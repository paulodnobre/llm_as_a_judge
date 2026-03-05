from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from openai import OpenAI


SYSTEM_PROMPT = """Voce e um comparador textual estritamente descritivo e imparcial.

Tarefa:
- Comparar Resposta A e Resposta B apenas por diferencas observaveis no texto.
- Nao decidir vencedora.
- Nao dar nota.
- Nao inferir qualidade factual.
- Nao elogiar, nao criticar, nao recomendar.
- Nao usar termos de superioridade/inferioridade.

Foque em aspectos observaveis:
- cobertura dos pontos pedidos;
- nivel de detalhe;
- extensao;
- objetividade vs elaboracao;
- tom, estrutura e organizacao;
- exemplos, ressalvas e explicacoes adicionais;
- sinais claros de repeticao/rodeio/generalidade.

Regras de linguagem:
- Use formulacoes neutras e comparativas.
- Nao use palavras como: melhor, pior, superior, inferior, mais correta, mais util, mais fraca, mais forte.

Formato de saida:
- Retorne SOMENTE JSON valido.
- Schema exato:
{
  "bullets": ["3 a 6 bullets objetivos"],
  "final_summary": "1 frase final sem julgamento"
}
"""

USER_TEMPLATE = """Pergunta:
{question}

Resposta A:
{answer_a}

Resposta B:
{answer_b}
"""


@dataclass
class CompareConfig:
    model: str = "gpt-4.1-mini"
    temperature: float = 0.0
    max_tokens: int = 700


class CompareParseError(RuntimeError):
    pass


def extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("{"):
        return json.loads(text)
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        return json.loads(match.group(0))
    raise CompareParseError("No JSON object found in model response")


def _sanitize_bullets(items: Any) -> list[str]:
    if not isinstance(items, list):
        return []
    out: list[str] = []
    for item in items:
        text = str(item).strip().lstrip("-").strip()
        if text:
            out.append(text[:240])
    if len(out) > 6:
        out = out[:6]
    return out


def normalize_result(raw: dict[str, Any]) -> dict[str, Any]:
    bullets = _sanitize_bullets(raw.get("bullets", []))
    final_summary = str(raw.get("final_summary", "")).strip()[:320]

    # Keep output shape stable if model under-delivers.
    if len(bullets) < 3:
        fallback = [
            "A Resposta A e a Resposta B abordam a pergunta com estruturas diferentes.",
            "Ha diferencas observaveis de extensao e nivel de detalhe entre os textos.",
            "Os textos variam na forma de organizacao e no grau de objetividade.",
        ]
        bullets = (bullets + fallback)[:3]
    if not final_summary:
        final_summary = "De forma geral, as respostas diferem em extensao, detalhamento e organizacao textual."

    return {
        "bullets": bullets,
        "final_summary": final_summary,
    }


def compare_answers(
    client: OpenAI,
    config: CompareConfig,
    question: str,
    answer_a: str,
    answer_b: str,
) -> dict[str, Any]:
    user_prompt = USER_TEMPLATE.format(
        question=question,
        answer_a=answer_a,
        answer_b=answer_b,
    )

    response = client.chat.completions.create(
        model=config.model,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )

    content = response.choices[0].message.content or "{}"
    parsed = extract_json(content)
    return normalize_result(parsed)
