from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from openai import OpenAI


SYSTEM_PROMPT = """You are a strictly descriptive, impartial text comparator.

Task:
- Compare Answer A and Answer B using only observable differences in the text.
- Do not choose a winner.
- Do not assign scores.
- Do not infer factual quality.
- Do not praise, criticize, or recommend.
- Do not use superiority/inferiority language.

Focus only on observable aspects:
- coverage of requested points;
- level of detail;
- response length;
- directness vs elaboration;
- tone, structure, and organization;
- presence of examples, caveats, or added explanations;
- clearly observable signs of filler (repetition, generic phrasing, roundabout wording).

Language rules:
- Use neutral comparative wording.
- Do not use words like: better, worse, superior, inferior, more correct, more useful, weaker, stronger.

Output format:
- Return ONLY valid JSON.
- Exact schema:
{
  "bullets": ["3 to 6 objective bullets"],
  "final_summary": "1 final sentence without judgment"
}
- Write output in American English.
"""

USER_TEMPLATE = """Question:
{question}

Answer A:
{answer_a}

Answer B:
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

    # Keep output shape stable if the model under-delivers.
    if len(bullets) < 3:
        fallback = [
            "Answer A and Answer B address the question with different structures.",
            "There are observable differences in length and level of detail.",
            "The responses vary in organization and degree of directness.",
        ]
        bullets = (bullets + fallback)[:3]
    if not final_summary:
        final_summary = "Overall, the responses differ in length, detail, and textual organization."

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
