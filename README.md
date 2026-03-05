# Comparador Descritivo A/B (OpenAI)

Projeto para comparar `Resposta A` e `Resposta B` de forma estritamente descritiva e imparcial.

O programa:
- nao escolhe vencedora;
- nao atribui nota;
- nao faz julgamento de qualidade factual;
- retorna apenas diferencas observaveis no texto.

Formato esperado de saida:
- 3 a 6 bullets objetivos
- 1 frase final de resumo sem julgamento

## Requisitos
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Credenciais OpenAI
Crie `.env` (ou exporte no shell):

```bash
OPENAI_API_KEY=seu_token_aqui
```

## 1) Interface Web (manual)
Suba o servidor:
```bash
python app.py
```

Abra no navegador:
- `http://127.0.0.1:8000`

Campos:
- Pergunta
- Resposta Modelo 1 (Resposta A)
- Resposta Modelo 2 (Resposta B)

Resultado:
- bullets descritivos
- frase final resumindo a diferenca geral
- JSON completo da analise

## 2) Modo Batch (JSONL -> relatorio)
Formato de entrada (`JSONL`, uma linha por caso):
- `id` (opcional)
- `question`
- `answer_a` e `answer_b`

Compatibilidade legada:
- tambem aceita `notebook_answer` e `internal_answer` no lugar de `answer_a` e `answer_b`.

Exemplo em [data/sample_pairs.jsonl](/Users/daniel/llm_as_a_judge/data/sample_pairs.jsonl).

Execucao:
```bash
python llm_judge.py \
  --input data/sample_pairs.jsonl \
  --model gpt-4.1-mini \
  --temperature 0.0 \
  --output-json reports/comparison_report.json \
  --output-md reports/comparison_report.md
```

Saidas:
- `reports/comparison_report.json`
- `reports/comparison_report.md`

## Arquitetura
- [judge_core.py](/Users/daniel/llm_as_a_judge/judge_core.py): prompt e parser do comparador descritivo
- [app.py](/Users/daniel/llm_as_a_judge/app.py): interface web
- [templates/index.html](/Users/daniel/llm_as_a_judge/templates/index.html): UI
- [llm_judge.py](/Users/daniel/llm_as_a_judge/llm_judge.py): pipeline batch
