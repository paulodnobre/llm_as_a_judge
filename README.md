# A/B Comparator with Optional Ground Truth (OpenAI)

This project supports two evaluation methodologies:

1. Descriptive mode (default)
- Compares Answer A vs Answer B objectively.
- No winner, no scoring.
- Returns 3-6 bullets + 1 neutral final summary sentence.

2. Ground Truth mode (checkbox enabled)
- Uses a human reference answer (`ground_truth`).
- Scores both answers on:
  - `faithfulness`
  - `answer_relevance`
  - `completeness`
  - `conciseness`
- Also returns objective bullets and a final summary.

## Requirements
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## OpenAI credentials
Create a `.env` file (or export in your shell):

```bash
OPENAI_API_KEY=your_api_key_here
```

## 1) Web interface
Run:
```bash
python app.py
```

Open:
- `http://127.0.0.1:8000`

Flow:
- Fill `Question`, `Model 1 Answer (A)`, `Model 2 Answer (B)`.
- Optional: enable `Use Ground Truth methodology` and fill `Human Ground Truth Answer`.
- Click `Analyze`.

Output:
- Descriptive mode: bullets + final summary.
- Ground Truth mode: vector score table for A/B + bullets + final summary.

## 2) Batch mode (JSONL -> report)
Input format (`JSONL`, one line per case):
- Required: `question`, `answer_a` and `answer_b`
- Optional: `ground_truth` (if present, ground-truth vector scoring is used)
- Optional: `id`

Legacy compatibility:
- also accepts `notebook_answer` and `internal_answer` instead of `answer_a` and `answer_b`.

Run:
```bash
python llm_judge.py \
  --input data/sample_pairs.jsonl \
  --model gpt-4.1-mini \
  --temperature 0.0 \
  --output-json reports/comparison_report.json \
  --output-md reports/comparison_report.md
```

## Architecture
- [judge_core.py](/Users/daniel/llm_as_a_judge/judge_core.py): descriptive + ground-truth evaluation logic
- [app.py](/Users/daniel/llm_as_a_judge/app.py): web interface with checkbox-driven mode
- [templates/index.html](/Users/daniel/llm_as_a_judge/templates/index.html): UI
- [llm_judge.py](/Users/daniel/llm_as_a_judge/llm_judge.py): batch pipeline
