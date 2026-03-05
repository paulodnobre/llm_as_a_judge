# Descriptive A/B Comparator (OpenAI)

This project compares `Answer A` and `Answer B` in a strictly descriptive and impartial way.

The program:
- does not choose a winner;
- does not assign scores;
- does not infer factual quality;
- only reports observable differences in the text.

Expected output format:
- 3 to 6 objective bullets
- 1 final sentence summarizing the overall difference without judgment

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

## 1) Web interface (manual)
Start the server:
```bash
python app.py
```

Open in your browser:
- `http://127.0.0.1:8000`

Fields:
- Question
- Model 1 Answer (Answer A)
- Model 2 Answer (Answer B)

Result:
- descriptive bullets
- final summary sentence
- full comparison JSON

## 2) Batch mode (JSONL -> report)
Input format (`JSONL`, one line per case):
- `id` (optional)
- `question`
- `answer_a` and `answer_b`

Legacy compatibility:
- also accepts `notebook_answer` and `internal_answer` instead of `answer_a` and `answer_b`.

Example in [data/sample_pairs.jsonl](/Users/daniel/llm_as_a_judge/data/sample_pairs.jsonl).

Run:
```bash
python llm_judge.py \
  --input data/sample_pairs.jsonl \
  --model gpt-4.1-mini \
  --temperature 0.0 \
  --output-json reports/comparison_report.json \
  --output-md reports/comparison_report.md
```

Outputs:
- `reports/comparison_report.json`
- `reports/comparison_report.md`

## Architecture
- [judge_core.py](/Users/daniel/llm_as_a_judge/judge_core.py): descriptive comparator prompt and parser
- [app.py](/Users/daniel/llm_as_a_judge/app.py): web interface
- [templates/index.html](/Users/daniel/llm_as_a_judge/templates/index.html): UI
- [llm_judge.py](/Users/daniel/llm_as_a_judge/llm_judge.py): batch pipeline
