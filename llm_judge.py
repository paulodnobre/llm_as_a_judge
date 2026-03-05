#!/usr/bin/env python3
"""Descriptive A/B comparator using OpenAI."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from judge_core import CompareConfig, compare_answers


@dataclass
class Example:
    ex_id: str
    question: str
    answer_a: str
    answer_b: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run descriptive A/B comparison")
    parser.add_argument("--input", required=True, help="Path to JSONL dataset")
    parser.add_argument("--model", default="gpt-4.1-mini", help="OpenAI model")
    parser.add_argument("--temperature", type=float, default=0.0, help="Model temperature")
    parser.add_argument("--max-tokens", type=int, default=700, help="Max output tokens")
    parser.add_argument("--output-json", default="reports/comparison_report.json", help="Path for machine-readable report")
    parser.add_argument("--output-md", default="reports/comparison_report.md", help="Path for human-readable report")
    return parser.parse_args()


def load_examples(path: Path) -> list[Example]:
    examples: list[Example] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            question = row.get("question")
            answer_a = row.get("answer_a", row.get("notebook_answer"))
            answer_b = row.get("answer_b", row.get("internal_answer"))
            if not question or not answer_a or not answer_b:
                raise ValueError(
                    f"Missing required keys at line {i}. Expected question + (answer_a,answer_b) "
                    "or question + (notebook_answer,internal_answer)."
                )
            examples.append(
                Example(
                    ex_id=str(row.get("id", f"line-{i}")),
                    question=str(question),
                    answer_a=str(answer_a),
                    answer_b=str(answer_b),
                )
            )
    if not examples:
        raise ValueError("Input dataset is empty")
    return examples


def to_markdown(report: dict, model: str, temperature: float) -> str:
    lines = [
        "# Descriptive A/B Report",
        "",
        f"- Model: `{model}`",
        f"- Temperature: `{temperature}`",
        f"- Total cases: `{report['summary']['total_examples']}`",
        "",
    ]
    for case in report["per_example"]:
        lines.append(f"## {case['id']}")
        lines.append("")
        for bullet in case["comparison"]["bullets"]:
            lines.append(f"- {bullet}")
        lines.append("")
        lines.append(f"Final summary: {case['comparison']['final_summary']}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    load_dotenv()
    args = parse_args()

    client = OpenAI()
    config = CompareConfig(
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    examples = load_examples(Path(args.input))
    per_example: list[dict] = []
    for ex in examples:
        comparison = compare_answers(
            client=client,
            config=config,
            question=ex.question,
            answer_a=ex.answer_a,
            answer_b=ex.answer_b,
        )
        per_example.append(
            {
                "id": ex.ex_id,
                "question": ex.question,
                "comparison": comparison,
            }
        )

    report = {
        "summary": {
            "total_examples": len(per_example),
            "mode": "descriptive_only",
        },
        "per_example": per_example,
    }

    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)

    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    output_md.write_text(to_markdown(report, args.model, args.temperature), encoding="utf-8")

    print(f"Report JSON written to: {output_json}")
    print(f"Report MD written to:   {output_md}")


if __name__ == "__main__":
    main()
