#!/usr/bin/env python3
"""Descriptive/ground-truth A/B comparator using OpenAI."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from judge_core import CompareConfig, compare_answers, compare_answers_with_ground_truth


@dataclass
class Example:
    ex_id: str
    question: str
    answer_a: str
    answer_b: str
    ground_truth: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run A/B comparison")
    parser.add_argument("--input", required=True, help="Path to JSONL dataset")
    parser.add_argument("--model", default="gpt-4.1-mini", help="OpenAI model")
    parser.add_argument("--temperature", type=float, default=0.0, help="Model temperature")
    parser.add_argument("--max-tokens", type=int, default=900, help="Max output tokens")
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
            ground_truth = str(row.get("ground_truth", "")).strip()
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
                    ground_truth=ground_truth,
                )
            )
    if not examples:
        raise ValueError("Input dataset is empty")
    return examples


def to_markdown(report: dict, model: str, temperature: float) -> str:
    lines = [
        "# A/B Comparison Report",
        "",
        f"- Model: `{model}`",
        f"- Temperature: `{temperature}`",
        f"- Total cases: `{report['summary']['total_examples']}`",
        f"- Ground truth cases: `{report['summary']['ground_truth_cases']}`",
        "",
    ]
    for case in report["per_example"]:
        lines.append(f"## {case['id']} ({case['comparison']['mode']})")
        lines.append("")
        if case["comparison"]["mode"] == "ground_truth":
            lines.append(f"- A overall: `{case['comparison']['overall']['A']:.3f}`")
            lines.append(f"- B overall: `{case['comparison']['overall']['B']:.3f}`")
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
    gt_cases = 0

    for ex in examples:
        if ex.ground_truth:
            comparison = compare_answers_with_ground_truth(
                client=client,
                config=config,
                question=ex.question,
                ground_truth=ex.ground_truth,
                answer_a=ex.answer_a,
                answer_b=ex.answer_b,
            )
            gt_cases += 1
        else:
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
            "ground_truth_cases": gt_cases,
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
