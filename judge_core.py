from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from openai import OpenAI


DESCRIPTIVE_SYSTEM_PROMPT = """You are a strictly descriptive, impartial text comparator.

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

DESCRIPTIVE_USER_TEMPLATE = """Question:
{question}

Answer A:
{answer_a}

Answer B:
{answer_b}
"""

VECTOR_SYSTEM_PROMPT = """You are an impartial LLM evaluator.
Evaluate Answer A and Answer B using the provided Question and Human Ground Truth answer.

Scoring vectors (0.0 to 1.0) for each answer:
- faithfulness (groundedness): Does the answer avoid unsupported claims relative to what can be supported by the question + ground truth? Penalize hallucinations strongly.
- answer_relevance: Does the answer directly address the user question?
- completeness: Does the answer cover the key points present in the ground truth?
- conciseness: Is the answer clear and direct without unnecessary filler?

Rules:
- Be strict and impartial.
- Do not apply position bias.
- Do not reward verbosity by itself.
- Return ONLY valid JSON.
- Keep rationales short.
- Write output in American English.

Few-shot reference example:
Question: Do you happen to have expertise in setting up backend infrastructure on AWS to be HIPAA compliant?
Human ground truth answer: Many of our HIPAA compliant clients are using AWS, let me confirm which ones we set up from scratch and I'll let you know. In case it helps too, few of them uses Aptible (https://aptible.com/), which is a PaaS over AWS that helps managing the infra and ensures it to be HIPAA Compliant. what kind of details would it work for you? just which ones were the clients?
Bed Connect uses AWS., Cardiex uses Aptible. CareBridge, Catapult Health and Globo uses AWS (either ECS or EC2), but managed by their own devops teams
"""

VECTOR_USER_TEMPLATE = """Question:
{question}

Human Ground Truth Answer:
{ground_truth}

Answer A:
{answer_a}

Answer B:
{answer_b}

Return JSON with this exact schema:
{{
  "scores": {{
    "A": {{
      "faithfulness": <float 0..1>,
      "answer_relevance": <float 0..1>,
      "completeness": <float 0..1>,
      "conciseness": <float 0..1>
    }},
    "B": {{
      "faithfulness": <float 0..1>,
      "answer_relevance": <float 0..1>,
      "completeness": <float 0..1>,
      "conciseness": <float 0..1>
    }}
  }},
  "bullets": ["3 to 6 objective comparison bullets"],
  "final_summary": "1 sentence summary",
  "short_rationale_a": "brief rationale for A",
  "short_rationale_b": "brief rationale for B"
}}
"""

METRICS = [
    "faithfulness",
    "answer_relevance",
    "completeness",
    "conciseness",
]


@dataclass
class CompareConfig:
    model: str = "gpt-4.1-mini"
    temperature: float = 0.0
    max_tokens: int = 900


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


def _clamp01(value: Any) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, v))


def _avg_metric(scores: dict[str, float]) -> float:
    if not scores:
        return 0.0
    return sum(scores.values()) / len(scores)


def normalize_descriptive_result(raw: dict[str, Any]) -> dict[str, Any]:
    bullets = _sanitize_bullets(raw.get("bullets", []))
    final_summary = str(raw.get("final_summary", "")).strip()[:320]

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
        "mode": "descriptive",
        "bullets": bullets,
        "final_summary": final_summary,
    }


def normalize_ground_truth_result(raw: dict[str, Any]) -> dict[str, Any]:
    scores_raw = raw.get("scores", {}) if isinstance(raw.get("scores", {}), dict) else {}
    out_scores: dict[str, dict[str, float]] = {"A": {}, "B": {}}
    for side in ("A", "B"):
        side_in = scores_raw.get(side, {}) if isinstance(scores_raw, dict) else {}
        for metric in METRICS:
            out_scores[side][metric] = _clamp01(side_in.get(metric, 0.0))

    bullets = _sanitize_bullets(raw.get("bullets", []))
    if len(bullets) < 3:
        bullets = [
            "The answers differ in how closely they align with the provided ground truth.",
            "The answers show different levels of detail and point coverage.",
            "The responses vary in directness and conciseness.",
        ]

    final_summary = str(raw.get("final_summary", "")).strip()[:320]
    if not final_summary:
        final_summary = "Overall, the answers differ across grounding, relevance, completeness, and conciseness."

    rationale_a = str(raw.get("short_rationale_a", "")).strip()[:220]
    rationale_b = str(raw.get("short_rationale_b", "")).strip()[:220]

    overall_a = _avg_metric(out_scores["A"])
    overall_b = _avg_metric(out_scores["B"])

    return {
        "mode": "ground_truth",
        "scores": out_scores,
        "overall": {
            "A": overall_a,
            "B": overall_b,
        },
        "bullets": bullets,
        "final_summary": final_summary,
        "short_rationale_a": rationale_a,
        "short_rationale_b": rationale_b,
    }


def compare_answers(
    client: OpenAI,
    config: CompareConfig,
    question: str,
    answer_a: str,
    answer_b: str,
) -> dict[str, Any]:
    user_prompt = DESCRIPTIVE_USER_TEMPLATE.format(
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
            {"role": "system", "content": DESCRIPTIVE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )

    content = response.choices[0].message.content or "{}"
    parsed = extract_json(content)
    return normalize_descriptive_result(parsed)


def compare_answers_with_ground_truth(
    client: OpenAI,
    config: CompareConfig,
    question: str,
    ground_truth: str,
    answer_a: str,
    answer_b: str,
) -> dict[str, Any]:
    user_prompt = VECTOR_USER_TEMPLATE.format(
        question=question,
        ground_truth=ground_truth,
        answer_a=answer_a,
        answer_b=answer_b,
    )

    response = client.chat.completions.create(
        model=config.model,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": VECTOR_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )

    content = response.choices[0].message.content or "{}"
    parsed = extract_json(content)
    return normalize_ground_truth_result(parsed)
