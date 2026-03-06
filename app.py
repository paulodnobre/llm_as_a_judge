#!/usr/bin/env python3
from __future__ import annotations

import json

from dotenv import load_dotenv
from flask import Flask, render_template, request
from openai import OpenAI

from judge_core import CompareConfig, compare_answers, compare_answers_with_ground_truth


load_dotenv()
app = Flask(__name__)


def render_home(**kwargs):
    defaults = {
        "model": "gpt-4.1-mini",
        "temperature": "0.0",
        "question": "",
        "answer_1": "",
        "answer_2": "",
        "use_ground_truth": False,
        "ground_truth": "",
        "error": "",
        "result": None,
        "result_json": "",
    }
    defaults.update(kwargs)
    return render_template("index.html", **defaults)


@app.get("/")
def home():
    return render_home()


@app.post("/analyze")
def analyze():
    model = request.form.get("model", "gpt-4.1-mini").strip()
    temperature_raw = request.form.get("temperature", "0.0").strip()
    question = request.form.get("question", "").strip()
    answer_1 = request.form.get("answer_1", "").strip()
    answer_2 = request.form.get("answer_2", "").strip()
    use_ground_truth = request.form.get("use_ground_truth") == "on"
    ground_truth = request.form.get("ground_truth", "").strip()

    if not all([model, question, answer_1, answer_2]):
        return render_home(
            model=model,
            temperature=temperature_raw,
            question=question,
            answer_1=answer_1,
            answer_2=answer_2,
            use_ground_truth=use_ground_truth,
            ground_truth=ground_truth,
            error="Please fill in all required fields.",
        )

    if use_ground_truth and not ground_truth:
        return render_home(
            model=model,
            temperature=temperature_raw,
            question=question,
            answer_1=answer_1,
            answer_2=answer_2,
            use_ground_truth=use_ground_truth,
            ground_truth=ground_truth,
            error="Ground truth is required when the checkbox is enabled.",
        )

    try:
        temperature = float(temperature_raw)
    except ValueError:
        return render_home(
            model=model,
            temperature=temperature_raw,
            question=question,
            answer_1=answer_1,
            answer_2=answer_2,
            use_ground_truth=use_ground_truth,
            ground_truth=ground_truth,
            error="Invalid temperature value.",
        )

    try:
        client = OpenAI()
        config = CompareConfig(model=model, temperature=temperature, max_tokens=900)
        if use_ground_truth:
            raw_result = compare_answers_with_ground_truth(
                client=client,
                config=config,
                question=question,
                ground_truth=ground_truth,
                answer_a=answer_1,
                answer_b=answer_2,
            )
        else:
            raw_result = compare_answers(
                client=client,
                config=config,
                question=question,
                answer_a=answer_1,
                answer_b=answer_2,
            )
    except Exception as exc:
        return render_home(
            model=model,
            temperature=temperature_raw,
            question=question,
            answer_1=answer_1,
            answer_2=answer_2,
            use_ground_truth=use_ground_truth,
            ground_truth=ground_truth,
            error=str(exc),
        )

    return render_home(
        model=model,
        temperature=temperature_raw,
        question=question,
        answer_1=answer_1,
        answer_2=answer_2,
        use_ground_truth=use_ground_truth,
        ground_truth=ground_truth,
        result=raw_result,
        result_json=json.dumps(raw_result, ensure_ascii=False, indent=2),
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
