#!/usr/bin/env python3
from __future__ import annotations

import json

from dotenv import load_dotenv
from flask import Flask, render_template, request
from openai import OpenAI

from judge_core import CompareConfig, compare_answers


load_dotenv()
app = Flask(__name__)


def render_home(**kwargs):
    defaults = {
        "model": "gpt-4.1-mini",
        "temperature": "0.0",
        "question": "",
        "answer_1": "",
        "answer_2": "",
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

    if not all([model, question, answer_1, answer_2]):
        return render_home(
            model=model,
            temperature=temperature_raw,
            question=question,
            answer_1=answer_1,
            answer_2=answer_2,
            error="Preencha todos os campos.",
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
            error="Temperature invalida.",
        )

    try:
        client = OpenAI()
        config = CompareConfig(model=model, temperature=temperature, max_tokens=700)
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
            error=str(exc),
        )

    return render_home(
        model=model,
        temperature=temperature_raw,
        question=question,
        answer_1=answer_1,
        answer_2=answer_2,
        result=raw_result,
        result_json=json.dumps(raw_result, ensure_ascii=False, indent=2),
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
