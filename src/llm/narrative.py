import json
from pathlib import Path

import pandas as pd
import requests

from src.utils.config import CONFIG


def build_facts(results_dir):
    """Pulls the numeric facts the narrative is allowed to cite. Also used by
    grounding_check.py as the source of truth to verify the LLM's output against."""
    results_dir = Path(results_dir)
    ml = pd.read_csv(results_dir / "ml_results.csv")
    prophet = pd.read_csv(results_dir / "prophet_results.csv")
    sarima = pd.read_csv(results_dir / "sarima_results.csv")
    anomaly_metrics = pd.read_csv(results_dir / "anomaly_eval_metrics.csv", index_col=0)

    ml_holdout = ml[ml["fold"] == "holdout"]
    best_model = ml_holdout.sort_values("mase").iloc[0]

    return {
        "best_ml_model": best_model["model"],
        "best_ml_mase_holdout": round(float(best_model["mase"]), 3),
        "best_ml_mape_holdout": round(float(best_model["mape"]), 2),
        "best_ml_wape_holdout": round(float(best_model["wape"]), 2),
        "prophet_mase_holdout": round(float(prophet[prophet["fold"] == "holdout"]["mase"].mean()), 3),
        "sarima_mase_holdout": round(float(sarima[sarima["fold"] == "holdout"]["mase"].mean()), 3),
        "control_limit_precision": round(float(anomaly_metrics.loc["control_limit_flag_injected", "precision"]), 3),
        "control_limit_recall": round(float(anomaly_metrics.loc["control_limit_flag_injected", "recall"]), 3),
        "isoforest_precision": round(float(anomaly_metrics.loc["isoforest_flag_injected", "precision"]), 3),
        "isoforest_recall": round(float(anomaly_metrics.loc["isoforest_flag_injected", "recall"]), 3),
    }


def build_prompt(facts):
    return f"""You are a senior retail analytics consultant preparing a results brief for
leadership. Write a structured, insightful report using ONLY the numbers given below - do
not invent, estimate, or derive new comparison numbers beyond what's directly stated (e.g.
"X is lower than Y" is fine; computing a new percentage difference that isn't in the facts
is not).

Format the report in Markdown with exactly these sections:

## Executive Summary
2-3 bullet points: the single most important takeaway from forecasting, the single most
important takeaway from anomaly detection, and one clear recommendation.

## Forecasting Performance
Compare the three approaches using MASE (primary), MAPE, and WAPE. Explain what the
winning model's MASE means in plain terms relative to a naive seasonal baseline.

## Anomaly Detection Performance
Compare control limits vs. Isolation Forest on precision and recall. Explain the practical
tradeoff between them in one or two sentences.

## Recommendation
One short paragraph: which model to deploy and why, and how the two anomaly methods could
be used together if the numbers justify it.

Keep total length to 300-400 words. Do not restate the raw JSON. Do not add a title (the
page already has one).

Facts:
{json.dumps(facts, indent=2)}
"""


def _post_json(url, headers, payload):
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    if not resp.ok:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:400]}")
    return resp.json()


def _call_nim(prompt, api_key, model, max_tokens, temperature):
    data = _post_json(
        "https://integrate.api.nvidia.com/v1/chat/completions",
        {"Authorization": f"Bearer {api_key}"},
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
    )
    return data["choices"][0]["message"]["content"]


def _call_groq(prompt, api_key, model, max_tokens, temperature):
    data = _post_json(
        "https://api.groq.com/openai/v1/chat/completions",
        {"Authorization": f"Bearer {api_key}"},
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
    )
    return data["choices"][0]["message"]["content"]


def _call_gemini(prompt, api_key, model, max_tokens, temperature):
    data = _post_json(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}",
        {},
        {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"maxOutputTokens": max_tokens, "temperature": temperature},
        },
    )
    return data["candidates"][0]["content"]["parts"][0]["text"]


PROVIDERS = {"nim": _call_nim, "groq": _call_groq, "gemini": _call_gemini}
KEY_NAMES = {"nim": "nim_api_key", "groq": "groq_api_key", "gemini": "gemini_api_key"}


def generate_narrative(results_dir, config=CONFIG):
    facts = build_facts(results_dir)
    prompt = build_prompt(facts)

    llm_cfg = config["llm"]
    env = config["env"]
    attempts = []

    for provider in llm_cfg["provider_priority"]:
        api_key = env.get(KEY_NAMES[provider])
        if not api_key:
            attempts.append({"provider": provider, "success": False, "error": "no API key configured"})
            continue
        try:
            text = PROVIDERS[provider](
                prompt, api_key, llm_cfg["models"][provider],
                llm_cfg["max_tokens"], llm_cfg["temperature"],
            )
            attempts.append({"provider": provider, "success": True, "error": None})
            return {"text": text, "provider": provider, "facts": facts, "attempts": attempts}
        except Exception as e:
            attempts.append({"provider": provider, "success": False, "error": str(e)})

    raise RuntimeError(f"All LLM providers failed: {attempts}")