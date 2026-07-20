from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

from src.llm.grounding_check import check_grounding
from src.llm.narrative import generate_narrative
from src.storage.supabase_client import fetch_reports, save_report
from src.utils.config import CONFIG

st.title("\U0001F916 AI-Generated Results Report")

DATA_DIR = str(Path(CONFIG["data"]["kaggle_outputs_dir"]))


def build_report_markdown(text, facts, provider, grounding_ratio, created_at=None):
    created_at = created_at or datetime.now(timezone.utc).isoformat()
    facts_lines = "\n".join(f"- **{k}**: {v}" for k, v in facts.items())
    return f"""# RetailCast AI Report

**Generated:** {created_at}
**Provider:** {provider}
**Grounding ratio:** {grounding_ratio * 100:.0f}%

---

{text}

---

## Source facts

{facts_lines}
"""


def download_filename(provider, created_at):
    safe_ts = str(created_at).replace(":", "-").replace(" ", "_")
    return f"retailcast_report_{provider}_{safe_ts}.md"


if st.button("Generate new report"):
    with st.spinner("Calling LLM provider..."):
        try:
            result = generate_narrative(DATA_DIR)
        except Exception as e:
            st.error(f"Report generation failed: {e}")
            st.stop()

    grounding = check_grounding(result["text"], result["facts"])
    facts = result["facts"]
    generated_at = datetime.now(timezone.utc).isoformat()

    st.success(f"Generated using: {result['provider']}")

    with st.expander("Provider fallback log", expanded=any(a["provider"] != result["provider"] for a in result["attempts"])):
        for a in result["attempts"]:
            icon = "\u2705" if a["success"] else "\u274c"
            st.write(f"{icon} **{a['provider']}** — {a['error'] or 'succeeded'}")

    st.subheader("Key metrics at a glance")
    col1, col2 = st.columns(2)
    with col1:
        st.caption("Forecasting: MASE on holdout (lower is better)")
        model_mase = pd.DataFrame(
            {"mase": [facts["best_ml_mase_holdout"], facts["prophet_mase_holdout"], facts["sarima_mase_holdout"]]},
            index=[facts["best_ml_model"], "prophet", "sarima"],
        )
        st.bar_chart(model_mase)
    with col2:
        st.caption("Anomaly detection: precision vs recall")
        anomaly_pr = pd.DataFrame(
            {
                "precision": [facts["control_limit_precision"], facts["isoforest_precision"]],
                "recall": [facts["control_limit_recall"], facts["isoforest_recall"]],
            },
            index=["control_limit", "isoforest"],
        )
        st.bar_chart(anomaly_pr)

    st.metric("Grounding ratio", f"{grounding['grounded_ratio'] * 100:.0f}%")
    st.caption(
        "Regex-based numeric check, not full claim verification. It can miss "
        "paraphrased claims with no literal number, and can flag numbers that are "
        "correct but simply aren't in the source facts. Treat a low ratio as "
        "'needs review,' not 'definitely wrong.'"
    )

    st.subheader("Narrative")
    with st.container(border=True):
        st.markdown(result["text"])

    st.download_button(
        "\U0001F4E5 Download this report (.md)",
        data=build_report_markdown(result["text"], facts, result["provider"], grounding["grounded_ratio"], generated_at),
        file_name=download_filename(result["provider"], generated_at),
        mime="text/markdown",
    )

    with st.expander("Flagged numeric claims"):
        ungrounded = [c for c in grounding["claims"] if not c["grounded"]]
        st.write(ungrounded if ungrounded else "None - every extracted number matched a source figure.")

    with st.expander("Source facts used"):
        st.json(facts)

    try:
        save_report(result["text"], facts, result["provider"], grounding["grounded_ratio"])
        st.caption("Saved to Supabase.")
    except Exception as e:
        st.caption(f"Not saved to Supabase: {e}")

st.subheader("Past reports")
try:
    past = fetch_reports()
    if past:
        for r in past:
            created_at = r.get("created_at", "unknown date")
            provider = r.get("provider", "unknown provider")
            with st.expander(f"{created_at} - {provider}"):
                st.markdown(r["report_text"])
                st.download_button(
                    "\U0001F4E5 Download this report (.md)",
                    data=build_report_markdown(
                        r["report_text"], r.get("facts", {}), provider,
                        r.get("grounding_ratio", 0), created_at,
                    ),
                    file_name=download_filename(provider, created_at),
                    mime="text/markdown",
                    key=f"download_{r.get('id', created_at)}",
                )
    else:
        st.caption("No saved reports yet.")
except Exception as e:
    st.caption(f"Could not load past reports: {e}")