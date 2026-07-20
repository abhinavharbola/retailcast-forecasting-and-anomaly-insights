from pathlib import Path

import pandas as pd
import streamlit as st

from src.storage.supabase_client import save_anomaly_flags
from src.utils.config import CONFIG

st.title("\U0001F6A8 Anomaly Detection")

DATA_DIR = Path(CONFIG["data"]["kaggle_outputs_dir"])
FILES = CONFIG["data"]["files"]


@st.cache_data
def load_anomaly_data():
    results = pd.read_parquet(DATA_DIR / FILES["anomaly_results"])
    eval_metrics = pd.read_csv(DATA_DIR / FILES["anomaly_eval_metrics"], index_col=0)
    return results, eval_metrics


results, eval_metrics = load_anomaly_data()

st.subheader("Synthetic-injection evaluation")
with st.container(border=True):
    st.dataframe(eval_metrics, use_container_width=True)
    st.caption(
        "Control limits: threshold computed per store-family from that series' own clean "
        "residual std (k=2.5). IsolationForest: contamination fixed at 5%, so its recall "
        "is structurally capped by that rate regardless of the true anomaly count."
    )

st.divider()
st.subheader("Flagged anomalies on real holdout data")
col1, col2 = st.columns(2)
col1.metric("Control-limit flags", int(results["control_limit_flag"].sum()))
col2.metric("IsolationForest flags", int(results["isoforest_flag"].sum()))

method = st.radio("Filter by method", ["control_limit_flag", "isoforest_flag", "either"], horizontal=True)
if method == "either":
    flagged = results[(results["control_limit_flag"] == 1) | (results["isoforest_flag"] == 1)]
else:
    flagged = results[results[method] == 1]

with st.container(border=True):
    st.dataframe(
        flagged[
            ["date", "store_nbr", "family", "sales", "forecast", "residual", "control_limit_flag", "isoforest_flag"]
        ].sort_values("date"),
        use_container_width=True,
    )
    if st.button("Log flagged anomalies to Supabase"):
        to_log = flagged[
            ["date", "store_nbr", "family", "sales", "forecast", "residual", "control_limit_flag", "isoforest_flag"]
        ].copy()
        to_log["date"] = to_log["date"].dt.strftime("%Y-%m-%d")
        try:
            save_anomaly_flags(to_log)
            st.success(f"Logged {len(to_log)} rows to Supabase.")
        except Exception as e:
            st.error(f"Logging failed: {e}")