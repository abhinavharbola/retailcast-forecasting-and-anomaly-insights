from pathlib import Path

import pandas as pd
import streamlit as st

from src.storage.supabase_client import save_forecast_run
from src.utils.config import CONFIG

st.title("\U0001F4C9 Forecast Explorer")

DATA_DIR = Path(CONFIG["data"]["kaggle_outputs_dir"])
FILES = CONFIG["data"]["files"]


@st.cache_data
def load_results():
    prophet = pd.read_csv(DATA_DIR / FILES["prophet_results"])
    sarima = pd.read_csv(DATA_DIR / FILES["sarima_results"])
    ml = pd.read_csv(DATA_DIR / FILES["ml_results"])
    holdout = pd.read_parquet(DATA_DIR / FILES["final_holdout_predictions"])
    return prophet, sarima, ml, holdout


prophet, sarima, ml, holdout = load_results()

ml_holdout = ml[ml["fold"] == "holdout"][["model", "mape", "wape", "mase"]].assign(source="ML (global)")
prophet_holdout = (
    prophet[prophet["fold"] == "holdout"][["mape", "wape", "mase"]]
    .mean().to_frame().T.assign(model="prophet", source="Prophet (60-series avg)")
)
sarima_holdout = (
    sarima[sarima["fold"] == "holdout"][["mape", "wape", "mase"]]
    .mean().to_frame().T.assign(model="sarima", source="SARIMA (3-series avg)")
)
comparison = pd.concat([ml_holdout, prophet_holdout, sarima_holdout], ignore_index=True)
comparison["fold"] = comparison.get("fold", "holdout")
comparison["fold"] = comparison["fold"].fillna("holdout")

best_row = comparison.sort_values("mase").iloc[0]
m1, m2, m3 = st.columns(3)
m1.metric("Best model", str(best_row["model"]))
m2.metric("Holdout MASE", f"{float(best_row['mase']):.3f}")
m3.metric("Holdout MAPE", f"{float(best_row['mape']):.2f}%")

st.subheader("Model comparison (holdout)")
with st.container(border=True):
    st.dataframe(comparison[["source", "model", "mape", "wape", "mase"]], use_container_width=True)
    if st.button("Log this comparison to Supabase"):
        logged = 0
        error = None
        for _, row in comparison[["model", "fold", "mape", "wape", "mase"]].iterrows():
            try:
                save_forecast_run(row.to_dict())
                logged += 1
            except Exception as e:
                error = str(e)
                break
        if error:
            st.error(f"Logging failed after {logged} rows: {error}")
        else:
            st.success(f"Logged {logged} rows to Supabase.")

st.divider()
st.subheader("Store-family forecast vs. actual")
col1, col2 = st.columns(2)
store = col1.selectbox("Store", sorted(holdout["store_nbr"].unique()))
family = col2.selectbox("Family", sorted(holdout["family"].unique()))

series = holdout[(holdout["store_nbr"] == store) & (holdout["family"] == family)].sort_values("date")
with st.container(border=True):
    if series.empty:
        st.warning("No holdout predictions for this store/family combination.")
    else:
        st.line_chart(series.set_index("date")[["sales", "forecast"]])
        st.caption(f"Predictions shown are from {best_row['model']}, the best model on holdout MASE.")