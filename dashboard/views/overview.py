import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.utils.config import CONFIG

st.title("\U0001F4CA Dataset Overview")

DATA_DIR = Path(CONFIG["data"]["kaggle_outputs_dir"])
FILES = CONFIG["data"]["files"]


@st.cache_data
def load_overview_data():
    with open(DATA_DIR / FILES["subset_config"]) as f:
        config = json.load(f)
    stationarity = pd.read_csv(DATA_DIR / FILES["stationarity"])
    pattern = pd.read_csv(DATA_DIR / FILES["demand_pattern"])
    return config, stationarity, pattern


config, stationarity, pattern = load_overview_data()

col1, col2 = st.columns(2)
with col1:
    with st.container(border=True):
        st.markdown("#### Selected stores")
        st.write(config["selected_stores"])
with col2:
    with st.container(border=True):
        st.markdown("#### Selected families")
        st.write(config["selected_families"])

st.divider()
st.subheader("Demand pattern classification")
st.caption(
    "Syntetos-Boylan classification (ADI / CV\u00b2). MAPE is unreliable on "
    "intermittent/erratic series - see the Forecast Explorer's WAPE/MASE for those."
)
with st.container(border=True):
    st.bar_chart(pattern["pattern"].value_counts())
    st.dataframe(pattern, use_container_width=True)

st.divider()
st.subheader("Stationarity (ADF test)")
pct_stationary = stationarity["likely_stationary_adf"].mean() * 100
with st.container(border=True):
    st.metric("Series stationary by ADF", f"{pct_stationary:.1f}%")
    st.dataframe(stationarity, use_container_width=True)