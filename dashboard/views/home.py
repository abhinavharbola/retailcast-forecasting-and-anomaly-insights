import streamlit as st

st.markdown(
    """
    <div style="padding-top: 1rem;">
        <h1 style="margin-bottom: 0;">\U0001F4C8 RetailCast</h1>
        <p style="font-size: 1.05rem; color: #9CA3AF; margin-top: 0.4rem; max-width: 780px;">
            Retail demand forecasting, anomaly detection, and a self-verifying GenAI narrative
            layer — benchmarked across Prophet, SARIMA, LightGBM, and XGBoost on 60 store-family
            series from the Favorita Store Sales dataset.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.divider()
st.subheader("Explore")

col1, col2, col3, col4 = st.columns(4)

with col1:
    with st.container(border=True):
        st.markdown("#### \U0001F4CA Overview")
        st.caption("Dataset scope, demand pattern classification, stationarity tests.")
        st.page_link("views/overview.py", label="Open", icon="\u27A1\uFE0F")

with col2:
    with st.container(border=True):
        st.markdown("#### \U0001F4C9 Forecast Explorer")
        st.caption("Model comparison and holdout forecast vs. actual, by store and family.")
        st.page_link("views/forecast_explorer.py", label="Open", icon="\u27A1\uFE0F")

with col3:
    with st.container(border=True):
        st.markdown("#### \U0001F6A8 Anomaly View")
        st.caption("Control limits vs. Isolation Forest, synthetic-injection evaluation.")
        st.page_link("views/anomaly_view.py", label="Open", icon="\u27A1\uFE0F")

with col4:
    with st.container(border=True):
        st.markdown("#### \U0001F916 AI Report")
        st.caption("Grounded GenAI narrative with numeric claim verification.")
        st.page_link("views/ai_report.py", label="Open", icon="\u27A1\uFE0F")

st.divider()
st.caption(
    "10 stores \u00d7 6 product families \u2022 15-day walk-forward-validated forecast horizon \u2022 "
    "expanding-window CV, 4 folds + holdout"
)