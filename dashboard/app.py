import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR.parent))

import streamlit as st

st.set_page_config(page_title="RetailCast", page_icon="\U0001F4C8", layout="wide")

home = st.Page(str(APP_DIR / "views" / "home.py"), title="Home Page", icon="\U0001F3E0", default=True)
overview = st.Page(str(APP_DIR / "views" / "overview.py"), title="Overview", icon="\U0001F4CA")
forecast_explorer = st.Page(str(APP_DIR / "views" / "forecast_explorer.py"), title="Forecast Explorer", icon="\U0001F4C9")
anomaly_view = st.Page(str(APP_DIR / "views" / "anomaly_view.py"), title="Anomaly View", icon="\U0001F6A8")
ai_report = st.Page(str(APP_DIR / "views" / "ai_report.py"), title="AI Report", icon="\U0001F916")

pg = st.navigation([home, overview, forecast_explorer, anomaly_view, ai_report])
pg.run()