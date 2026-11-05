"""Streamlit UI. Run: streamlit run app.py"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))
from agents import run_workflow  # noqa: E402
from config import DATA_DIR, FEATURES  # noqa: E402

st.set_page_config(page_title="Investment decision agents", layout="wide")
st.title("Agentic multicriteria investment decision")
st.caption("3 LangGraph agents (ingestion, analysis, decision) + ML scoring + AHP/TOPSIS. Synthetic data.")

src = st.radio("Data source", ["Sample from test set", "Upload CSV"], horizontal=True)
if src == "Upload CSV":
    up = st.file_uploader("CSV with project_id + " + ", ".join(FEATURES), type="csv")
    df = pd.read_csv(up) if up else None
else:
    n = st.slider("Number of projects", 4, 20, 12)
    df = pd.read_csv(DATA_DIR / "test.csv").dropna(subset=FEATURES).head(n)

if df is not None:
    st.subheader("Input projects")
    st.dataframe(df[["project_id", *FEATURES]], use_container_width=True)
    if st.button("Run agents"):
        res = run_workflow(df[["project_id", *FEATURES]].to_dict("records"))
        st.write("Ingestion:", res["ingestion"])
        dec = res["decision"]
        st.metric("AHP consistency ratio", dec["ahp_consistency_ratio"],
                  help="Below 0.10 is acceptable")
        st.subheader("Ranking")
        st.dataframe(pd.DataFrame(dec["recommendations"]), use_container_width=True)
        st.subheader("AHP weights")
        st.bar_chart(pd.Series(dec["ahp_weights"]))
