"""Streamlit UI.  Run:  streamlit run src/invest_agents/ui.py"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from invest_agents.agents import DecisionPolicy, WorkflowResult, run_workflow
from invest_agents.config import FEATURES, ID_COL, get_paths
from invest_agents.data import generate_projects
from invest_agents.scoring import get_scorer

VERDICT_ICON = {"INVEST": "✅ INVEST", "REVIEW": "🟡 REVIEW", "REJECT": "⛔ REJECT"}


@st.cache_data(show_spinner=False)
def sample_pool() -> pd.DataFrame:
    """Held-out test projects if a pipeline run exists, else a fresh synthetic pool."""
    f = get_paths().data / "test.csv"
    if f.exists():
        return pd.read_csv(f).dropna(subset=FEATURES).reset_index(drop=True)
    return generate_projects(n=200, seed=1, defects=False)


def sidebar_policy() -> DecisionPolicy | None:
    st.sidebar.header("Decision policy")
    invest = st.sidebar.slider("Invest if P(invest) ≥", 0.0, 1.0, 0.6, 0.05)
    reject = st.sidebar.slider("Reject if P(invest) <", 0.0, 1.0, 0.4, 0.05)
    top = st.sidebar.slider("Top share of the pool", 0.1, 1.0, 0.5, 0.05)
    knock = st.sidebar.slider("Compliance knockout below", 0.0, 100.0, 40.0, 5.0)
    sims = st.sidebar.slider("Weight-uncertainty simulations", 50, 2000, 300, 50)
    if reject > invest:
        st.sidebar.error("'Reject below' cannot exceed 'Invest from'.")
        return None
    return DecisionPolicy(invest_proba=invest, reject_proba=reject, top_fraction=top,
                          min_conformite=knock, n_sensitivity_sims=sims)


def load_input() -> pd.DataFrame | None:
    source = st.radio("Projects to evaluate", ["Sample pool", "Upload CSV"], horizontal=True)
    if source == "Upload CSV":
        up = st.file_uploader(f"CSV with columns {ID_COL}, " + ", ".join(FEATURES), type="csv")
        return pd.read_csv(up) if up else None
    n = st.slider("Number of projects", 4, 30, 12)
    return sample_pool().head(n)


def show_result(res: WorkflowResult) -> None:
    ing, dec = res.ingestion, res.decision
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Valid projects", f"{ing.n_valid} / {ing.n_received}")
    c2.metric("AHP consistency ratio", f"{dec.ahp_consistency_ratio:.2f}", help="Below 0.10 is OK")
    stability = "n/a" if dec.rank_stability is None else f"{dec.rank_stability:.2f}"
    c3.metric("Rank stability", stability,
              help="Mean Spearman correlation of the ranking under random weight perturbations")
    c4.metric("Model", res.model)
    for w in dec.warnings:
        st.warning(w)
    if ing.rejected:
        with st.expander(f"{len(ing.rejected)} rejected record(s)"):
            st.dataframe(pd.DataFrame([{"project": i.project_id, "problems": "; ".join(i.problems)}
                                       for i in ing.rejected]), width="stretch")
    if not dec.recommendations:
        return

    st.subheader("Recommendations")
    table = pd.DataFrame([r.model_dump() for r in dec.recommendations])
    table["verdict"] = table["verdict"].map(VERDICT_ICON)
    cols = ["rank", "project_id", "verdict", "topsis_score", "proba_go", "rank_mean", "rank_std"]
    st.dataframe(table[cols], width="stretch", hide_index=True)

    pick = st.selectbox("Why this verdict?", [r.project_id for r in dec.recommendations])
    rec = next(r for r in dec.recommendations if r.project_id == pick)
    st.info(rec.rationale)
    drivers = next(a for a in res.analysis if a.project_id == pick).drivers
    st.dataframe(pd.DataFrame([d.model_dump() for d in drivers]), width="stretch", hide_index=True)

    st.subheader("Criteria weights (AHP)")
    st.bar_chart(pd.Series(dec.ahp_weights, name="weight"))
    with st.expander("Agent trace"):
        st.code("\n".join(res.trace))


def main() -> None:
    st.set_page_config(page_title="Investment decision agents", layout="wide")
    st.title("Agentic multicriteria investment decision")
    st.caption("Three LangGraph agents: ingestion → ML scoring with SHAP → AHP/TOPSIS decision. "
               "Trained on synthetic data.")
    try:
        scorer = get_scorer()
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()
    policy = sidebar_policy()
    df = load_input()
    if df is not None:
        st.dataframe(df.head(50), width="stretch", hide_index=True)
    if st.button("Rank projects", type="primary", disabled=df is None or policy is None):
        cols = [c for c in [ID_COL, *FEATURES] if c in df.columns]
        st.session_state["result"] = run_workflow(
            df[cols].to_dict("records"), policy, scorer)
    if "result" in st.session_state:
        show_result(st.session_state["result"])


main()
