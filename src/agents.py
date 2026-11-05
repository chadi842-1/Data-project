"""Multi-agent decision workflow (LangGraph): ingestion -> analysis -> decision.

Each agent is a graph node that calls explicit tools (plain Python functions) and
returns output validated by Pydantic models (typed JSON).

NOTE: no LLM is called here. The agents are deterministic tool-calling nodes, which
keeps the pipeline reproducible and runnable offline. An LLM can be plugged into
`decision_agent` to write the narrative rationale; see README.
"""
from typing import TypedDict

import numpy as np
import pandas as pd
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field, ValidationError

from config import BOUNDS, FEATURES
from etl import ProjectRecord
from mcda import rank_projects
from models import explain_one


# ---------- validated outputs (Pydantic) ----------
class IngestionOutput(BaseModel):
    n_received: int
    n_valid: int
    rejected: list[str] = Field(default_factory=list)


class ProjectAnalysis(BaseModel):
    project_id: str
    proba_go: float = Field(ge=0, le=1)
    top_drivers: list[str]


class AnalysisOutput(BaseModel):
    analyses: list[ProjectAnalysis]


class Recommendation(BaseModel):
    project_id: str
    rank: int
    topsis_score: float
    proba_go: float
    verdict: str  # "INVEST" | "REVIEW" | "REJECT"
    rationale: str


class DecisionOutput(BaseModel):
    ahp_consistency_ratio: float
    ahp_weights: dict[str, float]
    recommendations: list[Recommendation]


class State(TypedDict, total=False):
    raw: list[dict]
    valid: list[dict]
    ingestion: dict
    analysis: dict
    decision: dict


# ---------- tools ----------
def tool_validate(rec: dict) -> ProjectRecord:
    """Schema check on one record. Missing numeric values fail validation here."""
    return ProjectRecord(**rec)


def tool_score_and_explain(rec: dict) -> dict:
    return explain_one(rec)


def tool_rank(df: pd.DataFrame):
    return rank_projects(df)


# ---------- agents (graph nodes) ----------
def ingestion_agent(state: State) -> State:
    valid, rejected = [], []
    for rec in state["raw"]:
        try:
            tool_validate(rec)
            valid.append(rec)
        except ValidationError as e:
            rejected.append(f"{rec.get('project_id', '?')}: {e.errors()[0]['loc'][0]}")
    out = IngestionOutput(n_received=len(state["raw"]), n_valid=len(valid), rejected=rejected)
    return {"valid": valid, "ingestion": out.model_dump()}


def analysis_agent(state: State) -> State:
    analyses = []
    for rec in state["valid"]:
        res = tool_score_and_explain(rec)
        drivers = [f"{k} ({'+' if v > 0 else '-'})" for k, v in list(res["shap"].items())[:3]]
        analyses.append(ProjectAnalysis(project_id=rec["project_id"],
                                        proba_go=res["proba_go"], top_drivers=drivers))
    return {"analysis": AnalysisOutput(analyses=analyses).model_dump()}


def decision_agent(state: State) -> State:
    df = pd.DataFrame(state["valid"])
    ranked, weights, cr = tool_rank(df)
    proba = {a["project_id"]: a for a in state["analysis"]["analyses"]}
    recs = []
    for _, r in ranked.iterrows():
        a = proba[r["project_id"]]
        p, s = a["proba_go"], float(r["topsis_score"])
        verdict = "INVEST" if (p >= 0.6 and s >= 0.5) else "REJECT" if (p < 0.4 and s < 0.5) else "REVIEW"
        recs.append(Recommendation(
            project_id=r["project_id"], rank=int(r["rank"]), topsis_score=round(s, 4),
            proba_go=round(p, 4), verdict=verdict,
            rationale=f"TOPSIS {s:.2f} (rank {int(r['rank'])}), P(go)={p:.2f}; "
                      f"main drivers: {', '.join(a['top_drivers'])}"))
    out = DecisionOutput(ahp_consistency_ratio=round(cr, 4),
                         ahp_weights={k: round(v, 4) for k, v in weights.items()},
                         recommendations=recs)
    return {"decision": out.model_dump()}


def build_graph():
    g = StateGraph(State)
    g.add_node("ingestion", ingestion_agent)
    g.add_node("analysis", analysis_agent)
    g.add_node("decision", decision_agent)
    g.add_edge(START, "ingestion")
    g.add_edge("ingestion", "analysis")
    g.add_edge("analysis", "decision")
    g.add_edge("decision", END)
    return g.compile()


def run_workflow(projects: list[dict]) -> dict:
    """Entry point used by the API and the Streamlit app."""
    clean = [{k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in p.items()}
             for p in projects]
    return build_graph().invoke({"raw": clean})


if __name__ == "__main__":
    from config import DATA_DIR
    df = pd.read_csv(DATA_DIR / "test.csv").dropna(subset=FEATURES).head(12)
    cols = ["project_id", *FEATURES]
    res = run_workflow(df[cols].to_dict("records"))
    print(res["ingestion"])
    for r in res["decision"]["recommendations"][:5]:
        print(r["rank"], r["project_id"], r["verdict"], r["rationale"])
