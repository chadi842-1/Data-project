"""Three-agent LangGraph workflow: ingestion -> analysis -> decision.

Each agent is a graph node that calls explicit tools and returns Pydantic-validated output.
No LLM is called: the agents are deterministic, reproducible and run offline. To let an LLM
write the human-readable rationale, pass `rationale_fn` to `run_workflow`; it receives the
already-computed numbers, so the LLM explains decisions but never makes them.

- ingestion agent: validates every record, reports per-record problems instead of failing.
- analysis agent: P(invest) from the trained model plus the SHAP drivers behind it.
- decision agent: AHP weights + TOPSIS ranking + rank stability under weight uncertainty,
  combined with the model probability under an explicit, auditable DecisionPolicy.
"""
from __future__ import annotations

import math
import operator
from collections.abc import Callable
from functools import partial
from typing import Annotated, Literal, TypedDict

import numpy as np
import pandas as pd
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field, ValidationError, model_validator

from .config import FEATURES, ID_COL
from .mcda import ahp, default_ahp, topsis, weight_sensitivity
from .schemas import ProjectIn
from .scoring import Scorer, get_scorer

Verdict = Literal["INVEST", "REVIEW", "REJECT"]


# ------------------------------------------------------------------ typed inputs / outputs
class DecisionPolicy(BaseModel):
    """The committee rules, explicit and editable. All thresholds are applied in this order:
    knockout -> INVEST -> REJECT -> REVIEW."""

    invest_proba: float = Field(0.6, ge=0, le=1, description="min P(invest) to invest")
    reject_proba: float = Field(0.4, ge=0, le=1, description="P(invest) below this may be rejected")
    top_fraction: float = Field(0.5, gt=0, le=1, description="share of the pool counted as 'top'")
    min_conformite: float = Field(40.0, ge=0, le=100, description="compliance knockout threshold")
    ahp_pairwise: list[list[float]] | None = Field(
        None, description="8x8 Saaty pairwise matrix in criteria order; default = illustrative")
    n_sensitivity_sims: int = Field(300, ge=10, le=5000)
    weight_concentration: float = Field(50.0, gt=0, description="Dirichlet concentration")

    @model_validator(mode="after")
    def _ordered(self):
        if self.reject_proba > self.invest_proba:
            raise ValueError("reject_proba must not exceed invest_proba")
        return self


class Issue(BaseModel):
    project_id: str
    problems: list[str]


class IngestionReport(BaseModel):
    n_received: int
    n_valid: int
    rejected: list[Issue] = Field(default_factory=list)


class Driver(BaseModel):
    feature: str
    value: float
    shap: float


class ProjectAnalysis(BaseModel):
    project_id: str
    proba_go: float = Field(ge=0, le=1)
    ml_go: bool
    drivers: list[Driver]


class Recommendation(BaseModel):
    project_id: str
    rank: int
    topsis_score: float
    proba_go: float
    rank_mean: float
    rank_std: float
    verdict: Verdict
    knockout: str | None = None
    rationale: str


class DecisionOutput(BaseModel):
    ahp_weights: dict[str, float]
    ahp_consistency_ratio: float
    rank_stability: float | None = None
    recommendations: list[Recommendation] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class WorkflowResult(BaseModel):
    model: str
    threshold: float
    ingestion: IngestionReport
    analysis: list[ProjectAnalysis]
    decision: DecisionOutput
    trace: list[str]


class State(TypedDict, total=False):
    raw: list[dict]
    valid: list[dict]
    ingestion: IngestionReport
    analysis: list[ProjectAnalysis]
    decision: DecisionOutput
    trace: Annotated[list[str], operator.add]


# ----------------------------------------------------------------------------- policy
def verdict_for(
    proba: float, rank: int, n: int, conformite: float, policy: DecisionPolicy
) -> tuple[Verdict, str | None]:
    """Pure decision rule (easy to unit-test and to audit)."""
    if conformite < policy.min_conformite:
        why = f"conformite {conformite:.1f} < {policy.min_conformite:g} (policy knockout)"
        return "REJECT", why
    in_top = rank / n <= policy.top_fraction
    if proba >= policy.invest_proba and in_top:
        return "INVEST", None
    if proba < policy.reject_proba and not in_top:
        return "REJECT", None
    return "REVIEW", None


def default_rationale(info: dict) -> str:
    """Template rationale. Pass `rationale_fn` to let an LLM word it (the numbers stay fixed)."""
    parts = [f"Rank {info['rank']}/{info['n']} (TOPSIS {info['topsis_score']:.2f}), "
             f"P(invest) {info['proba_go']:.0%}."]
    if info["knockout"]:
        parts.append(f"Rejected by policy: {info['knockout']}.")
    drivers = ", ".join(
        f"{d['feature']} ({'+' if d['shap'] > 0 else '-'})" for d in info["drivers"])
    parts.append(f"Main drivers: {drivers}.")
    if info["rank_std"] >= 1.5:
        parts.append(f"Rank is sensitive to the criteria weights (std {info['rank_std']:.1f}).")
    return " ".join(parts)


# ---------------------------------------------------------------------------- agents
def _is_missing(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def ingestion_agent(state: State) -> dict:
    valid, rejected, seen = [], [], set()
    for i, rec in enumerate(state["raw"]):
        pid = str(rec.get(ID_COL, f"row{i}"))
        problems = []
        if pid in seen:
            problems.append("duplicate project_id")
        missing = [f for f in FEATURES if _is_missing(rec.get(f))]
        if missing:
            problems.append("missing criteria: " + ", ".join(missing))
        else:
            try:
                ProjectIn(**{ID_COL: pid, **{f: rec[f] for f in FEATURES}})
            except ValidationError as e:
                problems += [
                    f"{'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors()]
        if problems:
            rejected.append(Issue(project_id=pid, problems=problems))
        else:
            seen.add(pid)
            valid.append({ID_COL: pid, **{f: float(rec[f]) for f in FEATURES}})
    report = IngestionReport(n_received=len(state["raw"]), n_valid=len(valid), rejected=rejected)
    return {"valid": valid, "ingestion": report,
            "trace": [f"ingestion: {len(valid)}/{len(state['raw'])} records valid"]}


def analysis_agent(state: State, scorer: Scorer) -> dict:
    if not state["valid"]:
        return {"analysis": [], "trace": ["analysis: nothing to analyse"]}
    df = pd.DataFrame(state["valid"])
    proba = scorer.predict_proba(df)
    drivers = scorer.explain(df, top_k=3)
    analyses = [
        ProjectAnalysis(
            project_id=pid, proba_go=float(p), ml_go=bool(p >= scorer.threshold),
            drivers=[Driver(**d) for d in ds])
        for pid, p, ds in zip(df[ID_COL], proba, drivers, strict=True)
    ]
    return {"analysis": analyses,
            "trace": [f"analysis: scored {len(analyses)} projects with {scorer.model_name}"]}


def decision_agent(
    state: State, policy: DecisionPolicy, rationale_fn: Callable[[dict], str]
) -> dict:
    res = default_ahp() if policy.ahp_pairwise is None else ahp(np.array(policy.ahp_pairwise))
    warnings = []
    if not res.is_consistent:
        warnings.append(f"AHP consistency ratio {res.consistency_ratio:.2f} >= 0.10: "
                        "revise the pairwise judgments")
    out = DecisionOutput(ahp_weights=res.weights, ahp_consistency_ratio=res.consistency_ratio,
                         warnings=warnings)
    valid = state["valid"]
    if len(valid) < 2:
        out.warnings.append("need at least 2 valid projects to rank; no recommendation made")
        return {"decision": out, "trace": ["decision: not enough valid projects to rank"]}

    df = pd.DataFrame(valid)
    ranked = topsis(df, res.weights)
    sens = weight_sensitivity(df, res.weights, n_sims=policy.n_sensitivity_sims,
                              concentration=policy.weight_concentration, seed=0)
    analysis = {a.project_id: a for a in state["analysis"]}
    n = len(ranked)
    recs = []
    for _, row in ranked.iterrows():
        pid = row[ID_COL]
        a = analysis[pid]
        verdict, knockout = verdict_for(a.proba_go, int(row["rank"]), n, row["conformite"], policy)
        stab = sens.per_project.loc[pid]
        info = {"project_id": pid, "rank": int(row["rank"]), "n": n,
                "topsis_score": float(row["topsis_score"]), "proba_go": a.proba_go,
                "verdict": verdict, "knockout": knockout, "rank_std": float(stab["rank_std"]),
                "drivers": [d.model_dump() for d in a.drivers]}
        recs.append(Recommendation(
            project_id=pid, rank=info["rank"], topsis_score=round(info["topsis_score"], 4),
            proba_go=round(a.proba_go, 4), rank_mean=round(float(stab["rank_mean"]), 2),
            rank_std=round(info["rank_std"], 2), verdict=verdict, knockout=knockout,
            rationale=rationale_fn(info)))
    out.recommendations = recs
    out.rank_stability = round(sens.spearman_mean, 4)
    counts = {v: sum(r.verdict == v for r in recs) for v in ("INVEST", "REVIEW", "REJECT")}
    return {"decision": out, "trace": [f"decision: ranked {n} projects, verdicts {counts}"]}


# ---------------------------------------------------------------------------- workflow
def build_graph(
    scorer: Scorer, policy: DecisionPolicy, rationale_fn: Callable[[dict], str] = default_rationale
):
    g = StateGraph(State)
    g.add_node("ingestion", ingestion_agent)
    g.add_node("analysis", partial(analysis_agent, scorer=scorer))
    g.add_node("decision", partial(decision_agent, policy=policy, rationale_fn=rationale_fn))
    g.add_edge(START, "ingestion")
    g.add_edge("ingestion", "analysis")
    g.add_edge("analysis", "decision")
    g.add_edge("decision", END)
    return g.compile()


def run_workflow(
    projects: list[dict | BaseModel],
    policy: DecisionPolicy | None = None,
    scorer: Scorer | None = None,
    rationale_fn: Callable[[dict], str] | None = None,
) -> WorkflowResult:
    """Run the three agents on a pool of projects and return a fully typed result."""
    scorer = scorer or get_scorer()
    rows = [p.model_dump() if isinstance(p, BaseModel) else dict(p) for p in projects]
    graph = build_graph(scorer, policy or DecisionPolicy(), rationale_fn or default_rationale)
    final = graph.invoke({"raw": rows, "trace": []})
    return WorkflowResult(
        model=scorer.model_name, threshold=scorer.threshold, ingestion=final["ingestion"],
        analysis=final["analysis"], decision=final["decision"], trace=final["trace"])
