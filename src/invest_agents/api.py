"""FastAPI service. Run:  uvicorn invest_agents.api:app --port 8000

POST /decide takes a pool of projects (>= 2 valid ones are needed to rank) and an optional
DecisionPolicy. Per-record problems (out of range, missing criteria, duplicate ids) do not fail
the request: they come back in `ingestion.rejected`. If fewer than 2 records are valid the
service answers 422 and lists why.
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import __version__
from .agents import DecisionPolicy, WorkflowResult, run_workflow
from .schemas import ProjectLoose
from .scoring import Scorer, get_scorer

app = FastAPI(
    title="Multicriteria investment decision agents",
    version=__version__,
    description="ML scoring + AHP/TOPSIS ranking orchestrated by three LangGraph agents.",
)


class DecideRequest(BaseModel):
    projects: list[ProjectLoose] = Field(min_length=1, max_length=1000)
    policy: DecisionPolicy | None = None


def _scorer() -> Scorer:
    try:
        return get_scorer()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e


@app.get("/health")
def health() -> dict:
    """Liveness: the process is up."""
    return {"status": "ok", "version": __version__}


@app.get("/ready")
def ready() -> dict:
    """Readiness: the trained model can be loaded."""
    scorer = _scorer()
    return {"status": "ready", "model": scorer.model_name}


@app.get("/model-info")
def model_info() -> dict:
    """Selected model, decision threshold and its held-out test metrics."""
    return _scorer().info()


@app.post("/decide", response_model=WorkflowResult)
def decide(req: DecideRequest) -> WorkflowResult:
    scorer = _scorer()
    result = run_workflow(req.projects, req.policy, scorer)
    if result.ingestion.n_valid < 2:
        raise HTTPException(status_code=422, detail={
            "message": "Need at least 2 valid projects to rank.",
            "rejected": [i.model_dump() for i in result.ingestion.rejected],
        })
    return result
