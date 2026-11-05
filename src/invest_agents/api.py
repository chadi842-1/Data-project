"""FastAPI service. Run:  uvicorn invest_agents.api:app --port 8000

POST /decide takes a pool of projects (>= 2 valid ones are needed to rank) and an optional
DecisionPolicy. Per-record problems (out of range, missing criteria, duplicate ids) do not fail
the request: they come back in `ingestion.rejected`. If fewer than 2 records are valid the
service answers 422 and lists why. POST /decide/csv does the same from an uploaded CSV.

Operational niceties: a request id and latency on every log line and response header, configurable
CORS, and a uniform JSON error envelope for unexpected failures.
"""
from __future__ import annotations

import io
import logging
import os
import time
import uuid
from collections.abc import Sequence
from typing import Annotated, Any

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from . import __version__
from .agents import DecisionPolicy, WorkflowResult, run_workflow
from .config import FEATURES, ID_COL
from .schemas import ProjectLoose
from .scoring import Scorer, get_scorer

logger = logging.getLogger("invest_agents.api")
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=os.environ.get("INVEST_AGENTS_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

_EXAMPLE: dict[str, Any] = {
    "projects": [
        {"project_id": "A", "rentabilite": 22, "risque": 35, "delai": 18, "impact": 70,
         "cout": 12, "liquidite": 60, "maturite": 70, "conformite": 85},
        {"project_id": "B", "rentabilite": 9, "risque": 80, "delai": 40, "impact": 30,
         "cout": 80, "liquidite": 20, "maturite": 40, "conformite": 55},
    ],
    "policy": {"invest_proba": 0.6, "min_conformite": 40},
}

app = FastAPI(
    title="Multicriteria investment decision agents",
    version=__version__,
    description="ML scoring + AHP/TOPSIS ranking orchestrated by three LangGraph agents.",
)

_cors = os.environ.get("INVEST_AGENTS_CORS_ORIGINS", "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if _cors.strip() == "*" else [o.strip() for o in _cors.split(",") if o],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_request_context(request: Request, call_next):
    """Tag each request with an id and log its method, path, status and latency."""
    rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        elapsed = (time.perf_counter() - start) * 1000
        logger.exception("rid=%s %s %s unhandled error after %.1f ms",
                         rid, request.method, request.url.path, elapsed)
        raise
    elapsed = (time.perf_counter() - start) * 1000
    response.headers["x-request-id"] = rid
    response.headers["x-process-time-ms"] = f"{elapsed:.1f}"
    logger.info("rid=%s %s %s -> %s in %.1f ms",
                rid, request.method, request.url.path, response.status_code, elapsed)
    return response


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Uniform JSON envelope for anything not already an HTTPException."""
    rid = request.headers.get("x-request-id", "")
    return JSONResponse(
        status_code=500,
        content={"error": {"type": exc.__class__.__name__,
                           "message": "internal server error", "request_id": rid}},
    )


class DecideRequest(BaseModel):
    projects: list[ProjectLoose] = Field(min_length=1, max_length=1000)
    policy: DecisionPolicy | None = None

    model_config = ConfigDict(json_schema_extra={"examples": [_EXAMPLE]})


def _scorer() -> Scorer:
    try:
        return get_scorer()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e


def _decide(projects: Sequence[dict | BaseModel], policy: DecisionPolicy | None) -> WorkflowResult:
    scorer = _scorer()
    result = run_workflow(projects, policy, scorer)
    if result.ingestion.n_valid < 2:
        raise HTTPException(status_code=422, detail={
            "message": "Need at least 2 valid projects to rank.",
            "rejected": [i.model_dump() for i in result.ingestion.rejected],
        })
    return result


@app.get("/")
def root() -> dict:
    """Service metadata and where to go next."""
    return {"service": "invest-agents", "version": __version__,
            "docs": "/docs", "openapi": "/openapi.json",
            "endpoints": ["/health", "/ready", "/model-info", "/decide", "/decide/csv"]}


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
    return _decide(req.projects, req.policy)


@app.post("/decide/csv", response_model=WorkflowResult)
async def decide_csv(
    file: Annotated[UploadFile, File(description="CSV: a project_id column and the 8 criteria")],
    invest_proba: Annotated[float | None, Query(ge=0, le=1)] = None,
    reject_proba: Annotated[float | None, Query(ge=0, le=1)] = None,
    min_conformite: Annotated[float | None, Query(ge=0, le=100)] = None,
    top_fraction: Annotated[float | None, Query(gt=0, le=1)] = None,
) -> WorkflowResult:
    """Rank a pool uploaded as CSV. Policy knobs are optional query parameters; omit them for
    the defaults. Unknown columns are ignored; missing criteria are reported per record."""
    raw = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"could not parse CSV: {e}") from e
    if ID_COL not in df.columns:
        df.insert(0, ID_COL, [f"row{i}" for i in range(len(df))])
    keep = [ID_COL, *[c for c in FEATURES if c in df.columns]]
    projects = df[keep].to_dict("records")
    if not projects:
        raise HTTPException(status_code=422, detail="empty CSV")
    overrides = {k: v for k, v in {
        "invest_proba": invest_proba, "reject_proba": reject_proba,
        "min_conformite": min_conformite, "top_fraction": top_fraction}.items() if v is not None}
    policy = DecisionPolicy.model_validate(overrides) if overrides else None
    return _decide(projects, policy)
