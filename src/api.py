"""FastAPI service. Run: uvicorn api:app --app-dir src --reload"""
from fastapi import FastAPI, HTTPException

from agents import run_workflow
from config import FEATURES

app = FastAPI(title="Multicriteria investment decision agents")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/decide")
def decide(projects: list[dict]):
    """Body: list of projects with project_id + the 8 criteria. Needs >= 2 projects for ranking."""
    if len(projects) < 2:
        raise HTTPException(422, "Provide at least 2 projects to rank.")
    for p in projects:
        missing = [c for c in ["project_id", *FEATURES] if c not in p]
        if missing:
            raise HTTPException(422, f"{p.get('project_id', '?')}: missing {missing}")
    res = run_workflow(projects)
    if res["ingestion"]["n_valid"] < 2:
        raise HTTPException(422, f"Fewer than 2 valid projects: {res['ingestion']['rejected']}")
    return {"ingestion": res["ingestion"], "decision": res["decision"]}
