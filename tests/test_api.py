import io

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from invest_agents.agents import WorkflowResult
from invest_agents.api import app
from invest_agents.config import FEATURES, ID_COL
from invest_agents.scoring import get_scorer


@pytest.fixture
def client(home_env):
    return TestClient(app)


@pytest.fixture
def body(home_env):
    df = pd.read_csv(home_env.data / "test.csv").dropna(subset=FEATURES).head(6)
    return [{k: (v if k == ID_COL else float(v)) for k, v in r.items()}
            for r in df[[ID_COL, *FEATURES]].to_dict("records")]


def test_health_ready_and_model_info(client):
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/ready").json()["status"] == "ready"
    info = client.get("/model-info").json()
    assert info["features"] == FEATURES and 0.2 <= info["threshold"] <= 0.8


def test_decide_returns_a_typed_ranking(client, body):
    r = client.post("/decide", json={"projects": body})
    assert r.status_code == 200
    res = WorkflowResult.model_validate(r.json())
    assert sorted(x.rank for x in res.decision.recommendations) == list(range(1, 7))
    assert res.ingestion.n_valid == 6


def test_bad_records_are_reported_not_fatal(client, body):
    body[0]["risque"] = 999.0
    body[1]["impact"] = None
    r = client.post("/decide", json={"projects": body})
    assert r.status_code == 200
    rejected = {i["project_id"] for i in r.json()["ingestion"]["rejected"]}
    assert rejected == {body[0][ID_COL], body[1][ID_COL]}
    assert len(r.json()["decision"]["recommendations"]) == 4


def test_fewer_than_two_valid_is_422_with_reasons(client, body):
    for p in body[1:]:
        p["conformite"] = -5.0
    r = client.post("/decide", json={"projects": body})
    assert r.status_code == 422
    assert r.json()["detail"]["rejected"][0]["problems"]


def test_policy_is_honoured(client, body):
    strict = {"invest_proba": 1.0, "reject_proba": 0.0}
    r = client.post("/decide", json={"projects": body, "policy": strict})
    assert all(x["verdict"] != "INVEST" for x in r.json()["decision"]["recommendations"])


def test_request_level_validation(client, body):
    assert client.post("/decide", json={"projects": []}).status_code == 422
    non_numeric = {"projects": [{**body[0], "cout": "abc"}]}
    assert client.post("/decide", json=non_numeric).status_code == 422
    bad_policy = {"projects": body, "policy": {"invest_proba": 0.2, "reject_proba": 0.8}}
    assert client.post("/decide", json=bad_policy).status_code == 422


def test_openapi_documents_the_endpoints(client):
    paths = client.get("/openapi.json").json()["paths"]
    assert {"/", "/decide", "/decide/csv", "/health", "/ready", "/model-info"} <= set(paths)


def test_root_lists_endpoints(client):
    r = client.get("/")
    assert r.status_code == 200
    body = r.json()
    assert body["service"] == "invest-agents" and "/decide" in body["endpoints"]


def test_every_response_carries_request_id_and_timing(client):
    r = client.get("/health")
    assert r.headers["x-request-id"]
    assert float(r.headers["x-process-time-ms"]) >= 0
    # a client-supplied request id is echoed back
    r2 = client.get("/health", headers={"x-request-id": "trace-123"})
    assert r2.headers["x-request-id"] == "trace-123"


def test_decide_csv_ranks_an_uploaded_pool(client, body):
    df = pd.DataFrame(body)
    buf = io.BytesIO(df.to_csv(index=False).encode())
    r = client.post("/decide/csv", files={"file": ("pool.csv", buf, "text/csv")})
    assert r.status_code == 200
    res = WorkflowResult.model_validate(r.json())
    assert res.ingestion.n_valid == 6 and len(res.decision.recommendations) == 6


def test_decide_csv_honours_policy_query_params(client, body):
    df = pd.DataFrame(body)
    buf = io.BytesIO(df.to_csv(index=False).encode())
    r = client.post("/decide/csv?invest_proba=1.0&reject_proba=0.0",
                    files={"file": ("pool.csv", buf, "text/csv")})
    assert r.status_code == 200
    assert all(x["verdict"] != "INVEST" for x in r.json()["decision"]["recommendations"])


def test_decide_csv_rejects_unparsable_upload(client):
    buf = io.BytesIO(b"\x00\x01not,a,valid\x00csv")
    r = client.post("/decide/csv", files={"file": ("bad.csv", buf, "text/csv")})
    assert r.status_code == 422


def test_service_reports_503_when_the_model_is_missing(tmp_path, monkeypatch, body):
    monkeypatch.setenv("INVEST_AGENTS_HOME", str(tmp_path))
    get_scorer.cache_clear()
    c = TestClient(app)
    assert c.get("/health").status_code == 200           # liveness does not need the model
    assert c.get("/ready").status_code == 503
    r = c.post("/decide", json={"projects": body})
    assert r.status_code == 503 and "pipeline" in r.json()["detail"]
    get_scorer.cache_clear()
