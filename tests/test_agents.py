import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from invest_agents.agents import (
    DecisionPolicy,
    WorkflowResult,
    default_rationale,
    run_workflow,
    verdict_for,
)
from invest_agents.config import FEATURES, ID_COL
from invest_agents.mcda import ahp


@pytest.fixture
def pool(home_env):
    df = pd.read_csv(home_env.data / "test.csv").dropna(subset=FEATURES).head(10)
    return df[[ID_COL, *FEATURES]].to_dict("records")


# ---- policy (pure function)
def test_verdict_rule_branches():
    pol = DecisionPolicy()
    assert verdict_for(0.9, 1, 10, 80, pol)[0] == "INVEST"
    assert verdict_for(0.9, 9, 10, 80, pol)[0] == "REVIEW"      # likely but ranked low
    assert verdict_for(0.2, 9, 10, 80, pol)[0] == "REJECT"
    assert verdict_for(0.2, 1, 10, 80, pol)[0] == "REVIEW"      # ranked high, model says no
    v, why = verdict_for(0.99, 1, 10, 39.9, pol)
    assert v == "REJECT" and "knockout" in why


def test_policy_validation():
    with pytest.raises(ValidationError):
        DecisionPolicy(invest_proba=0.3, reject_proba=0.6)
    with pytest.raises(ValidationError):
        DecisionPolicy(invest_proba=1.5)


# ---- workflow
def test_workflow_end_to_end(pool):
    res = run_workflow(pool)
    assert isinstance(res, WorkflowResult)
    assert res.ingestion.n_received == 10 and res.ingestion.n_valid == 10
    assert len(res.analysis) == 10
    ranks = sorted(r.rank for r in res.decision.recommendations)
    assert ranks == list(range(1, 11))
    assert res.decision.ahp_consistency_ratio < 0.10
    assert 0 <= res.decision.rank_stability <= 1
    assert [t.split(":")[0] for t in res.trace] == ["ingestion", "analysis", "decision"]
    for rec in res.decision.recommendations:
        assert rec.verdict in {"INVEST", "REVIEW", "REJECT"}
        assert 0 <= rec.proba_go <= 1 and rec.rank_std >= 0
        assert rec.rationale.startswith("Rank ")


def test_recommendations_are_sorted_by_rank_and_deterministic(pool):
    a, b = run_workflow(pool), run_workflow(pool)
    assert [r.rank for r in a.decision.recommendations] == sorted(
        r.rank for r in a.decision.recommendations)
    assert a.model_dump() == b.model_dump()


def test_result_round_trips_through_json(pool):
    res = run_workflow(pool)
    assert WorkflowResult.model_validate_json(res.model_dump_json()) == res


def test_ingestion_reports_every_kind_of_problem(pool):
    bad = [dict(p) for p in pool]
    bad[0]["risque"] = 150.0                      # out of range
    bad[1]["impact"] = None                        # missing
    bad[2]["delai"] = float("nan")                 # missing (NaN)
    bad[3]["cout"] = "abc"                         # not a number
    bad.append(dict(pool[5]))                      # duplicate id
    res = run_workflow(bad)
    issues = {i.project_id: " ".join(i.problems) for i in res.ingestion.rejected}
    assert res.ingestion.n_received == 11 and res.ingestion.n_valid == 6
    assert "risque" in issues[bad[0][ID_COL]]
    assert "missing" in issues[bad[1][ID_COL]] and "missing" in issues[bad[2][ID_COL]]
    assert "cout" in issues[bad[3][ID_COL]]
    assert "duplicate" in issues[pool[5][ID_COL]]
    ranked_ids = {r.project_id for r in res.decision.recommendations}
    assert len(res.decision.recommendations) == 6
    assert not ranked_ids & {bad[i][ID_COL] for i in range(4)}   # rejected rows never get ranked


def test_knockout_forces_rejection_even_for_a_strong_project(pool):
    strong = [dict(p) for p in pool]
    best = max(strong, key=lambda p: p["rentabilite"])
    best["conformite"] = 20.0
    res = run_workflow(strong)
    rec = next(r for r in res.decision.recommendations if r.project_id == best[ID_COL])
    assert rec.verdict == "REJECT" and "conformite" in rec.knockout
    assert "Rejected by policy" in rec.rationale


def test_fewer_than_two_valid_projects_gives_a_warning_not_a_crash(pool):
    res = run_workflow(pool[:1])
    assert res.decision.recommendations == []
    assert any("at least 2" in w for w in res.decision.warnings)
    empty = run_workflow([])
    assert empty.ingestion.n_valid == 0 and empty.analysis == []


def test_custom_rationale_hook_is_used_and_receives_the_numbers(pool):
    seen = []

    def fake_llm(info):
        seen.append(info)
        return f"LLM says {info['project_id']}"

    res = run_workflow(pool, rationale_fn=fake_llm)
    assert len(seen) == 10 and seen[0]["drivers"] and "proba_go" in seen[0]
    assert all(r.rationale.startswith("LLM says") for r in res.decision.recommendations)


def test_policy_changes_the_verdicts(pool):
    lenient = run_workflow(pool, DecisionPolicy(invest_proba=0.01, reject_proba=0.0,
                                                top_fraction=1.0, min_conformite=0))
    strict = run_workflow(pool, DecisionPolicy(invest_proba=1.0, reject_proba=0.0))
    def n_invest(result):
        return sum(x.verdict == "INVEST" for x in result.decision.recommendations)

    assert n_invest(lenient) == 10 and n_invest(strict) == 0


def test_inconsistent_ahp_matrix_triggers_a_warning(pool):
    rng = np.random.default_rng(0)
    k = len(FEATURES)
    m = np.ones((k, k))
    for i in range(k):
        for j in range(i + 1, k):
            v = float(rng.choice([1 / 9, 1 / 5, 1 / 3, 3, 5, 9]))
            m[i, j], m[j, i] = v, 1 / v
    assert ahp(m).consistency_ratio > 0.10
    res = run_workflow(pool, DecisionPolicy(ahp_pairwise=m.tolist()))
    assert any("consistency" in w for w in res.decision.warnings)


def test_default_rationale_mentions_sensitivity_when_rank_is_unstable():
    info = {"rank": 3, "n": 10, "topsis_score": 0.5, "proba_go": 0.7, "knockout": None,
            "drivers": [{"feature": "risque", "shap": -0.4}], "rank_std": 2.0}
    assert "sensitive" in default_rationale(info)
    info["rank_std"] = 0.2
    assert "sensitive" not in default_rationale(info)
