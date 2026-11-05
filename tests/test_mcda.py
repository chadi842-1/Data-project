import numpy as np
import pandas as pd
import pytest

from invest_agents.config import BOUNDS, FEATURES, ID_COL
from invest_agents.data import generate_projects
from invest_agents.mcda import (
    DEFAULT_PRIORITIES,
    ahp,
    check_pairwise,
    default_ahp,
    priorities_to_pairwise,
    rank_projects,
    topsis,
    topsis_scores,
    weight_sensitivity,
)


def _consistent(w):
    w = np.asarray(w, float)
    return w[:, None] / w[None, :]


# ---- AHP
def test_ahp_recovers_weights_of_a_perfectly_consistent_matrix():
    res = ahp(_consistent([5, 3, 2]), names=["a", "b", "c"])
    assert res.weights == pytest.approx({"a": 0.5, "b": 0.3, "c": 0.2})
    assert res.consistency_ratio == pytest.approx(0, abs=1e-9)
    assert res.is_consistent


def test_ahp_flags_a_contradictory_matrix():
    cyc = np.array([[1, 9, 1 / 9], [1 / 9, 1, 9], [9, 1 / 9, 1]])
    res = ahp(cyc, names=["a", "b", "c"])
    assert res.consistency_ratio > 0.5 and not res.is_consistent


def test_ahp_rejects_bad_matrices():
    with pytest.raises(ValueError, match="reciprocal"):
        check_pairwise(np.array([[1, 3], [3, 1]]))
    with pytest.raises(ValueError, match="square"):
        check_pairwise(np.ones((2, 3)))
    with pytest.raises(ValueError, match="positive"):
        check_pairwise(np.array([[1, -1], [-1, 1]]))
    with pytest.raises(ValueError, match="names"):
        ahp(_consistent([1, 2, 3]), names=["a", "b"])


def test_default_ahp_is_acceptable_and_ordered_by_priority():
    res = default_ahp()
    assert sum(res.weights.values()) == pytest.approx(1.0)
    assert res.is_consistent
    assert max(res.weights, key=res.weights.get) == "rentabilite"
    assert min(res.weights, key=res.weights.get) == "maturite"
    assert set(res.weights) == set(FEATURES)


def test_priorities_matrix_is_reciprocal_and_on_the_saaty_scale():
    m = priorities_to_pairwise(DEFAULT_PRIORITIES)
    check_pairwise(m)
    assert set(np.round(m[m >= 1]).astype(int)) <= set(range(1, 10))


# ---- TOPSIS
def _pool():
    mid = {f: (lo + hi) / 2 for f, (lo, hi) in BOUNDS.items()}
    rows = []
    for i in range(5):
        rows.append({ID_COL: f"P{i}", **mid, "rentabilite": 10 + 2 * i, "risque": 60 - 5 * i})
    return pd.DataFrame(rows)


def test_dominating_project_ranks_first_and_dominated_last():
    df = _pool()
    ranked = topsis(df, default_ahp().weights)
    assert ranked.iloc[0][ID_COL] == "P4" and ranked.iloc[-1][ID_COL] == "P0"
    assert ranked["rank"].tolist() == [1, 2, 3, 4, 5]
    assert ranked["topsis_score"].between(0, 1).all()


def test_cost_criteria_are_minimised():
    df = _pool().iloc[:2].copy()
    df["rentabilite"], df["risque"] = 12.0, 40.0
    df.loc[0, "cout"], df.loc[1, "cout"] = 5.0, 50.0       # same except cost
    ranked = topsis(df, default_ahp().weights)
    assert ranked.iloc[0][ID_COL] == "P0"                   # the cheaper one wins


def test_score_is_invariant_to_rescaling_a_column():
    df = _pool()
    w = default_ahp().weights
    a = topsis(df, w).sort_values(ID_COL)["topsis_score"].to_numpy()
    scaled = df.copy()
    scaled["cout"] = scaled["cout"] * 10
    b = topsis(scaled, w).sort_values(ID_COL)["topsis_score"].to_numpy()
    assert np.allclose(a, b)


def test_degenerate_pool_does_not_crash():
    X = np.ones((3, len(FEATURES)))
    s = topsis_scores(X, np.full(len(FEATURES), 1 / len(FEATURES)), np.ones(len(FEATURES)))
    assert np.allclose(s, 0.5)


def test_rank_projects_wrapper():
    ranked, res = rank_projects(generate_projects(n=12, defects=False))
    assert len(ranked) == 12 and res.is_consistent


# ---- sensitivity
def test_sensitivity_is_stable_when_weights_are_certain():
    df = _pool()
    sens = weight_sensitivity(df, default_ahp().weights, n_sims=200, concentration=1e7)
    assert sens.spearman_mean == pytest.approx(1.0, abs=1e-3)
    assert (sens.per_project["rank_std"] < 0.05).all()
    assert sens.per_project.loc["P4", "p_top1"] == pytest.approx(1.0)


def test_sensitivity_grows_with_weight_uncertainty():
    df = generate_projects(n=12, seed=3, defects=False)
    w = default_ahp().weights
    tight = weight_sensitivity(df, w, n_sims=300, concentration=500, seed=1)
    loose = weight_sensitivity(df, w, n_sims=300, concentration=5, seed=1)
    assert loose.per_project["rank_std"].mean() > tight.per_project["rank_std"].mean()
    assert loose.spearman_mean < tight.spearman_mean


def test_sensitivity_is_reproducible_and_well_formed():
    df = generate_projects(n=10, seed=4, defects=False)
    w = default_ahp().weights
    a = weight_sensitivity(df, w, n_sims=100, seed=9)
    b = weight_sensitivity(df, w, n_sims=100, seed=9)
    pd.testing.assert_frame_equal(a.per_project, b.per_project)
    assert a.per_project["p_top1"].sum() == pytest.approx(1.0)
    assert a.per_project["base_rank"].tolist() == sorted(a.per_project["base_rank"])
