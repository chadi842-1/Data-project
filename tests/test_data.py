import numpy as np

from invest_agents.config import BOUNDS, FEATURES, TARGET
from invest_agents.data import (
    N_CORRUPT,
    N_DUPLICATES,
    correlation_matrix,
    generate_projects,
    latent_utility,
)


def test_clean_generation_shape_and_bounds():
    df = generate_projects(n=500, defects=False)
    assert len(df) == 500
    assert not df[FEATURES].isna().any().any()
    for f, (lo, hi) in BOUNDS.items():
        assert df[f].between(lo, hi).all(), f


def test_defects_add_expected_rows():
    df = generate_projects(n=500, defects=True)
    assert len(df) == 500 + N_DUPLICATES + N_CORRUPT
    assert df.duplicated(subset=FEATURES + [TARGET]).sum() == N_DUPLICATES
    assert df[FEATURES].isna().any().any()
    out_of_bounds = sum((~df[f].between(*BOUNDS[f]) & df[f].notna()).sum() for f in FEATURES)
    assert out_of_bounds == N_CORRUPT


def test_same_seed_is_reproducible_and_seeds_differ():
    a = generate_projects(n=200, seed=1)
    b = generate_projects(n=200, seed=1)
    c = generate_projects(n=200, seed=2)
    assert a.equals(b)
    assert not a[FEATURES].equals(c[FEATURES])


def test_prevalence_matches_invest_rate():
    df = generate_projects(n=1000, invest_rate=0.45, defects=False)
    assert df[TARGET].sum() == 450


def test_correlation_structure_is_realistic():
    df = generate_projects(n=6000, defects=False)
    corr = df[FEATURES].corr(method="spearman")
    assert corr.loc["rentabilite", "risque"] > 0.25
    assert corr.loc["delai", "cout"] > 0.35
    assert corr.loc["risque", "maturite"] < -0.2
    assert np.linalg.eigvalsh(correlation_matrix()).min() > 0


def test_additive_regime_label_is_exactly_the_linear_utility():
    df = generate_projects(n=800, nonlinearity=0.0, noise_sd=0.0, defects=False)
    util = latent_utility(df, nonlinearity=0.0)
    expected = (util > np.quantile(util, 1 - 0.45)).astype(int)
    assert (expected == df[TARGET].to_numpy()).all()


def test_nonlinear_terms_change_the_utility():
    df = generate_projects(n=800, defects=False)
    assert not np.allclose(latent_utility(df, 0.0), latent_utility(df, 1.0))


def test_knockout_rule_lowers_utility():
    df = generate_projects(n=10, defects=False).iloc[[0]].copy()
    df["conformite"] = 80.0
    good = latent_utility(df, 1.0)[0]
    df["conformite"] = 30.0
    bad = latent_utility(df, 1.0)[0]
    assert bad < good - 1.0
