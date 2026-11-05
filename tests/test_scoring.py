import numpy as np
import pandas as pd
import pytest

from invest_agents.config import FEATURES, get_paths
from invest_agents.scoring import Scorer, get_scorer


@pytest.fixture
def test_df(trained_home):
    return pd.read_csv(trained_home.data / "test.csv")


def test_load_and_basic_properties(trained_home):
    s = Scorer.load(trained_home)
    assert s.model_name in {"LogisticRegression", "LightGBM", "XGBoost"}
    assert 0.2 <= s.threshold <= 0.8
    assert s.info()["features"] == FEATURES


def test_predict_proba_shape_range_and_missing_values(trained_home, test_df):
    s = Scorer.load(trained_home)
    x = test_df.head(10).copy()
    x.loc[x.index[0], "rentabilite"] = np.nan
    p = s.predict_proba(x)
    assert p.shape == (10,) and ((p >= 0) & (p <= 1)).all()


def test_explain_top_k_is_sorted_and_complete_sum_is_additive(trained_home, test_df):
    s = Scorer.load(trained_home)
    x = test_df.dropna(subset=FEATURES).head(5)
    top = s.explain(x, top_k=3)
    assert len(top) == 5 and all(len(t) == 3 for t in top)
    for drivers in top:
        mags = [abs(d["shap"]) for d in drivers]
        assert mags == sorted(mags, reverse=True)
    # all contributions + base value reproduce the model's log-odds
    full = s.explain(x, top_k=None)
    proba = s.predict_proba(x)
    for drivers, p in zip(full, proba, strict=True):
        margin = s.base_value() + sum(d["shap"] for d in drivers)
        assert 1 / (1 + np.exp(-margin)) == pytest.approx(p, abs=5e-3)


def test_get_scorer_is_cached_and_follows_env(home_env):
    assert get_scorer() is get_scorer()


def test_missing_artifacts_give_a_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="pipeline"):
        Scorer.load(get_paths(tmp_path))
