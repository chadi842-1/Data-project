import pandas as pd
import pytest

from invest_agents.config import get_paths
from invest_agents.experiments import (
    MODELS,
    render_report,
    run_experiment,
    run_study,
    summarise,
)


@pytest.fixture(scope="module")
def raw():
    return run_study(levels=(0.0, 2.0), seeds=range(3), n=400)


def test_study_has_one_row_per_level_seed_and_model(raw):
    assert len(raw) == 2 * 3 * len(MODELS)
    assert set(raw["model"]) == set(MODELS)
    assert raw["auc"].between(0.5, 1).all() and raw["f1"].between(0, 1).all()


def test_study_is_reproducible():
    a = run_study(levels=(1.0,), seeds=range(2), n=300)
    b = run_study(levels=(1.0,), seeds=range(2), n=300)
    pd.testing.assert_frame_equal(a, b)


def test_additive_regime_is_easy_for_the_linear_model(raw):
    lr = raw[(raw["model"] == "LogisticRegression") & (raw["nonlinearity"] == 0.0)]
    assert lr["auc"].mean() > 0.8


def test_summary_columns_and_gaps_are_consistent(raw):
    s = summarise(raw)
    assert list(s["nonlinearity"]) == [0.0, 2.0]
    gap = s["auc_XGBoost"] - s["auc_LogisticRegression"]
    assert gap.tolist() == pytest.approx(s["auc_gap_XGBoost"].tolist())
    assert s["auc_gap_wins_LightGBM"].between(0, 1).all()


def test_report_and_files_are_written(tmp_path):
    summary = run_experiment(get_paths(tmp_path), n_seeds=2, n=300)
    for f in ("nonlinearity_study.csv", "nonlinearity_study.png", "nonlinearity_study.md"):
        assert (tmp_path / "reports" / f).exists()
    md = (tmp_path / "reports" / "nonlinearity_study.md").read_text()
    assert md.startswith("# When do trees beat")
    assert len(summary) == 5


def test_render_report_lists_every_level(raw):
    md = render_report(raw, summarise(raw))
    assert "| 0 |" in md and "| 2 |" in md
