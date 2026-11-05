import json

import pytest

from invest_agents.config import get_paths
from invest_agents.pipeline import main, ranking_validation, render_results, run_pipeline


def test_ranking_validation_structure_and_sanity(trained_home):
    val = ranking_validation(trained_home, n_pools=40, pool_size=10, seed=1)
    assert val["chance_pick_rank"] == 5.5
    for key in ("topsis_vs_truth", "ml_vs_truth", "topsis_vs_ml"):
        assert -1 <= val[key]["p05"] <= val[key]["mean"] <= val[key]["p95"] <= 1
    # both rankings should carry real signal about the true utility
    assert val["topsis_vs_truth"]["mean"] > 0.2 and val["ml_vs_truth"]["mean"] > 0.2
    assert 1 <= val["topsis_pick_true_rank"]["mean"] < val["chance_pick_rank"]


def test_ranking_validation_is_reproducible_and_checks_pool_size(trained_home):
    a = ranking_validation(trained_home, n_pools=20, pool_size=8, seed=3)
    b = ranking_validation(trained_home, n_pools=20, pool_size=8, seed=3)
    assert a == b
    with pytest.raises(ValueError, match="need"):
        ranking_validation(trained_home, pool_size=10_000)


def test_full_pipeline_writes_every_report(tmp_path):
    paths = get_paths(tmp_path)
    run_pipeline(paths, n=500, n_iter=2, n_boot=100, cv_folds=3, n_jobs=1, seed=2)
    for f in ("metrics.json", "ranking_validation.json", "workflow_demo.json", "ahp.json",
              "etl_report.json", "run_config.json", "results.md", "model_comparison.png",
              "feature_importance.png", "shap_summary.png"):
        assert (paths.reports / f).exists(), f
    assert (paths.artifacts / "model.joblib").exists()
    demo = json.loads((paths.reports / "workflow_demo.json").read_text())
    assert len(demo["decision"]["recommendations"]) == 12
    md = (paths.reports / "results.md").read_text()
    for heading in ("## Data and ETL", "## Models", "## Ranking validation", "## AHP weights"):
        assert heading in md
    assert md == render_results(paths)             # rendering is a pure function of the JSON files


def test_cli_entry_point(tmp_path):
    main(["--root", str(tmp_path), "--n", "400", "--n-iter", "2", "--n-boot", "50"])
    assert (tmp_path / "reports" / "results.md").exists()
