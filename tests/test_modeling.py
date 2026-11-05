import numpy as np
import pytest

from invest_agents.config import FEATURES, TARGET
from invest_agents.data import generate_projects
from invest_agents.etl import split
from invest_agents.modeling import (
    BASELINE,
    THRESHOLD_GRID,
    best_threshold,
    best_threshold_cost,
    bootstrap_f1,
    expected_calibration_error,
    expected_value,
    global_explanations,
    make_explainer,
    percentile_ci,
    plot_calibration,
    select_model,
    shap_matrix,
    train_and_evaluate,
    write_model_card,
)


@pytest.fixture(scope="module")
def splits():
    df = generate_projects(n=500, seed=11, defects=False)
    return split(df, seed=11)


@pytest.fixture(scope="module")
def trained(splits):
    train, val, test = splits
    return train_and_evaluate(train, val, test, n_iter=2, cv_folds=3, n_boot=300, n_jobs=1)


# ---- pure functions
def test_bootstrap_f1_matches_a_plain_loop():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 60)
    pred = rng.integers(0, 2, 60)
    idx = rng.integers(0, 60, size=(25, 60))
    from sklearn.metrics import f1_score

    expected = [f1_score(y[i], pred[i], zero_division=0) for i in idx]
    assert np.allclose(bootstrap_f1(y, pred, idx), expected)


def test_best_threshold_on_separable_data_and_tie_break():
    y = np.array([1, 1, 0, 0])
    p = np.array([0.9, 0.8, 0.1, 0.2])
    t, f1 = best_threshold(y, p)
    assert f1 == 1.0 and t == pytest.approx(0.5)
    assert THRESHOLD_GRID.min() <= t <= THRESHOLD_GRID.max()


def test_percentile_ci_orders_bounds():
    lo, hi = percentile_ci(np.arange(101))
    assert lo == pytest.approx(2.5) and hi == pytest.approx(97.5)


def test_expected_calibration_error_is_zero_when_perfect():
    # predicted probability equals the observed rate in every bin -> ECE 0
    assert expected_calibration_error(
        np.array([0, 0, 1, 1]), np.array([0.0, 0.0, 1.0, 1.0])) == pytest.approx(0.0, abs=1e-9)
    # a bin where half the labels are positive and p = 0.5 is also perfectly calibrated
    assert expected_calibration_error(
        np.array([0, 1, 0, 1]), np.array([0.5, 0.5, 0.5, 0.5])) == pytest.approx(0.0, abs=1e-9)
    # a confidently wrong model is maximally miscalibrated
    assert expected_calibration_error(np.array([0, 0]), np.array([1.0, 1.0])) == pytest.approx(1.0)
    assert 0.0 <= expected_calibration_error(
        np.array([0, 1, 0, 1]), np.array([0.3, 0.3, 0.7, 0.7])) <= 1.0


def test_best_threshold_cost_shifts_with_asymmetric_costs():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    p = np.clip(0.5 * y + rng.normal(0.25, 0.2, 200), 0, 1)
    t_fn, _ = best_threshold_cost(y, p, cost_fp=1.0, cost_fn=5.0)   # misses are costly
    t_fp, _ = best_threshold_cost(y, p, cost_fp=5.0, cost_fn=1.0)   # false bets are costly
    assert t_fn <= t_fp
    t_eq, cost = best_threshold_cost(y, p, 1.0, 1.0)
    assert THRESHOLD_GRID.min() <= t_eq <= THRESHOLD_GRID.max() and cost >= 0


def test_parsimony_rule():
    res = {BASELINE: {"val_f1": 0.80}, "LightGBM": {"val_f1": 0.805}, "XGBoost": {"val_f1": 0.83}}
    assert select_model(res, 0.01)[0] == "XGBoost"
    assert select_model(res, 0.05)[0] == BASELINE
    res["XGBoost"]["val_f1"] = 0.805
    assert select_model(res, 0.01)[0] == BASELINE


# ---- training
def test_pipelines_do_preprocessing_inside_the_model(trained):
    for pipe in trained.pipelines.values():
        assert [name for name, _ in pipe.steps] == ["impute", "scale", "clf"]


def test_training_result_is_complete_and_sane(trained):
    assert set(trained.results) == {"LogisticRegression", "LightGBM", "XGBoost"}
    for name, r in trained.results.items():
        assert 0 <= r["test"]["f1"] <= 1 and 0.5 <= r["test"]["auc"] <= 1
        lo, hi = r["test_f1_ci95"]
        assert lo <= hi
        assert THRESHOLD_GRID.min() <= r["threshold"] <= THRESHOLD_GRID.max()
        assert ("f1_diff_vs_baseline" in r) == (name != BASELINE)
    assert trained.best in trained.results


def test_models_predict_probabilities_and_handle_missing(trained, splits):
    _, _, test = splits
    x = test[FEATURES].copy()
    x.iloc[0, 0] = np.nan
    for pipe in trained.pipelines.values():
        p = pipe.predict_proba(x)[:, 1]
        assert p.shape == (len(x),) and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()


def test_models_beat_chance_on_this_dgp(trained):
    assert all(r["test"]["auc"] > 0.75 for r in trained.results.values())


# ---- explanation
def _margin(clf, x):
    name = type(clf).__name__
    if name == "LogisticRegression":
        return clf.decision_function(x)
    if name == "LGBMClassifier":
        return clf.predict(x, raw_score=True)
    return clf.predict(x, output_margin=True)


def test_shap_values_are_additive_for_every_model(trained, splits):
    _, val, _ = splits
    for name, pipe in trained.pipelines.items():
        x = pipe[:-1].transform(val[FEATURES].head(40))
        clf = pipe.named_steps["clf"]
        ex = make_explainer(clf, pipe[:-1].transform(splits[0][FEATURES]))
        sv = shap_matrix(ex, x)
        assert sv.shape == (40, len(FEATURES)), name
        assert np.allclose(sv.sum(1) + expected_value(ex), _margin(clf, x), atol=1e-3), name


def test_training_keeps_test_probabilities_and_reports_calibration(trained, splits):
    _, _, test = splits
    assert trained.y_test is not None and len(trained.y_test) == len(test)
    for name, r in trained.results.items():
        assert trained.test_proba[name].shape == (len(test),)
        assert 0.0 <= r["test"]["ece"] <= 1.0


def test_plot_calibration_and_model_card_write_files(trained, tmp_path):
    plot_calibration(trained, tmp_path / "calibration.png")
    assert (tmp_path / "calibration.png").exists()
    write_model_card(trained, tmp_path / "model_card.md")
    card = (tmp_path / "model_card.md").read_text()
    assert trained.best in card and "calibration error" in card.lower()


def test_global_explanations_write_files(trained, splits, tmp_path):
    _, val, _ = splits
    out = global_explanations(trained.pipelines[BASELINE], val[FEATURES], val[TARGET].to_numpy(),
                              tmp_path)
    assert set(out) == {"shap", "permutation"}
    for fname in ("shap_importance.csv", "permutation_importance.csv", "shap_summary.png",
                  "feature_importance.png"):
        assert (tmp_path / fname).exists()
    assert out["shap"].index[0] in FEATURES
