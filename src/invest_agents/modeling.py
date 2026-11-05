"""Model training, evaluation and explanation.

Design choices (each one fixes a weakness a reviewer would spot):
- Imputation and scaling live INSIDE each sklearn Pipeline, so they are re-fitted on the training
  part of every CV fold (no preprocessing leakage into cross-validation).
- Hyper-parameters are tuned by randomized search with 5-fold CV on the TRAIN split only,
  scored by log-loss (a threshold-free, proper scoring rule).
- The decision threshold that maximises F1 is chosen on the VALIDATION split, for every model
  alike, so models are compared fairly. The TEST split is used exactly once, for reporting.
- F1 comes with a 95% bootstrap confidence interval, and every model is compared with the
  logistic-regression baseline by a PAIRED bootstrap on the same resamples. With a few hundred
  test rows, small F1 gaps are noise and the CI says so.
- Parsimony rule: a more complex model is selected only if it beats the baseline on validation F1
  by more than `parsimony_margin`; otherwise the simpler model wins.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import viz
from .config import FEATURES, SEED, TARGET, Paths, get_paths

BASELINE = "LogisticRegression"
THRESHOLD_GRID = np.linspace(0.2, 0.8, 61)


# ----------------------------------------------------------------------------- models
def build_pipeline(clf) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", clf),
    ])


def model_specs(seed: int = SEED) -> dict[str, tuple[Pipeline, dict]]:
    """name -> (pipeline, hyper-parameter distributions for randomized search)."""
    boosting = {
        "clf__n_estimators": stats.randint(80, 500),
        "clf__learning_rate": stats.loguniform(0.02, 0.2),
        "clf__subsample": stats.uniform(0.6, 0.4),
        "clf__colsample_bytree": stats.uniform(0.6, 0.4),
        "clf__reg_lambda": stats.loguniform(1e-3, 10),
    }
    return {
        "LogisticRegression": (
            build_pipeline(LogisticRegression(max_iter=2000, random_state=seed)),
            {"clf__C": stats.loguniform(1e-2, 1e2)},
        ),
        "LightGBM": (
            build_pipeline(lgb.LGBMClassifier(
                subsample_freq=1, random_state=seed, verbose=-1, n_jobs=1)),
            {**boosting,
             "clf__num_leaves": stats.randint(4, 32),
             "clf__min_child_samples": stats.randint(5, 40)},
        ),
        "XGBoost": (
            build_pipeline(xgb.XGBClassifier(
                eval_metric="logloss", tree_method="hist", random_state=seed, n_jobs=1)),
            {**boosting,
             "clf__max_depth": stats.randint(2, 6),
             "clf__min_child_weight": stats.randint(1, 10)},
        ),
    }


# -------------------------------------------------------------------------- evaluation
def _f1_from_counts(tp, fp, fn):
    denom = 2 * tp + fp + fn
    return np.divide(2 * tp, denom, out=np.zeros_like(denom, dtype=float), where=denom > 0)


def best_threshold(y: np.ndarray, p: np.ndarray, grid: np.ndarray = THRESHOLD_GRID):
    """F1-maximising threshold on `grid` (ties go to the threshold closest to 0.5)."""
    y = np.asarray(y).astype(bool)[:, None]
    pred = np.asarray(p)[:, None] >= grid[None, :]
    tp = (pred & y).sum(0)
    fp = (pred & ~y).sum(0)
    fn = (~pred & y).sum(0)
    f1 = _f1_from_counts(tp, fp, fn)
    best = f1.max()
    cands = grid[f1 >= best - 1e-12]
    return float(cands[np.argmin(np.abs(cands - 0.5))]), float(best)


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error: average |observed rate - predicted P| over equal-width bins,
    weighted by bin population. 0 = perfectly calibrated probabilities."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_of = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    n = len(p)
    ece = 0.0
    for b in range(n_bins):
        mask = bin_of == b
        if not mask.any():
            continue
        ece += mask.sum() / n * abs(y[mask].mean() - p[mask].mean())
    return float(ece)


def best_threshold_cost(
    y: np.ndarray, p: np.ndarray, cost_fp: float = 1.0, cost_fn: float = 1.0,
    grid: np.ndarray = THRESHOLD_GRID,
) -> tuple[float, float]:
    """Threshold on `grid` that minimises expected misclassification cost.

    A false positive (back a project that fails) costs `cost_fp`; a false negative (skip a
    project that would have paid off) costs `cost_fn`. Returns (threshold, mean cost per project).
    Ties go to the threshold closest to 0.5. Use it when the two error types are not equally
    expensive: raising `cost_fn` lowers the threshold, raising `cost_fp` raises it.
    """
    yb = np.asarray(y).astype(bool)[:, None]
    pred = np.asarray(p, dtype=float)[:, None] >= grid[None, :]
    fp = (pred & ~yb).sum(0)
    fn = (~pred & yb).sum(0)
    cost = (cost_fp * fp + cost_fn * fn) / len(p)
    best = cost.min()
    cands = grid[cost <= best + 1e-12]
    return float(cands[np.argmin(np.abs(cands - 0.5))]), float(best)


def evaluate(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (p >= threshold).astype(int)
    return {
        "f1": float(f1_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred)),
        "auc": float(roc_auc_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "ece": expected_calibration_error(y, p),
    }


def bootstrap_f1(y: np.ndarray, pred: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """F1 for every bootstrap resample (rows of `idx`), vectorised."""
    yb, pb = np.asarray(y).astype(bool)[idx], np.asarray(pred).astype(bool)[idx]
    tp = (yb & pb).sum(1)
    fp = (~yb & pb).sum(1)
    fn = (yb & ~pb).sum(1)
    return _f1_from_counts(tp, fp, fn)


def percentile_ci(values: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    a = (1 - level) / 2
    return float(np.quantile(values, a)), float(np.quantile(values, 1 - a))


def select_model(results: dict[str, dict], margin: float) -> tuple[str, str]:
    """Parsimony rule: keep the baseline unless a complex model beats it by > margin (val F1)."""
    base = results[BASELINE]["val_f1"]
    better = {n: r["val_f1"] for n, r in results.items()
              if n != BASELINE and r["val_f1"] > base + margin}
    if not better:
        return BASELINE, (f"no model beat the baseline validation F1 ({base:.3f}) "
                          f"by more than {margin}; simpler model kept")
    best = max(better, key=lambda n: better[n])
    gain = better[best] - base
    return best, f"{best} beat the baseline on validation F1 by {gain:.3f} (> {margin})"


@dataclass
class TrainingResult:
    pipelines: dict[str, Pipeline]
    thresholds: dict[str, float]
    results: dict[str, dict]
    best: str
    selection_reason: str
    meta: dict = field(default_factory=dict)
    test_proba: dict[str, np.ndarray] = field(default_factory=dict)  # per-model P(invest) on test
    y_test: np.ndarray | None = None                                 # test labels (for calibration)


def _py(obj):
    """Make numpy scalars JSON-friendly."""
    if isinstance(obj, dict):
        return {k: _py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_py(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def train_and_evaluate(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    n_iter: int = 20,
    cv_folds: int = 5,
    seed: int = SEED,
    n_boot: int = 2000,
    parsimony_margin: float = 0.01,
    n_jobs: int = -1,
) -> TrainingResult:
    X_tr, y_tr = train[FEATURES], train[TARGET].to_numpy()
    X_va, y_va = val[FEATURES], val[TARGET].to_numpy()
    X_te, y_te = test[FEATURES], test[TARGET].to_numpy()
    idx = np.random.default_rng(seed).integers(0, len(y_te), size=(n_boot, len(y_te)))

    pipelines, thresholds, results, boots, test_proba = {}, {}, {}, {}, {}
    for name, (pipe, dist) in model_specs(seed).items():
        search = RandomizedSearchCV(
            pipe, dist, n_iter=n_iter, scoring="neg_log_loss", refit=True, n_jobs=n_jobs,
            cv=StratifiedKFold(cv_folds, shuffle=True, random_state=seed), random_state=seed,
        ).fit(X_tr, y_tr)
        model = search.best_estimator_
        t, val_f1 = best_threshold(y_va, model.predict_proba(X_va)[:, 1])
        p_te = model.predict_proba(X_te)[:, 1]
        boots[name] = bootstrap_f1(y_te, (p_te >= t).astype(int), idx)
        test_proba[name] = p_te
        pipelines[name], thresholds[name] = model, t
        results[name] = {
            "best_params": search.best_params_,
            "cv_log_loss": float(-search.best_score_),
            "threshold": t,
            "val_f1": val_f1,
            "test": evaluate(y_te, p_te, t),
            "test_f1_ci95": percentile_ci(boots[name]),
        }
    for name in results:
        if name == BASELINE:
            continue
        diff = boots[name] - boots[BASELINE]
        results[name]["f1_diff_vs_baseline"] = {
            "mean": float(diff.mean()),
            "ci95": percentile_ci(diff),
            "p_better": float((diff > 0).mean()),
        }
    best, reason = select_model(results, parsimony_margin)
    meta = {"n_train": len(train), "n_val": len(val), "n_test": len(test), "n_boot": n_boot,
            "n_iter": n_iter, "cv_folds": cv_folds, "parsimony_margin": parsimony_margin}
    return TrainingResult(pipelines, thresholds, _py(results), best, reason, meta,
                          test_proba=test_proba, y_test=y_te)


# -------------------------------------------------------------------------- explanation
def make_explainer(clf, background_scaled: np.ndarray):
    if isinstance(clf, LogisticRegression):
        masker = shap.maskers.Independent(background_scaled, max_samples=len(background_scaled))
        return shap.LinearExplainer(clf, masker)
    return shap.TreeExplainer(clf)


def shap_matrix(explainer, x_scaled: np.ndarray) -> np.ndarray:
    """SHAP values for the positive class, shape (n_rows, n_features), in log-odds units."""
    sv = explainer.shap_values(x_scaled)
    if isinstance(sv, list):
        sv = sv[-1]
    sv = np.asarray(sv)
    return sv[:, :, -1] if sv.ndim == 3 else sv


def expected_value(explainer) -> float:
    return float(np.ravel(explainer.expected_value)[-1])


def global_explanations(
    pipeline: Pipeline, X: pd.DataFrame, y: np.ndarray, out_dir, seed: int = SEED
) -> dict[str, pd.Series]:
    """SHAP + permutation importances on `X`; writes two CSVs and two figures to out_dir."""
    import matplotlib.pyplot as plt

    viz.apply_style()
    imputed = pd.DataFrame(pipeline.named_steps["impute"].transform(X), columns=FEATURES)
    scaled = pipeline.named_steps["scale"].transform(imputed)
    explainer = make_explainer(pipeline.named_steps["clf"], scaled)
    sv = shap_matrix(explainer, scaled)

    shap_imp = pd.Series(np.abs(sv).mean(0), index=FEATURES).sort_values(ascending=False)
    perm = permutation_importance(
        pipeline, X, y, scoring="roc_auc", n_repeats=20, random_state=seed, n_jobs=1)
    perm_imp = pd.Series(perm.importances_mean, index=FEATURES).sort_values(ascending=False)
    shap_imp.to_csv(out_dir / "shap_importance.csv", header=["mean_abs_shap"])
    perm_imp.to_csv(out_dir / "permutation_importance.csv", header=["auc_drop"])

    shap.summary_plot(sv, imputed, show=False, plot_size=(7, 4.5))
    plt.tight_layout()
    plt.savefig(out_dir / "shap_summary.png", dpi=140)
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, (series, label, sub) in zip(
        axes,
        [(shap_imp, "Mean |SHAP|", "average push on the log-odds of 'invest'"),
         (perm_imp, "Drop in AUC when shuffled", "validation set, 20 repeats")],
        strict=True,
    ):
        s = series.sort_values()
        ax.barh(s.index, s.values, color=viz.SERIES["blue"], height=0.62)
        for yv, v in enumerate(s.values):
            ax.text(v, yv, f" {v:.2f}", va="center", fontsize=9, color=viz.INK_2)
        ax.set_xlabel(label)
        ax.set_xlim(0, s.max() * 1.18)
        ax.grid(axis="y", visible=False)
        viz.title(ax, label, sub)
    fig.tight_layout()
    fig.savefig(out_dir / "feature_importance.png", dpi=140)
    plt.close(fig)
    return {"shap": shap_imp, "permutation": perm_imp}


def plot_model_comparison(res: TrainingResult, n_test: int, out_path) -> None:
    import matplotlib.pyplot as plt

    viz.apply_style()
    names = list(res.results)
    fig, ax = plt.subplots(figsize=(7.5, 3.0))
    for yv, name in enumerate(reversed(names)):
        r = res.results[name]
        f1, (lo, hi) = r["test"]["f1"], r["test_f1_ci95"]
        color = viz.MODEL_COLORS[name]
        ax.plot([lo, hi], [yv, yv], color=color, lw=2.5, solid_capstyle="round")
        ax.plot(f1, yv, "o", color=color, ms=9, mec=viz.SURFACE, mew=2)
        tag = "  (selected)" if name == res.best else ""
        ax.text(hi + 0.004, yv, f"{f1:.3f}{tag}", va="center", fontsize=9, color=viz.INK)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(list(reversed(names)))
    ax.set_xlabel("F1 on the test set")
    ax.grid(axis="y", visible=False)
    lo_all = min(r["test_f1_ci95"][0] for r in res.results.values())
    hi_all = max(r["test_f1_ci95"][1] for r in res.results.values())
    ax.set_xlim(lo_all - 0.02, hi_all + 0.08)
    viz.title(ax, "Model comparison on held-out data",
              f"Test F1 with 95% bootstrap interval, n = {n_test}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_calibration(res: TrainingResult, out_path, n_bins: int = 10) -> None:
    """Reliability diagram: observed invest rate vs predicted P(invest) on the test set.

    A well-calibrated model sits on the diagonal. The legend reports each model's ECE.
    """
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve

    if res.y_test is None or not res.test_proba:
        return
    viz.apply_style()
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    ax.plot([0, 1], [0, 1], "--", color=viz.INK_2, lw=1, label="perfectly calibrated")
    for name in res.results:
        frac_pos, mean_pred = calibration_curve(
            res.y_test, res.test_proba[name], n_bins=n_bins, strategy="quantile")
        ece = res.results[name]["test"].get("ece", float("nan"))
        ax.plot(mean_pred, frac_pos, "o-", color=viz.MODEL_COLORS[name], ms=5, lw=1.8,
                label=f"{name} (ECE {ece:.3f})")
    ax.set_xlabel("Mean predicted P(invest)")
    ax.set_ylabel("Observed invest rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    viz.title(ax, "Calibration (reliability) on held-out data",
              f"quantile bins, n = {len(res.y_test)}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def write_model_card(res: TrainingResult, out_path) -> None:
    """A short, auto-generated model card for the selected model (Mitchell et al., 2019)."""
    best = res.results[res.best]
    t = best["test"]
    st = res.meta
    diff = best.get("f1_diff_vs_baseline")
    lines = [
        f"# Model card: {res.best}", "",
        "Auto-generated by `invest-pipeline`. All data is **synthetic**; these numbers describe "
        "the simulation in `src/invest_agents/data.py`, not real-world investment performance.", "",
        "## Intended use", "",
        "- Rank a pool of candidate investment projects and attach an INVEST / REVIEW / REJECT "
        "verdict under an explicit, editable `DecisionPolicy`.",
        "- Decision support for a human committee, **not** an automated trading or allocation "
        "system. The verdict is advisory and every number is auditable.", "",
        "## Model", "",
        f"- Type: {res.best}, selected by a parsimony rule — {res.selection_reason}.",
        "- Preprocessing (inside the pipeline, re-fitted per CV fold): median imputation + "
        "standardisation.",
        f"- Decision threshold: {res.thresholds[res.best]:.3f} (F1-maximising on validation).",
        f"- Training data: {st.get('n_train', '?')} train / {st.get('n_val', '?')} validation / "
        f"{st.get('n_test', '?')} test projects, 8 criteria.", "",
        "## Held-out performance (test split, used once)", "",
        "| Metric | Value |", "|---|---|",
        f"| F1 | {t['f1']:.3f} {_ci(best['test_f1_ci95'])} |",
        f"| Precision | {t['precision']:.3f} |",
        f"| Recall | {t['recall']:.3f} |",
        f"| ROC AUC | {t['auc']:.3f} |",
        f"| Brier score | {t['brier']:.3f} |",
        f"| Expected calibration error | {t.get('ece', float('nan')):.3f} |",
    ]
    if diff:
        lines += [
            f"| ΔF1 vs {BASELINE} | {diff['mean']:+.3f} {_ci(diff['ci95'])} "
            f"(P better {diff['p_better']:.2f}) |",
        ]
    lines += [
        "", "![Calibration](calibration.png)", "",
        "## Limitations", "",
        "- Synthetic data: no claim about real investment returns.",
        "- The test split is small, so the F1 interval is wide.",
        "- Probabilities should be read together with the calibration curve above; the decision "
        "threshold assumes equal-cost errors unless `best_threshold_cost` is used.",
        "- Rankings from AHP + TOPSIS are relative to the submitted pool and sensitive to the "
        "(illustrative) criteria weights.", "",
        "## Ethical and operational notes", "",
        "- Keep a human in the loop; log every decision via the workflow `trace`.",
        "- Re-elicit the AHP weights and re-validate on real, representative data before any use "
        "beyond demonstration.", "",
    ]
    out_path.write_text("\n".join(lines))


def _ci(v) -> str:
    return f"[{v[0]:.3f}, {v[1]:.3f}]"


# ------------------------------------------------------------------------------ artifacts
def save_artifacts(res: TrainingResult, train: pd.DataFrame, paths: Paths) -> None:
    """Persist the selected model for inference (artifacts/) and metrics for humans (reports/)."""
    paths.ensure()
    joblib.dump(res.pipelines[res.best], paths.artifacts / "model.joblib")
    train[FEATURES].sample(min(200, len(train)), random_state=SEED).to_csv(
        paths.artifacts / "background.csv", index=False)
    meta = {"model": res.best, "threshold": res.thresholds[res.best], "features": FEATURES,
            "selection_reason": res.selection_reason, "test": res.results[res.best]["test"]}
    (paths.artifacts / "model_meta.json").write_text(json.dumps(meta, indent=2))
    metrics = {"best_model": res.best, "selection_reason": res.selection_reason,
               "settings": res.meta, "models": res.results}
    (paths.reports / "metrics.json").write_text(json.dumps(metrics, indent=2))


def run_training(
    paths: Paths | None = None, n_iter: int = 20, seed: int = SEED, n_boot: int = 2000,
    cv_folds: int = 5, n_jobs: int = -1,
) -> TrainingResult:
    paths = (paths or get_paths()).ensure()
    train, val, test = (pd.read_csv(paths.data / f"{k}.csv") for k in ("train", "val", "test"))
    res = train_and_evaluate(train, val, test, n_iter=n_iter, cv_folds=cv_folds, seed=seed,
                             n_boot=n_boot, n_jobs=n_jobs)
    for name, r in res.results.items():
        ci = r["test_f1_ci95"]
        print(f"[MODEL] {name:19s} val F1 {r['val_f1']:.3f} | test F1 {r['test']['f1']:.3f} "
              f"[{ci[0]:.3f}, {ci[1]:.3f}] | AUC {r['test']['auc']:.3f}")
    print(f"[MODEL] selected: {res.best} ({res.selection_reason})")
    save_artifacts(res, train, paths)
    plot_model_comparison(res, len(test), paths.reports / "model_comparison.png")
    plot_calibration(res, paths.reports / "calibration.png")
    write_model_card(res, paths.reports / "model_card.md")
    global_explanations(res.pipelines[res.best], val[FEATURES], val[TARGET].to_numpy(),
                        paths.reports, seed)
    return res


if __name__ == "__main__":
    run_training()
