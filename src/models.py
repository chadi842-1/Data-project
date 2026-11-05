"""Train and compare 3 models (logistic regression, LightGBM, XGBoost); explain with SHAP.

Model selection uses the VALIDATION set; the TEST set is only used once for the final report.
"""
import json
from functools import lru_cache

import joblib
import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score

from config import ARTIFACTS_DIR, DATA_DIR, FEATURES, SEED, TARGET

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load():
    pre = joblib.load(ARTIFACTS_DIR / "preprocessor.joblib")
    parts = {k: pd.read_csv(DATA_DIR / f"{k}.csv") for k in ("train", "val", "test")}
    X = {k: pre.transform(v[FEATURES]) for k, v in parts.items()}
    y = {k: v[TARGET].values for k, v in parts.items()}
    return X, y, pre


def make_models():
    return {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=SEED),
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=300, learning_rate=0.05, num_leaves=15,
            min_child_samples=15, subsample=0.8, colsample_bytree=0.8,
            random_state=SEED, verbose=-1),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=4,
            subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
            random_state=SEED),
    }


def run() -> dict:
    X, y, pre = load()
    models, results = make_models(), {}
    for name, m in models.items():
        m.fit(X["train"], y["train"])
        pv, pt = m.predict(X["val"]), m.predict(X["test"])
        results[name] = {
            "f1_val": round(float(f1_score(y["val"], pv)), 4),
            "f1_test": round(float(f1_score(y["test"], pt)), 4),
            "auc_test": round(float(roc_auc_score(y["test"], m.predict_proba(X["test"])[:, 1])), 4),
        }
        print(f"[MODEL] {name:20s} {results[name]}")

    best = max(results, key=lambda k: results[k]["f1_val"])  # pick on validation
    results["best_model"] = best
    base, top = results["LogisticRegression"]["f1_test"], results[best]["f1_test"]
    results["f1_test_gain_over_baseline"] = round(top - base, 4)
    print(f"[MODEL] best on validation: {best} | test F1 {top} vs baseline {base}")

    joblib.dump(models[best], ARTIFACTS_DIR / "best_model.joblib")
    (ARTIFACTS_DIR / "metrics.json").write_text(json.dumps(results, indent=2))

    # SHAP on the best model (tree explainer for trees, linear for LR)
    mdl = models[best]
    Xt = pd.DataFrame(X["test"], columns=FEATURES)
    explainer = (shap.LinearExplainer(mdl, X["train"]) if best == "LogisticRegression"
                 else shap.TreeExplainer(mdl))
    sv = explainer.shap_values(Xt)
    sv = sv[1] if isinstance(sv, list) else sv
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[:, :, 1]
    shap.summary_plot(sv, Xt, show=False)
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "shap_summary.png", dpi=130)
    plt.close()
    imp = pd.Series(np.abs(sv).mean(axis=0), index=FEATURES).sort_values(ascending=False)
    imp.to_csv(ARTIFACTS_DIR / "shap_importance.csv", header=["mean_abs_shap"])
    print("[SHAP] top drivers:", ", ".join(f"{k}={v:.3f}" for k, v in imp.head(4).items()))
    return results


@lru_cache(maxsize=1)
def _load_explainer():
    """Load preprocessor, model and SHAP explainer once per process."""
    pre = joblib.load(ARTIFACTS_DIR / "preprocessor.joblib")
    mdl = joblib.load(ARTIFACTS_DIR / "best_model.joblib")
    train = pre.transform(pd.read_csv(DATA_DIR / "train.csv")[FEATURES])
    if isinstance(mdl, LogisticRegression):
        ex = shap.LinearExplainer(mdl, shap.maskers.Independent(train, max_samples=len(train)))
    else:
        ex = shap.TreeExplainer(mdl)
    return pre, mdl, ex


def explain_one(row: dict) -> dict:
    """Probability + per-feature SHAP contributions for one project (used by the agents)."""
    pre, mdl, ex = _load_explainer()
    x = pre.transform(pd.DataFrame([row])[FEATURES])
    proba = float(mdl.predict_proba(x)[0, 1])
    sv = ex.shap_values(x)
    sv = sv[1] if isinstance(sv, list) else sv
    sv = np.asarray(sv)
    sv = sv[0, :, 1] if sv.ndim == 3 else sv[0]
    contrib = dict(sorted(zip(FEATURES, map(float, sv)), key=lambda kv: -abs(kv[1])))
    return {"proba_go": proba, "shap": contrib}


if __name__ == "__main__":
    run()
