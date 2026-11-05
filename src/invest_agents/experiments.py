"""Experiment: when do gradient-boosted trees beat logistic regression?

The data generator has a `nonlinearity` knob (0 = purely additive utility, 1 = the default
committee rule set, >1 = stronger knockouts and interactions). For each level we generate several
independent datasets (seeds), fit the three models with FIXED, reasonable hyper-parameters (no
tuning, to isolate the effect of the function class) and compare held-out performance.

AUC is the main yardstick (threshold-free, less noisy); F1 uses the threshold picked on validation.

    python -m invest_agents.experiments [--seeds 8] [--n 2000]
"""
from __future__ import annotations

import argparse

import lightgbm as lgb
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression

from . import viz
from .config import FEATURES, SEED, TARGET, Paths, get_paths
from .data import generate_projects
from .etl import split
from .modeling import BASELINE, best_threshold, build_pipeline, evaluate

LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0)
MODELS = ("LogisticRegression", "LightGBM", "XGBoost")


def _fixed_models(seed: int) -> dict:
    return {
        "LogisticRegression": build_pipeline(LogisticRegression(max_iter=2000)),
        "LightGBM": build_pipeline(lgb.LGBMClassifier(
            n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=15,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
            random_state=seed, verbose=-1, n_jobs=1)),
        "XGBoost": build_pipeline(xgb.XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.8,
            colsample_bytree=0.8, eval_metric="logloss", tree_method="hist",
            random_state=seed, n_jobs=1)),
    }


def evaluate_once(df: pd.DataFrame, seed: int) -> list[dict]:
    train, val, test = split(df, seed)
    rows = []
    for name, model in _fixed_models(seed).items():
        model.fit(train[FEATURES], train[TARGET])
        t, _ = best_threshold(val[TARGET].to_numpy(), model.predict_proba(val[FEATURES])[:, 1])
        m = evaluate(test[TARGET].to_numpy(), model.predict_proba(test[FEATURES])[:, 1], t)
        rows.append({"model": name, "f1": m["f1"], "auc": m["auc"]})
    return rows


def run_study(
    levels: tuple[float, ...] = LEVELS, seeds: range | list[int] = range(8), n: int = 2000
) -> pd.DataFrame:
    """Long table: nonlinearity, seed, model, f1, auc."""
    out = []
    for lam in levels:
        for s in seeds:
            df = generate_projects(n=n, seed=1000 + s, nonlinearity=lam, defects=False)
            for row in evaluate_once(df, seed=SEED + s):
                out.append({"nonlinearity": lam, "seed": s, **row})
    return pd.DataFrame(out)


def summarise(raw: pd.DataFrame) -> pd.DataFrame:
    """Per level: mean/sd of F1 and AUC per model, and the paired AUC gap vs the baseline."""
    rows = []
    for lam, g in raw.groupby("nonlinearity"):
        piv_auc = g.pivot(index="seed", columns="model", values="auc")
        piv_f1 = g.pivot(index="seed", columns="model", values="f1")
        row = {"nonlinearity": lam}
        for m in MODELS:
            row[f"f1_{m}"] = piv_f1[m].mean()
            row[f"f1_sd_{m}"] = piv_f1[m].std(ddof=1)
            row[f"auc_{m}"] = piv_auc[m].mean()
        for m in MODELS:
            if m == BASELINE:
                continue
            gap = piv_auc[m] - piv_auc[BASELINE]
            row[f"auc_gap_{m}"] = gap.mean()
            row[f"auc_gap_sd_{m}"] = gap.std(ddof=1)
            row[f"auc_gap_wins_{m}"] = float((gap > 0).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def plot_study(raw: pd.DataFrame, out_path) -> None:
    import matplotlib.pyplot as plt

    viz.apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for ax, (metric, label) in zip(axes, [("auc", "Test AUC"), ("f1", "Test F1")], strict=True):
        agg = raw.groupby(["model", "nonlinearity"])[metric].agg(["mean", "std"]).reset_index()
        for m in MODELS:
            a = agg[agg["model"] == m].sort_values("nonlinearity")
            color = viz.MODEL_COLORS[m]
            ax.fill_between(a["nonlinearity"], a["mean"] - a["std"], a["mean"] + a["std"],
                            color=color, alpha=0.12, lw=0)
            ax.plot(a["nonlinearity"], a["mean"], "-o", color=color, lw=2, ms=6,
                    mec=viz.SURFACE, mew=1.5)
        last = agg[agg["nonlinearity"] == agg["nonlinearity"].max()].set_index("model")["mean"]
        for m in MODELS:  # direct labels at the right edge, in ink color
            ax.annotate(m, (agg["nonlinearity"].max(), last[m]), xytext=(8, 0),
                        textcoords="offset points", va="center", fontsize=9, color=viz.INK)
        ax.set_xlabel("Nonlinearity of the committee rule (0 = additive)")
        ax.set_xlim(-0.1, agg["nonlinearity"].max() * 1.45)
        viz.title(ax, label, "mean over seeds, band = ±1 sd")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _gap_cell(row: pd.Series, model: str) -> str:
    return (f"{row[f'auc_gap_{model}']:+.3f} ± {row[f'auc_gap_sd_{model}']:.3f}, "
            f"{row[f'auc_gap_wins_{model}']:.0%}")


def render_report(raw: pd.DataFrame, summary: pd.DataFrame) -> str:
    n_seeds = raw["seed"].nunique()
    L = ["# When do trees beat logistic regression?", "",
         "Auto-generated by `python -m invest_agents.experiments`. Synthetic data.", "",
         f"{len(summary)} nonlinearity levels x {n_seeds} independent datasets, three models with "
         "fixed hyper-parameters (no tuning), identical splits for all models.", "",
         "| Nonlinearity | LR AUC | LightGBM AUC | XGBoost AUC | ΔAUC LightGBM-LR "
         "(mean ± sd, share of seeds > 0) | ΔAUC XGBoost-LR |",
         "|---|---|---|---|---|---|"]
    for _, r in summary.iterrows():
        L.append(f"| {r['nonlinearity']:g} | {r['auc_LogisticRegression']:.3f} | "
                 f"{r['auc_LightGBM']:.3f} | {r['auc_XGBoost']:.3f} | "
                 f"{_gap_cell(r, 'LightGBM')} | {_gap_cell(r, 'XGBoost')} |")
    L += ["", "![Nonlinearity study](nonlinearity_study.png)", ""]
    return "\n".join(L)


def run_experiment(paths: Paths | None = None, n_seeds: int = 8, n: int = 2000) -> pd.DataFrame:
    paths = (paths or get_paths()).ensure()
    raw = run_study(seeds=range(n_seeds), n=n)
    summary = summarise(raw)
    raw.to_csv(paths.reports / "nonlinearity_study.csv", index=False)
    plot_study(raw, paths.reports / "nonlinearity_study.png")
    (paths.reports / "nonlinearity_study.md").write_text(render_report(raw, summary))
    print(summary.round(3).to_string(index=False))
    return summary


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--n", type=int, default=2000)
    a = ap.parse_args(argv)
    run_experiment(get_paths(a.root), a.seeds, a.n)


if __name__ == "__main__":
    main()
