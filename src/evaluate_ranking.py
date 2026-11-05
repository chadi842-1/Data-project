"""Check how well the AHP+TOPSIS ranking agrees with the ML model's P(go).

Takes the test set, draws many random pools of 12 alternatives, and reports the mean
Spearman correlation between the TOPSIS ranking and the ML probability ranking.
This is agreement between two automated methods, NOT validation against a human expert.
"""
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from config import ARTIFACTS_DIR, DATA_DIR, FEATURES, SEED
from mcda import rank_projects
from models import _load_explainer


def run(n_pools: int = 200, pool_size: int = 12) -> dict:
    df = pd.read_csv(DATA_DIR / "test.csv").dropna(subset=FEATURES).reset_index(drop=True)
    pre, mdl, _ = _load_explainer()
    df["proba"] = mdl.predict_proba(pre.transform(df[FEATURES]))[:, 1]
    rng = np.random.default_rng(SEED)
    rhos = []
    for _ in range(n_pools):
        pool = df.sample(pool_size, random_state=int(rng.integers(1e9)))
        ranked, _, _ = rank_projects(pool)
        rho, _ = spearmanr(ranked["topsis_score"], ranked["proba"])
        rhos.append(rho)
    res = {"pools": n_pools, "pool_size": pool_size,
           "spearman_mean": round(float(np.mean(rhos)), 3),
           "spearman_min": round(float(np.min(rhos)), 3),
           "spearman_max": round(float(np.max(rhos)), 3)}
    (ARTIFACTS_DIR / "ranking_agreement.json").write_text(json.dumps(res, indent=2))
    print("[RANKING]", res)
    return res


if __name__ == "__main__":
    run()
