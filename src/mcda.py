"""Multicriteria decision: AHP (weights) + TOPSIS (ranking)."""
import numpy as np
import pandas as pd

from config import CRITERIA, FEATURES

# Saaty random consistency index
_RI = {1: 0, 2: 0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}


def ahp_weights(pairwise: np.ndarray) -> tuple[np.ndarray, float]:
    """Return (weights, consistency ratio) from a pairwise comparison matrix (Saaty scale).

    Weights use the principal eigenvector. CR < 0.10 is the usual acceptability rule.
    """
    pairwise = np.asarray(pairwise, dtype=float)
    n = pairwise.shape[0]
    vals, vecs = np.linalg.eig(pairwise)
    k = np.argmax(vals.real)
    w = np.abs(vecs[:, k].real)
    w = w / w.sum()
    ci = (vals[k].real - n) / (n - 1)
    cr = ci / _RI[n] if _RI.get(n, 0) else 0.0
    return w, float(cr)


def default_pairwise() -> np.ndarray:
    """Illustrative expert judgments (rentabilite and risque matter most). Edit to your own."""
    # order = FEATURES: rentabilite, risque, delai, impact, cout, liquidite, maturite, conformite
    importance = np.array([9, 8, 4, 6, 5, 3, 2, 4], dtype=float)
    n = len(importance)
    m = np.ones((n, n))
    for i in range(n):
        for j in range(n):
            ratio = importance[i] / importance[j]
            # snap to the Saaty 1-9 scale (with reciprocals)
            s = int(np.clip(round(ratio if ratio >= 1 else 1 / ratio), 1, 9))
            m[i, j] = s if ratio >= 1 else 1 / s
    return m


def topsis(df: pd.DataFrame, weights: np.ndarray) -> pd.DataFrame:
    """Rank alternatives (rows) over the 8 criteria. Returns df + closeness score + rank."""
    X = df[FEATURES].astype(float).values
    norm = X / np.sqrt((X ** 2).sum(axis=0))
    V = norm * weights
    sign = np.array([CRITERIA[c] for c in FEATURES])
    ideal = np.where(sign > 0, V.max(axis=0), V.min(axis=0))
    anti = np.where(sign > 0, V.min(axis=0), V.max(axis=0))
    d_pos = np.sqrt(((V - ideal) ** 2).sum(axis=1))
    d_neg = np.sqrt(((V - anti) ** 2).sum(axis=1))
    out = df.copy()
    out["topsis_score"] = d_neg / (d_pos + d_neg)
    out["rank"] = out["topsis_score"].rank(ascending=False, method="min").astype(int)
    return out.sort_values("rank")


def rank_projects(df: pd.DataFrame, pairwise: np.ndarray | None = None):
    w, cr = ahp_weights(default_pairwise() if pairwise is None else pairwise)
    return topsis(df, w), dict(zip(FEATURES, map(float, w))), cr


if __name__ == "__main__":
    from config import DATA_DIR
    demo = pd.read_csv(DATA_DIR / "test.csv").dropna(subset=FEATURES).head(12)
    ranked, w, cr = rank_projects(demo)
    print("AHP weights:", {k: round(v, 3) for k, v in w.items()}, "| CR =", round(cr, 3))
    print(ranked[["project_id", "topsis_score", "rank"]].to_string(index=False))
