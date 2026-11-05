"""Multicriteria decision analysis: AHP weights, TOPSIS ranking and weight-sensitivity analysis.

Two caveats worth stating plainly:
- TOPSIS scores are RELATIVE to the pool being ranked (vector normalisation and ideal points
  are computed on the pool), so the same project can score differently in a different pool.
- The default AHP priorities below are illustrative placeholders, not elicited from experts.
  Pass your own pairwise matrix (or priorities) to `ahp` / the agents.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import CRITERIA, FEATURES, ID_COL

CR_THRESHOLD = 0.10
# Saaty's random consistency index, by matrix size.
_RI = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}

# Illustrative importance on a 1-9 scale (NOT elicited from experts).
DEFAULT_PRIORITIES: dict[str, float] = {
    "rentabilite": 9, "risque": 8, "impact": 6, "cout": 5,
    "delai": 4, "conformite": 4, "liquidite": 3, "maturite": 2,
}


# -------------------------------------------------------------------------------- AHP
@dataclass(frozen=True)
class AHPResult:
    weights: dict[str, float]
    lambda_max: float
    consistency_ratio: float

    @property
    def is_consistent(self) -> bool:
        return self.consistency_ratio < CR_THRESHOLD


def check_pairwise(matrix: np.ndarray, atol: float = 1e-6) -> np.ndarray:
    """Validate a Saaty pairwise matrix: square, positive, unit diagonal, reciprocal."""
    m = np.asarray(matrix, dtype=float)
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        raise ValueError("pairwise matrix must be square")
    if (m <= 0).any():
        raise ValueError("pairwise matrix must be strictly positive")
    if not np.allclose(np.diag(m), 1.0, atol=atol):
        raise ValueError("pairwise matrix must have ones on the diagonal")
    if not np.allclose(m * m.T, 1.0, atol=1e-4):
        raise ValueError("pairwise matrix must be reciprocal (a_ij * a_ji = 1)")
    if m.shape[0] > max(_RI):
        raise ValueError(f"at most {max(_RI)} criteria are supported")
    return m


def ahp(pairwise: np.ndarray, names: list[str] | None = None) -> AHPResult:
    """AHP weights (principal eigenvector) and consistency ratio. CR < 0.10 is acceptable."""
    m = check_pairwise(pairwise)
    names = names or FEATURES
    if len(names) != m.shape[0]:
        raise ValueError(f"matrix is {m.shape[0]}x{m.shape[0]} but {len(names)} names were given")
    n = m.shape[0]
    vals, vecs = np.linalg.eig(m)
    k = int(np.argmax(vals.real))
    w = np.abs(vecs[:, k].real)
    w = w / w.sum()
    lam = float(vals[k].real)
    ci = (lam - n) / (n - 1) if n > 1 else 0.0
    cr = ci / _RI[n] if _RI[n] else 0.0
    return AHPResult(dict(zip(names, map(float, w), strict=True)), lam, float(max(cr, 0.0)))


def priorities_to_pairwise(priorities: dict[str, float], names: list[str] | None = None):
    """Encode 1-9 importance scores as a reciprocal pairwise matrix on the Saaty scale.

    a_ij = priority_i / priority_j snapped to the nearest of 1..9 (or its reciprocal), so the
    consistency ratio reflects the loss from that rounding.
    """
    names = names or FEATURES
    p = np.array([priorities[n] for n in names], dtype=float)
    m = np.ones((len(p), len(p)))
    for i in range(len(p)):
        for j in range(len(p)):
            ratio = p[i] / p[j]
            level = int(np.clip(round(ratio if ratio >= 1 else 1 / ratio), 1, 9))
            m[i, j] = level if ratio >= 1 else 1 / level
    return m


def default_ahp() -> AHPResult:
    return ahp(priorities_to_pairwise(DEFAULT_PRIORITIES))


# ---------------------------------------------------------------------------------- TOPSIS
def _directions(names: list[str]) -> np.ndarray:
    return np.array([CRITERIA[n] for n in names])


def topsis_scores(X: np.ndarray, weights: np.ndarray, directions: np.ndarray) -> np.ndarray:
    """Closeness to the ideal solution in [0, 1] for each row of X (alternatives x criteria)."""
    X = np.asarray(X, dtype=float)
    norm = np.sqrt((X**2).sum(axis=0))
    norm[norm == 0] = 1.0
    v = (X / norm) * weights
    ideal = np.where(directions > 0, v.max(axis=0), v.min(axis=0))
    anti = np.where(directions > 0, v.min(axis=0), v.max(axis=0))
    d_pos = np.sqrt(((v - ideal) ** 2).sum(axis=1))
    d_neg = np.sqrt(((v - anti) ** 2).sum(axis=1))
    denom = d_pos + d_neg
    return np.divide(d_neg, denom, out=np.full_like(denom, 0.5), where=denom > 0)


def topsis(df: pd.DataFrame, weights: dict[str, float] | np.ndarray) -> pd.DataFrame:
    """Rank the rows of `df` (needs the 8 criteria). Adds topsis_score and rank (1 = best)."""
    w = _as_vector(weights)
    out = df.copy()
    out["topsis_score"] = topsis_scores(df[FEATURES].to_numpy(float), w, _directions(FEATURES))
    out["rank"] = out["topsis_score"].rank(ascending=False, method="min").astype(int)
    return out.sort_values(["rank"], kind="stable")


def _as_vector(weights: dict[str, float] | np.ndarray) -> np.ndarray:
    if isinstance(weights, dict):
        return np.array([weights[f] for f in FEATURES], dtype=float)
    return np.asarray(weights, dtype=float)


def rank_projects(df: pd.DataFrame, pairwise: np.ndarray | None = None):
    """AHP + TOPSIS in one call. Returns (ranked dataframe, AHPResult)."""
    res = default_ahp() if pairwise is None else ahp(pairwise)
    return topsis(df, res.weights), res


# --------------------------------------------------------------------- weight sensitivity
@dataclass
class SensitivityResult:
    per_project: pd.DataFrame  # base_rank, rank_mean, rank_std, p_top1, p_top3
    spearman_mean: float       # mean rank correlation between perturbed and base rankings


def weight_sensitivity(
    df: pd.DataFrame,
    weights: dict[str, float] | np.ndarray,
    n_sims: int = 1000,
    concentration: float = 50.0,
    seed: int = 0,
) -> SensitivityResult:
    """How stable is the TOPSIS ranking when the weights are uncertain?

    Draws weight vectors from a Dirichlet centred on `weights` (higher `concentration` = less
    uncertainty), re-ranks the pool each time and summarises each project's rank distribution.
    """
    w0 = _as_vector(weights)
    X = df[FEATURES].to_numpy(float)
    sign = _directions(FEATURES)
    norm = np.sqrt((X**2).sum(axis=0))
    norm[norm == 0] = 1.0
    N = X / norm
    rng = np.random.default_rng(seed)
    W = rng.dirichlet(w0 / w0.sum() * concentration, size=n_sims)  # (S, k)

    n_hi, n_lo = N.max(axis=0), N.min(axis=0)
    ideal = np.where(sign > 0, n_hi, n_lo)[None, :] * W
    anti = np.where(sign > 0, n_lo, n_hi)[None, :] * W
    V = N[None, :, :] * W[:, None, :]                          # (S, n, k)
    d_pos = np.sqrt(((V - ideal[:, None, :]) ** 2).sum(axis=2))
    d_neg = np.sqrt(((V - anti[:, None, :]) ** 2).sum(axis=2))
    denom = d_pos + d_neg
    scores = np.divide(d_neg, denom, out=np.full_like(denom, 0.5), where=denom > 0)
    ranks = (-scores).argsort(axis=1).argsort(axis=1) + 1      # (S, n), 1 = best

    base = topsis_scores(X, w0 / w0.sum(), sign)
    base_rank = (-base).argsort().argsort() + 1
    a = ranks - ranks.mean(axis=1, keepdims=True)
    b = base_rank - base_rank.mean()
    denom_rho = np.sqrt((a**2).sum(axis=1) * (b**2).sum())
    rho = np.divide((a * b).sum(axis=1), denom_rho, out=np.ones(n_sims), where=denom_rho > 0)

    index = df[ID_COL] if ID_COL in df.columns else df.index
    per = pd.DataFrame({
        "base_rank": base_rank,
        "rank_mean": ranks.mean(axis=0),
        "rank_std": ranks.std(axis=0),
        "p_top1": (ranks == 1).mean(axis=0),
        "p_top3": (ranks <= 3).mean(axis=0),
    }, index=index).sort_values("base_rank")
    return SensitivityResult(per, float(rho.mean()))
