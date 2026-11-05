"""Synthetic investment-project generator with a documented data-generating process (DGP).

Everything here is SIMULATED. No real project data is used.

DGP, fixed on modelling grounds before any model was trained:

1. Criteria are drawn from a Gaussian copula so they are correlated like real projects
   (riskier projects pay more, bigger projects take longer, mature projects are less risky...),
   then mapped to bounded marginals (beta / gamma / log-normal / normal).
2. An investment committee's latent attractiveness is the sum of
     - a smooth additive utility (return, risk, delay, impact, cost, liquidity, ...), and
     - committee-style NON-LINEAR terms: diminishing returns on profitability, a return x maturity
       synergy, a bonus for high-impact profitable projects, and two knockout rules
       (low compliance; high risk with low liquidity).
   The non-linear terms are scaled by `nonlinearity` (default 1.0 = the full rule set,
   0.0 = purely additive). Normal committees DO use knockout rules, which is why the default is 1.0;
   `invest_agents.experiments` studies how model rankings change as this knob moves.
3. Gaussian noise is added to the latent value, and the top `invest_rate` share becomes "invest".
4. Real-world mess is injected on request: ~2% missing values, duplicates and a few corrupt rows,
   so the ETL step has genuine work to do.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .config import BOUNDS, FEATURES, ID_COL, SEED, TARGET

N_DUPLICATES = 25
N_CORRUPT = 5
MISSING_RATE = 0.02
MISSING_COLUMNS = ("rentabilite", "delai", "liquidite", "maturite")

# Pairwise correlations of the latent Gaussian (everything not listed is independent).
_CORR_PAIRS: dict[tuple[str, str], float] = {
    ("rentabilite", "risque"): 0.40,
    ("risque", "maturite"): -0.35,
    ("risque", "liquidite"): -0.30,
    ("delai", "cout"): 0.50,
    ("impact", "cout"): 0.20,
    ("maturite", "conformite"): 0.30,
    ("rentabilite", "impact"): 0.15,
}

_MARGINALS = {
    "rentabilite": stats.norm(loc=14, scale=8),
    "risque": stats.beta(2.2, 2.8, scale=100),
    "delai": stats.gamma(4, loc=3, scale=5),
    "impact": stats.beta(2.5, 2.5, scale=100),
    "cout": stats.lognorm(s=0.9, scale=float(np.exp(2.8))),
    "liquidite": stats.beta(2, 2, scale=100),
    "maturite": stats.beta(3, 2, scale=100),
    "conformite": stats.beta(4, 1.8, scale=100),
}


def correlation_matrix() -> np.ndarray:
    """Latent correlation matrix in FEATURES order (checked to be positive definite)."""
    idx = {f: i for i, f in enumerate(FEATURES)}
    corr = np.eye(len(FEATURES))
    for (a, b), value in _CORR_PAIRS.items():
        corr[idx[a], idx[b]] = corr[idx[b], idx[a]] = value
    if np.linalg.eigvalsh(corr).min() <= 0:
        raise ValueError("correlation matrix is not positive definite")
    return corr


def sample_criteria(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Draw n correlated, bounded criteria vectors."""
    z = rng.multivariate_normal(np.zeros(len(FEATURES)), correlation_matrix(), size=n)
    u = np.clip(stats.norm.cdf(z), 1e-6, 1 - 1e-6)
    cols = {}
    for i, f in enumerate(FEATURES):
        lo, hi = BOUNDS[f]
        cols[f] = np.clip(_MARGINALS[f].ppf(u[:, i]), lo, hi)
    return pd.DataFrame(cols)


def latent_utility(df: pd.DataFrame, nonlinearity: float = 1.0) -> np.ndarray:
    """Noise-free committee attractiveness (a ground-truth ranking, on synthetic data only)."""
    r, k, d, i, c, l_, m, f = (df[name].to_numpy(dtype=float) for name in FEATURES)
    linear = (
        0.11 * r - 0.035 * k - 0.03 * d + 0.022 * i - 0.25 * np.log(c)
        + 0.012 * l_ + 0.010 * m + 0.015 * f
    )
    nonlinear = (
        0.11 * (20 * np.tanh(r / 20) - r)      # diminishing returns on profitability
        + 0.00016 * r * m                       # return x maturity synergy
        - 1.4 * ((k > 70) & (l_ < 35))          # knockout: very risky AND illiquid
        - 1.2 * (f < 40)                        # knockout: poor regulatory compliance
        + 0.9 * ((i > 70) & (r > 12))           # bonus: high impact AND profitable
    )
    return linear + nonlinearity * nonlinear


def _inject_defects(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    df = df.copy()
    n = len(df)
    for col in MISSING_COLUMNS:  # MCAR missing values
        idx = rng.choice(n, size=int(MISSING_RATE * n), replace=False)
        df.loc[idx, col] = np.nan
    dups = df.sample(N_DUPLICATES, random_state=int(rng.integers(1 << 31)))
    corrupt = df.sample(N_CORRUPT, random_state=int(rng.integers(1 << 31))).copy()
    for row_idx in corrupt.index:  # impossible values: schema validation must reject these
        feat = FEATURES[int(rng.integers(len(FEATURES)))]
        lo, hi = BOUNDS[feat]
        corrupt.loc[row_idx, feat] = hi * 2 + 1 if rng.random() < 0.5 else lo - abs(lo) - 1
    return pd.concat([df, dups, corrupt], ignore_index=True)


def generate_projects(
    n: int = 2000,
    seed: int = SEED,
    nonlinearity: float = 1.0,
    noise_sd: float = 0.9,
    invest_rate: float = 0.45,
    defects: bool = True,
) -> pd.DataFrame:
    """Generate `n` projects. With defects=True: n + 30 rows (25 duplicates, 5 corrupt)."""
    rng = np.random.default_rng(seed)
    df = sample_criteria(n, rng)
    z = latent_utility(df, nonlinearity) + rng.normal(0, noise_sd, n)
    df[TARGET] = (z > np.quantile(z, 1 - invest_rate)).astype(int)
    if defects:
        df = _inject_defects(df, rng)
    df.insert(0, ID_COL, [f"P{i:05d}" for i in range(len(df))])
    return df


if __name__ == "__main__":
    from .config import get_paths

    paths = get_paths().ensure()
    out = generate_projects()
    out.to_csv(paths.data / "projects_raw.csv", index=False)
    print(f"Wrote {len(out)} rows to {paths.data / 'projects_raw.csv'}")
