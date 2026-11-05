"""Generate a SYNTHETIC dataset of 2,000 investment projects.

The data is fully synthetic: no real project data is used. The go/no-go label is
produced by a nonlinear latent score plus noise, so tree models can legitimately
outperform a linear baseline, but the gap is whatever the simulation yields.
A few missing values and duplicates are injected so the ETL step has real work to do.
"""
import numpy as np
import pandas as pd

from config import BOUNDS, DATA_DIR, SEED, TARGET


def generate(n: int = 2000, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "rentabilite": rng.normal(14, 8, n),
        "risque": rng.beta(2.2, 2.8, n) * 100,
        "delai": rng.gamma(4, 5, n) + 3,
        "impact": rng.beta(2.5, 2.5, n) * 100,
        "cout": rng.lognormal(2.8, 0.9, n),
        "liquidite": rng.beta(2, 2, n) * 100,
        "maturite": rng.beta(3, 2, n) * 100,
        "conformite": rng.beta(4, 1.8, n) * 100,
    })
    # correlated structure: riskier projects tend to pay more
    df["rentabilite"] += 0.08 * (df["risque"] - 45)

    # Nonlinear latent attractiveness (interactions + thresholds)
    z = (
        0.11 * df["rentabilite"]
        - 0.035 * df["risque"]
        - 0.03 * df["delai"]
        + 0.022 * df["impact"]
        - 0.25 * np.log(df["cout"])
        + 0.012 * df["liquidite"]
        + 0.01 * df["maturite"]
        + 0.015 * df["conformite"]
        # interactions / thresholds
        + 0.0016 * (df["rentabilite"] * df["maturite"] / 10)
        - 1.4 * ((df["risque"] > 70) & (df["liquidite"] < 35)).astype(float)
        - 1.2 * (df["conformite"] < 40).astype(float)
        + 0.9 * ((df["impact"] > 70) & (df["rentabilite"] > 12)).astype(float)
    )
    z = z + rng.normal(0, 0.9, n)
    df[TARGET] = (z > np.quantile(z, 0.55)).astype(int)

    for c, (lo, hi) in BOUNDS.items():
        df[c] = df[c].clip(lo, hi)

    # inject realistic mess
    for c in ["rentabilite", "delai", "liquidite", "maturite"]:
        idx = rng.choice(n, size=int(0.02 * n), replace=False)
        df.loc[idx, c] = np.nan
    dup = df.sample(25, random_state=seed)
    df = pd.concat([df, dup], ignore_index=True)
    df.insert(0, "project_id", [f"P{i:05d}" for i in range(len(df))])
    return df


if __name__ == "__main__":
    DATA_DIR.mkdir(exist_ok=True)
    out = generate()
    out.to_csv(DATA_DIR / "projects_raw.csv", index=False)
    print(f"Wrote {len(out)} rows to data/projects_raw.csv")
