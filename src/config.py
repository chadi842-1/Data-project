"""Shared configuration: criteria, paths, seeds."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts"
SEED = 42

# 8 decision criteria. direction: +1 = benefit (maximize), -1 = cost (minimize)
CRITERIA = {
    "rentabilite": +1,   # expected return (%)
    "risque": -1,        # risk score (0-100)
    "delai": -1,         # time to completion (months)
    "impact": +1,        # social / strategic impact (0-100)
    "cout": -1,          # initial investment (MMAD)
    "liquidite": +1,     # ease of exit (0-100)
    "maturite": +1,      # technology / market maturity (0-100)
    "conformite": +1,    # regulatory compliance (0-100)
}
FEATURES = list(CRITERIA.keys())
TARGET = "decision"  # 1 = invest (go), 0 = reject (no-go)

# Plausible bounds used by the schema validation step
BOUNDS = {
    "rentabilite": (-10.0, 60.0),
    "risque": (0.0, 100.0),
    "delai": (1.0, 84.0),
    "impact": (0.0, 100.0),
    "cout": (0.1, 500.0),
    "liquidite": (0.0, 100.0),
    "maturite": (0.0, 100.0),
    "conformite": (0.0, 100.0),
}
