"""Central configuration: criteria, bounds, seeds and working-directory layout."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

SEED = 42
ID_COL = "project_id"
TARGET = "decision"  # 1 = invest (go), 0 = reject (no-go)

# The 8 decision criteria. Direction: +1 = benefit (higher is better), -1 = cost (lower is better).
CRITERIA: dict[str, int] = {
    "rentabilite": +1,  # expected return (%)
    "risque": -1,       # risk score (0-100)
    "delai": -1,        # time to completion (months)
    "impact": +1,       # social / strategic impact (0-100)
    "cout": -1,         # initial investment (MMAD)
    "liquidite": +1,    # ease of exit (0-100)
    "maturite": +1,     # technology / market maturity (0-100)
    "conformite": +1,   # regulatory compliance (0-100)
}
FEATURES: list[str] = list(CRITERIA)

# Plausible bounds, used for schema validation and to clip generated data.
BOUNDS: dict[str, tuple[float, float]] = {
    "rentabilite": (-10.0, 60.0),
    "risque": (0.0, 100.0),
    "delai": (1.0, 84.0),
    "impact": (0.0, 100.0),
    "cout": (0.1, 500.0),
    "liquidite": (0.0, 100.0),
    "maturite": (0.0, 100.0),
    "conformite": (0.0, 100.0),
}


@dataclass(frozen=True)
class Paths:
    """Working-directory layout. Everything is relative to `root`."""

    root: Path

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def artifacts(self) -> Path:
        return self.root / "artifacts"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    def ensure(self) -> Paths:
        for d in (self.data, self.artifacts, self.reports):
            d.mkdir(parents=True, exist_ok=True)
        return self


def get_paths(root: str | os.PathLike | None = None) -> Paths:
    """Resolve the working directory: explicit arg > $INVEST_AGENTS_HOME > current directory."""
    chosen = root or os.environ.get("INVEST_AGENTS_HOME") or Path.cwd()
    return Paths(Path(chosen).resolve())
