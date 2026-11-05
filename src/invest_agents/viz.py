"""Shared matplotlib style. Colors are the first slots of a validated categorical palette
(blue, orange, aqua), recessive grid, text in ink colors (never in the series color)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SERIES = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a"}
MODEL_COLORS = {
    "LogisticRegression": SERIES["blue"],
    "LightGBM": SERIES["orange"],
    "XGBoost": SERIES["aqua"],
}


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_2,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "text.color": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "font.size": 10,
    })


def title(ax, main: str, sub: str | None = None) -> None:
    """Left-aligned title with an optional muted subtitle."""
    ax.set_title(main, loc="left", fontsize=12, fontweight="bold", color=INK, pad=22 if sub else 10)
    if sub:
        ax.text(0, 1.03, sub, transform=ax.transAxes, fontsize=9, color=INK_2, va="bottom")
