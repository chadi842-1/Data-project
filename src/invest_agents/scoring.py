"""Inference: load the selected model and score / explain projects.

The agents and the API only talk to `Scorer`; they never touch sklearn or SHAP directly.
"""
from __future__ import annotations

import json
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from .config import FEATURES, Paths, get_paths
from .modeling import expected_value, make_explainer, shap_matrix


class Scorer:
    def __init__(self, pipeline, meta: dict, background: pd.DataFrame):
        self.pipeline = pipeline
        self.meta = meta
        self._pre = pipeline[:-1]  # imputer + scaler
        self._explainer = make_explainer(
            pipeline.named_steps["clf"], self._pre.transform(background[FEATURES]))

    @classmethod
    def load(cls, paths: Paths | None = None) -> Scorer:
        paths = paths or get_paths()
        names = ("model.joblib", "model_meta.json", "background.csv")
        needed = [paths.artifacts / n for n in names]
        missing = [p.name for p in needed if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing model artifacts {missing} in {paths.artifacts}. "
                "Run `python -m invest_agents.pipeline` first.")
        return cls(
            joblib.load(needed[0]),
            json.loads(needed[1].read_text()),
            pd.read_csv(needed[2]),
        )

    @property
    def model_name(self) -> str:
        return self.meta["model"]

    @property
    def threshold(self) -> float:
        return float(self.meta["threshold"])

    def info(self) -> dict:
        return dict(self.meta)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """P(invest) for each row. Missing criteria are imputed with the training medians."""
        return self.pipeline.predict_proba(df[FEATURES])[:, 1]

    def explain(self, df: pd.DataFrame, top_k: int | None = 3) -> list[list[dict]]:
        """Per project, the criteria that moved P(invest) most (SHAP, in log-odds units).

        Each driver: {feature, value, shap}. shap > 0 pushes towards 'invest'.
        """
        x = df[FEATURES].reset_index(drop=True)
        sv = shap_matrix(self._explainer, self._pre.transform(x))
        out = []
        for i in range(len(x)):
            order = np.argsort(-np.abs(sv[i]))[:top_k]
            out.append([
                {"feature": FEATURES[j], "value": float(x.iloc[i, j]), "shap": float(sv[i, j])}
                for j in order
            ])
        return out

    def base_value(self) -> float:
        return expected_value(self._explainer)


@lru_cache(maxsize=4)
def get_scorer(root: str | None = None) -> Scorer:
    """Cached scorer for a working directory (None = $INVEST_AGENTS_HOME or cwd)."""
    return Scorer.load(get_paths(root))
