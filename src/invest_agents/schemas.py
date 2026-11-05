"""Pydantic schemas for one project, in three strictness levels.

- ProjectIn:      all 8 criteria present and inside their bounds (what the agents accept).
- ProjectRecord:  criteria may be missing (None) but, when present, must be inside bounds
                  (what the ETL accepts; missing values are imputed downstream).
- ProjectLoose:   no range checks, for API request bodies; the ingestion agent reports
                  per-record problems instead of failing the whole request.
"""
from __future__ import annotations

from typing import Annotated, Optional

from pydantic import Field, create_model

from .config import BOUNDS, FEATURES, ID_COL


def _bounded(feature: str):
    lo, hi = BOUNDS[feature]
    return Annotated[float, Field(ge=lo, le=hi)]


ProjectIn = create_model(
    "ProjectIn",
    **{ID_COL: (str, ...)},
    **{f: (_bounded(f), ...) for f in FEATURES},
)

ProjectRecord = create_model(
    "ProjectRecord",
    **{ID_COL: (str, ...)},
    **{f: (Optional[_bounded(f)], None) for f in FEATURES},
)

ProjectLoose = create_model(
    "ProjectLoose",
    **{ID_COL: (str, ...)},
    **{f: (Optional[float], None) for f in FEATURES},
)
