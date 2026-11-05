"""ETL: cleaning, schema validation, leak-free 70/15/15 split.

Leakage control: duplicates are dropped BEFORE splitting, and the imputer and
scaler are fit on the training split only, then applied to validation and test.
"""
import joblib
import pandas as pd
from pydantic import BaseModel, Field, ValidationError
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from config import ARTIFACTS_DIR, BOUNDS, DATA_DIR, FEATURES, SEED, TARGET


class ProjectRecord(BaseModel):
    """Schema for one project row (bounds come from config.BOUNDS)."""
    project_id: str
    rentabilite: float = Field(ge=BOUNDS["rentabilite"][0], le=BOUNDS["rentabilite"][1])
    risque: float = Field(ge=BOUNDS["risque"][0], le=BOUNDS["risque"][1])
    delai: float = Field(ge=BOUNDS["delai"][0], le=BOUNDS["delai"][1])
    impact: float = Field(ge=BOUNDS["impact"][0], le=BOUNDS["impact"][1])
    cout: float = Field(ge=BOUNDS["cout"][0], le=BOUNDS["cout"][1])
    liquidite: float = Field(ge=BOUNDS["liquidite"][0], le=BOUNDS["liquidite"][1])
    maturite: float = Field(ge=BOUNDS["maturite"][0], le=BOUNDS["maturite"][1])
    conformite: float = Field(ge=BOUNDS["conformite"][0], le=BOUNDS["conformite"][1])


def validate_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows that violate the schema (missing values are allowed here and imputed later)."""
    good, bad = [], 0
    for row in df.to_dict("records"):
        probe = {k: (None if pd.isna(v) else v) for k, v in row.items()}
        # fill NaN features with a bound-midpoint only for the check itself
        for f in FEATURES:
            if probe[f] is None:
                lo, hi = BOUNDS[f]
                probe[f] = (lo + hi) / 2
        try:
            ProjectRecord(**{k: probe[k] for k in ["project_id", *FEATURES]})
            good.append(row)
        except ValidationError:
            bad += 1
    print(f"[ETL] schema validation: kept {len(good)}, rejected {bad}")
    return pd.DataFrame(good)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    n0 = len(df)
    df = df.drop_duplicates(subset=FEATURES + [TARGET]).reset_index(drop=True)
    print(f"[ETL] dropped {n0 - len(df)} duplicate rows")
    return validate_schema(df)


def split(df: pd.DataFrame):
    """Stratified 70/15/15 split."""
    train, tmp = train_test_split(
        df, test_size=0.30, stratify=df[TARGET], random_state=SEED)
    val, test = train_test_split(
        tmp, test_size=0.50, stratify=tmp[TARGET], random_state=SEED)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def build_preprocessor() -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])


def run() -> dict:
    DATA_DIR.mkdir(exist_ok=True)
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    raw = pd.read_csv(DATA_DIR / "projects_raw.csv")
    df = clean(raw)
    train, val, test = split(df)

    pre = build_preprocessor().fit(train[FEATURES])  # fit on TRAIN only
    joblib.dump(pre, ARTIFACTS_DIR / "preprocessor.joblib")

    for name, part in {"train": train, "val": val, "test": test}.items():
        part.to_csv(DATA_DIR / f"{name}.csv", index=False)

    # Leakage check: no project appears in two splits
    ids = [set(p["project_id"]) for p in (train, val, test)]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2]), "split leakage!"
    sizes = {k: len(v) for k, v in zip(["train", "val", "test"], (train, val, test))}
    print(f"[ETL] split sizes: {sizes}")
    return sizes


if __name__ == "__main__":
    run()
