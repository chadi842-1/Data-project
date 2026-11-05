"""ETL: deduplicate, validate against the schema, and make a leak-free stratified 70/15/15 split.

Leakage controls:
- duplicates are dropped BEFORE splitting, so a copy of a row cannot land in two splits;
- imputation and scaling are NOT fitted here: they live inside the model pipelines
  (see modeling.py), so they are re-fitted on the training part of every CV fold;
- `assert_no_leakage` checks that no project id and no feature vector appears in two splits.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

import pandas as pd
from pydantic import ValidationError
from sklearn.model_selection import train_test_split

from .config import FEATURES, ID_COL, SEED, TARGET, Paths, get_paths
from .schemas import ProjectRecord


@dataclass
class EtlReport:
    n_raw: int = 0
    n_duplicates_dropped: int = 0
    n_schema_rejected: int = 0
    n_final: int = 0
    missing_rate: dict[str, float] = field(default_factory=dict)
    split_sizes: dict[str, int] = field(default_factory=dict)
    invest_rate: dict[str, float] = field(default_factory=dict)


def validate_schema(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split rows into (valid, rejected) using the ProjectRecord schema."""
    ok_idx, bad_idx = [], []
    records = df[[ID_COL, *FEATURES]].to_dict("records")
    for idx, rec in zip(df.index, records, strict=True):
        rec = {k: (None if pd.isna(v) else v) for k, v in rec.items()}
        try:
            ProjectRecord(**rec)
            ok_idx.append(idx)
        except ValidationError:
            bad_idx.append(idx)
    return df.loc[ok_idx].reset_index(drop=True), df.loc[bad_idx].reset_index(drop=True)


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, EtlReport]:
    """Drop duplicates (on features + label), then drop schema violations."""
    report = EtlReport(n_raw=len(df))
    deduped = df.drop_duplicates(subset=FEATURES + [TARGET]).reset_index(drop=True)
    report.n_duplicates_dropped = len(df) - len(deduped)
    valid, rejected = validate_schema(deduped)
    report.n_schema_rejected = len(rejected)
    report.n_final = len(valid)
    report.missing_rate = {f: round(float(valid[f].isna().mean()), 4) for f in FEATURES}
    return valid, report


def split(
    df: pd.DataFrame, seed: int = SEED
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified 70/15/15 split on the label."""
    train, rest = train_test_split(df, test_size=0.30, stratify=df[TARGET], random_state=seed)
    val, test = train_test_split(rest, test_size=0.50, stratify=rest[TARGET], random_state=seed)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def assert_no_leakage(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> None:
    parts = {"train": train, "val": val, "test": test}
    names = list(parts)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            shared_ids = set(parts[a][ID_COL]) & set(parts[b][ID_COL])
            if shared_ids:
                raise AssertionError(f"{len(shared_ids)} project ids shared by {a} and {b}")
            merged = pd.concat(
                [parts[a][FEATURES].drop_duplicates(), parts[b][FEATURES].drop_duplicates()]
            )
            if merged.duplicated().any():
                raise AssertionError(f"identical feature vectors shared by {a} and {b}")


def run_etl(paths: Paths | None = None, seed: int = SEED) -> EtlReport:
    """Read data/projects_raw.csv, clean it, write data/{train,val,test}.csv and etl_report.json."""
    paths = (paths or get_paths()).ensure()
    raw = pd.read_csv(paths.data / "projects_raw.csv")
    df, report = clean(raw)
    train, val, test = split(df, seed)
    assert_no_leakage(train, val, test)
    for name, part in {"train": train, "val": val, "test": test}.items():
        part.to_csv(paths.data / f"{name}.csv", index=False)
        report.split_sizes[name] = len(part)
        report.invest_rate[name] = round(float(part[TARGET].mean()), 4)
    (paths.data / "etl_report.json").write_text(json.dumps(asdict(report), indent=2))
    return report


if __name__ == "__main__":
    print(json.dumps(asdict(run_etl()), indent=2))
