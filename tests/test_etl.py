import json

import pandas as pd
import pytest
from pydantic import ValidationError

from invest_agents.config import FEATURES, ID_COL, TARGET, get_paths
from invest_agents.data import N_CORRUPT, N_DUPLICATES, generate_projects
from invest_agents.etl import assert_no_leakage, clean, run_etl, split, validate_schema
from invest_agents.schemas import ProjectIn, ProjectLoose, ProjectRecord

N = 400


@pytest.fixture(scope="module")
def raw():
    return generate_projects(n=N, seed=7, defects=True)


def test_clean_recovers_exactly_the_clean_projects(raw):
    df, rep = clean(raw)
    assert rep.n_raw == N + N_DUPLICATES + N_CORRUPT
    assert rep.n_duplicates_dropped == N_DUPLICATES
    assert rep.n_schema_rejected == N_CORRUPT
    assert rep.n_final == len(df) == N


def test_clean_keeps_missing_values_for_later_imputation(raw):
    df, rep = clean(raw)
    assert df[FEATURES].isna().any().any()
    assert rep.missing_rate["rentabilite"] > 0
    assert rep.missing_rate["risque"] == 0


def test_split_sizes_and_stratification(raw):
    df, _ = clean(raw)
    train, val, test = split(df, seed=1)
    assert (len(train), len(val), len(test)) == (280, 60, 60)
    for part in (train, val, test):
        assert abs(part[TARGET].mean() - df[TARGET].mean()) < 0.03


def test_split_is_leak_free_and_detects_leaks(raw):
    df, _ = clean(raw)
    train, val, test = split(df, seed=1)
    assert_no_leakage(train, val, test)
    with pytest.raises(AssertionError):
        assert_no_leakage(train, pd.concat([val, train.head(3)]), test)


def test_leak_check_also_catches_same_features_under_new_id(raw):
    df, _ = clean(raw)
    train, val, test = split(df, seed=1)
    sneaky = train.head(2).copy()
    sneaky[ID_COL] = ["X1", "X2"]
    with pytest.raises(AssertionError):
        assert_no_leakage(train, pd.concat([val, sneaky]), test)


def test_validate_schema_flags_out_of_range_rows():
    df = generate_projects(n=20, defects=False)
    df.loc[3, "risque"] = 150.0
    valid, rejected = validate_schema(df)
    assert len(rejected) == 1 and rejected.loc[0, ID_COL] == df.loc[3, ID_COL]
    assert len(valid) == 19


def test_schemas_levels():
    ok = {ID_COL: "P1", **{f: 10.0 for f in FEATURES}, "risque": 40.0, "delai": 12.0}
    ProjectIn(**ok)
    with pytest.raises(ValidationError):
        ProjectIn(**{**ok, "risque": 101.0})
    missing = {k: v for k, v in ok.items() if k != "impact"}
    with pytest.raises(ValidationError):
        ProjectIn(**missing)
    ProjectRecord(**missing)                      # missing allowed for ETL
    with pytest.raises(ValidationError):
        ProjectRecord(**{**missing, "risque": -3.0})  # but not out of range
    ProjectLoose(**{**ok, "risque": 999.0})       # loose accepts anything numeric


def test_run_etl_writes_files(tmp_path):
    paths = get_paths(tmp_path).ensure()
    generate_projects(n=300, seed=3).to_csv(paths.data / "projects_raw.csv", index=False)
    rep = run_etl(paths, seed=3)
    for name in ("train", "val", "test"):
        assert (paths.data / f"{name}.csv").exists()
    assert rep.n_final == 300
    saved = json.loads((paths.data / "etl_report.json").read_text())
    assert saved["split_sizes"] == {"train": 210, "val": 45, "test": 45}
