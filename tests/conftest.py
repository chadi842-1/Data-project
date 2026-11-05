"""Shared fixtures. `trained_home` trains one tiny model once per test session."""
import pytest

from invest_agents.config import get_paths
from invest_agents.data import generate_projects
from invest_agents.etl import run_etl
from invest_agents.modeling import run_training


@pytest.fixture(scope="session")
def trained_home(tmp_path_factory):
    root = tmp_path_factory.mktemp("invest_home")
    paths = get_paths(root).ensure()
    generate_projects(n=600, seed=5).to_csv(paths.data / "projects_raw.csv", index=False)
    run_etl(paths, seed=5)
    run_training(paths, n_iter=3, cv_folds=3, n_boot=200, n_jobs=1, seed=5)
    return paths


@pytest.fixture
def home_env(trained_home, monkeypatch):
    """Point the package at the trained working directory and reset the scorer cache."""
    from invest_agents.scoring import get_scorer

    monkeypatch.setenv("INVEST_AGENTS_HOME", str(trained_home.root))
    get_scorer.cache_clear()
    yield trained_home
    get_scorer.cache_clear()
