from invest_agents.config import BOUNDS, CRITERIA, FEATURES, get_paths


def test_eight_criteria_with_valid_directions():
    assert len(FEATURES) == 8
    assert set(CRITERIA.values()) <= {-1, 1}


def test_every_criterion_has_ordered_bounds():
    assert set(BOUNDS) == set(FEATURES)
    assert all(lo < hi for lo, hi in BOUNDS.values())


def test_paths_ensure_creates_directories(tmp_path):
    p = get_paths(tmp_path).ensure()
    assert p.data.is_dir() and p.artifacts.is_dir() and p.reports.is_dir()


def test_env_var_overrides_cwd(tmp_path, monkeypatch):
    monkeypatch.setenv("INVEST_AGENTS_HOME", str(tmp_path))
    assert get_paths().root == tmp_path.resolve()
