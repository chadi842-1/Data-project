import tomllib
from importlib import import_module
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_console_scripts_point_at_real_callables():
    scripts = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["scripts"]
    assert set(scripts) == {"invest-pipeline", "invest-experiments", "invest-ui"}
    for target in scripts.values():
        module, func = target.split(":")
        assert callable(getattr(import_module(module), func)), target


def test_dockerfile_uses_only_installed_entry_points():
    text = (ROOT / "Dockerfile").read_text()
    scripts = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["scripts"]
    assert "invest-pipeline" in text and "invest-pipeline" in scripts
    assert "invest_agents.api:app" in text
    assert "invest-ui" in text


def test_ci_runs_lint_and_tests():
    text = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    assert "ruff check" in text and "pytest" in text
