from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import invest_agents.ui as ui_module

UI = str(Path(ui_module.__file__))


def test_ui_loads_runs_and_shows_a_ranking(home_env):
    at = AppTest.from_file(UI, default_timeout=120).run()
    assert not at.exception
    assert at.title[0].value.startswith("Agentic")
    at.button[0].click().run()
    assert not at.exception
    assert [m.label for m in at.metric][:2] == ["Valid projects", "AHP consistency ratio"]
    assert at.selectbox[0].options                      # a project can be inspected
    assert len(at.dataframe) >= 3                        # input, recommendations, drivers


def test_ui_survives_interaction_after_ranking(home_env):
    at = AppTest.from_file(UI, default_timeout=120).run()
    at.button[0].click().run()
    options = at.selectbox[0].options
    at.selectbox[0].select(options[-1]).run()           # rerun must keep the result
    assert not at.exception and at.metric


def test_ui_blocks_an_impossible_policy(home_env):
    at = AppTest.from_file(UI, default_timeout=120).run()
    at.sidebar.slider[1].set_value(0.9).run()            # reject threshold above invest threshold
    assert at.sidebar.error
    assert at.button[0].disabled


def test_ui_explains_missing_model(tmp_path, monkeypatch):
    from invest_agents.scoring import get_scorer

    monkeypatch.setenv("INVEST_AGENTS_HOME", str(tmp_path))
    get_scorer.cache_clear()
    at = AppTest.from_file(UI, default_timeout=60).run()
    assert at.error and "pipeline" in at.error[0].value
    get_scorer.cache_clear()


@pytest.fixture(autouse=True)
def _clear_streamlit_cache():
    import streamlit as st

    st.cache_data.clear()
    yield
