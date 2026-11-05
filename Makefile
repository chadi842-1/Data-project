.PHONY: install lint typecheck test cov check pipeline api ui clean

install:          ## install the package with dev extras
	pip install -e ".[dev]"

lint:             ## static lint (ruff)
	ruff check src tests

typecheck:        ## static type check (mypy)
	mypy

test:             ## run the test suite
	pytest -q

cov:              ## run tests with a coverage report
	pytest --cov --cov-report=term-missing

check: lint typecheck test   ## everything CI runs

pipeline:         ## generate data, train, validate and write reports
	invest-pipeline

api:              ## serve the FastAPI app on :8000
	uvicorn invest_agents.api:app --port 8000 --reload

ui:               ## launch the Streamlit UI
	invest-ui

clean:            ## remove generated data/artifacts and caches
	rm -rf data artifacts .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
