FROM python:3.12-slim

# libgomp is needed by LightGBM / XGBoost
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

# Train at build time so the image is self-contained (synthetic data, well under a minute).
ENV INVEST_AGENTS_HOME=/app
RUN invest-pipeline --n-iter 10 --n-boot 500

EXPOSE 8000 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8000/health')"

# API (default):   docker run -p 8000:8000 invest-agents
# Streamlit UI:    docker run -p 8501:8501 invest-agents invest-ui --server.address 0.0.0.0
CMD ["uvicorn", "invest_agents.api:app", "--host", "0.0.0.0", "--port", "8000"]
