FROM python:3.14-slim

# uv - modern package manager (copied from the official image)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/

WORKDIR /app

# Two-phase install: dependencies first (cached layer), source second.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src/ src/
RUN uv sync --frozen --no-dev

# Configuration and raw data needed to serve the API and the pipelines.
COPY conf/ conf/
COPY data/01_raw/ data/01_raw/

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "diabetes.api:app", "--host", "0.0.0.0", "--port", "8000"]
