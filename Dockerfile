# syntax=docker/dockerfile:1.7
#
# Container image for the derm-pipeline FastAPI service.
#
# Build:    docker build -t derm-pipeline:local .
# Run:      docker run --rm -p 8000:8000 derm-pipeline:local
# Compose:  docker compose up --build
#
# The image uses `uv` (the same tool the project uses locally) to resolve
# dependencies from `uv.lock` deterministically, and bundles the project
# source under `/app/src/derm_pipeline`.

FROM python:3.11-slim

# ---------------------------------------------------------------------------
# System libraries
# ---------------------------------------------------------------------------
# - libgl1 / libglib2.0-0: required by opencv-python for image I/O.
# - ca-certificates: HTTPS to PyPI / GitHub when fetching wheels.
#
# We deliberately do NOT install libsm6/libxrender/etc. — matplotlib uses the
# headless `Agg` backend (forced in `cli._select_matplotlib_backend` and
# `api.py` via `matplotlib.use("Agg")`), so no X11 libs are required.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        ca-certificates \
        libgl1 \
        libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# uv (project's build backend and dependency manager)
# ---------------------------------------------------------------------------
# Pinned so the build is reproducible. Bump alongside the locally-installed
# `uv` version (see `uv self version`).
COPY --from=ghcr.io/astral-sh/uv:0.5.20 /uv /uvx /usr/local/bin/

# ---------------------------------------------------------------------------
# Project layout
# ---------------------------------------------------------------------------
WORKDIR /app

# Resolve dependencies first so this layer caches well when only source code
# changes. `--no-install-project` skips installing our package itself — the
# second `uv sync` below picks it up after the source is copied in.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

# Now copy the source and install the package itself.
COPY src ./src
RUN uv sync --frozen

# ---------------------------------------------------------------------------
# Runtime configuration
# ---------------------------------------------------------------------------
EXPOSE 8000

# Default location for `keep_artifacts=true` runs. The docker-compose file
# mounts a named volume here so PNGs / CSVs / PDFs survive container restarts.
ENV API_OUTPUT_ROOT=/app/api_outputs
RUN mkdir -p "$API_OUTPUT_ROOT"

# Build-time sanity check: fail fast if the import breaks.
RUN uv run python -c "from derm_pipeline.api import app"

# ---------------------------------------------------------------------------
# Health check (no curl dependency — Python only)
# ---------------------------------------------------------------------------
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD ["python", "-c", "import urllib.request, sys; \
sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=3).getcode() == 200 else 1)"]

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
# `--host 0.0.0.0` is required to accept connections from outside the container
# (loopback only by default). Workers stay at 1 — each request holds ~100 MB of
# matplotlib + numpy state, and uvicorn workers are not process-safe here
# because matplotlib's pyplot state would need per-process re-init anyway.
CMD ["uv", "run", "uvicorn", "derm_pipeline.api:app", "--host", "0.0.0.0", "--port", "8000"]
