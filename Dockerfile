# syntax=docker/dockerfile:1.6
# Multi-stage build for SHL Assessment Recommender.
#  Stage 1 (builder): install deps, pre-bake the catalog index artifacts.
#  Stage 2 (runtime): slim runtime image with just deps + source + artifacts.

ARG PYTHON_VERSION=3.11

# ---------- Stage 1: builder -------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install uv (matches local dev environment).
RUN pip install --no-cache-dir uv==0.5.4

# Resolve and freeze deps first so layer cache is good.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev

# Copy the source + build script + raw catalog + pre-built artifacts.
# data/build/ MUST exist locally — run `uv run python scripts/build_index.py`
# (with credentials in `.env`) before building the image. The build context
# carries those artifacts straight into the runtime layer.
COPY src ./src
COPY scripts ./scripts
COPY data/shl_product_catalog.json ./data/shl_product_catalog.json
COPY data/build ./data/build

# ---------- Stage 2: runtime -------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080 \
    SHL_INDEX_DIR=/app/data/build \
    PYTHONPATH=/app/src

WORKDIR /app

# Pull the resolved venv + source + artifacts from builder.
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src
COPY --from=builder /app/data/build /app/data/build

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8080

# Cloud Run honors $PORT — uvicorn is bound to it.
CMD ["sh", "-c", "uvicorn shl_recommender.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
