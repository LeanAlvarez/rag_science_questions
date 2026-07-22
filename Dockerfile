# syntax=docker/dockerfile:1.7
# =============================================================================
# arxiv-rag — production image for VPS deploy (Dokploy).
#
# Multi-stage build:
#   1. frontend-builder — npm ci + vite build → /app/static
#   2. python-builder   — uv sync (torch CPU on Linux via [tool.uv.sources])
#   3. model-baker      — pre-download embedding + rerank weights
#   4. runtime          — slim python + venv + src + static + HF cache
#
# Final image ~1.3 GB. Only stage 4 ships; toolchain and node_modules stay
# behind in the intermediate stages.
# =============================================================================


# ─── Stage 1: build the React SPA ───────────────────────────────────────────
FROM node:22-alpine AS frontend-builder

WORKDIR /app/frontend

# Copy manifests first so `npm ci` is cached when only source changes.
COPY src/web/frontend/package.json src/web/frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund

# Copy the rest of the frontend and build. Vite's outDir="../static" writes
# the built assets to /app/static in this stage.
COPY src/web/frontend/ ./
RUN npm run build


# ─── Stage 2: Python deps with uv ──────────────────────────────────────────
# The [tool.uv.sources] override in pyproject.toml routes torch to the CPU
# wheel index on Linux automatically — no extra flags needed.
FROM python:3.12-slim-bookworm AS python-builder

# Pull the uv static binary from the official image (small, no pip needed).
COPY --from=ghcr.io/astral-sh/uv:0.8.22 /uv /uvx /usr/local/bin/

WORKDIR /app

ENV \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONDONTWRITEBYTECODE=1

# Only the resolver-relevant files, so the (long) dep install is cached
# independently from source-code changes.
COPY pyproject.toml uv.lock ./

#   --frozen              refuse to update the lockfile (deterministic)
#   --no-dev              skip the [dependency-groups] dev = [...] group
#   --no-install-project  do not attempt to install our app as a package
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project


# ─── Stage 3: pre-bake model weights ───────────────────────────────────────
# Adds ~220 MB to the image but eliminates the multi-second cold download
# on the first user query after each deploy. Model IDs are ARG-passed so a
# future model swap only means changing the docker build args (or CI vars),
# not editing this file.
FROM python-builder AS model-baker

ARG EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
ARG RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2

ENV HF_HOME=/opt/hf-cache \
    HF_HUB_DISABLE_TELEMETRY=1

RUN .venv/bin/python -c "\
from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('${EMBEDDING_MODEL}'); \
CrossEncoder('${RERANKER_MODEL}'); \
print('Cached ${EMBEDDING_MODEL} + ${RERANKER_MODEL} at /opt/hf-cache')"


# ─── Stage 4: runtime ──────────────────────────────────────────────────────
# No apt installs: psycopg[binary] bundles libpq, and httpx uses stdlib SSL
# from the manylinux python image. Nothing else is needed.
FROM python:3.12-slim-bookworm AS runtime

# Non-root user — good hygiene for a public HTTP service.
RUN groupadd --system app && \
    useradd  --system --gid app --home-dir /app --shell /usr/sbin/nologin app

WORKDIR /app

# The resolved virtualenv from stage 2.
COPY --from=python-builder --chown=app:app /app/.venv /app/.venv

# The pre-baked HuggingFace cache from stage 3.
COPY --from=model-baker    --chown=app:app /opt/hf-cache /opt/hf-cache

# The built SPA from stage 1 → the exact path FastAPI's StaticFiles mount
# looks for (src/web/api.py: `_STATIC_DIR = Path(__file__).parent / "static"`).
COPY --from=frontend-builder --chown=app:app /app/static /app/src/web/static

# Application source — explicit per-subdir COPY so the runtime image only
# ships what the running app actually imports. src/web/frontend/ is NOT
# copied (its built output landed at src/web/static in the previous COPY).
# src/bot/ IS copied — the FastAPI lifespan boots it alongside the web when
# TELEGRAM_BOT_TOKEN is set (shares the interpreter to save VPS RAM).
COPY --chown=app:app src/__init__.py src/config.py src/db.py /app/src/
COPY --chown=app:app src/core/                                /app/src/core/
COPY --chown=app:app src/ingest/                              /app/src/ingest/
COPY --chown=app:app src/query/                               /app/src/query/
COPY --chown=app:app src/bot/                                 /app/src/bot/
COPY --chown=app:app src/web/__init__.py src/web/api.py src/web/run_api.py /app/src/web/

# Runtime environment:
#   HF_HUB_OFFLINE=1  refuse to download models at runtime — if a model is
#                     missing from /opt/hf-cache the app fails fast instead
#                     of hanging a request while it downloads.
#   WEB_HOST=0.0.0.0  container must bind on 0.0.0.0 (not 127.0.0.1) so
#                     Dokploy's reverse proxy can reach it.
ENV \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/opt/hf-cache \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HUB_OFFLINE=1 \
    WEB_HOST=0.0.0.0 \
    WEB_PORT=8000 \
    MODEL_DEVICE=cpu

USER app

EXPOSE 8000

# Docker/Dokploy uses this to decide "healthy". start-period gives the
# process time to import torch + load models before probes count against it.
HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=3).status == 200 else 1)"

# Uvicorn directly — one worker is fine for a small VPS + a global embedding
# model (loading it once per worker would 3x the RAM). Scale horizontally
# via Dokploy replicas if needed instead of upping --workers.
CMD ["uvicorn", "src.web.api:app", "--host", "0.0.0.0", "--port", "8000"]
