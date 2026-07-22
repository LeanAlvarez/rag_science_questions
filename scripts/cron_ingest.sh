#!/usr/bin/env bash
# Incremental-ingest wrapper for a classic cron / systemd-timer setup.
#
# Not needed when using Dokploy's Schedules tab — Dokploy execs a command
# directly inside the running container, so it can call the CLI itself:
#
#     /app/.venv/bin/python -m src.ingest.run_ingest incremental
#
# This wrapper is here as a fallback for hosts where you'd run the ingest
# from OUTSIDE the container (or without a container at all):
#
#     crontab -e
#     0 */6 * * *  /path/to/repo/scripts/cron_ingest.sh >> /var/log/arxiv-rag-ingest.log 2>&1
#
set -euo pipefail

# Anchor to the repo root so relative imports and .env discovery work no
# matter where cron invokes us from.
cd "$(dirname "$0")/.."

# uv run auto-activates the project's venv AND respects uv.lock, so the deps
# are always the ones the app was built against.
exec uv run python -m src.ingest.run_ingest incremental "$@"
