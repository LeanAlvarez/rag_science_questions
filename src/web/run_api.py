"""Convenience CLI for launching the FastAPI dev server.

Usage:
    uv run python -m src.web.run_api                  # uses WEB_HOST / WEB_PORT from .env
    uv run python -m src.web.run_api --reload         # dev mode with autoreload
    uv run python -m src.web.run_api --port 9000
"""
from __future__ import annotations

import argparse

import uvicorn

from src.config import settings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="run_api")
    parser.add_argument("--host", default=settings.WEB_HOST)
    parser.add_argument("--port", type=int, default=settings.WEB_PORT)
    parser.add_argument("--reload", action="store_true", help="Autoreload on code changes")
    args = parser.parse_args(argv)

    uvicorn.run(
        "src.web.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
