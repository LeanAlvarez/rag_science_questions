"""CLI entrypoint for ingestion.

Usage:
    uv run python -m src.ingest.run_ingest backfill cs.CL [--max-papers 500]
    uv run python -m src.ingest.run_ingest incremental
"""
from __future__ import annotations

import argparse
import logging
import sys

from src.db import close_pool
from src.ingest.backfill import run_backfill
from src.ingest.incremental import IngestStats, run_incremental


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _print_summary(label: str, stats: IngestStats) -> None:
    print()
    print(f"=== {label} ===")
    print(f"  seen              : {stats.seen}")
    print(f"  inserted (new)    : {stats.inserted}")
    print(f"  reindexed (hash Δ): {stats.reindexed}")
    print(f"  skipped unchanged : {stats.skipped_unchanged}")
    print(f"  failed            : {stats.failed}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="run_ingest")
    sub = parser.add_subparsers(dest="mode", required=True)

    p_back = sub.add_parser("backfill", help="Bulk-load one category (bounded)")
    p_back.add_argument("category", help="arXiv category code, e.g. cs.CL")
    p_back.add_argument("--max-papers", type=int, default=None)
    p_back.add_argument(
        "--page-size",
        type=int,
        default=None,
        help=(
            "arXiv API page size (max_results per request). "
            "Overrides ARXIV_MAX_RESULTS_PER_PAGE for this run. "
            "Lower this (e.g. 25) if arXiv keeps returning 429."
        ),
    )

    p_incr = sub.add_parser(
        "incremental", help="Fetch only new/changed papers since last run"
    )
    p_incr.add_argument(
        "--page-size",
        type=int,
        default=None,
        help=(
            "arXiv API page size (max_results per request). "
            "Overrides ARXIV_MAX_RESULTS_PER_PAGE for this run."
        ),
    )

    args = parser.parse_args(argv)
    _configure_logging()

    try:
        if args.mode == "backfill":
            stats = run_backfill(
                args.category,
                max_papers=args.max_papers,
                page_size=args.page_size,
            )
            _print_summary(f"BACKFILL {args.category}", stats)
        else:  # incremental
            stats = run_incremental(page_size=args.page_size)
            _print_summary("INCREMENTAL", stats)
    finally:
        close_pool()
    return 0


if __name__ == "__main__":
    sys.exit(main())
