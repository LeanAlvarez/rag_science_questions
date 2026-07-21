"""Backfill: bulk historical load of a single category.

Run this once by hand when you enable a new category, then let the incremental
pass keep it fresh from that point on.

Idempotent: re-running backfill on a populated category just skips the papers
whose hash hasn't changed — the same code path incremental uses.
"""
from __future__ import annotations

import logging

from src.config import settings
from src.db import transaction
from src.ingest.arxiv_client import ArxivClient
from src.ingest.incremental import IngestStats, process_paper

log = logging.getLogger(__name__)


def ensure_category_registered(category: str) -> None:
    """Make sure the category is in active_categories (so incremental picks it up next)."""
    with transaction() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO active_categories (category)
            VALUES (%s)
            ON CONFLICT (category) DO NOTHING
            """,
            (category,),
        )


def run_backfill(category: str, *, max_papers: int | None = None) -> IngestStats:
    """Load up to `max_papers` newest papers from `category`."""
    limit = max_papers if max_papers is not None else settings.BACKFILL_MAX_PAPERS
    ensure_category_registered(category)

    stats = IngestStats()
    with ArxivClient() as client:
        for paper in client.iter_category(category, max_results=limit):
            stats.seen += 1
            try:
                outcome = process_paper(paper)
            except Exception:  # noqa: BLE001
                log.exception("Failed to process %s", paper.arxiv_id)
                stats.failed += 1
                continue

            if outcome == "inserted":
                stats.inserted += 1
            elif outcome == "reindexed":
                stats.reindexed += 1
            else:
                stats.skipped_unchanged += 1

            # Periodic progress log so long backfills don't feel dead.
            if stats.seen % 10 == 0:
                log.info(
                    "  progress: seen=%d inserted=%d reindexed=%d skipped=%d failed=%d",
                    stats.seen, stats.inserted, stats.reindexed,
                    stats.skipped_unchanged, stats.failed,
                )
    return stats
