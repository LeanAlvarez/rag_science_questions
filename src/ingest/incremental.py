"""Incremental ingestion.

The two pieces that matter here:

  * `upsert_paper()` is the transactional core. DELETE chunks + UPSERT paper
    + INSERT new chunks all run in ONE transaction (see src/db.py::transaction).
    On any exception the whole thing rolls back — the corpus never sees a
    half-updated paper.

  * `process_paper()` is a cheap "should we bother?" wrapper that reads the
    stored hash BEFORE running the (expensive) chunker + embedder. If the
    hash is unchanged, we skip the paper entirely.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from psycopg import Connection
from psycopg.rows import tuple_row

from src.core.chunking import chunk_text
from src.core.embeddings import embed
from src.db import transaction
from src.ingest.arxiv_client import ArxivClient, ArxivPaper

log = logging.getLogger(__name__)

# After N contiguous "unchanged" papers we assume we've caught up. Papers are
# sorted by submittedDate DESC, so once we've reconfirmed the newest N, every
# older paper is (by construction) also already indexed.
STOP_AFTER_UNCHANGED = 20


@dataclass(slots=True)
class IngestStats:
    seen: int = 0
    inserted: int = 0
    reindexed: int = 0
    skipped_unchanged: int = 0
    failed: int = 0

    def merge(self, other: IngestStats) -> None:
        self.seen += other.seen
        self.inserted += other.inserted
        self.reindexed += other.reindexed
        self.skipped_unchanged += other.skipped_unchanged
        self.failed += other.failed


def get_active_categories(conn: Connection) -> list[str]:
    with conn.cursor(row_factory=tuple_row) as cur:
        cur.execute(
            "SELECT category FROM active_categories WHERE enabled = TRUE ORDER BY category"
        )
        return [row[0] for row in cur.fetchall()]


def _existing_hash(conn: Connection, arxiv_id: str) -> str | None:
    """Return the stored content_hash for a paper, or None if we've never seen it."""
    with conn.cursor(row_factory=tuple_row) as cur:
        cur.execute(
            "SELECT content_hash FROM ingested_papers WHERE arxiv_id = %s",
            (arxiv_id,),
        )
        row = cur.fetchone()
        return row[0] if row else None


def upsert_paper(paper: ArxivPaper) -> str:
    """Transactionally (re-)index a single paper.

    The transaction runs these steps in order, atomically:
      1. DELETE FROM chunks WHERE arxiv_id = X  — explicit, not via CASCADE,
         because we're UPDATE-ing the parent row (not DELETE-ing it), so
         CASCADE would never fire.
      2. INSERT ... ON CONFLICT DO UPDATE      — upsert the state row.
      3. INSERT INTO chunks (executemany)      — persist new chunks+embeddings.
      4. COMMIT                                 — automatic on clean exit;
                                                  ROLLBACK on any exception.

    Returns "inserted" (first time) or "reindexed" (existing paper, hash changed).
    The caller (`process_paper`) is responsible for the up-front hash check that
    keeps us from re-embedding unchanged papers.
    """
    chunks = chunk_text(paper.indexable_text)
    if not chunks:
        raise ValueError(f"{paper.arxiv_id}: chunking produced 0 chunks (empty text?)")
    embeddings = embed([c.content for c in chunks])

    with transaction() as conn:
        prior = _existing_hash(conn, paper.arxiv_id)
        action = "inserted" if prior is None else "reindexed"

        with conn.cursor() as cur:
            # 1. Wipe old chunks so two versions never coexist in the corpus.
            cur.execute("DELETE FROM chunks WHERE arxiv_id = %s", (paper.arxiv_id,))

            # 2. Upsert the state row. content_hash is what drives future skips.
            cur.execute(
                """
                INSERT INTO ingested_papers (
                    arxiv_id, content_hash, title, primary_category, published_at
                ) VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (arxiv_id) DO UPDATE SET
                    content_hash     = EXCLUDED.content_hash,
                    title            = EXCLUDED.title,
                    primary_category = EXCLUDED.primary_category,
                    published_at     = EXCLUDED.published_at,
                    last_indexed_at  = NOW()
                """,
                (
                    paper.arxiv_id,
                    paper.content_hash,
                    paper.title,
                    paper.primary_category,
                    paper.published_at,
                ),
            )

            # 3. Insert new chunks. executemany is fine — chunk count per paper is small.
            cur.executemany(
                """
                INSERT INTO chunks (arxiv_id, chunk_index, content, embedding)
                VALUES (%s, %s, %s, %s)
                """,
                [
                    (paper.arxiv_id, c.index, c.content, embeddings[i].tolist())
                    for i, c in enumerate(chunks)
                ],
            )
    return action


def process_paper(paper: ArxivPaper) -> str:
    """Decide-and-do wrapper. Returns 'inserted' | 'reindexed' | 'skipped'."""
    # Cheap SELECT first — skipping unchanged papers is the whole point of
    # incremental ingestion. Embedding is orders of magnitude more expensive
    # than a primary-key lookup.
    with transaction() as conn:
        prior_hash = _existing_hash(conn, paper.arxiv_id)
    if prior_hash == paper.content_hash:
        return "skipped"
    return upsert_paper(paper)


def run_incremental_for_category(client: ArxivClient, category: str) -> IngestStats:
    """Walk a category newest-first, stopping once we hit a run of unchanged papers."""
    stats = IngestStats()
    unchanged_streak = 0

    for paper in client.iter_category(category):
        stats.seen += 1
        try:
            outcome = process_paper(paper)
        except Exception:  # noqa: BLE001 — one bad paper shouldn't kill the pass
            log.exception("Failed to process %s", paper.arxiv_id)
            stats.failed += 1
            continue

        if outcome == "inserted":
            stats.inserted += 1
            unchanged_streak = 0
        elif outcome == "reindexed":
            stats.reindexed += 1
            unchanged_streak = 0
        else:
            stats.skipped_unchanged += 1
            unchanged_streak += 1
            if unchanged_streak >= STOP_AFTER_UNCHANGED:
                log.info(
                    "%s: %d consecutive unchanged papers → assuming caught up",
                    category, unchanged_streak,
                )
                break
    return stats


def run_incremental() -> IngestStats:
    """Run one incremental pass across all enabled categories."""
    with transaction() as conn:
        categories = get_active_categories(conn)
    if not categories:
        raise RuntimeError(
            "active_categories is empty. Add one, e.g.:\n"
            "  INSERT INTO active_categories (category) VALUES ('cs.CL');\n"
            "or run: uv run python -m src.ingest.run_ingest backfill cs.CL"
        )

    total = IngestStats()
    with ArxivClient() as client:
        for category in categories:
            log.info("Incremental pass: category=%s", category)
            per_cat = run_incremental_for_category(client, category)
            log.info(
                "  → %s: seen=%d inserted=%d reindexed=%d skipped=%d failed=%d",
                category, per_cat.seen, per_cat.inserted, per_cat.reindexed,
                per_cat.skipped_unchanged, per_cat.failed,
            )
            total.merge(per_cat)
    return total
