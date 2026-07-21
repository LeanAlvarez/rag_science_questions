"""Hybrid retrieval: vector nearest-neighbour + Postgres full-text, fused with RRF.

Why hybrid, not pure vector:
  * Vector search alone misses exact-keyword hits (acronyms, model names,
    identifier tokens, rare technical terms) because the embedding smooths
    them into a general concept vector.
  * Keyword search alone misses paraphrases and semantic reformulations.
  * The two are complementary and the errors are largely uncorrelated, so
    fusing their rankings recovers most of both.

Reciprocal Rank Fusion (RRF) is the standard, calibration-free way to combine
ranked lists:
    RRF_score(doc) = Σ over lists  ( 1 / (k + rank_in_that_list) )
It ignores the absolute scores each retriever produces (which are on wildly
different scales) and only uses the ORDER — so it doesn't matter that cosine
similarities and ts_rank_cd values aren't comparable.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.config import settings
from src.core.embeddings import embed_one
from src.db import get_connection


@dataclass
class Candidate:
    """A retrieved chunk plus every score picked up along the pipeline.

    We keep per-stage scores side-by-side (rather than collapsing them) so the
    web debug view / CLI can show WHY a chunk survived: was it strong vector
    similarity, a great keyword hit, both, or a rerank rescue?
    """
    chunk_id: int
    arxiv_id: str
    chunk_index: int
    content: str
    title: str
    vector_similarity: float | None = None  # cosine similarity in [-1, 1]; higher is better
    vector_rank: int | None = None           # 1-indexed rank in the vector-search list
    keyword_score: float | None = None       # ts_rank_cd; higher is better
    keyword_rank: int | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None        # cross-encoder logit; higher is better


def vector_search(query_vec: np.ndarray, *, top_k: int) -> list[Candidate]:
    """K-nearest-neighbour search over chunk embeddings (cosine, HNSW-indexed).

    `<=>` is pgvector's cosine DISTANCE. For L2-normalized vectors (bge-m3 gives
    us those) `1 - distance == cosine similarity`, which we expose so callers
    can reason about "how similar" in the usual [-1, 1] range.
    """
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                c.id, c.arxiv_id, c.chunk_index, c.content, ip.title,
                1 - (c.embedding <=> %s) AS similarity
            FROM chunks c
            JOIN ingested_papers ip ON ip.arxiv_id = c.arxiv_id
            ORDER BY c.embedding <=> %s
            LIMIT %s
            """,
            (query_vec, query_vec, top_k),
        )
        rows = cur.fetchall()
    return [
        Candidate(
            chunk_id=row[0],
            arxiv_id=row[1],
            chunk_index=row[2],
            content=row[3],
            title=row[4],
            vector_similarity=float(row[5]),
            vector_rank=i + 1,
        )
        for i, row in enumerate(rows)
    ]


def keyword_search(query_text: str, *, top_k: int) -> list[Candidate]:
    """Postgres full-text search against the GENERATED tsvector column.

    Uses `websearch_to_tsquery` (not `plainto_tsquery`) because it accepts the
    user-search-box syntax people actually type — `"exact phrase"`, `-excluded`,
    `word OR word` — without erroring on odd punctuation. `ts_rank_cd` scores
    with cover-density, which rewards matches that appear close together in
    the text.
    """
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                c.id, c.arxiv_id, c.chunk_index, c.content, ip.title,
                ts_rank_cd(c.content_tsv, q) AS score
            FROM chunks c
            JOIN ingested_papers ip ON ip.arxiv_id = c.arxiv_id,
                 websearch_to_tsquery('english', %s) q
            WHERE c.content_tsv @@ q
            ORDER BY score DESC
            LIMIT %s
            """,
            (query_text, top_k),
        )
        rows = cur.fetchall()
    return [
        Candidate(
            chunk_id=row[0],
            arxiv_id=row[1],
            chunk_index=row[2],
            content=row[3],
            title=row[4],
            keyword_score=float(row[5]),
            keyword_rank=i + 1,
        )
        for i, row in enumerate(rows)
    ]


def rrf_fuse(
    rankings: list[list[Candidate]],
    *,
    top_k: int,
    k: int | None = None,
) -> list[Candidate]:
    """Reciprocal Rank Fusion of two or more ranked candidate lists.

    Merges by `chunk_id`: when a chunk appears in more than one input list, we
    keep a single Candidate and copy over whichever per-stage scores each list
    brought. The Candidate's final `rrf_score` is the sum of `1/(k+rank)`
    contributions.
    """
    k = k if k is not None else settings.RRF_K
    merged: dict[int, Candidate] = {}
    scores: dict[int, float] = {}

    for ranking in rankings:
        for rank, cand in enumerate(ranking, start=1):
            if cand.chunk_id not in merged:
                merged[cand.chunk_id] = cand
            else:
                # Same chunk seen from another retriever — enrich the row with
                # scores this ranking has that the existing one didn't.
                existing = merged[cand.chunk_id]
                if cand.vector_similarity is not None and existing.vector_similarity is None:
                    existing.vector_similarity = cand.vector_similarity
                    existing.vector_rank = cand.vector_rank
                if cand.keyword_score is not None and existing.keyword_score is None:
                    existing.keyword_score = cand.keyword_score
                    existing.keyword_rank = cand.keyword_rank
            scores[cand.chunk_id] = scores.get(cand.chunk_id, 0.0) + 1.0 / (k + rank)

    for chunk_id, score in scores.items():
        merged[chunk_id].rrf_score = score

    ordered = sorted(merged.values(), key=lambda c: c.rrf_score or 0.0, reverse=True)
    return ordered[:top_k]


def hybrid_search(query: str, *, top_k: int | None = None) -> list[Candidate]:
    """Vector + keyword search fused with RRF. Returns the top-K survivors.

    This is the "recall" stage — it's cheap and aims to include the right
    passages in the top ~20. The subsequent rerank is the "precision" stage.
    """
    top_k = top_k if top_k is not None else settings.RETRIEVAL_TOP_K_AFTER_RRF

    query_vec = embed_one(query)
    vec = vector_search(query_vec, top_k=settings.RETRIEVAL_TOP_K_VECTOR)
    kw = keyword_search(query, top_k=settings.RETRIEVAL_TOP_K_KEYWORD)

    return rrf_fuse([vec, kw], top_k=top_k)
