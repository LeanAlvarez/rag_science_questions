"""Cross-encoder reranking.

A bi-encoder (the embedding model) turns query and chunk into vectors
INDEPENDENTLY and then compares them — fast but coarse. A cross-encoder feeds
`[query, chunk]` as a SINGLE input to a transformer and outputs one relevance
score — much more accurate on the last mile, but too slow to score the whole
corpus. So we only rerank the top-K survivors from hybrid retrieval.

Model: BAAI/bge-reranker-v2-m3.
  * Multilingual.
  * ~600 MB weights (small enough for CPU).
  * Outputs an UNBOUNDED logit; higher = more relevant. Negative scores are
    fine and mean "probably not relevant". Do NOT interpret the score as a
    probability without a sigmoid.
"""
from __future__ import annotations

from functools import lru_cache

from sentence_transformers import CrossEncoder

from src.config import settings
from src.core.retrieval import Candidate


@lru_cache
def _model() -> CrossEncoder:
    """Lazy-load the reranker once per process (~600 MB weights)."""
    return CrossEncoder(settings.RERANKER_MODEL, device=settings.MODEL_DEVICE)


def rerank(
    query: str,
    candidates: list[Candidate],
    *,
    top_k: int | None = None,
) -> list[Candidate]:
    """Re-score candidates against `query` and return the top-K by rerank_score.

    The input Candidate objects are mutated in place with `rerank_score` set —
    keeping the earlier vector / keyword / RRF scores intact so the caller can
    still see the full journey each chunk took.
    """
    top_k = top_k if top_k is not None else settings.RERANK_TOP_K
    if not candidates:
        return []

    pairs = [(query, c.content) for c in candidates]
    scores = _model().predict(pairs, show_progress_bar=False)

    for cand, score in zip(candidates, scores, strict=True):
        cand.rerank_score = float(score)

    return sorted(
        candidates,
        key=lambda c: c.rerank_score if c.rerank_score is not None else float("-inf"),
        reverse=True,
    )[:top_k]
