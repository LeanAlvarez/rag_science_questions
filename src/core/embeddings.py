"""Local embedding model wrapper.

Loads sentence-transformers on first use (weights are downloaded once and
cached), then serves batched encode() calls cheaply.

The default model (bge-small-en-v1.5) returns L2-normalized 384-d vectors,
which is why the pgvector index in sql/schema.sql uses vector_cosine_ops.
If you swap models: update EMBEDDING_MODEL, EMBEDDING_DIMENSION, and the
VECTOR(N) column in sql/schema.sql, then re-index the corpus.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import settings


@lru_cache
def _model() -> SentenceTransformer:
    return SentenceTransformer(settings.EMBEDDING_MODEL, device=settings.MODEL_DEVICE)


def embed(texts: list[str], *, batch_size: int = 16) -> np.ndarray:
    """Encode a batch of strings. Returns (N, EMBEDDING_DIMENSION) float32."""
    if not texts:
        return np.empty((0, settings.EMBEDDING_DIMENSION), dtype=np.float32)

    vectors = _model().encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,  # bge-* models expect/recommend L2-normalized
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    # Fail loudly on a dimension mismatch so a bad config surfaces here, not
    # as a cryptic pgvector error deep in the INSERT.
    if vectors.shape[1] != settings.EMBEDDING_DIMENSION:
        raise ValueError(
            f"Embedding model returned dim {vectors.shape[1]} but config says "
            f"EMBEDDING_DIMENSION={settings.EMBEDDING_DIMENSION}. Update the env "
            f"var AND the VECTOR(N) column in sql/schema.sql to match."
        )
    return vectors.astype(np.float32)


def embed_one(text: str) -> np.ndarray:
    """Convenience wrapper for single-text encoding."""
    return embed([text])[0]
