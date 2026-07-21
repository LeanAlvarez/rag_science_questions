"""Text → list of overlapping chunks, sized in TOKENS not characters.

Why the embedding model's own tokenizer (not tiktoken, not char count):
  Each embedding model has its own vocabulary (bge-small-en-v1.5 uses BERT
  WordPiece; bge-m3 used SentencePiece). If we bound chunks by characters
  or by an OpenAI tokenizer, we can silently overshoot the model's context
  window (512 tokens for bge-small-en-v1.5), or waste embedding capacity by
  undershooting. Loading only the tokenizer (a few MB, not the full model
  weights) is cheap. CHUNK_SIZE_TOKENS should stay ≤ 512 for the default model.

Strategy:
  1. Split the text on blank lines into paragraphs.
  2. Greedily pack paragraphs into a chunk until adding the next one would
     exceed CHUNK_SIZE_TOKENS.
  3. Start each new chunk with the last CHUNK_OVERLAP_TOKENS of the previous
     one — so a sentence straddling a boundary is still retrievable.
  4. If a single paragraph is bigger than the chunk budget, split it at the
     token level.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from src.config import settings


@dataclass(slots=True)
class Chunk:
    index: int
    content: str


@lru_cache
def _tokenizer() -> PreTrainedTokenizerBase:
    """Lazy-load the embedding model's tokenizer (weights are NOT loaded)."""
    return AutoTokenizer.from_pretrained(settings.EMBEDDING_MODEL)


def _tok_len(text: str) -> int:
    return len(_tokenizer().encode(text, add_special_tokens=False))


def _split_paragraphs(text: str) -> list[str]:
    """Split on blank lines. Falls back to line-by-line if no blank lines exist."""
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    return paras or [line.strip() for line in text.splitlines() if line.strip()]


def _split_long_paragraph(paragraph: str, max_tokens: int) -> list[str]:
    """Break a paragraph that's bigger than a whole chunk into token-bounded pieces."""
    tok = _tokenizer()
    ids = tok.encode(paragraph, add_special_tokens=False)
    return [
        tok.decode(ids[start : start + max_tokens], skip_special_tokens=True)
        for start in range(0, len(ids), max_tokens)
    ]


def _overlap_tail(text: str, overlap_tokens: int) -> str:
    """Return the last `overlap_tokens` tokens of `text` as a string, or ''."""
    if overlap_tokens <= 0:
        return ""
    tok = _tokenizer()
    ids = tok.encode(text, add_special_tokens=False)
    if not ids:
        return ""
    return tok.decode(ids[-overlap_tokens:], skip_special_tokens=True)


def chunk_text(
    text: str,
    *,
    max_tokens: int | None = None,
    overlap_tokens: int | None = None,
) -> list[Chunk]:
    """Split `text` into overlapping chunks bounded by token counts."""
    max_tokens = max_tokens if max_tokens is not None else settings.CHUNK_SIZE_TOKENS
    overlap_tokens = overlap_tokens if overlap_tokens is not None else settings.CHUNK_OVERLAP_TOKENS

    if not text.strip():
        return []

    paragraphs = _split_paragraphs(text)
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = _tok_len(para)

        # A single paragraph that's larger than the whole budget: flush what
        # we have, then emit the paragraph as its own token-sliced chunks.
        if para_tokens > max_tokens:
            if current:
                chunks.append("\n\n".join(current))
                current, current_tokens = [], 0
            chunks.extend(_split_long_paragraph(para, max_tokens))
            continue

        # Adding this paragraph would overflow — close the current chunk,
        # then seed the next one with the overlap tail of what we just closed.
        if current and current_tokens + para_tokens > max_tokens:
            chunks.append("\n\n".join(current))
            tail = _overlap_tail(chunks[-1], overlap_tokens)
            current = [tail] if tail else []
            current_tokens = _tok_len(tail) if tail else 0

        current.append(para)
        current_tokens += para_tokens

    if current:
        chunks.append("\n\n".join(current))

    return [Chunk(index=i, content=c) for i, c in enumerate(chunks)]
