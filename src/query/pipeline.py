"""Shared query engine — the "brain" reused by both the web API (Phase 4) and
the Telegram bot (Phase 5). No entry point ever runs retrieval, rerank, or
generation on its own; they always go through this file. That's how we prevent
the "wait, does the web use the same retriever as the bot?" divergence.

Phase 3 completes the pipeline:
  retrieve_and_rerank()  → hybrid + cross-encoder, no LLM (also useful on its own)
  answer_question()      → same, then hands the fragments to OpenRouter for grounded generation
"""
from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass

from src.core.generation import openrouter_client
from src.core.rerank import rerank
from src.core.retrieval import Candidate, hybrid_search

# Some OpenRouter free-tier providers inject a safety classifier line into the
# CONTENT of the response (not as a separate field on the delta), so our SSE
# parser can't just "ignore" it — the model gets it as part of its output text.
# Match ONLY lines whose whole content is a `<label>: <value>` classifier tag.
# A sentence like "The user safety implications are ..." will NOT match because
# it doesn't fit the end-anchored pattern.
_SAFETY_CLASSIFIER_LINE_RE = re.compile(
    r"^\**\s*"
    r"(user\s+safety|content\s+safety|content\s+category|safety|classification)"
    r"\s*[:\-]\s*\S+\s*\**\s*$",
    re.IGNORECASE,
)


@dataclass(slots=True)
class Source:
    """A cited paper — deduplicated view derived from the fragments used."""
    arxiv_id: str
    title: str
    url: str  # canonical https://arxiv.org/abs/{arxiv_id}


@dataclass(slots=True)
class Answer:
    """End-to-end result of `answer_question`.

    `candidates` is preserved so the web/bot layers can expose the evidence
    (which fragments the model was actually shown, with their scores).
    `sources` is a deduplicated, ready-to-render list of the papers those
    fragments came from.
    """
    text: str
    sources: list[Source]
    candidates: list[Candidate]
    model_used: str
    context_used: bool  # False = we skipped generation because retrieval found nothing


# Anti-hallucination system prompt. The three rules — cite inline, ground in
# fragments, admit ignorance — are what make this a RAG answer instead of the
# model's parametric memory dressed up in fragments.
SYSTEM_PROMPT = (
    "You are a research assistant that answers questions using ONLY the "
    "numbered fragments provided. Each fragment is a passage from an arXiv paper.\n\n"
    "Rules:\n"
    "- Ground every claim in the fragments.\n"
    "- Cite the fragments you use inline with [1], [2], etc.\n"
    "- If the fragments do not contain enough information, say so directly. "
    "Do not invent facts.\n"
    "- Be concise. Do not include a bibliography — sources are appended "
    "programmatically."
)


def retrieve_and_rerank(question: str) -> list[Candidate]:
    """Two-stage retrieval (Phase 2): recall via hybrid RRF, then precision via cross-encoder."""
    candidates = hybrid_search(question)
    return rerank(question, candidates)


def _build_messages(question: str, candidates: list[Candidate]) -> list[dict]:
    fragment_blocks = [
        f"[{i}] arxiv:{c.arxiv_id} — {c.title}\n{c.content}\n"
        for i, c in enumerate(candidates, start=1)
    ]
    fragments_text = "\n".join(fragment_blocks)
    user = f"Question: {question}\n\nFragments:\n\n{fragments_text}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def _dedupe_sources(candidates: list[Candidate]) -> list[Source]:
    """One Source per unique arxiv_id, in the order they were first surfaced."""
    seen: set[str] = set()
    sources: list[Source] = []
    for c in candidates:
        if c.arxiv_id in seen:
            continue
        seen.add(c.arxiv_id)
        sources.append(
            Source(
                arxiv_id=c.arxiv_id,
                title=c.title,
                url=f"https://arxiv.org/abs/{c.arxiv_id}",
            )
        )
    return sources


def answer_question(question: str) -> Answer:
    """Full RAG pipeline: retrieval → rerank → LLM generation with citations."""
    candidates = retrieve_and_rerank(question)
    if not candidates:
        return Answer(
            text=(
                "I don't have any indexed material that matches this question. "
                "Try running an ingest first, or rephrase."
            ),
            sources=[],
            candidates=[],
            model_used="",
            context_used=False,
        )

    messages = _build_messages(question, candidates)
    completion = openrouter_client().complete(messages)

    return Answer(
        text=completion.text,
        sources=_dedupe_sources(candidates),
        candidates=candidates,
        model_used=completion.model_used,
        context_used=True,
    )


def answer_question_stream(question: str) -> Iterator[dict]:
    """Streaming variant of `answer_question`. Yields dicts shaped for SSE:

        {"event": "context", "data": {"sources": [...], "candidates": [...]}}
        {"event": "token",   "data": {"text": "delta"}}       (many)
        {"event": "done",    "data": {"model_used": "...", "context_used": True}}

    The `context` event is emitted BEFORE any tokens so the UI can render
    sources + evidence skeletons while the model is still generating — that's
    the whole win of streaming for RAG.
    """
    candidates = retrieve_and_rerank(question)
    sources = _dedupe_sources(candidates)
    yield {
        "event": "context",
        "data": {
            "sources": [
                {"arxiv_id": s.arxiv_id, "title": s.title, "url": s.url}
                for s in sources
            ],
            "candidates": [_candidate_to_dict(c) for c in candidates],
        },
    }

    if not candidates:
        yield {
            "event": "done",
            "data": {"model_used": "", "context_used": False},
        }
        return

    messages = _build_messages(question, candidates)
    model_used = ""
    # Line-boundary buffer so we can drop safety-classifier lines cleanly.
    # In-line deltas that don't cross a newline are held until the line ends;
    # in practice LLM answers have paragraph breaks often enough that this
    # feels like paragraph-paced streaming, not batch delivery.
    line_buf = ""
    just_dropped_noise = False
    for delta, model_id in openrouter_client().complete_stream(messages):
        model_used = model_id
        line_buf += delta
        while "\n" in line_buf:
            line, line_buf = line_buf.split("\n", 1)
            if _is_safety_classifier_noise(line):
                just_dropped_noise = True
                continue
            if just_dropped_noise and not line.strip():
                # Swallow the blank line that usually follows the classifier
                # so the answer doesn't start with a stray empty line.
                just_dropped_noise = False
                continue
            just_dropped_noise = False
            yield {"event": "token", "data": {"text": line + "\n"}}

    # Flush trailing content (final line without a terminating \n).
    if line_buf and not _is_safety_classifier_noise(line_buf):
        yield {"event": "token", "data": {"text": line_buf}}

    yield {
        "event": "done",
        "data": {"model_used": model_used, "context_used": True},
    }


def _is_safety_classifier_noise(line: str) -> bool:
    """True if `line` is a `<label>: <value>` safety-classifier tag on its own."""
    stripped = line.strip()
    if not stripped:
        return False
    return bool(_SAFETY_CLASSIFIER_LINE_RE.match(stripped))


def _candidate_to_dict(c: Candidate) -> dict:
    """Serialize a Candidate for the wire. Kept explicit (not asdict) so we
    control exactly which fields are exposed to the frontend."""
    return {
        "chunk_id": c.chunk_id,
        "arxiv_id": c.arxiv_id,
        "chunk_index": c.chunk_index,
        "content": c.content,
        "title": c.title,
        "vector_similarity": c.vector_similarity,
        "vector_rank": c.vector_rank,
        "keyword_score": c.keyword_score,
        "keyword_rank": c.keyword_rank,
        "rrf_score": c.rrf_score,
        "rerank_score": c.rerank_score,
    }
