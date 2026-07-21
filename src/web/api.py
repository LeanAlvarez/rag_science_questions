"""FastAPI app for the web-facing side of arxiv-rag.

Three routes:
    POST /api/ask         → full RAG answer as JSON (no streaming)
    POST /api/ask/stream  → same, but as an SSE stream so the UI can render
                            sources immediately and tokens as they arrive
    GET  /api/health      → cheap liveness probe

Plus a static mount for the built React app at `/`. That way, in production,
FastAPI serves both the API and the UI from a single origin — no CORS needed.
In development, the Vite dev server runs on :5173 and proxies /api/* to :8000,
so CORS IS needed for browsers to allow the requests.
"""
from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.config import settings
from src.core.generation import NoModelSucceeded
from src.query.pipeline import (
    answer_question,
    answer_question_stream,
)

log = logging.getLogger(__name__)

app = FastAPI(
    title="arxiv-rag",
    description="RAG over arXiv papers — Phase 4 web API.",
    version="0.1.0",
)

if settings.web_allowed_origins_list:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.web_allowed_origins_list,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )


# ---------------------------------------------------------------------------
# Request / response schemas — Pydantic gives us free validation + OpenAPI docs.
# ---------------------------------------------------------------------------
class AskRequest(BaseModel):
    question: str = Field(
        ..., min_length=1, max_length=2000,
        description="The user's question, in any language.",
    )


class SourceOut(BaseModel):
    arxiv_id: str
    title: str
    url: str


class CandidateOut(BaseModel):
    chunk_id: int
    arxiv_id: str
    chunk_index: int
    content: str
    title: str
    vector_similarity: float | None = None
    vector_rank: int | None = None
    keyword_score: float | None = None
    keyword_rank: int | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None


class AnswerResponse(BaseModel):
    text: str
    sources: list[SourceOut]
    candidates: list[CandidateOut]
    model_used: str
    context_used: bool


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/api/health")
def health() -> dict:
    """Liveness probe. Doesn't touch the DB or any model — always cheap."""
    return {"status": "ok"}


@app.post("/api/ask", response_model=AnswerResponse)
def ask(req: AskRequest) -> AnswerResponse:
    """Blocking end-to-end call: retrieval + rerank + generation."""
    try:
        ans = answer_question(req.question)
    except NoModelSucceeded as e:
        raise HTTPException(status_code=502, detail=str(e)) from e

    return AnswerResponse(
        text=ans.text,
        sources=[SourceOut(**s.__dict__) for s in ans.sources],
        candidates=[
            CandidateOut(**{k: getattr(c, k) for k in CandidateOut.model_fields})
            for c in ans.candidates
        ],
        model_used=ans.model_used,
        context_used=ans.context_used,
    )


@app.post("/api/ask/stream")
def ask_stream(req: AskRequest) -> StreamingResponse:
    """SSE stream: context event first, then token events, then done event."""
    def sse_events() -> Iterator[bytes]:
        try:
            for event in answer_question_stream(req.question):
                payload = json.dumps(event["data"], ensure_ascii=False)
                yield f"event: {event['event']}\ndata: {payload}\n\n".encode()
        except NoModelSucceeded as e:
            # Turn the exception into a final error event so the browser sees
            # it as data (not a broken connection).
            err = json.dumps({"message": str(e)}, ensure_ascii=False)
            yield f"event: error\ndata: {err}\n\n".encode()
        except Exception as e:  # noqa: BLE001
            log.exception("unexpected error during stream")
            err = json.dumps({"message": f"internal error: {e}"}, ensure_ascii=False)
            yield f"event: error\ndata: {err}\n\n".encode()

    return StreamingResponse(
        sse_events(),
        media_type="text/event-stream",
        headers={
            # Nginx / proxies love to buffer SSE unless told not to.
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# ---------------------------------------------------------------------------
# Static mount — production only. In dev the Vite server serves the UI on :5173
# and hot-reloads; we skip the mount if the built assets aren't there yet.
# ---------------------------------------------------------------------------
_STATIC_DIR = Path(__file__).parent / "static"
if _STATIC_DIR.exists() and (_STATIC_DIR / "index.html").exists():
    # `html=True` makes StaticFiles serve index.html for unknown paths — the
    # SPA behaviour we want (client-side routing later, direct URLs work).
    app.mount("/", StaticFiles(directory=_STATIC_DIR, html=True), name="ui")
