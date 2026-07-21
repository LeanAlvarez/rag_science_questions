"""Phase 3 CLI: end-to-end RAG in the terminal.

By default this runs the whole pipeline — retrieval + rerank + generation —
and prints the answer plus the papers it cites. Two flags for debugging:

  --retrieval-only   Skip the LLM entirely (Phase 2 behavior). Useful when
                     you want to check "is my retriever surfacing the right
                     passages?" without spending an OpenRouter call.
  --show-context     Also print each reranked fragment with all its scores.

Usage:
    uv run python -m src.query.run_query "your question here"
    uv run python -m src.query.run_query "..." --show-context
    uv run python -m src.query.run_query "..." --retrieval-only
"""
from __future__ import annotations

import argparse
import logging
import sys
import textwrap

from src.core.generation import close_openrouter_client
from src.core.retrieval import Candidate
from src.db import close_pool
from src.query.pipeline import Answer, answer_question, retrieve_and_rerank


def _fmt(x: float | None, spec: str = ".4f") -> str:
    return f"{x:{spec}}" if x is not None else "  -   "


def _print_candidate(idx: int, cand: Candidate) -> None:
    print()
    print(f"─── #{idx}  arxiv:{cand.arxiv_id}  chunk:{cand.chunk_index} ───")
    print(
        f"  scores  "
        f"vec_sim={_fmt(cand.vector_similarity)}  "
        f"kw={_fmt(cand.keyword_score)}  "
        f"rrf={_fmt(cand.rrf_score)}  "
        f"rerank={_fmt(cand.rerank_score, '+.4f')}"
    )
    print(
        f"  ranks   "
        f"vec_rank={cand.vector_rank if cand.vector_rank is not None else '-'}  "
        f"kw_rank={cand.keyword_rank if cand.keyword_rank is not None else '-'}"
    )
    print(f"  title   {cand.title}")
    body = textwrap.shorten(cand.content, width=400, placeholder=" …")
    print(f"  body    {body}")


def _print_answer(ans: Answer) -> None:
    print()
    print("─── Answer ───")
    print()
    # Rewrap paragraphs for readability at ~100 chars but preserve blank lines.
    for para in ans.text.split("\n"):
        if not para.strip():
            print()
            continue
        for line in textwrap.wrap(para, width=100):
            print(f"  {line}")

    print()
    print("─── Sources ───")
    if not ans.sources:
        print("  (none)")
    for i, src in enumerate(ans.sources, start=1):
        print(f"  [{i}] arxiv:{src.arxiv_id} — {src.title}")
        print(f"      {src.url}")

    print()
    if ans.model_used:
        print(f"Model used: {ans.model_used}")
    if not ans.context_used:
        print("(No evidence was retrieved — the answer above is a canned fallback.)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="run_query")
    parser.add_argument("question", nargs="+", help="Free-text question")
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Skip the LLM. Just show the retrieved+reranked chunks.",
    )
    parser.add_argument(
        "--show-context",
        action="store_true",
        help="Also print the reranked fragments alongside the answer.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="DEBUG logging")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    question = " ".join(args.question)
    print(f"Q: {question}")

    try:
        if args.retrieval_only:
            candidates = retrieve_and_rerank(question)
            if not candidates:
                print("\n(no results — did you run backfill / incremental first?)")
                return 0
            for i, cand in enumerate(candidates, start=1):
                _print_candidate(i, cand)
            print()
            return 0

        ans = answer_question(question)
        _print_answer(ans)
        if args.show_context and ans.candidates:
            print()
            print("─── Context (fragments the model saw) ───")
            for i, cand in enumerate(ans.candidates, start=1):
                _print_candidate(i, cand)
        print()
    finally:
        close_pool()
        close_openrouter_client()

    return 0


if __name__ == "__main__":
    sys.exit(main())
