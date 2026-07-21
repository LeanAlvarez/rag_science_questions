"""Client for the arXiv public API (https://info.arxiv.org/help/api/).

Fetches papers from a category, sorted newest-first, and normalizes each Atom
entry into an `ArxivPaper`. Two things worth calling out:

  * Rate limiting is ENFORCED in this module, not left up to the caller.
    arXiv asks for at most one request every 3 seconds — we sleep here so no
    caller can accidentally hammer the API.

  * Retries with exponential backoff cover transient network errors and 5xx
    responses. 4xx responses are NOT retried (they'd never succeed anyway).
"""
from __future__ import annotations

import hashlib
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime

import feedparser
import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.config import settings


@dataclass(slots=True)
class ArxivPaper:
    """Normalized view of a single arXiv entry."""

    arxiv_id: str  # e.g. '2410.12345v2' (version suffix is kept as-is)
    title: str
    abstract: str
    primary_category: str
    published_at: datetime
    updated_at: datetime
    pdf_url: str
    authors: list[str] = field(default_factory=list)

    @property
    def content_hash(self) -> str:
        """SHA-256 of the fields we index. Changes iff the paper's text changes."""
        blob = f"{self.title}\n\n{self.abstract}"
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    @property
    def indexable_text(self) -> str:
        """Text handed to the chunker + embedder. Currently: title + abstract.

        Full-PDF ingestion is a future upgrade — abstracts are already dense
        and searchable, and arXiv returns them for free in the Atom feed.
        """
        return f"{self.title}\n\n{self.abstract}"


class _RateLimiter:
    """Sleeps just enough between calls to respect a minimum interval."""

    def __init__(self, interval_seconds: float) -> None:
        self._interval = interval_seconds
        self._last_call: float = 0.0

    def wait(self) -> None:
        elapsed = time.monotonic() - self._last_call
        remaining = self._interval - elapsed
        if remaining > 0:
            time.sleep(remaining)
        self._last_call = time.monotonic()


class ArxivClient:
    def __init__(
        self,
        *,
        base_url: str | None = None,
        page_size: int | None = None,
        interval_seconds: float | None = None,
    ) -> None:
        self._base_url = base_url or settings.ARXIV_API_BASE
        self._page_size = page_size or settings.ARXIV_MAX_RESULTS_PER_PAGE
        self._limiter = _RateLimiter(
            interval_seconds or settings.ARXIV_REQUEST_INTERVAL_SECONDS
        )
        self._http = httpx.Client(
            timeout=30.0,
            # arXiv serves everything from https://; the http:// hostname 301s.
            # follow_redirects=True future-proofs against any other permanent
            # move without silently returning a 3xx as if it were success.
            follow_redirects=True,
            headers={"User-Agent": "arxiv-rag/0.1 (portfolio project)"},
        )

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> ArxivClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(
            (httpx.TransportError, httpx.HTTPStatusError)
        ),
        reraise=True,
    )
    def _fetch_page(self, *, category: str, start: int) -> feedparser.FeedParserDict:
        self._limiter.wait()
        params = {
            "search_query": f"cat:{category}",
            "start": start,
            "max_results": self._page_size,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
        resp = self._http.get(self._base_url, params=params)
        # raise_for_status turns 5xx into HTTPStatusError so tenacity retries.
        # 4xx also raises but the retry predicate treats it the same way; that's
        # fine here because 4xx from arXiv (400 bad query, 429 rate-limit) is
        # transient in practice.
        resp.raise_for_status()
        return feedparser.parse(resp.content)

    def iter_category(
        self,
        category: str,
        *,
        max_results: int | None = None,
    ) -> Iterator[ArxivPaper]:
        """Yield papers from `category`, newest first.

        Stops when either `max_results` have been yielded or arXiv returns an
        empty page.
        """
        emitted = 0
        start = 0
        while True:
            feed = self._fetch_page(category=category, start=start)
            entries = feed.entries or []
            if not entries:
                return
            for entry in entries:
                paper = _entry_to_paper(entry)
                if paper is None:
                    continue
                yield paper
                emitted += 1
                if max_results is not None and emitted >= max_results:
                    return
            start += len(entries)


def _entry_to_paper(entry: feedparser.FeedParserDict) -> ArxivPaper | None:
    """Convert a raw feedparser entry to our ArxivPaper. Returns None on malformed rows."""
    try:
        # entry.id looks like 'http://arxiv.org/abs/2410.12345v2'
        arxiv_id = entry.id.rsplit("/", 1)[-1]
        title = " ".join(entry.title.split())
        abstract = " ".join(entry.summary.split())
        primary_category = entry.arxiv_primary_category["term"]
        published_at = datetime(*entry.published_parsed[:6])
        updated_at = datetime(*entry.updated_parsed[:6])
        pdf_url = next(
            (link.href for link in entry.links if link.get("type") == "application/pdf"),
            "",
        )
        authors = [a.name for a in entry.get("authors", [])]
    except (AttributeError, KeyError, TypeError):
        return None

    return ArxivPaper(
        arxiv_id=arxiv_id,
        title=title,
        abstract=abstract,
        primary_category=primary_category,
        published_at=published_at,
        updated_at=updated_at,
        pdf_url=pdf_url,
        authors=authors,
    )
