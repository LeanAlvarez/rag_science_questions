"""Client for the arXiv public API (https://info.arxiv.org/help/api/).

Fetches papers from a category, sorted newest-first, and normalizes each Atom
entry into an `ArxivPaper`. Four things worth calling out:

  * Rate limiting is ENFORCED in this module, not left up to the caller.
    arXiv asks for at most one request every 3 seconds — we sleep here so no
    caller can accidentally hammer the API. The wait also applies BEFORE the
    first request; a scheduler that restarts the process every few seconds
    would otherwise burst past the limit.

  * The User-Agent is identifiable (project + repo URL). arXiv 429s anonymous
    or default-python-httpx clients on first contact.

  * Retries: 429 (rate-limited) uses a MUCH longer backoff than 5xx / transport
    errors — arXiv 429 windows are measured in minutes. When arXiv returns a
    `Retry-After` header we honor it exactly.

  * `iter_category` never asks arXiv for more entries per page than the caller
    actually wants (a --max-papers 20 request sends max_results=20, not 100).
"""
from __future__ import annotations

import hashlib
import logging
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime

import feedparser
import httpx
from tenacity import RetryCallState, retry, stop_after_attempt

from src.config import settings

log = logging.getLogger(__name__)


# Retry policy — see _should_retry / _wait_policy below.
_MAX_ATTEMPTS = 6
# 5xx / TransportError: quick exponential 2..30s. Network blips clear fast.
_TRANSIENT_WAIT_MIN = 2.0
_TRANSIENT_WAIT_MAX = 30.0
# 429: much longer exponential 60..600s. arXiv rate-limit windows are in minutes.
_RATE_LIMIT_WAIT_MIN = 60.0
_RATE_LIMIT_WAIT_MAX = 600.0


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
    """Sleeps just enough between calls to respect a minimum interval.

    `_last_call` is initialised to "now" (not 0.0), so the very first `.wait()`
    also blocks for up to `interval` seconds. This prevents a burst-on-startup
    when the same process is restarted repeatedly by a scheduler.
    """

    def __init__(self, interval_seconds: float) -> None:
        self._interval = interval_seconds
        self._last_call: float = time.monotonic()

    def wait(self) -> None:
        elapsed = time.monotonic() - self._last_call
        remaining = self._interval - elapsed
        if remaining > 0:
            time.sleep(remaining)
        self._last_call = time.monotonic()


def _should_retry(retry_state: RetryCallState) -> bool:
    """Retry on transient failures only: TransportError, 5xx, 429."""
    if retry_state.outcome is None:
        return False
    exc = retry_state.outcome.exception()
    if exc is None:
        return False
    if isinstance(exc, httpx.TransportError):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        return code == 429 or 500 <= code < 600
    return False


def _wait_policy(retry_state: RetryCallState) -> float:
    """Pick a backoff length based on WHY the previous attempt failed.

      * 429 with Retry-After header  → honor the header exactly.
      * 429 without header           → exponential 60..600s.
      * TransportError / 5xx         → fast exponential 2..30s.
    """
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    attempt = retry_state.attempt_number  # 1-based: 1 = after first failure

    if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 429:
        retry_after = exc.response.headers.get("Retry-After")
        if retry_after:
            try:
                wait = float(retry_after)
                log.warning("arXiv 429: honoring Retry-After=%.1fs", wait)
                return wait
            except ValueError:
                # RFC 7231 also allows an HTTP-date form here; we don't parse
                # it — fall through to the exponential backoff.
                pass
        wait = min(
            _RATE_LIMIT_WAIT_MIN * (2 ** (attempt - 1)), _RATE_LIMIT_WAIT_MAX
        )
        log.warning(
            "arXiv 429 (no Retry-After): backing off %.0fs (attempt %d/%d)",
            wait, attempt, _MAX_ATTEMPTS,
        )
        return wait

    return min(_TRANSIENT_WAIT_MIN * (2 ** (attempt - 1)), _TRANSIENT_WAIT_MAX)


class ArxivClient:
    def __init__(
        self,
        *,
        base_url: str | None = None,
        page_size: int | None = None,
        interval_seconds: float | None = None,
        user_agent: str | None = None,
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
            headers={
                # arXiv's Terms of Use ask for an identifiable UA. Anonymous
                # or default python-httpx UAs get 429'd on first contact.
                "User-Agent": user_agent or settings.ARXIV_USER_AGENT,
                # Explicit Accept keeps arXiv's response routing predictable.
                "Accept": "application/atom+xml",
            },
        )

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> ArxivClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @retry(
        stop=stop_after_attempt(_MAX_ATTEMPTS),
        wait=_wait_policy,
        retry=_should_retry,
        reraise=True,
    )
    def _fetch_page(
        self, *, category: str, start: int, page_size: int
    ) -> feedparser.FeedParserDict:
        self._limiter.wait()
        params = {
            "search_query": f"cat:{category}",
            "start": start,
            "max_results": page_size,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
        resp = self._http.get(self._base_url, params=params)
        # 4xx (except 429) will raise here and NOT be retried — see _should_retry.
        # 5xx and 429 raise here too, but the retry decorator will catch and back off.
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
        empty page. Never requests more per page than we actually still need —
        so `max_results=20` sends `max_results=20` to arXiv, not the default
        page_size of 100.
        """
        emitted = 0
        start = 0
        while True:
            if max_results is not None:
                remaining = max_results - emitted
                if remaining <= 0:
                    return
                this_page = min(self._page_size, remaining)
            else:
                this_page = self._page_size

            feed = self._fetch_page(
                category=category, start=start, page_size=this_page
            )
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
