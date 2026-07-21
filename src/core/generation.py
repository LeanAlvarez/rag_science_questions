"""OpenRouter chat completion client with model fallback.

OpenRouter is OpenAI-API-compatible, so the actual HTTP shape is trivial. What
makes this file more than "just call an endpoint" is the resilience story
described in CLAUDE.md requirement #7:

  * The `:free` tier rotates constantly — a model that worked yesterday can
    404 today. We take an ORDERED list of models in config and walk it
    top-to-bottom until one answers.

  * The free tier is rate-limited (~20 req/min, ~50 req/day per account).
    A 429 is NOT "the model is broken", it's "try again later" — so we sleep
    (honouring Retry-After if the server sent it) and retry the SAME model.

  * A non-429 4xx (404 = deprecated, 401 = key issue, 400 = bad request)
    means this model is never going to work: skip to the next one immediately
    instead of burning attempts.

  * 5xx and network errors are transient. Retry per-model with exponential
    backoff, then fall over to the next model.

The last resort — every configured model has failed — raises `NoModelSucceeded`
with the accumulated per-model errors so the caller (the CLI, the web endpoint,
the bot) can surface a real diagnosis instead of a generic "try later".
"""
from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterator
from dataclasses import dataclass
from functools import lru_cache

import httpx

from src.config import settings

log = logging.getLogger(__name__)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MAX_RETRIES_PER_MODEL = 3
BACKOFF_CAP_SECONDS = 30


class ModelFailure(Exception):
    """One model didn't answer after its per-model retries were exhausted."""


class NoModelSucceeded(Exception):
    """Every model in the fallback list failed. Message aggregates all reasons."""


@dataclass(slots=True)
class Completion:
    text: str
    model_used: str


class OpenRouterClient:
    def __init__(
        self,
        *,
        api_key: str,
        models: list[str],
        referer: str = "",
        title: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.2,
    ) -> None:
        if not api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is empty. Get a key at "
                "https://openrouter.ai/keys and set it in .env."
            )
        if not models:
            raise RuntimeError(
                "OPENROUTER_MODELS is empty. Set a comma-separated ordered list "
                "of free-tier model IDs in .env — pick current ones from "
                "https://openrouter.ai/models?max_price=0 (they end in ':free')."
            )
        self._models = models
        self._max_tokens = max_tokens
        self._temperature = temperature

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        # OpenRouter uses these for attribution on their public leaderboard. Optional.
        if referer:
            headers["HTTP-Referer"] = referer
        if title:
            headers["X-Title"] = title

        self._http = httpx.Client(timeout=60.0, headers=headers)

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> OpenRouterClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def complete(self, messages: list[dict]) -> Completion:
        """Try each configured model in order until one answers (non-streaming)."""
        errors: list[str] = []
        for model_id in self._models:
            try:
                text = self._try_one(model_id, messages)
                return Completion(text=text, model_used=model_id)
            except ModelFailure as e:
                log.warning("model %s failed, falling back: %s", model_id, e)
                errors.append(f"{model_id}: {e}")

        raise NoModelSucceeded(
            "All configured OpenRouter models failed:\n  - "
            + "\n  - ".join(errors)
        )

    def complete_stream(self, messages: list[dict]) -> Iterator[tuple[str, str]]:
        """Streaming variant of `complete`. Yields (delta_text, model_used).

        Fallback semantics differ from the non-streaming path: once we have
        started emitting tokens for a model, we can NOT cleanly fall back
        to another one — the client has already begun rendering. So the
        model-fallback loop only runs during the CONNECTION HANDSHAKE
        (before the first token). A mid-stream failure re-raises.
        """
        errors: list[str] = []
        for model_id in self._models:
            yielded = False
            try:
                for delta in self._stream_one(model_id, messages):
                    yielded = True
                    yield delta, model_id
                return  # stream completed cleanly
            except ModelFailure as e:
                if yielded:
                    # Mid-stream failure: bubble up — can't safely fall back.
                    raise
                log.warning("model %s failed, falling back: %s", model_id, e)
                errors.append(f"{model_id}: {e}")

        raise NoModelSucceeded(
            "All configured OpenRouter models failed:\n  - "
            + "\n  - ".join(errors)
        )

    def _try_one(self, model_id: str, messages: list[dict]) -> str:
        body = {
            "model": model_id,
            "messages": messages,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
        }
        for attempt in range(1, MAX_RETRIES_PER_MODEL + 1):
            try:
                resp = self._http.post(OPENROUTER_URL, json=body)
            except httpx.TransportError as e:
                log.info("network error on %s attempt %d: %s", model_id, attempt, e)
                time.sleep(min(2 ** attempt, BACKOFF_CAP_SECONDS))
                continue

            if resp.status_code == 200:
                return self._extract_text(resp.json())

            if resp.status_code == 429:
                # Rate-limited. Same model is fine, just needs to cool off.
                delay = _parse_retry_after(resp) or min(2 ** attempt, BACKOFF_CAP_SECONDS)
                log.info(
                    "rate-limited on %s attempt %d, sleeping %.1fs",
                    model_id, attempt, delay,
                )
                time.sleep(delay)
                continue

            if 500 <= resp.status_code < 600:
                log.info("%d on %s attempt %d", resp.status_code, model_id, attempt)
                time.sleep(min(2 ** attempt, BACKOFF_CAP_SECONDS))
                continue

            # Non-429 4xx (404 deprecated, 401 auth, 400 bad request) — waiting
            # doesn't help. Bubble up so the caller falls over to the next model.
            raise ModelFailure(f"HTTP {resp.status_code}: {resp.text[:400]}")

        raise ModelFailure(f"exhausted {MAX_RETRIES_PER_MODEL} retries")

    @staticmethod
    def _extract_text(payload: dict) -> str:
        try:
            return payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise ModelFailure(f"unexpected response shape: {payload!r}") from e

    def _stream_one(self, model_id: str, messages: list[dict]) -> Iterator[str]:
        """Establish a streaming connection with one model. Yields content deltas.

        The retry loop applies to the HANDSHAKE only — once we hit the 200 path
        and start iterating lines, mid-stream errors bubble up (see the note in
        `complete_stream`).
        """
        body = {
            "model": model_id,
            "messages": messages,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
            "stream": True,
        }
        for attempt in range(1, MAX_RETRIES_PER_MODEL + 1):
            status = 0
            body_text = ""
            try:
                with self._http.stream("POST", OPENROUTER_URL, json=body) as resp:
                    status = resp.status_code
                    if status == 200:
                        yield from self._parse_sse_stream(resp)
                        return
                    # Non-200: read body so we can log a useful diagnosis, then
                    # fall out of the with-block and decide what to do.
                    body_text = resp.read().decode("utf-8", errors="replace")
            except httpx.TransportError as e:
                log.info("network error on %s attempt %d: %s", model_id, attempt, e)
                time.sleep(min(2 ** attempt, BACKOFF_CAP_SECONDS))
                continue

            if status == 429:
                delay = min(2 ** attempt, BACKOFF_CAP_SECONDS)
                log.info(
                    "rate-limited on %s attempt %d, sleeping %.1fs",
                    model_id, attempt, delay,
                )
                time.sleep(delay)
                continue
            if 500 <= status < 600:
                log.info("%d on %s attempt %d", status, model_id, attempt)
                time.sleep(min(2 ** attempt, BACKOFF_CAP_SECONDS))
                continue

            # Non-429 4xx — deprecated model / auth / bad request.
            raise ModelFailure(f"HTTP {status}: {body_text[:400]}")

        raise ModelFailure(f"exhausted {MAX_RETRIES_PER_MODEL} retries")

    @staticmethod
    def _parse_sse_stream(resp: httpx.Response) -> Iterator[str]:
        """Parse an OpenAI-shaped SSE stream and yield the content deltas.

        Events look like:
            data: {"choices":[{"delta":{"content":"Hello"},"index":0}]}
            ...
            data: [DONE]
        Lines that aren't `data:`, plus keepalive comments and heartbeats, are
        skipped. Malformed JSON is skipped (defensive — protects against
        provider-specific noise in the stream).
        """
        for raw in resp.iter_lines():
            line = raw.strip() if isinstance(raw, str) else raw.decode("utf-8", errors="ignore").strip()
            if not line or line.startswith(":"):
                continue
            if not line.startswith("data:"):
                continue
            payload = line[len("data:"):].strip()
            if payload == "[DONE]":
                return
            try:
                obj = json.loads(payload)
                delta = obj["choices"][0]["delta"].get("content")
            except (json.JSONDecodeError, KeyError, IndexError, TypeError):
                continue
            if delta:
                yield delta


def _parse_retry_after(resp: httpx.Response) -> float | None:
    """Return seconds to wait per the Retry-After header, or None if absent/HTTP-date."""
    header = resp.headers.get("Retry-After") or resp.headers.get("retry-after")
    if not header:
        return None
    try:
        return float(header)
    except ValueError:
        # HTTP-date form ("Wed, 21 Oct 2015 07:28:00 GMT"). Rare; skip it.
        return None


@lru_cache
def openrouter_client() -> OpenRouterClient:
    """Process-wide OpenRouter client (loads config once)."""
    return OpenRouterClient(
        api_key=settings.OPENROUTER_API_KEY,
        models=settings.openrouter_models_list,
        referer=settings.OPENROUTER_REFERER,
        title=settings.OPENROUTER_TITLE,
        max_tokens=settings.OPENROUTER_MAX_TOKENS,
        temperature=settings.OPENROUTER_TEMPERATURE,
    )


def close_openrouter_client() -> None:
    """Close the cached client (call before process exit)."""
    if openrouter_client.cache_info().currsize > 0:
        openrouter_client().close()
    openrouter_client.cache_clear()
