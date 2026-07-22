"""Configuration loaded and validated from environment variables.

pydantic-settings gives us:
  * fail-fast on missing/malformed vars at import time (not deep in the pipeline);
  * type coercion (`ARXIV_REQUEST_INTERVAL_SECONDS=3` → float 3.0);
  * one place to see every knob the system reads.

Import `settings` anywhere in the codebase — it's cached (lru_cache) so
reading env vars happens only once per process.
"""
from __future__ import annotations

from functools import lru_cache

from pydantic import computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # tolerate unknown vars in .env instead of crashing
    )

    # --- PostgreSQL ---
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_DB: str
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432

    @computed_field  # type: ignore[misc]
    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    # --- Local ML models ---
    # Defaults are tuned for a RAM-limited VPS (~400 MB total ML footprint).
    # bge-small-en-v1.5:      ~130 MB, 384-d, English-only (matches arXiv corpus).
    # ms-marco-MiniLM-L-6-v2: ~90 MB, English-only, fast on CPU.
    # EMBEDDING_DIMENSION must match the model output AND the VECTOR(N)
    # column in sql/schema.sql — changing this requires a re-index.
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
    EMBEDDING_DIMENSION: int = 384
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    MODEL_DEVICE: str = "cpu"  # cpu | mps | cuda

    # --- arXiv ---
    ARXIV_API_BASE: str = "https://export.arxiv.org/api/query"
    ARXIV_REQUEST_INTERVAL_SECONDS: float = 3.0
    ARXIV_MAX_RESULTS_PER_PAGE: int = 100
    BACKFILL_MAX_PAPERS: int = 500
    # arXiv requires an identifiable User-Agent. Default embeds the public
    # repo URL as contact vector. Override in .env to add a mailto: — e.g.
    # ARXIV_USER_AGENT="arxiv-rag/1.0 (+https://example.com; mailto:you@x.com)"
    ARXIV_USER_AGENT: str = (
        "arxiv-rag/1.0 (+https://github.com/LeanAlvarez/rag_science_questions)"
    )

    # --- Chunking ---
    CHUNK_SIZE_TOKENS: int = 400
    CHUNK_OVERLAP_TOKENS: int = 50

    # --- Retrieval + rerank (unused in Phase 1 but validated up front) ---
    RETRIEVAL_TOP_K_VECTOR: int = 30
    RETRIEVAL_TOP_K_KEYWORD: int = 30
    RETRIEVAL_TOP_K_AFTER_RRF: int = 20
    RERANK_TOP_K: int = 5
    RRF_K: int = 60

    # --- OpenRouter (Phase 3+) ---
    OPENROUTER_API_KEY: str = ""
    OPENROUTER_MODELS: str = ""  # comma-separated ordered fallback list
    OPENROUTER_REFERER: str = ""
    OPENROUTER_TITLE: str = ""
    OPENROUTER_MAX_TOKENS: int = 1024
    OPENROUTER_TEMPERATURE: float = 0.2

    # --- Telegram (Phase 5) ---
    TELEGRAM_BOT_TOKEN: str = ""
    # Comma-separated Telegram user IDs allowed to talk to the bot. Empty =
    # allow anyone (dev / testing). Populate for private use.
    TELEGRAM_ALLOWED_USER_IDS: str = ""

    @property
    def telegram_allowed_user_ids_list(self) -> list[int]:
        """Parsed allowlist. Non-numeric entries are silently dropped."""
        return [
            int(x.strip())
            for x in self.TELEGRAM_ALLOWED_USER_IDS.split(",")
            if x.strip().lstrip("-").isdigit()
        ]

    # --- Web (Phase 4) ---
    WEB_HOST: str = "127.0.0.1"
    WEB_PORT: int = 8000
    # Comma-separated list of origins the FastAPI CORS middleware will allow.
    # In dev, the Vite frontend runs on http://localhost:5173. In prod (single
    # origin) this can stay empty and CORS is effectively off.
    WEB_ALLOWED_ORIGINS: str = "http://localhost:5173,http://127.0.0.1:5173"

    @property
    def web_allowed_origins_list(self) -> list[str]:
        return [o.strip() for o in self.WEB_ALLOWED_ORIGINS.split(",") if o.strip()]

    @property
    def openrouter_models_list(self) -> list[str]:
        """Parsed fallback model list, in priority order."""
        return [m.strip() for m in self.OPENROUTER_MODELS.split(",") if m.strip()]



@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]


settings = get_settings()
