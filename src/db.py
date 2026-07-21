"""PostgreSQL access layer.

Uses psycopg3 with a connection pool. Registers the pgvector adapter on every
pooled connection so Python lists / numpy arrays flow into VECTOR columns
without manual `str(...)` formatting.

Two public context managers:
  * `get_connection()` — borrow a pooled connection, no automatic transaction.
  * `transaction()`    — borrow a connection AND wrap the block in a single
                          DB transaction (commit on clean exit, rollback on
                          any exception). This is the primitive used for the
                          re-index upsert.
"""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from pgvector.psycopg import register_vector
from psycopg import Connection
from psycopg_pool import ConnectionPool

from src.config import settings


def _configure(conn: Connection) -> None:
    """Runs once per pooled connection when it's first opened."""
    register_vector(conn)


_pool: ConnectionPool | None = None


def get_pool() -> ConnectionPool:
    """Lazily initialize the process-wide connection pool."""
    global _pool
    if _pool is None:
        _pool = ConnectionPool(
            conninfo=settings.postgres_dsn,
            min_size=1,
            max_size=5,
            configure=_configure,
            open=True,
        )
    return _pool


@contextmanager
def get_connection() -> Iterator[Connection]:
    """Borrow a connection from the pool. Autocommit stays False."""
    with get_pool().connection() as conn:
        yield conn


@contextmanager
def transaction() -> Iterator[Connection]:
    """Borrow a connection AND run the block inside a single DB transaction.

    Any exception raised inside the `with` block triggers a ROLLBACK; a clean
    exit COMMITs. This is exactly the semantics the re-index upsert needs:
    DELETE chunks + UPSERT paper + INSERT chunks all-or-nothing.
    """
    with get_pool().connection() as conn, conn.transaction():
        yield conn


def close_pool() -> None:
    """Close the pool. Call before process exit (e.g. from a CLI's finally)."""
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None
