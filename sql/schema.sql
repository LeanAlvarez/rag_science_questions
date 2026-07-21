-- =============================================================================
-- arxiv-rag schema.
--
-- Design principles:
--   * ONE database for everything (chunks, ingestion state, active categories)
--     so writes are transactional across all three concerns.
--   * Categories to track live in a TABLE (active_categories), not in code.
--     Adding a new category is an INSERT, not a redeploy.
--   * Incremental ingestion uses ingested_papers as a state table:
--     content_hash detects paper revisions; last_indexed_at drives dedup.
--   * On re-index, DELETE all chunks for the paper and INSERT the new set in
--     the SAME transaction as the ingested_papers UPDATE. Never inconsistent.
-- =============================================================================

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- -----------------------------------------------------------------------------
-- active_categories: which arXiv categories the ingester tracks.
-- Add a category with:  INSERT INTO active_categories (category) VALUES ('cs.CL');
-- Disable one without deleting history:  UPDATE active_categories SET enabled = FALSE WHERE ...
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS active_categories (
    category     TEXT PRIMARY KEY,          -- arXiv category code, e.g. 'cs.CL'
    display_name TEXT,                       -- optional human label
    added_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    enabled      BOOLEAN     NOT NULL DEFAULT TRUE
);

-- -----------------------------------------------------------------------------
-- ingested_papers: state table for incremental ingestion.
-- content_hash is computed by the ingester (e.g. SHA-256 of title+abstract);
-- if it changes on a later run, the paper is re-chunked and re-embedded.
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ingested_papers (
    arxiv_id         TEXT PRIMARY KEY,       -- canonical id, e.g. '2410.12345v2'
    content_hash     TEXT NOT NULL,
    title            TEXT NOT NULL,
    primary_category TEXT NOT NULL,
    published_at     TIMESTAMPTZ,
    first_seen_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_indexed_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS ingested_papers_category_idx
    ON ingested_papers (primary_category);

CREATE INDEX IF NOT EXISTS ingested_papers_published_idx
    ON ingested_papers (published_at DESC);

-- -----------------------------------------------------------------------------
-- chunks: text fragments + embedding + generated tsvector for hybrid retrieval.
--
-- VECTOR(1024) is chosen to match BAAI/bge-m3. If you switch embedding model,
-- update BOTH this dimension AND the EMBEDDING_DIMENSION env var.
--
-- ON DELETE CASCADE on the FK means a DELETE on ingested_papers wipes all its
-- chunks in one shot — useful when a paper is removed from the corpus.
-- For re-indexing (same paper, new content_hash) the ingester should DELETE
-- WHERE arxiv_id = X and INSERT the new chunks inside a single transaction.
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS chunks (
    id          BIGSERIAL   PRIMARY KEY,
    arxiv_id    TEXT        NOT NULL REFERENCES ingested_papers(arxiv_id) ON DELETE CASCADE,
    chunk_index INT         NOT NULL,
    content     TEXT        NOT NULL,
    embedding   VECTOR(1024) NOT NULL,
    content_tsv TSVECTOR    GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (arxiv_id, chunk_index)
);

-- HNSW index on the vector column: fast approximate nearest neighbour.
-- vector_cosine_ops matches normalized embeddings (bge-m3 outputs are L2-normalized).
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops);

-- GIN index on the generated tsvector powers the keyword side of hybrid retrieval.
CREATE INDEX IF NOT EXISTS chunks_content_tsv_gin
    ON chunks USING gin (content_tsv);

CREATE INDEX IF NOT EXISTS chunks_arxiv_id_idx
    ON chunks (arxiv_id);
