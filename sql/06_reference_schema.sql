-- =============================================================================
-- Reference Documents Schema (Phase 11 RAG Citations)
-- =============================================================================
--
-- Authoritative source documents pulled from the FRED API per series:
--   - series_notes   : methodology paragraph from FRED's series metadata
--   - release_info   : parent release name, link, and description
--   - category_path  : flattened category hierarchy for the series
--
-- Dashboard "Show sources" panels cite these rows by id. The vector store
-- (ChromaDB) embeds the same rows for retrieval at generation time.
--
-- Note: the citations_json column on ai_insights is added in 01_schema.sql for
-- fresh builds. For existing pre-Phase-11 databases, db_setup.py's
-- _ensure_citations_column() adds the column via a PRAGMA-guarded ALTER TABLE.
-- =============================================================================

CREATE TABLE IF NOT EXISTS reference_docs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    series_id   TEXT NOT NULL,
    doc_type    TEXT NOT NULL,  -- 'series_notes' | 'release_info' | 'category_path'
    title       TEXT NOT NULL,  -- human-readable label for dashboard display
    content     TEXT NOT NULL,  -- the actual text retrieved by the LLM
    source_url  TEXT,           -- FRED / BLS link, NULL for category_path
    fetched_at  TEXT NOT NULL,  -- ISO timestamp
    FOREIGN KEY (series_id) REFERENCES series_metadata(series_id),
    UNIQUE (series_id, doc_type)
);

CREATE INDEX IF NOT EXISTS idx_reference_docs_series_id
    ON reference_docs(series_id);
