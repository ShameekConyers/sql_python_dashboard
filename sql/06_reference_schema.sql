-- =============================================================================
-- Reference Documents Schema (Phase 11 RAG Citations, extended in Phase 12)
-- =============================================================================
--
-- Authoritative source documents pulled from the FRED API per series:
--   - series_notes   : methodology paragraph from FRED's series metadata
--   - release_info   : parent release name, link, and description
--   - category_path  : flattened category hierarchy for the series
--
-- Phase 12 added curated US federal-government publications (BLS, BEA, EIA,
-- CRS, CBO, CEA, Treasury, GAO — all public domain under 17 USC 105). To
-- support multiple scholarly sources per series without changing the UNIQUE
-- (series_id, doc_type) constraint, scholarly rows encode their slug in the
-- doc_type value:
--   - doc_type = 'scholarly:<slug>'   e.g. 'scholarly:cea_erp_labor_market'
-- Downstream code that wants to filter to scholarly rows only treats any
-- doc_type starting with 'scholarly' (or 'scholarly:') as scholarly; the
-- embedder and retriever are doc_type-agnostic.
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
    doc_type    TEXT NOT NULL,  -- 'series_notes' | 'release_info' | 'category_path' | 'scholarly:<slug>'
    title       TEXT NOT NULL,  -- human-readable label for dashboard display
    content     TEXT NOT NULL,  -- the actual text retrieved by the LLM
    source_url  TEXT,           -- FRED / BLS / federal-agency link, NULL permitted
    fetched_at  TEXT NOT NULL,  -- ISO timestamp
    FOREIGN KEY (series_id) REFERENCES series_metadata(series_id),
    UNIQUE (series_id, doc_type)
);

CREATE INDEX IF NOT EXISTS idx_reference_docs_series_id
    ON reference_docs(series_id);
