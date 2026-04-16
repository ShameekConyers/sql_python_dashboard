-- =============================================================================
-- Macro Economic Dashboard: Star Schema
-- =============================================================================
--
-- Star schema with two dimension tables and one fact table, plus a cached
-- narratives table for batch-generated AI insights.
--
-- Design rationale:
--   - series_metadata (dimension): one row per FRED series. Stores identity,
--     grouping category, and ingestion metadata. Category lives here as a
--     column rather than a separate table because we have only 6 flat
--     categories across 10 series with no hierarchy or category-level
--     attributes worth normalizing.
--
--   - observations (fact): the core analytical table. Each row is one
--     data point for one series on one date. The unique constraint on
--     (series_id, date) enforces idempotent loads so re-running ingestion
--     never creates duplicates.
--
--   - ai_insights (cached narratives): stores batch-generated LLM insights
--     with structured claims and programmatic verification results. The
--     dashboard reads from this table directly and never calls the LLM live.
-- =============================================================================

-- Dimension: one row per FRED series
CREATE TABLE IF NOT EXISTS series_metadata (
    series_id               TEXT PRIMARY KEY,       -- FRED series ID (e.g. 'UNRATE')
    name                    TEXT NOT NULL,           -- human-readable name
    category                TEXT NOT NULL,           -- grouping key (e.g. 'labor_market', 'yield_curve')
    frequency               TEXT NOT NULL,           -- 'daily', 'monthly', or 'quarterly'
    units                   TEXT,                    -- unit label from FRED (e.g. 'Percent', 'Thousands')
    seasonal_adjustment     TEXT,                    -- e.g. 'Seasonally Adjusted', 'Not Seasonally Adjusted'
    last_updated            TEXT                     -- ISO timestamp of last FRED update
);

-- Fact: one row per (series, date) observation
CREATE TABLE IF NOT EXISTS observations (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    series_id               TEXT NOT NULL,
    date                    TEXT NOT NULL,            -- ISO date string (YYYY-MM-DD)
    value                   REAL NOT NULL,            -- observed numeric value (raw from FRED)
    value_covid_adjusted    REAL NOT NULL,            -- ARIMA counterfactual for COVID window,
                                                     -- raw value elsewhere. See covid_adjustment.py
    FOREIGN KEY (series_id) REFERENCES series_metadata(series_id),
    UNIQUE (series_id, date)
);

-- Index for the most common access pattern: filtering observations by series
CREATE INDEX IF NOT EXISTS idx_observations_series_id ON observations(series_id);

-- Index for time-range queries across all series
CREATE INDEX IF NOT EXISTS idx_observations_date ON observations(date);

-- Cached AI narratives: batch-generated insights with verification
CREATE TABLE IF NOT EXISTS ai_insights (
    id                  INTEGER PRIMARY KEY,
    metric_key          TEXT NOT NULL,               -- which series or metric this insight covers
    slice_key           TEXT NOT NULL,               -- time period or segment (e.g. '2020-2024')
    insight_type        TEXT NOT NULL,               -- 'trend', 'correlation', 'anomaly', 'comparison'
    narrative           TEXT NOT NULL,               -- human-readable insight paragraph
    claims_json         TEXT NOT NULL,               -- JSON array of structured, testable claims
    verification_json   TEXT NOT NULL,               -- JSON object with verification results per claim
    all_verified        BOOLEAN NOT NULL,            -- 1 if every claim passed verification, 0 otherwise
    model_used          TEXT NOT NULL,               -- e.g. 'claude-haiku-4-5-20251001'
    generated_at        TIMESTAMP NOT NULL,          -- when the insight was generated
    citations_json      TEXT NOT NULL DEFAULT '[]',  -- JSON array of RAG citation records (Phase 11)
    UNIQUE (metric_key, slice_key, insight_type)
);
