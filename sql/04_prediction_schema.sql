-- =============================================================================
-- Recession Predictions: Model Output Storage
-- =============================================================================
--
-- Stores pre-computed recession probability predictions from the classifier
-- in src/recession_model.py. The dashboard reads from this table directly
-- and never runs the model live.
--
-- One row per month. UPSERT on date so re-running the model overwrites
-- previous predictions without creating duplicates.
-- =============================================================================

CREATE TABLE IF NOT EXISTS recession_predictions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    date            TEXT NOT NULL UNIQUE,            -- ISO date (YYYY-MM-DD), first of month
    probability     REAL NOT NULL,                   -- 0.0 to 1.0 recession probability
    prediction      INTEGER NOT NULL,                -- 0 or 1 binary classification
    actual          INTEGER,                         -- NULL for future months, 0/1 from USREC
    model_name      TEXT NOT NULL,                   -- e.g. 'random_forest', 'logistic_regression'
    features_json   TEXT NOT NULL,                   -- JSON dict of feature values for this month
    generated_at    TEXT NOT NULL                    -- ISO timestamp of generation
);
