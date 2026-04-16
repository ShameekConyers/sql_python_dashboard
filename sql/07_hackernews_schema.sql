-- =============================================================================
-- Hacker News Stories + Monthly Sentiment Schema (Phase 13)
-- =============================================================================
--
-- Two tables supporting the HN ingestion + sentiment pipeline:
--
--   - hn_stories (fact): one row per HN story id. Thinned -- story text is
--     truncated to STORY_TEXT_CHAR_LIMIT characters and normalized to ASCII
--     during ingestion. Full story bodies live only in the gitignored raw/
--     cache.
--
--   - hn_sentiment_monthly (aggregate): one row per YYYY-MM month for which
--     at least one story exists. Computed from hn_stories via SQL GROUP BY
--     during db_setup. Nothing in Phase 13 consumes this table; Phase 14
--     adds classifier features and a dashboard chart that read from it.
--
-- Why a single-channel schema (not per-topic like the planned Reddit subs):
--   - HN is a single site -- no natural per-channel axis.
--   - Phase 14's classifier wants one volume feature, one sentiment feature,
--     one layoff-frequency feature. Flat corpus matches this exactly.
--
-- Month format: 'YYYY-MM-01' ISO date string. Same convention as
-- observations.date, so joining on month is straightforward.
-- =============================================================================

CREATE TABLE IF NOT EXISTS hn_stories (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    story_id            INTEGER NOT NULL,    -- HN item ID (Algolia objectID)
    created_utc         TEXT NOT NULL,        -- ISO8601 UTC string
    month               TEXT NOT NULL,        -- 'YYYY-MM-01' bucket at load
    title               TEXT NOT NULL,        -- full title, ASCII-normalized
    text_excerpt        TEXT NOT NULL,        -- <=500 chars, ASCII-normalized;
                                              --   empty string for link posts
    score               INTEGER NOT NULL,     -- Algolia 'points' at pull time
    num_comments        INTEGER NOT NULL,     -- Algolia 'num_comments'
    url                 TEXT,                 -- external URL; NULL for self-posts
    hn_permalink        TEXT NOT NULL,        -- news.ycombinator.com item link
    sentiment_score     REAL NOT NULL,        -- compound in [-1, 1]
    sentiment_label     TEXT NOT NULL,        -- 'positive'|'neutral'|'negative'
    UNIQUE (story_id)
);

CREATE INDEX IF NOT EXISTS idx_hn_stories_month ON hn_stories(month);

CREATE TABLE IF NOT EXISTS hn_sentiment_monthly (
    month                  TEXT PRIMARY KEY,  -- 'YYYY-MM-01'
    mean_sentiment         REAL NOT NULL,     -- unweighted mean of sentiment_score
    story_count            INTEGER NOT NULL,  -- stories in the bucket
    layoff_story_count     INTEGER NOT NULL   -- stories matching LAYOFF_KEYWORDS
);
