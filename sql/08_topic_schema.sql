-- Phase 15: Topic modeling tables for NMF-derived HN story topics
-- and monthly bigram frequency counts.

CREATE TABLE IF NOT EXISTS hn_topics (
    topic_id    INTEGER PRIMARY KEY,
    label       TEXT NOT NULL,
    top_terms   TEXT NOT NULL,
    story_count INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS hn_topic_assignments (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    story_id    INTEGER NOT NULL,
    topic_id    INTEGER NOT NULL,
    score       REAL NOT NULL,
    UNIQUE (story_id),
    FOREIGN KEY (story_id) REFERENCES hn_stories(story_id),
    FOREIGN KEY (topic_id) REFERENCES hn_topics(topic_id)
);

CREATE INDEX IF NOT EXISTS idx_topic_assignments_topic
    ON hn_topic_assignments(topic_id);

CREATE TABLE IF NOT EXISTS hn_ngram_monthly (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    month   TEXT NOT NULL,
    ngram   TEXT NOT NULL,
    count   INTEGER NOT NULL,
    UNIQUE (month, ngram)
);
