"""Tests for the NMF topic model pipeline (src/topic_model.py)."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

import topic_model


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SQL_DIR: Path = Path(__file__).resolve().parent.parent / "sql"


def _create_test_db(conn: sqlite3.Connection, stories: list[dict]) -> None:
    """Create the HN schema and insert test stories.

    Args:
        conn: In-memory SQLite connection.
        stories: List of dicts with story_id, title, text_excerpt, month.
    """
    # Minimal hn_stories schema
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS hn_stories (
            story_id INTEGER PRIMARY KEY,
            created_utc TEXT,
            month TEXT,
            title TEXT,
            text_excerpt TEXT,
            score INTEGER DEFAULT 1,
            num_comments INTEGER DEFAULT 0,
            url TEXT,
            hn_permalink TEXT,
            sentiment_score REAL DEFAULT 0.0,
            sentiment_label TEXT DEFAULT 'neutral'
        );
    """)
    # Topic schema
    schema_path: Path = SQL_DIR / "08_topic_schema.sql"
    conn.executescript(schema_path.read_text())

    for s in stories:
        conn.execute(
            """
            INSERT INTO hn_stories (story_id, created_utc, month, title,
                text_excerpt, score, num_comments, hn_permalink,
                sentiment_score, sentiment_label)
            VALUES (?, ?, ?, ?, ?, 10, 5, 'https://hn.example.com', 0.0, 'neutral')
            """,
            (s["story_id"], f"{s['month'][:7]}-15T00:00:00", s["month"],
             s["title"], s.get("text_excerpt", "")),
        )
    conn.commit()


@pytest.fixture()
def test_db() -> sqlite3.Connection:
    """Create an in-memory DB with sample HN stories for topic modeling.

    Returns:
        Open SQLite connection with test data.
    """
    conn: sqlite3.Connection = sqlite3.connect(":memory:")
    stories: list[dict] = [
        # AI / automation cluster
        {"story_id": 1, "title": "AI will replace software engineers",
         "text_excerpt": "artificial intelligence automation jobs", "month": "2023-01-01"},
        {"story_id": 2, "title": "CEO says AI replaces coders",
         "text_excerpt": "ai replace programming developers", "month": "2023-02-01"},
        {"story_id": 3, "title": "AI tools replacing data scientists",
         "text_excerpt": "machine learning automation data jobs", "month": "2023-03-01"},
        # Layoff cluster
        {"story_id": 4, "title": "Tech layoffs hit 50000 workers",
         "text_excerpt": "mass layoffs technology companies workers", "month": "2023-01-01"},
        {"story_id": 5, "title": "Microsoft layoffs 10000 employees",
         "text_excerpt": "microsoft mass layoffs cuts workers", "month": "2023-02-01"},
        {"story_id": 6, "title": "Meta layoffs continue into 2023",
         "text_excerpt": "meta facebook layoffs employees workers", "month": "2023-03-01"},
        # Hiring freeze cluster
        {"story_id": 7, "title": "Amazon hiring freeze extends",
         "text_excerpt": "hiring freeze amazon extended", "month": "2023-01-01"},
        {"story_id": 8, "title": "Google hiring freeze impacts teams",
         "text_excerpt": "hiring freeze google teams impact", "month": "2023-02-01"},
        {"story_id": 9, "title": "Hiring freezes across tech industry",
         "text_excerpt": "hiring freeze tech companies industry", "month": "2023-03-01"},
        # Career / job search cluster
        {"story_id": 10, "title": "Software engineer job market 2023",
         "text_excerpt": "software engineer job market career", "month": "2023-01-01"},
        {"story_id": 11, "title": "Job market tough for new grads",
         "text_excerpt": "job market graduates career software", "month": "2023-02-01"},
        {"story_id": 12, "title": "Career advice for software engineers",
         "text_excerpt": "career advice software engineer tips", "month": "2023-03-01"},
    ]
    _create_test_db(conn, stories)
    return conn


# ---------------------------------------------------------------------------
# Tests: internal helpers
# ---------------------------------------------------------------------------


def test_load_documents(test_db: sqlite3.Connection) -> None:
    """_load_documents returns all stories as (id, text) tuples."""
    docs: list[tuple[int, str]] = topic_model._load_documents(test_db)
    assert len(docs) == 12
    assert all(isinstance(d[0], int) for d in docs)
    assert all(isinstance(d[1], str) and len(d[1]) > 0 for d in docs)


def test_build_tfidf() -> None:
    """_build_tfidf returns a sparse matrix and a fitted vectorizer."""
    texts: list[str] = [
        "software engineer job market",
        "ai replace programmer automation",
        "layoffs tech workers industry",
    ]
    matrix, vectorizer = topic_model._build_tfidf(texts, max_features=100, min_df=1, max_df=0.9)
    assert matrix.shape[0] == 3
    assert matrix.shape[1] > 0
    assert len(vectorizer.get_feature_names_out()) == matrix.shape[1]


def test_fit_nmf() -> None:
    """_fit_nmf returns a fitted NMF model with the right number of components."""
    texts: list[str] = [
        "software engineer job",
        "ai replace automation",
        "layoffs workers tech",
        "hiring freeze companies",
    ]
    matrix, _ = topic_model._build_tfidf(texts, max_features=100, min_df=1, max_df=0.9)
    model = topic_model._fit_nmf(matrix, n_topics=2)
    assert model.components_.shape[0] == 2


def test_extract_topic_labels() -> None:
    """_extract_topic_labels returns labels and top terms for each topic."""
    texts: list[str] = [
        "software engineer job market",
        "ai replace automation coder",
        "layoffs workers mass tech",
        "hiring freeze companies extended",
    ]
    matrix, vectorizer = topic_model._build_tfidf(texts, max_features=100, min_df=1, max_df=0.9)
    model = topic_model._fit_nmf(matrix, n_topics=2)
    labels: list[tuple[str, list[str]]] = topic_model._extract_topic_labels(model, vectorizer)
    assert len(labels) == 2
    for label, terms in labels:
        assert isinstance(label, str)
        assert len(terms) == 5  # default n_top_terms


def test_assign_topics() -> None:
    """_assign_topics assigns each doc to exactly one topic."""
    texts: list[str] = [
        "software engineer job",
        "ai replace automation",
    ]
    ids: list[int] = [1, 2]
    matrix, _ = topic_model._build_tfidf(texts, max_features=100, min_df=1, max_df=0.9)
    model = topic_model._fit_nmf(matrix, n_topics=2)
    assignments: list[tuple[int, int, float]] = topic_model._assign_topics(model, matrix, ids)
    assert len(assignments) == 2
    for story_id, topic_id, score in assignments:
        assert story_id in ids
        assert 0 <= topic_id < 2
        assert score >= 0.0


def test_write_results(test_db: sqlite3.Connection) -> None:
    """_write_results populates hn_topics and hn_topic_assignments."""
    topics: list[tuple[str, list[str]]] = [
        ("Topic A", ["term1", "term2", "term3"]),
        ("Topic B", ["term4", "term5", "term6"]),
    ]
    assignments: list[tuple[int, int, float]] = [
        (1, 0, 0.5), (2, 0, 0.3), (3, 1, 0.8),
        (4, 1, 0.6), (5, 0, 0.4), (6, 1, 0.7),
        (7, 0, 0.2), (8, 1, 0.5), (9, 0, 0.3),
        (10, 1, 0.4), (11, 0, 0.6), (12, 1, 0.9),
    ]
    topic_model._write_results(test_db, topics, assignments)

    topic_rows: list[tuple] = test_db.execute(
        "SELECT topic_id, label, top_terms, story_count FROM hn_topics ORDER BY topic_id"
    ).fetchall()
    assert len(topic_rows) == 2
    assert topic_rows[0][1] == "Topic A"
    assert json.loads(topic_rows[0][2]) == ["term1", "term2", "term3"]
    assert topic_rows[0][3] == 6  # stories assigned to topic 0

    assign_count: int = test_db.execute(
        "SELECT COUNT(*) FROM hn_topic_assignments"
    ).fetchone()[0]
    assert assign_count == 12


def test_compute_ngrams(test_db: sqlite3.Connection) -> None:
    """_compute_ngrams writes bigram counts per month."""
    count: int = topic_model._compute_ngrams(test_db, top_n=5)
    assert count > 0

    rows: list[tuple] = test_db.execute(
        "SELECT month, ngram, count FROM hn_ngram_monthly ORDER BY month, count DESC"
    ).fetchall()
    assert len(rows) > 0
    # Every row should have a bigram (two words)
    for _, ngram, cnt in rows:
        assert " " in ngram
        assert cnt >= 1


def test_compute_ngrams_idempotent(test_db: sqlite3.Connection) -> None:
    """Running _compute_ngrams twice produces the same result."""
    count1: int = topic_model._compute_ngrams(test_db, top_n=5)
    count2: int = topic_model._compute_ngrams(test_db, top_n=5)
    assert count1 == count2


# ---------------------------------------------------------------------------
# Tests: full pipeline
# ---------------------------------------------------------------------------


def test_fit_topics_end_to_end(tmp_path: Path) -> None:
    """fit_topics runs the full pipeline and populates all three tables."""
    db_path: Path = tmp_path / "test.db"
    conn: sqlite3.Connection = sqlite3.connect(db_path)

    stories: list[dict] = [
        {"story_id": i, "title": f"AI layoffs software engineer job {i}",
         "text_excerpt": f"tech workers hiring freeze market {i}",
         "month": f"2023-{(i % 3) + 1:02d}-01"}
        for i in range(1, 21)
    ]
    _create_test_db(conn, stories)
    conn.close()

    topic_model.fit_topics(db_path, n_topics=3, min_df=1)

    conn = sqlite3.connect(db_path)
    topic_count: int = conn.execute("SELECT COUNT(*) FROM hn_topics").fetchone()[0]
    assert topic_count == 3

    assign_count: int = conn.execute("SELECT COUNT(*) FROM hn_topic_assignments").fetchone()[0]
    assert assign_count == 20

    ngram_count: int = conn.execute("SELECT COUNT(*) FROM hn_ngram_monthly").fetchone()[0]
    assert ngram_count > 0

    conn.close()


def test_fit_topics_empty_db(tmp_path: Path) -> None:
    """fit_topics handles an empty hn_stories table gracefully."""
    db_path: Path = tmp_path / "empty.db"
    conn: sqlite3.Connection = sqlite3.connect(db_path)
    _create_test_db(conn, [])
    conn.close()

    # Should not raise
    topic_model.fit_topics(db_path, n_topics=3, min_df=1)

    conn = sqlite3.connect(db_path)
    topic_count: int = conn.execute("SELECT COUNT(*) FROM hn_topics").fetchone()[0]
    assert topic_count == 0
    conn.close()
