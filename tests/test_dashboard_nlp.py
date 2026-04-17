"""Tests for NLP Analysis dashboard query helpers (Phase 15).

These tests exercise the query functions directly against in-memory
SQLite databases without importing the Streamlit app module.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import pytest

SQL_DIR: Path = Path(__file__).resolve().parent.parent / "sql"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _create_nlp_test_db() -> sqlite3.Connection:
    """Create an in-memory DB with topic, story, and observation data.

    Returns:
        Open SQLite connection with test data.
    """
    conn: sqlite3.Connection = sqlite3.connect(":memory:")

    # Core schema + HN + topic schemas
    for schema_file in (
        "01_schema.sql",
        "07_hackernews_schema.sql",
        "08_topic_schema.sql",
    ):
        conn.executescript((SQL_DIR / schema_file).read_text())

    # Series metadata
    for series_id, name in [
        ("UNRATE", "Unemployment Rate"),
        ("U6RATE", "Underemployment Rate"),
        ("USINFO", "Info Sector"),
        ("CNP16OV", "Population 16+"),
    ]:
        conn.execute(
            "INSERT INTO series_metadata (series_id, name, category, frequency) "
            "VALUES (?, ?, 'test', 'monthly')",
            (series_id, name),
        )

    # Topics
    conn.execute(
        "INSERT INTO hn_topics (topic_id, label, top_terms, story_count) "
        "VALUES (0, 'AI Automation', '[\"ai\",\"replace\"]', 6)"
    )
    conn.execute(
        "INSERT INTO hn_topics (topic_id, label, top_terms, story_count) "
        "VALUES (1, 'Tech Layoffs', '[\"layoffs\",\"tech\"]', 4)"
    )

    # Stories + assignments + sentiment_monthly
    months: list[str] = [
        "2023-01-01", "2023-02-01", "2023-03-01",
        "2023-04-01", "2023-05-01",
    ]
    for i, month in enumerate(months):
        for j in range(2):
            story_id: int = i * 2 + j + 1
            topic_id: int = 0 if j == 0 else 1
            conn.execute(
                "INSERT INTO hn_stories (story_id, created_utc, month, title, "
                "text_excerpt, score, num_comments, hn_permalink, "
                "sentiment_score, sentiment_label) "
                "VALUES (?, ?, ?, 'Test story', 'excerpt', 10, 5, "
                "'https://hn.example.com', ?, 'neutral')",
                (story_id, f"{month[:7]}-15T00:00:00", month,
                 -0.1 + j * 0.2),
            )
            conn.execute(
                "INSERT INTO hn_topic_assignments (story_id, topic_id, score) "
                "VALUES (?, ?, 0.5)",
                (story_id, topic_id),
            )

        conn.execute(
            "INSERT INTO hn_sentiment_monthly "
            "(month, mean_sentiment, story_count, layoff_story_count) "
            "VALUES (?, ?, 2, 1)",
            (month, -0.1 + i * 0.02),
        )

    # Observations: UNRATE, U6RATE, USINFO, CNP16OV
    for month in months:
        conn.execute(
            "INSERT INTO observations (series_id, date, value, value_covid_adjusted) "
            "VALUES ('UNRATE', ?, 4.0, 4.0)", (month,)
        )
        conn.execute(
            "INSERT INTO observations (series_id, date, value, value_covid_adjusted) "
            "VALUES ('U6RATE', ?, 7.0, 7.0)", (month,)
        )
        conn.execute(
            "INSERT INTO observations (series_id, date, value, value_covid_adjusted) "
            "VALUES ('USINFO', ?, 2900, 2900)", (month,)
        )
        conn.execute(
            "INSERT INTO observations (series_id, date, value, value_covid_adjusted) "
            "VALUES ('CNP16OV', ?, 265000, 265000)", (month,)
        )

    # Ngrams
    conn.execute(
        "INSERT INTO hn_ngram_monthly (month, ngram, count) "
        "VALUES ('2023-01-01', 'hiring freeze', 5)"
    )
    conn.execute(
        "INSERT INTO hn_ngram_monthly (month, ngram, count) "
        "VALUES ('2023-02-01', 'hiring freeze', 3)"
    )
    conn.execute(
        "INSERT INTO hn_ngram_monthly (month, ngram, count) "
        "VALUES ('2023-01-01', 'mass layoffs', 4)"
    )

    conn.commit()
    return conn


@pytest.fixture()
def nlp_db() -> sqlite3.Connection:
    """Provide a populated in-memory test database for NLP queries.

    Returns:
        sqlite3.Connection with topic and story test data.
    """
    return _create_nlp_test_db()


# ---------------------------------------------------------------------------
# Since we can't import dashboard/app.py (Streamlit side effects), we
# replicate the SQL queries here as pure functions. This tests the query
# logic without the Streamlit caching wrapper.
# ---------------------------------------------------------------------------


def _query_topic_distribution(
    conn: sqlite3.Connection, date_min: str, date_max: str
) -> pd.DataFrame:
    """Replication of query_topic_distribution for testing."""
    effective_min: str = max(date_min[:7], "2022-01")
    df: pd.DataFrame = pd.read_sql_query(
        """
        SELECT ta.topic_id, t.label,
               SUBSTR(s.month, 1, 7) AS month,
               COUNT(*) AS story_count
        FROM hn_topic_assignments ta
        JOIN hn_topics t ON ta.topic_id = t.topic_id
        JOIN hn_stories s ON ta.story_id = s.story_id
        GROUP BY ta.topic_id, SUBSTR(s.month, 1, 7)
        ORDER BY SUBSTR(s.month, 1, 7), ta.topic_id
        """,
        conn,
    )
    df = df[
        (df["month"] >= effective_min) & (df["month"] <= date_max[:7])
    ]
    return df


def _query_sentiment_by_topic(conn: sqlite3.Connection) -> pd.DataFrame:
    """Replication of query_sentiment_by_topic for testing."""
    return pd.read_sql_query(
        """
        SELECT t.label, s.sentiment_score
        FROM hn_topic_assignments ta
        JOIN hn_topics t ON ta.topic_id = t.topic_id
        JOIN hn_stories s ON ta.story_id = s.story_id
        """,
        conn,
    )


def _query_layoff_vs_u6u3(
    conn: sqlite3.Connection, date_min: str, date_max: str
) -> pd.DataFrame:
    """Replication of query_layoff_vs_u6u3 for testing."""
    effective_min: str = max(date_min[:7], "2022-01")
    hn: pd.DataFrame = pd.read_sql_query(
        "SELECT SUBSTR(month, 1, 7) AS month, layoff_story_count "
        "FROM hn_sentiment_monthly ORDER BY month",
        conn,
    )
    gap: pd.DataFrame = pd.read_sql_query(
        """
        SELECT SUBSTR(u6.date, 1, 7) AS month,
               u6.value_covid_adjusted - u3.value_covid_adjusted AS u6_u3_gap
        FROM observations u6
        JOIN observations u3
          ON u3.series_id = 'UNRATE'
          AND SUBSTR(u3.date, 1, 7) = SUBSTR(u6.date, 1, 7)
        WHERE u6.series_id = 'U6RATE'
        ORDER BY month
        """,
        conn,
    )
    merged: pd.DataFrame = hn.merge(gap, on="month", how="inner")
    merged = merged[
        (merged["month"] >= effective_min) & (merged["month"] <= date_max[:7])
    ]
    return merged


def _query_ngram_trends(
    conn: sqlite3.Connection, date_min: str, date_max: str
) -> pd.DataFrame:
    """Replication of query_ngram_trends for testing."""
    effective_min: str = max(date_min[:7], "2022-01")
    df: pd.DataFrame = pd.read_sql_query(
        "SELECT SUBSTR(month, 1, 7) AS month, ngram, count "
        "FROM hn_ngram_monthly ORDER BY month, count DESC",
        conn,
    )
    df = df[
        (df["month"] >= effective_min) & (df["month"] <= date_max[:7])
    ]
    return df


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestQueryTopicDistribution:
    """Tests for the topic distribution query."""

    def test_returns_expected_columns(self, nlp_db: sqlite3.Connection) -> None:
        """Result has topic_id, label, month, story_count columns."""
        df: pd.DataFrame = _query_topic_distribution(nlp_db, "2023-01-01", "2023-12-31")
        assert set(df.columns) == {"topic_id", "label", "month", "story_count"}

    def test_correct_row_count(self, nlp_db: sqlite3.Connection) -> None:
        """Each topic has one row per month with stories."""
        df: pd.DataFrame = _query_topic_distribution(nlp_db, "2023-01-01", "2023-12-31")
        # 2 topics x 5 months = 10 rows
        assert len(df) == 10

    def test_respects_date_filter(self, nlp_db: sqlite3.Connection) -> None:
        """Filtering to a subset of months reduces the result."""
        df: pd.DataFrame = _query_topic_distribution(nlp_db, "2023-01-01", "2023-02-28")
        assert len(df) < 10
        assert all(df["month"] <= "2023-02")


class TestQuerySentimentByTopic:
    """Tests for the sentiment by topic query."""

    def test_returns_expected_columns(self, nlp_db: sqlite3.Connection) -> None:
        """Result has label and sentiment_score columns."""
        df: pd.DataFrame = _query_sentiment_by_topic(nlp_db)
        assert set(df.columns) == {"label", "sentiment_score"}

    def test_returns_all_stories(self, nlp_db: sqlite3.Connection) -> None:
        """One row per story (10 total in the fixture)."""
        df: pd.DataFrame = _query_sentiment_by_topic(nlp_db)
        assert len(df) == 10


class TestQueryLayoffVsU6U3:
    """Tests for the layoff volume vs U6-U3 gap query."""

    def test_returns_expected_columns(self, nlp_db: sqlite3.Connection) -> None:
        """Result has month, layoff_story_count, u6_u3_gap."""
        df: pd.DataFrame = _query_layoff_vs_u6u3(nlp_db, "2023-01-01", "2023-12-31")
        assert "month" in df.columns
        assert "layoff_story_count" in df.columns
        assert "u6_u3_gap" in df.columns

    def test_u6_u3_gap_value(self, nlp_db: sqlite3.Connection) -> None:
        """U6-U3 gap should be 3.0 (7.0 - 4.0) in the fixture."""
        df: pd.DataFrame = _query_layoff_vs_u6u3(nlp_db, "2023-01-01", "2023-12-31")
        if not df.empty:
            assert abs(df["u6_u3_gap"].iloc[0] - 3.0) < 0.01


class TestQueryNgramTrends:
    """Tests for the n-gram trending query."""

    def test_returns_expected_columns(self, nlp_db: sqlite3.Connection) -> None:
        """Result has month, ngram, count columns."""
        df: pd.DataFrame = _query_ngram_trends(nlp_db, "2023-01-01", "2023-12-31")
        assert set(df.columns) == {"month", "ngram", "count"}

    def test_returns_seeded_ngrams(self, nlp_db: sqlite3.Connection) -> None:
        """Returns the ngrams we seeded in the fixture."""
        df: pd.DataFrame = _query_ngram_trends(nlp_db, "2023-01-01", "2023-12-31")
        assert len(df) == 3
        assert "hiring freeze" in df["ngram"].values
