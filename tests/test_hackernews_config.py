"""Tests for src/hackernews_config.py.

Pure-constants validation. Ensures the curated keyword lists and
window anchors stay in the documented shape so downstream code
(``hackernews_pull.py`` and the SQL layoff-count aggregate) can rely
on the invariants.
"""

from __future__ import annotations

from datetime import datetime, timezone

import hackernews_config as cfg


class TestCoverageStart:
    """Tests for the coverage-window anchor constants."""

    def test_unix_corresponds_to_2022_11_01_utc(self) -> None:
        """COVERAGE_START_UNIX decodes to the expected UTC date."""
        dt: datetime = datetime.fromtimestamp(
            cfg.COVERAGE_START_UNIX, tz=timezone.utc
        )
        assert dt.year == 2022
        assert dt.month == 11
        assert dt.day == 1
        assert dt.hour == 0
        assert dt.minute == 0
        assert dt.second == 0

    def test_iso_string_matches(self) -> None:
        """The ISO anchor string matches the unix anchor."""
        assert cfg.COVERAGE_START_ISO == "2022-11-01"


class TestLimits:
    """Tests for the numeric limit constants."""

    def test_monthly_story_limit_is_positive(self) -> None:
        """MONTHLY_STORY_LIMIT > 0 so we actually keep stories."""
        assert cfg.MONTHLY_STORY_LIMIT > 0

    def test_story_text_char_limit_reasonable(self) -> None:
        """Character cap is at least 100 so excerpts carry signal."""
        assert cfg.STORY_TEXT_CHAR_LIMIT >= 100


class TestSearchQueries:
    """Tests for the Algolia query list."""

    def test_non_empty(self) -> None:
        """At least one query is configured."""
        assert len(cfg.SEARCH_QUERIES) > 0

    def test_all_lowercase_ascii(self) -> None:
        """Every query is lowercase ASCII so dedupe + config diffs stay clean."""
        for q in cfg.SEARCH_QUERIES:
            assert q == q.lower()
            assert q.isascii()
            assert q.strip() == q

    def test_no_empty_queries(self) -> None:
        """No query is the empty string."""
        assert all(q for q in cfg.SEARCH_QUERIES)


class TestAlgoliaConstants:
    """Tests for the Algolia endpoint constants."""

    def test_base_url_prefix(self) -> None:
        """ALGOLIA_BASE_URL points at hn.algolia.com."""
        assert cfg.ALGOLIA_BASE_URL.startswith("https://hn.algolia.com")

    def test_hits_per_page_positive(self) -> None:
        """Page size is positive."""
        assert cfg.ALGOLIA_HITS_PER_PAGE > 0

    def test_request_delay_positive(self) -> None:
        """Rate-limit delay is positive so pull_all sleeps between calls."""
        assert cfg.ALGOLIA_REQUEST_DELAY > 0


class TestLayoffKeywords:
    """Tests for the layoff-keyword list used by the monthly aggregate."""

    def test_non_empty(self) -> None:
        """At least one keyword is configured."""
        assert len(cfg.LAYOFF_KEYWORDS) > 0

    def test_all_lowercase_ascii(self) -> None:
        """All keywords are lowercase ASCII (SQLite LIKE is ASCII-case-insensitive)."""
        for kw in cfg.LAYOFF_KEYWORDS:
            assert kw == kw.lower()
            assert kw.isascii()
