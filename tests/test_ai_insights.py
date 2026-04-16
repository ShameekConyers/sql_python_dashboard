"""Tests for the AI insight generation pipeline.

Covers context functions, response parsing, DB storage, and the
generate_insight orchestration with mocked LLM calls.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import ai_insights

SCHEMA_SQL: str = (
    ai_insights.PROJECT_ROOT / "sql" / "01_schema.sql"
).read_text()
HACKERNEWS_SCHEMA_SQL: str = (
    ai_insights.PROJECT_ROOT / "sql" / "07_hackernews_schema.sql"
).read_text()
REFERENCE_SCHEMA_SQL: str = (
    ai_insights.PROJECT_ROOT / "sql" / "06_reference_schema.sql"
).read_text()

# ---------------------------------------------------------------------------
# Test data helpers
# ---------------------------------------------------------------------------

SERIES_META: list[tuple[str, str, str, str, str]] = [
    ("UNRATE", "Unemployment Rate", "labor_market", "monthly", "Percent"),
    ("U6RATE", "Underemployment Rate", "labor_market", "monthly", "Percent"),
    ("T10Y2Y", "10Y-2Y Spread", "yield_curve", "daily", "Percent"),
    ("GDPC1", "Real GDP", "output_growth", "quarterly", "Billions"),
    ("CPIAUCSL", "CPI", "prices", "monthly", "Index"),
    ("USINFO", "Info Sector", "ai_labor", "monthly", "Thousands"),
    (
        "CES2023800001",
        "Specialty Trades",
        "ai_labor",
        "monthly",
        "Thousands",
    ),
    ("CNP16OV", "Population 16+", "population", "monthly", "Thousands"),
    ("IPG2211S", "Electric Power", "ai_energy", "monthly", "Index"),
    ("USREC", "Recession Indicator", "recession", "monthly", "Binary"),
]


def _make_test_db() -> sqlite3.Connection:
    """Create an in-memory DB with 24 months of predictable data.

    Returns:
        Connection to the populated in-memory database.
    """
    conn: sqlite3.Connection = sqlite3.connect(":memory:")
    conn.executescript(SCHEMA_SQL)
    conn.executescript(HACKERNEWS_SCHEMA_SQL)
    conn.executescript(REFERENCE_SCHEMA_SQL)

    for meta in SERIES_META:
        conn.execute(
            "INSERT INTO series_metadata "
            "(series_id, name, category, frequency, units) "
            "VALUES (?, ?, ?, ?, ?)",
            meta,
        )

    months: list[str] = [f"2023-{m:02d}-01" for m in range(1, 13)] + [
        f"2024-{m:02d}-01" for m in range(1, 13)
    ]

    # UNRATE: 3.5 → 4.1 over 24 months
    for i, d in enumerate(months):
        val: float = 3.5 + i * 0.025
        conn.execute(
            "INSERT INTO observations "
            "(series_id, date, value, value_covid_adjusted) "
            "VALUES ('UNRATE', ?, ?, ?)",
            (d, val, val),
        )

    # U6RATE: 6.5 → 7.7 over 24 months
    for i, d in enumerate(months):
        val = 6.5 + i * 0.05
        conn.execute(
            "INSERT INTO observations "
            "(series_id, date, value, value_covid_adjusted) "
            "VALUES ('U6RATE', ?, ?, ?)",
            (d, val, val),
        )

    # T10Y2Y: starts positive, goes negative mid-way, recovers
    for i, d in enumerate(months):
        val = 1.0 - i * 0.15 if i < 16 else -1.4 + (i - 16) * 0.3
        conn.execute(
            "INSERT INTO observations "
            "(series_id, date, value, value_covid_adjusted) "
            "VALUES ('T10Y2Y', ?, ?, ?)",
            (d, val, val),
        )

    # GDPC1: quarterly, 8 observations
    quarters: list[str] = [
        "2023-01-01",
        "2023-04-01",
        "2023-07-01",
        "2023-10-01",
        "2024-01-01",
        "2024-04-01",
        "2024-07-01",
        "2024-10-01",
    ]
    gdp_vals: list[float] = [
        20000, 20100, 20250, 20350, 20500, 20600, 20750, 20900,
    ]
    for d, val in zip(quarters, gdp_vals):
        conn.execute(
            "INSERT INTO observations "
            "(series_id, date, value, value_covid_adjusted) "
            "VALUES ('GDPC1', ?, ?, ?)",
            (d, val, val),
        )

    # CPIAUCSL: 300 + 0.5 per month
    for i, d in enumerate(months):
        val = 300.0 + i * 0.5
        conn.execute(
            "INSERT INTO observations "
            "(series_id, date, value, value_covid_adjusted) "
            "VALUES ('CPIAUCSL', ?, ?, ?)",
            (d, val, val),
        )

    # USINFO: declining from 2900 to 2780
    for i, d in enumerate(months):
        val = 2900.0 - i * 5.0
        conn.execute(
            "INSERT INTO observations "
            "(series_id, date, value, value_covid_adjusted) "
            "VALUES ('USINFO', ?, ?, ?)",
            (d, val, val),
        )

    # CES2023800001: rising from 1800 to 1920
    for i, d in enumerate(months):
        val = 1800.0 + i * 5.0
        conn.execute(
            "INSERT INTO observations "
            "(series_id, date, value, value_covid_adjusted) "
            "VALUES ('CES2023800001', ?, ?, ?)",
            (d, val, val),
        )

    # CNP16OV: stable at 265000
    for d in months:
        conn.execute(
            "INSERT INTO observations "
            "(series_id, date, value, value_covid_adjusted) "
            "VALUES ('CNP16OV', ?, 265000, 265000)",
            (d,),
        )

    # IPG2211S: 105 + 0.3 per month
    for i, d in enumerate(months):
        val = 105.0 + i * 0.3
        conn.execute(
            "INSERT INTO observations "
            "(series_id, date, value, value_covid_adjusted) "
            "VALUES ('IPG2211S', ?, ?, ?)",
            (d, val, val),
        )

    # USREC: all 0 (no recession)
    for d in months:
        conn.execute(
            "INSERT INTO observations "
            "(series_id, date, value, value_covid_adjusted) "
            "VALUES ('USREC', ?, 0, 0)",
            (d,),
        )

    # Add Feb 2020 for COVID recovery context
    conn.execute(
        "INSERT INTO observations "
        "(series_id, date, value, value_covid_adjusted) "
        "VALUES ('USINFO', '2020-02-01', 2850, 2850)"
    )
    conn.execute(
        "INSERT INTO observations "
        "(series_id, date, value, value_covid_adjusted) "
        "VALUES ('CES2023800001', '2020-02-01', 1780, 1780)"
    )

    # HN sentiment: seed the same 24 months so the Phase 14 slice has
    # current + prior 12-month windows to aggregate.
    for i, d in enumerate(months):
        sentiment: float = -0.1 if i >= 12 else 0.0
        conn.execute(
            "INSERT INTO hn_sentiment_monthly "
            "(month, mean_sentiment, story_count, layoff_story_count) "
            "VALUES (?, ?, ?, ?)",
            (d, sentiment, 30, 6 if i >= 12 else 2),
        )

    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def test_db() -> sqlite3.Connection:
    """Provide a populated in-memory test database.

    Returns:
        sqlite3.Connection with schema and test data.
    """
    return _make_test_db()


VALID_LLM_RESPONSE: str = json.dumps(
    {
        "narrative": "The unemployment rate has risen steadily.",
        "claims": [
            {
                "description": "UNRATE reached 4.1% in Dec 2024",
                "metric": "UNRATE",
                "value": 4.1,
                "claim_type": "value",
                "aggregation": "latest",
                "period_start": "2024-12",
                "period_end": "2024-12",
                "per_capita": False,
                "use_raw": False,
                "threshold": None,
                "comparison_metric": None,
            },
            {
                "description": "Unemployment trending upward",
                "metric": "UNRATE",
                "value": 1.0,
                "claim_type": "trend",
                "aggregation": "direction",
                "period_start": "2023-01",
                "period_end": "2024-12",
                "per_capita": False,
                "use_raw": False,
                "threshold": None,
                "comparison_metric": None,
            },
        ],
    }
)


# ---------------------------------------------------------------------------
# Context function tests
# ---------------------------------------------------------------------------


class TestContextYieldCurveUnemployment:
    """Tests for the yield curve + unemployment context builder."""

    def test_returns_expected_keys(self, test_db: sqlite3.Connection) -> None:
        """Context dict contains context_text and data_points."""
        result: dict[str, Any] = (
            ai_insights._context_yield_curve_unemployment(test_db)
        )
        assert "context_text" in result
        assert "data_points" in result
        dp: dict[str, Any] = result["data_points"]
        assert "current_spread" in dp
        assert "total_inversion_months" in dp
        assert "longest_inversion" in dp
        assert "current_unrate" in dp

    def test_inversion_detection(
        self, test_db: sqlite3.Connection
    ) -> None:
        """Detects inverted months in the yield curve data."""
        result: dict[str, Any] = (
            ai_insights._context_yield_curve_unemployment(test_db)
        )
        assert result["data_points"]["total_inversion_months"] > 0
        assert result["data_points"]["longest_inversion"] > 0


class TestContextGdpTrend:
    """Tests for the GDP trend context builder."""

    def test_returns_expected_keys(self, test_db: sqlite3.Connection) -> None:
        """Context dict contains the right data_points keys."""
        result: dict[str, Any] = ai_insights._context_gdp_trend(test_db)
        dp: dict[str, Any] = result["data_points"]
        assert "latest_growth" in dp
        assert "avg_growth_5y" in dp
        assert "negative_quarters" in dp
        assert "current_recession" in dp

    def test_no_recession(self, test_db: sqlite3.Connection) -> None:
        """USREC is all zeros, so no recession detected."""
        result: dict[str, Any] = ai_insights._context_gdp_trend(test_db)
        assert result["data_points"]["current_recession"] is False


class TestContextCpiTrend:
    """Tests for the CPI inflation context builder."""

    def test_positive_yoy(self, test_db: sqlite3.Connection) -> None:
        """CPI is rising, so YoY should be positive."""
        result: dict[str, Any] = ai_insights._context_cpi_trend(test_db)
        assert result["data_points"]["latest_yoy"] > 0


class TestContextInfoVsTrades:
    """Tests for the info vs trades per-capita comparison context."""

    def test_divergence_direction(
        self, test_db: sqlite3.Connection
    ) -> None:
        """Trades growing while info declining produces positive divergence."""
        result: dict[str, Any] = ai_insights._context_info_vs_trades(
            test_db
        )
        dp: dict[str, Any] = result["data_points"]
        assert dp["info_change_pct"] < 0
        assert dp["trades_change_pct"] > 0
        assert dp["divergence"] > 0


class TestContextEmploymentGrowth:
    """Tests for the rolling employment growth context builder."""

    def test_returns_both_series(
        self, test_db: sqlite3.Connection
    ) -> None:
        """Data points include both USINFO and CES2023800001."""
        result: dict[str, Any] = ai_insights._context_employment_growth(
            test_db
        )
        dp: dict[str, Any] = result["data_points"]
        assert "USINFO" in dp
        assert "CES2023800001" in dp


class TestContextCovidRecovery:
    """Tests for the COVID recovery context builder."""

    def test_recovery_percentages(
        self, test_db: sqlite3.Connection
    ) -> None:
        """Recovery percentages are computed relative to Feb 2020 peak."""
        result: dict[str, Any] = ai_insights._context_covid_recovery(
            test_db
        )
        dp: dict[str, Any] = result["data_points"]
        assert dp["info_peak"] == 2850
        assert dp["trades_peak"] == 1780
        assert dp["info_recovery_pct"] > 0
        assert dp["trades_recovery_pct"] > 0


class TestContextU6U3Gap:
    """Tests for the U6-U3 gap context builder."""

    def test_gap_positive(self, test_db: sqlite3.Connection) -> None:
        """U6 > U3 so gap should be positive."""
        result: dict[str, Any] = ai_insights._context_u6_u3_gap(test_db)
        assert result["data_points"]["current_gap"] > 0


class TestContextPowerVsInfo:
    """Tests for the power vs info employment context builder."""

    def test_index_values(self, test_db: sqlite3.Connection) -> None:
        """Indexed values should exist and be reasonable."""
        result: dict[str, Any] = ai_insights._context_power_vs_info(
            test_db
        )
        dp: dict[str, Any] = result["data_points"]
        assert dp["power_index_latest"] > 100  # rising series
        assert dp["info_index_latest"] < 100  # declining series


class TestContextSynthesis:
    """Tests for the cross-metric synthesis context builder."""

    def test_returns_cross_metric_data(
        self, test_db: sqlite3.Connection
    ) -> None:
        """Synthesis pulls data from multiple context functions."""
        result: dict[str, Any] = ai_insights._context_synthesis(test_db)
        dp: dict[str, Any] = result["data_points"]
        assert "longest_inversion" in dp
        assert "info_change_pct" in dp
        assert "trades_change_pct" in dp
        assert "power_info_divergence" in dp


# ---------------------------------------------------------------------------
# Response parsing tests
# ---------------------------------------------------------------------------


class TestParseNarrative:
    """Tests for LLM narrative parsing."""

    def test_plain_text(self) -> None:
        """Returns plain text as-is."""
        result: str = ai_insights._parse_narrative(
            "The economy is growing steadily."
        )
        assert result == "The economy is growing steadily."

    def test_strips_code_fences(self) -> None:
        """Strips markdown code fences."""
        result: str = ai_insights._parse_narrative(
            "```\nSome narrative here.\n```"
        )
        assert result == "Some narrative here."

    def test_strips_quotes(self) -> None:
        """Strips wrapping double quotes."""
        result: str = ai_insights._parse_narrative(
            '"The unemployment rate has risen."'
        )
        assert result == "The unemployment rate has risen."

    def test_extracts_from_json(self) -> None:
        """Extracts narrative field if LLM returned JSON anyway."""
        json_resp: str = json.dumps(
            {"narrative": "Found in JSON.", "claims": []}
        )
        result: str = ai_insights._parse_narrative(json_resp)
        assert result == "Found in JSON."

    def test_empty_raises(self) -> None:
        """Raises ValueError for empty input."""
        with pytest.raises(ValueError, match="empty"):
            ai_insights._parse_narrative("   ")


# ---------------------------------------------------------------------------
# DB storage tests
# ---------------------------------------------------------------------------


class TestStoreInsight:
    """Tests for storing insights in the database."""

    def test_insert(self, test_db: sqlite3.Connection) -> None:
        """Stores an insight row in ai_insights."""
        claims: list[dict[str, Any]] = [
            {"description": "test", "metric": "UNRATE", "value": 4.1}
        ]
        ai_insights._store_insight(
            test_db, "TEST_METRIC", "2023-2024", "trend",
            "Test narrative.", claims, [], "llama3.1:8b",
        )
        row = test_db.execute(
            "SELECT metric_key, insight_type, narrative, all_verified, "
            "       citations_json "
            "FROM ai_insights WHERE metric_key = 'TEST_METRIC'"
        ).fetchone()
        assert row is not None
        assert row[0] == "TEST_METRIC"
        assert row[1] == "trend"
        assert row[3] == 0  # all_verified defaults to 0
        assert row[4] == "[]"  # empty citations

    def test_upsert_idempotent(
        self, test_db: sqlite3.Connection
    ) -> None:
        """Re-inserting the same key replaces without duplicates."""
        claims: list[dict[str, Any]] = [
            {"description": "test", "metric": "UNRATE", "value": 4.1}
        ]
        ai_insights._store_insight(
            test_db, "UPSERT_KEY", "2023-2024", "trend",
            "Narrative v1.", claims, [], "llama3.1:8b",
        )
        ai_insights._store_insight(
            test_db, "UPSERT_KEY", "2023-2024", "trend",
            "Narrative v2.", claims, [], "llama3.1:8b",
        )
        count = test_db.execute(
            "SELECT COUNT(*) FROM ai_insights "
            "WHERE metric_key = 'UPSERT_KEY'"
        ).fetchone()[0]
        assert count == 1

    def test_stores_citations_json(
        self, test_db: sqlite3.Connection
    ) -> None:
        """Citations are persisted as a JSON array in citations_json."""
        citations: list[dict[str, Any]] = [
            {
                "ref_id": 12,
                "doc_type": "series_notes",
                "series_id": "UNRATE",
                "title": "FRED Series Notes — UNRATE",
                "source_url": "https://fred.stlouisfed.org/series/UNRATE",
                "excerpt": "The unemployment rate represents...",
            }
        ]
        ai_insights._store_insight(
            test_db, "CIT_KEY", "2023-2024", "trend",
            "Narrative.", [], citations, "llama3.1:8b",
        )
        row = test_db.execute(
            "SELECT citations_json FROM ai_insights "
            "WHERE metric_key = 'CIT_KEY'"
        ).fetchone()
        stored: list[dict[str, Any]] = json.loads(row[0])
        assert stored == citations


# ---------------------------------------------------------------------------
# Generate insight orchestration tests
# ---------------------------------------------------------------------------


class TestGenerateInsight:
    """Tests for the generate_insight function with mocked LLM."""

    @patch(
        "ai_insights._call_llm",
        return_value="The yield curve has been signaling trouble.",
    )
    def test_success(
        self, mock_llm: MagicMock, test_db: sqlite3.Connection
    ) -> None:
        """Generates and stores an insight when LLM returns a narrative."""
        slice_config: dict[str, Any] = {
            "metric_key": "T10Y2Y_UNRATE",
            "insight_type": "correlation",
            "context_fn": ai_insights._context_yield_curve_unemployment,
            "claims_fn": ai_insights._claims_yield_curve_unemployment,
            "analysis_prompt": "Test prompt",
        }
        result: bool = ai_insights.generate_insight(
            test_db, "llama3.1:8b", slice_config, "2023-2024"
        )
        assert result is True

        row = test_db.execute(
            "SELECT metric_key, narrative FROM ai_insights "
            "WHERE metric_key = 'T10Y2Y_UNRATE'"
        ).fetchone()
        assert row is not None
        assert "yield curve" in row[1].lower()

    @patch("ai_insights._call_llm", return_value="   ")
    def test_empty_response_returns_false(
        self, mock_llm: MagicMock, test_db: sqlite3.Connection
    ) -> None:
        """Returns False when LLM returns empty text."""
        slice_config: dict[str, Any] = {
            "metric_key": "BAD_METRIC",
            "insight_type": "trend",
            "context_fn": ai_insights._context_gdp_trend,
            "claims_fn": ai_insights._claims_gdp_trend,
            "analysis_prompt": "Test",
        }
        result: bool = ai_insights.generate_insight(
            test_db, "llama3.1:8b", slice_config, "2023-2024"
        )
        assert result is False


class TestComputeSliceKey:
    """Tests for the slice key computation."""

    def test_returns_year_range(
        self, test_db: sqlite3.Connection
    ) -> None:
        """Slice key spans from min to max year in observations."""
        key: str = ai_insights._compute_slice_key(test_db)
        assert key.startswith("20")
        assert "-" in key


class TestFmtSeries:
    """Tests for the series formatting helper."""

    def test_formats_tail(self) -> None:
        """Formats the tail of a pandas Series as comma-separated string."""
        import pandas as pd

        s: pd.Series = pd.Series([1.0, 2.0, 3.0])
        result: str = ai_insights._fmt_series(s, 1)
        assert "1.0" in result
        assert "3.0" in result


# ---------------------------------------------------------------------------
# Phase 10: narrative validator and unit-aware formatter tests
# ---------------------------------------------------------------------------


class TestValidateNarrative:
    """Tests for the post-generation narrative anti-pattern validator."""

    def test_catches_probability_language(self) -> None:
        """Flags '67% likelihood of a recession' as probability language."""
        warnings: list[str] = ai_insights._validate_narrative(
            "The Scenario Explorer reveals a roughly 67% likelihood of a "
            "recession occurring within the next year.",
            "scenario_explorer",
        )
        assert warnings
        assert any("probability" in w.lower() for w in warnings)

    def test_catches_productivity_terminology(self) -> None:
        """Flags 'decline in productivity' as wrong-terminology usage."""
        warnings: list[str] = ai_insights._validate_narrative(
            "The information sector has experienced a significant decline "
            "in productivity over the past year.",
            "deep_divergence",
        )
        assert warnings
        assert any("employment" in w.lower() for w in warnings)

    def test_catches_causal_verbs(self) -> None:
        """Flags 'AI is displacing workers' as a causal assertion."""
        warnings: list[str] = ai_insights._validate_narrative(
            "AI is displacing workers in the information sector.",
            "IPG2211S_USINFO",
        )
        assert warnings
        assert any(
            "coincides" in w.lower() or "causation" in w.lower()
            for w in warnings
        )

    def test_catches_quarterly_expansion_dollars(self) -> None:
        """Flags dollar-billion 'quarterly expansion' phrasing."""
        warnings: list[str] = ai_insights._validate_narrative(
            "Notably lower than the average quarterly expansion of "
            "$23800B over the past year.",
            "GDPC1",
        )
        assert warnings
        assert any(
            "level" in w.lower() or "expansion" in w.lower()
            for w in warnings
        )

    def test_clean_narrative_returns_empty(self) -> None:
        """A compliant narrative produces no warnings."""
        warnings: list[str] = ai_insights._validate_narrative(
            "Unemployment ticked up to 4.4%, a +0.1pp MoM move from 4.3%. "
            "The risk score reads 0.67, consistent with a cooling labor "
            "market that coincides with the rising yield spread.",
            "overview",
        )
        assert warnings == []


class TestFmtChange:
    """Tests for the unit-aware change formatter."""

    def test_rate_series_reports_pp(self) -> None:
        """Rate-type series (UNRATE) reports percentage points first."""
        result: str = ai_insights._fmt_change("UNRATE", 4.3, 4.4)
        assert "+0.10pp" in result
        assert "report in pp" in result

    def test_level_series_reports_percent(self) -> None:
        """Level-type series (GDPC1) reports relative percent change."""
        result: str = ai_insights._fmt_change("GDPC1", 23000.0, 23500.0)
        # (23500 - 23000) / 23000 = +2.17%
        assert "+2.17%" in result
        assert "report as percent change" in result


class TestSeriesKind:
    """Tests for the SERIES_KIND lookup coverage."""

    def test_covers_all_known_series(self) -> None:
        """Every FRED series in SERIES_META has an entry in SERIES_KIND."""
        known_series: list[str] = [sid for sid, *_ in SERIES_META]
        for series_id in known_series:
            assert series_id in ai_insights.SERIES_KIND, (
                f"{series_id} missing from SERIES_KIND"
            )


# ---------------------------------------------------------------------------
# Phase 11: RAG citation tests
# ---------------------------------------------------------------------------


def _make_chunk(
    doc_id: int,
    series_id: str = "UNRATE",
    doc_type: str = "series_notes",
    title: str = "FRED Series Notes — UNRATE",
    content: str = "The unemployment rate represents the unemployed as a percentage of the labor force.",
    source_url: str | None = "https://fred.stlouisfed.org/series/UNRATE",
) -> Any:
    """Build a RetrievedChunk for tests. Imports locally to avoid top-level."""
    from rag_retrieval import RetrievedChunk

    return RetrievedChunk(
        doc_id=doc_id,
        series_id=series_id,
        doc_type=doc_type,
        title=title,
        content=content,
        source_url=source_url,
        score=0.1,
    )


class TestRule7InSystemPrompt:
    """Tests for the Phase 11 RULE 7 clause in SYSTEM_PROMPT."""

    def test_rule_7_present(self) -> None:
        """SYSTEM_PROMPT contains the RULE 7 header."""
        assert "7. CITATIONS." in ai_insights.SYSTEM_PROMPT

    def test_rule_7_references_ref_tag_format(self) -> None:
        """RULE 7 instructs the LLM to use [ref:N] tags."""
        assert "[ref:N]" in ai_insights.SYSTEM_PROMPT

    def test_rule_7_limits_citations(self) -> None:
        """RULE 7 asks for at most two citations."""
        assert "at most two" in ai_insights.SYSTEM_PROMPT.lower()

    def test_rules_1_through_6_still_present(self) -> None:
        """Phase 10 rules 1-6 are preserved (not rewritten)."""
        for n in range(1, 7):
            assert f"{n}. " in ai_insights.SYSTEM_PROMPT


class TestSeriesHintForMetric:
    """Tests for _series_hint_for_metric."""

    def test_single_series_returns_self(self) -> None:
        """A metric_key that is a FRED series id returns itself."""
        assert ai_insights._series_hint_for_metric("GDPC1") == "GDPC1"

    def test_composite_returns_leading_series(self) -> None:
        """Composite keys resolve to the first series id."""
        assert (
            ai_insights._series_hint_for_metric("T10Y2Y_UNRATE") == "T10Y2Y"
        )
        assert (
            ai_insights._series_hint_for_metric("USINFO_CES2023800001")
            == "USINFO"
        )

    def test_cross_series_returns_none(self) -> None:
        """Cross-series slices skip the filter."""
        for key in (
            "dashboard_intro",
            "recession_risk",
            "synthesis",
            "scenario_explorer",
        ):
            assert ai_insights._series_hint_for_metric(key) is None


class TestBuildReferenceContext:
    """Tests for _build_reference_context."""

    def test_empty_returns_none_marker(self) -> None:
        """Empty retrieval produces a '(none)' placeholder block."""
        block, provided = ai_insights._build_reference_context([])
        assert "REFERENCE CONTEXT: (none)" in block
        assert provided == {}

    def test_emits_ref_tags_and_source_urls(self) -> None:
        """Each unique doc_id gets a [ref:N] entry with its URL."""
        chunks = [
            _make_chunk(doc_id=12, title="FRED Series Notes — UNRATE"),
            _make_chunk(
                doc_id=15,
                doc_type="release_info",
                title="Release — Employment Situation",
                content="Monthly BLS release of payroll data.",
                source_url="https://www.bls.gov/empsit",
            ),
        ]
        block, provided = ai_insights._build_reference_context(chunks)
        assert "[ref:12]" in block
        assert "[ref:15]" in block
        assert "https://fred.stlouisfed.org/series/UNRATE" in block
        assert "https://www.bls.gov/empsit" in block
        assert set(provided.keys()) == {12, 15}

    def test_deduplicates_by_doc_id(self) -> None:
        """Multiple chunks from the same doc_id yield one entry."""
        chunks = [
            _make_chunk(doc_id=12, content="First chunk of the same doc."),
            _make_chunk(doc_id=12, content="Second chunk of the same doc."),
        ]
        block, provided = ai_insights._build_reference_context(chunks)
        # Only one [ref:12] block
        assert block.count("[ref:12]") == 1
        assert set(provided.keys()) == {12}


class TestExtractCitations:
    """Tests for _extract_citations."""

    def test_valid_tag_is_stripped_and_recorded(self) -> None:
        """A valid [ref:N] produces a citation and disappears from prose."""
        provided = {12: _make_chunk(doc_id=12)}
        narrative: str = "Unemployment sits at 4% [ref:12]."

        stripped, records, warnings = ai_insights._extract_citations(
            narrative, provided
        )

        assert "[ref:12]" not in stripped
        assert stripped.endswith(".")
        assert len(records) == 1
        assert records[0]["ref_id"] == 12
        assert records[0]["series_id"] == "UNRATE"
        assert records[0]["source_url"].startswith("https://fred")
        assert warnings == []

    def test_invalid_tag_stays_and_warns(self) -> None:
        """Tags whose id is not in provided stay visible and warn."""
        provided = {12: _make_chunk(doc_id=12)}
        narrative: str = "Unemployment sits at 4% [ref:99]."

        stripped, records, warnings = ai_insights._extract_citations(
            narrative, provided
        )

        assert "[ref:99]" in stripped
        assert records == []
        assert warnings
        assert "hallucinated" in warnings[0].lower()

    def test_duplicate_valid_tags_dedupe(self) -> None:
        """Multiple occurrences of the same valid id produce one record."""
        provided = {12: _make_chunk(doc_id=12)}
        narrative: str = (
            "Unemployment is stable [ref:12]. Methodology is [ref:12] "
            "defined by BLS."
        )

        stripped, records, warnings = ai_insights._extract_citations(
            narrative, provided
        )

        assert "[ref:12]" not in stripped
        assert len(records) == 1
        assert records[0]["ref_id"] == 12

    def test_mixed_valid_and_invalid(self) -> None:
        """Valid tags are stripped; invalid tags survive; records only valid."""
        provided = {12: _make_chunk(doc_id=12)}
        narrative: str = (
            "Unemployment [ref:12] is at 4% [ref:77]."
        )

        stripped, records, warnings = ai_insights._extract_citations(
            narrative, provided
        )

        assert "[ref:12]" not in stripped
        assert "[ref:77]" in stripped
        assert len(records) == 1
        assert records[0]["ref_id"] == 12
        assert warnings

    def test_no_tags_returns_narrative_unchanged(self) -> None:
        """A narrative without [ref:N] tags passes through unchanged."""
        provided = {12: _make_chunk(doc_id=12)}
        narrative: str = "Nothing to cite here."

        stripped, records, warnings = ai_insights._extract_citations(
            narrative, provided
        )

        assert stripped == narrative
        assert records == []
        assert warnings == []

    def test_excerpt_truncated_for_long_content(self) -> None:
        """Long source content is truncated to a reasonable excerpt length."""
        long_content: str = "x" * 800
        provided = {
            42: _make_chunk(doc_id=42, content=long_content)
        }
        narrative: str = "Something [ref:42]."

        _, records, _ = ai_insights._extract_citations(narrative, provided)

        assert len(records) == 1
        assert len(records[0]["excerpt"]) <= 240
        assert records[0]["excerpt"].endswith("...")


class TestValidateNarrativeCitations:
    """Tests for citation-aware extensions to _validate_narrative."""

    def test_merges_citation_warnings(self) -> None:
        """Citation warnings from _extract_citations are merged into output."""
        warnings = ai_insights._validate_narrative(
            "Clean narrative.",
            "UNRATE",
            citation_warnings=[
                "Narrative cites [ref:99] but that id was not in REFERENCE "
                "CONTEXT (hallucinated citation)."
            ],
            retrieval_empty=False,
            citation_count=0,
        )
        assert any("hallucinated" in w.lower() for w in warnings)

    def test_methodology_without_citation_when_refs_present(self) -> None:
        """Methodology language without citations warns when refs were present."""
        warnings = ai_insights._validate_narrative(
            "Unemployment is published by the Bureau of Labor Statistics "
            "each month.",
            "UNRATE",
            retrieval_empty=False,
            citation_count=0,
        )
        assert any("methodology" in w.lower() for w in warnings)

    def test_skips_methodology_check_when_retrieval_empty(self) -> None:
        """If REFERENCE CONTEXT was empty, methodology phrasing is OK."""
        warnings = ai_insights._validate_narrative(
            "Unemployment is published by the Bureau of Labor Statistics.",
            "UNRATE",
            retrieval_empty=True,
            citation_count=0,
        )
        assert not any("methodology" in w.lower() for w in warnings)

    def test_warns_when_too_many_citations(self) -> None:
        """RULE 7 cap: >2 citations triggers a warning."""
        warnings = ai_insights._validate_narrative(
            "Plain narrative.",
            "UNRATE",
            retrieval_empty=False,
            citation_count=4,
        )
        assert any(
            "at most two" in w.lower() or "4 citations" in w.lower()
            for w in warnings
        )

    def test_accepts_two_citations(self) -> None:
        """Exactly two citations passes without warning."""
        warnings = ai_insights._validate_narrative(
            "Plain narrative.",
            "UNRATE",
            retrieval_empty=False,
            citation_count=2,
        )
        assert not any(
            "at most two" in w.lower() or "citations" in w.lower()
            for w in warnings
        )


class TestGenerateInsightRagWiring:
    """Tests for the RAG integration inside generate_insight."""

    @patch("ai_insights._call_llm")
    @patch("ai_insights.rag_retrieval.retrieve")
    def test_reference_context_injected_into_prompt(
        self,
        mock_retrieve: MagicMock,
        mock_llm: MagicMock,
        test_db: sqlite3.Connection,
    ) -> None:
        """generate_insight inserts REFERENCE CONTEXT into the user prompt."""
        mock_retrieve.return_value = [_make_chunk(doc_id=12)]
        mock_llm.return_value = (
            "Unemployment sits at 4% [ref:12], consistent with cooling."
        )

        slice_config: dict[str, Any] = {
            "metric_key": "UNRATE",
            "insight_type": "trend",
            "context_fn": ai_insights._context_gdp_trend,
            "claims_fn": ai_insights._claims_gdp_trend,
            "analysis_prompt": "Test prompt for UNRATE analysis.",
        }

        result: bool = ai_insights.generate_insight(
            test_db, "llama3.1:8b", slice_config, "2023-2024"
        )
        assert result is True

        # Verify the LLM was called with a prompt containing REFERENCE CONTEXT
        prompt_sent: str = mock_llm.call_args[0][1]
        assert "REFERENCE CONTEXT:" in prompt_sent
        assert "[ref:12]" in prompt_sent

        # Verify retrieval was called with the UNRATE series_hint
        retrieve_kwargs = mock_retrieve.call_args.kwargs
        assert retrieve_kwargs["series_hint"] == "UNRATE"

        # The stored narrative has the tag stripped
        row = test_db.execute(
            "SELECT narrative, citations_json FROM ai_insights "
            "WHERE metric_key = 'UNRATE'"
        ).fetchone()
        assert "[ref:12]" not in row[0]
        citations: list[dict[str, Any]] = json.loads(row[1])
        assert len(citations) == 1
        assert citations[0]["ref_id"] == 12

    @patch("ai_insights._call_llm")
    @patch("ai_insights.rag_retrieval.retrieve")
    def test_empty_retrieval_emits_none_marker(
        self,
        mock_retrieve: MagicMock,
        mock_llm: MagicMock,
        test_db: sqlite3.Connection,
    ) -> None:
        """With retrieval empty, the user prompt shows 'REFERENCE CONTEXT: (none)'."""
        mock_retrieve.return_value = []
        mock_llm.return_value = "Plain narrative without any citations."

        slice_config: dict[str, Any] = {
            "metric_key": "GDPC1",
            "insight_type": "trend",
            "context_fn": ai_insights._context_gdp_trend,
            "claims_fn": ai_insights._claims_gdp_trend,
            "analysis_prompt": "GDP analysis.",
        }

        ai_insights.generate_insight(
            test_db, "llama3.1:8b", slice_config, "2023-2024"
        )

        prompt_sent: str = mock_llm.call_args[0][1]
        assert "REFERENCE CONTEXT: (none)" in prompt_sent
        row = test_db.execute(
            "SELECT citations_json FROM ai_insights "
            "WHERE metric_key = 'GDPC1'"
        ).fetchone()
        assert row[0] == "[]"

    @patch("ai_insights._call_llm")
    @patch("ai_insights.rag_retrieval.retrieve")
    def test_cross_series_metric_uses_none_hint(
        self,
        mock_retrieve: MagicMock,
        mock_llm: MagicMock,
        test_db: sqlite3.Connection,
    ) -> None:
        """Cross-series slices pass series_hint=None to retrieval."""
        mock_retrieve.return_value = []
        mock_llm.return_value = "Synthesis narrative."

        slice_config: dict[str, Any] = {
            "metric_key": "synthesis",
            "insight_type": "trend",
            "context_fn": ai_insights._context_synthesis,
            "claims_fn": ai_insights._claims_synthesis,
            "analysis_prompt": "Synthesize everything.",
        }

        ai_insights.generate_insight(
            test_db, "llama3.1:8b", slice_config, "2023-2024"
        )

        assert mock_retrieve.call_args.kwargs["series_hint"] is None


# ---------------------------------------------------------------------------
# HN labor sentiment slice (Phase 14)
# ---------------------------------------------------------------------------


class TestHnLaborSentiment:
    """Tests for the Phase 14 HN labor sentiment slice."""

    def test_composite_metric_hints_routes_hn_to_usinfo(self) -> None:
        """``hn_labor_sentiment`` resolves to ``USINFO`` via hints, not cross-series."""
        assert ai_insights._COMPOSITE_METRIC_HINTS["hn_labor_sentiment"] == "USINFO"
        assert (
            ai_insights._series_hint_for_metric("hn_labor_sentiment") == "USINFO"
        )
        assert "hn_labor_sentiment" not in ai_insights._CROSS_SERIES_METRICS

    def test_hn_labor_sentiment_slice_in_insight_slices(self) -> None:
        """The new slice appears in INSIGHT_SLICES."""
        keys: list[str] = [s["metric_key"] for s in ai_insights.INSIGHT_SLICES]
        assert "hn_labor_sentiment" in keys

    def test_context_hn_labor_sentiment_returns_required_keys(
        self, test_db: sqlite3.Connection
    ) -> None:
        """Context builder returns the expected dict shape."""
        ctx: dict[str, Any] = ai_insights._context_hn_labor_sentiment(test_db)
        assert "context_text" in ctx
        assert "data_points" in ctx
        dp: dict[str, Any] = ctx["data_points"]
        assert dp, "data_points should not be empty with the test fixture"
        for key in (
            "latest_month",
            "window_start",
            "hn_sentiment_current_12m",
            "hn_sentiment_prior_12m",
            "hn_sentiment_change",
            "hn_total_stories",
            "usinfo_change_pct",
            "usinfo_change_raw_pct",
        ):
            assert key in dp, f"Missing data_point key: {key}"

    def test_claims_hn_labor_sentiment_uses_existing_verifier_types(
        self, test_db: sqlite3.Connection
    ) -> None:
        """Every claim uses an aggregation in the existing verifier registry."""
        import verify_insights

        valid_aggs: set[str] = set(verify_insights.VERIFIERS.keys())
        claims: list[dict[str, Any]] = ai_insights._claims_hn_labor_sentiment(
            test_db
        )
        assert len(claims) > 0
        for claim in claims:
            assert claim["aggregation"] in valid_aggs, (
                f"Unknown aggregation type: {claim['aggregation']}"
            )

    def test_usinfo_analysis_prompt_mentions_hacker_news(self) -> None:
        """Four updated AI-labor slices mention Hacker News in their prompt."""
        target_keys: set[str] = {
            "USINFO_CES2023800001",
            "deep_divergence",
            "synthesis",
            "deep_synthesis_charts",
        }
        for s in ai_insights.INSIGHT_SLICES:
            if s["metric_key"] in target_keys:
                assert "Hacker News" in s["analysis_prompt"] or \
                       "Hacker\nNews" in s["analysis_prompt"], (
                    f"Slice {s['metric_key']} missing HN invite in prompt"
                )

    @patch("ai_insights._call_llm")
    @patch("ai_insights.rag_retrieval.retrieve")
    def test_generate_insight_for_hn_labor_sentiment_with_stubs(
        self,
        mock_retrieve: MagicMock,
        mock_llm: MagicMock,
        test_db: sqlite3.Connection,
    ) -> None:
        """End-to-end with stubbed retrieval + LLM produces an HN citation."""
        import rag_retrieval as rr

        # Seed a social reference_docs row that the citation extractor will find.
        test_db.execute(
            "INSERT INTO reference_docs "
            "(id, series_id, doc_type, title, content, source_url, fetched_at) "
            "VALUES (9001, 'USINFO', 'social:hn:99999999', "
            "'AI layoffs accelerate at BigCo', "
            "'AI layoffs accelerate at BigCo', "
            "'https://news.ycombinator.com/item?id=99999999', "
            "'2024-03-01T00:00:00+00:00')"
        )
        test_db.commit()

        fake_chunk = rr.RetrievedChunk(
            doc_id=9001,
            series_id="USINFO",
            doc_type="social:hn:99999999",
            title="AI layoffs accelerate at BigCo",
            content="AI layoffs accelerate at BigCo",
            source_url="https://news.ycombinator.com/item?id=99999999",
            score=0.2,
        )
        mock_retrieve.return_value = [fake_chunk]
        mock_llm.return_value = (
            "HN tech-practitioner sentiment has declined by 0.10 points "
            "over the trailing 12 months, coinciding with a decline in "
            "info-sector employment. This trend tracks alongside broader "
            "AI hiring shifts [ref:9001]."
        )

        slice_config: dict[str, Any] = {
            "metric_key": "hn_labor_sentiment",
            "insight_type": "correlation",
            "context_fn": ai_insights._context_hn_labor_sentiment,
            "claims_fn": ai_insights._claims_hn_labor_sentiment,
            "analysis_prompt": "Describe HN sentiment.",
        }

        ai_insights.generate_insight(
            test_db, "llama3.1:8b", slice_config, "2023-2024"
        )

        row = test_db.execute(
            "SELECT citations_json FROM ai_insights "
            "WHERE metric_key = 'hn_labor_sentiment'"
        ).fetchone()
        assert row is not None
        citations: list[dict[str, Any]] = json.loads(row[0])
        assert len(citations) >= 1
        assert citations[0]["ref_id"] == 9001
