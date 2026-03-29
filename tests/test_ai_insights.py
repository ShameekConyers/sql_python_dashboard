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
            "Test narrative.", claims, "llama3.1:8b",
        )
        row = test_db.execute(
            "SELECT metric_key, insight_type, narrative, all_verified "
            "FROM ai_insights WHERE metric_key = 'TEST_METRIC'"
        ).fetchone()
        assert row is not None
        assert row[0] == "TEST_METRIC"
        assert row[1] == "trend"
        assert row[3] == 0  # all_verified defaults to 0

    def test_upsert_idempotent(
        self, test_db: sqlite3.Connection
    ) -> None:
        """Re-inserting the same key replaces without duplicates."""
        claims: list[dict[str, Any]] = [
            {"description": "test", "metric": "UNRATE", "value": 4.1}
        ]
        ai_insights._store_insight(
            test_db, "UPSERT_KEY", "2023-2024", "trend",
            "Narrative v1.", claims, "llama3.1:8b",
        )
        ai_insights._store_insight(
            test_db, "UPSERT_KEY", "2023-2024", "trend",
            "Narrative v2.", claims, "llama3.1:8b",
        )
        count = test_db.execute(
            "SELECT COUNT(*) FROM ai_insights "
            "WHERE metric_key = 'UPSERT_KEY'"
        ).fetchone()[0]
        assert count == 1


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
