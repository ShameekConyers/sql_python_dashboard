"""Tests for the AI insight verification pipeline.

Covers tolerance checking, each verification function (latest, change_pct,
pct_of_start, average, direction, count_months), dispatch logic, and the
full verify_insight orchestration.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

import pytest

import verify_insights

SCHEMA_SQL: str = (
    verify_insights.PROJECT_ROOT / "sql" / "01_schema.sql"
).read_text()

# ---------------------------------------------------------------------------
# Test data helpers
# ---------------------------------------------------------------------------

SERIES_META: list[tuple[str, str, str, str, str]] = [
    ("UNRATE", "Unemployment Rate", "labor_market", "monthly", "Percent"),
    ("U6RATE", "Underemployment Rate", "labor_market", "monthly", "Percent"),
    ("T10Y2Y", "10Y-2Y Spread", "yield_curve", "daily", "Percent"),
    ("USINFO", "Info Sector", "ai_labor", "monthly", "Thousands"),
    (
        "CES2023800001",
        "Specialty Trades",
        "ai_labor",
        "monthly",
        "Thousands",
    ),
    ("CNP16OV", "Population 16+", "population", "monthly", "Thousands"),
]


def _make_test_db() -> sqlite3.Connection:
    """Create an in-memory DB with known values for verification tests.

    Data is deliberately simple and predictable so expected verification
    results can be computed by hand.

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

    # UNRATE: Jan 2024 = 3.5, Jun 2024 = 4.0, Dec 2024 = 4.2
    unrate_vals: list[tuple[str, float]] = [
        ("2024-01-01", 3.5),
        ("2024-02-01", 3.6),
        ("2024-03-01", 3.7),
        ("2024-04-01", 3.8),
        ("2024-05-01", 3.9),
        ("2024-06-01", 4.0),
        ("2024-07-01", 4.0),
        ("2024-08-01", 4.0),
        ("2024-09-01", 4.1),
        ("2024-10-01", 4.1),
        ("2024-11-01", 4.2),
        ("2024-12-01", 4.2),
    ]
    for date, val in unrate_vals:
        conn.execute(
            "INSERT INTO observations "
            "(series_id, date, value, value_covid_adjusted) "
            "VALUES ('UNRATE', ?, ?, ?)",
            (date, val, val),
        )

    # T10Y2Y: months 1-6 positive, months 7-12 negative
    t10y2y_vals: list[tuple[str, float]] = [
        ("2024-01-01", 0.5),
        ("2024-02-01", 0.3),
        ("2024-03-01", 0.1),
        ("2024-04-01", -0.1),
        ("2024-05-01", -0.3),
        ("2024-06-01", -0.5),
        ("2024-07-01", -0.7),
        ("2024-08-01", -0.8),
        ("2024-09-01", -0.6),
        ("2024-10-01", -0.4),
        ("2024-11-01", -0.2),
        ("2024-12-01", 0.1),
    ]
    for date, val in t10y2y_vals:
        conn.execute(
            "INSERT INTO observations "
            "(series_id, date, value, value_covid_adjusted) "
            "VALUES ('T10Y2Y', ?, ?, ?)",
            (date, val, val),
        )

    # USINFO: 2800 in Jan, 2700 in Dec (declining)
    for i in range(12):
        d: str = f"2024-{i + 1:02d}-01"
        val: float = 2800.0 - i * (100.0 / 11)
        conn.execute(
            "INSERT INTO observations "
            "(series_id, date, value, value_covid_adjusted) "
            "VALUES ('USINFO', ?, ?, ?)",
            (d, val, val),
        )

    # CES2023800001: 1800 in Jan, 1900 in Dec (rising)
    for i in range(12):
        d = f"2024-{i + 1:02d}-01"
        val = 1800.0 + i * (100.0 / 11)
        conn.execute(
            "INSERT INTO observations "
            "(series_id, date, value, value_covid_adjusted) "
            "VALUES ('CES2023800001', ?, ?, ?)",
            (d, val, val),
        )

    # CNP16OV: constant 265000 for per-capita normalization
    for i in range(12):
        d = f"2024-{i + 1:02d}-01"
        conn.execute(
            "INSERT INTO observations "
            "(series_id, date, value, value_covid_adjusted) "
            "VALUES ('CNP16OV', ?, 265000, 265000)",
            (d,),
        )

    # U6RATE: Jan 2024 = 7.0, Dec 2024 = 8.0
    for i in range(12):
        d = f"2024-{i + 1:02d}-01"
        val = 7.0 + i * (1.0 / 11)
        conn.execute(
            "INSERT INTO observations "
            "(series_id, date, value, value_covid_adjusted) "
            "VALUES ('U6RATE', ?, ?, ?)",
            (d, val, val),
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
        sqlite3.Connection with schema and known test data.
    """
    return _make_test_db()


# ---------------------------------------------------------------------------
# Tolerance tests
# ---------------------------------------------------------------------------


class TestWithinTolerance:
    """Tests for the within_tolerance function."""

    def test_exact_match(self) -> None:
        """Exact match passes."""
        assert verify_insights.within_tolerance(4.2, 4.2) is True

    def test_within_relative(self) -> None:
        """Value within 5% relative tolerance passes."""
        # 4.2 * 0.05 = 0.21, so 4.0 is within range
        assert verify_insights.within_tolerance(4.2, 4.0) is True

    def test_outside_relative(self) -> None:
        """Value outside 5% relative tolerance fails."""
        assert verify_insights.within_tolerance(4.2, 5.0) is False

    def test_within_absolute(self) -> None:
        """Small value within 0.5 absolute tolerance passes."""
        assert verify_insights.within_tolerance(0.1, 0.3) is True

    def test_zero_expected(self) -> None:
        """Zero expected uses absolute tolerance only."""
        assert verify_insights.within_tolerance(0, 0.4) is True
        assert verify_insights.within_tolerance(0, 0.6) is False

    def test_negative_values(self) -> None:
        """Tolerance works for negative values."""
        assert verify_insights.within_tolerance(-7.2, -7.1) is True
        assert verify_insights.within_tolerance(-7.2, -10.0) is False


# ---------------------------------------------------------------------------
# Verify latest tests
# ---------------------------------------------------------------------------


class TestVerifyLatest:
    """Tests for the _verify_latest function."""

    def test_exact_value(self, test_db: sqlite3.Connection) -> None:
        """Verifies the latest UNRATE value in Dec 2024."""
        claim: dict[str, Any] = {
            "metric": "UNRATE",
            "value": 4.2,
            "period_start": "2024-01",
            "period_end": "2024-12",
        }
        result: dict[str, Any] = verify_insights._verify_latest(
            test_db, claim
        )
        assert result["passed"] is True
        assert result["actual_value"] == 4.2

    def test_no_data(self, test_db: sqlite3.Connection) -> None:
        """Returns failed when no data found in range."""
        claim: dict[str, Any] = {
            "metric": "UNRATE",
            "value": 4.0,
            "period_start": "2025-01",
            "period_end": "2025-12",
        }
        result: dict[str, Any] = verify_insights._verify_latest(
            test_db, claim
        )
        assert result["passed"] is False
        assert result["actual_value"] is None

    def test_per_capita(self, test_db: sqlite3.Connection) -> None:
        """Per-capita normalization divides by CNP16OV * 1000."""
        claim: dict[str, Any] = {
            "metric": "USINFO",
            "value": 2700 / 265000 * 1000,  # expected per-capita
            "period_start": "2024-12",
            "period_end": "2024-12",
            "per_capita": True,
        }
        result: dict[str, Any] = verify_insights._verify_latest(
            test_db, claim
        )
        assert result["passed"] is True


# ---------------------------------------------------------------------------
# Verify change_pct tests
# ---------------------------------------------------------------------------


class TestVerifyChangePct:
    """Tests for the _verify_change_pct function."""

    def test_positive_change(self, test_db: sqlite3.Connection) -> None:
        """Detects positive percentage change for UNRATE."""
        # UNRATE: 3.5 → 4.2, change = (4.2 - 3.5) / 3.5 * 100 = 20.0
        claim: dict[str, Any] = {
            "metric": "UNRATE",
            "value": 20.0,
            "period_start": "2024-01",
            "period_end": "2024-12",
        }
        result: dict[str, Any] = verify_insights._verify_change_pct(
            test_db, claim
        )
        assert result["passed"] is True

    def test_negative_change(self, test_db: sqlite3.Connection) -> None:
        """Detects negative percentage change for USINFO."""
        # USINFO: 2800 → ~2700, change ≈ -3.6%
        claim: dict[str, Any] = {
            "metric": "USINFO",
            "value": -3.6,
            "period_start": "2024-01",
            "period_end": "2024-12",
        }
        result: dict[str, Any] = verify_insights._verify_change_pct(
            test_db, claim
        )
        assert result["passed"] is True

    def test_missing_data(self, test_db: sqlite3.Connection) -> None:
        """Fails when start or end data is missing."""
        claim: dict[str, Any] = {
            "metric": "UNRATE",
            "value": 10.0,
            "period_start": "2020-01",
            "period_end": "2020-12",
        }
        result: dict[str, Any] = verify_insights._verify_change_pct(
            test_db, claim
        )
        assert result["passed"] is False


# ---------------------------------------------------------------------------
# Verify pct_of_start tests
# ---------------------------------------------------------------------------


class TestVerifyPctOfStart:
    """Tests for the _verify_pct_of_start function."""

    def test_ratio_above_100(self, test_db: sqlite3.Connection) -> None:
        """UNRATE rose so end/start > 100%."""
        # UNRATE: 3.5 → 4.2, pct_of_start = 120.0
        claim: dict[str, Any] = {
            "metric": "UNRATE",
            "value": 120.0,
            "period_start": "2024-01",
            "period_end": "2024-12",
        }
        result: dict[str, Any] = verify_insights._verify_pct_of_start(
            test_db, claim
        )
        assert result["passed"] is True

    def test_ratio_below_100(self, test_db: sqlite3.Connection) -> None:
        """USINFO declined so end/start < 100%."""
        # USINFO: 2800 → ~2700, pct ≈ 96.4%
        claim: dict[str, Any] = {
            "metric": "USINFO",
            "value": 96.4,
            "period_start": "2024-01",
            "period_end": "2024-12",
        }
        result: dict[str, Any] = verify_insights._verify_pct_of_start(
            test_db, claim
        )
        assert result["passed"] is True


# ---------------------------------------------------------------------------
# Verify average tests
# ---------------------------------------------------------------------------


class TestVerifyAverage:
    """Tests for the _verify_average function."""

    def test_unrate_average(self, test_db: sqlite3.Connection) -> None:
        """Average UNRATE across 12 months is near 3.9."""
        claim: dict[str, Any] = {
            "metric": "UNRATE",
            "value": 3.9,
            "period_start": "2024-01",
            "period_end": "2024-12",
        }
        result: dict[str, Any] = verify_insights._verify_average(
            test_db, claim
        )
        assert result["passed"] is True

    def test_no_data(self, test_db: sqlite3.Connection) -> None:
        """Returns failed when no observations in range."""
        claim: dict[str, Any] = {
            "metric": "UNRATE",
            "value": 4.0,
            "period_start": "2030-01",
            "period_end": "2030-12",
        }
        result: dict[str, Any] = verify_insights._verify_average(
            test_db, claim
        )
        assert result["passed"] is False


# ---------------------------------------------------------------------------
# Verify direction tests
# ---------------------------------------------------------------------------


class TestVerifyDirection:
    """Tests for the _verify_direction function."""

    def test_increasing(self, test_db: sqlite3.Connection) -> None:
        """UNRATE rose from Jan to Dec, direction = +1."""
        claim: dict[str, Any] = {
            "metric": "UNRATE",
            "value": 1.0,
            "period_start": "2024-01",
            "period_end": "2024-12",
        }
        result: dict[str, Any] = verify_insights._verify_direction(
            test_db, claim
        )
        assert result["passed"] is True
        assert result["actual_value"] == 1.0

    def test_decreasing(self, test_db: sqlite3.Connection) -> None:
        """USINFO declined, direction = -1."""
        claim: dict[str, Any] = {
            "metric": "USINFO",
            "value": -1.0,
            "period_start": "2024-01",
            "period_end": "2024-12",
        }
        result: dict[str, Any] = verify_insights._verify_direction(
            test_db, claim
        )
        assert result["passed"] is True

    def test_mismatch(self, test_db: sqlite3.Connection) -> None:
        """Wrong direction assertion fails."""
        claim: dict[str, Any] = {
            "metric": "UNRATE",
            "value": -1.0,
            "period_start": "2024-01",
            "period_end": "2024-12",
        }
        result: dict[str, Any] = verify_insights._verify_direction(
            test_db, claim
        )
        assert result["passed"] is False


# ---------------------------------------------------------------------------
# Verify count_months tests
# ---------------------------------------------------------------------------


class TestVerifyCountMonths:
    """Tests for the _verify_count_months function."""

    def test_count_months_below(
        self, test_db: sqlite3.Connection
    ) -> None:
        """Counts months where T10Y2Y spread < 0."""
        # T10Y2Y: months 4-11 are negative (8 months)
        claim: dict[str, Any] = {
            "metric": "T10Y2Y",
            "value": 8,
            "aggregation": "count_months_below",
            "threshold": 0,
            "period_start": "2024-01",
            "period_end": "2024-12",
        }
        result: dict[str, Any] = verify_insights._verify_count_months(
            test_db, claim
        )
        assert result["passed"] is True

    def test_count_months_above(
        self, test_db: sqlite3.Connection
    ) -> None:
        """Counts months where UNRATE > 4.0."""
        # UNRATE > 4.0: months 9-12 (4.1, 4.1, 4.2, 4.2) = 4 months
        claim: dict[str, Any] = {
            "metric": "UNRATE",
            "value": 4,
            "aggregation": "count_months_above",
            "threshold": 4.0,
            "period_start": "2024-01",
            "period_end": "2024-12",
        }
        result: dict[str, Any] = verify_insights._verify_count_months(
            test_db, claim
        )
        assert result["passed"] is True

    def test_count_tolerance(
        self, test_db: sqlite3.Connection
    ) -> None:
        """Count passes when within +/- 2 tolerance."""
        claim: dict[str, Any] = {
            "metric": "T10Y2Y",
            "value": 7,  # actual is 8, within tolerance of 2
            "aggregation": "count_months_below",
            "threshold": 0,
            "period_start": "2024-01",
            "period_end": "2024-12",
        }
        result: dict[str, Any] = verify_insights._verify_count_months(
            test_db, claim
        )
        assert result["passed"] is True


# ---------------------------------------------------------------------------
# Verify claim dispatch tests
# ---------------------------------------------------------------------------


class TestVerifyClaim:
    """Tests for the verify_claim dispatch function."""

    def test_dispatches_latest(
        self, test_db: sqlite3.Connection
    ) -> None:
        """Routes 'latest' aggregation to _verify_latest."""
        claim: dict[str, Any] = {
            "metric": "UNRATE",
            "value": 4.2,
            "aggregation": "latest",
            "period_start": "2024-12",
            "period_end": "2024-12",
        }
        result: dict[str, Any] = verify_insights.verify_claim(
            test_db, claim
        )
        assert result["passed"] is True

    def test_unknown_aggregation(
        self, test_db: sqlite3.Connection
    ) -> None:
        """Unknown aggregation type fails gracefully."""
        claim: dict[str, Any] = {
            "metric": "UNRATE",
            "value": 4.2,
            "aggregation": "nonexistent_type",
            "period_start": "2024-12",
            "period_end": "2024-12",
        }
        result: dict[str, Any] = verify_insights.verify_claim(
            test_db, claim
        )
        assert result["passed"] is False
        assert "unknown" in result["reason"]

    def test_defaults_to_latest(
        self, test_db: sqlite3.Connection
    ) -> None:
        """Claims without aggregation key default to latest."""
        claim: dict[str, Any] = {
            "metric": "UNRATE",
            "value": 4.2,
            "period_start": "2024-12",
            "period_end": "2024-12",
        }
        result: dict[str, Any] = verify_insights.verify_claim(
            test_db, claim
        )
        assert result["passed"] is True

    def test_handles_exceptions(
        self, test_db: sqlite3.Connection
    ) -> None:
        """Catches exceptions from verifiers and returns a failure."""
        claim: dict[str, Any] = {
            "metric": "NONEXISTENT",
            "value": 999,
            "aggregation": "change_pct",
            "period_start": "2024-01",
            "period_end": "2024-12",
        }
        result: dict[str, Any] = verify_insights.verify_claim(
            test_db, claim
        )
        assert result["passed"] is False


# ---------------------------------------------------------------------------
# Verify insight (end-to-end) tests
# ---------------------------------------------------------------------------


class TestVerifyInsight:
    """Tests for the verify_insight orchestration function."""

    def test_all_pass(self, test_db: sqlite3.Connection) -> None:
        """All claims verified produces all_verified = True."""
        claims: list[dict[str, Any]] = [
            {
                "description": "UNRATE at 4.2",
                "metric": "UNRATE",
                "value": 4.2,
                "aggregation": "latest",
                "period_start": "2024-12",
                "period_end": "2024-12",
            },
            {
                "description": "UNRATE trending up",
                "metric": "UNRATE",
                "value": 1.0,
                "aggregation": "direction",
                "period_start": "2024-01",
                "period_end": "2024-12",
            },
        ]
        claims_json: str = json.dumps(claims)
        verification, all_ok = verify_insights.verify_insight(
            test_db, 1, claims_json
        )

        assert all_ok is True
        assert verification["0"]["passed"] is True
        assert verification["1"]["passed"] is True
        assert "verified_at" in verification

    def test_one_fails(self, test_db: sqlite3.Connection) -> None:
        """One failed claim makes all_verified = False."""
        claims: list[dict[str, Any]] = [
            {
                "description": "UNRATE at 4.2",
                "metric": "UNRATE",
                "value": 4.2,
                "aggregation": "latest",
                "period_start": "2024-12",
                "period_end": "2024-12",
            },
            {
                "description": "Wrong: UNRATE decreasing",
                "metric": "UNRATE",
                "value": -1.0,
                "aggregation": "direction",
                "period_start": "2024-01",
                "period_end": "2024-12",
            },
        ]
        claims_json: str = json.dumps(claims)
        verification, all_ok = verify_insights.verify_insight(
            test_db, 1, claims_json
        )

        assert all_ok is False
        assert verification["0"]["passed"] is True
        assert verification["1"]["passed"] is False

    def test_empty_claims(self, test_db: sqlite3.Connection) -> None:
        """Empty claims array returns all_verified = True."""
        claims_json: str = "[]"
        verification, all_ok = verify_insights.verify_insight(
            test_db, 1, claims_json
        )
        assert all_ok is True

    def test_verification_json_structure(
        self, test_db: sqlite3.Connection
    ) -> None:
        """Verification JSON has string-indexed results matching claims."""
        claims: list[dict[str, Any]] = [
            {
                "description": "UNRATE average",
                "metric": "UNRATE",
                "value": 3.9,
                "aggregation": "average",
                "period_start": "2024-01",
                "period_end": "2024-12",
            },
        ]
        verification, _ = verify_insights.verify_insight(
            test_db, 1, json.dumps(claims)
        )

        assert "0" in verification
        assert "actual_value" in verification["0"]
        assert "passed" in verification["0"]
        assert "reason" in verification["0"]


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Tests for boundary conditions and unusual inputs."""

    def test_per_capita_change_pct(
        self, test_db: sqlite3.Connection
    ) -> None:
        """Per-capita change_pct works with CNP16OV normalization."""
        # USINFO per-capita: 2800/265000*1000 → 2700/265000*1000
        start_pc: float = 2800 / 265000 * 1000
        end_pc: float = 2700 / 265000 * 1000
        expected_change: float = (end_pc - start_pc) / start_pc * 100

        claim: dict[str, Any] = {
            "metric": "USINFO",
            "value": round(expected_change, 1),
            "aggregation": "change_pct",
            "period_start": "2024-01",
            "period_end": "2024-12",
            "per_capita": True,
        }
        result: dict[str, Any] = verify_insights._verify_change_pct(
            test_db, claim
        )
        assert result["passed"] is True

    def test_use_raw_flag(self, test_db: sqlite3.Connection) -> None:
        """use_raw=True reads from value column instead of adjusted."""
        claim: dict[str, Any] = {
            "metric": "UNRATE",
            "value": 4.2,
            "aggregation": "latest",
            "period_start": "2024-12",
            "period_end": "2024-12",
            "use_raw": True,
        }
        result: dict[str, Any] = verify_insights._verify_latest(
            test_db, claim
        )
        # In test data, value == value_covid_adjusted
        assert result["passed"] is True

    def test_boundary_value_start(
        self, test_db: sqlite3.Connection
    ) -> None:
        """_get_boundary_value with position='start' gets earliest."""
        val: float | None = verify_insights._get_boundary_value(
            test_db, "UNRATE", "2024-01", "2024-12", "start"
        )
        assert val == pytest.approx(3.5)

    def test_boundary_value_end(
        self, test_db: sqlite3.Connection
    ) -> None:
        """_get_boundary_value with position='end' gets latest."""
        val: float | None = verify_insights._get_boundary_value(
            test_db, "UNRATE", "2024-01", "2024-12", "end"
        )
        assert val == pytest.approx(4.2)

    def test_boundary_value_no_data(
        self, test_db: sqlite3.Connection
    ) -> None:
        """_get_boundary_value returns None for empty range."""
        val: float | None = verify_insights._get_boundary_value(
            test_db, "UNRATE", "2030-01", "2030-12", "end"
        )
        assert val is None


# ---------------------------------------------------------------------------
# Phase 11: citation verification
# ---------------------------------------------------------------------------


REFERENCE_SCHEMA_SQL: str = (
    verify_insights.PROJECT_ROOT / "sql" / "06_reference_schema.sql"
).read_text()


@pytest.fixture()
def db_with_refs() -> sqlite3.Connection:
    """Return an in-memory DB with reference_docs populated."""
    conn: sqlite3.Connection = sqlite3.connect(":memory:")
    conn.executescript(SCHEMA_SQL)
    conn.executescript(REFERENCE_SCHEMA_SQL)
    # One minimal series_metadata row for FK integrity
    conn.execute(
        "INSERT INTO series_metadata "
        "(series_id, name, category, frequency, units) "
        "VALUES (?, ?, ?, ?, ?)",
        ("UNRATE", "Unemployment Rate", "labor_market", "monthly", "Percent"),
    )
    conn.execute(
        "INSERT INTO reference_docs "
        "(id, series_id, doc_type, title, content, source_url, fetched_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            12,
            "UNRATE",
            "series_notes",
            "FRED Series Notes — UNRATE",
            (
                "The unemployment rate represents the number of unemployed "
                "as a percentage of the labor force. Labor force data are "
                "restricted to people 16 years of age and older."
            ),
            "https://fred.stlouisfed.org/series/UNRATE",
            "2026-04-16T00:00:00+00:00",
        ),
    )
    conn.commit()
    return conn


class TestVerifyCitations:
    """Tests for _verify_citations."""

    def test_valid_ref_and_excerpt(
        self, db_with_refs: sqlite3.Connection
    ) -> None:
        """Existing ref with matching excerpt passes both flags."""
        citations: list[dict[str, Any]] = [
            {
                "ref_id": 12,
                "doc_type": "series_notes",
                "series_id": "UNRATE",
                "title": "FRED Series Notes — UNRATE",
                "source_url": "https://fred.stlouisfed.org/series/UNRATE",
                "excerpt": "The unemployment rate represents the number of unemployed...",
            }
        ]
        results = verify_insights._verify_citations(db_with_refs, citations)
        assert len(results) == 1
        assert results[0]["ref_exists"] is True
        assert results[0]["excerpt_matches"] is True

    def test_missing_ref(self, db_with_refs: sqlite3.Connection) -> None:
        """A ref_id that does not exist in reference_docs fails both."""
        citations: list[dict[str, Any]] = [
            {
                "ref_id": 999,
                "title": "Fake Ref",
                "excerpt": "irrelevant",
            }
        ]
        results = verify_insights._verify_citations(db_with_refs, citations)
        assert len(results) == 1
        assert results[0]["ref_exists"] is False
        assert results[0]["excerpt_matches"] is False

    def test_fabricated_excerpt(
        self, db_with_refs: sqlite3.Connection
    ) -> None:
        """Real ref_id but invented excerpt passes exists, fails match."""
        citations: list[dict[str, Any]] = [
            {
                "ref_id": 12,
                "title": "FRED Series Notes — UNRATE",
                "excerpt": "This excerpt was invented by the LLM entirely.",
            }
        ]
        results = verify_insights._verify_citations(db_with_refs, citations)
        assert results[0]["ref_exists"] is True
        assert results[0]["excerpt_matches"] is False

    def test_excerpt_matches_is_case_insensitive(
        self, db_with_refs: sqlite3.Connection
    ) -> None:
        """Case differences in the excerpt do not defeat the match."""
        citations: list[dict[str, Any]] = [
            {
                "ref_id": 12,
                "excerpt": "THE UNEMPLOYMENT RATE REPRESENTS THE NUMBER OF UNEMPLOYED",
            }
        ]
        results = verify_insights._verify_citations(db_with_refs, citations)
        assert results[0]["excerpt_matches"] is True

    def test_empty_citations_returns_empty(
        self, db_with_refs: sqlite3.Connection
    ) -> None:
        """An empty citations list returns an empty verification list."""
        assert verify_insights._verify_citations(db_with_refs, []) == []

    def test_preserves_original_keys(
        self, db_with_refs: sqlite3.Connection
    ) -> None:
        """Original citation fields are preserved in the augmented result."""
        citations: list[dict[str, Any]] = [
            {
                "ref_id": 12,
                "title": "FRED Series Notes — UNRATE",
                "excerpt": "The unemployment rate represents the number...",
                "source_url": "https://fred.stlouisfed.org/series/UNRATE",
            }
        ]
        results = verify_insights._verify_citations(db_with_refs, citations)
        assert results[0]["title"] == "FRED Series Notes — UNRATE"
        assert (
            results[0]["source_url"] == "https://fred.stlouisfed.org/series/UNRATE"
        )


class TestVerifyInsightWithCitations:
    """Tests for verify_insight when citations_json is included."""

    def test_verification_json_carries_citations_block(
        self, db_with_refs: sqlite3.Connection
    ) -> None:
        """verify_insight adds a 'citations' sub-array to verification_json."""
        citations_json: str = json.dumps(
            [
                {
                    "ref_id": 12,
                    "title": "FRED Series Notes — UNRATE",
                    "excerpt": "The unemployment rate represents the number...",
                }
            ]
        )

        verification, all_verified = verify_insights.verify_insight(
            db_with_refs, row_id=1, claims_json="[]", citations_json=citations_json
        )

        assert "citations" in verification
        assert len(verification["citations"]) == 1
        assert verification["citations"][0]["ref_exists"] is True
        # Citation failures do NOT flip all_verified; with no claims it stays True
        assert all_verified is True

    def test_citation_failure_does_not_change_all_verified(
        self, db_with_refs: sqlite3.Connection
    ) -> None:
        """A missing ref_id must not flip the main verification banner."""
        citations_json: str = json.dumps(
            [{"ref_id": 9999, "excerpt": "fake"}]
        )

        verification, all_verified = verify_insights.verify_insight(
            db_with_refs, row_id=1, claims_json="[]", citations_json=citations_json
        )

        assert all_verified is True
        assert verification["citations"][0]["ref_exists"] is False

    def test_default_citations_json_is_empty_list(
        self, db_with_refs: sqlite3.Connection
    ) -> None:
        """Callers that omit citations_json still get a 'citations' key."""
        verification, _ = verify_insights.verify_insight(
            db_with_refs, row_id=1, claims_json="[]"
        )
        assert verification["citations"] == []

    def test_malformed_citations_json_is_tolerated(
        self, db_with_refs: sqlite3.Connection
    ) -> None:
        """Bad JSON in citations_json does not break verification."""
        verification, _ = verify_insights.verify_insight(
            db_with_refs, row_id=1, claims_json="[]", citations_json="not-json"
        )
        assert verification["citations"] == []


# ---------------------------------------------------------------------------
# HN reference citation verification (Phase 14)
# ---------------------------------------------------------------------------


class TestVerifyHnCitation:
    """Social reference_docs rows pass through the same path as FRED refs."""

    def test_verifies_hn_ref_id_through_reference_docs(
        self, db_with_refs: sqlite3.Connection
    ) -> None:
        """An HN social reference_docs row verifies through _verify_citations."""
        # Seed a USINFO metadata row so the FK allows inserting under that series.
        db_with_refs.execute(
            "INSERT OR IGNORE INTO series_metadata "
            "(series_id, name, category, frequency, units) "
            "VALUES (?, ?, ?, ?, ?)",
            ("USINFO", "Info Sector", "ai_labor", "monthly", "Thousands"),
        )
        db_with_refs.execute(
            "INSERT INTO reference_docs "
            "(id, series_id, doc_type, title, content, source_url, fetched_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                5001,
                "USINFO",
                "social:hn:8888888",
                "Tech layoffs at BigCo",
                "Tech layoffs at BigCo: details in the filing.",
                "https://news.ycombinator.com/item?id=8888888",
                "2024-06-01T00:00:00+00:00",
            ),
        )
        db_with_refs.commit()

        citations: list[dict[str, Any]] = [
            {
                "ref_id": 5001,
                "doc_type": "social:hn:8888888",
                "series_id": "USINFO",
                "title": "Tech layoffs at BigCo",
                "source_url": "https://news.ycombinator.com/item?id=8888888",
                "excerpt": "Tech layoffs at BigCo",
            }
        ]

        results = verify_insights._verify_citations(db_with_refs, citations)
        assert len(results) == 1
        assert results[0]["ref_exists"] is True
        assert results[0]["excerpt_matches"] is True
