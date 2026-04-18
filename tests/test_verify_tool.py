"""Tests for src/agent/tools/verify_tool.py.

Covers claim parsing, individual verification, aggregate status logic,
integration with mocked LLM responses, and regression for the batch
verify_insights.verify_claim interface.
"""

from __future__ import annotations

import dataclasses
import json
import sqlite3
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage

from src.agent.tools.verify_tool import (
    Claim,
    ClaimResult,
    VerificationResult,
    _claim_to_verifier_dict,
    parse_agent_response,
    verify_agent_claim,
    verify_all_claims,
)
from src.verify_insights import verify_claim


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def verify_db(tmp_path: pytest.TempPathFactory) -> str:
    """Create a small SQLite DB with UNRATE data for verification tests.

    Returns:
        Path to the temporary database file.
    """
    db_path = str(tmp_path / "verify_test.db")
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE series_metadata "
        "(series_id TEXT PRIMARY KEY, name TEXT, category TEXT, "
        "units TEXT, frequency TEXT)"
    )
    conn.execute(
        "INSERT INTO series_metadata VALUES "
        "('UNRATE', 'Unemployment Rate', 'Labor', 'Percent', 'Monthly')"
    )
    conn.execute(
        "CREATE TABLE observations "
        "(id INTEGER PRIMARY KEY, series_id TEXT, date TEXT, "
        "value REAL, value_covid_adjusted REAL)"
    )
    conn.executemany(
        "INSERT INTO observations (series_id, date, value, "
        "value_covid_adjusted) VALUES (?, ?, ?, ?)",
        [
            ("UNRATE", "2025-10-01", 4.0, 4.0),
            ("UNRATE", "2025-11-01", 4.1, 4.1),
            ("UNRATE", "2025-12-01", 4.2, 4.2),
            ("UNRATE", "2026-01-01", 4.3, 4.3),
            ("UNRATE", "2026-02-01", 4.4, 4.4),
        ],
    )
    conn.execute(
        "CREATE TABLE recession_predictions "
        "(id INTEGER PRIMARY KEY, month TEXT, lr_probability REAL, "
        "rf_probability REAL, lr_prediction INTEGER, "
        "rf_prediction INTEGER, actual_recession INTEGER, "
        "model_name TEXT, features_json TEXT, "
        "date TEXT, probability REAL, prediction INTEGER)"
    )
    conn.executemany(
        "INSERT INTO recession_predictions "
        "(month, date, probability, prediction, model_name, features_json) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        [
            ("2025-12", "2025-12-01", 0.15, 0, "lr", "{}"),
            ("2026-01", "2026-01-01", 0.20, 0, "lr", "{}"),
            ("2026-02", "2026-02-01", 0.25, 0, "lr", "{}"),
        ],
    )
    conn.commit()
    conn.close()
    return db_path


# ---------------------------------------------------------------------------
# Parsing tests (~8)
# ---------------------------------------------------------------------------


class TestParseAgentResponse:
    """Tests for parse_agent_response JSON parsing."""

    def test_valid_json_with_claims(self) -> None:
        """Valid JSON with narrative + claims returns correct Claim list."""
        raw = json.dumps({
            "narrative": "Unemployment is at 4.4%.",
            "claims": [
                {
                    "statement": "unemployment at 4.4%",
                    "metric_type": "latest",
                    "series_id": "UNRATE",
                    "expected_value": 4.4,
                    "date_range": ["2026-02", "2026-02"],
                }
            ],
        })
        narrative, claims = parse_agent_response(raw)
        assert narrative == "Unemployment is at 4.4%."
        assert len(claims) == 1
        assert claims[0].metric_type == "latest"
        assert claims[0].series_id == "UNRATE"
        assert claims[0].expected_value == 4.4
        assert claims[0].date_range == ("2026-02", "2026-02")

    def test_valid_json_empty_claims(self) -> None:
        """Valid JSON with empty claims array returns empty list."""
        raw = json.dumps({
            "narrative": "This is a conceptual answer.",
            "claims": [],
        })
        narrative, claims = parse_agent_response(raw)
        assert narrative == "This is a conceptual answer."
        assert claims == []

    def test_plain_text_refusal(self) -> None:
        """Plain text (off-topic refusal) returns raw text, empty list."""
        raw = "I can only answer questions about the economic data."
        narrative, claims = parse_agent_response(raw)
        assert narrative == raw
        assert claims == []

    def test_malformed_json(self) -> None:
        """Malformed JSON (missing closing brace) returns raw text."""
        raw = '{"narrative": "test", "claims": ['
        narrative, claims = parse_agent_response(raw)
        assert narrative == raw
        assert claims == []

    def test_extra_fields_ignored(self) -> None:
        """Extra fields in JSON are ignored without error."""
        raw = json.dumps({
            "narrative": "Answer here.",
            "claims": [],
            "extra_field": "should be ignored",
            "another": 42,
        })
        narrative, claims = parse_agent_response(raw)
        assert narrative == "Answer here."
        assert claims == []

    def test_missing_metric_type_skipped(self) -> None:
        """A claim missing metric_type is skipped gracefully."""
        raw = json.dumps({
            "narrative": "Some answer.",
            "claims": [
                {
                    "statement": "incomplete claim",
                    "series_id": "UNRATE",
                    "expected_value": 4.4,
                },
                {
                    "statement": "complete claim",
                    "metric_type": "latest",
                    "series_id": "UNRATE",
                    "expected_value": 4.4,
                },
            ],
        })
        narrative, claims = parse_agent_response(raw)
        assert len(claims) == 1
        assert claims[0].statement == "complete claim"

    def test_missing_date_range_is_none(self) -> None:
        """A claim without date_range gets date_range=None."""
        raw = json.dumps({
            "narrative": "Answer.",
            "claims": [
                {
                    "statement": "prediction latest",
                    "metric_type": "prediction_latest",
                    "series_id": "recession",
                    "expected_value": 0.25,
                }
            ],
        })
        _, claims = parse_agent_response(raw)
        assert len(claims) == 1
        assert claims[0].date_range is None

    def test_markdown_json_fences(self) -> None:
        """JSON wrapped in markdown fences is parsed correctly."""
        inner = json.dumps({
            "narrative": "Fenced answer.",
            "claims": [
                {
                    "statement": "test",
                    "metric_type": "latest",
                    "series_id": "UNRATE",
                    "expected_value": 4.4,
                    "date_range": ["2026-02", "2026-02"],
                }
            ],
        })
        raw = f"```json\n{inner}\n```"
        narrative, claims = parse_agent_response(raw)
        assert narrative == "Fenced answer."
        assert len(claims) == 1


# ---------------------------------------------------------------------------
# Verification tests (~10)
# ---------------------------------------------------------------------------


class TestVerifyAgentClaim:
    """Tests for verify_agent_claim against an in-memory-style SQLite DB."""

    def test_unrate_latest_passes(self, verify_db: str) -> None:
        """UNRATE latest value within tolerance passes."""
        claim = Claim(
            statement="unemployment at 4.4%",
            metric_type="latest",
            series_id="UNRATE",
            expected_value=4.4,
            date_range=("2026-02", "2026-02"),
        )
        result = verify_agent_claim(claim, verify_db)
        assert result.passed is True
        assert result.actual_value == 4.4

    def test_value_outside_tolerance_fails(self, verify_db: str) -> None:
        """Value outside 5% relative AND 0.5 absolute tolerance fails."""
        claim = Claim(
            statement="unemployment at 5.5%",
            metric_type="latest",
            series_id="UNRATE",
            expected_value=5.5,
            date_range=("2026-02", "2026-02"),
        )
        result = verify_agent_claim(claim, verify_db)
        assert result.passed is False

    def test_within_absolute_tolerance_passes(self, verify_db: str) -> None:
        """Value within absolute tolerance (0.5) but outside relative passes."""
        # Actual is 4.4, claiming 4.8 => diff=0.4 < 0.5 absolute
        claim = Claim(
            statement="unemployment around 4.8%",
            metric_type="latest",
            series_id="UNRATE",
            expected_value=4.8,
            date_range=("2026-02", "2026-02"),
        )
        result = verify_agent_claim(claim, verify_db)
        assert result.passed is True

    def test_direction_increasing_passes(self, verify_db: str) -> None:
        """Direction claim for increasing trend passes when data increases."""
        claim = Claim(
            statement="unemployment trending up",
            metric_type="direction",
            series_id="UNRATE",
            expected_value=1.0,
            date_range=("2025-10", "2026-02"),
        )
        result = verify_agent_claim(claim, verify_db)
        assert result.passed is True
        assert result.actual_value == 1.0

    def test_direction_decreasing_fails(self, verify_db: str) -> None:
        """Direction claim for decreasing fails when data is increasing."""
        claim = Claim(
            statement="unemployment trending down",
            metric_type="direction",
            series_id="UNRATE",
            expected_value=-1.0,
            date_range=("2025-10", "2026-02"),
        )
        result = verify_agent_claim(claim, verify_db)
        assert result.passed is False

    def test_average_within_tolerance(self, verify_db: str) -> None:
        """Average claim within tolerance passes."""
        # Average of 4.0, 4.1, 4.2, 4.3, 4.4 = 4.2
        claim = Claim(
            statement="average unemployment 4.2%",
            metric_type="average",
            series_id="UNRATE",
            expected_value=4.2,
            date_range=("2025-10", "2026-02"),
        )
        result = verify_agent_claim(claim, verify_db)
        assert result.passed is True

    def test_unknown_metric_type_fails(self, verify_db: str) -> None:
        """Unknown metric_type fails with descriptive reason."""
        claim = Claim(
            statement="bogus claim",
            metric_type="nonexistent_aggregation",
            series_id="UNRATE",
            expected_value=4.4,
            date_range=("2026-02", "2026-02"),
        )
        result = verify_agent_claim(claim, verify_db)
        assert result.passed is False
        assert "unknown aggregation type" in result.reason

    def test_empty_series_no_data(self, verify_db: str) -> None:
        """Nonexistent series fails with 'no data found'."""
        claim = Claim(
            statement="fake series",
            metric_type="latest",
            series_id="NONEXISTENT",
            expected_value=1.0,
            date_range=("2026-01", "2026-02"),
        )
        result = verify_agent_claim(claim, verify_db)
        assert result.passed is False
        assert "no data" in result.reason.lower()

    def test_prediction_latest_passes(self, verify_db: str) -> None:
        """prediction_latest claim within tolerance passes."""
        claim = Claim(
            statement="latest recession probability 0.25",
            metric_type="prediction_latest",
            series_id="recession",
            expected_value=0.25,
        )
        result = verify_agent_claim(claim, verify_db)
        assert result.passed is True

    def test_date_range_none_errors_gracefully(
        self, verify_db: str
    ) -> None:
        """Claim with date_range=None for a type that needs it errors."""
        claim = Claim(
            statement="latest UNRATE no dates",
            metric_type="latest",
            series_id="UNRATE",
            expected_value=4.4,
            date_range=None,
        )
        result = verify_agent_claim(claim, verify_db)
        assert result.passed is False
        assert "error" in result.reason.lower()


# ---------------------------------------------------------------------------
# Aggregate status tests (~4)
# ---------------------------------------------------------------------------


class TestVerifyAllClaims:
    """Tests for verify_all_claims aggregate status logic."""

    def test_all_pass_verified(self, verify_db: str) -> None:
        """All claims passing results in 'Verified'."""
        claims = [
            Claim(
                statement="u1",
                metric_type="latest",
                series_id="UNRATE",
                expected_value=4.4,
                date_range=("2026-02", "2026-02"),
            ),
            Claim(
                statement="u2",
                metric_type="latest",
                series_id="UNRATE",
                expected_value=4.3,
                date_range=("2026-01", "2026-01"),
            ),
        ]
        result = verify_all_claims(claims, verify_db)
        assert result.status == "Verified"
        assert result.all_verified is True
        assert result.passed_count == 2
        assert result.total == 2

    def test_mixed_partially_verified(self, verify_db: str) -> None:
        """Mixed pass/fail results in 'Partially Verified'."""
        claims = [
            Claim(
                statement="correct",
                metric_type="latest",
                series_id="UNRATE",
                expected_value=4.4,
                date_range=("2026-02", "2026-02"),
            ),
            Claim(
                statement="wrong",
                metric_type="latest",
                series_id="UNRATE",
                expected_value=9.9,
                date_range=("2026-02", "2026-02"),
            ),
        ]
        result = verify_all_claims(claims, verify_db)
        assert result.status == "Partially Verified"
        assert result.all_verified is False
        assert result.passed_count == 1

    def test_all_fail_unverified(self, verify_db: str) -> None:
        """All claims failing results in 'Unverified'."""
        claims = [
            Claim(
                statement="wrong1",
                metric_type="latest",
                series_id="UNRATE",
                expected_value=9.9,
                date_range=("2026-02", "2026-02"),
            ),
            Claim(
                statement="wrong2",
                metric_type="latest",
                series_id="NONEXISTENT",
                expected_value=1.0,
                date_range=("2026-02", "2026-02"),
            ),
        ]
        result = verify_all_claims(claims, verify_db)
        assert result.status == "Unverified"
        assert result.all_verified is False
        assert result.passed_count == 0

    def test_zero_claims_verified(self, verify_db: str) -> None:
        """Zero claims results in 'Verified' (conceptual answer)."""
        result = verify_all_claims([], verify_db)
        assert result.status == "Verified"
        assert result.all_verified is True
        assert result.total == 0
        assert result.passed_count == 0


# ---------------------------------------------------------------------------
# Integration tests with mocked LLM (~2)
# ---------------------------------------------------------------------------


class TestIntegration:
    """Integration tests: mock LLM → parse → verify → AgentResponse."""

    def test_structured_json_response(self, verify_db: str) -> None:
        """Mock LLM structured JSON → parse → verify → populated fields."""
        from src.agent.config import AgentConfig
        from src.agent.graph import AgentResponse

        llm_output = json.dumps({
            "narrative": "The latest unemployment rate is 4.4%.",
            "claims": [
                {
                    "statement": "unemployment at 4.4%",
                    "metric_type": "latest",
                    "series_id": "UNRATE",
                    "expected_value": 4.4,
                    "date_range": ["2026-02", "2026-02"],
                }
            ],
        })

        narrative, claims = parse_agent_response(llm_output)
        verification = verify_all_claims(claims, verify_db)

        resp = AgentResponse(
            answer=narrative,
            tool_calls=[],
            claims=[dataclasses.asdict(c) for c in claims],
            verification={
                "status": verification.status,
                "all_verified": verification.all_verified,
                "results": [
                    {
                        "statement": r.claim.statement,
                        "passed": r.passed,
                        "actual_value": r.actual_value,
                        "reason": r.reason,
                    }
                    for r in verification.results
                ],
                "total": verification.total,
                "passed_count": verification.passed_count,
            },
        )

        assert resp.answer == "The latest unemployment rate is 4.4%."
        assert len(resp.claims) == 1
        assert resp.verification["status"] == "Verified"
        assert resp.verification["all_verified"] is True

    def test_plain_text_refusal_response(self, verify_db: str) -> None:
        """Mock LLM plain text refusal → empty claims, Verified status."""
        from src.agent.graph import AgentResponse

        llm_output = "I can only answer questions about the economic data."
        narrative, claims = parse_agent_response(llm_output)
        verification = verify_all_claims(claims, verify_db)

        resp = AgentResponse(
            answer=narrative,
            claims=[dataclasses.asdict(c) for c in claims],
            verification={
                "status": verification.status,
                "all_verified": verification.all_verified,
                "total": verification.total,
                "passed_count": verification.passed_count,
            },
        )

        assert resp.claims == []
        assert resp.verification["status"] == "Verified"
        assert resp.verification["all_verified"] is True


# ---------------------------------------------------------------------------
# Regression test for batch verify_insights.verify_claim (~1)
# ---------------------------------------------------------------------------


class TestBatchVerifierRegression:
    """Confirm batch verify_insights.verify_claim still works identically."""

    def test_batch_claim_dict_format(self, verify_db: str) -> None:
        """Batch verifier accepts the standard claim dict format."""
        conn = sqlite3.connect(verify_db)
        claim_dict: dict = {
            "metric": "UNRATE",
            "value": 4.4,
            "aggregation": "latest",
            "period_start": "2026-02",
            "period_end": "2026-02",
            "description": "regression test claim",
        }
        result = verify_claim(conn, claim_dict)
        conn.close()

        assert result["passed"] is True
        assert result["actual_value"] == 4.4
        assert result["reason"] == "within tolerance"
