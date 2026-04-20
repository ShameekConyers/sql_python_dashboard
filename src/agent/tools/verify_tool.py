"""Claim parsing and verification for agent responses.

Parses structured JSON claims from the agent's output and verifies each
claim against the database using the same tolerance system as the batch
pipeline (verify_insights.py).
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from dataclasses import dataclass
from typing import Any

from src.verify_insights import (
    ABSOLUTE_TOLERANCE,
    COUNT_TOLERANCE,
    RELATIVE_TOLERANCE,
    verify_claim,
)

logger: logging.Logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class Claim:
    """A single verifiable numeric assertion from the agent's response.

    Attributes:
        statement: Human-readable claim text.
        metric_type: Aggregation type mapping to VERIFIERS keys.
        series_id: FRED series or prediction table target.
        expected_value: The asserted numeric value.
        date_range: (period_start, period_end) or None.
        use_raw: Use raw value instead of COVID-adjusted.
        per_capita: Normalize by CNP16OV.
        threshold: For count_months_* types.
    """

    statement: str
    metric_type: str
    series_id: str
    expected_value: float
    date_range: tuple[str, str] | None = None
    use_raw: bool = False
    per_capita: bool = False
    threshold: float | None = None


@dataclass
class ClaimResult:
    """Verification result for a single claim.

    Attributes:
        claim: The original claim that was verified.
        passed: Whether the claim passed verification.
        actual_value: The actual value from the database.
        reason: Human-readable explanation of the result.
    """

    claim: Claim
    passed: bool
    actual_value: float | None
    reason: str


@dataclass
class VerificationResult:
    """Aggregate verification result for all claims.

    Attributes:
        status: "Verified", "Partially Verified", or "Unverified".
        all_verified: True only if every claim passed.
        results: Per-claim verification results.
        total: Total number of claims.
        passed_count: Number of claims that passed.
    """

    status: str
    all_verified: bool
    results: list[ClaimResult]
    total: int
    passed_count: int


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------


def _extract_json(raw: str) -> dict[str, Any] | None:
    """Attempt to extract a JSON object from a raw string.

    Tries three strategies in order:
    1. Parse the full string as JSON.
    2. Strip markdown fences and parse.
    3. Find the outermost { ... } substring and parse.

    Args:
        raw: Raw string from the agent.

    Returns:
        Parsed dict if successful, None otherwise.
    """
    # Strategy 1: full string
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
    except (json.JSONDecodeError, ValueError):
        pass

    # Strategy 2: strip markdown fences
    stripped = re.sub(
        r"```(?:json)?\s*\n?(.*?)\n?\s*```", r"\1", raw, flags=re.DOTALL
    )
    if stripped != raw:
        try:
            obj = json.loads(stripped)
            if isinstance(obj, dict):
                return obj
        except (json.JSONDecodeError, ValueError):
            pass

    # Strategy 3: outermost { ... }
    first_brace = raw.find("{")
    last_brace = raw.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        candidate = raw[first_brace : last_brace + 1]
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except (json.JSONDecodeError, ValueError):
            pass

    return None


def parse_agent_response(raw: str) -> tuple[str, list[Claim]]:
    """Parse an agent response into a narrative and list of claims.

    Args:
        raw: The raw string content from the agent's final message.

    Returns:
        Tuple of (narrative, claims). If the response is not valid JSON
        or has no claims, returns (raw_text, []).
    """
    obj = _extract_json(raw)
    if obj is None:
        return (raw, [])

    narrative: str = obj.get("narrative", raw)
    raw_claims: list[dict[str, Any]] = obj.get("claims", [])
    if not isinstance(raw_claims, list):
        return (narrative, [])

    claims: list[Claim] = []
    for claim_dict in raw_claims:
        # Validate required fields
        if not all(
            k in claim_dict
            for k in ("metric_type", "series_id")
        ):
            logger.warning(
                "Skipping claim missing required fields: %s", claim_dict
            )
            continue

        raw_expected = claim_dict.get("expected_value")
        if raw_expected is None:
            logger.warning(
                "Skipping claim with null expected_value: %s", claim_dict
            )
            continue

        try:
            expected_value = float(raw_expected)
        except (TypeError, ValueError):
            logger.warning(
                "Skipping claim with non-numeric expected_value: %s",
                claim_dict,
            )
            continue

        date_range: tuple[str, str] | None = None
        raw_range = claim_dict.get("date_range")
        if raw_range and isinstance(raw_range, (list, tuple)) and len(raw_range) == 2:
            date_range = (str(raw_range[0]), str(raw_range[1]))

        claims.append(
            Claim(
                statement=claim_dict.get("statement", ""),
                metric_type=claim_dict["metric_type"],
                series_id=claim_dict["series_id"],
                expected_value=expected_value,
                date_range=date_range,
                use_raw=bool(claim_dict.get("use_raw", False)),
                per_capita=bool(claim_dict.get("per_capita", False)),
                threshold=claim_dict.get("threshold"),
            )
        )

    return (narrative, claims)


# ---------------------------------------------------------------------------
# Verifier bridge
# ---------------------------------------------------------------------------


def _claim_to_verifier_dict(claim: Claim) -> dict[str, Any]:
    """Convert an agent Claim to the dict format verify_insights expects.

    Args:
        claim: An agent claim dataclass.

    Returns:
        A dict compatible with verify_insights.verify_claim().
    """
    result: dict[str, Any] = {
        "metric": claim.series_id,
        "value": claim.expected_value,
        "aggregation": claim.metric_type,
        "description": claim.statement,
        "use_raw": claim.use_raw,
        "per_capita": claim.per_capita,
    }

    if claim.date_range is not None:
        result["period_start"] = claim.date_range[0]
        result["period_end"] = claim.date_range[1]

    if claim.threshold is not None:
        result["threshold"] = claim.threshold

    return result


def verify_agent_claim(claim: Claim, db_path: str) -> ClaimResult:
    """Verify a single agent claim against the database.

    Args:
        claim: The claim to verify.
        db_path: Path to the SQLite database.

    Returns:
        A ClaimResult with the verification outcome.
    """
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        claim_dict = _claim_to_verifier_dict(claim)
        result = verify_claim(conn, claim_dict)
        return ClaimResult(
            claim=claim,
            passed=result["passed"],
            actual_value=result.get("actual_value"),
            reason=result["reason"],
        )
    finally:
        conn.close()


def verify_all_claims(
    claims: list[Claim], db_path: str
) -> VerificationResult:
    """Verify all claims and compute aggregate status.

    Args:
        claims: List of claims to verify.
        db_path: Path to the SQLite database.

    Returns:
        Aggregate VerificationResult with per-claim results.
    """
    if not claims:
        return VerificationResult(
            status="Verified",
            all_verified=True,
            results=[],
            total=0,
            passed_count=0,
        )

    results: list[ClaimResult] = []
    for claim in claims:
        results.append(verify_agent_claim(claim, db_path))

    passed_count = sum(1 for r in results if r.passed)
    total = len(results)

    if passed_count == total:
        status = "Verified"
    elif passed_count > 0:
        status = "Partially Verified"
    else:
        status = "Unverified"

    return VerificationResult(
        status=status,
        all_verified=passed_count == total,
        results=results,
        total=total,
        passed_count=passed_count,
    )
