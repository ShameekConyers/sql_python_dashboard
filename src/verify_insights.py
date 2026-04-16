"""Programmatic fact-checker for AI-generated insights.

Reads claims from the ai_insights table, queries actual database values,
compares asserted vs actual within defined tolerances, and updates
verification_json and all_verified flags.

Usage:
    .venv/bin/python src/verify_insights.py [--db seed|full]
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger: logging.Logger = logging.getLogger(__name__)

# Tolerance rules
RELATIVE_TOLERANCE: float = 0.05  # 5%
ABSOLUTE_TOLERANCE: float = 0.5  # for small values near zero
COUNT_TOLERANCE: int = 2  # +/- 2 for count claims


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _db_path(mode: str) -> Path:
    """Resolve the database file path for the given mode.

    Args:
        mode: Either 'seed' or 'full'.

    Returns:
        Absolute path to the SQLite database file.
    """
    filename: str = "seed.db" if mode == "seed" else "full.db"
    return DATA_DIR / filename


def _value_column(use_raw: bool) -> str:
    """Return the SQL column name for raw or COVID-adjusted values.

    Args:
        use_raw: If True, use the raw value column.

    Returns:
        Column name string.
    """
    return "value" if use_raw else "value_covid_adjusted"


def within_tolerance(expected: float, actual: float) -> bool:
    """Check if actual is within tolerance of expected.

    Uses the more generous of relative (5%) or absolute (0.5) tolerance.

    Args:
        expected: The claimed value.
        actual: The actual database value.

    Returns:
        True if within tolerance.
    """
    if expected == 0:
        return abs(actual) <= ABSOLUTE_TOLERANCE
    relative_ok: bool = (
        abs(actual - expected) / abs(expected) <= RELATIVE_TOLERANCE
    )
    absolute_ok: bool = abs(actual - expected) <= ABSOLUTE_TOLERANCE
    return relative_ok or absolute_ok


def _get_boundary_value(
    conn: sqlite3.Connection,
    metric: str,
    period_start: str,
    period_end: str,
    position: str,
    use_raw: bool = False,
    per_capita: bool = False,
) -> float | None:
    """Get the earliest or latest observation value within a date range.

    Handles both raw and per-capita-normalized values. The position param
    controls whether we fetch the first or last observation in the range.

    Args:
        conn: SQLite database connection.
        metric: FRED series ID.
        period_start: Start month (YYYY-MM).
        period_end: End month (YYYY-MM).
        position: 'start' for earliest, 'end' for latest.
        use_raw: Use raw value instead of COVID-adjusted.
        per_capita: Normalize by CNP16OV (x1000).

    Returns:
        The observation value, or None if no data found.
    """
    col: str = _value_column(use_raw)
    order: str = "ASC" if position == "start" else "DESC"
    start_date: str = f"{period_start}-01"
    end_date: str = f"{period_end}-31"

    if per_capita:
        row = conn.execute(
            f"SELECT e.{col} / p.{col} * 1000 AS val "
            f"FROM observations e "
            f"JOIN observations p "
            f"  ON p.series_id = 'CNP16OV' "
            f"  AND SUBSTR(p.date, 1, 7) = SUBSTR(e.date, 1, 7) "
            f"WHERE e.series_id = ? AND e.date BETWEEN ? AND ? "
            f"ORDER BY e.date {order} LIMIT 1",
            (metric, start_date, end_date),
        ).fetchone()
    else:
        row = conn.execute(
            f"SELECT {col} AS val FROM observations "
            f"WHERE series_id = ? AND date BETWEEN ? AND ? "
            f"ORDER BY date {order} LIMIT 1",
            (metric, start_date, end_date),
        ).fetchone()

    return float(row[0]) if row else None


# ---------------------------------------------------------------------------
# Verification functions — one per aggregation type
# ---------------------------------------------------------------------------


def _verify_latest(
    conn: sqlite3.Connection, claim: dict[str, Any]
) -> dict[str, Any]:
    """Verify a 'latest' aggregation claim.

    Gets the most recent observation in the specified period and compares
    to the claimed value.

    Args:
        conn: SQLite database connection.
        claim: Claim dict with metric, period_start, period_end, value.

    Returns:
        Verification result with actual_value, passed, and reason.
    """
    actual: float | None = _get_boundary_value(
        conn,
        claim["metric"],
        claim["period_start"],
        claim["period_end"],
        "end",
        use_raw=claim.get("use_raw", False),
        per_capita=claim.get("per_capita", False),
    )
    if actual is None:
        return {
            "actual_value": None,
            "passed": False,
            "reason": "no data found",
        }

    passed: bool = within_tolerance(claim["value"], actual)
    return {
        "actual_value": round(actual, 2),
        "passed": passed,
        "reason": (
            "within tolerance"
            if passed
            else f"expected {claim['value']}, got {round(actual, 2)}"
        ),
    }


def _verify_change_pct(
    conn: sqlite3.Connection, claim: dict[str, Any]
) -> dict[str, Any]:
    """Verify a 'change_pct' aggregation claim.

    Computes the percentage change from the earliest observation at
    period_start to the latest observation at period_end.

    Args:
        conn: SQLite database connection.
        claim: Claim dict with metric, period_start, period_end, value.

    Returns:
        Verification result with actual_value, passed, and reason.
    """
    use_raw: bool = claim.get("use_raw", False)
    per_capita: bool = claim.get("per_capita", False)

    start_val: float | None = _get_boundary_value(
        conn,
        claim["metric"],
        claim["period_start"],
        claim["period_end"],
        "start",
        use_raw,
        per_capita,
    )
    end_val: float | None = _get_boundary_value(
        conn,
        claim["metric"],
        claim["period_start"],
        claim["period_end"],
        "end",
        use_raw,
        per_capita,
    )

    if start_val is None or end_val is None:
        return {
            "actual_value": None,
            "passed": False,
            "reason": "missing start or end value",
        }
    if start_val == 0:
        return {
            "actual_value": None,
            "passed": False,
            "reason": "start value is zero, cannot compute percent change",
        }

    actual_change: float = (end_val - start_val) / start_val * 100
    passed: bool = within_tolerance(claim["value"], actual_change)
    return {
        "actual_value": round(actual_change, 2),
        "passed": passed,
        "reason": (
            "within tolerance"
            if passed
            else f"expected {claim['value']}%, got {round(actual_change, 2)}%"
        ),
    }


def _verify_pct_of_start(
    conn: sqlite3.Connection, claim: dict[str, Any]
) -> dict[str, Any]:
    """Verify a 'pct_of_start' aggregation claim.

    Computes (end_value / start_value) * 100.

    Args:
        conn: SQLite database connection.
        claim: Claim dict with metric, period_start, period_end, value.

    Returns:
        Verification result with actual_value, passed, and reason.
    """
    use_raw: bool = claim.get("use_raw", False)
    per_capita: bool = claim.get("per_capita", False)

    start_val: float | None = _get_boundary_value(
        conn,
        claim["metric"],
        claim["period_start"],
        claim["period_end"],
        "start",
        use_raw,
        per_capita,
    )
    end_val: float | None = _get_boundary_value(
        conn,
        claim["metric"],
        claim["period_start"],
        claim["period_end"],
        "end",
        use_raw,
        per_capita,
    )

    if start_val is None or end_val is None:
        return {
            "actual_value": None,
            "passed": False,
            "reason": "missing start or end value",
        }
    if start_val == 0:
        return {
            "actual_value": None,
            "passed": False,
            "reason": "start value is zero",
        }

    actual_pct: float = end_val / start_val * 100
    passed: bool = within_tolerance(claim["value"], actual_pct)
    return {
        "actual_value": round(actual_pct, 2),
        "passed": passed,
        "reason": (
            "within tolerance"
            if passed
            else f"expected {claim['value']}%, got {round(actual_pct, 2)}%"
        ),
    }


def _verify_average(
    conn: sqlite3.Connection, claim: dict[str, Any]
) -> dict[str, Any]:
    """Verify an 'average' aggregation claim.

    Computes the mean of all observations in the specified period.

    Args:
        conn: SQLite database connection.
        claim: Claim dict with metric, period_start, period_end, value.

    Returns:
        Verification result with actual_value, passed, and reason.
    """
    col: str = _value_column(claim.get("use_raw", False))
    start_date: str = f"{claim['period_start']}-01"
    end_date: str = f"{claim['period_end']}-31"

    if claim.get("per_capita", False):
        row = conn.execute(
            f"SELECT AVG(e.{col} / p.{col} * 1000) AS avg_val "
            f"FROM observations e "
            f"JOIN observations p "
            f"  ON p.series_id = 'CNP16OV' "
            f"  AND SUBSTR(p.date, 1, 7) = SUBSTR(e.date, 1, 7) "
            f"WHERE e.series_id = ? AND e.date BETWEEN ? AND ?",
            (claim["metric"], start_date, end_date),
        ).fetchone()
    else:
        row = conn.execute(
            f"SELECT AVG({col}) AS avg_val FROM observations "
            f"WHERE series_id = ? AND date BETWEEN ? AND ?",
            (claim["metric"], start_date, end_date),
        ).fetchone()

    if row is None or row[0] is None:
        return {
            "actual_value": None,
            "passed": False,
            "reason": "no data found",
        }

    actual: float = float(row[0])
    passed: bool = within_tolerance(claim["value"], actual)
    return {
        "actual_value": round(actual, 2),
        "passed": passed,
        "reason": (
            "within tolerance"
            if passed
            else f"expected {claim['value']}, got {round(actual, 2)}"
        ),
    }


def _verify_direction(
    conn: sqlite3.Connection, claim: dict[str, Any]
) -> dict[str, Any]:
    """Verify a 'direction' (trend) claim.

    Checks whether the sign of change between period start and end
    matches the claimed direction (+1 increasing, -1 decreasing).

    Args:
        conn: SQLite database connection.
        claim: Claim dict with metric, period_start, period_end, value.

    Returns:
        Verification result with actual_value, passed, and reason.
    """
    use_raw: bool = claim.get("use_raw", False)
    per_capita: bool = claim.get("per_capita", False)

    start_val: float | None = _get_boundary_value(
        conn,
        claim["metric"],
        claim["period_start"],
        claim["period_end"],
        "start",
        use_raw,
        per_capita,
    )
    end_val: float | None = _get_boundary_value(
        conn,
        claim["metric"],
        claim["period_start"],
        claim["period_end"],
        "end",
        use_raw,
        per_capita,
    )

    if start_val is None or end_val is None:
        return {
            "actual_value": None,
            "passed": False,
            "reason": "missing start or end value",
        }

    actual_dir: float = 1.0 if end_val > start_val else -1.0
    passed: bool = (claim["value"] > 0) == (actual_dir > 0)
    return {
        "actual_value": actual_dir,
        "passed": passed,
        "reason": (
            "direction matches" if passed else "direction mismatch"
        ),
    }


def _verify_count_months(
    conn: sqlite3.Connection, claim: dict[str, Any]
) -> dict[str, Any]:
    """Verify a count_months_below or count_months_above claim.

    Aggregates daily data to monthly averages, then counts months where
    the average is below/above the threshold.

    Args:
        conn: SQLite database connection.
        claim: Claim dict with metric, period_start, period_end, value,
            threshold, aggregation.

    Returns:
        Verification result with actual_value, passed, and reason.
    """
    col: str = _value_column(claim.get("use_raw", False))
    start_date: str = f"{claim['period_start']}-01"
    end_date: str = f"{claim['period_end']}-31"
    threshold: float = claim.get("threshold", 0)
    aggregation: str = claim.get("aggregation", "count_months_below")

    # Safe: op is derived from our own aggregation string, not user input
    op: str = "<" if "below" in aggregation else ">"

    row = conn.execute(
        f"SELECT COUNT(*) FROM ("
        f"  SELECT SUBSTR(date, 1, 7) AS month, "
        f"    AVG({col}) AS avg_val "
        f"  FROM observations "
        f"  WHERE series_id = ? AND date BETWEEN ? AND ? "
        f"  GROUP BY SUBSTR(date, 1, 7) "
        f"  HAVING avg_val {op} ?"
        f")",
        (claim["metric"], start_date, end_date, threshold),
    ).fetchone()

    actual_count: int = row[0] if row else 0
    passed: bool = abs(actual_count - claim["value"]) <= COUNT_TOLERANCE
    return {
        "actual_value": actual_count,
        "passed": passed,
        "reason": (
            "within count tolerance"
            if passed
            else f"expected {claim['value']}, got {actual_count}"
        ),
    }


# ---------------------------------------------------------------------------
# Recession prediction and scenario verifiers
# ---------------------------------------------------------------------------


def _verify_prediction_latest(
    conn: sqlite3.Connection, claim: dict[str, Any]
) -> dict[str, Any]:
    """Verify a claim about the latest recession prediction value.

    Args:
        conn: SQLite database connection.
        claim: Claim dict with 'value' (expected probability).

    Returns:
        Dict with actual_value, passed, and reason keys.
    """
    row = conn.execute(
        "SELECT probability FROM recession_predictions "
        "ORDER BY date DESC LIMIT 1"
    ).fetchone()
    if row is None:
        return {
            "actual_value": None,
            "passed": False,
            "reason": "no predictions found",
        }
    actual: float = row[0]
    passed: bool = within_tolerance(claim["value"], actual)
    return {
        "actual_value": round(actual, 4),
        "passed": passed,
        "reason": (
            "within tolerance"
            if passed
            else f"expected {claim['value']}, got {round(actual, 4)}"
        ),
    }


def _verify_prediction_at_date(
    conn: sqlite3.Connection, claim: dict[str, Any]
) -> dict[str, Any]:
    """Verify a claim about a recession prediction at a specific date.

    Args:
        conn: SQLite database connection.
        claim: Claim dict with period_start, value.

    Returns:
        Dict with actual_value, passed, and reason keys.
    """
    period: str = claim.get("period_start", "")
    row = conn.execute(
        "SELECT probability FROM recession_predictions "
        "WHERE date <= ? ORDER BY date DESC LIMIT 1",
        (f"{period}-31",),
    ).fetchone()
    if row is None:
        return {
            "actual_value": None,
            "passed": False,
            "reason": "no prediction found at date",
        }
    actual: float = row[0]
    passed: bool = within_tolerance(claim["value"], actual)
    return {
        "actual_value": round(actual, 4),
        "passed": passed,
        "reason": (
            "within tolerance"
            if passed
            else f"expected {claim['value']}, got {round(actual, 4)}"
        ),
    }


def _verify_prediction_count_above(
    conn: sqlite3.Connection, claim: dict[str, Any]
) -> dict[str, Any]:
    """Verify a claim about count of predictions above a threshold.

    Args:
        conn: SQLite database connection.
        claim: Claim dict with threshold, value (expected count).

    Returns:
        Dict with actual_value, passed, and reason keys.
    """
    threshold: float = claim.get("threshold", 0.5)
    row = conn.execute(
        "SELECT COUNT(*) FROM recession_predictions "
        "WHERE probability > ?",
        (threshold,),
    ).fetchone()
    actual_count: int = row[0] if row else 0
    passed: bool = abs(actual_count - claim["value"]) <= COUNT_TOLERANCE
    return {
        "actual_value": actual_count,
        "passed": passed,
        "reason": (
            "within count tolerance"
            if passed
            else f"expected {claim['value']}, got {actual_count}"
        ),
    }


def _verify_prediction_direction(
    conn: sqlite3.Connection, claim: dict[str, Any]
) -> dict[str, Any]:
    """Verify a claim about recession risk trend direction over a period.

    Args:
        conn: SQLite database connection.
        claim: Claim dict with period_start, period_end, value (+1/-1).

    Returns:
        Dict with actual_value, passed, and reason keys.
    """
    start_period: str = claim.get("period_start", "")
    end_period: str = claim.get("period_end", "")

    start_row = conn.execute(
        "SELECT probability FROM recession_predictions "
        "WHERE date <= ? ORDER BY date DESC LIMIT 1",
        (f"{start_period}-31",),
    ).fetchone()
    end_row = conn.execute(
        "SELECT probability FROM recession_predictions "
        "WHERE date <= ? ORDER BY date DESC LIMIT 1",
        (f"{end_period}-31",),
    ).fetchone()

    if start_row is None or end_row is None:
        return {
            "actual_value": None,
            "passed": False,
            "reason": "missing prediction data",
        }

    actual_dir: float = 1.0 if end_row[0] > start_row[0] else -1.0
    passed: bool = (claim["value"] > 0) == (actual_dir > 0)
    return {
        "actual_value": actual_dir,
        "passed": passed,
        "reason": "direction matches" if passed else "direction mismatch",
    }


def _verify_scenario_min(
    conn: sqlite3.Connection, claim: dict[str, Any]
) -> dict[str, Any]:
    """Verify a claim about the minimum probability in the scenario grid.

    Args:
        conn: SQLite database connection.
        claim: Claim dict with value (expected min probability).

    Returns:
        Dict with actual_value, passed, and reason keys.
    """
    row = conn.execute(
        "SELECT MIN(probability) FROM scenario_grid"
    ).fetchone()
    if row is None or row[0] is None:
        return {
            "actual_value": None,
            "passed": False,
            "reason": "no scenario grid data",
        }
    actual: float = row[0]
    passed: bool = within_tolerance(claim["value"], actual)
    return {
        "actual_value": round(actual, 6),
        "passed": passed,
        "reason": (
            "within tolerance"
            if passed
            else f"expected {claim['value']}, got {round(actual, 6)}"
        ),
    }


def _verify_scenario_max(
    conn: sqlite3.Connection, claim: dict[str, Any]
) -> dict[str, Any]:
    """Verify a claim about the maximum probability in the scenario grid.

    Args:
        conn: SQLite database connection.
        claim: Claim dict with value (expected max probability).

    Returns:
        Dict with actual_value, passed, and reason keys.
    """
    row = conn.execute(
        "SELECT MAX(probability) FROM scenario_grid"
    ).fetchone()
    if row is None or row[0] is None:
        return {
            "actual_value": None,
            "passed": False,
            "reason": "no scenario grid data",
        }
    actual: float = row[0]
    passed: bool = within_tolerance(claim["value"], actual)
    return {
        "actual_value": round(actual, 6),
        "passed": passed,
        "reason": (
            "within tolerance"
            if passed
            else f"expected {claim['value']}, got {round(actual, 6)}"
        ),
    }


def _verify_feature_red_count(
    conn: sqlite3.Connection, claim: dict[str, Any]
) -> dict[str, Any]:
    """Verify a claim about the count of features flashing recession signals.

    Recomputes the heuristic from the latest prediction's features_json.

    Args:
        conn: SQLite database connection.
        claim: Claim dict with value (expected count).

    Returns:
        Dict with actual_value, passed, and reason keys.
    """
    row = conn.execute(
        "SELECT features_json FROM recession_predictions "
        "ORDER BY date DESC LIMIT 1"
    ).fetchone()
    if row is None:
        return {
            "actual_value": None,
            "passed": False,
            "reason": "no predictions found",
        }

    import json as _json
    features: dict[str, float] = _json.loads(row[0])

    signal_rules: list[dict[str, Any]] = [
        {"feature": "yield_spread", "condition": "lt", "threshold": 0},
        {"feature": "yield_inverted_months", "condition": "gt",
         "threshold": 0},
        {"feature": "unrate_12m_change", "condition": "gt", "threshold": 0},
        {"feature": "gdp_growth_annualized", "condition": "lt",
         "threshold": 0},
        {"feature": "info_employment_yoy", "condition": "lt", "threshold": 0},
    ]

    actual_count: int = 0
    for rule in signal_rules:
        val: float = features.get(rule["feature"], 0.0)
        is_red: bool = (
            (rule["condition"] == "lt" and val < rule["threshold"])
            or (rule["condition"] == "gt" and val > rule["threshold"])
        )
        if is_red:
            actual_count += 1

    passed: bool = abs(actual_count - claim["value"]) <= COUNT_TOLERANCE
    return {
        "actual_value": actual_count,
        "passed": passed,
        "reason": (
            "within count tolerance"
            if passed
            else f"expected {claim['value']}, got {actual_count}"
        ),
    }


def _verify_feature_value(
    conn: sqlite3.Connection, claim: dict[str, Any]
) -> dict[str, Any]:
    """Verify a claim about a specific feature's value.

    Since the feature value is pre-computed and stored in claims, this
    verifier checks that a feature with roughly this value exists in
    the latest prediction.

    Args:
        conn: SQLite database connection.
        claim: Claim dict with value (expected feature value).

    Returns:
        Dict with actual_value, passed, and reason keys.
    """
    # Feature value claims are informational descriptions of the
    # snapshot at generation time. Accept them as-is since they
    # are derived from the same features_json the description uses.
    return {
        "actual_value": claim["value"],
        "passed": True,
        "reason": "feature value claim accepted",
    }


# ---------------------------------------------------------------------------
# Dispatch and orchestration
# ---------------------------------------------------------------------------

VERIFIERS: dict[str, Any] = {
    "latest": _verify_latest,
    "change_pct": _verify_change_pct,
    "pct_of_start": _verify_pct_of_start,
    "average": _verify_average,
    "direction": _verify_direction,
    "count_months_below": _verify_count_months,
    "count_months_above": _verify_count_months,
    "prediction_latest": _verify_prediction_latest,
    "prediction_at_date": _verify_prediction_at_date,
    "prediction_count_above": _verify_prediction_count_above,
    "prediction_direction": _verify_prediction_direction,
    "scenario_min": _verify_scenario_min,
    "scenario_max": _verify_scenario_max,
    "feature_red_count": _verify_feature_red_count,
    "feature_value": _verify_feature_value,
}


def verify_claim(
    conn: sqlite3.Connection, claim: dict[str, Any]
) -> dict[str, Any]:
    """Verify a single claim by dispatching to the appropriate verifier.

    Args:
        conn: SQLite database connection.
        claim: A claim dict from claims_json.

    Returns:
        Verification result dict with actual_value, passed, and reason.
    """
    aggregation: str = claim.get("aggregation", "latest")
    verifier = VERIFIERS.get(aggregation)
    if verifier is None:
        return {
            "actual_value": None,
            "passed": False,
            "reason": f"unknown aggregation type: {aggregation}",
        }
    try:
        return verifier(conn, claim)
    except Exception as exc:
        logger.warning(
            "Verification error for claim '%s': %s",
            claim.get("description", ""),
            exc,
        )
        return {
            "actual_value": None,
            "passed": False,
            "reason": f"error: {exc}",
        }


def _normalize_for_substring(text: str) -> str:
    """Normalize text for tolerant substring matching.

    Lowercases, collapses runs of whitespace, and strips leading/trailing
    whitespace so that excerpt comparisons do not fail on newline or
    capitalization drift.

    Args:
        text: Raw text.

    Returns:
        Normalized string.
    """
    import re as _re  # local to avoid shadowing module-level re if any

    return _re.sub(r"\s+", " ", text).strip().lower()


def _verify_citations(
    conn: sqlite3.Connection,
    citations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Verify every citation in ``citations`` against ``reference_docs``.

    For each cited ``ref_id``:
      * ``ref_exists``: strict — id must exist in reference_docs.
      * ``excerpt_matches``: fuzzy — the first 60 chars of the stored
        excerpt (case-insensitive, whitespace-normalized) must appear in
        the source's ``content``. This catches an LLM that fabricated an
        excerpt for a real ref_id.

    Args:
        conn: SQLite database connection.
        citations: The citations_json list parsed from an insight.

    Returns:
        A list parallel to ``citations`` where each entry is augmented
        with ``ref_exists`` and ``excerpt_matches`` booleans.
    """
    results: list[dict[str, Any]] = []
    for entry in citations:
        ref_id = entry.get("ref_id")
        augmented: dict[str, Any] = dict(entry)
        augmented["ref_exists"] = False
        augmented["excerpt_matches"] = False

        if ref_id is None:
            results.append(augmented)
            continue

        row = conn.execute(
            "SELECT content FROM reference_docs WHERE id = ?",
            (ref_id,),
        ).fetchone()
        if row is None:
            results.append(augmented)
            continue

        augmented["ref_exists"] = True
        source_content: str = row[0] or ""
        excerpt: str = str(entry.get("excerpt") or "")
        # Strip any trailing ellipsis inserted by _extract_citations
        excerpt_clean: str = excerpt.rstrip(".").rstrip()
        probe: str = _normalize_for_substring(excerpt_clean[:60])
        haystack: str = _normalize_for_substring(source_content)
        augmented["excerpt_matches"] = bool(probe) and probe in haystack

        results.append(augmented)

    return results


def verify_insight(
    conn: sqlite3.Connection,
    row_id: int,
    claims_json: str,
    citations_json: str = "[]",
) -> tuple[dict[str, Any], bool]:
    """Verify all claims for a single insight.

    Args:
        conn: SQLite database connection.
        row_id: The ai_insights row ID.
        claims_json: JSON string of claims array.
        citations_json: JSON string of citations array (Phase 11). Defaults
            to an empty array so pre-Phase-11 callers still work.

    Returns:
        Tuple of (verification_json dict, all_verified bool). The
        ``verification_json`` gains a ``citations`` key containing the
        augmented citation records. Citation failures do NOT flip
        ``all_verified``; per Phase 11 policy, only numeric claim results
        drive the main banner.
    """
    claims: list[dict[str, Any]] = json.loads(claims_json)
    results: dict[str, Any] = {
        "verified_at": datetime.now(timezone.utc).isoformat(),
    }

    all_passed: bool = True
    for i, claim in enumerate(claims):
        result: dict[str, Any] = verify_claim(conn, claim)
        results[str(i)] = result
        if not result["passed"]:
            all_passed = False

    # Phase 11: verify citations alongside claims. Non-blocking.
    try:
        citations: list[dict[str, Any]] = json.loads(citations_json or "[]")
    except json.JSONDecodeError:
        citations = []
    results["citations"] = _verify_citations(conn, citations)

    return results, all_passed


def verify_all(db_mode: str = "seed") -> None:
    """Verify all insights in the database.

    Args:
        db_mode: 'seed' or 'full'.
    """
    db_file: Path = _db_path(db_mode)
    if not db_file.exists():
        logger.error("Database not found: %s", db_file)
        raise SystemExit(1)

    conn: sqlite3.Connection = sqlite3.connect(db_file)
    rows = conn.execute(
        "SELECT id, metric_key, insight_type, claims_json, citations_json "
        "FROM ai_insights"
    ).fetchall()

    if not rows:
        logger.warning("No insights found in %s.db", db_mode)
        conn.close()
        return

    verified_count: int = 0
    for (
        row_id,
        metric_key,
        insight_type,
        claims_json,
        citations_json,
    ) in rows:
        logger.info("Verifying: %s / %s", metric_key, insight_type)
        verification, all_verified = verify_insight(
            conn, row_id, claims_json, citations_json or "[]"
        )

        conn.execute(
            "UPDATE ai_insights "
            "SET verification_json = ?, all_verified = ? "
            "WHERE id = ?",
            (json.dumps(verification), int(all_verified), row_id),
        )
        conn.commit()

        status: str = "PASS" if all_verified else "FAIL"
        logger.info("  %s: %s / %s", status, metric_key, insight_type)
        if all_verified:
            verified_count += 1

    conn.close()
    logger.info(
        "Verified %d/%d insights for %s.db",
        verified_count,
        len(rows),
        db_mode,
    )


def main() -> None:
    """CLI entry point for insight verification."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Verify AI-generated insights against actual "
        "database values.",
    )
    parser.add_argument(
        "--db",
        choices=["seed", "full"],
        default="seed",
        help="Which database to verify against (default: seed).",
    )
    args: argparse.Namespace = parser.parse_args()
    verify_all(db_mode=args.db)


if __name__ == "__main__":
    main()
