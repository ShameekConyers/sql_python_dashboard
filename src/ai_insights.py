"""Batch-generate AI narrative insights for the macro economic dashboard.

Queries the database for metric aggregations, sends structured prompts to
a local LLM via Ollama, parses the response into narrative + claims JSON,
and stores results in the ai_insights table. Never called live by the
dashboard.

Usage:
    .venv/bin/python src/ai_insights.py [--db seed|full] [--metric KEY]
                                        [--model MODEL]
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

import rag_retrieval

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger: logging.Logger = logging.getLogger(__name__)

DEFAULT_MODEL: str = "llama3.1:8b"
OLLAMA_BASE_URL: str = "http://localhost:11434"
CHATGPT_LAUNCH: str = "2022-11"
"""Year-month of ChatGPT public release, used as an analytical anchor point."""

SERIES_KIND: dict[str, str] = {
    # Rate-type series — report changes in percentage points
    "UNRATE": "rate",
    "U6RATE": "rate",
    "T10Y2Y": "rate",
    "CPIAUCSL_YOY": "rate",  # virtual key for CPI YoY framing
    # Level-type series — report changes as percent
    "GDPC1": "level",
    "CPIAUCSL": "level",
    "USINFO": "level",
    "CES2023800001": "level",
    "IPG2211S": "level",
    "CNP16OV": "level",
    "USREC": "level",  # binary, never meaningfully report changes
}
"""Classification of every FRED series into rate vs level.

Rate-type series are percentages (unemployment, yield spread, CPI YoY) and
period-over-period changes should be reported in percentage points. Level-type
series are dollar values or indexes where percent change is the natural unit.
"""

SIGNAL_RULES: list[dict[str, Any]] = [
    {"feature": "yield_spread", "condition": "lt", "threshold": 0},
    {"feature": "yield_inverted_months", "condition": "gt", "threshold": 0},
    {"feature": "unrate_12m_change", "condition": "gt", "threshold": 0},
    {"feature": "gdp_growth_annualized", "condition": "lt", "threshold": 0},
    {"feature": "info_employment_yoy", "condition": "lt", "threshold": 0},
]
"""Heuristic thresholds that classify recession-feature values as RED.

Shared by _claims_feature_snapshot() (which counts red features) and
_context_feature_snapshot() (which labels each feature RED/GREEN/NEUTRAL in
the LLM context string).
"""

NARRATIVE_PATTERNS: list[tuple[str, str]] = [
    # Category 3 — probability language applied to recession risk score
    (
        r"(?i)\b\d+(?:\.\d+)?\s*%\s+(?:likelihood|chance|probability)\b",
        "Describe recession scores as a 0-1 reading, not a probability.",
    ),
    (
        r"(?i)\blikelihood of (?:a )?recession\b",
        "Do not describe recession score as a likelihood.",
    ),
    (
        r"(?i)\bprobability of (?:a )?recession\b",
        "Do not describe recession score as a probability.",
    ),
    # Category 1 — wrong terminology (productivity vs employment)
    (
        r"(?i)\bdecline in productivity\b|\bproductivity decline\b",
        "Use 'employment' not 'productivity' for labor series.",
    ),
    (
        r"(?i)per-capita (?:output|activity)\b",
        "Use 'per-capita employment index' not 'per-capita output/activity'.",
    ),
    # Category 6 — causal verbs
    (
        r"(?i)\b(?:is|are)\s+(?:causing|driving|displacing)\b",
        "Use 'coincides with' or 'tracks alongside' instead of causal verbs.",
    ),
    (
        r"(?i)\b(?:is|are)\s+responsible for\b",
        "Do not assert causation.",
    ),
    # Category 2 — hallucinated dollar-billion expansions from GDP level
    (
        r"(?i)(?:quarterly|annualized|annual|monthly)\s+"
        r"(?:expansion|growth|increase|gain)\s+of\s+\$\d",
        "Dollar figures from GDP claims are levels, not expansions.",
    ),
    (
        r"(?i)\$\d[\d,]*\s*B?\s+(?:expansion|gain|increase)",
        "Do not describe dollar level amounts as expansions or gains.",
    ),
]
"""Regex anti-patterns flagged after narrative generation.

Each tuple is (pattern, human-readable message). Warnings are logged; storage
is never blocked. Operators judge whether to regenerate flagged slices.
"""

METHODOLOGY_LANGUAGE_RE: re.Pattern[str] = re.compile(
    r"(?i)\b("
    r"published by"
    r"|seasonally adjusted"
    r"|as defined by"
    r"|measured by"
    r"|(?:the\s+)?(?:bls|bureau of labor statistics|bea|federal reserve)"
    r")\b"
)
"""Methodology phrasing that should be accompanied by a citation.

Only consulted in _validate_narrative when retrieval actually returned
chunks. If REFERENCE CONTEXT was empty, this check is skipped because the
narrative had no refs to cite.
"""

SYSTEM_PROMPT: str = """\
You are an economic analyst writing insights for a macro economic dashboard \
that tracks recession indicators, AI's impact on the labor market, and energy \
production.

Write a 2-3 sentence analyst-quality narrative paragraph. Be specific, \
reference the numbers from KEY FINDINGS, and avoid hedging. Write as if \
briefing a portfolio manager.

RULES:
1. NUMBERS. Only reference numbers that appear in KEY FINDINGS or DATA \
CONTEXT. Do not invent, extrapolate, or round figures beyond what is \
provided. Dollar values labeled "level" are levels, not changes or \
expansions.
2. TERMINOLOGY. Use the exact labels from the context. Employment is \
"employment," not "productivity" or "output." An index is an "index," not a \
"level" or "rate" unless explicitly labeled as such.
3. UNITS. For rate-type series (unemployment rate, U6, yield spread, CPI \
YoY), describe changes in percentage points (e.g., "+0.1pp") when the \
context provides a pp framing. For levels and indexes (GDP, CPI level, \
employment counts, indexes), describe changes as percent changes. Follow \
the unit cue in the context string. Unit directives that appear after an \
em-dash (e.g. "— report in pp") are instructions to you — do NOT repeat \
them in the narrative; just follow the unit.
4. MODEL OUTPUTS. The recession risk score is a 0-1 value indicating \
resemblance to historical pre-recession periods. Do NOT call it a \
"probability," "likelihood," "chance," or "percent chance." Refer to it as \
a "score," "reading," or "risk score."
5. CAUSATION. Use "coincides with," "is consistent with," "tracks \
alongside," or "suggests" when interpreting correlations. Do NOT use \
"causes," "drives," "displaces," "is responsible for," or similar \
definitive causal verbs. The dashboard explores a thesis; it does not \
prove it.
6. SCALES. Do not subtract or compare numbers on different scales (e.g. a \
per-capita employment index and a power output index). If the context \
provides a precomputed gap, reference it as given; do not compute a new \
gap in the narrative.
7. CITATIONS. When you reference methodology, how a series is measured, \
who publishes it, or what category it belongs to, cite the supporting \
reference using its [ref:N] tag exactly as it appears in REFERENCE \
CONTEXT. Do not invent tags. Do not cite a tag that is not in REFERENCE \
CONTEXT. If REFERENCE CONTEXT is empty, write the narrative without \
citations. Cite at most two references per narrative.

Return ONLY the narrative text. No JSON, no markdown, no bullet points, \
no preamble like "Here is...". Just the paragraph.
"""


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


def _compute_slice_key(conn: sqlite3.Connection) -> str:
    """Compute a human-readable slice key from the database date range.

    Returns:
        A string like '2016-2026' representing the data span.
    """
    row = conn.execute(
        "SELECT MIN(SUBSTR(date, 1, 4)), MAX(SUBSTR(date, 1, 4)) "
        "FROM observations"
    ).fetchone()
    return f"{row[0]}-{row[1]}"


def _check_ollama() -> None:
    """Verify that Ollama is running and reachable.

    Raises:
        SystemExit: If Ollama is not responding at OLLAMA_BASE_URL.
    """
    try:
        resp: httpx.Response = httpx.get(
            f"{OLLAMA_BASE_URL}/api/tags", timeout=5.0
        )
        resp.raise_for_status()
    except (httpx.ConnectError, httpx.HTTPStatusError) as exc:
        logger.error(
            "Ollama not reachable at %s. "
            "Start it with: brew services start ollama. Error: %s",
            OLLAMA_BASE_URL,
            exc,
        )
        raise SystemExit(1)


def _latest_month(conn: sqlite3.Connection, series_id: str) -> str:
    """Get the latest month (YYYY-MM) for a series in the database.

    Args:
        conn: SQLite database connection.
        series_id: FRED series ID.

    Returns:
        Month string in YYYY-MM format.
    """
    return conn.execute(
        "SELECT MAX(SUBSTR(date, 1, 7)) FROM observations "
        "WHERE series_id = ?",
        (series_id,),
    ).fetchone()[0]


def _earliest_month(conn: sqlite3.Connection, series_id: str) -> str:
    """Get the earliest month (YYYY-MM) for a series in the database.

    Args:
        conn: SQLite database connection.
        series_id: FRED series ID.

    Returns:
        Month string in YYYY-MM format.
    """
    return conn.execute(
        "SELECT MIN(SUBSTR(date, 1, 7)) FROM observations "
        "WHERE series_id = ?",
        (series_id,),
    ).fetchone()[0]


def _month_offset(month: str, offset: int) -> str:
    """Compute a YYYY-MM string shifted by N months.

    Args:
        month: Base month in YYYY-MM format.
        offset: Number of months to shift (negative for past).

    Returns:
        Shifted month string in YYYY-MM format.
    """
    year: int = int(month[:4])
    mon: int = int(month[5:7])
    total: int = year * 12 + mon - 1 + offset
    return f"{total // 12}-{total % 12 + 1:02d}"


def _boundary_value(
    conn: sqlite3.Connection,
    series_id: str,
    period_start: str,
    period_end: str,
    position: str = "end",
    use_raw: bool = False,
    per_capita: bool = False,
) -> float | None:
    """Get the earliest or latest observation in a date range.

    Matches the query pattern in verify_insights._get_boundary_value so
    any claim value built from this will pass verification.

    Args:
        conn: SQLite database connection.
        series_id: FRED series ID.
        period_start: Start month (YYYY-MM).
        period_end: End month (YYYY-MM).
        position: 'start' for earliest, 'end' for latest.
        use_raw: Use raw value instead of COVID-adjusted.
        per_capita: Normalize by CNP16OV (x1000).

    Returns:
        The observation value, or None if no data found.
    """
    col: str = "value" if use_raw else "value_covid_adjusted"
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
            (series_id, start_date, end_date),
        ).fetchone()
    else:
        row = conn.execute(
            f"SELECT {col} AS val FROM observations "
            f"WHERE series_id = ? AND date BETWEEN ? AND ? "
            f"ORDER BY date {order} LIMIT 1",
            (series_id, start_date, end_date),
        ).fetchone()
    return float(row[0]) if row else None


def _compute_change(
    conn: sqlite3.Connection,
    series_id: str,
    period_start: str,
    period_end: str,
    use_raw: bool = False,
    per_capita: bool = False,
) -> float:
    """Compute percent change matching the verifier's change_pct logic.

    Args:
        conn: SQLite database connection.
        series_id: FRED series ID.
        period_start: Start month (YYYY-MM).
        period_end: End month (YYYY-MM).
        use_raw: Use raw values.
        per_capita: Normalize by CNP16OV.

    Returns:
        Percentage change rounded to 2 decimal places.
    """
    start_val: float | None = _boundary_value(
        conn, series_id, period_start, period_end,
        "start", use_raw, per_capita,
    )
    end_val: float | None = _boundary_value(
        conn, series_id, period_start, period_end,
        "end", use_raw, per_capita,
    )
    if start_val is None or end_val is None or start_val == 0:
        return 0.0
    return round((end_val - start_val) / start_val * 100, 2)


def _compute_pct_of_start(
    conn: sqlite3.Connection,
    series_id: str,
    period_start: str,
    period_end: str,
    use_raw: bool = False,
    per_capita: bool = False,
) -> float:
    """Compute end as percentage of start, matching verifier's pct_of_start.

    Args:
        conn: SQLite database connection.
        series_id: FRED series ID.
        period_start: Start month (YYYY-MM).
        period_end: End month (YYYY-MM).
        use_raw: Use raw values.
        per_capita: Normalize by CNP16OV.

    Returns:
        Percentage rounded to 2 decimal places.
    """
    start_val: float | None = _boundary_value(
        conn, series_id, period_start, period_end,
        "start", use_raw, per_capita,
    )
    end_val: float | None = _boundary_value(
        conn, series_id, period_start, period_end,
        "end", use_raw, per_capita,
    )
    if start_val is None or end_val is None or start_val == 0:
        return 0.0
    return round(end_val / start_val * 100, 2)


def _fmt_change(
    series_id: str,
    start_val: float,
    end_val: float,
) -> str:
    """Format a period change with explicit units and a unit cue.

    Rate-type series (per SERIES_KIND) report absolute percentage-point
    change first with the relative percent change in parentheses. Level-type
    series report the percent change first with the absolute delta in
    parentheses. The returned string ends with an em-dash hint ("— report in
    pp" or "— report as percent change") so the LLM follows the system-prompt
    unit rule rather than guessing. The system prompt tells the model not to
    echo the hint in the narrative.

    Args:
        series_id: FRED series ID (or 'CPIAUCSL_YOY' for CPI YoY framing).
        start_val: Starting value.
        end_val: Ending value.

    Returns:
        Formatted change string for inclusion in a context_text block.

    Raises:
        KeyError: If the series is not registered in SERIES_KIND.
    """
    kind: str = SERIES_KIND[series_id]
    pp: float = end_val - start_val
    rel: float = (
        (end_val - start_val) / start_val * 100 if start_val else 0.0
    )
    if kind == "rate":
        return (
            f"{pp:+.2f}pp (from {start_val:.2f} to {end_val:.2f}; "
            f"relative {rel:+.2f}%) — report in pp"
        )
    return (
        f"{rel:+.2f}% (from {start_val:.2f} to {end_val:.2f}; "
        f"absolute delta {pp:+.2f}) — report as percent change"
    )


def _make_claim(
    description: str,
    metric: str,
    value: float,
    claim_type: str = "value",
    aggregation: str = "latest",
    period_start: str = "",
    period_end: str = "",
    per_capita: bool = False,
    use_raw: bool = False,
    threshold: float | None = None,
    comparison_metric: str | None = None,
) -> dict[str, Any]:
    """Build a structured claim dict.

    Args:
        description: Human-readable claim text for the dashboard.
        metric: FRED series ID.
        value: The asserted numeric value.
        claim_type: value, trend, comparison, or count.
        aggregation: How to verify the value.
        period_start: Start month (YYYY-MM).
        period_end: End month (YYYY-MM).
        per_capita: Whether the value is per-capita normalized.
        use_raw: Whether to use raw values (not COVID-adjusted).
        threshold: Threshold for count aggregations.
        comparison_metric: Second series for comparison claims.

    Returns:
        Structured claim dict compatible with the dashboard and verifier.
    """
    claim: dict[str, Any] = {
        "description": description,
        "metric": metric,
        "value": value,
        "claim_type": claim_type,
        "aggregation": aggregation,
        "period_start": period_start,
        "period_end": period_end,
        "per_capita": per_capita,
        "use_raw": use_raw,
    }
    if threshold is not None:
        claim["threshold"] = threshold
    if comparison_metric is not None:
        claim["comparison_metric"] = comparison_metric
    return claim


# ---------------------------------------------------------------------------
# Claims builders — one per insight slice
# ---------------------------------------------------------------------------


def _claims_yield_curve_unemployment(
    conn: sqlite3.Connection,
) -> list[dict[str, Any]]:
    """Build verifiable claims for yield curve vs unemployment.

    Args:
        conn: SQLite database connection.

    Returns:
        List of structured claim dicts.
    """
    latest: str = _latest_month(conn, "UNRATE")
    earliest: str = _earliest_month(conn, "T10Y2Y")
    yr_ago: str = _month_offset(latest, -12)

    inv_count: int = conn.execute(
        "SELECT COUNT(*) FROM ("
        "  SELECT SUBSTR(date, 1, 7) AS m, "
        "    AVG(value_covid_adjusted) AS avg_v "
        "  FROM observations WHERE series_id = 'T10Y2Y' "
        "  GROUP BY m HAVING avg_v < 0"
        ")",
    ).fetchone()[0]

    unrate_12m_chg: float = _compute_change(
        conn, "UNRATE", yr_ago, latest
    )
    avg_spread: float = round(
        conn.execute(
            "SELECT AVG(avg_v) FROM ("
            "  SELECT AVG(value_covid_adjusted) AS avg_v "
            "  FROM observations WHERE series_id = 'T10Y2Y' "
            "    AND date >= ? "
            "  GROUP BY SUBSTR(date, 1, 7)"
            ")",
            (f"{yr_ago}-01",),
        ).fetchone()[0],
        2,
    )

    return [
        _make_claim(
            f"Yield curve inverted in {inv_count} of the months "
            "in this dataset",
            "T10Y2Y", inv_count, "count", "count_months_below",
            earliest, latest, threshold=0,
        ),
        _make_claim(
            f"Unemployment changed {unrate_12m_chg:+.2f}% over the "
            "past 12 months",
            "UNRATE", unrate_12m_chg, "value", "change_pct",
            yr_ago, latest,
        ),
        _make_claim(
            f"Average monthly yield spread was {avg_spread}% over "
            "the past 12 months",
            "T10Y2Y", avg_spread, "value", "average",
            yr_ago, latest,
        ),
    ]


def _claims_gdp_trend(
    conn: sqlite3.Connection,
) -> list[dict[str, Any]]:
    """Build verifiable claims for GDP growth trend.

    Args:
        conn: SQLite database connection.

    Returns:
        List of structured claim dicts.
    """
    latest: str = _latest_month(conn, "GDPC1")
    earliest: str = _earliest_month(conn, "GDPC1")
    yr_ago: str = _month_offset(latest, -12)

    full_chg: float = _compute_change(conn, "GDPC1", earliest, latest)
    recent_chg: float = _compute_change(conn, "GDPC1", yr_ago, latest)
    avg_gdp: float = round(
        conn.execute(
            "SELECT AVG(value_covid_adjusted) FROM observations "
            "WHERE series_id = 'GDPC1' AND date >= ?",
            (f"{yr_ago}-01",),
        ).fetchone()[0],
        2,
    )

    return [
        _make_claim(
            f"Real GDP grew {full_chg}% over the full dataset period",
            "GDPC1", full_chg, "value", "change_pct",
            earliest, latest,
        ),
        _make_claim(
            f"GDP expanded {recent_chg}% over the most recent "
            "4 quarters",
            "GDPC1", recent_chg, "value", "change_pct",
            yr_ago, latest,
        ),
        _make_claim(
            f"Average real GDP value over the past 4 quarters: "
            f"${avg_gdp:,.0f}B (this is a level, not a change)",
            "GDPC1", avg_gdp, "value", "average",
            yr_ago, latest,
        ),
    ]


def _claims_cpi_trend(
    conn: sqlite3.Connection,
) -> list[dict[str, Any]]:
    """Build verifiable claims for CPI inflation trend.

    Args:
        conn: SQLite database connection.

    Returns:
        List of structured claim dicts.
    """
    latest: str = _latest_month(conn, "CPIAUCSL")
    yr_ago: str = _month_offset(latest, -12)
    five_yr_ago: str = _month_offset(latest, -60)

    yoy_chg: float = _compute_change(conn, "CPIAUCSL", yr_ago, latest)
    five_yr_chg: float = _compute_change(
        conn, "CPIAUCSL", five_yr_ago, latest
    )
    avg_cpi: float = round(
        conn.execute(
            "SELECT AVG(value_covid_adjusted) FROM observations "
            "WHERE series_id = 'CPIAUCSL' AND date >= ?",
            (f"{yr_ago}-01",),
        ).fetchone()[0],
        2,
    )

    return [
        _make_claim(
            f"Consumer prices rose {yoy_chg}% year-over-year",
            "CPIAUCSL", yoy_chg, "value", "change_pct",
            yr_ago, latest,
        ),
        _make_claim(
            f"Cumulative 5-year inflation reached {five_yr_chg}%",
            "CPIAUCSL", five_yr_chg, "value", "change_pct",
            five_yr_ago, latest,
        ),
        _make_claim(
            f"Average CPI index over the past 12 months: {avg_cpi}",
            "CPIAUCSL", avg_cpi, "value", "average",
            yr_ago, latest,
        ),
    ]


def _claims_info_vs_trades(
    conn: sqlite3.Connection,
) -> list[dict[str, Any]]:
    """Build verifiable claims for info vs trades per-capita comparison.

    Args:
        conn: SQLite database connection.

    Returns:
        List of structured claim dicts.
    """
    latest: str = _latest_month(conn, "USINFO")
    earliest: str = _earliest_month(conn, "USINFO")
    chatgpt: str = CHATGPT_LAUNCH

    info_full_chg: float = _compute_change(
        conn, "USINFO", earliest, latest, per_capita=True,
    )
    trades_full_chg: float = _compute_change(
        conn, "CES2023800001", earliest, latest, per_capita=True,
    )
    info_post_ai: float = _compute_change(
        conn, "USINFO", chatgpt, latest, per_capita=True,
    )
    trades_post_ai: float = _compute_change(
        conn, "CES2023800001", chatgpt, latest, per_capita=True,
    )

    return [
        _make_claim(
            f"Info sector employment changed {info_full_chg:+.1f}% "
            "per capita over the dataset period",
            "USINFO", info_full_chg, "value", "change_pct",
            earliest, latest, per_capita=True,
        ),
        _make_claim(
            f"Specialty trades grew {trades_full_chg:+.1f}% per capita "
            "over the same period",
            "CES2023800001", trades_full_chg, "value", "change_pct",
            earliest, latest, per_capita=True,
        ),
        _make_claim(
            f"Since ChatGPT launch (Nov 2022), info sector per capita "
            f"changed {info_post_ai:+.1f}%",
            "USINFO", info_post_ai, "value", "change_pct",
            chatgpt, latest, per_capita=True,
        ),
        _make_claim(
            f"Trades per capita changed {trades_post_ai:+.1f}% since "
            "ChatGPT launch",
            "CES2023800001", trades_post_ai, "value", "change_pct",
            chatgpt, latest, per_capita=True,
        ),
    ]


def _claims_employment_growth(
    conn: sqlite3.Connection,
) -> list[dict[str, Any]]:
    """Build verifiable claims for rolling employment growth trends.

    Args:
        conn: SQLite database connection.

    Returns:
        List of structured claim dicts.
    """
    latest: str = _latest_month(conn, "USINFO")
    two_yr_ago: str = _month_offset(latest, -24)

    info_24m: float = _compute_change(
        conn, "USINFO", two_yr_ago, latest, per_capita=True,
    )
    trades_24m: float = _compute_change(
        conn, "CES2023800001", two_yr_ago, latest, per_capita=True,
    )
    info_pct: float = _compute_pct_of_start(
        conn, "USINFO", two_yr_ago, latest, per_capita=True,
    )

    return [
        _make_claim(
            f"Info sector per-capita employment changed "
            f"{info_24m:+.2f}% over the past 24 months",
            "USINFO", info_24m, "value", "change_pct",
            two_yr_ago, latest, per_capita=True,
        ),
        _make_claim(
            f"Trades per-capita employment changed "
            f"{trades_24m:+.2f}% over the past 24 months",
            "CES2023800001", trades_24m, "value", "change_pct",
            two_yr_ago, latest, per_capita=True,
        ),
        _make_claim(
            f"Info sector per-capita now at {info_pct}% of its "
            "level 24 months ago",
            "USINFO", info_pct, "value", "pct_of_start",
            two_yr_ago, latest, per_capita=True,
        ),
    ]


def _claims_covid_recovery(
    conn: sqlite3.Connection,
) -> list[dict[str, Any]]:
    """Build verifiable claims for COVID recovery comparison.

    Args:
        conn: SQLite database connection.

    Returns:
        List of structured claim dicts.
    """
    latest: str = _latest_month(conn, "USINFO")
    pre_covid: str = "2020-02"

    info_pct: float = _compute_pct_of_start(
        conn, "USINFO", pre_covid, latest, use_raw=True,
    )
    trades_pct: float = _compute_pct_of_start(
        conn, "CES2023800001", pre_covid, latest, use_raw=True,
    )
    info_chg: float = _compute_change(
        conn, "USINFO", pre_covid, latest, use_raw=True,
    )
    trades_chg: float = _compute_change(
        conn, "CES2023800001", pre_covid, latest, use_raw=True,
    )

    return [
        _make_claim(
            f"Info sector at {info_pct}% of Feb 2020 pre-COVID peak",
            "USINFO", info_pct, "value", "pct_of_start",
            pre_covid, latest, use_raw=True,
        ),
        _make_claim(
            f"Trades at {trades_pct}% of Feb 2020 peak, "
            f"{'surpassing' if trades_pct > 100 else 'below'} "
            "pre-COVID levels",
            "CES2023800001", trades_pct, "value", "pct_of_start",
            pre_covid, latest, use_raw=True,
        ),
        _make_claim(
            f"Info employment changed {info_chg:+.1f}% since "
            "Feb 2020",
            "USINFO", info_chg, "value", "change_pct",
            pre_covid, latest, use_raw=True,
        ),
        _make_claim(
            f"Trades employment changed {trades_chg:+.1f}% since "
            "Feb 2020",
            "CES2023800001", trades_chg, "value", "change_pct",
            pre_covid, latest, use_raw=True,
        ),
    ]


def _claims_u6_u3_gap(
    conn: sqlite3.Connection,
) -> list[dict[str, Any]]:
    """Build verifiable claims for U6 vs U3 unemployment gap.

    Args:
        conn: SQLite database connection.

    Returns:
        List of structured claim dicts.
    """
    latest: str = _latest_month(conn, "UNRATE")
    yr_ago: str = _month_offset(latest, -12)
    earliest: str = _earliest_month(conn, "U6RATE")

    u3_chg: float = _compute_change(conn, "UNRATE", yr_ago, latest)
    u6_chg: float = _compute_change(conn, "U6RATE", yr_ago, latest)
    u6_above_count: int = conn.execute(
        "SELECT COUNT(*) FROM ("
        "  SELECT SUBSTR(date, 1, 7) AS m, "
        "    AVG(value_covid_adjusted) AS avg_v "
        "  FROM observations WHERE series_id = 'U6RATE' "
        "  GROUP BY m HAVING avg_v > 7.0"
        ")",
    ).fetchone()[0]

    return [
        _make_claim(
            f"U3 changed {u3_chg:+.2f}% over the past 12 months",
            "UNRATE", u3_chg, "value", "change_pct",
            yr_ago, latest,
        ),
        _make_claim(
            f"U6 changed {u6_chg:+.2f}% over the past 12 months",
            "U6RATE", u6_chg, "value", "change_pct",
            yr_ago, latest,
        ),
        _make_claim(
            f"U6 exceeded 7% in {u6_above_count} months across "
            "the dataset",
            "U6RATE", u6_above_count, "count",
            "count_months_above", earliest, latest,
            threshold=7.0,
        ),
    ]


def _claims_power_vs_info(
    conn: sqlite3.Connection,
) -> list[dict[str, Any]]:
    """Build verifiable claims for power vs info since ChatGPT launch.

    Args:
        conn: SQLite database connection.

    Returns:
        List of structured claim dicts.
    """
    latest: str = _latest_month(conn, "IPG2211S")
    chatgpt: str = CHATGPT_LAUNCH

    power_chg: float = _compute_change(
        conn, "IPG2211S", chatgpt, latest,
    )
    info_chg: float = _compute_change(
        conn, "USINFO", chatgpt, latest,
    )
    power_pct: float = _compute_pct_of_start(
        conn, "IPG2211S", chatgpt, latest,
    )
    info_pct: float = _compute_pct_of_start(
        conn, "USINFO", chatgpt, latest,
    )

    return [
        _make_claim(
            f"Power output changed {power_chg:+.2f}% since ChatGPT "
            "launch (Nov 2022)",
            "IPG2211S", power_chg, "value", "change_pct",
            chatgpt, latest,
        ),
        _make_claim(
            f"Info employment changed {info_chg:+.2f}% over the "
            "same period",
            "USINFO", info_chg, "value", "change_pct",
            chatgpt, latest,
        ),
        _make_claim(
            f"Power output at {power_pct}% of Nov 2022 level",
            "IPG2211S", power_pct, "value", "pct_of_start",
            chatgpt, latest,
        ),
        _make_claim(
            f"Info employment at {info_pct}% of Nov 2022 level",
            "USINFO", info_pct, "value", "pct_of_start",
            chatgpt, latest,
        ),
    ]


def _claims_dashboard_intro(
    conn: sqlite3.Connection,
) -> list[dict[str, Any]]:
    """Build verifiable claims for the dashboard intro insight.

    Args:
        conn: SQLite database connection.

    Returns:
        List of structured claim dicts.
    """
    latest_unrate: str = _latest_month(conn, "UNRATE")
    latest_gdp: str = _latest_month(conn, "GDPC1")
    latest_t10y2y: str = _latest_month(conn, "T10Y2Y")
    earliest: str = _earliest_month(conn, "UNRATE")

    unrate_val: float | None = _boundary_value(
        conn, "UNRATE", earliest, latest_unrate, "end",
    )
    t10y2y_val: float | None = _boundary_value(
        conn, "T10Y2Y", earliest, latest_t10y2y, "end",
    )
    gdp_yr_ago: str = _month_offset(latest_gdp, -12)
    gdp_chg: float = _compute_change(conn, "GDPC1", gdp_yr_ago, latest_gdp)

    total_months: int = conn.execute(
        "SELECT COUNT(DISTINCT SUBSTR(date, 1, 7)) FROM observations"
    ).fetchone()[0]

    claims: list[dict[str, Any]] = [
        _make_claim(
            f"Unemployment at {unrate_val:.1f}%",
            "UNRATE", round(unrate_val or 0, 1), "value", "latest",
            latest_unrate, latest_unrate,
        ),
        _make_claim(
            f"GDP changed {gdp_chg:+.2f}% over the past year",
            "GDPC1", gdp_chg, "value", "change_pct",
            gdp_yr_ago, latest_gdp,
        ),
        _make_claim(
            f"Yield spread at {t10y2y_val:.2f}%, "
            f"{'inverted' if (t10y2y_val or 0) < 0 else 'normal'}",
            "T10Y2Y", round(t10y2y_val or 0, 2), "value", "latest",
            latest_t10y2y, latest_t10y2y,
        ),
    ]
    return claims


def _claims_overview(
    conn: sqlite3.Connection,
) -> list[dict[str, Any]]:
    """Build verifiable claims for the overview KPI synthesis insight.

    Args:
        conn: SQLite database connection.

    Returns:
        List of structured claim dicts.
    """
    latest_unrate: str = _latest_month(conn, "UNRATE")
    latest_gdp: str = _latest_month(conn, "GDPC1")
    latest_cpi: str = _latest_month(conn, "CPIAUCSL")
    yr_ago_unrate: str = _month_offset(latest_unrate, -1)
    yr_ago_cpi: str = _month_offset(latest_cpi, -12)

    unrate_chg: float = _compute_change(
        conn, "UNRATE", yr_ago_unrate, latest_unrate,
    )
    u6_chg: float = _compute_change(
        conn, "U6RATE", yr_ago_unrate, latest_unrate,
    )
    gdp_yr_ago: str = _month_offset(latest_gdp, -12)
    gdp_chg: float = _compute_change(conn, "GDPC1", gdp_yr_ago, latest_gdp)
    cpi_yoy: float = _compute_change(
        conn, "CPIAUCSL", yr_ago_cpi, latest_cpi,
    )

    return [
        _make_claim(
            f"UNRATE MoM change: {unrate_chg:+.2f}%",
            "UNRATE", unrate_chg, "value", "change_pct",
            yr_ago_unrate, latest_unrate,
        ),
        _make_claim(
            f"U6RATE MoM change: {u6_chg:+.2f}%",
            "U6RATE", u6_chg, "value", "change_pct",
            yr_ago_unrate, latest_unrate,
        ),
        _make_claim(
            f"GDP growth {'positive' if gdp_chg > 0 else 'negative'} "
            f"at {gdp_chg:+.2f}%",
            "GDPC1", gdp_chg, "value", "change_pct",
            gdp_yr_ago, latest_gdp,
        ),
        _make_claim(
            f"CPI YoY at {cpi_yoy:.2f}%",
            "CPIAUCSL", cpi_yoy, "value", "change_pct",
            yr_ago_cpi, latest_cpi,
        ),
    ]


def _claims_recession_risk(
    conn: sqlite3.Connection,
) -> list[dict[str, Any]]:
    """Build verifiable claims for the recession risk timeline insight.

    Args:
        conn: SQLite database connection.

    Returns:
        List of structured claim dicts.
    """
    latest_row = conn.execute(
        "SELECT date, probability FROM recession_predictions "
        "ORDER BY date DESC LIMIT 1"
    ).fetchone()
    latest_date: str = latest_row[0][:7]
    latest_prob: float = round(latest_row[1], 4)

    yr_ago_date: str = _month_offset(latest_date, -12)
    yr_ago_row = conn.execute(
        "SELECT probability FROM recession_predictions "
        "WHERE date <= ? ORDER BY date DESC LIMIT 1",
        (f"{yr_ago_date}-31",),
    ).fetchone()
    yr_ago_prob: float = round(yr_ago_row[0], 4) if yr_ago_row else 0.0

    above_count: int = conn.execute(
        "SELECT COUNT(*) FROM recession_predictions WHERE probability > 0.5"
    ).fetchone()[0]

    # 6-month trend direction
    six_ago: str = _month_offset(latest_date, -6)
    six_row = conn.execute(
        "SELECT probability FROM recession_predictions "
        "WHERE date <= ? ORDER BY date DESC LIMIT 1",
        (f"{six_ago}-31",),
    ).fetchone()
    six_prob: float = six_row[0] if six_row else latest_prob
    trend_dir: float = 1.0 if latest_prob > six_prob else -1.0

    return [
        _make_claim(
            f"Latest recession risk score: {latest_prob:.4f}",
            "recession_risk", latest_prob, "value", "prediction_latest",
            latest_date, latest_date,
        ),
        _make_claim(
            f"Risk score 12 months ago: {yr_ago_prob:.4f}",
            "recession_risk", yr_ago_prob, "value", "prediction_at_date",
            yr_ago_date, yr_ago_date,
        ),
        _make_claim(
            f"{above_count} months above 0.5 threshold",
            "recession_risk", above_count, "count",
            "prediction_count_above",
            period_start="", period_end="",
            threshold=0.5,
        ),
        _make_claim(
            f"6-month trend: {'rising' if trend_dir > 0 else 'falling'}",
            "recession_risk", trend_dir, "trend", "prediction_direction",
            six_ago, latest_date,
        ),
    ]


def _claims_feature_snapshot(
    conn: sqlite3.Connection,
) -> list[dict[str, Any]]:
    """Build verifiable claims for the feature contribution snapshot insight.

    Args:
        conn: SQLite database connection.

    Returns:
        List of structured claim dicts.
    """
    latest_row = conn.execute(
        "SELECT features_json FROM recession_predictions "
        "ORDER BY date DESC LIMIT 1"
    ).fetchone()
    features: dict[str, float] = json.loads(latest_row[0])

    red_count: int = 0
    strongest_red: str = ""
    strongest_red_val: float = 0.0
    strongest_green: str = ""
    strongest_green_val: float = 0.0

    for rule in SIGNAL_RULES:
        feat: str = rule["feature"]
        val: float = features.get(feat, 0.0)
        is_red: bool = (
            (rule["condition"] == "lt" and val < rule["threshold"])
            or (rule["condition"] == "gt" and val > rule["threshold"])
        )
        if is_red:
            red_count += 1
            if abs(val) > abs(strongest_red_val):
                strongest_red = feat
                strongest_red_val = val
        else:
            if abs(val) > abs(strongest_green_val):
                strongest_green = feat
                strongest_green_val = val

    return [
        _make_claim(
            f"{red_count} features flashing recession signals",
            "feature_snapshot", red_count, "count",
            "feature_red_count",
        ),
        _make_claim(
            f"Strongest recession signal: {strongest_red}",
            "feature_snapshot", strongest_red_val, "value",
            "feature_value",
        ),
        _make_claim(
            f"Strongest healthy signal: {strongest_green}",
            "feature_snapshot", strongest_green_val, "value",
            "feature_value",
        ),
    ]


def _claims_scenario_explorer(
    conn: sqlite3.Connection,
) -> list[dict[str, Any]]:
    """Build verifiable claims for the scenario explorer insight.

    Args:
        conn: SQLite database connection.

    Returns:
        List of structured claim dicts.
    """
    # Baseline: latest prediction probability
    baseline_row = conn.execute(
        "SELECT probability FROM recession_predictions "
        "ORDER BY date DESC LIMIT 1"
    ).fetchone()
    baseline: float = round(baseline_row[0], 4)

    extremes = conn.execute(
        "SELECT MIN(probability), MAX(probability) FROM scenario_grid"
    ).fetchone()
    grid_min: float = round(extremes[0], 6)
    grid_max: float = round(extremes[1], 6)

    return [
        _make_claim(
            f"Baseline scenario risk score: {baseline:.4f}",
            "scenario_explorer", baseline, "value", "prediction_latest",
        ),
        _make_claim(
            f"Lowest risk in scenario grid: {grid_min:.6f}",
            "scenario_explorer", grid_min, "value", "scenario_min",
        ),
        _make_claim(
            f"Highest risk in scenario grid: {grid_max:.6f}",
            "scenario_explorer", grid_max, "value", "scenario_max",
        ),
    ]


def _claims_deep_divergence(
    conn: sqlite3.Connection,
) -> list[dict[str, Any]]:
    """Build verifiable claims for the deep dive employment divergence.

    Focuses on the post-ChatGPT window (Nov 2022 onward).

    Args:
        conn: SQLite database connection.

    Returns:
        List of structured claim dicts.
    """
    latest: str = _latest_month(conn, "USINFO")
    chatgpt: str = CHATGPT_LAUNCH

    info_pct: float = _compute_pct_of_start(
        conn, "USINFO", chatgpt, latest, per_capita=True,
    )
    trades_pct: float = _compute_pct_of_start(
        conn, "CES2023800001", chatgpt, latest, per_capita=True,
    )
    info_chg: float = _compute_change(
        conn, "USINFO", chatgpt, latest, per_capita=True,
    )
    gap: float = round(trades_pct - info_pct, 2)

    return [
        _make_claim(
            f"Info sector per-capita index at {info_pct}% of Nov 2022 level",
            "USINFO", info_pct, "value", "pct_of_start",
            chatgpt, latest, per_capita=True,
        ),
        _make_claim(
            f"Trades per-capita index at {trades_pct}% of Nov 2022 level",
            "CES2023800001", trades_pct, "value", "pct_of_start",
            chatgpt, latest, per_capita=True,
        ),
        _make_claim(
            f"Post-ChatGPT info sector per-capita change: {info_chg:+.2f}%",
            "USINFO", info_chg, "value", "change_pct",
            chatgpt, latest, per_capita=True,
        ),
    ]


def _claims_deep_energy(
    conn: sqlite3.Connection,
) -> list[dict[str, Any]]:
    """Build verifiable claims for the deep dive energy paradox.

    Focuses on the post-ChatGPT window (Nov 2022 onward).

    Args:
        conn: SQLite database connection.

    Returns:
        List of structured claim dicts.
    """
    latest: str = _latest_month(conn, "IPG2211S")
    chatgpt: str = CHATGPT_LAUNCH

    power_pct: float = _compute_pct_of_start(
        conn, "IPG2211S", chatgpt, latest,
    )
    info_pct: float = _compute_pct_of_start(
        conn, "USINFO", chatgpt, latest,
    )
    power_chg: float = _compute_change(
        conn, "IPG2211S", chatgpt, latest,
    )
    info_chg: float = _compute_change(
        conn, "USINFO", chatgpt, latest,
    )

    return [
        _make_claim(
            f"Power output at {power_pct}% of Nov 2022 level",
            "IPG2211S", power_pct, "value", "pct_of_start",
            chatgpt, latest,
        ),
        _make_claim(
            f"Info employment at {info_pct}% of Nov 2022 level",
            "USINFO", info_pct, "value", "pct_of_start",
            chatgpt, latest,
        ),
        _make_claim(
            f"Power output changed {power_chg:+.2f}% post-ChatGPT",
            "IPG2211S", power_chg, "value", "change_pct",
            chatgpt, latest,
        ),
        _make_claim(
            f"Info employment changed {info_chg:+.2f}% post-ChatGPT",
            "USINFO", info_chg, "value", "change_pct",
            chatgpt, latest,
        ),
    ]


def _claims_deep_synthesis_charts(
    conn: sqlite3.Connection,
) -> list[dict[str, Any]]:
    """Build verifiable claims for the deep dive synthesis insight.

    Connects employment divergence and energy paradox narratives.

    Args:
        conn: SQLite database connection.

    Returns:
        List of structured claim dicts.
    """
    latest: str = _latest_month(conn, "USINFO")
    chatgpt: str = CHATGPT_LAUNCH

    info_pct: float = _compute_pct_of_start(
        conn, "USINFO", chatgpt, latest, per_capita=True,
    )
    trades_pct: float = _compute_pct_of_start(
        conn, "CES2023800001", chatgpt, latest, per_capita=True,
    )
    divergence_gap: float = round(trades_pct - info_pct, 2)

    power_pct: float = _compute_pct_of_start(
        conn, "IPG2211S", chatgpt, latest,
    )
    power_info_gap: float = round(power_pct - info_pct, 2)

    return [
        _make_claim(
            f"Trades-info per-capita divergence: {divergence_gap:+.2f}pp "
            "since ChatGPT",
            "CES2023800001", trades_pct, "value", "pct_of_start",
            chatgpt, latest, per_capita=True,
        ),
        _make_claim(
            f"Power-info gap: {power_info_gap:+.2f}pp post-ChatGPT",
            "IPG2211S", power_pct, "value", "pct_of_start",
            chatgpt, latest,
        ),
        _make_claim(
            f"Info sector per-capita at {info_pct}% of Nov 2022",
            "USINFO", info_pct, "value", "pct_of_start",
            chatgpt, latest, per_capita=True,
        ),
    ]


def _claims_synthesis(
    conn: sqlite3.Connection,
) -> list[dict[str, Any]]:
    """Build verifiable claims for the cross-metric synthesis.

    Args:
        conn: SQLite database connection.

    Returns:
        List of structured claim dicts.
    """
    latest: str = _latest_month(conn, "UNRATE")
    yr_ago: str = _month_offset(latest, -12)
    earliest_t10: str = _earliest_month(conn, "T10Y2Y")
    earliest_info: str = _earliest_month(conn, "USINFO")

    inv_count: int = conn.execute(
        "SELECT COUNT(*) FROM ("
        "  SELECT SUBSTR(date, 1, 7) AS m, "
        "    AVG(value_covid_adjusted) AS avg_v "
        "  FROM observations WHERE series_id = 'T10Y2Y' "
        "  GROUP BY m HAVING avg_v < 0"
        ")",
    ).fetchone()[0]
    info_chg: float = _compute_change(
        conn, "USINFO", earliest_info, latest, per_capita=True,
    )
    trades_chg: float = _compute_change(
        conn, "CES2023800001", earliest_info, latest, per_capita=True,
    )
    cpi_yoy: float = _compute_change(
        conn, "CPIAUCSL", yr_ago, latest,
    )
    unrate_chg: float = _compute_change(
        conn, "UNRATE", yr_ago, latest,
    )

    return [
        _make_claim(
            f"Yield curve inverted in {inv_count} months, signaling "
            "sustained recession risk",
            "T10Y2Y", inv_count, "count", "count_months_below",
            earliest_t10, latest, threshold=0,
        ),
        _make_claim(
            f"Info sector shrank {info_chg:+.1f}% per capita while "
            f"trades grew {trades_chg:+.1f}%",
            "USINFO", info_chg, "value", "change_pct",
            earliest_info, latest, per_capita=True,
        ),
        _make_claim(
            f"CPI inflation running at {cpi_yoy}% year-over-year",
            "CPIAUCSL", cpi_yoy, "value", "change_pct",
            yr_ago, latest,
        ),
        _make_claim(
            f"Unemployment changed {unrate_chg:+.2f}% over the "
            "past 12 months",
            "UNRATE", unrate_chg, "value", "change_pct",
            yr_ago, latest,
        ),
    ]


def _fmt_series(series: pd.Series, decimals: int = 1) -> str:
    """Format a pandas Series tail as a comma-separated string.

    Args:
        series: Numeric pandas Series.
        decimals: Number of decimal places.

    Returns:
        Comma-separated string of the last 6 values.
    """
    fmt: str = f"{{:.{decimals}f}}"
    return ", ".join(fmt.format(v) for v in series.tail(6))


# ---------------------------------------------------------------------------
# Context functions — one per insight slice
# ---------------------------------------------------------------------------


def _context_yield_curve_unemployment(
    conn: sqlite3.Connection,
) -> dict[str, Any]:
    """Build data context for yield curve vs unemployment correlation.

    Args:
        conn: SQLite database connection.

    Returns:
        Dict with 'context_text' for the LLM prompt and 'data_points'.
    """
    spread_df: pd.DataFrame = pd.read_sql_query(
        "SELECT SUBSTR(date, 1, 7) AS month, "
        "AVG(value_covid_adjusted) AS spread "
        "FROM observations WHERE series_id = 'T10Y2Y' "
        "GROUP BY SUBSTR(date, 1, 7) ORDER BY month",
        conn,
    )
    unrate_df: pd.DataFrame = pd.read_sql_query(
        "SELECT SUBSTR(date, 1, 7) AS month, "
        "value_covid_adjusted AS unrate "
        "FROM observations WHERE series_id = 'UNRATE' ORDER BY month",
        conn,
    )

    current_spread: float = float(spread_df["spread"].iloc[-1])
    total_inv: int = int((spread_df["spread"] < 0).sum())

    # Longest consecutive inversion streak
    inverted: pd.Series = spread_df["spread"] < 0
    groups: pd.Series = inverted.ne(inverted.shift()).cumsum()
    streaks: pd.Series = spread_df[inverted].groupby(groups).size()
    longest_inv: int = int(streaks.max()) if len(streaks) > 0 else 0

    current_unrate: float = float(unrate_df["unrate"].iloc[-1])
    unrate_12m_ago: float = float(
        unrate_df["unrate"].iloc[-13]
        if len(unrate_df) >= 13
        else unrate_df["unrate"].iloc[0]
    )
    unrate_change: float = current_unrate - unrate_12m_ago

    context_text: str = (
        f"Yield Curve (T10Y2Y) and Unemployment (UNRATE) data:\n"
        f"- Current monthly avg yield spread: {current_spread:.2f}% "
        f"(rate-type; describe changes in pp)\n"
        f"- Total months with inverted curve (spread < 0): {total_inv}\n"
        f"- Longest consecutive inversion: {longest_inv} months\n"
        f"- Current UNRATE: {current_unrate:.1f}% (rate-type; describe "
        f"changes in pp)\n"
        f"- UNRATE 12 months ago: {unrate_12m_ago:.1f}%\n"
        f"- 12-month UNRATE change: "
        f"{_fmt_change('UNRATE', unrate_12m_ago, current_unrate)}\n"
        f"- Recent spreads (last 6 months): "
        f"{_fmt_series(spread_df['spread'], 2)}\n"
        f"- Recent UNRATE (last 6 months): "
        f"{_fmt_series(unrate_df['unrate'])}"
    )

    return {
        "context_text": context_text,
        "data_points": {
            "current_spread": round(current_spread, 2),
            "total_inversion_months": total_inv,
            "longest_inversion": longest_inv,
            "current_unrate": round(current_unrate, 1),
            "unrate_12m_change": round(unrate_change, 1),
        },
    }


def _context_gdp_trend(conn: sqlite3.Connection) -> dict[str, Any]:
    """Build data context for GDP growth trend.

    Args:
        conn: SQLite database connection.

    Returns:
        Dict with 'context_text' and 'data_points'.
    """
    gdp_df: pd.DataFrame = pd.read_sql_query(
        "SELECT date, value_covid_adjusted AS gdp "
        "FROM observations WHERE series_id = 'GDPC1' ORDER BY date",
        conn,
    )
    gdp_df["growth"] = gdp_df["gdp"].pct_change() * 400  # annualized QoQ

    rec_df: pd.DataFrame = pd.read_sql_query(
        "SELECT SUBSTR(date, 1, 7) AS month, "
        "value_covid_adjusted AS recession "
        "FROM observations WHERE series_id = 'USREC' ORDER BY month",
        conn,
    )

    latest_growth: float = float(gdp_df["growth"].iloc[-1])
    tail_20: pd.Series = gdp_df["growth"].tail(20)
    avg_growth_5y: float = float(tail_20.mean())
    negative_quarters: int = int((tail_20 < 0).sum())
    current_recession: bool = bool(rec_df["recession"].iloc[-1] > 0)

    context_text: str = (
        f"Real GDP (GDPC1) trend data:\n"
        f"- Latest quarter annualized growth: {latest_growth:.1f}%\n"
        f"- Average annualized growth (last 20 quarters): "
        f"{avg_growth_5y:.1f}%\n"
        f"- Negative growth quarters (last 20): {negative_quarters}\n"
        f"- Currently in NBER recession: "
        f"{'Yes' if current_recession else 'No'}\n"
        f"- Recent quarterly growth rates: "
        f"{_fmt_series(gdp_df['growth'].dropna())}"
    )

    return {
        "context_text": context_text,
        "data_points": {
            "latest_growth": round(latest_growth, 1),
            "avg_growth_5y": round(avg_growth_5y, 1),
            "negative_quarters": negative_quarters,
            "current_recession": current_recession,
        },
    }


def _context_cpi_trend(conn: sqlite3.Connection) -> dict[str, Any]:
    """Build data context for CPI inflation trend.

    Args:
        conn: SQLite database connection.

    Returns:
        Dict with 'context_text' and 'data_points'.
    """
    cpi_df: pd.DataFrame = pd.read_sql_query(
        "SELECT date, value_covid_adjusted AS cpi "
        "FROM observations WHERE series_id = 'CPIAUCSL' ORDER BY date",
        conn,
    )
    cpi_df["mom"] = cpi_df["cpi"].pct_change() * 100
    cpi_df["yoy"] = cpi_df["cpi"].pct_change(12) * 100

    latest_mom: float = float(cpi_df["mom"].iloc[-1])
    latest_yoy: float = float(cpi_df["yoy"].iloc[-1])
    peak_yoy: float = float(cpi_df["yoy"].max())
    avg_yoy_2y: float = float(cpi_df["yoy"].tail(24).mean())

    context_text: str = (
        f"CPI Inflation (CPIAUCSL) data:\n"
        f"- Latest month-over-month change on the CPI level: "
        f"{latest_mom:.2f}% (level-type; already a percent change)\n"
        f"- Latest year-over-year rate: {latest_yoy:.1f}% (rate-type; "
        f"describe changes to this rate in percentage points)\n"
        f"- Peak YoY inflation in dataset: {peak_yoy:.1f}%\n"
        f"- Average YoY rate (last 24 months): {avg_yoy_2y:.1f}%\n"
        f"- Recent MoM (last 6): {_fmt_series(cpi_df['mom'], 2)}\n"
        f"- Recent YoY (last 6): "
        f"{_fmt_series(cpi_df['yoy'].dropna())}"
    )

    return {
        "context_text": context_text,
        "data_points": {
            "latest_mom": round(latest_mom, 2),
            "latest_yoy": round(latest_yoy, 1),
            "peak_yoy": round(peak_yoy, 1),
            "avg_yoy_2y": round(avg_yoy_2y, 1),
        },
    }


def _context_info_vs_trades(conn: sqlite3.Connection) -> dict[str, Any]:
    """Build data context for info vs trades per-capita comparison.

    Args:
        conn: SQLite database connection.

    Returns:
        Dict with 'context_text' and 'data_points'.
    """
    df: pd.DataFrame = pd.read_sql_query(
        "SELECT SUBSTR(e.date, 1, 7) AS month, e.series_id, "
        "e.value_covid_adjusted / p.value_covid_adjusted * 1000 AS per_capita "
        "FROM observations e "
        "JOIN observations p ON p.series_id = 'CNP16OV' "
        "  AND SUBSTR(p.date, 1, 7) = SUBSTR(e.date, 1, 7) "
        "WHERE e.series_id IN ('USINFO', 'CES2023800001') "
        "ORDER BY month",
        conn,
    )
    info: pd.DataFrame = df[df["series_id"] == "USINFO"].reset_index(drop=True)
    trades: pd.DataFrame = (
        df[df["series_id"] == "CES2023800001"].reset_index(drop=True)
    )

    info_start: float = float(info["per_capita"].iloc[0])
    info_latest: float = float(info["per_capita"].iloc[-1])
    trades_start: float = float(trades["per_capita"].iloc[0])
    trades_latest: float = float(trades["per_capita"].iloc[-1])

    info_chg: float = (info_latest - info_start) / info_start * 100
    trades_chg: float = (trades_latest - trades_start) / trades_start * 100
    divergence: float = trades_chg - info_chg

    context_text: str = (
        f"Info Sector (USINFO) vs Specialty Trades (CES2023800001), "
        f"per-capita normalized (per 1K working-age):\n"
        f"- Info start ({info['month'].iloc[0]}): {info_start:.2f}\n"
        f"- Info latest ({info['month'].iloc[-1]}): {info_latest:.2f}\n"
        f"- Info change: {info_chg:+.1f}%\n"
        f"- Trades start: {trades_start:.2f}\n"
        f"- Trades latest: {trades_latest:.2f}\n"
        f"- Trades change: {trades_chg:+.1f}%\n"
        f"- Divergence (trades - info change): {divergence:+.1f} pp"
    )

    return {
        "context_text": context_text,
        "data_points": {
            "info_start": round(info_start, 2),
            "info_latest": round(info_latest, 2),
            "info_change_pct": round(info_chg, 1),
            "trades_start": round(trades_start, 2),
            "trades_latest": round(trades_latest, 2),
            "trades_change_pct": round(trades_chg, 1),
            "divergence": round(divergence, 1),
            "start_month": str(info["month"].iloc[0]),
            "latest_month": str(info["month"].iloc[-1]),
        },
    }


def _context_employment_growth(conn: sqlite3.Connection) -> dict[str, Any]:
    """Build data context for rolling 12-month per-capita employment growth.

    Args:
        conn: SQLite database connection.

    Returns:
        Dict with 'context_text' and 'data_points'.
    """
    df: pd.DataFrame = pd.read_sql_query(
        "SELECT SUBSTR(e.date, 1, 7) AS month, e.series_id, "
        "e.value_covid_adjusted / p.value_covid_adjusted * 1000 AS per_capita "
        "FROM observations e "
        "JOIN observations p ON p.series_id = 'CNP16OV' "
        "  AND SUBSTR(p.date, 1, 7) = SUBSTR(e.date, 1, 7) "
        "WHERE e.series_id IN ('USINFO', 'CES2023800001') "
        "ORDER BY month",
        conn,
    )

    results: dict[str, dict[str, float]] = {}
    for sid in ("USINFO", "CES2023800001"):
        sdf: pd.DataFrame = df[df["series_id"] == sid].reset_index(drop=True)
        sdf["yoy"] = sdf["per_capita"].pct_change(12) * 100
        tail: pd.Series = sdf["yoy"].tail(24)
        results[sid] = {
            "latest_growth": round(float(sdf["yoy"].iloc[-1]), 1)
            if len(sdf) > 12
            else 0.0,
            "negative_months_24": int((tail < 0).sum()),
            "avg_growth_24m": round(float(tail.mean()), 1),
        }

    context_text: str = (
        f"Rolling 12-Month Per-Capita Employment Growth:\n"
        f"- USINFO latest YoY: "
        f"{results['USINFO']['latest_growth']:+.1f}%\n"
        f"- USINFO negative months (last 24): "
        f"{results['USINFO']['negative_months_24']}\n"
        f"- USINFO avg growth (24m): "
        f"{results['USINFO']['avg_growth_24m']:+.1f}%\n"
        f"- CES2023800001 latest YoY: "
        f"{results['CES2023800001']['latest_growth']:+.1f}%\n"
        f"- CES2023800001 negative months (last 24): "
        f"{results['CES2023800001']['negative_months_24']}\n"
        f"- CES2023800001 avg growth (24m): "
        f"{results['CES2023800001']['avg_growth_24m']:+.1f}%\n"
        f"Note: All values are per-capita (divided by CNP16OV, x1000)."
    )

    return {"context_text": context_text, "data_points": results}


def _context_covid_recovery(conn: sqlite3.Connection) -> dict[str, Any]:
    """Build data context for COVID recovery comparison using raw values.

    Args:
        conn: SQLite database connection.

    Returns:
        Dict with 'context_text' and 'data_points'.
    """
    feb_2020: pd.DataFrame = pd.read_sql_query(
        "SELECT series_id, value AS peak "
        "FROM observations "
        "WHERE series_id IN ('USINFO', 'CES2023800001') "
        "  AND SUBSTR(date, 1, 7) = '2020-02'",
        conn,
    )
    peaks: dict[str, float] = dict(
        zip(feb_2020["series_id"], feb_2020["peak"])
    )

    latest: pd.DataFrame = pd.read_sql_query(
        "SELECT series_id, value AS latest_val, date "
        "FROM observations "
        "WHERE series_id IN ('USINFO', 'CES2023800001') "
        "  AND date = ("
        "    SELECT MAX(date) FROM observations "
        "    WHERE series_id = 'USINFO'"
        "  )",
        conn,
    )
    latest_vals: dict[str, float] = dict(
        zip(latest["series_id"], latest["latest_val"])
    )
    latest_date: str = (
        str(latest["date"].iloc[0])[:7] if not latest.empty else "unknown"
    )

    info_peak: float = peaks.get("USINFO", 1)
    trades_peak: float = peaks.get("CES2023800001", 1)
    info_latest: float = latest_vals.get("USINFO", 0)
    trades_latest: float = latest_vals.get("CES2023800001", 0)
    info_recovery: float = info_latest / info_peak * 100
    trades_recovery: float = trades_latest / trades_peak * 100

    context_text: str = (
        f"COVID Recovery (raw values, % of Feb 2020 peak):\n"
        f"- USINFO peak (Feb 2020): {info_peak:.0f} thousand\n"
        f"- USINFO latest ({latest_date}): {info_latest:.0f} thousand\n"
        f"- USINFO recovery: {info_recovery:.1f}% of peak\n"
        f"- CES2023800001 peak (Feb 2020): {trades_peak:.0f} thousand\n"
        f"- CES2023800001 latest ({latest_date}): "
        f"{trades_latest:.0f} thousand\n"
        f"- CES2023800001 recovery: {trades_recovery:.1f}% of peak\n"
        f"Note: Uses raw values (not COVID-adjusted) to show actual impact."
    )

    return {
        "context_text": context_text,
        "data_points": {
            "info_peak": info_peak,
            "info_latest": info_latest,
            "info_recovery_pct": round(info_recovery, 1),
            "trades_peak": trades_peak,
            "trades_latest": trades_latest,
            "trades_recovery_pct": round(trades_recovery, 1),
            "latest_date": latest_date,
        },
    }


def _context_u6_u3_gap(conn: sqlite3.Connection) -> dict[str, Any]:
    """Build data context for U6 vs U3 unemployment gap.

    Args:
        conn: SQLite database connection.

    Returns:
        Dict with 'context_text' and 'data_points'.
    """
    df: pd.DataFrame = pd.read_sql_query(
        "SELECT u3.date, "
        "  u3.value_covid_adjusted AS u3, "
        "  u6.value_covid_adjusted AS u6, "
        "  u6.value_covid_adjusted - u3.value_covid_adjusted AS gap "
        "FROM observations u3 "
        "JOIN observations u6 "
        "  ON u6.series_id = 'U6RATE' AND u3.date = u6.date "
        "WHERE u3.series_id = 'UNRATE' ORDER BY u3.date",
        conn,
    )

    current_u3: float = float(df["u3"].iloc[-1])
    current_u6: float = float(df["u6"].iloc[-1])
    current_gap: float = float(df["gap"].iloc[-1])

    df["u3_yoy"] = df["u3"].diff(12)
    df["u6_yoy"] = df["u6"].diff(12)
    recent: pd.DataFrame = df.tail(24).dropna(subset=["u3_yoy", "u6_yoy"])
    u6_faster: int = int((recent["u6_yoy"] > recent["u3_yoy"]).sum())
    total_recent: int = len(recent)

    context_text: str = (
        f"U6 vs U3 Unemployment Gap (both are rate-type series; describe "
        f"changes in percentage points, not relative percent):\n"
        f"- Current U3 (UNRATE): {current_u3:.1f}%\n"
        f"- Current U6 (U6RATE): {current_u6:.1f}%\n"
        f"- Current gap (U6 - U3): {current_gap:.1f} pp\n"
        f"- Months where U6 YoY change > U3 YoY change (last 24): "
        f"{u6_faster}/{total_recent}\n"
        f"- Recent U3 (last 6): {_fmt_series(df['u3'])}\n"
        f"- Recent U6 (last 6): {_fmt_series(df['u6'])}\n"
        f"- Recent gap (last 6): {_fmt_series(df['gap'])}"
    )

    return {
        "context_text": context_text,
        "data_points": {
            "current_u3": round(current_u3, 1),
            "current_u6": round(current_u6, 1),
            "current_gap": round(current_gap, 1),
            "u6_faster_months": u6_faster,
            "total_recent_months": total_recent,
        },
    }


def _context_power_vs_info(conn: sqlite3.Connection) -> dict[str, Any]:
    """Build data context for electric power vs info employment correlation.

    Args:
        conn: SQLite database connection.

    Returns:
        Dict with 'context_text' and 'data_points'.
    """
    df: pd.DataFrame = pd.read_sql_query(
        "SELECT SUBSTR(p.date, 1, 7) AS month, "
        "  p.value_covid_adjusted AS power_val, "
        "  i.value_covid_adjusted AS info_val "
        "FROM observations p "
        "JOIN observations i "
        "  ON i.series_id = 'USINFO' "
        "  AND SUBSTR(i.date, 1, 7) = SUBSTR(p.date, 1, 7) "
        "WHERE p.series_id = 'IPG2211S' ORDER BY month",
        conn,
    )

    power_base: float = float(df["power_val"].iloc[0])
    info_base: float = float(df["info_val"].iloc[0])
    df["power_idx"] = df["power_val"] / power_base * 100
    df["info_idx"] = df["info_val"] / info_base * 100

    # Post-ChatGPT (Nov 2022)
    post_mask: pd.Series = df["month"] >= CHATGPT_LAUNCH
    post: pd.DataFrame = df[post_mask]

    if len(post) > 0:
        power_change: float = (
            float(post["power_idx"].iloc[-1])
            - float(post["power_idx"].iloc[0])
        )
        info_change: float = (
            float(post["info_idx"].iloc[-1])
            - float(post["info_idx"].iloc[0])
        )
    else:
        power_change = info_change = 0.0

    context_text: str = (
        f"Electric Power (IPG2211S) vs Info Employment (USINFO):\n"
        f"- Both indexed to 100 at dataset start\n"
        f"- Latest power index: {float(df['power_idx'].iloc[-1]):.1f}\n"
        f"- Latest info index: {float(df['info_idx'].iloc[-1]):.1f}\n"
        f"- Post-ChatGPT (Nov 2022) power change: "
        f"{power_change:+.1f} pp\n"
        f"- Post-ChatGPT info change: {info_change:+.1f} pp\n"
        f"- Divergence since ChatGPT: "
        f"{power_change - info_change:+.1f} pp"
    )

    return {
        "context_text": context_text,
        "data_points": {
            "power_index_latest": round(
                float(df["power_idx"].iloc[-1]), 1
            ),
            "info_index_latest": round(
                float(df["info_idx"].iloc[-1]), 1
            ),
            "post_chatgpt_power_change": round(power_change, 1),
            "post_chatgpt_info_change": round(info_change, 1),
            "divergence": round(power_change - info_change, 1),
        },
    }


def _context_dashboard_intro(conn: sqlite3.Connection) -> dict[str, Any]:
    """Build data context for the dashboard intro macro landscape summary.

    Args:
        conn: SQLite database connection.

    Returns:
        Dict with 'context_text' and 'data_points'.
    """
    latest_unrate: str = _latest_month(conn, "UNRATE")
    unrate_val: float = float(
        _boundary_value(conn, "UNRATE", latest_unrate, latest_unrate, "end")
        or 0
    )
    gdp_latest: str = _latest_month(conn, "GDPC1")
    gdp_yr_ago: str = _month_offset(gdp_latest, -12)
    gdp_chg: float = _compute_change(conn, "GDPC1", gdp_yr_ago, gdp_latest)

    t10y2y_latest: str = _latest_month(conn, "T10Y2Y")
    t10y2y_val: float = float(
        _boundary_value(conn, "T10Y2Y", t10y2y_latest, t10y2y_latest, "end")
        or 0
    )
    cpi_latest: str = _latest_month(conn, "CPIAUCSL")
    cpi_yr_ago: str = _month_offset(cpi_latest, -12)
    cpi_yoy: float = _compute_change(
        conn, "CPIAUCSL", cpi_yr_ago, cpi_latest
    )

    total_months: int = conn.execute(
        "SELECT COUNT(DISTINCT SUBSTR(date, 1, 7)) FROM observations"
    ).fetchone()[0]

    context_text: str = (
        f"Dashboard Macro Landscape Summary:\n"
        f"- Latest UNRATE: {unrate_val:.1f}% ({latest_unrate}) "
        f"[rate-type series; describe changes in percentage points]\n"
        f"- GDP YoY change: {gdp_chg:+.2f}% ({gdp_latest}) "
        f"[already a percent change]\n"
        f"- Yield spread (T10Y2Y): {t10y2y_val:.2f}% "
        f"({'inverted' if t10y2y_val < 0 else 'normal'}) "
        f"[rate-type series; describe changes in percentage points]\n"
        f"- CPI YoY inflation: {cpi_yoy:.2f}% "
        f"[rate-type series; describe changes in percentage points]\n"
        f"- Dataset spans {total_months} months of observations"
    )

    return {
        "context_text": context_text,
        "data_points": {
            "unrate": round(unrate_val, 1),
            "gdp_change": round(gdp_chg, 2),
            "t10y2y": round(t10y2y_val, 2),
            "cpi_yoy": round(cpi_yoy, 2),
            "total_months": total_months,
        },
    }


def _context_overview(conn: sqlite3.Connection) -> dict[str, Any]:
    """Build data context for the overview KPI synthesis insight.

    Args:
        conn: SQLite database connection.

    Returns:
        Dict with 'context_text' and 'data_points'.
    """
    latest_unrate: str = _latest_month(conn, "UNRATE")
    prev_unrate: str = _month_offset(latest_unrate, -1)
    latest_gdp: str = _latest_month(conn, "GDPC1")
    latest_cpi: str = _latest_month(conn, "CPIAUCSL")
    cpi_yr_ago: str = _month_offset(latest_cpi, -12)

    unrate_val: float = float(
        _boundary_value(conn, "UNRATE", latest_unrate, latest_unrate, "end")
        or 0
    )
    u6_val: float = float(
        _boundary_value(conn, "U6RATE", latest_unrate, latest_unrate, "end")
        or 0
    )
    prev_unrate_val: float = float(
        _boundary_value(conn, "UNRATE", prev_unrate, prev_unrate, "end")
        or 0
    )
    prev_u6_val: float = float(
        _boundary_value(conn, "U6RATE", prev_unrate, prev_unrate, "end")
        or 0
    )
    unrate_chg: float = _compute_change(
        conn, "UNRATE", prev_unrate, latest_unrate
    )
    u6_chg: float = _compute_change(
        conn, "U6RATE", prev_unrate, latest_unrate
    )
    gdp_yr_ago: str = _month_offset(latest_gdp, -12)
    gdp_chg: float = _compute_change(conn, "GDPC1", gdp_yr_ago, latest_gdp)
    cpi_yoy: float = _compute_change(
        conn, "CPIAUCSL", cpi_yr_ago, latest_cpi
    )
    t10y2y_val: float = float(
        _boundary_value(
            conn, "T10Y2Y",
            _latest_month(conn, "T10Y2Y"),
            _latest_month(conn, "T10Y2Y"), "end",
        ) or 0
    )

    context_text: str = (
        f"Overview KPI Synthesis:\n"
        f"- UNRATE: {unrate_val:.1f}%, MoM change "
        f"{_fmt_change('UNRATE', prev_unrate_val, unrate_val)}\n"
        f"- U6RATE: {u6_val:.1f}%, MoM change "
        f"{_fmt_change('U6RATE', prev_u6_val, u6_val)}\n"
        f"- GDP YoY growth: {gdp_chg:+.2f}% (already a percent change)\n"
        f"- CPI YoY: {cpi_yoy:.2f}% "
        f"({'above' if cpi_yoy > 3 else 'at or below'} 3%; this is a rate, "
        f"describe its own changes in pp)\n"
        f"- Yield spread (T10Y2Y): {t10y2y_val:.2f}% "
        f"({'inverted' if t10y2y_val < 0 else 'normal'})"
    )

    return {
        "context_text": context_text,
        "data_points": {
            "unrate": round(unrate_val, 1),
            "u6": round(u6_val, 1),
            "unrate_mom": round(unrate_chg, 2),
            "u6_mom": round(u6_chg, 2),
            "unrate_mom_pp": round(unrate_val - prev_unrate_val, 2),
            "u6_mom_pp": round(u6_val - prev_u6_val, 2),
            "gdp_yoy": round(gdp_chg, 2),
            "cpi_yoy": round(cpi_yoy, 2),
            "t10y2y": round(t10y2y_val, 2),
        },
    }


def _context_recession_risk(conn: sqlite3.Connection) -> dict[str, Any]:
    """Build data context for the recession risk timeline insight.

    Args:
        conn: SQLite database connection.

    Returns:
        Dict with 'context_text' and 'data_points'.
    """
    latest_row = conn.execute(
        "SELECT date, probability FROM recession_predictions "
        "ORDER BY date DESC LIMIT 1"
    ).fetchone()
    latest_date: str = latest_row[0][:7]
    latest_prob: float = latest_row[1]

    yr_ago_date: str = _month_offset(latest_date, -12)
    yr_ago_row = conn.execute(
        "SELECT probability FROM recession_predictions "
        "WHERE date <= ? ORDER BY date DESC LIMIT 1",
        (f"{yr_ago_date}-31",),
    ).fetchone()
    yr_ago_prob: float = yr_ago_row[0] if yr_ago_row else 0.0

    above_count: int = conn.execute(
        "SELECT COUNT(*) FROM recession_predictions WHERE probability > 0.5"
    ).fetchone()[0]
    total_count: int = conn.execute(
        "SELECT COUNT(*) FROM recession_predictions"
    ).fetchone()[0]

    six_ago: str = _month_offset(latest_date, -6)
    six_row = conn.execute(
        "SELECT probability FROM recession_predictions "
        "WHERE date <= ? ORDER BY date DESC LIMIT 1",
        (f"{six_ago}-31",),
    ).fetchone()
    six_prob: float = six_row[0] if six_row else latest_prob

    context_text: str = (
        f"Recession Risk Timeline:\n"
        f"- Latest risk score: {latest_prob:.4f} ({latest_date})\n"
        f"- Score 12 months ago: {yr_ago_prob:.4f}\n"
        f"- Change: {latest_prob - yr_ago_prob:+.4f}\n"
        f"- Months above 0.5: {above_count}/{total_count}\n"
        f"- 6-month trend: {latest_prob:.4f} vs {six_prob:.4f} "
        f"({'rising' if latest_prob > six_prob else 'falling'})"
    )

    return {
        "context_text": context_text,
        "data_points": {
            "latest_prob": round(latest_prob, 4),
            "yr_ago_prob": round(yr_ago_prob, 4),
            "above_threshold": above_count,
            "total_predictions": total_count,
            "six_month_ago_prob": round(six_prob, 4),
        },
    }


def _context_feature_snapshot(conn: sqlite3.Connection) -> dict[str, Any]:
    """Build data context for the feature contribution snapshot insight.

    Args:
        conn: SQLite database connection.

    Returns:
        Dict with 'context_text' and 'data_points'.
    """
    latest_row = conn.execute(
        "SELECT date, probability, features_json FROM recession_predictions "
        "ORDER BY date DESC LIMIT 1"
    ).fetchone()
    date: str = latest_row[0][:7]
    prob: float = latest_row[1]
    features: dict[str, float] = json.loads(latest_row[2])

    # Label each feature tracked by SIGNAL_RULES with an explicit status so
    # the LLM cannot invent warning signs from feature names alone.
    labeled_features: set[str] = set()
    feature_lines: list[str] = []
    red_count: int = 0
    for rule in SIGNAL_RULES:
        feat: str = rule["feature"]
        val: float = features.get(feat, 0.0)
        is_red: bool = (
            (rule["condition"] == "lt" and val < rule["threshold"])
            or (rule["condition"] == "gt" and val > rule["threshold"])
        )
        if is_red:
            label: str = "RED (warning)"
            red_count += 1
        elif val == 0:
            label = "NEUTRAL"
        else:
            label = "GREEN (reassuring)"
        feature_lines.append(f"  {feat}: {val:.4f}  [{label}]")
        labeled_features.add(feat)
    # Any additional features not in SIGNAL_RULES are shown without a label.
    for feat, val in sorted(features.items()):
        if feat not in labeled_features:
            feature_lines.append(f"  {feat}: {val:.4f}  [untracked]")

    context_text: str = (
        f"Feature Snapshot ({date}, risk score={prob:.4f}, "
        f"red features={red_count}):\n"
        + "\n".join(feature_lines)
        + "\n(The 'risk score' is a 0-1 reading, NOT a probability. "
        "Treat only features marked RED as warning signs.)"
    )

    return {
        "context_text": context_text,
        "data_points": {
            "date": date,
            "probability": round(prob, 4),
            "features": features,
            "red_count": red_count,
        },
    }


def _context_scenario_explorer(conn: sqlite3.Connection) -> dict[str, Any]:
    """Build data context for the scenario explorer insight.

    Args:
        conn: SQLite database connection.

    Returns:
        Dict with 'context_text' and 'data_points'.
    """
    baseline_row = conn.execute(
        "SELECT probability FROM recession_predictions "
        "ORDER BY date DESC LIMIT 1"
    ).fetchone()
    baseline: float = baseline_row[0]

    extremes = conn.execute(
        "SELECT MIN(probability), MAX(probability) FROM scenario_grid"
    ).fetchone()
    grid_min: float = extremes[0]
    grid_max: float = extremes[1]
    grid_count: int = conn.execute(
        "SELECT COUNT(*) FROM scenario_grid"
    ).fetchone()[0]

    # Which slider has biggest range of effect?
    slider_cols = [
        "yield_spread", "unrate", "gdp_growth_annualized", "cpi_yoy",
    ]
    sensitivity: dict[str, float] = {}
    for col in slider_cols:
        row = conn.execute(
            f"SELECT MIN(probability), MAX(probability) FROM scenario_grid "
            f"GROUP BY {col} ORDER BY MAX(probability) - MIN(probability) "
            f"DESC LIMIT 1"
        ).fetchone()
        if row:
            sensitivity[col] = round(row[1] - row[0], 4)

    most_sensitive: str = max(sensitivity, key=sensitivity.get)  # type: ignore[arg-type]

    context_text: str = (
        f"Scenario Explorer:\n"
        f"- Baseline risk score (current features, this IS the "
        f"baseline): {baseline:.4f}\n"
        f"- Grid range: {grid_min:.6f} to {grid_max:.6f}\n"
        f"- Grid size: {grid_count} scenarios\n"
        f"- Most sensitive input: {most_sensitive}\n"
        f"- Sensitivity RANGE per slider (max probability minus min "
        f"probability when varying that slider, NOT a baseline or a "
        f"probability): "
        + ", ".join(f"{k}={v:.4f}" for k, v in sensitivity.items())
    )

    return {
        "context_text": context_text,
        "data_points": {
            "baseline": round(baseline, 4),
            "grid_min": round(grid_min, 6),
            "grid_max": round(grid_max, 6),
            "grid_count": grid_count,
            "most_sensitive": most_sensitive,
            "sensitivity": sensitivity,
        },
    }


def _context_deep_divergence(conn: sqlite3.Connection) -> dict[str, Any]:
    """Build data context for the deep dive employment divergence insight.

    Focuses on post-ChatGPT window.

    Args:
        conn: SQLite database connection.

    Returns:
        Dict with 'context_text' and 'data_points'.
    """
    latest: str = _latest_month(conn, "USINFO")
    chatgpt: str = CHATGPT_LAUNCH

    info_pct: float = _compute_pct_of_start(
        conn, "USINFO", chatgpt, latest, per_capita=True,
    )
    trades_pct: float = _compute_pct_of_start(
        conn, "CES2023800001", chatgpt, latest, per_capita=True,
    )
    info_chg: float = _compute_change(
        conn, "USINFO", chatgpt, latest, per_capita=True,
    )
    trades_chg: float = _compute_change(
        conn, "CES2023800001", chatgpt, latest, per_capita=True,
    )
    gap: float = round(trades_pct - info_pct, 2)

    context_text: str = (
        f"Deep Dive: Employment Divergence (post-ChatGPT):\n"
        f"- Info per-capita index: {info_pct}% of Nov 2022\n"
        f"- Trades per-capita index: {trades_pct}% of Nov 2022\n"
        f"- Info change: {info_chg:+.2f}%\n"
        f"- Trades change: {trades_chg:+.2f}%\n"
        f"- Gap (trades - info): {gap:+.2f}pp\n"
        f"Focus on: what changed after Nov 2022?"
    )

    return {
        "context_text": context_text,
        "data_points": {
            "info_pct": round(info_pct, 2),
            "trades_pct": round(trades_pct, 2),
            "info_change": round(info_chg, 2),
            "trades_change": round(trades_chg, 2),
            "gap": gap,
        },
    }


def _context_deep_energy(conn: sqlite3.Connection) -> dict[str, Any]:
    """Build data context for the deep dive energy paradox insight.

    Focuses on post-ChatGPT window.

    Args:
        conn: SQLite database connection.

    Returns:
        Dict with 'context_text' and 'data_points'.
    """
    latest: str = _latest_month(conn, "IPG2211S")
    chatgpt: str = CHATGPT_LAUNCH

    power_pct: float = _compute_pct_of_start(
        conn, "IPG2211S", chatgpt, latest,
    )
    info_pct: float = _compute_pct_of_start(
        conn, "USINFO", chatgpt, latest,
    )
    power_chg: float = _compute_change(conn, "IPG2211S", chatgpt, latest)
    info_chg: float = _compute_change(conn, "USINFO", chatgpt, latest)
    gap: float = round(power_pct - info_pct, 2)

    context_text: str = (
        f"Deep Dive: Energy Paradox (post-ChatGPT):\n"
        f"- Power output index: {power_pct}% of Nov 2022\n"
        f"- Info employment index: {info_pct}% of Nov 2022\n"
        f"- Power change: {power_chg:+.2f}%\n"
        f"- Info change: {info_chg:+.2f}%\n"
        f"- Power-info gap: {gap:+.2f}pp\n"
        f"Focus on: power demand rising while info employment falls."
    )

    return {
        "context_text": context_text,
        "data_points": {
            "power_pct": round(power_pct, 2),
            "info_pct": round(info_pct, 2),
            "power_change": round(power_chg, 2),
            "info_change": round(info_chg, 2),
            "gap": gap,
        },
    }


def _context_deep_synthesis_charts(
    conn: sqlite3.Connection,
) -> dict[str, Any]:
    """Build data context for the deep dive synthesis insight.

    Connects employment divergence and energy paradox.

    Args:
        conn: SQLite database connection.

    Returns:
        Dict with 'context_text' and 'data_points'.
    """
    latest: str = _latest_month(conn, "USINFO")
    chatgpt: str = CHATGPT_LAUNCH

    info_pct: float = _compute_pct_of_start(
        conn, "USINFO", chatgpt, latest, per_capita=True,
    )
    trades_pct: float = _compute_pct_of_start(
        conn, "CES2023800001", chatgpt, latest, per_capita=True,
    )
    power_pct: float = _compute_pct_of_start(
        conn, "IPG2211S", chatgpt, latest,
    )
    divergence_gap: float = round(trades_pct - info_pct, 2)
    power_info_gap: float = round(power_pct - info_pct, 2)
    both_widening: bool = divergence_gap > 0 and power_info_gap > 0

    context_text: str = (
        f"Deep Dive Synthesis (post-ChatGPT):\n"
        f"- Trades-info divergence: {divergence_gap:+.2f}pp\n"
        f"- Power-info gap: {power_info_gap:+.2f}pp\n"
        f"- Both gaps widening: {'Yes' if both_widening else 'No'}\n"
        f"- Info per-capita: {info_pct}% of Nov 2022\n"
        f"- Trades per-capita: {trades_pct}% of Nov 2022\n"
        f"- Power index: {power_pct}% of Nov 2022\n"
        f"Connect the two charts into a single AI structural impact "
        f"narrative."
    )

    return {
        "context_text": context_text,
        "data_points": {
            "divergence_gap": divergence_gap,
            "power_info_gap": power_info_gap,
            "both_widening": both_widening,
            "info_pct": round(info_pct, 2),
            "trades_pct": round(trades_pct, 2),
            "power_pct": round(power_pct, 2),
        },
    }


def _context_synthesis(conn: sqlite3.Connection) -> dict[str, Any]:
    """Build cross-metric synthesis context for the deep dive insight.

    Args:
        conn: SQLite database connection.

    Returns:
        Dict with 'context_text' and 'data_points'.
    """
    yc: dict[str, Any] = _context_yield_curve_unemployment(conn)
    gdp: dict[str, Any] = _context_gdp_trend(conn)
    it: dict[str, Any] = _context_info_vs_trades(conn)
    u6u3: dict[str, Any] = _context_u6_u3_gap(conn)
    pwr: dict[str, Any] = _context_power_vs_info(conn)

    context_text: str = (
        f"Cross-Metric Synthesis — AI Labor Market Impact:\n"
        f"- Yield curve: {yc['data_points']['longest_inversion']}m longest "
        f"inversion, spread now {yc['data_points']['current_spread']}%\n"
        f"- GDP: latest {gdp['data_points']['latest_growth']}% growth, "
        f"{'in' if gdp['data_points']['current_recession'] else 'not in'} "
        f"recession\n"
        f"- Employment: info {it['data_points']['info_change_pct']:+.1f}% "
        f"vs trades {it['data_points']['trades_change_pct']:+.1f}% "
        f"per capita\n"
        f"- U6-U3 gap: {u6u3['data_points']['current_gap']:.1f} pp, "
        f"U6 faster in {u6u3['data_points']['u6_faster_months']}/"
        f"{u6u3['data_points']['total_recent_months']} recent months\n"
        f"- Power vs info: {pwr['data_points']['divergence']:+.1f} pp "
        f"divergence since ChatGPT\n"
        f"- Core question: how do traditional recession signals interact "
        f"with AI-driven employment divergence?"
    )

    return {
        "context_text": context_text,
        "data_points": {
            "longest_inversion": yc["data_points"]["longest_inversion"],
            "info_change_pct": it["data_points"]["info_change_pct"],
            "trades_change_pct": it["data_points"]["trades_change_pct"],
            "u6_faster_months": u6u3["data_points"]["u6_faster_months"],
            "power_info_divergence": pwr["data_points"]["divergence"],
        },
    }


# ---------------------------------------------------------------------------
# Insight slice configuration
# ---------------------------------------------------------------------------

INSIGHT_SLICES: list[dict[str, Any]] = [
    {
        "metric_key": "dashboard_intro",
        "insight_type": "trend",
        "context_fn": _context_dashboard_intro,
        "claims_fn": _claims_dashboard_intro,
        "analysis_prompt": (
            "Ignore the 2-3 sentence instruction. Write two substantial "
            "paragraphs (8-10 sentences total) suitable for the top of "
            "an executive dashboard.\n\n"
            "FIRST PARAGRAPH — Synthesis: Weave the headline numbers "
            "into a narrative that tells a coherent story. Do not list "
            "stats one after another. Connect them: what does "
            "unemployment at this level, combined with this GDP growth "
            "rate and this yield spread, actually mean for the economy "
            "right now? Compare to historical norms where useful. A "
            "C-suite reader should finish this paragraph understanding "
            "the macro landscape without looking at a single chart.\n\n"
            "SECOND PARAGRAPH — So What: Answer 'what should we watch "
            "and why does it matter.' Identify the key tension or risk "
            "in the data, explain what would change your outlook (e.g., "
            "if the yield curve inverts, if unemployment crosses a "
            "threshold), and give a forward-looking read. This paragraph "
            "should feel like actionable intelligence, not a summary.\n\n"
            "Follow all system-prompt RULES — no causal verbs, no "
            "probability language, no invented numbers."
        ),
    },
    {
        "metric_key": "overview",
        "insight_type": "trend",
        "context_fn": _context_overview,
        "claims_fn": _claims_overview,
        "analysis_prompt": (
            "Synthesize the 5 headline indicators into a cohesive "
            "2-sentence snapshot. Are they pointing in the same "
            "direction or diverging? UNRATE and U6RATE are rates; "
            "describe their MoM changes in percentage points, not "
            "relative percent."
        ),
    },
    {
        "metric_key": "recession_risk",
        "insight_type": "trend",
        "context_fn": _context_recession_risk,
        "claims_fn": _claims_recession_risk,
        "analysis_prompt": (
            "Interpret the recession risk timeline. Is the risk score "
            "rising, falling, or stable? How does the current reading "
            "compare to a year ago? The risk score is a 0-1 reading, "
            "NOT a probability. Never say 'likelihood,' 'chance,' or "
            "'probability.'"
        ),
    },
    {
        "metric_key": "feature_snapshot",
        "insight_type": "trend",
        "context_fn": _context_feature_snapshot,
        "claims_fn": _claims_feature_snapshot,
        "analysis_prompt": (
            "Explain what the feature snapshot says about current "
            "recession risk. Only describe features labeled RED as "
            "warning signs. Do not flag NEUTRAL or GREEN features as "
            "concerning. Mention GREEN features as reassuring. Note "
            "that several features (unrate_12m_change, "
            "info_employment_yoy, gdp_growth_annualized) ARE ALREADY "
            "deltas — describe their values as readings or levels "
            "(e.g. 'reading +0.4pp', 'at -3.0%'), not as 'increased "
            "by' or 'declined by' their own values. For yield_spread, "
            "call a positive value 'positive' (not 'low'); 'low' "
            "implies inversion risk and is reserved for values near "
            "or below zero."
        ),
    },
    {
        "metric_key": "scenario_explorer",
        "insight_type": "comparison",
        "context_fn": _context_scenario_explorer,
        "claims_fn": _claims_scenario_explorer,
        "analysis_prompt": (
            "Describe what the scenario explorer reveals about "
            "recession sensitivity. Which input has the strongest "
            "effect on risk? Risk scores are on a 0-1 scale and are "
            "NOT probabilities. Describe the baseline as a 'score of "
            "X' not 'X% likelihood.' The baseline risk score is the "
            "value labeled 'Baseline risk score' in the context. The "
            "sensitivity RANGE values are NOT baselines and NOT "
            "slider settings — they are the swing in risk score when "
            "that slider is varied across the grid. Do not describe "
            "a sensitivity value as a baseline or a slider setting."
        ),
    },
    {
        "metric_key": "deep_divergence",
        "insight_type": "comparison",
        "context_fn": _context_deep_divergence,
        "claims_fn": _claims_deep_divergence,
        "analysis_prompt": (
            "Narrate the employment divergence between info and "
            "trades sectors. Focus on the post-ChatGPT period. "
            "Do not repeat the full-range trend. What changed "
            "after Nov 2022? The series are per-capita EMPLOYMENT "
            "indexes, not productivity or output. Use 'employment' "
            "explicitly."
        ),
    },
    {
        "metric_key": "deep_energy",
        "insight_type": "correlation",
        "context_fn": _context_deep_energy,
        "claims_fn": _claims_deep_energy,
        "analysis_prompt": (
            "Explain the energy paradox: power demand rising while "
            "info employment falls. Focus on the post-ChatGPT "
            "period. What does this suggest about AI infrastructure? "
            "Describe the relationship as 'coincides with' or "
            "'tracks alongside'; the data does not prove that AI "
            "causes energy demand."
        ),
    },
    {
        "metric_key": "deep_synthesis_charts",
        "insight_type": "comparison",
        "context_fn": _context_deep_synthesis_charts,
        "claims_fn": _claims_deep_synthesis_charts,
        "analysis_prompt": (
            "Connect the employment divergence and energy paradox "
            "into a single narrative about AI's structural impact "
            "on work. Do not subtract the per-capita employment "
            "index from the power output index — they are on "
            "different scales. Reference each series on its own. "
            "The info and trades series are EMPLOYMENT indexes; "
            "never call them 'output,' 'productivity,' or "
            "'activity.' Only IPG2211S is an output (power "
            "production) index."
        ),
    },
    {
        "metric_key": "T10Y2Y_UNRATE",
        "insight_type": "correlation",
        "context_fn": _context_yield_curve_unemployment,
        "claims_fn": _claims_yield_curve_unemployment,
        "analysis_prompt": (
            "Analyze the correlation between yield curve inversions "
            "and unemployment rate changes. Report unemployment "
            "changes in percentage points. The yield spread average "
            "is itself a rate value, report its level in percent."
        ),
    },
    {
        "metric_key": "GDPC1",
        "insight_type": "trend",
        "context_fn": _context_gdp_trend,
        "claims_fn": _claims_gdp_trend,
        "analysis_prompt": (
            "Analyze GDP growth trends. The 'average quarterly GDP "
            "level' is a dollar LEVEL, not a change. Do not describe "
            "it as expansion or growth. Do not reference any "
            "recession risk score, probability, or classifier "
            "output — this slice is GDP-only. Every number in your "
            "narrative must come from the KEY FINDINGS or DATA "
            "CONTEXT for GDP."
        ),
    },
    {
        "metric_key": "CPIAUCSL",
        "insight_type": "trend",
        "context_fn": _context_cpi_trend,
        "claims_fn": _claims_cpi_trend,
        "analysis_prompt": (
            "Analyze CPI inflation trends. Is inflation cooling, "
            "accelerating, or stabilizing? CPI level changes are "
            "percent changes; CPI YoY is a rate, describe its own "
            "changes in percentage points."
        ),
    },
    {
        "metric_key": "USINFO_CES2023800001",
        "insight_type": "comparison",
        "context_fn": _context_info_vs_trades,
        "claims_fn": _claims_info_vs_trades,
        "analysis_prompt": (
            "Compare info sector vs specialty trades employment "
            "per capita. What does the divergence reveal about "
            "AI's impact on the labor market? These are per-capita "
            "EMPLOYMENT indexes. Do not call them productivity, "
            "output, or activity. Describe the relationship with "
            "'coincides with.'"
        ),
    },
    {
        "metric_key": "employment_growth",
        "insight_type": "trend",
        "context_fn": _context_employment_growth,
        "claims_fn": _claims_employment_growth,
        "analysis_prompt": (
            "Analyze rolling per-capita employment growth for both "
            "sectors. Which shows stronger momentum? YoY growth is "
            "already a percent change; do not add a 'pp' suffix. "
            "Describe direction without asserting causation."
        ),
    },
    {
        "metric_key": "covid_recovery",
        "insight_type": "comparison",
        "context_fn": _context_covid_recovery,
        "claims_fn": _claims_covid_recovery,
        "analysis_prompt": (
            "Compare COVID recovery trajectories between info and "
            "trades sectors. These are recovery ratios (percent of "
            "pre-COVID peak). Describe as percent recovery, not "
            "productivity."
        ),
    },
    {
        "metric_key": "U6_U3",
        "insight_type": "trend",
        "context_fn": _context_u6_u3_gap,
        "claims_fn": _claims_u6_u3_gap,
        "analysis_prompt": (
            "Analyze the U6 vs U3 gap and what it reveals about "
            "hidden labor market slack. Both rates are percentages. "
            "Describe the gap and its changes in percentage points."
        ),
    },
    {
        "metric_key": "IPG2211S_USINFO",
        "insight_type": "correlation",
        "context_fn": _context_power_vs_info,
        "claims_fn": _claims_power_vs_info,
        "analysis_prompt": (
            "Analyze electric power output vs info employment. "
            "Use 'coincides with' or 'tracks alongside' — the data "
            "does not prove AI causes the divergence. Do not say "
            "'displaces' or 'drives.'"
        ),
    },
    {
        "metric_key": "synthesis",
        "insight_type": "trend",
        "context_fn": _context_synthesis,
        "claims_fn": _claims_synthesis,
        "analysis_prompt": (
            "Synthesize all macro indicators into a cohesive "
            "narrative about recession risk and AI's labor market "
            "impact. Describe any recession risk score as a 0-1 "
            "reading. Use correlation language ('coincides with,' "
            "'tracks alongside') for multi-series relationships. No "
            "causal verbs."
        ),
    },
]


# ---------------------------------------------------------------------------
# Core generation logic
# ---------------------------------------------------------------------------


def _call_llm(model: str, user_prompt: str) -> str:
    """Send a prompt to the local Ollama model and return the response text.

    Args:
        model: Ollama model name (e.g. 'llama3.1:8b').
        user_prompt: The full user prompt with data context.

    Returns:
        Raw text response from the model.

    Raises:
        httpx.HTTPStatusError: If Ollama returns a non-200 status.
    """
    resp: httpx.Response = httpx.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {"temperature": 0.3, "num_predict": 2048},
        },
        timeout=120.0,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def _parse_narrative(raw_text: str) -> str:
    """Extract a clean narrative paragraph from the LLM response.

    Strips markdown fences, quotes, and preamble if present.

    Args:
        raw_text: Raw text from the LLM.

    Returns:
        Clean narrative string.

    Raises:
        ValueError: If the response is empty after cleaning.
    """
    text: str = raw_text.strip()

    # Strip markdown code fences
    if text.startswith("```"):
        lines: list[str] = text.split("\n")
        lines = [ln for ln in lines if not ln.strip().startswith("```")]
        text = "\n".join(lines).strip()

    # Strip wrapping quotes
    if (text.startswith('"') and text.endswith('"')) or (
        text.startswith("'") and text.endswith("'")
    ):
        text = text[1:-1].strip()

    # Try to extract narrative from JSON if the model returned it anyway
    if text.startswith("{"):
        try:
            parsed: dict[str, Any] = json.loads(text)
            if "narrative" in parsed:
                text = parsed["narrative"]
        except json.JSONDecodeError:
            pass

    if not text:
        raise ValueError("LLM returned empty narrative")

    return text


def _validate_narrative(
    narrative: str,
    metric_key: str,
    *,
    citation_warnings: list[str] | None = None,
    retrieval_empty: bool = False,
    citation_count: int = 0,
) -> list[str]:
    """Scan a narrative for known anti-patterns and emit warnings.

    Runs after _parse_narrative(). Each regex in NARRATIVE_PATTERNS that
    matches produces a warning string. Phase 11 adds three citation-aware
    checks: stray [ref:N] tags (emitted by _extract_citations via
    ``citation_warnings``), methodology phrasing without any citation when
    refs were actually available, and more than two citations per narrative.
    Warnings are logged and returned for optional re-generation. Storage is
    NOT blocked — operators decide whether to rerun.

    Args:
        narrative: Cleaned narrative text, AFTER _extract_citations has run.
        metric_key: The slice's metric_key, for log context.
        citation_warnings: Warnings already produced by _extract_citations
            (stray tags). Merged into the returned list so the caller does
            not need to reason about two separate warning streams.
        retrieval_empty: True when REFERENCE CONTEXT for this slice was
            empty. Skips the methodology-without-citation check because no
            refs were available to cite.
        citation_count: Number of citation records produced. Warns when
            greater than 2 (RULE 7 asks for "at most two references").

    Returns:
        List of warning strings. Empty list means the narrative is clean.
    """
    warnings: list[str] = []
    for pattern, message in NARRATIVE_PATTERNS:
        if re.search(pattern, narrative):
            warnings.append(f"[{metric_key}] {message}")
            logger.warning(
                "Narrative validator flagged %s: %s", metric_key, message
            )

    if citation_warnings:
        for w in citation_warnings:
            warnings.append(f"[{metric_key}] {w}")
            logger.warning("Narrative validator flagged %s: %s", metric_key, w)

    # Methodology language without any citation — only fires when refs were
    # available. Skips the check when retrieval returned nothing so the
    # narrative cannot be faulted for failing to cite something that never
    # existed in REFERENCE CONTEXT.
    if not retrieval_empty and citation_count == 0:
        if METHODOLOGY_LANGUAGE_RE.search(narrative):
            msg: str = (
                "Narrative uses methodology phrasing but has no "
                "citations — consider whether a [ref:N] was warranted."
            )
            warnings.append(f"[{metric_key}] {msg}")
            logger.warning("Narrative validator flagged %s: %s", metric_key, msg)

    if citation_count > 2:
        msg = (
            f"Narrative has {citation_count} citations; RULE 7 asks for "
            "at most two."
        )
        warnings.append(f"[{metric_key}] {msg}")
        logger.warning("Narrative validator flagged %s: %s", metric_key, msg)

    return warnings


def _store_insight(
    conn: sqlite3.Connection,
    metric_key: str,
    slice_key: str,
    insight_type: str,
    narrative: str,
    claims: list[dict[str, Any]],
    citations: list[dict[str, Any]],
    model: str,
) -> None:
    """Store a generated insight in the ai_insights table.

    Uses INSERT OR REPLACE to handle re-runs idempotently.

    Args:
        conn: SQLite database connection.
        metric_key: The metric key for this insight.
        slice_key: The time period slice key.
        insight_type: The insight type (trend, correlation, comparison).
        narrative: LLM-generated narrative paragraph, with valid [ref:N]
            tags already stripped by _extract_citations.
        claims: Pre-computed verifiable claims list.
        citations: List of citation records. Each record has keys
            ``ref_id``, ``doc_type``, ``series_id``, ``title``,
            ``source_url``, and ``excerpt`` (see _extract_citations).
            Stored as ``citations_json``.
        model: Model name used for generation.
    """
    conn.execute(
        "INSERT OR REPLACE INTO ai_insights "
        "  (metric_key, slice_key, insight_type, narrative, claims_json, "
        "   verification_json, all_verified, model_used, generated_at, "
        "   citations_json) "
        "VALUES (?, ?, ?, ?, ?, '{}', 0, ?, ?, ?)",
        (
            metric_key,
            slice_key,
            insight_type,
            narrative,
            json.dumps(claims),
            model,
            datetime.now(timezone.utc).isoformat(),
            json.dumps(citations),
        ),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# RAG citation helpers (Phase 11)
# ---------------------------------------------------------------------------


_REF_TAG_RE: re.Pattern[str] = re.compile(r"\[ref:(\d+)\]")
"""Match tags like [ref:12] and capture the numeric id."""


# Cross-series slices where a single series_hint would be too narrow.
# For these, retrieval runs unfiltered and similarity ranking decides.
_CROSS_SERIES_METRICS: frozenset[str] = frozenset(
    {
        "dashboard_intro",
        "overview",
        "recession_risk",
        "feature_snapshot",
        "scenario_explorer",
        "deep_divergence",
        "deep_energy",
        "deep_synthesis_charts",
        "employment_growth",
        "covid_recovery",
        "synthesis",
    }
)

# Composite metric_key → leading series id used as the series_hint.
_COMPOSITE_METRIC_HINTS: dict[str, str] = {
    "T10Y2Y_UNRATE": "T10Y2Y",
    "USINFO_CES2023800001": "USINFO",
    "IPG2211S_USINFO": "IPG2211S",
    "U6_U3": "U6RATE",
}


def _series_hint_for_metric(metric_key: str) -> str | None:
    """Choose a series_hint for a slice's retrieval query.

    Cross-series slices return None; composite keys resolve to the leading
    series id; everything else is assumed to be a single series id that
    matches a reference_docs row.

    Args:
        metric_key: The slice's metric_key.

    Returns:
        A FRED series id or None.
    """
    if metric_key in _CROSS_SERIES_METRICS:
        return None
    if metric_key in _COMPOSITE_METRIC_HINTS:
        return _COMPOSITE_METRIC_HINTS[metric_key]
    return metric_key


def _build_reference_context(
    chunks: list[rag_retrieval.RetrievedChunk],
) -> tuple[str, dict[int, rag_retrieval.RetrievedChunk]]:
    """Format retrieved chunks as a REFERENCE CONTEXT block for the LLM.

    Deduplicates by ``doc_id`` so each ``[ref:N]`` tag corresponds to one
    reference_docs row, even when multiple chunks of that row matched.

    Args:
        chunks: Retrieval results. Pass an empty list when retrieval
            returned nothing.

    Returns:
        Tuple of (context_block, provided_by_id). ``context_block`` is the
        text to inject into the user prompt. ``provided_by_id`` maps each
        ``doc_id`` that appeared in the context to its best chunk; it is
        passed to _extract_citations as the whitelist of valid ids.
    """
    if not chunks:
        return "REFERENCE CONTEXT: (none)", {}

    # Keep first occurrence per doc_id (retrieval returns best-first).
    provided: dict[int, rag_retrieval.RetrievedChunk] = {}
    for c in chunks:
        if c.doc_id not in provided:
            provided[c.doc_id] = c

    lines: list[str] = ["REFERENCE CONTEXT:"]
    for doc_id, chunk in provided.items():
        lines.append(f"[ref:{doc_id}] {chunk.title}")
        lines.append(chunk.content.strip())
        if chunk.source_url:
            lines.append(f"Source: {chunk.source_url}")
        lines.append("")  # blank line separator

    return "\n".join(lines).rstrip(), provided


def _extract_citations(
    narrative: str,
    provided: dict[int, rag_retrieval.RetrievedChunk],
) -> tuple[str, list[dict[str, Any]], list[str]]:
    """Parse [ref:N] tags out of a narrative, match to provided refs.

    Behavior, matching the Phase 11 plan:
      * Valid tag (N is a key in ``provided``): record one citation entry
        per unique id and STRIP the tag from the narrative.
      * Invalid tag (N not in ``provided``): emit a warning, leave the tag
        in place so reviewers see the failure.
      * Deduplicated: a single citation record per id even when the same id
        is cited multiple times.

    Args:
        narrative: Raw narrative from the LLM (with [ref:N] tags).
        provided: Mapping of doc_id → retrieved chunk, as produced by
            _build_reference_context.

    Returns:
        Tuple of (stripped_narrative, citation_records, warnings).
    """
    warnings: list[str] = []
    records: list[dict[str, Any]] = []
    seen_valid: set[int] = set()

    # Walk all matches in order so we process invalids (for warnings)
    # before stripping the valid ones.
    for match in _REF_TAG_RE.finditer(narrative):
        ref_id: int = int(match.group(1))
        if ref_id in provided:
            if ref_id in seen_valid:
                continue
            seen_valid.add(ref_id)
            chunk: rag_retrieval.RetrievedChunk = provided[ref_id]
            excerpt: str = chunk.content.strip()
            if len(excerpt) > 240:
                excerpt = excerpt[:237].rstrip() + "..."
            records.append(
                {
                    "ref_id": ref_id,
                    "doc_type": chunk.doc_type,
                    "series_id": chunk.series_id,
                    "title": chunk.title,
                    "source_url": chunk.source_url,
                    "excerpt": excerpt,
                }
            )
        else:
            warnings.append(
                f"Narrative cites [ref:{ref_id}] but that id was not in "
                "REFERENCE CONTEXT (hallucinated citation)."
            )

    # Strip only the tags whose id is valid. Attach a single space to
    # collapse the surrounding whitespace without mangling the sentence.
    def _replace_valid(match: re.Match[str]) -> str:
        """Return empty string for valid tags, original for invalid."""
        tag_id: int = int(match.group(1))
        return "" if tag_id in seen_valid else match.group(0)

    stripped: str = _REF_TAG_RE.sub(_replace_valid, narrative)
    # Collapse double spaces introduced by tag removal
    stripped = re.sub(r"\s{2,}", " ", stripped)
    stripped = re.sub(r"\s+([,.;:!?])", r"\1", stripped)
    stripped = stripped.strip()

    return stripped, records, warnings


def generate_insight(
    conn: sqlite3.Connection,
    model: str,
    slice_config: dict[str, Any],
    slice_key: str,
) -> bool:
    """Generate a single AI insight for the given slice configuration.

    Phase 11: retrieves reference chunks from the local ChromaDB store,
    injects a REFERENCE CONTEXT block into the user prompt, and extracts
    ``[ref:N]`` citations from the LLM response into ``citations_json``.

    Args:
        conn: SQLite database connection.
        model: Ollama model name.
        slice_config: Dict with metric_key, insight_type, context_fn,
            analysis_prompt.
        slice_key: The time period slice key.

    Returns:
        True if the insight was generated and stored successfully.
    """
    metric_key: str = slice_config["metric_key"]
    insight_type: str = slice_config["insight_type"]

    logger.info("Generating insight: %s / %s", metric_key, insight_type)

    # Build verifiable claims from DB (independent of LLM)
    claims: list[dict[str, Any]] = slice_config["claims_fn"](conn)

    # Build context and key findings for the LLM prompt
    context: dict[str, Any] = slice_config["context_fn"](conn)
    findings: str = "\n".join(
        f"- {c['description']}" for c in claims
    )

    # Retrieve supporting reference chunks (Phase 11 RAG step)
    query: str = f"{metric_key} {slice_config['analysis_prompt'][:200]}"
    series_hint: str | None = _series_hint_for_metric(metric_key)
    retrieved: list[rag_retrieval.RetrievedChunk] = rag_retrieval.retrieve(
        query, k=7, series_hint=series_hint, min_scholarly=3
    )
    reference_block, provided = _build_reference_context(retrieved)
    retrieval_empty: bool = not provided

    user_prompt: str = (
        f"DATA CONTEXT:\n{context['context_text']}\n\n"
        f"{reference_block}\n\n"
        f"KEY FINDINGS:\n{findings}\n\n"
        f"ANALYSIS DIRECTION:\n{slice_config['analysis_prompt']}"
    )

    try:
        raw_response: str = _call_llm(model, user_prompt)
        narrative: str = _parse_narrative(raw_response)
        stripped_narrative, citation_records, citation_warnings = (
            _extract_citations(narrative, provided)
        )
        warnings: list[str] = _validate_narrative(
            stripped_narrative,
            metric_key,
            citation_warnings=citation_warnings,
            retrieval_empty=retrieval_empty,
            citation_count=len(citation_records),
        )
        _store_insight(
            conn, metric_key, slice_key, insight_type,
            stripped_narrative, claims, citation_records, model,
        )
        if warnings:
            logger.warning(
                "Stored %s/%s with %d validator warning(s) — "
                "review narrative",
                metric_key,
                insight_type,
                len(warnings),
            )
        logger.info(
            "Stored insight: %s / %s (%d claims, %d citations)",
            metric_key,
            insight_type,
            len(claims),
            len(citation_records),
        )
        return True
    except ValueError as exc:
        logger.error(
            "Failed to parse LLM response for %s / %s: %s",
            metric_key,
            insight_type,
            exc,
        )
        return False
    except httpx.HTTPStatusError as exc:
        logger.error(
            "Ollama error for %s / %s: %s",
            metric_key,
            insight_type,
            exc,
        )
        return False


def generate_all(
    db_mode: str = "seed",
    metric_filter: str | None = None,
    model: str = DEFAULT_MODEL,
) -> None:
    """Generate AI insights for all (or one) metric slices.

    Args:
        db_mode: 'seed' or 'full'.
        metric_filter: If provided, only generate for this metric_key.
        model: Ollama model name to use.
    """
    db_file: Path = _db_path(db_mode)
    if not db_file.exists():
        logger.error("Database not found: %s", db_file)
        raise SystemExit(1)

    _check_ollama()

    conn: sqlite3.Connection = sqlite3.connect(db_file)
    slice_key: str = _compute_slice_key(conn)

    slices: list[dict[str, Any]] = INSIGHT_SLICES
    if metric_filter:
        slices = [s for s in slices if s["metric_key"] == metric_filter]
        if not slices:
            logger.error("Unknown metric key: %s", metric_filter)
            conn.close()
            raise SystemExit(1)

    success: int = 0
    total: int = len(slices)
    for s in slices:
        if generate_insight(conn, model, s, slice_key):
            success += 1

    conn.close()
    logger.info("Generated %d/%d insights for %s.db", success, total, db_mode)


def main() -> None:
    """CLI entry point for batch insight generation."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Batch-generate AI insights for the macro economic "
        "dashboard using a local Ollama model.",
    )
    parser.add_argument(
        "--db",
        choices=["seed", "full"],
        default="seed",
        help="Which database to use (default: seed).",
    )
    parser.add_argument(
        "--metric",
        default=None,
        help="Generate only for this metric_key (default: all).",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Ollama model to use (default: {DEFAULT_MODEL}).",
    )
    args: argparse.Namespace = parser.parse_args()
    generate_all(
        db_mode=args.db, metric_filter=args.metric, model=args.model
    )


if __name__ == "__main__":
    main()
