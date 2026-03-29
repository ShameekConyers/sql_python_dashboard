"""Macro Economic Dashboard: Recession Risk and AI's Labor Market Impact.

Streamlit dashboard that queries seed.db (or full.db) directly, surfaces
interactive Plotly charts for 8 analysis queries, and renders AI-verified
insight blocks when available.
"""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
SEED_DB: Path = PROJECT_ROOT / "data" / "seed.db"
FULL_DB: Path = PROJECT_ROOT / "data" / "full.db"

# Consistent color palette across all charts
COLORS: dict[str, str] = {
    "info": "#636EFA",        # blue  — information sector
    "trades": "#EF553B",      # red   — specialty trades
    "unemployment": "#00CC96", # teal  — UNRATE / U3
    "u6": "#AB63FA",          # purple — U6
    "gdp": "#FFA15A",         # orange — GDP growth
    "yield": "#19D3F3",       # cyan  — yield curve
    "cpi": "#FF6692",         # pink  — CPI / inflation
    "power": "#B6E880",       # lime  — electric power
    "recession": "rgba(200,200,200,0.3)",  # light gray shading
}

CHATGPT_LAUNCH: str = "2022-11-30"


# ---------------------------------------------------------------------------
# Data layer
# ---------------------------------------------------------------------------

def get_connection(full: bool = False) -> sqlite3.Connection:
    """Return a SQLite connection to the appropriate database file.

    Args:
        full: If True and full.db exists, connect to it instead of seed.db.

    Returns:
        An open sqlite3.Connection.
    """
    db_path = FULL_DB if full and FULL_DB.exists() else SEED_DB
    return sqlite3.connect(str(db_path))


@st.cache_data(ttl=300)
def run_query(sql: str, full: bool = False) -> pd.DataFrame:
    """Execute a SQL query and return the result as a DataFrame.

    Args:
        sql: The SQL query string to execute.
        full: Whether to use full.db instead of seed.db.

    Returns:
        A pandas DataFrame containing the query results.
    """
    conn = get_connection(full)
    try:
        return pd.read_sql_query(sql, conn)
    finally:
        conn.close()


@st.cache_data(ttl=300)
def load_series_metadata(full: bool = False) -> pd.DataFrame:
    """Load the series_metadata table for display names and categories.

    Args:
        full: Whether to use full.db instead of seed.db.

    Returns:
        DataFrame with series_id, name, category, units columns.
    """
    return run_query("SELECT series_id, name, category, units FROM series_metadata", full)


@st.cache_data(ttl=300)
def get_date_range(full: bool = False) -> tuple[str, str]:
    """Get the min and max observation dates from the database.

    Args:
        full: Whether to use full.db instead of seed.db.

    Returns:
        Tuple of (min_date, max_date) as ISO date strings.
    """
    df = run_query("SELECT MIN(date) AS min_d, MAX(date) AS max_d FROM observations", full)
    return df["min_d"].iloc[0], df["max_d"].iloc[0]


@st.cache_data(ttl=300)
def get_ai_insight(metric_key: str, insight_type: str, full: bool = False) -> dict | None:
    """Query ai_insights for a given metric_key and insight_type.

    Args:
        metric_key: The metric key to look up.
        insight_type: The insight type (trend, correlation, anomaly, comparison).
        full: Whether to use full.db instead of seed.db.

    Returns:
        A dict with narrative, claims_json, verification_json, all_verified,
        or None if no matching row exists.
    """
    df = run_query(
        f"""
        SELECT narrative, claims_json, verification_json, all_verified
        FROM ai_insights
        WHERE metric_key = '{metric_key}' AND insight_type = '{insight_type}'
        LIMIT 1
        """,
        full,
    )
    if df.empty:
        return None
    row = df.iloc[0]
    return {
        "narrative": row["narrative"],
        "claims_json": row["claims_json"],
        "verification_json": row["verification_json"],
        "all_verified": bool(row["all_verified"]),
    }


def has_ai_insights(full: bool = False) -> bool:
    """Check whether the ai_insights table has any rows.

    Args:
        full: Whether to use full.db instead of seed.db.

    Returns:
        True if at least one insight exists.
    """
    df = run_query("SELECT COUNT(*) AS cnt FROM ai_insights", full)
    return int(df["cnt"].iloc[0]) > 0


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------

def add_annotated_vline(
    fig: go.Figure,
    x: str,
    line_dash: str = "dash",
    line_color: str = "orange",
    text: str = "",
    font_size: int = 10,
) -> None:
    """Add a vertical line with optional annotation, compatible with Plotly v6.

    Plotly v6's add_vline tries to sum() the x values for annotation placement,
    which fails on date strings. This helper uses add_shape + add_annotation
    directly to avoid the issue.

    Args:
        fig: The Plotly figure.
        x: The x-axis value for the vertical line (ISO date string).
        line_dash: Dash style for the line.
        line_color: Color of the line.
        text: Annotation text (empty string to skip annotation).
        font_size: Font size for the annotation.
    """
    fig.add_shape(
        type="line", x0=x, x1=x, y0=0, y1=1,
        yref="paper", line=dict(dash=line_dash, color=line_color, width=1),
    )
    if text:
        fig.add_annotation(
            x=x, y=1, yref="paper",
            text=text, showarrow=False,
            font=dict(size=font_size, color=line_color),
            yanchor="bottom",
        )


def add_annotated_vrect(
    fig: go.Figure,
    x0: str,
    x1: str,
    fillcolor: str = "rgba(200,200,200,0.3)",
    text: str = "",
    font_size: int = 9,
    font_color: str = "gray",
) -> None:
    """Add a shaded rectangle with optional annotation, compatible with Plotly v6.

    Args:
        fig: The Plotly figure.
        x0: Left bound x-axis value.
        x1: Right bound x-axis value.
        fillcolor: Fill color for the rectangle.
        text: Annotation text (empty to skip).
        font_size: Font size for the annotation.
        font_color: Color for the annotation text.
    """
    fig.add_shape(
        type="rect", x0=x0, x1=x1, y0=0, y1=1,
        yref="paper", fillcolor=fillcolor, line_width=0,
    )
    if text:
        fig.add_annotation(
            x=x0, y=1, yref="paper",
            text=text, showarrow=False,
            font=dict(size=font_size, color=font_color),
            xanchor="left", yanchor="bottom",
        )


def add_recession_shading(fig: go.Figure, df_recession: pd.DataFrame) -> None:
    """Add semi-transparent gray vrects for NBER recession periods.

    Args:
        fig: The Plotly figure to add shading to.
        df_recession: DataFrame with 'date' and 'in_recession' columns.
    """
    in_rec = False
    start = None
    for _, row in df_recession.iterrows():
        if row["in_recession"] == 1 and not in_rec:
            start = row["date"]
            in_rec = True
        elif row["in_recession"] == 0 and in_rec:
            add_annotated_vrect(
                fig, start, row["date"],
                fillcolor=COLORS["recession"],
                text="Recession", font_color="gray",
            )
            in_rec = False
    if in_rec and start:
        add_annotated_vrect(
            fig, start, df_recession["date"].iloc[-1],
            fillcolor=COLORS["recession"],
        )


def get_recession_data(date_min: str, date_max: str, full: bool = False) -> pd.DataFrame:
    """Fetch monthly USREC data for recession shading.

    Args:
        date_min: Start date filter (ISO string).
        date_max: End date filter (ISO string).
        full: Whether to use full.db instead of seed.db.

    Returns:
        DataFrame with date and in_recession columns.
    """
    return run_query(
        f"""
        SELECT date, CAST(value_covid_adjusted AS INTEGER) AS in_recession
        FROM observations
        WHERE series_id = 'USREC'
          AND date BETWEEN '{date_min}' AND '{date_max}'
        ORDER BY date
        """,
        full,
    )


def style_figure(fig: go.Figure, title: str = "") -> go.Figure:
    """Apply consistent styling to a Plotly figure.

    Args:
        fig: The Plotly figure to style.
        title: Optional chart title.

    Returns:
        The styled figure.
    """
    fig.update_layout(
        title=title,
        template="plotly_white",
        hovermode="x unified",
        margin=dict(l=40, r=20, t=50, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(tickformat="%b %Y"),
    )
    return fig


def render_ai_insight_block(metric_key: str, insight_type: str, full: bool = False) -> None:
    """Render an AI insight expander block below a chart section.

    Only renders if a matching row exists in ai_insights. Silently skips
    otherwise (no error, no placeholder text).

    Args:
        metric_key: The metric key to look up.
        insight_type: The insight type to look up.
        full: Whether to use full.db instead of seed.db.
    """
    insight = get_ai_insight(metric_key, insight_type, full)
    if insight is None:
        return

    # Determine three-tier verification status from individual claims
    try:
        claims = json.loads(insight["claims_json"])
        verification = json.loads(insight["verification_json"])
        pass_count: int = sum(
            1
            for i in range(len(claims))
            if verification.get(str(i), verification.get(i, {})).get("passed", False)
        )
        total: int = len(claims)
    except (json.JSONDecodeError, TypeError, AttributeError):
        claims, verification, pass_count, total = [], {}, 0, 0

    if total > 0 and pass_count == total:
        banner = ":green[**Verified**] — all claims checked against the database."
    elif pass_count > 0:
        banner = (
            f":orange[**Partially Verified**] — {pass_count} of "
            f"{total} claims confirmed."
        )
    else:
        banner = ":red[**Unverified**] — no claims could be confirmed."

    with st.expander("AI Insight", expanded=True):
        st.markdown(banner)
        st.markdown(insight["narrative"])
        with st.expander("Show sources"):
            if not claims:
                st.text("Could not parse verification data.")
            else:
                source_rows = []
                for i, claim in enumerate(claims):
                    v = verification.get(str(i), verification.get(i, {}))
                    passed = v.get("passed", False)
                    source_rows.append({
                        "Claim": claim.get("description", claim.get("metric", str(claim))),
                        "Expected": claim.get("value", ""),
                        "Actual": v.get("actual_value", ""),
                        "Status": "Verified" if passed else "Unverified",
                    })
                df = pd.DataFrame(source_rows)

                def _color_status(val: object) -> str:
                    """Apply green/red color to the Status column."""
                    if val == "Verified":
                        return "color: green; font-weight: bold"
                    return "color: red; font-weight: bold"

                styled = df.style.map(  # type: ignore[arg-type]
                    _color_status, subset=["Status"]
                )
                st.dataframe(styled, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Query functions (map to 03_analysis_queries.sql)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300)
def query_q1(date_min: str, date_max: str, full: bool = False) -> pd.DataFrame:
    """Q1: Yield curve inversions vs unemployment.

    Args:
        date_min: Start date filter.
        date_max: End date filter.
        full: Whether to use full.db.

    Returns:
        DataFrame with month, avg_spread, inverted, unemployment_rate columns.
    """
    return run_query(
        f"""
        WITH monthly_spread AS (
            SELECT
                SUBSTR(date, 1, 7) AS month,
                ROUND(AVG(value_covid_adjusted), 3) AS avg_spread
            FROM observations
            WHERE series_id = 'T10Y2Y'
              AND date BETWEEN '{date_min}' AND '{date_max}'
            GROUP BY SUBSTR(date, 1, 7)
        ),
        unemployment AS (
            SELECT
                SUBSTR(date, 1, 7) AS month,
                value_covid_adjusted AS unemployment_rate
            FROM observations
            WHERE series_id = 'UNRATE'
              AND date BETWEEN '{date_min}' AND '{date_max}'
        )
        SELECT
            ms.month,
            ms.avg_spread,
            CASE WHEN ms.avg_spread < 0 THEN 1 ELSE 0 END AS inverted,
            u.unemployment_rate,
            ROUND(
                u.unemployment_rate - LAG(u.unemployment_rate, 12)
                    OVER (ORDER BY ms.month),
                2
            ) AS unemployment_yoy_change
        FROM monthly_spread ms
        LEFT JOIN unemployment u ON ms.month = u.month
        ORDER BY ms.month
        """,
        full,
    )


@st.cache_data(ttl=300)
def query_q2(date_min: str, date_max: str, full: bool = False) -> pd.DataFrame:
    """Q2: Info vs trades divergence, per-capita indexed.

    Args:
        date_min: Start date filter.
        date_max: End date filter.
        full: Whether to use full.db.

    Returns:
        DataFrame with date, info/trades indices, and divergence gap.
    """
    return run_query(
        f"""
        WITH population AS (
            SELECT date, value_covid_adjusted AS pop
            FROM observations
            WHERE series_id = 'CNP16OV'
        ),
        per_capita AS (
            SELECT
                o.date, o.series_id,
                o.value_covid_adjusted / p.pop * 1000 AS per_1k_pop,
                o.value_covid_adjusted AS employment
            FROM observations o
            JOIN population p ON o.date = p.date
            WHERE o.series_id IN ('USINFO', 'CES2023800001')
        ),
        indexed AS (
            SELECT date, series_id, employment, per_1k_pop,
                ROUND(100.0 * per_1k_pop / FIRST_VALUE(per_1k_pop)
                    OVER (PARTITION BY series_id ORDER BY date
                          ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING), 2
                ) AS pc_index
            FROM per_capita
        )
        SELECT
            i.date,
            i.employment AS info_employment,
            i.pc_index AS info_pc_index,
            t.employment AS trades_employment,
            t.pc_index AS trades_pc_index,
            ROUND(t.pc_index - i.pc_index, 2) AS divergence_gap
        FROM indexed i
        JOIN indexed t ON i.date = t.date
        WHERE i.series_id = 'USINFO'
          AND t.series_id = 'CES2023800001'
          AND i.date BETWEEN '{date_min}' AND '{date_max}'
        ORDER BY i.date
        """,
        full,
    )


@st.cache_data(ttl=300)
def query_q3(date_min: str, date_max: str, full: bool = False) -> pd.DataFrame:
    """Q3: GDP annualized growth with NBER recession context.

    Args:
        date_min: Start date filter.
        date_max: End date filter.
        full: Whether to use full.db.

    Returns:
        DataFrame with date, real_gdp, annualized_growth_pct, nber_recession.
    """
    return run_query(
        f"""
        WITH gdp_growth AS (
            SELECT
                date,
                value_covid_adjusted AS real_gdp,
                ROUND(
                    (value_covid_adjusted / LAG(value_covid_adjusted)
                        OVER (ORDER BY date) - 1) * 400, 2
                ) AS annualized_growth_pct
            FROM observations
            WHERE series_id = 'GDPC1'
        ),
        recession_flag AS (
            SELECT
                SUBSTR(date, 1, 7) AS month,
                MAX(CAST(value_covid_adjusted AS INTEGER)) AS in_recession
            FROM observations
            WHERE series_id = 'USREC'
            GROUP BY SUBSTR(date, 1, 7)
        )
        SELECT
            g.date, g.real_gdp, g.annualized_growth_pct,
            COALESCE(r.in_recession, 0) AS nber_recession
        FROM gdp_growth g
        LEFT JOIN recession_flag r ON SUBSTR(g.date, 1, 7) = r.month
        WHERE g.annualized_growth_pct IS NOT NULL
          AND g.date BETWEEN '{date_min}' AND '{date_max}'
        ORDER BY g.date
        """,
        full,
    )


@st.cache_data(ttl=300)
def query_q4(date_min: str, date_max: str, full: bool = False) -> pd.DataFrame:
    """Q4: Rolling 12-month per-capita employment growth by sector.

    Args:
        date_min: Start date filter.
        date_max: End date filter.
        full: Whether to use full.db.

    Returns:
        DataFrame with series_id, series_name, date, yoy_pct_change_pc.
    """
    return run_query(
        f"""
        WITH per_capita AS (
            SELECT
                o.series_id, o.date,
                CASE
                    WHEN o.series_id = 'UNRATE' THEN o.value_covid_adjusted
                    ELSE o.value_covid_adjusted / p.value_covid_adjusted * 1000
                END AS adj_value,
                o.value_covid_adjusted AS raw_value
            FROM observations o
            LEFT JOIN observations p
                ON o.date = p.date AND p.series_id = 'CNP16OV'
            WHERE o.series_id IN ('UNRATE', 'USINFO', 'CES2023800001')
        ),
        lagged AS (
            SELECT series_id, date, adj_value,
                LAG(adj_value, 12) OVER (PARTITION BY series_id ORDER BY date) AS adj_12m_ago
            FROM per_capita
        )
        SELECT
            series_id,
            CASE series_id
                WHEN 'UNRATE' THEN 'Unemployment Rate'
                WHEN 'USINFO' THEN 'Information Sector'
                WHEN 'CES2023800001' THEN 'Specialty Trades'
            END AS series_name,
            date,
            ROUND((adj_value - adj_12m_ago) / adj_12m_ago * 100, 2) AS yoy_pct_change_pc
        FROM lagged
        WHERE adj_12m_ago IS NOT NULL
          AND date BETWEEN '{date_min}' AND '{date_max}'
        ORDER BY series_id, date
        """,
        full,
    )


@st.cache_data(ttl=300)
def query_q5(date_min: str, date_max: str, full: bool = False) -> pd.DataFrame:
    """Q5: COVID recovery comparison (raw values, pct of Feb 2020 peak).

    Args:
        date_min: Start date filter.
        date_max: End date filter.
        full: Whether to use full.db.

    Returns:
        DataFrame with date, info/trades employment and pct_of_peak.
    """
    return run_query(
        f"""
        WITH pre_covid_peak AS (
            SELECT series_id, value AS peak_value
            FROM observations
            WHERE series_id IN ('USINFO', 'CES2023800001')
              AND date = '2020-02-01'
        ),
        recovery AS (
            SELECT
                o.series_id, o.date, o.value, p.peak_value,
                ROUND(o.value / p.peak_value * 100, 2) AS pct_of_peak
            FROM observations o
            JOIN pre_covid_peak p ON o.series_id = p.series_id
            WHERE o.date >= '2020-01-01'
              AND o.date <= '{date_max}'
              AND o.series_id IN ('USINFO', 'CES2023800001')
        )
        SELECT
            date,
            MAX(CASE WHEN series_id = 'USINFO' THEN value END) AS info_employment,
            MAX(CASE WHEN series_id = 'USINFO' THEN pct_of_peak END) AS info_pct_of_peak,
            MAX(CASE WHEN series_id = 'CES2023800001' THEN value END) AS trades_employment,
            MAX(CASE WHEN series_id = 'CES2023800001' THEN pct_of_peak END) AS trades_pct_of_peak
        FROM recovery
        GROUP BY date
        ORDER BY date
        """,
        full,
    )


@st.cache_data(ttl=300)
def query_q6(date_min: str, date_max: str, full: bool = False) -> pd.DataFrame:
    """Q6: U6 vs U3 unemployment gap.

    Args:
        date_min: Start date filter.
        date_max: End date filter.
        full: Whether to use full.db.

    Returns:
        DataFrame with date, u3_rate, u6_rate, u6_u3_gap, YoY changes.
    """
    return run_query(
        f"""
        SELECT
            u3.date,
            u3.value_covid_adjusted AS u3_rate,
            u6.value_covid_adjusted AS u6_rate,
            ROUND(u6.value_covid_adjusted - u3.value_covid_adjusted, 2) AS u6_u3_gap,
            ROUND(
                u3.value_covid_adjusted - LAG(u3.value_covid_adjusted, 12)
                    OVER (ORDER BY u3.date), 2
            ) AS u3_yoy_change,
            ROUND(
                u6.value_covid_adjusted - LAG(u6.value_covid_adjusted, 12)
                    OVER (ORDER BY u3.date), 2
            ) AS u6_yoy_change
        FROM observations u3
        JOIN observations u6 ON u3.date = u6.date AND u6.series_id = 'U6RATE'
        WHERE u3.series_id = 'UNRATE'
          AND u3.date BETWEEN '{date_min}' AND '{date_max}'
        ORDER BY u3.date
        """,
        full,
    )


@st.cache_data(ttl=300)
def query_q7(date_min: str, date_max: str, full: bool = False) -> pd.DataFrame:
    """Q7: Electric power output vs information sector employment (indexed).

    Args:
        date_min: Start date filter.
        date_max: End date filter.
        full: Whether to use full.db.

    Returns:
        DataFrame with date, power_index, info_index, power_vs_info_gap.
    """
    return run_query(
        f"""
        WITH power AS (
            SELECT date, value_covid_adjusted AS power_index_raw,
                ROUND(100.0 * value_covid_adjusted / FIRST_VALUE(value_covid_adjusted)
                    OVER (ORDER BY date ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING), 2
                ) AS power_index
            FROM observations WHERE series_id = 'IPG2211S'
        ),
        info AS (
            SELECT date, value_covid_adjusted AS info_employment,
                ROUND(100.0 * value_covid_adjusted / FIRST_VALUE(value_covid_adjusted)
                    OVER (ORDER BY date ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING), 2
                ) AS info_index
            FROM observations WHERE series_id = 'USINFO'
        )
        SELECT
            p.date, p.power_index, i.info_index,
            ROUND(p.power_index - i.info_index, 2) AS power_vs_info_gap
        FROM power p
        JOIN info i ON p.date = i.date
        WHERE p.date BETWEEN '{date_min}' AND '{date_max}'
        ORDER BY p.date
        """,
        full,
    )


@st.cache_data(ttl=300)
def query_q8(date_min: str, date_max: str, full: bool = False) -> pd.DataFrame:
    """Q8: CPI inflation MoM and YoY.

    Args:
        date_min: Start date filter.
        date_max: End date filter.
        full: Whether to use full.db.

    Returns:
        DataFrame with date, cpi_level, mom_pct_change, yoy_pct_change.
    """
    return run_query(
        f"""
        SELECT
            date, value_covid_adjusted AS cpi_level,
            ROUND((value_covid_adjusted / LAG(value_covid_adjusted, 1)
                OVER (ORDER BY date) - 1) * 100, 3) AS mom_pct_change,
            ROUND((value_covid_adjusted / LAG(value_covid_adjusted, 12)
                OVER (ORDER BY date) - 1) * 100, 2) AS yoy_pct_change
        FROM observations
        WHERE series_id = 'CPIAUCSL'
          AND date BETWEEN '{date_min}' AND '{date_max}'
        ORDER BY date
        """,
        full,
    )


# ---------------------------------------------------------------------------
# KPI helpers
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300)
def get_latest_metric(
    series_id: str, full: bool = False
) -> tuple[float, float, str]:
    """Get the latest value and previous value for a series.

    Args:
        series_id: The FRED series ID.
        full: Whether to use full.db.

    Returns:
        Tuple of (latest_value, delta, latest_date).
    """
    df = run_query(
        f"""
        SELECT date, value_covid_adjusted AS val
        FROM observations
        WHERE series_id = '{series_id}'
        ORDER BY date DESC
        LIMIT 2
        """,
        full,
    )
    if len(df) < 2:
        return (df["val"].iloc[0], 0.0, df["date"].iloc[0]) if len(df) == 1 else (0.0, 0.0, "")
    latest = df["val"].iloc[0]
    prev = df["val"].iloc[1]
    return latest, round(latest - prev, 3), df["date"].iloc[0]


@st.cache_data(ttl=300)
def get_gdp_latest_growth(full: bool = False) -> tuple[float, str]:
    """Get the most recent GDP annualized growth rate.

    Args:
        full: Whether to use full.db.

    Returns:
        Tuple of (annualized_growth_pct, date).
    """
    df = run_query(
        """
        SELECT date,
            ROUND((value_covid_adjusted / LAG(value_covid_adjusted)
                OVER (ORDER BY date) - 1) * 400, 2) AS growth
        FROM observations
        WHERE series_id = 'GDPC1'
        ORDER BY date DESC
        LIMIT 2
        """,
        full,
    )
    if df.empty or pd.isna(df["growth"].iloc[0]):
        return 0.0, ""
    return df["growth"].iloc[0], df["date"].iloc[0]


@st.cache_data(ttl=300)
def get_cpi_yoy_latest(full: bool = False) -> tuple[float, str]:
    """Get the most recent CPI YoY inflation rate.

    Args:
        full: Whether to use full.db.

    Returns:
        Tuple of (yoy_pct, date).
    """
    df = run_query(
        """
        SELECT date,
            ROUND((value_covid_adjusted / LAG(value_covid_adjusted, 12)
                OVER (ORDER BY date) - 1) * 100, 2) AS yoy
        FROM observations
        WHERE series_id = 'CPIAUCSL'
        ORDER BY date DESC
        LIMIT 13
        """,
        full,
    )
    df = df.dropna(subset=["yoy"])
    if df.empty:
        return 0.0, ""
    return df["yoy"].iloc[0], df["date"].iloc[0]


@st.cache_data(ttl=300)
def get_sparkline_data(
    series_id: str, months: int = 24, full: bool = False
) -> pd.DataFrame:
    """Get recent data for a sparkline mini-chart.

    Args:
        series_id: The FRED series ID.
        months: Number of recent months to fetch.
        full: Whether to use full.db.

    Returns:
        DataFrame with date and value columns.
    """
    return run_query(
        f"""
        SELECT date, value_covid_adjusted AS value
        FROM observations
        WHERE series_id = '{series_id}'
        ORDER BY date DESC
        LIMIT {months}
        """,
        full,
    ).sort_values("date")


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Macro Economic Dashboard",
    page_icon="📊",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("Macro Economic Dashboard")
    st.caption("Recession risk and AI's labor market impact")

    use_full = False
    if FULL_DB.exists():
        use_full = st.toggle("Use full dataset", value=False, help="Switch from seed.db to full.db")

    metadata = load_series_metadata(use_full)
    date_min_db, date_max_db = get_date_range(use_full)

    date_range = st.date_input(
        "Date range",
        value=(pd.to_datetime(date_min_db).date(), pd.to_datetime(date_max_db).date()),
        min_value=pd.to_datetime(date_min_db).date(),
        max_value=pd.to_datetime(date_max_db).date(),
    )

    if isinstance(date_range, tuple) and len(date_range) == 2:
        date_min = str(date_range[0])
        date_max = str(date_range[1])
    else:
        date_min = date_min_db
        date_max = date_max_db

    all_categories = sorted(metadata["category"].unique().tolist())
    # Exclude population — it's a normalizer, not a dashboard category
    display_categories = [c for c in all_categories if c != "population"]
    selected_categories = st.multiselect(
        "Categories",
        options=display_categories,
        default=display_categories,
        help="Filter which chart sections are visible",
    )

    use_covid_adjusted = st.checkbox("Use COVID-adjusted values", value=True, help="Toggle ARIMA counterfactual for the COVID window")

    st.divider()
    st.caption(f"Data range: {date_min_db} to {date_max_db}")
    st.caption(f"Observations: {run_query('SELECT COUNT(*) AS n FROM observations', use_full)['n'].iloc[0]:,}")


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown("# Macro Economic Dashboard")
st.markdown(
    "*As AI reshapes white-collar work, how do traditional recession signals "
    "interact with diverging employment trends between information-sector "
    "workers and skilled trades?*"
)

last_updated = run_query(
    "SELECT MAX(last_updated) AS lu FROM series_metadata", use_full
)["lu"].iloc[0]
if last_updated:
    st.caption(f"Data last updated: {last_updated}")

st.divider()


# ---------------------------------------------------------------------------
# Section A: Overview — KPI cards + sparklines
# ---------------------------------------------------------------------------

st.header("Overview")

# KPI cards
kpi_cols = st.columns(5)

unrate_val, unrate_delta, unrate_date = get_latest_metric("UNRATE", use_full)
with kpi_cols[0]:
    st.metric("Unemployment (U3)", f"{unrate_val:.1f}%", f"{unrate_delta:+.1f}pp MoM")

u6_val, u6_delta, u6_date = get_latest_metric("U6RATE", use_full)
with kpi_cols[1]:
    st.metric("Underemployment (U6)", f"{u6_val:.1f}%", f"{u6_delta:+.1f}pp MoM")

gdp_growth, gdp_date = get_gdp_latest_growth(use_full)
with kpi_cols[2]:
    st.metric("GDP Growth (Ann.)", f"{gdp_growth:+.1f}%", gdp_date[:7] if gdp_date else "")

t10y2y_val, t10y2y_delta, t10y2y_date = get_latest_metric("T10Y2Y", use_full)
inverted_label = "INVERTED" if t10y2y_val < 0 else "Normal"
with kpi_cols[3]:
    st.metric("Yield Spread (10Y-2Y)", f"{t10y2y_val:.2f}%", inverted_label)

cpi_yoy, cpi_date = get_cpi_yoy_latest(use_full)
with kpi_cols[4]:
    st.metric("CPI Inflation (YoY)", f"{cpi_yoy:.1f}%", cpi_date[:7] if cpi_date else "")

# Sparklines
spark_cols = st.columns(5)
spark_configs: list[tuple[str, str, str]] = [
    ("UNRATE", "Unemployment", COLORS["unemployment"]),
    ("U6RATE", "U6 Rate", COLORS["u6"]),
    ("GDPC1", "Real GDP", COLORS["gdp"]),
    ("T10Y2Y", "Yield Spread", COLORS["yield"]),
    ("CPIAUCSL", "CPI Level", COLORS["cpi"]),
]
for col, (sid, label, color) in zip(spark_cols, spark_configs):
    spark_df = get_sparkline_data(sid, 24, use_full)
    fig = go.Figure(go.Scatter(
        x=spark_df["date"], y=spark_df["value"],
        mode="lines", line=dict(color=color, width=2),
        hovertemplate="%{x|%b %Y}: %{y:.2f}<extra></extra>",
    ))
    fig.update_layout(
        height=100, margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        showlegend=False,
    )
    with col:
        st.plotly_chart(fig, use_container_width=True, key=f"spark_{sid}")

st.divider()


# ---------------------------------------------------------------------------
# Section B: Exploration — tabbed interactive charts
# ---------------------------------------------------------------------------

st.header("Exploration")

tab_recession, tab_labor, tab_broader = st.tabs([
    "Recession Signals", "AI Labor Impact", "Broader Signals"
])

# --- Tab 1: Recession Signals ---
with tab_recession:
    show_recession_tab = any(
        c in selected_categories for c in ["yield_curve", "output_growth", "labor_market", "prices"]
    )
    if not show_recession_tab:
        st.info("Select yield_curve, output_growth, labor_market, or prices categories to view this tab.")
    else:
        recession_data = get_recession_data(date_min, date_max, use_full)

        # Q1: Yield curve vs unemployment
        if "yield_curve" in selected_categories or "labor_market" in selected_categories:
            st.subheader("Yield Curve Inversions vs Unemployment")
            df_q1 = query_q1(date_min, date_max, use_full)

            if not df_q1.empty:
                fig_q1 = make_subplots(specs=[[{"secondary_y": True}]])
                # Shade inversion periods
                inv_start = None
                for _, row in df_q1.iterrows():
                    if row["inverted"] == 1 and inv_start is None:
                        inv_start = row["month"] + "-01"
                    elif row["inverted"] == 0 and inv_start is not None:
                        add_annotated_vrect(
                            fig_q1, inv_start, row["month"] + "-01",
                            fillcolor="rgba(255,200,200,0.3)",
                        )
                        inv_start = None
                if inv_start:
                    add_annotated_vrect(
                        fig_q1, inv_start, df_q1["month"].iloc[-1] + "-01",
                        fillcolor="rgba(255,200,200,0.3)",
                        text="Inverted",
                    )

                fig_q1.add_trace(
                    go.Scatter(
                        x=df_q1["month"], y=df_q1["avg_spread"],
                        name="10Y-2Y Spread", line=dict(color=COLORS["yield"]),
                        hovertemplate="%{x}: %{y:.3f}%<extra>Spread</extra>",
                    ),
                    secondary_y=False,
                )
                fig_q1.add_trace(
                    go.Scatter(
                        x=df_q1["month"], y=df_q1["unemployment_rate"],
                        name="Unemployment Rate", line=dict(color=COLORS["unemployment"]),
                        hovertemplate="%{x}: %{y:.1f}%<extra>UNRATE</extra>",
                    ),
                    secondary_y=True,
                )
                fig_q1.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, secondary_y=False)
                fig_q1.update_yaxes(title_text="Yield Spread (%)", secondary_y=False)
                fig_q1.update_yaxes(title_text="Unemployment Rate (%)", secondary_y=True)
                style_figure(fig_q1, "Yield Curve Spread vs Unemployment Rate")
                st.plotly_chart(fig_q1, use_container_width=True, key="q1")
                render_ai_insight_block("T10Y2Y_UNRATE", "correlation", use_full)

        # Q3: GDP growth with recession shading
        if "output_growth" in selected_categories:
            st.subheader("GDP Annualized Growth")
            df_q3 = query_q3(date_min, date_max, use_full)

            if not df_q3.empty:
                fig_q3 = go.Figure()
                bar_colors = [
                    COLORS["gdp"] if g >= 0 else "#EF553B"
                    for g in df_q3["annualized_growth_pct"]
                ]
                fig_q3.add_trace(go.Bar(
                    x=df_q3["date"], y=df_q3["annualized_growth_pct"],
                    marker_color=bar_colors, name="GDP Growth",
                    hovertemplate="%{x|%b %Y}: %{y:+.2f}%<extra></extra>",
                ))
                # Add recession shading from NBER
                rec_q3 = df_q3[df_q3["nber_recession"] == 1]
                if not rec_q3.empty:
                    in_rec = False
                    start = None
                    for _, row in df_q3.iterrows():
                        if row["nber_recession"] == 1 and not in_rec:
                            start = row["date"]
                            in_rec = True
                        elif row["nber_recession"] == 0 and in_rec:
                            add_annotated_vrect(
                                fig_q3, start, row["date"],
                                fillcolor=COLORS["recession"],
                            )
                            in_rec = False
                    if in_rec and start:
                        add_annotated_vrect(
                            fig_q3, start, df_q3["date"].iloc[-1],
                            fillcolor=COLORS["recession"],
                        )
                fig_q3.update_yaxes(title_text="Annualized Growth (%)")
                style_figure(fig_q3, "Real GDP Quarter-over-Quarter Annualized Growth")
                st.plotly_chart(fig_q3, use_container_width=True, key="q3")
                render_ai_insight_block("GDPC1", "trend", use_full)

        # Q8: CPI inflation
        if "prices" in selected_categories:
            st.subheader("CPI Inflation")
            df_q8 = query_q8(date_min, date_max, use_full)

            if not df_q8.empty:
                fig_q8 = make_subplots(specs=[[{"secondary_y": True}]])
                fig_q8.add_trace(
                    go.Scatter(
                        x=df_q8["date"], y=df_q8["yoy_pct_change"],
                        name="YoY %", line=dict(color=COLORS["cpi"], width=2),
                        hovertemplate="%{x|%b %Y}: %{y:.2f}%<extra>YoY</extra>",
                    ),
                    secondary_y=False,
                )
                fig_q8.add_trace(
                    go.Scatter(
                        x=df_q8["date"], y=df_q8["mom_pct_change"],
                        name="MoM %", line=dict(color=COLORS["cpi"], width=1, dash="dot"),
                        opacity=0.6,
                        hovertemplate="%{x|%b %Y}: %{y:.3f}%<extra>MoM</extra>",
                    ),
                    secondary_y=True,
                )
                fig_q8.update_yaxes(title_text="YoY Inflation (%)", secondary_y=False)
                fig_q8.update_yaxes(title_text="MoM Change (%)", secondary_y=True)
                style_figure(fig_q8, "CPI Inflation: Month-over-Month and Year-over-Year")
                st.plotly_chart(fig_q8, use_container_width=True, key="q8")
                render_ai_insight_block("CPIAUCSL", "trend", use_full)


# --- Tab 2: AI Labor Impact ---
with tab_labor:
    show_labor_tab = "ai_labor" in selected_categories or "labor_market" in selected_categories
    if not show_labor_tab:
        st.info("Select ai_labor or labor_market categories to view this tab.")
    else:
        # Q2: Info vs Trades divergence
        if "ai_labor" in selected_categories:
            st.subheader("Information Sector vs Specialty Trades (Per-Capita Indexed)")
            df_q2 = query_q2(date_min, date_max, use_full)

            if not df_q2.empty:
                fig_q2 = go.Figure()
                fig_q2.add_trace(go.Scatter(
                    x=df_q2["date"], y=df_q2["info_pc_index"],
                    name="Information Sector", line=dict(color=COLORS["info"], width=2),
                    hovertemplate="%{x|%b %Y}: %{y:.1f}<extra>Info</extra>",
                ))
                fig_q2.add_trace(go.Scatter(
                    x=df_q2["date"], y=df_q2["trades_pc_index"],
                    name="Specialty Trades", line=dict(color=COLORS["trades"], width=2),
                    hovertemplate="%{x|%b %Y}: %{y:.1f}<extra>Trades</extra>",
                ))
                fig_q2.add_trace(go.Scatter(
                    x=df_q2["date"], y=df_q2["divergence_gap"],
                    name="Divergence Gap", line=dict(color="gray", width=1, dash="dash"),
                    hovertemplate="%{x|%b %Y}: %{y:+.1f}pp<extra>Gap</extra>",
                ))
                fig_q2.add_hline(y=100, line_dash="dot", line_color="gray", opacity=0.4)
                add_annotated_vline(fig_q2, CHATGPT_LAUNCH, text="ChatGPT Launch")
                fig_q2.update_yaxes(title_text="Per-Capita Index (start = 100)")
                style_figure(fig_q2, "Employment Divergence: Info Sector vs Specialty Trades")
                st.plotly_chart(fig_q2, use_container_width=True, key="q2")
                render_ai_insight_block("USINFO_CES2023800001", "comparison", use_full)

        # Q4: Rolling 12-month growth
        if "ai_labor" in selected_categories or "labor_market" in selected_categories:
            st.subheader("Rolling 12-Month Per-Capita Employment Growth")
            df_q4 = query_q4(date_min, date_max, use_full)

            if not df_q4.empty:
                fig_q4 = go.Figure()
                color_map = {
                    "USINFO": COLORS["info"],
                    "CES2023800001": COLORS["trades"],
                    "UNRATE": COLORS["unemployment"],
                }
                for sid in df_q4["series_id"].unique():
                    sub = df_q4[df_q4["series_id"] == sid]
                    fig_q4.add_trace(go.Scatter(
                        x=sub["date"], y=sub["yoy_pct_change_pc"],
                        name=sub["series_name"].iloc[0],
                        line=dict(color=color_map.get(sid, "gray"), width=2),
                        hovertemplate="%{x|%b %Y}: %{y:+.2f}%<extra></extra>",
                    ))
                fig_q4.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
                add_annotated_vline(fig_q4, CHATGPT_LAUNCH, text="ChatGPT Launch")
                fig_q4.update_yaxes(title_text="YoY Growth (%)")
                style_figure(fig_q4, "Rolling 12-Month Per-Capita Employment Growth")
                st.plotly_chart(fig_q4, use_container_width=True, key="q4")
                render_ai_insight_block("employment_growth", "trend", use_full)

        # Q5: COVID recovery
        if "ai_labor" in selected_categories:
            st.subheader("COVID Recovery: Employment as % of Feb 2020 Peak")
            df_q5 = query_q5(date_min, date_max, use_full)

            if not df_q5.empty:
                fig_q5 = go.Figure()
                fig_q5.add_trace(go.Scatter(
                    x=df_q5["date"], y=df_q5["info_pct_of_peak"],
                    name="Information Sector", line=dict(color=COLORS["info"], width=2),
                    hovertemplate="%{x|%b %Y}: %{y:.1f}%<extra>Info</extra>",
                ))
                fig_q5.add_trace(go.Scatter(
                    x=df_q5["date"], y=df_q5["trades_pct_of_peak"],
                    name="Specialty Trades", line=dict(color=COLORS["trades"], width=2),
                    hovertemplate="%{x|%b %Y}: %{y:.1f}%<extra>Trades</extra>",
                ))
                fig_q5.add_hline(y=100, line_dash="dot", line_color="gray", opacity=0.5,
                                 annotation_text="Feb 2020 Peak", annotation_position="bottom right")
                fig_q5.update_yaxes(title_text="% of Feb 2020 Employment")
                style_figure(fig_q5, "COVID Recovery: Info Sector vs Specialty Trades")
                st.plotly_chart(fig_q5, use_container_width=True, key="q5")
                render_ai_insight_block("covid_recovery", "comparison", use_full)


# --- Tab 3: Broader Signals ---
with tab_broader:
    show_broader_tab = any(
        c in selected_categories for c in ["labor_market", "ai_energy"]
    )
    if not show_broader_tab:
        st.info("Select labor_market or ai_energy categories to view this tab.")
    else:
        # Q6: U6 vs U3 gap
        if "labor_market" in selected_categories:
            st.subheader("U6 vs U3 Unemployment Gap")
            df_q6 = query_q6(date_min, date_max, use_full)

            if not df_q6.empty:
                fig_q6 = go.Figure()
                fig_q6.add_trace(go.Scatter(
                    x=df_q6["date"], y=df_q6["u3_rate"],
                    name="U3 (Official)", line=dict(color=COLORS["unemployment"], width=2),
                    hovertemplate="%{x|%b %Y}: %{y:.1f}%<extra>U3</extra>",
                ))
                fig_q6.add_trace(go.Scatter(
                    x=df_q6["date"], y=df_q6["u6_rate"],
                    name="U6 (Broad)", line=dict(color=COLORS["u6"], width=2),
                    hovertemplate="%{x|%b %Y}: %{y:.1f}%<extra>U6</extra>",
                ))
                fig_q6.add_trace(go.Scatter(
                    x=df_q6["date"], y=df_q6["u6_u3_gap"],
                    name="U6-U3 Gap", fill="tozeroy",
                    line=dict(color="rgba(171,99,250,0.4)", width=1),
                    fillcolor="rgba(171,99,250,0.15)",
                    hovertemplate="%{x|%b %Y}: %{y:.2f}pp<extra>Gap</extra>",
                ))
                fig_q6.update_yaxes(title_text="Rate (%)")
                style_figure(fig_q6, "Hidden Labor Market Slack: U6 vs U3 Gap")
                st.plotly_chart(fig_q6, use_container_width=True, key="q6")
                render_ai_insight_block("U6_U3", "trend", use_full)

        # Q7: Power vs Info employment
        if "ai_energy" in selected_categories:
            st.subheader("Electric Power Output vs Info Sector Employment")
            df_q7 = query_q7(date_min, date_max, use_full)

            if not df_q7.empty:
                fig_q7 = make_subplots(specs=[[{"secondary_y": True}]])
                fig_q7.add_trace(
                    go.Scatter(
                        x=df_q7["date"], y=df_q7["power_index"],
                        name="Electric Power (Index)", line=dict(color=COLORS["power"], width=2),
                        hovertemplate="%{x|%b %Y}: %{y:.1f}<extra>Power</extra>",
                    ),
                    secondary_y=False,
                )
                fig_q7.add_trace(
                    go.Scatter(
                        x=df_q7["date"], y=df_q7["info_index"],
                        name="Info Employment (Index)", line=dict(color=COLORS["info"], width=2),
                        hovertemplate="%{x|%b %Y}: %{y:.1f}<extra>Info</extra>",
                    ),
                    secondary_y=True,
                )
                add_annotated_vline(fig_q7, CHATGPT_LAUNCH, text="ChatGPT Launch")
                fig_q7.add_hline(y=100, line_dash="dot", line_color="gray", opacity=0.4, secondary_y=False)
                fig_q7.update_yaxes(title_text="Power Index (start=100)", secondary_y=False)
                fig_q7.update_yaxes(title_text="Info Employment Index (start=100)", secondary_y=True)
                style_figure(fig_q7, "AI Energy Paradox: Power Output vs Info Employment")
                st.plotly_chart(fig_q7, use_container_width=True, key="q7")
                render_ai_insight_block("IPG2211S_USINFO", "correlation", use_full)

st.divider()


# ---------------------------------------------------------------------------
# Section C: Deep Dive — the core question
# ---------------------------------------------------------------------------

st.header("Deep Dive: AI's Reshaping of Work")

st.markdown(
    """
As AI tools become embedded in knowledge work, the macro data tells a story of divergence.
Information-sector employment, once a reliable growth engine, has decoupled from the broader
economy. Meanwhile, skilled trades that require physical presence and hands-on expertise
continue to expand. The question isn't whether AI is changing the labor market. It's whether
traditional recession indicators still capture what's happening underneath.
"""
)

# Side-by-side comparison
deep_col1, deep_col2 = st.columns(2)

df_q2_deep = query_q2(date_min, date_max, use_full)
df_q7_deep = query_q7(date_min, date_max, use_full)

with deep_col1:
    st.subheader("Employment Divergence")
    if not df_q2_deep.empty:
        fig_deep1 = go.Figure()
        fig_deep1.add_trace(go.Scatter(
            x=df_q2_deep["date"], y=df_q2_deep["info_pc_index"],
            name="Info Sector", line=dict(color=COLORS["info"], width=2),
        ))
        fig_deep1.add_trace(go.Scatter(
            x=df_q2_deep["date"], y=df_q2_deep["trades_pc_index"],
            name="Specialty Trades", line=dict(color=COLORS["trades"], width=2),
        ))
        add_annotated_vline(fig_deep1, CHATGPT_LAUNCH)
        fig_deep1.add_hline(y=100, line_dash="dot", line_color="gray", opacity=0.4)
        fig_deep1.update_yaxes(title_text="Per-Capita Index")
        style_figure(fig_deep1, "Info vs Trades (Per-Capita Indexed)")
        st.plotly_chart(fig_deep1, use_container_width=True, key="deep_q2")

with deep_col2:
    st.subheader("Energy Paradox")
    if not df_q7_deep.empty:
        fig_deep2 = go.Figure()
        fig_deep2.add_trace(go.Scatter(
            x=df_q7_deep["date"], y=df_q7_deep["power_index"],
            name="Electric Power", line=dict(color=COLORS["power"], width=2),
        ))
        fig_deep2.add_trace(go.Scatter(
            x=df_q7_deep["date"], y=df_q7_deep["info_index"],
            name="Info Employment", line=dict(color=COLORS["info"], width=2),
        ))
        add_annotated_vline(fig_deep2, CHATGPT_LAUNCH)
        fig_deep2.add_hline(y=100, line_dash="dot", line_color="gray", opacity=0.4)
        fig_deep2.update_yaxes(title_text="Index (start=100)")
        style_figure(fig_deep2, "Power Output vs Info Employment")
        st.plotly_chart(fig_deep2, use_container_width=True, key="deep_q7")

# Key findings
st.subheader("Key Findings")

# Compute dynamic findings from the data
findings: list[str] = []

if not df_q2_deep.empty:
    latest_q2 = df_q2_deep.iloc[-1]
    info_change = latest_q2["info_pc_index"] - 100
    trades_change = latest_q2["trades_pc_index"] - 100
    findings.append(
        f"Info sector **{info_change:+.1f}%** per capita vs trades **{trades_change:+.1f}%** "
        f"since {df_q2_deep['date'].iloc[0][:7]}"
    )

if not df_q7_deep.empty:
    # Post-ChatGPT divergence
    post_gpt = df_q7_deep[df_q7_deep["date"] >= CHATGPT_LAUNCH]
    if not post_gpt.empty:
        power_delta = post_gpt["power_index"].iloc[-1] - post_gpt["power_index"].iloc[0]
        info_delta = post_gpt["info_index"].iloc[-1] - post_gpt["info_index"].iloc[0]
        findings.append(
            f"Since ChatGPT launch: power output **{power_delta:+.1f}pp** while info employment "
            f"**{info_delta:+.1f}pp**"
        )

df_q1_deep = query_q1(date_min, date_max, use_full)
if not df_q1_deep.empty:
    inv_months = df_q1_deep[df_q1_deep["inverted"] == 1]
    if not inv_months.empty:
        # Find longest consecutive inversion streak
        inv_sorted = inv_months["month"].sort_values().tolist()
        max_streak = 1
        current_streak = 1
        for i in range(1, len(inv_sorted)):
            # Check if consecutive month
            prev_y, prev_m = int(inv_sorted[i - 1][:4]), int(inv_sorted[i - 1][5:7])
            curr_y, curr_m = int(inv_sorted[i][:4]), int(inv_sorted[i][5:7])
            expected_m = prev_m + 1 if prev_m < 12 else 1
            expected_y = prev_y if prev_m < 12 else prev_y + 1
            if curr_y == expected_y and curr_m == expected_m:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 1
        findings.append(
            f"Yield curve inverted **{max_streak}** consecutive months (longest streak in dataset)"
        )

df_q6_deep = query_q6(date_min, date_max, use_full)
if not df_q6_deep.empty:
    recent_q6 = df_q6_deep[df_q6_deep["date"] >= "2024-01-01"]
    if not recent_q6.empty and "u6_yoy_change" in recent_q6.columns and "u3_yoy_change" in recent_q6.columns:
        valid = recent_q6.dropna(subset=["u6_yoy_change", "u3_yoy_change"])
        if not valid.empty:
            u6_faster = (valid["u6_yoy_change"] > valid["u3_yoy_change"]).sum()
            total = len(valid)
            findings.append(
                f"U6 rising faster than U3 in **{u6_faster}/{total}** months since Jan 2024"
            )

if findings:
    findings_md = "\n".join(f"- {f}" for f in findings)
    st.info(findings_md)

# Recommendations
st.subheader("Takeaways")
st.markdown(
    """
1. **Traditional recession signals need context.** The yield curve inversion signaled stress,
   but the labor market divergence shows the impact is sector-specific rather than broad-based.

2. **Per-capita normalization matters.** Raw employment figures mask population-driven growth.
   When you control for working-age population, the info sector's decline is starker.

3. **Watch the U6-U3 gap.** If AI pushes knowledge workers into underemployment rather than
   outright unemployment, U3 alone will understate labor market slack.

4. **Energy demand tells the supply-side story.** Rising power output alongside falling info
   employment suggests AI is creating infrastructure demand while reducing the workforce that
   builds on it.
"""
)

render_ai_insight_block("synthesis", "trend", use_full)

st.divider()
st.caption("Data source: FRED (Federal Reserve Economic Data) | Built with Streamlit + Plotly")
