"""Macro Economic Dashboard: Recession Risk and AI's Labor Market Impact.

Streamlit dashboard that queries seed.db (or full.db) directly, surfaces
interactive Plotly charts for 8 analysis queries, and renders AI-verified
insight blocks when available.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

try:
    from dashboard.ask_the_data import render_ask_the_data

    _ASK_THE_DATA_AVAILABLE = True
except ImportError:
    _ASK_THE_DATA_AVAILABLE = False

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
        and citations_json, or None if no matching row exists.
    """
    df = run_query(
        f"""
        SELECT narrative, claims_json, verification_json, all_verified,
               citations_json
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
        "citations_json": row["citations_json"] or "[]",
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
    start: str = ""
    for _, row in df_recession.iterrows():
        if row["in_recession"] == 1 and not in_rec:
            start = str(row["date"])
            in_rec = True
        elif row["in_recession"] == 0 and in_rec:
            add_annotated_vrect(
                fig, start, str(row["date"]),
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


def style_figure(
    fig: go.Figure, title: str = "", height: int = 450
) -> go.Figure:
    """Apply consistent styling to a Plotly figure.

    Args:
        fig: The Plotly figure to style.
        title: Optional chart title.
        height: Chart height in pixels.

    Returns:
        The styled figure.
    """
    fig.update_layout(
        title=dict(text=title, y=0.98, yanchor="top"),
        template="plotly_white",
        hovermode="x unified",
        height=height,
        margin=dict(l=50, r=20, t=80, b=30),
        legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
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
        with st.expander("AI Insight", expanded=False):
            st.caption("No AI-generated insight available for this section yet.")
        return

    # Determine three-tier verification status from individual claims
    try:
        claims = json.loads(insight["claims_json"])
        verification = json.loads(insight["verification_json"])
        # Dual-key lookup: verification keys may be strings ("0") or ints (0)
        # depending on JSON serialization path
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

    # Parse citations for the References section (Phase 11)
    try:
        citations: list[dict] = json.loads(insight.get("citations_json", "[]"))
    except (json.JSONDecodeError, TypeError):
        citations = []

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

            if citations:
                st.markdown("---")
                st.markdown("**Reference Citations**")
                for cit in citations:
                    ref_id: int = int(cit.get("ref_id", 0))
                    title: str = cit.get("title", f"Reference {ref_id}")
                    excerpt: str = cit.get("excerpt", "")
                    source_url: str | None = cit.get("source_url")
                    st.markdown(f"**{title}**")
                    if excerpt:
                        st.markdown(f"> {excerpt}")
                    if source_url:
                        st.markdown(f"Source: [{source_url}]({source_url})")
                    st.markdown("")


# ---------------------------------------------------------------------------
# Recession Risk data layer
# ---------------------------------------------------------------------------

FEATURE_LABELS: dict[str, str] = {
    "yield_spread": "Yield Curve Spread",
    "yield_spread_3m_avg": "Yield Spread (3-mo Avg)",
    "yield_inverted_months": "Months Inverted (trailing 6)",
    "unrate": "Unemployment Rate",
    "unrate_12m_change": "Unemployment 12-mo Change",
    "u6_u3_gap": "U6-U3 Gap",
    "gdp_growth_annualized": "GDP Growth (Annualized)",
    "cpi_yoy": "CPI Inflation (YoY)",
    "info_trades_divergence": "Info vs Trades Divergence",
    "info_employment_yoy": "Info Employment YoY Growth",
    "power_output_yoy": "Power Output YoY Growth",
    # Phase 14 — HN-derived features
    "hn_sentiment_3m_avg": "HN Sentiment (3-mo avg)",
    "hn_story_volume_yoy": "HN Story Volume YoY",
    "layoff_story_freq": "Layoff Story Share",
}

# Heuristics for recession signal coloring (red = recession-like)
RECESSION_SIGNAL_RULES: dict[str, str] = {
    "yield_spread": "negative",        # < 0 means inverted curve
    "yield_inverted_months": "positive",  # > 0 means months inverted
    "unrate_12m_change": "positive",    # > 0 means rising unemployment
    "gdp_growth_annualized": "negative",  # < 0 means contracting
    "info_employment_yoy": "negative",  # < 0 means info sector shrinking
}
# u6_u3_gap uses median comparison, handled separately


def _table_exists(table_name: str, full: bool = False) -> bool:
    """Check whether a table exists in the database.

    Args:
        table_name: Name of the table to check.
        full: Whether to use full.db instead of seed.db.

    Returns:
        True if the table exists.
    """
    conn = get_connection(full)
    try:
        result = conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,),
        ).fetchone()
        return result[0] > 0
    finally:
        conn.close()


@st.cache_data(ttl=300)
def query_predictions(
    date_min: str, date_max: str, full: bool = False
) -> pd.DataFrame:
    """Query recession predictions for the timeline chart.

    Args:
        date_min: Start date filter (ISO string).
        date_max: End date filter (ISO string).
        full: Whether to use full.db instead of seed.db.

    Returns:
        DataFrame with date, probability, prediction, actual, model_name.
    """
    return run_query(
        f"""
        SELECT date, probability, prediction, actual, model_name
        FROM recession_predictions
        WHERE date BETWEEN '{date_min}' AND '{date_max}'
        ORDER BY date
        """,
        full,
    )


@st.cache_data(ttl=300)
def get_latest_prediction(full: bool = False) -> dict | None:
    """Get the most recent prediction row with parsed features.

    Args:
        full: Whether to use full.db instead of seed.db.

    Returns:
        Dict with date, probability, prediction, model_name, features
        (parsed dict), or None if no predictions exist.
    """
    df = run_query(
        """
        SELECT date, probability, prediction, model_name, features_json
        FROM recession_predictions
        ORDER BY date DESC
        LIMIT 1
        """,
        full,
    )
    if df.empty:
        return None
    row = df.iloc[0]
    return {
        "date": row["date"],
        "probability": row["probability"],
        "prediction": row["prediction"],
        "model_name": row["model_name"],
        "features": json.loads(row["features_json"]),
    }


@st.cache_data(ttl=300)
def query_scenario_grid(full: bool = False) -> pd.DataFrame:
    """Load the entire scenario_grid table for nearest-neighbor lookup.

    Args:
        full: Whether to use full.db instead of seed.db.

    Returns:
        DataFrame with yield_spread, unrate, gdp_growth_annualized,
        cpi_yoy, probability, model_name columns.
    """
    return run_query(
        """
        SELECT yield_spread, unrate, gdp_growth_annualized,
               cpi_yoy, probability, model_name
        FROM scenario_grid
        """,
        full,
    )


def get_feature_signal_color(feature: str, value: float, u6_u3_median: float = 3.5) -> str:
    """Determine recession signal color for a feature value.

    Args:
        feature: Feature column name.
        value: Current feature value.
        u6_u3_median: Historical median for u6_u3_gap comparison.

    Returns:
        Color string: red (recession-like), green (healthy), or gray (neutral).
    """
    if feature == "u6_u3_gap":
        return "red" if value > u6_u3_median else "green"

    rule = RECESSION_SIGNAL_RULES.get(feature)
    if rule == "negative":
        return "red" if value < 0 else "green"
    if rule == "positive":
        return "red" if value > 0 else "green"
    return "gray"


def find_nearest_scenario(
    grid_df: pd.DataFrame,
    yield_spread: float,
    unrate: float,
    gdp_growth: float,
    cpi_yoy: float,
) -> pd.Series:
    """Find the nearest scenario in the grid using normalized Euclidean distance.

    Args:
        grid_df: Scenario grid DataFrame.
        yield_spread: Slider value for yield spread.
        unrate: Slider value for unemployment rate.
        gdp_growth: Slider value for GDP growth.
        cpi_yoy: Slider value for CPI YoY.

    Returns:
        The nearest row from the grid as a pandas Series.
    """
    slider_cols = ["yield_spread", "unrate", "gdp_growth_annualized", "cpi_yoy"]
    slider_vals = np.array([yield_spread, unrate, gdp_growth, cpi_yoy])

    # Min-max normalize for scale-invariant distance
    grid_vals = grid_df[slider_cols].values
    col_min = grid_vals.min(axis=0)
    col_max = grid_vals.max(axis=0)
    col_range = col_max - col_min
    col_range[col_range == 0] = 1.0  # avoid division by zero

    grid_norm = (grid_vals - col_min) / col_range
    slider_norm = (slider_vals - col_min) / col_range

    distances = np.sqrt(((grid_norm - slider_norm) ** 2).sum(axis=1))
    nearest_idx = distances.argmin()
    return grid_df.iloc[nearest_idx]


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
def query_hn_sentiment_overlay(
    date_min: str, date_max: str, full: bool = False
) -> pd.DataFrame:
    """HN sentiment 3-month rolling avg alongside USINFO per-capita index.

    Drops the trailing partial-month row (current month) so the chart
    does not show incomplete data. Returns an empty DataFrame when
    ``hn_sentiment_monthly`` is empty or not yet present.

    Args:
        date_min: Start date filter.
        date_max: End date filter.
        full: Whether to use ``full.db``.

    Returns:
        DataFrame with columns ``month``, ``mean_sentiment_3m_avg``,
        ``info_pc_index``. Empty when no HN data falls in the range.
    """
    from datetime import datetime, timezone  # noqa: PLC0415

    conn: sqlite3.Connection = get_connection(full)
    try:
        # Check table existence first to avoid errors in pre-Phase-13 DBs.
        exists: int = conn.execute(
            "SELECT COUNT(*) FROM sqlite_master "
            "WHERE type='table' AND name='hn_sentiment_monthly'"
        ).fetchone()[0]
        if not exists:
            return pd.DataFrame(
                columns=["month", "mean_sentiment_3m_avg", "info_pc_index"]
            )

        hn: pd.DataFrame = pd.read_sql_query(
            "SELECT SUBSTR(month, 1, 7) AS month, mean_sentiment "
            "FROM hn_sentiment_monthly ORDER BY month",
            conn,
        )
        if hn.empty:
            return pd.DataFrame(
                columns=["month", "mean_sentiment_3m_avg", "info_pc_index"]
            )

        hn["mean_sentiment_3m_avg"] = (
            hn["mean_sentiment"].rolling(3, min_periods=1).mean()
        )

        # USINFO per-capita index using same convention as query_q2.
        info: pd.DataFrame = pd.read_sql_query(
            """
            SELECT
                SUBSTR(o.date, 1, 7) AS month,
                o.value_covid_adjusted / p.value_covid_adjusted * 1000 AS info_pc
            FROM observations o
            JOIN observations p
              ON p.series_id = 'CNP16OV'
              AND SUBSTR(p.date, 1, 7) = SUBSTR(o.date, 1, 7)
            WHERE o.series_id = 'USINFO'
            ORDER BY month
            """,
            conn,
        )
    finally:
        conn.close()

    if info.empty:
        return pd.DataFrame(
            columns=["month", "mean_sentiment_3m_avg", "info_pc_index"]
        )

    # Index info_pc to 100 at the earliest month in the HN window.
    hn_start: str = hn["month"].iloc[0]
    merged: pd.DataFrame = info.merge(hn, on="month", how="inner")
    if merged.empty:
        return pd.DataFrame(
            columns=["month", "mean_sentiment_3m_avg", "info_pc_index"]
        )
    base_row = merged.loc[merged["month"] >= hn_start, "info_pc"]
    base: float = float(base_row.iloc[0]) if not base_row.empty else 1.0
    merged["info_pc_index"] = merged["info_pc"] / base * 100

    # Mask partial current month.
    today_ym: str = datetime.now(timezone.utc).strftime("%Y-%m")
    merged = merged[merged["month"] != today_ym]

    # Apply date range clipping.
    effective_min: str = max(date_min[:7], hn_start)
    merged = merged[
        (merged["month"] >= effective_min) & (merged["month"] <= date_max[:7])
    ]

    return merged[["month", "mean_sentiment_3m_avg", "info_pc_index"]]


# ---------------------------------------------------------------------------
# Phase 15: NLP Analysis query helpers
# ---------------------------------------------------------------------------


@st.cache_data(ttl=300)
def query_topic_distribution(
    date_min: str, date_max: str, full: bool = False
) -> pd.DataFrame:
    """Topic distribution over time for Chart 1 (stacked area).

    Args:
        date_min: Start date filter.
        date_max: End date filter.
        full: Whether to use ``full.db``.

    Returns:
        DataFrame with columns ``topic_id``, ``label``, ``month``,
        ``story_count``.
    """
    from datetime import datetime, timezone  # noqa: PLC0415

    effective_min: str = max(date_min[:7], "2022-01")
    conn: sqlite3.Connection = get_connection(full)
    try:
        exists: int = conn.execute(
            "SELECT COUNT(*) FROM sqlite_master "
            "WHERE type='table' AND name='hn_topic_assignments'"
        ).fetchone()[0]
        if not exists:
            return pd.DataFrame(
                columns=["topic_id", "label", "month", "story_count"]
            )
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
    finally:
        conn.close()

    if df.empty:
        return df

    today_ym: str = datetime.now(timezone.utc).strftime("%Y-%m")
    df = df[df["month"] != today_ym]
    df = df[
        (df["month"] >= effective_min) & (df["month"] <= date_max[:7])
    ]
    return df


@st.cache_data(ttl=300)
def query_sentiment_by_topic(full: bool = False) -> pd.DataFrame:
    """Sentiment distribution per topic for Chart 2 (box plot).

    Args:
        full: Whether to use ``full.db``.

    Returns:
        DataFrame with columns ``label``, ``sentiment_score``.
    """
    conn: sqlite3.Connection = get_connection(full)
    try:
        exists: int = conn.execute(
            "SELECT COUNT(*) FROM sqlite_master "
            "WHERE type='table' AND name='hn_topic_assignments'"
        ).fetchone()[0]
        if not exists:
            return pd.DataFrame(columns=["label", "sentiment_score"])
        df: pd.DataFrame = pd.read_sql_query(
            """
            SELECT t.label, s.sentiment_score
            FROM hn_topic_assignments ta
            JOIN hn_topics t ON ta.topic_id = t.topic_id
            JOIN hn_stories s ON ta.story_id = s.story_id
            """,
            conn,
        )
    finally:
        conn.close()
    return df


@st.cache_data(ttl=300)
def query_layoff_vs_u6u3(
    date_min: str, date_max: str, full: bool = False
) -> pd.DataFrame:
    """Layoff story volume vs U6-U3 gap for Chart 3 (dual-axis).

    Args:
        date_min: Start date filter.
        date_max: End date filter.
        full: Whether to use ``full.db``.

    Returns:
        DataFrame with columns ``month``, ``layoff_story_count``,
        ``u6_u3_gap``.
    """
    from datetime import datetime, timezone  # noqa: PLC0415

    effective_min: str = max(date_min[:7], "2022-01")
    conn: sqlite3.Connection = get_connection(full)
    try:
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
    finally:
        conn.close()

    merged: pd.DataFrame = hn.merge(gap, on="month", how="inner")
    if merged.empty:
        return pd.DataFrame(
            columns=["month", "layoff_story_count", "u6_u3_gap"]
        )

    today_ym: str = datetime.now(timezone.utc).strftime("%Y-%m")
    merged = merged[merged["month"] != today_ym]
    merged = merged[
        (merged["month"] >= effective_min) & (merged["month"] <= date_max[:7])
    ]
    return merged


@st.cache_data(ttl=300)
def query_topic_sentiment_vs_usinfo(
    date_min: str, date_max: str, full: bool = False
) -> pd.DataFrame:
    """Topic sentiment vs USINFO per-capita index for Chart 4 (dual-axis).

    Returns the USINFO per-capita index (indexed to 100 at the earliest
    HN month) and mean monthly sentiment for the top-2 most populated
    topics.

    Args:
        date_min: Start date filter.
        date_max: End date filter.
        full: Whether to use ``full.db``.

    Returns:
        DataFrame with columns ``month``, ``info_pc_index``, and one
        column per top-2 topic label containing monthly mean sentiment.
    """
    from datetime import datetime, timezone  # noqa: PLC0415

    effective_min: str = max(date_min[:7], "2022-01")
    conn: sqlite3.Connection = get_connection(full)
    try:
        # Top-2 topics by story_count
        top2: list[tuple[int, str]] = conn.execute(
            "SELECT topic_id, label FROM hn_topics "
            "ORDER BY story_count DESC LIMIT 2"
        ).fetchall()
        if not top2:
            return pd.DataFrame(columns=["month", "info_pc_index"])

        top2_ids: list[int] = [t[0] for t in top2]
        top2_labels: list[str] = [t[1] for t in top2]
        placeholders: str = ",".join("?" * len(top2_ids))

        # Monthly sentiment per top-2 topic
        topic_sent: pd.DataFrame = pd.read_sql_query(
            f"""
            SELECT t.label, SUBSTR(s.month, 1, 7) AS month,
                   AVG(s.sentiment_score) AS avg_sent
            FROM hn_topic_assignments ta
            JOIN hn_topics t ON ta.topic_id = t.topic_id
            JOIN hn_stories s ON ta.story_id = s.story_id
            WHERE ta.topic_id IN ({placeholders})
            GROUP BY t.label, SUBSTR(s.month, 1, 7)
            ORDER BY month
            """,
            conn,
            params=top2_ids,
        )

        # USINFO per-capita index
        info: pd.DataFrame = pd.read_sql_query(
            """
            SELECT
                SUBSTR(o.date, 1, 7) AS month,
                o.value_covid_adjusted / p.value_covid_adjusted * 1000 AS info_pc
            FROM observations o
            JOIN observations p
              ON p.series_id = 'CNP16OV'
              AND SUBSTR(p.date, 1, 7) = SUBSTR(o.date, 1, 7)
            WHERE o.series_id = 'USINFO'
            ORDER BY month
            """,
            conn,
        )
    finally:
        conn.close()

    if topic_sent.empty or info.empty:
        return pd.DataFrame(columns=["month", "info_pc_index"])

    # Pivot topic sentiment: one column per topic label
    pivot: pd.DataFrame = topic_sent.pivot_table(
        index="month", columns="label", values="avg_sent"
    ).reset_index()

    # Index info_pc to 100 at earliest HN month
    hn_start: str = pivot["month"].iloc[0]
    base_row = info.loc[info["month"] >= hn_start, "info_pc"]
    base: float = float(base_row.iloc[0]) if not base_row.empty else 1.0
    info["info_pc_index"] = info["info_pc"] / base * 100

    merged: pd.DataFrame = info[["month", "info_pc_index"]].merge(
        pivot, on="month", how="inner"
    )

    today_ym: str = datetime.now(timezone.utc).strftime("%Y-%m")
    merged = merged[merged["month"] != today_ym]
    merged = merged[
        (merged["month"] >= effective_min) & (merged["month"] <= date_max[:7])
    ]
    return merged


@st.cache_data(ttl=300)
def query_ngram_trends(
    date_min: str, date_max: str, full: bool = False
) -> pd.DataFrame:
    """Monthly bigram frequencies for the n-gram trending section.

    Args:
        date_min: Start date filter.
        date_max: End date filter.
        full: Whether to use ``full.db``.

    Returns:
        DataFrame with columns ``month``, ``ngram``, ``count``.
    """
    effective_min: str = max(date_min[:7], "2022-01")
    conn: sqlite3.Connection = get_connection(full)
    try:
        exists: int = conn.execute(
            "SELECT COUNT(*) FROM sqlite_master "
            "WHERE type='table' AND name='hn_ngram_monthly'"
        ).fetchone()[0]
        if not exists:
            return pd.DataFrame(columns=["month", "ngram", "count"])
        df: pd.DataFrame = pd.read_sql_query(
            "SELECT SUBSTR(month, 1, 7) AS month, ngram, count "
            "FROM hn_ngram_monthly ORDER BY month, count DESC",
            conn,
        )
    finally:
        conn.close()

    if df.empty:
        return df

    df = df[
        (df["month"] >= effective_min) & (df["month"] <= date_max[:7])
    ]
    return df


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


@st.cache_data(ttl=300)
def get_gdp_growth_sparkline(months: int = 24, full: bool = False) -> pd.DataFrame:
    """Get recent GDP annualized growth rates for a sparkline.

    Args:
        months: Number of recent quarters to include.
        full: Whether to use full.db.

    Returns:
        DataFrame with date and value columns (annualized growth %).
    """
    df = run_query(
        """
        SELECT date,
            ROUND((value_covid_adjusted / LAG(value_covid_adjusted)
                OVER (ORDER BY date) - 1) * 400, 2) AS value
        FROM observations
        WHERE series_id = 'GDPC1'
        ORDER BY date
        """,
        full,
    )
    df = df.dropna(subset=["value"])
    return df.tail(months)


@st.cache_data(ttl=300)
def get_cpi_yoy_sparkline(months: int = 24, full: bool = False) -> pd.DataFrame:
    """Get recent CPI year-over-year inflation rates for a sparkline.

    Args:
        months: Number of recent months to include.
        full: Whether to use full.db.

    Returns:
        DataFrame with date and value columns (YoY %).
    """
    df = run_query(
        """
        SELECT date,
            ROUND((value_covid_adjusted / LAG(value_covid_adjusted, 12)
                OVER (ORDER BY date) - 1) * 100, 2) AS value
        FROM observations
        WHERE series_id = 'CPIAUCSL'
        ORDER BY date
        """,
        full,
    )
    df = df.dropna(subset=["value"])
    return df.tail(months)


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Macro Economic Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
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

    default_start = max(
        pd.to_datetime("2022-01-01").date(),
        pd.to_datetime(date_min_db).date(),
    )
    date_range = st.date_input(
        "Date range",
        value=(default_start, pd.to_datetime(date_max_db).date()),
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

    # Prediction model info
    if _table_exists("recession_predictions", use_full):
        pred_meta = run_query(
            """
            SELECT model_name, MAX(generated_at) AS last_gen
            FROM recession_predictions
            """,
            use_full,
        )
        if not pred_meta.empty and pred_meta["last_gen"].iloc[0]:
            st.caption(f"Predictions: {pred_meta['model_name'].iloc[0]}")
            st.caption(f"Model run: {pred_meta['last_gen'].iloc[0][:10]}")


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

render_ai_insight_block("dashboard_intro", "trend", use_full)

# Inline glossary of economic terms
GLOSSARY_TERMS: list[tuple[str, str]] = [
    (
        "U3 (Official Unemployment Rate)",
        "Percentage of the labor force that is jobless and actively seeking "
        "work. The most commonly cited unemployment figure.",
    ),
    (
        "U6 (Broad Unemployment Rate)",
        "Includes U3 plus discouraged workers and those working part-time "
        "for economic reasons. A wider measure of labor market slack.",
    ),
    (
        "U6-U3 Gap",
        "The difference between U6 and U3. A widening gap signals rising "
        "underemployment even when headline unemployment looks stable.",
    ),
    (
        "Yield Curve / Yield Spread (10Y-2Y)",
        "The difference between 10-year and 2-year Treasury yields. "
        "Normally positive; when investors demand more for short-term "
        "risk, it can invert.",
    ),
    (
        "Yield Curve Inversion",
        "When the 10Y-2Y spread turns negative. Historically precedes "
        "recessions by 6-18 months, though timing varies.",
    ),
    (
        "GDP (Gross Domestic Product)",
        "Total value of goods and services produced in the U.S. The "
        "broadest measure of economic output.",
    ),
    (
        "Hacker News Sentiment Signal",
        "Compound sentiment score (-1 to +1) computed by a transformer "
        "model on titles and excerpts of the top Hacker News stories "
        "per month tagged AI hiring, layoffs, or careers. Reflects a "
        "self-selected audience of tech practitioners, not a "
        "representative labor-market sentiment measure.",
    ),
    (
        "Annualized Growth Rate",
        "A quarterly GDP change scaled to a full year. A 0.5% quarterly "
        "gain reports as roughly 2% annualized.",
    ),
    (
        "CPI (Consumer Price Index)",
        "Tracks the average price change for a basket of consumer goods "
        "and services. The primary inflation gauge.",
    ),
    (
        "YoY (Year-over-Year)",
        "Compares a value to the same month one year earlier. Removes "
        "seasonal effects for cleaner trend reading.",
    ),
    (
        "MoM (Month-over-Month)",
        "Compares a value to the prior month. More volatile than YoY "
        "but captures turning points faster.",
    ),
    (
        "Per-Capita Normalization",
        "Dividing employment counts by working-age population (CNP16OV). "
        "Controls for population growth so sector trends reflect real "
        "demand shifts, not demographic drift.",
    ),
    (
        "NBER Recession",
        "A period of significant economic decline as designated by the "
        "National Bureau of Economic Research. The official U.S. "
        "recession arbiter.",
    ),
    (
        "ARIMA / COVID Adjustment",
        "An ARIMA model trained on pre-COVID data fills in a "
        "counterfactual for the Mar 2020 - Jan 2022 shock window. "
        "Prevents the COVID outlier from distorting trend calculations.",
    ),
    (
        "Recession Risk Score",
        "The model's output (0-1) indicating how closely current "
        "conditions resemble historical pre-recession patterns. Not a "
        "calibrated probability. Higher = more recession-like.",
    ),
    (
        "FRED (Federal Reserve Economic Data)",
        "A database of 800,000+ economic time series maintained by the "
        "Federal Reserve Bank of St. Louis. The data source for this "
        "dashboard.",
    ),
    (
        "NMF (Non-negative Matrix Factorization)",
        "A topic modeling technique that decomposes a document-term matrix "
        "into non-negative factors, producing interpretable topic clusters. "
        "Used here to group HN stories by theme.",
    ),
    (
        "TF-IDF (Term Frequency-Inverse Document Frequency)",
        "A text weighting scheme that scores words higher when they appear "
        "frequently in a document but rarely across the corpus. Reduces "
        "the influence of common words like 'the' and 'is'.",
    ),
    (
        "Topic Model",
        "An unsupervised NLP method that discovers latent themes in a "
        "collection of documents. Each topic is characterized by its "
        "highest-weighted terms.",
    ),
    (
        "Ask the Data",
        "A conversational AI feature that lets you ask natural-language "
        "questions about the economic data. Answers are fact-checked "
        "against the database and cited with reference sources.",
    ),
    (
        "RAG (Retrieval-Augmented Generation)",
        "A technique that retrieves relevant documents from a knowledge "
        "base before generating an answer. Grounds the LLM's response "
        "in real data instead of relying on memorized training patterns.",
    ),
]

with st.expander("Glossary of Economic Terms"):
    g_col1, g_col2 = st.columns(2)
    midpoint: int = (len(GLOSSARY_TERMS) + 1) // 2
    for i, (term, definition) in enumerate(GLOSSARY_TERMS):
        target = g_col1 if i < midpoint else g_col2
        with target:
            st.markdown(f"**{term}**  \n{definition}")

st.divider()


# ---------------------------------------------------------------------------
# Section A: Overview — KPI cards + sparklines
# ---------------------------------------------------------------------------

st.header("Overview")

# KPI data
unrate_val, unrate_delta, unrate_date = get_latest_metric("UNRATE", use_full)
u6_val, u6_delta, u6_date = get_latest_metric("U6RATE", use_full)
gdp_growth, gdp_date = get_gdp_latest_growth(use_full)
t10y2y_val, t10y2y_delta, t10y2y_date = get_latest_metric("T10Y2Y", use_full)
inverted_label = "INVERTED" if t10y2y_val < 0 else "Normal"
cpi_yoy, cpi_date = get_cpi_yoy_latest(use_full)

# Sparkline helper
spark_configs: dict[str, tuple[str, str]] = {
    "UNRATE": ("Unemployment", COLORS["unemployment"]),
    "U6RATE": ("U6 Rate", COLORS["u6"]),
    "GDPC1": ("Real GDP", COLORS["gdp"]),
    "T10Y2Y": ("Yield Spread", COLORS["yield"]),
    "CPIAUCSL": ("CPI Level", COLORS["cpi"]),
}


def _render_sparkline(series_id: str) -> None:
    """Render a sparkline mini-chart for the given series.

    Uses derived percentage data for GDP (annualized growth) and CPI
    (YoY inflation) so the sparkline matches the KPI card metric.

    Args:
        series_id: FRED series identifier.
    """
    label, color = spark_configs[series_id]

    # Use derived % series for GDP and CPI to match KPI card values
    if series_id == "GDPC1":
        spark_df = get_gdp_growth_sparkline(24, use_full)
        hover_fmt = "%{x|%b %Y}: %{y:+.1f}%<extra></extra>"
    elif series_id == "CPIAUCSL":
        spark_df = get_cpi_yoy_sparkline(24, use_full)
        hover_fmt = "%{x|%b %Y}: %{y:.1f}%<extra></extra>"
    else:
        spark_df = get_sparkline_data(series_id, 24, use_full)
        hover_fmt = "%{x|%b %Y}: %{y:.2f}<extra></extra>"

    fig = go.Figure(go.Scatter(
        x=spark_df["date"], y=spark_df["value"],
        mode="lines", line=dict(color=color, width=2),
        hovertemplate=hover_fmt,
    ))
    show_zero = series_id in ("GDPC1", "T10Y2Y", "CPIAUCSL")
    if show_zero:
        fig.add_hline(y=0, line_color="#333333", line_width=1, opacity=0.8)
        fig.add_annotation(
            x=0, xref="paper", xanchor="right",
            y=0, yref="y",
            text="<b>0</b>", showarrow=False,
            font=dict(size=11, color="#333333"),
            xshift=-4,
        )
    fig.update_layout(
        height=100,
        margin=dict(l=20 if show_zero else 0, r=0, t=0, b=0),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True, key=f"spark_{series_id}")


# Row 1: GDP, Yield Spread, CPI
row1 = st.columns(3)
with row1[0]:
    st.metric("GDP Growth (Ann.)", f"{gdp_growth:+.1f}%", gdp_date[:7] if gdp_date else "")
    _render_sparkline("GDPC1")
with row1[1]:
    st.metric("Yield Spread (10Y-2Y)", f"{t10y2y_val:.2f}%", inverted_label)
    _render_sparkline("T10Y2Y")
with row1[2]:
    st.metric("CPI Inflation (YoY)", f"{cpi_yoy:.1f}%", cpi_date[:7] if cpi_date else "")
    _render_sparkline("CPIAUCSL")

# Row 2: Employment metrics (2 columns since only 2 metrics)
row2 = st.columns(2)
with row2[0]:
    st.metric("Unemployment (U3)", f"{unrate_val:.1f}%", f"{unrate_delta:+.1f}pp MoM")
    _render_sparkline("UNRATE")
with row2[1]:
    st.metric("Underemployment (U6)", f"{u6_val:.1f}%", f"{u6_delta:+.1f}pp MoM")
    _render_sparkline("U6RATE")

render_ai_insight_block("overview", "trend", use_full)

st.divider()


# ---------------------------------------------------------------------------
# Section B: Exploration — tabbed interactive charts
# ---------------------------------------------------------------------------

st.header("Exploration")

tab_recession, tab_labor, tab_broader, tab_predictions = st.tabs([
    "Recession Signals",
    "AI Labor Impact",
    "Broader Signals",
    "Recession Risk",
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
                    start: str = ""
                    for _, row in df_q3.iterrows():
                        if row["nber_recession"] == 1 and not in_rec:
                            start = str(row["date"])
                            in_rec = True
                        elif row["nber_recession"] == 0 and in_rec:
                            add_annotated_vrect(
                                fig_q3, start, str(row["date"]),
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


# --- Tab 4: Recession Risk ---
with tab_predictions:
    has_predictions = _table_exists("recession_predictions", use_full)
    has_scenarios = _table_exists("scenario_grid", use_full)

    if not has_predictions:
        st.info("Run recession_model.py to generate predictions.")
    else:
        # --- 1a: Recession Risk Score Timeline ---
        st.subheader("Recession Risk Score Timeline")
        df_pred = query_predictions(date_min, date_max, use_full)

        if not df_pred.empty:
            recession_data_pred = get_recession_data(date_min, date_max, use_full)

            fig_timeline = go.Figure()

            # Recession shading
            add_recession_shading(fig_timeline, recession_data_pred)

            # Threshold line at 0.5
            fig_timeline.add_hline(
                y=0.5, line_dash="dash", line_color="gray", opacity=0.5,
                annotation_text="Classification Threshold",
                annotation_position="bottom right",
            )

            # Color-code markers by threshold
            marker_colors = [
                "green" if p < 0.3 else ("goldenrod" if p < 0.6 else "red")
                for p in df_pred["probability"]
            ]

            # Split into evaluated vs forward predictions. Forward
            # predictions are trailing months where actual=0 but the
            # model's 12-month target window extends beyond available data.
            # Walk backward from end to find the last recession month; all
            # months after that are treated as forward predictions.
            forward_start = len(df_pred)
            for i in range(len(df_pred) - 1, -1, -1):
                if df_pred["actual"].iloc[i] == 1:
                    forward_start = i + 1
                    break

            df_evaluated = df_pred.iloc[:forward_start]
            df_forward = df_pred.iloc[max(0, forward_start - 1):]

            # Evaluated predictions — solid line with colored markers
            if not df_evaluated.empty:
                fig_timeline.add_trace(go.Scatter(
                    x=df_evaluated["date"],
                    y=df_evaluated["probability"],
                    mode="lines+markers",
                    name="Recession Risk Score",
                    line=dict(color="steelblue", width=2),
                    marker=dict(
                        color=marker_colors[:forward_start],
                        size=5,
                    ),
                    hovertemplate="%{x|%b %Y}: %{y:.3f}<extra>Risk Score</extra>",
                ))

            # Forward predictions — dashed line (genuine forecasts)
            if len(df_forward) > 1:
                fig_timeline.add_trace(go.Scatter(
                    x=df_forward["date"],
                    y=df_forward["probability"],
                    mode="lines+markers",
                    name="Forward Forecast",
                    line=dict(color="steelblue", width=2, dash="dash"),
                    marker=dict(
                        color=marker_colors[max(0, forward_start - 1):],
                        size=5,
                    ),
                    hovertemplate="%{x|%b %Y}: %{y:.3f}<extra>Forecast</extra>",
                ))

            # ChatGPT launch marker
            add_annotated_vline(
                fig_timeline, CHATGPT_LAUNCH, text="ChatGPT Launch"
            )

            fig_timeline.update_yaxes(
                title_text="Recession Risk Score", range=[0, 1]
            )
            style_figure(
                fig_timeline,
                "Recession Risk Score Over Time",
            )
            st.plotly_chart(fig_timeline, use_container_width=True, key="pred_timeline")

            # Model caption
            model_name = df_pred["model_name"].iloc[0]
            st.caption(
                f"Model: {model_name.replace('_', ' ').title()} | "
                "Scores indicate relative recession risk based on 11 "
                "macroeconomic features. Higher values mean more features "
                "align with historical pre-recession patterns. These are "
                "model outputs, not calibrated probabilities."
            )
            st.caption(
                "Recent predictions extend beyond the model's 12-month "
                "forward window. These represent genuine forecasts, not "
                "evaluated test results."
            )
            render_ai_insight_block("recession_risk", "trend", use_full)

        # --- 1b: Feature Contribution Snapshot ---
        st.subheader("Feature Contribution Snapshot")
        latest_pred = get_latest_prediction(use_full)

        if latest_pred is not None:
            features = latest_pred["features"]
            feature_names = list(features.keys())
            feature_values = [features[f] for f in feature_names]
            display_labels = [
                FEATURE_LABELS.get(f, f) for f in feature_names
            ]

            # Compute u6_u3_gap median from predictions for heuristic
            all_preds_for_median = run_query(
                "SELECT features_json FROM recession_predictions", use_full
            )
            u6_u3_values = []
            for fj in all_preds_for_median["features_json"]:
                parsed = json.loads(fj)
                if "u6_u3_gap" in parsed:
                    u6_u3_values.append(parsed["u6_u3_gap"])
            u6_u3_median = float(np.median(u6_u3_values)) if u6_u3_values else 3.5

            bar_colors = [
                get_feature_signal_color(f, v, u6_u3_median)
                for f, v in zip(feature_names, feature_values)
            ]

            fig_features = go.Figure(go.Bar(
                y=display_labels,
                x=feature_values,
                orientation="h",
                marker_color=bar_colors,
                text=[f"{v:.3f}" for v in feature_values],
                textposition="outside",
                hovertemplate="%{y}: %{x:.4f}<extra></extra>",
            ))
            fig_features.update_layout(
                template="plotly_white",
                height=450,
                margin=dict(l=200, r=80, t=40, b=30),
                xaxis_title="Feature Value",
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig_features, use_container_width=True, key="feature_snapshot")

            st.caption(
                f"Feature values as of {latest_pred['date']} | "
                f"Risk score: {latest_pred['probability']:.3f} | "
                "Red = recession-like signal, Green = healthy, Gray = neutral"
            )
            render_ai_insight_block("feature_snapshot", "trend", use_full)

        # --- 1c: What If Scenario Explorer ---
        st.subheader("What If Scenario Explorer")

        if not has_scenarios:
            st.info(
                "Run recession_model.py to generate the scenario grid."
            )
        else:
            grid_df = query_scenario_grid(use_full)

            if grid_df.empty:
                st.warning("Scenario grid is empty.")
            else:
                # Get latest feature values for slider defaults and fixed display
                if latest_pred is not None:
                    default_features = latest_pred["features"]
                else:
                    default_features = {
                        "yield_spread": 0.5,
                        "unrate": 4.0,
                        "gdp_growth_annualized": 2.0,
                        "cpi_yoy": 3.0,
                    }

                slider_col1, slider_col2 = st.columns(2)

                with slider_col1:
                    s_yield = st.slider(
                        "Yield Curve Spread",
                        min_value=-1.5,
                        max_value=3.0,
                        value=float(
                            np.clip(default_features.get("yield_spread", 0.5), -1.5, 3.0)
                        ),
                        step=0.25,
                        key="slider_yield",
                    )
                    s_unrate = st.slider(
                        "Unemployment Rate",
                        min_value=2.5,
                        max_value=8.0,
                        value=float(
                            np.clip(default_features.get("unrate", 4.0), 2.5, 8.0)
                        ),
                        step=0.25,
                        key="slider_unrate",
                    )

                with slider_col2:
                    s_gdp = st.slider(
                        "GDP Growth (Annualized)",
                        min_value=-6.0,
                        max_value=10.0,
                        value=float(
                            np.clip(
                                default_features.get("gdp_growth_annualized", 2.0),
                                -6.0, 10.0,
                            )
                        ),
                        step=0.5,
                        key="slider_gdp",
                    )
                    s_cpi = st.slider(
                        "CPI Inflation (YoY)",
                        min_value=0.0,
                        max_value=12.0,
                        value=float(
                            np.clip(default_features.get("cpi_yoy", 3.0), 0.0, 12.0)
                        ),
                        step=0.5,
                        key="slider_cpi",
                    )

                # Nearest-neighbor lookup
                nearest = find_nearest_scenario(
                    grid_df, s_yield, s_unrate, s_gdp, s_cpi
                )
                scenario_prob = float(nearest["probability"])

                # Display result
                result_col1, result_col2 = st.columns([1, 2])

                with result_col1:
                    if scenario_prob < 0.3:
                        risk_label = "Low Risk"
                    elif scenario_prob < 0.6:
                        risk_label = "Moderate Risk"
                    else:
                        risk_label = "High Risk"
                    st.metric(
                        "Recession Risk Score",
                        f"{scenario_prob:.3f}",
                        risk_label,
                    )

                with result_col2:
                    # Color bar gauge
                    gauge_color = (
                        "green" if scenario_prob < 0.3
                        else ("goldenrod" if scenario_prob < 0.6 else "red")
                    )
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=scenario_prob,
                        number=dict(suffix="", valueformat=".3f"),
                        gauge=dict(
                            axis=dict(range=[0, 1]),
                            bar=dict(color=gauge_color),
                            steps=[
                                dict(range=[0, 0.3], color="rgba(0,200,0,0.1)"),
                                dict(range=[0.3, 0.6], color="rgba(255,200,0,0.1)"),
                                dict(range=[0.6, 1], color="rgba(255,0,0,0.1)"),
                            ],
                            threshold=dict(
                                line=dict(color="black", width=2),
                                thickness=0.75,
                                value=0.5,
                            ),
                        ),
                    ))
                    fig_gauge.update_layout(
                        height=200,
                        margin=dict(l=20, r=20, t=30, b=10),
                    )
                    st.plotly_chart(fig_gauge, use_container_width=True, key="gauge")

                st.caption(
                    "Based on the closest pre-computed scenario. Other features "
                    "held at their most recent observed values."
                )

                # Show fixed features as read-only context
                if latest_pred is not None:
                    slider_keys = {
                        "yield_spread", "unrate",
                        "gdp_growth_annualized", "cpi_yoy",
                    }
                    fixed = {
                        FEATURE_LABELS.get(k, k): f"{v:.3f}"
                        for k, v in latest_pred["features"].items()
                        if k not in slider_keys
                    }
                    with st.expander("Fixed features (held at latest values)"):
                        fixed_df = pd.DataFrame(
                            list(fixed.items()),
                            columns=["Feature", "Value"],
                        )
                        st.dataframe(
                            fixed_df,
                            use_container_width=True,
                            hide_index=True,
                        )

                render_ai_insight_block("scenario_explorer", "comparison", use_full)

st.divider()


# ---------------------------------------------------------------------------
# Section C: NLP Analysis — topic modeling and cross-domain charts
# ---------------------------------------------------------------------------

st.header("NLP Analysis")

st.markdown(
    "Topic modeling (NMF) over 1,500+ Hacker News stories reveals *what* "
    "the tech-labor conversation is about, not just how negative it is. "
    "The charts below break down topic trends, sentiment by topic, and "
    "cross-domain connections to macro indicators."
)

render_ai_insight_block("hn_topic_analysis", "trend", use_full)

# Chart 1: Topic Distribution Over Time (stacked area)
nlp_topic_dist: pd.DataFrame = query_topic_distribution(date_min, date_max, use_full)
if not nlp_topic_dist.empty:
    fig_topic_dist = go.Figure()
    for label in nlp_topic_dist["label"].unique():
        subset = nlp_topic_dist[nlp_topic_dist["label"] == label]
        fig_topic_dist.add_trace(go.Scatter(
            x=subset["month"],
            y=subset["story_count"],
            mode="lines",
            name=label,
            stackgroup="one",
        ))
    fig_topic_dist.update_layout(
        title="Topic Distribution Over Time",
        xaxis_title="Month",
        yaxis_title="Stories",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=450,
    )
    add_annotated_vline(fig_topic_dist, CHATGPT_LAUNCH, text="ChatGPT Launch")
    st.plotly_chart(fig_topic_dist, use_container_width=True, key="nlp_topic_dist")
else:
    st.info("No topic distribution data available for the selected range.")

# Chart 2: Sentiment by Topic (horizontal box plot)
nlp_sent_by_topic: pd.DataFrame = query_sentiment_by_topic(use_full)
if not nlp_sent_by_topic.empty:
    fig_sent_topic = go.Figure()
    for label in nlp_sent_by_topic["label"].unique():
        subset = nlp_sent_by_topic[nlp_sent_by_topic["label"] == label]
        median_val: float = float(subset["sentiment_score"].median())
        if median_val < -0.1:
            box_color = "#EF553B"
        elif median_val > 0.1:
            box_color = "#00CC96"
        else:
            box_color = "#636EFA"
        fig_sent_topic.add_trace(go.Box(
            x=subset["sentiment_score"],
            name=label,
            orientation="h",
            marker_color=box_color,
        ))
    fig_sent_topic.update_layout(
        title="Sentiment Distribution by Topic",
        xaxis_title="Sentiment Score",
        height=400,
        showlegend=False,
    )
    fig_sent_topic.update_traces(hoverinfo="x", hovertemplate="%{x:.3f}<extra></extra>")
    st.plotly_chart(fig_sent_topic, use_container_width=True, key="nlp_sent_topic")
else:
    st.info("No topic sentiment data available.")

render_ai_insight_block("nlp_topic_sentiment", "trend", use_full)

# Charts 3 & 4 side by side
nlp_col1, nlp_col2 = st.columns(2)

# Chart 3: Layoff Volume vs U6-U3 Gap (dual-axis)
nlp_layoff_gap: pd.DataFrame = query_layoff_vs_u6u3(date_min, date_max, use_full)
with nlp_col1:
    if not nlp_layoff_gap.empty:
        fig_layoff = make_subplots(specs=[[{"secondary_y": True}]])
        fig_layoff.add_trace(
            go.Bar(
                x=nlp_layoff_gap["month"],
                y=nlp_layoff_gap["layoff_story_count"],
                name="Layoff Stories",
                marker_color="#EF553B",
                opacity=0.7,
            ),
            secondary_y=False,
        )
        fig_layoff.add_trace(
            go.Scatter(
                x=nlp_layoff_gap["month"],
                y=nlp_layoff_gap["u6_u3_gap"],
                name="U6-U3 Gap (pp)",
                line=dict(color="#00CC96", width=2),
            ),
            secondary_y=True,
        )
        fig_layoff.update_layout(
            title="Layoff Story Volume vs U6-U3 Gap",
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        fig_layoff.update_yaxes(title_text="Layoff Stories", secondary_y=False)
        fig_layoff.update_yaxes(title_text="U6-U3 Gap (pp)", secondary_y=True)
        add_annotated_vline(fig_layoff, CHATGPT_LAUNCH, text="ChatGPT")
        st.plotly_chart(fig_layoff, use_container_width=True, key="nlp_layoff_gap")
    else:
        st.info("No layoff vs U6-U3 data available.")

# Chart 4: Topic Sentiment vs USINFO Per-Capita
nlp_topic_usinfo: pd.DataFrame = query_topic_sentiment_vs_usinfo(
    date_min, date_max, use_full
)
with nlp_col2:
    if not nlp_topic_usinfo.empty and "info_pc_index" in nlp_topic_usinfo.columns:
        fig_topic_usinfo = make_subplots(specs=[[{"secondary_y": True}]])
        fig_topic_usinfo.add_trace(
            go.Scatter(
                x=nlp_topic_usinfo["month"],
                y=nlp_topic_usinfo["info_pc_index"],
                name="USINFO Per-Capita Index",
                line=dict(color="#636EFA", width=2),
            ),
            secondary_y=False,
        )
        # Add top-2 topic sentiment lines on secondary axis
        topic_cols: list[str] = [
            c for c in nlp_topic_usinfo.columns
            if c not in ("month", "info_pc_index")
        ]
        colors: list[str] = ["#EF553B", "#FFA15A"]
        for i, col in enumerate(topic_cols[:2]):
            fig_topic_usinfo.add_trace(
                go.Scatter(
                    x=nlp_topic_usinfo["month"],
                    y=nlp_topic_usinfo[col],
                    name=col,
                    line=dict(color=colors[i % len(colors)], width=2, dash="dot"),
                ),
                secondary_y=True,
            )
        fig_topic_usinfo.update_layout(
            title="Topic Sentiment vs USINFO Per-Capita",
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        fig_topic_usinfo.update_yaxes(title_text="USINFO Index (100=base)", secondary_y=False)
        fig_topic_usinfo.update_yaxes(title_text="Mean Sentiment", secondary_y=True)
        add_annotated_vline(fig_topic_usinfo, CHATGPT_LAUNCH, text="ChatGPT")
        st.plotly_chart(fig_topic_usinfo, use_container_width=True, key="nlp_topic_usinfo")
    else:
        st.info("No topic sentiment vs USINFO data available.")

# N-gram trending section
nlp_ngrams: pd.DataFrame = query_ngram_trends(date_min, date_max, use_full)
if not nlp_ngrams.empty:
    with st.expander("Trending Bigrams", expanded=False):
        # Pivot to quarterly aggregation for readability
        nlp_ngrams["quarter"] = nlp_ngrams["month"].str[:4] + "-Q" + (
            (pd.to_numeric(nlp_ngrams["month"].str[5:7]) - 1) // 3 + 1
        ).astype(str)
        quarterly: pd.DataFrame = (
            nlp_ngrams.groupby(["quarter", "ngram"])["count"]
            .sum()
            .reset_index()
        )
        # Top 8 ngrams overall for the heatmap
        top_ngrams: list[str] = (
            quarterly.groupby("ngram")["count"]
            .sum()
            .nlargest(8)
            .index.tolist()
        )
        heatmap_data: pd.DataFrame = quarterly[
            quarterly["ngram"].isin(top_ngrams)
        ].pivot_table(index="ngram", columns="quarter", values="count", fill_value=0)

        if not heatmap_data.empty:
            fig_ngram = go.Figure(data=go.Heatmap(
                z=heatmap_data.values,
                x=heatmap_data.columns.tolist(),
                y=heatmap_data.index.tolist(),
                colorscale="YlOrRd",
                hoverongaps=False,
            ))
            fig_ngram.update_layout(
                title="Top Bigrams by Quarter",
                xaxis_title="Quarter",
                yaxis_title="Bigram",
                height=350,
            )
            st.plotly_chart(fig_ngram, use_container_width=True, key="nlp_ngram_heatmap")

render_ai_insight_block("nlp_cross_domain", "correlation", use_full)

st.divider()


# ---------------------------------------------------------------------------
# Section D: Deep Dive — the core question
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
        render_ai_insight_block("deep_divergence", "comparison", use_full)

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
        render_ai_insight_block("deep_energy", "correlation", use_full)

# HN Sentiment Signal (Phase 14)
st.subheader("AI Labor Market Sentiment Signal")
st.caption(
    "Hacker News tech-practitioner sentiment (3-month rolling average) "
    "alongside per-capita info-sector employment. The sentiment score "
    "reflects a self-selected audience of tech practitioners, not a "
    "representative labor-market sentiment measure. HN data starts "
    "Jan 2022."
)

df_hn_overlay: pd.DataFrame = query_hn_sentiment_overlay(
    date_min, date_max, use_full
)
if not df_hn_overlay.empty:
    fig_hn = make_subplots(specs=[[{"secondary_y": True}]])

    fig_hn.add_trace(
        go.Scatter(
            x=df_hn_overlay["month"],
            y=df_hn_overlay["info_pc_index"],
            name="USINFO Per-Capita Index",
            line=dict(color="#636EFA"),
        ),
        secondary_y=False,
    )
    fig_hn.add_trace(
        go.Scatter(
            x=df_hn_overlay["month"],
            y=df_hn_overlay["mean_sentiment_3m_avg"],
            name="HN Sentiment (3-mo avg)",
            line=dict(color="#EF553B", dash="dot"),
        ),
        secondary_y=True,
    )

    add_annotated_vline(fig_hn, CHATGPT_LAUNCH)
    fig_hn.add_hline(
        y=0, line_dash="dot", line_color="gray", opacity=0.3,
        secondary_y=True,
    )
    fig_hn.update_yaxes(title_text="Per-Capita Index", secondary_y=False)
    fig_hn.update_yaxes(title_text="Sentiment Score", secondary_y=True)
    style_figure(fig_hn, "HN Sentiment vs Info Employment")
    st.plotly_chart(fig_hn, use_container_width=True, key="deep_hn")

render_ai_insight_block("hn_labor_sentiment", "correlation", use_full)

render_ai_insight_block("deep_synthesis_charts", "comparison", use_full)

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

# ---------------------------------------------------------------------------
# Ask the Data (Phase 19)
# ---------------------------------------------------------------------------

if _ASK_THE_DATA_AVAILABLE:
    st.header("Ask the Data")
    _active_db = str(FULL_DB if use_full else SEED_DB)
    _chroma_dir = str(PROJECT_ROOT / "data" / ".chroma")
    render_ask_the_data(db_path=_active_db, chroma_path=_chroma_dir)

st.divider()
st.caption("Data source: FRED (Federal Reserve Economic Data) | Built with Streamlit + Plotly")
