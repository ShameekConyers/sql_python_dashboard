"""COVID period adjustment using ARIMA counterfactual modeling.

Fits ARIMA models on pre-COVID data for each FRED series, forecasts through
the COVID window (Mar 2020 - Jun 2021), and tapers back to actual data with
a linear blend. Results are stored in the ``value_covid_adjusted`` column.

The adjustment treats COVID as a 500-year outlier that contaminates rolling
window calculations and trend analysis. The post-COVID structural changes
are fully preserved. Raw values remain in the ``value`` column so both views
are always available.

Special cases:
  - USREC (binary recession flag): set to 0 during COVID window since the
    counterfactual assumes no pandemic recession.
  - T10Y2Y (daily): aggregated to monthly before ARIMA, then the monthly
    counterfactual is applied to all daily observations in that month.

Usage:
    python src/covid_adjustment.py           # adjusts seed.db
    python src/covid_adjustment.py --full    # adjusts full.db

Run AFTER db_setup.py and BEFORE export_csv.py.
"""

import argparse
import logging
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from pmdarima import auto_arima

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger: logging.Logger = logging.getLogger(__name__)

# COVID window boundaries
COVID_START: str = "2020-03-01"
COVID_END: str = "2021-06-01"
TAPER_MONTHS: int = 3

# Series that need special handling
BINARY_SERIES: set[str] = {"USREC"}
DAILY_SERIES: set[str] = {"T10Y2Y"}


def _db_path(mode: str) -> Path:
    """Return the database file path for the given mode.

    Args:
        mode: Either 'seed' or 'full'.

    Returns:
        Path to the SQLite database file.
    """
    filename: str = "seed.db" if mode == "seed" else "full.db"
    return DATA_DIR / filename


def _get_series_ids(conn: sqlite3.Connection) -> list[str]:
    """Get all unique series IDs from the database.

    Args:
        conn: Open SQLite connection.

    Returns:
        List of series ID strings.
    """
    rows: list[tuple] = conn.execute(
        "SELECT DISTINCT series_id FROM observations ORDER BY series_id"
    ).fetchall()
    return [r[0] for r in rows]


def _get_frequency(conn: sqlite3.Connection, series_id: str) -> str:
    """Look up a series' frequency from metadata.

    Args:
        conn: Open SQLite connection.
        series_id: FRED series identifier.

    Returns:
        Frequency string ('daily', 'monthly', or 'quarterly').
    """
    row: tuple = conn.execute(
        "SELECT frequency FROM series_metadata WHERE series_id = ?",
        (series_id,),
    ).fetchone()
    return row[0]


def _get_observations_df(conn: sqlite3.Connection, series_id: str) -> pd.DataFrame:
    """Load all observations for a series into a DataFrame.

    Args:
        conn: Open SQLite connection.
        series_id: FRED series identifier.

    Returns:
        DataFrame with 'date' (datetime) and 'value' columns, sorted by date.
    """
    rows: list[tuple] = conn.execute(
        "SELECT date, value FROM observations WHERE series_id = ? ORDER BY date",
        (series_id,),
    ).fetchall()
    df: pd.DataFrame = pd.DataFrame(rows, columns=["date", "value"])
    df["date"] = pd.to_datetime(df["date"])
    return df


def _seasonality_period(frequency: str) -> int:
    """Map frequency string to the seasonal period for ARIMA.

    Args:
        frequency: One of 'daily', 'monthly', or 'quarterly'.

    Returns:
        Seasonal period (m) for SARIMA. Returns 1 for daily (no seasonality
        after monthly aggregation is handled upstream).
    """
    if frequency == "quarterly":
        return 4
    return 12  # monthly and aggregated-daily both use m=12


def _fit_and_forecast(
    pre_covid_values: np.ndarray, n_steps: int, seasonal_period: int
) -> tuple[np.ndarray, str]:
    """Fit auto_arima on pre-COVID data and forecast n_steps ahead.

    Args:
        pre_covid_values: Array of observed values before COVID.
        n_steps: Number of future periods to forecast.
        seasonal_period: Seasonal period for SARIMA (4 or 12).

    Returns:
        Tuple of (forecast array, model order string for logging).
    """
    use_seasonal: bool = len(pre_covid_values) >= 2 * seasonal_period
    model = auto_arima(
        pre_covid_values,
        seasonal=use_seasonal,
        m=seasonal_period if use_seasonal else 1,
        stepwise=True,
        suppress_warnings=True,
        max_p=3,
        max_q=3,
        max_P=2,
        max_Q=2,
        error_action="ignore",
    )
    forecast: np.ndarray = model.predict(n_periods=n_steps)
    order_str: str = str(model.order)
    if use_seasonal:
        order_str += str(model.seasonal_order)
    return forecast, order_str


def _apply_taper(
    forecast_values: np.ndarray,
    actual_values: np.ndarray,
    covid_dates: pd.DatetimeIndex,
) -> np.ndarray:
    """Blend ARIMA forecast back to actual values with a linear taper.

    The taper covers the last TAPER_MONTHS of the COVID window. Before the
    taper zone, the output is pure ARIMA forecast. Within the taper, the
    blend shifts linearly from forecast toward actual.

    Args:
        forecast_values: ARIMA counterfactual values for the COVID window.
        actual_values: Raw observed values for the COVID window.
        covid_dates: DatetimeIndex of observation dates in the COVID window.

    Returns:
        Array of adjusted values (same length as inputs).
    """
    taper_start: pd.Timestamp = pd.Timestamp(COVID_END) - pd.DateOffset(months=TAPER_MONTHS)
    taper_end: pd.Timestamp = pd.Timestamp(COVID_END) + pd.DateOffset(months=1)

    adjusted: np.ndarray = forecast_values.copy()
    for i, dt in enumerate(covid_dates):
        if dt >= taper_start:
            total_days: int = (taper_end - taper_start).days
            days_in: int = (dt - taper_start).days
            # weight goes from ~0 at taper_start to ~1 at taper_end
            weight_actual: float = min((days_in + 1) / (total_days + 1), 1.0)
            adjusted[i] = (1 - weight_actual) * forecast_values[i] + weight_actual * actual_values[i]

    return adjusted


def _adjust_binary_series(conn: sqlite3.Connection, series_id: str) -> None:
    """Set COVID window to 0 for binary indicator series.

    The counterfactual for USREC is 'no recession occurred.'

    Args:
        conn: Open SQLite connection.
        series_id: FRED series identifier.
    """
    conn.execute(
        """
        UPDATE observations SET value_covid_adjusted = 0
        WHERE series_id = ? AND date >= ? AND date <= ?
        """,
        (series_id, COVID_START, COVID_END),
    )
    logger.info("  %-16s  binary → set COVID window to 0", series_id)


def _adjust_daily_series(conn: sqlite3.Connection, series_id: str) -> None:
    """Adjust a daily series by fitting ARIMA on monthly aggregates.

    Daily observations within each COVID month receive the monthly
    counterfactual value. Non-COVID observations are unchanged.

    Args:
        conn: Open SQLite connection.
        series_id: FRED series identifier.
    """
    df: pd.DataFrame = _get_observations_df(conn, series_id)
    if df.empty:
        return

    # Aggregate to monthly means
    df["month"] = df["date"].dt.to_period("M")
    monthly: pd.DataFrame = df.groupby("month")["value"].mean().reset_index()
    monthly["date"] = monthly["month"].dt.to_timestamp()

    covid_mask: pd.Series = (monthly["date"] >= COVID_START) & (monthly["date"] <= COVID_END)
    pre_covid: pd.DataFrame = monthly[monthly["date"] < COVID_START]
    covid_window: pd.DataFrame = monthly[covid_mask]

    if covid_window.empty or len(pre_covid) < 12:
        logger.warning("  %-16s  skipped (insufficient data)", series_id)
        return

    forecast: np.ndarray
    order_str: str
    forecast, order_str = _fit_and_forecast(
        pre_covid["value"].values,
        n_steps=len(covid_window),
        seasonal_period=12,
    )

    adjusted: np.ndarray = _apply_taper(
        forecast,
        covid_window["value"].values,
        pd.DatetimeIndex(covid_window["date"]),
    )

    # Build month → adjusted_value lookup
    month_lookup: dict[str, float] = {}
    for dt, val in zip(covid_window["date"], adjusted):
        month_key: str = dt.strftime("%Y-%m")
        month_lookup[month_key] = float(val)

    # Update daily observations for each COVID month
    for month_key, adj_val in month_lookup.items():
        conn.execute(
            """
            UPDATE observations SET value_covid_adjusted = ?
            WHERE series_id = ? AND SUBSTR(date, 1, 7) = ?
            """,
            (adj_val, series_id, month_key),
        )

    logger.info(
        "  %-16s  daily→monthly ARIMA %s, %d months adjusted",
        series_id,
        order_str,
        len(month_lookup),
    )


def _adjust_standard_series(
    conn: sqlite3.Connection, series_id: str, frequency: str
) -> None:
    """Adjust a monthly or quarterly series with ARIMA counterfactual.

    Args:
        conn: Open SQLite connection.
        series_id: FRED series identifier.
        frequency: 'monthly' or 'quarterly'.
    """
    df: pd.DataFrame = _get_observations_df(conn, series_id)
    if df.empty:
        return

    covid_mask: pd.Series = (df["date"] >= COVID_START) & (df["date"] <= COVID_END)
    pre_covid: pd.DataFrame = df[df["date"] < COVID_START]
    covid_window: pd.DataFrame = df[covid_mask]

    if covid_window.empty:
        logger.info("  %-16s  no COVID-window observations, skipped", series_id)
        return

    if len(pre_covid) < 8:
        logger.warning(
            "  %-16s  only %d pre-COVID obs, skipped", series_id, len(pre_covid)
        )
        return

    seasonal_m: int = _seasonality_period(frequency)

    forecast: np.ndarray
    order_str: str
    forecast, order_str = _fit_and_forecast(
        pre_covid["value"].values,
        n_steps=len(covid_window),
        seasonal_period=seasonal_m,
    )

    adjusted: np.ndarray = _apply_taper(
        forecast,
        covid_window["value"].values,
        pd.DatetimeIndex(covid_window["date"]),
    )

    # Write adjusted values back to the database
    for dt, adj_val in zip(covid_window["date"], adjusted):
        date_str: str = dt.strftime("%Y-%m-%d")
        conn.execute(
            """
            UPDATE observations SET value_covid_adjusted = ?
            WHERE series_id = ? AND date = ?
            """,
            (float(adj_val), series_id, date_str),
        )

    logger.info(
        "  %-16s  %s ARIMA %s, %d obs adjusted, taper last %d months",
        series_id,
        frequency,
        order_str,
        len(covid_window),
        TAPER_MONTHS,
    )


def _print_summary(conn: sqlite3.Connection) -> None:
    """Print before/after comparison for COVID window observations.

    Args:
        conn: Open SQLite connection.
    """
    print("\n" + "=" * 70)
    print("COVID ADJUSTMENT SUMMARY")
    print("=" * 70)

    rows: list[tuple] = conn.execute(
        """
        SELECT
            series_id,
            COUNT(*) AS n_adjusted,
            ROUND(AVG(value), 2) AS avg_raw,
            ROUND(AVG(value_covid_adjusted), 2) AS avg_adjusted,
            ROUND(AVG(ABS(value - value_covid_adjusted)), 2) AS avg_abs_diff
        FROM observations
        WHERE date >= ? AND date <= ?
        GROUP BY series_id
        ORDER BY series_id
        """,
        (COVID_START, COVID_END),
    ).fetchall()

    print(f"  {'Series':<18} {'N':>4} {'Avg Raw':>10} {'Avg Adj':>10} {'Avg |Diff|':>11}")
    print(f"  {'-'*18} {'-'*4} {'-'*10} {'-'*10} {'-'*11}")
    for series_id, n, avg_raw, avg_adj, avg_diff in rows:
        print(
            f"  {series_id:<18} {n:>4} {avg_raw:>10.2f} {avg_adj:>10.2f} {avg_diff:>11.2f}"
        )

    # Verify non-COVID observations are unchanged (compare at month level
    # to handle daily series where the full month is adjusted)
    unchanged: tuple = conn.execute(
        """
        SELECT COUNT(*) FROM observations
        WHERE (SUBSTR(date, 1, 7) < SUBSTR(?, 1, 7)
            OR SUBSTR(date, 1, 7) > SUBSTR(?, 1, 7))
          AND value != value_covid_adjusted
        """,
        (COVID_START, COVID_END),
    ).fetchone()
    print(f"\n  Non-COVID rows where raw != adjusted: {unchanged[0]} (should be 0)")
    print("=" * 70 + "\n")


def adjust_all(mode: str = "seed") -> None:
    """Run COVID adjustment on all series in the database.

    Args:
        mode: 'seed' or 'full', determines which database to adjust.
    """
    db: Path = _db_path(mode)
    if not db.exists():
        logger.error("Database not found: %s", db)
        return

    conn: sqlite3.Connection = sqlite3.connect(db)
    series_ids: list[str] = _get_series_ids(conn)
    logger.info(
        "COVID adjustment: %s, window %s to %s, taper %d months",
        db.name,
        COVID_START,
        COVID_END,
        TAPER_MONTHS,
    )

    try:
        for series_id in series_ids:
            if series_id in BINARY_SERIES:
                _adjust_binary_series(conn, series_id)
            elif series_id in DAILY_SERIES:
                _adjust_daily_series(conn, series_id)
            else:
                frequency: str = _get_frequency(conn, series_id)
                _adjust_standard_series(conn, series_id, frequency)

        conn.commit()
        _print_summary(conn)
    finally:
        conn.close()

    logger.info("COVID adjustment complete: %s", db)


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Apply ARIMA-based COVID adjustment to the observations table.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Adjust full.db instead of seed.db.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for the COVID adjustment script."""
    args: argparse.Namespace = _parse_args()
    mode: str = "full" if args.full else "seed"
    adjust_all(mode)


if __name__ == "__main__":
    main()
