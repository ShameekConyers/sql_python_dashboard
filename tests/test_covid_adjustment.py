"""Tests for src/covid_adjustment.py.

Uses in-memory SQLite databases with synthetic data. ARIMA fitting is mocked
in most tests to keep the suite fast and deterministic.
"""

import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

import covid_adjustment

# ---------------------------------------------------------------------------
# Schema helper — reuse the real schema SQL
# ---------------------------------------------------------------------------

SCHEMA_SQL: str = (covid_adjustment.PROJECT_ROOT / "sql" / "01_schema.sql").read_text()


def _make_db(
    series_id: str = "UNRATE",
    frequency: str = "monthly",
    dates: list[str] | None = None,
    values: list[float] | None = None,
) -> sqlite3.Connection:
    """Build an in-memory DB with one series and its observations.

    Args:
        series_id: FRED series identifier.
        frequency: Series frequency string.
        dates: ISO date strings for observations.
        values: Observed values (same length as dates).

    Returns:
        Open SQLite connection with schema, metadata, and observations.
    """
    if dates is None:
        # 36 months: 2018-01 through 2020-12
        dates = [f"{2018 + (i // 12)}-{(i % 12) + 1:02d}-01" for i in range(36)]
    if values is None:
        values = [3.5 + 0.1 * i for i in range(len(dates))]

    conn: sqlite3.Connection = sqlite3.connect(":memory:")
    conn.executescript(SCHEMA_SQL)
    conn.execute(
        "INSERT INTO series_metadata (series_id, name, category, frequency) VALUES (?, ?, ?, ?)",
        (series_id, series_id, "test", frequency),
    )
    for d, v in zip(dates, values):
        conn.execute(
            "INSERT INTO observations (series_id, date, value, value_covid_adjusted) VALUES (?, ?, ?, ?)",
            (series_id, d, v, v),
        )
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# _db_path
# ---------------------------------------------------------------------------


class TestDbPath:
    """Tests for _db_path."""

    def test_seed_returns_seed_db(self) -> None:
        """Seed mode points to seed.db."""
        assert covid_adjustment._db_path("seed").name == "seed.db"

    def test_full_returns_full_db(self) -> None:
        """Full mode points to full.db."""
        assert covid_adjustment._db_path("full").name == "full.db"


# ---------------------------------------------------------------------------
# _get_series_ids
# ---------------------------------------------------------------------------


class TestGetSeriesIds:
    """Tests for _get_series_ids."""

    def test_returns_ids_from_observations(self) -> None:
        """Returns distinct series IDs present in observations."""
        conn: sqlite3.Connection = _make_db("UNRATE")
        # Add a second series
        conn.execute(
            "INSERT INTO series_metadata (series_id, name, category, frequency) VALUES (?, ?, ?, ?)",
            ("GDPC1", "GDP", "test", "quarterly"),
        )
        conn.execute(
            "INSERT INTO observations (series_id, date, value, value_covid_adjusted) VALUES (?, ?, ?, ?)",
            ("GDPC1", "2020-01-01", 100.0, 100.0),
        )
        conn.commit()

        ids: list[str] = covid_adjustment._get_series_ids(conn)
        assert "GDPC1" in ids
        assert "UNRATE" in ids

    def test_returns_sorted(self) -> None:
        """IDs are returned in alphabetical order."""
        conn: sqlite3.Connection = _make_db("ZEBRA")
        conn.execute(
            "INSERT INTO series_metadata (series_id, name, category, frequency) VALUES (?, ?, ?, ?)",
            ("ALPHA", "A", "test", "monthly"),
        )
        conn.execute(
            "INSERT INTO observations (series_id, date, value, value_covid_adjusted) VALUES (?, ?, ?, ?)",
            ("ALPHA", "2020-01-01", 1.0, 1.0),
        )
        conn.commit()

        ids: list[str] = covid_adjustment._get_series_ids(conn)
        assert ids == sorted(ids)


# ---------------------------------------------------------------------------
# _get_frequency
# ---------------------------------------------------------------------------


class TestGetFrequency:
    """Tests for _get_frequency."""

    def test_returns_frequency(self) -> None:
        """Returns the frequency stored in series_metadata."""
        conn: sqlite3.Connection = _make_db("UNRATE", frequency="monthly")
        assert covid_adjustment._get_frequency(conn, "UNRATE") == "monthly"


# ---------------------------------------------------------------------------
# _get_observations_df
# ---------------------------------------------------------------------------


class TestGetObservationsDf:
    """Tests for _get_observations_df."""

    def test_returns_dataframe(self) -> None:
        """Returns a DataFrame with date and value columns."""
        conn: sqlite3.Connection = _make_db("UNRATE")
        df: pd.DataFrame = covid_adjustment._get_observations_df(conn, "UNRATE")

        assert "date" in df.columns
        assert "value" in df.columns
        assert len(df) > 0

    def test_dates_are_datetime(self) -> None:
        """The date column is converted to datetime."""
        conn: sqlite3.Connection = _make_db("UNRATE")
        df: pd.DataFrame = covid_adjustment._get_observations_df(conn, "UNRATE")

        assert pd.api.types.is_datetime64_any_dtype(df["date"])

    def test_sorted_by_date(self) -> None:
        """Rows are sorted by date ascending."""
        conn: sqlite3.Connection = _make_db("UNRATE")
        df: pd.DataFrame = covid_adjustment._get_observations_df(conn, "UNRATE")

        assert df["date"].is_monotonic_increasing


# ---------------------------------------------------------------------------
# _seasonality_period
# ---------------------------------------------------------------------------


class TestSeasonalityPeriod:
    """Tests for _seasonality_period."""

    def test_quarterly_returns_4(self) -> None:
        """Quarterly frequency maps to seasonal period 4."""
        assert covid_adjustment._seasonality_period("quarterly") == 4

    def test_monthly_returns_12(self) -> None:
        """Monthly frequency maps to seasonal period 12."""
        assert covid_adjustment._seasonality_period("monthly") == 12

    def test_daily_returns_12(self) -> None:
        """Daily frequency (post-aggregation) maps to seasonal period 12."""
        assert covid_adjustment._seasonality_period("daily") == 12


# ---------------------------------------------------------------------------
# _apply_taper
# ---------------------------------------------------------------------------


class TestApplyTaper:
    """Tests for _apply_taper."""

    def test_pre_taper_values_are_forecast(self) -> None:
        """Before the taper zone, output equals the ARIMA forecast."""
        dates: pd.DatetimeIndex = pd.to_datetime(
            ["2020-03-01", "2020-04-01", "2020-05-01", "2020-06-01",
             "2020-07-01", "2020-08-01", "2020-09-01"]
        )
        forecast: np.ndarray = np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0])
        actual: np.ndarray = np.array([20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0])

        result: np.ndarray = covid_adjustment._apply_taper(forecast, actual, dates)

        # All dates are before the taper zone (COVID_END - 3mo = 2021-10-01),
        # so they should equal the forecast
        assert result[0] == pytest.approx(10.0)
        assert result[1] == pytest.approx(10.0)

    def test_taper_zone_blends(self) -> None:
        """Within the taper zone, values are between forecast and actual."""
        # Taper zone: COVID_END - 3mo to COVID_END + 1mo = 2021-10-01 to 2022-02-01
        dates: pd.DatetimeIndex = pd.to_datetime(
            ["2021-10-01", "2021-11-01", "2021-12-01", "2022-01-01"]
        )
        forecast: np.ndarray = np.array([10.0, 10.0, 10.0, 10.0])
        actual: np.ndarray = np.array([20.0, 20.0, 20.0, 20.0])

        result: np.ndarray = covid_adjustment._apply_taper(forecast, actual, dates)

        for val in result:
            assert 10.0 <= val <= 20.0

    def test_output_length_matches_input(self) -> None:
        """Output has the same length as input arrays."""
        dates: pd.DatetimeIndex = pd.to_datetime(["2020-06-01", "2020-07-01"])
        forecast: np.ndarray = np.array([5.0, 6.0])
        actual: np.ndarray = np.array([7.0, 8.0])

        result: np.ndarray = covid_adjustment._apply_taper(forecast, actual, dates)
        assert len(result) == 2

    def test_no_taper_when_all_before_taper_zone(self) -> None:
        """When all dates are before the taper zone, output equals forecast."""
        dates: pd.DatetimeIndex = pd.to_datetime(["2020-03-01", "2020-04-01"])
        forecast: np.ndarray = np.array([5.0, 6.0])
        actual: np.ndarray = np.array([10.0, 12.0])

        result: np.ndarray = covid_adjustment._apply_taper(forecast, actual, dates)
        np.testing.assert_array_equal(result, forecast)


# ---------------------------------------------------------------------------
# _adjust_binary_series
# ---------------------------------------------------------------------------


class TestAdjustBinarySeries:
    """Tests for _adjust_binary_series (USREC handling)."""

    def test_sets_covid_window_to_zero(self) -> None:
        """COVID-window observations get value_covid_adjusted = 0."""
        dates: list[str] = [
            "2019-12-01", "2020-03-01", "2020-06-01",
            "2021-06-01", "2022-01-01", "2022-02-01",
        ]
        values: list[float] = [0.0, 1.0, 1.0, 1.0, 1.0, 0.0]
        conn: sqlite3.Connection = _make_db("USREC", "monthly", dates, values)

        covid_adjustment._adjust_binary_series(conn, "USREC")
        conn.commit()

        rows: list[tuple] = conn.execute(
            "SELECT date, value, value_covid_adjusted FROM observations WHERE series_id = 'USREC' ORDER BY date"
        ).fetchall()

        for d, raw, adj in rows:
            if "2020-03-01" <= d <= "2022-01-01":
                assert adj == 0.0, f"Expected 0 for {d}, got {adj}"
            else:
                assert adj == raw, f"Non-COVID obs {d} should be unchanged"

    def test_leaves_non_covid_unchanged(self) -> None:
        """Observations outside the COVID window keep their original values."""
        dates: list[str] = ["2019-01-01", "2022-02-01"]
        values: list[float] = [0.0, 0.0]
        conn: sqlite3.Connection = _make_db("USREC", "monthly", dates, values)

        covid_adjustment._adjust_binary_series(conn, "USREC")
        conn.commit()

        rows: list[tuple] = conn.execute(
            "SELECT value, value_covid_adjusted FROM observations"
        ).fetchall()
        for raw, adj in rows:
            assert raw == adj


# ---------------------------------------------------------------------------
# _adjust_standard_series
# ---------------------------------------------------------------------------


class TestAdjustStandardSeries:
    """Tests for _adjust_standard_series with mocked ARIMA."""

    def test_updates_covid_window_observations(self) -> None:
        """COVID-window observations get different value_covid_adjusted values."""
        # Build dates: 24 months pre-COVID + 23 COVID months (Mar 2020 - Jan 2022)
        pre_dates: list[str] = [f"{2018 + (i // 12)}-{(i % 12) + 1:02d}-01" for i in range(24)]
        covid_dates: list[str] = [
            "2020-03-01", "2020-04-01", "2020-05-01", "2020-06-01",
            "2020-07-01", "2020-08-01", "2020-09-01", "2020-10-01",
            "2020-11-01", "2020-12-01", "2021-01-01", "2021-02-01",
            "2021-03-01", "2021-04-01", "2021-05-01", "2021-06-01",
            "2021-07-01", "2021-08-01", "2021-09-01", "2021-10-01",
            "2021-11-01", "2021-12-01", "2022-01-01",
        ]
        all_dates: list[str] = pre_dates + covid_dates
        all_values: list[float] = [3.5 + 0.05 * i for i in range(len(all_dates))]

        conn: sqlite3.Connection = _make_db("UNRATE", "monthly", all_dates, all_values)

        # Mock ARIMA to return a flat forecast
        n_covid: int = len(covid_dates)
        fake_forecast: np.ndarray = np.full(n_covid, 99.0)
        with patch.object(
            covid_adjustment,
            "_fit_and_forecast",
            return_value=(fake_forecast, "(1,1,1)"),
        ):
            covid_adjustment._adjust_standard_series(conn, "UNRATE", "monthly")

        conn.commit()

        # Check that COVID window adjusted values differ from raw
        rows: list[tuple] = conn.execute(
            """SELECT date, value, value_covid_adjusted FROM observations
               WHERE series_id = 'UNRATE' AND date >= '2020-03-01' AND date <= '2022-01-01'
               ORDER BY date""",
        ).fetchall()

        assert len(rows) == n_covid
        for d, raw, adj in rows:
            # Taper blends forecast→actual, so at least early values should differ
            # For the first observation, it should be the forecast (99.0) since it's before taper
            pass  # We just verify no exception was raised and rows were updated

        # At least the first COVID obs should be the pure forecast value
        first_adj: float = rows[0][2]
        assert first_adj == pytest.approx(99.0, abs=0.1)

    def test_skips_when_no_covid_observations(self) -> None:
        """Does nothing when the series has no observations in the COVID window."""
        dates: list[str] = ["2019-01-01", "2019-02-01", "2019-03-01"]
        values: list[float] = [3.5, 3.6, 3.7]
        conn: sqlite3.Connection = _make_db("UNRATE", "monthly", dates, values)

        # Should not raise
        covid_adjustment._adjust_standard_series(conn, "UNRATE", "monthly")

    def test_skips_when_insufficient_pre_covid_data(self) -> None:
        """Skips adjustment when there are fewer than 8 pre-COVID observations."""
        dates: list[str] = [
            "2020-01-01", "2020-02-01",  # only 2 pre-COVID
            "2020-03-01", "2020-04-01",  # COVID window
        ]
        values: list[float] = [3.5, 3.6, 11.0, 12.0]
        conn: sqlite3.Connection = _make_db("UNRATE", "monthly", dates, values)

        covid_adjustment._adjust_standard_series(conn, "UNRATE", "monthly")
        conn.commit()

        # Values should be unchanged (adjustment was skipped)
        rows: list[tuple] = conn.execute(
            "SELECT value, value_covid_adjusted FROM observations WHERE date >= '2020-03-01'"
        ).fetchall()
        for raw, adj in rows:
            assert raw == adj


# ---------------------------------------------------------------------------
# _adjust_daily_series
# ---------------------------------------------------------------------------


class TestAdjustDailySeries:
    """Tests for _adjust_daily_series with mocked ARIMA."""

    def test_updates_daily_obs_in_covid_window(self) -> None:
        """Daily observations within COVID months get monthly counterfactual values."""
        # Build 24 months of "daily" data pre-COVID (one obs per month for simplicity)
        pre_dates: list[str] = [f"{2018 + (i // 12)}-{(i % 12) + 1:02d}-01" for i in range(24)]
        # Add some daily obs in a single COVID month
        covid_dates: list[str] = [
            "2020-03-02", "2020-03-09", "2020-03-16",
            "2020-04-01", "2020-04-08",
        ]
        all_dates: list[str] = pre_dates + covid_dates
        all_values: list[float] = [1.5 + 0.01 * i for i in range(len(all_dates))]

        conn: sqlite3.Connection = _make_db("T10Y2Y", "daily", all_dates, all_values)

        # Mock ARIMA to return known forecast values
        fake_forecast: np.ndarray = np.array([50.0, 60.0])  # 2 COVID months
        with patch.object(
            covid_adjustment,
            "_fit_and_forecast",
            return_value=(fake_forecast, "(1,0,0)"),
        ):
            covid_adjustment._adjust_daily_series(conn, "T10Y2Y")

        conn.commit()

        # All March 2020 daily obs should get the same monthly value
        march_rows: list[tuple] = conn.execute(
            "SELECT value_covid_adjusted FROM observations WHERE series_id = 'T10Y2Y' AND date LIKE '2020-03%'"
        ).fetchall()
        march_vals: set[float] = {r[0] for r in march_rows}
        # They should all share the same adjusted value (from taper/forecast blend)
        assert len(march_vals) == 1

    def test_skips_empty_series(self) -> None:
        """Returns early when the series has no observations."""
        conn: sqlite3.Connection = sqlite3.connect(":memory:")
        conn.executescript(SCHEMA_SQL)
        conn.execute(
            "INSERT INTO series_metadata (series_id, name, category, frequency) VALUES (?, ?, ?, ?)",
            ("T10Y2Y", "T10Y2Y", "test", "daily"),
        )
        conn.commit()

        # Should not raise
        covid_adjustment._adjust_daily_series(conn, "T10Y2Y")


# ---------------------------------------------------------------------------
# adjust_all
# ---------------------------------------------------------------------------


class TestAdjustAll:
    """Tests for the adjust_all orchestrator."""

    def test_skips_when_db_missing(self, tmp_path: Path) -> None:
        """Returns early (no error) when the database file does not exist."""
        with patch.object(covid_adjustment, "DATA_DIR", tmp_path):
            # Should not raise
            covid_adjustment.adjust_all("seed")

    def test_dispatches_to_correct_handler(self, tmp_path: Path) -> None:
        """Routes USREC to binary, T10Y2Y to daily, others to standard."""
        # Create a real DB on disk with all three types
        db_file: Path = tmp_path / "seed.db"
        conn: sqlite3.Connection = sqlite3.connect(db_file)
        conn.executescript(SCHEMA_SQL)

        for sid, freq in [("USREC", "monthly"), ("T10Y2Y", "daily"), ("UNRATE", "monthly")]:
            conn.execute(
                "INSERT INTO series_metadata (series_id, name, category, frequency) VALUES (?, ?, ?, ?)",
                (sid, sid, "test", freq),
            )
            conn.execute(
                "INSERT INTO observations (series_id, date, value, value_covid_adjusted) VALUES (?, ?, ?, ?)",
                (sid, "2020-06-01", 5.0, 5.0),
            )
        conn.commit()
        conn.close()

        with (
            patch.object(covid_adjustment, "DATA_DIR", tmp_path),
            patch.object(covid_adjustment, "_adjust_binary_series") as mock_binary,
            patch.object(covid_adjustment, "_adjust_daily_series") as mock_daily,
            patch.object(covid_adjustment, "_adjust_standard_series") as mock_standard,
            patch.object(covid_adjustment, "_print_summary"),
        ):
            covid_adjustment.adjust_all("seed")

        # USREC → binary handler
        mock_binary.assert_called_once()
        assert mock_binary.call_args[0][1] == "USREC"

        # T10Y2Y → daily handler
        mock_daily.assert_called_once()
        assert mock_daily.call_args[0][1] == "T10Y2Y"

        # UNRATE → standard handler
        mock_standard.assert_called_once()
        assert mock_standard.call_args[0][1] == "UNRATE"


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


class TestMain:
    """Tests for the main() entry point."""

    @patch("covid_adjustment.adjust_all")
    @patch("covid_adjustment._parse_args")
    def test_default_is_seed(self, mock_args: MagicMock, mock_adjust: MagicMock) -> None:
        """Without --full, main calls adjust_all with 'seed'."""
        import argparse

        mock_args.return_value = argparse.Namespace(full=False)
        covid_adjustment.main()
        mock_adjust.assert_called_once_with("seed")

    @patch("covid_adjustment.adjust_all")
    @patch("covid_adjustment._parse_args")
    def test_full_flag(self, mock_args: MagicMock, mock_adjust: MagicMock) -> None:
        """--full flag passes 'full' to adjust_all."""
        import argparse

        mock_args.return_value = argparse.Namespace(full=True)
        covid_adjustment.main()
        mock_adjust.assert_called_once_with("full")
