"""Tests for src/recession_model.py.

All tests use in-memory SQLite databases with synthetic data. No real seed.db
required.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import recession_model


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SCHEMA_SQL: str = (recession_model.SQL_DIR / "01_schema.sql").read_text()
PREDICTION_SCHEMA_SQL: str = (
    recession_model.SQL_DIR / "04_prediction_schema.sql"
).read_text()

# Series needed for feature matrix
_SERIES_META: list[tuple[str, str, str, str]] = [
    ("T10Y2Y", "10Y-2Y Spread", "yield_curve", "daily"),
    ("UNRATE", "Unemployment Rate", "labor_market", "monthly"),
    ("U6RATE", "Underemployment Rate", "labor_market", "monthly"),
    ("GDPC1", "Real GDP", "output_growth", "quarterly"),
    ("CPIAUCSL", "CPI", "prices", "monthly"),
    ("CNP16OV", "Population 16+", "population", "monthly"),
    ("USINFO", "Info Employment", "ai_labor", "monthly"),
    ("CES2023800001", "Trades Employment", "ai_labor", "monthly"),
    ("IPG2211S", "Electric Power", "ai_energy", "monthly"),
    ("USREC", "Recession Indicator", "recession", "monthly"),
]


def _populate_synthetic_db(conn: sqlite3.Connection) -> None:
    """Insert 9 years of synthetic monthly data into an in-memory DB.

    Creates 108 months (2017-01 to 2025-12) with two recession windows
    so that both training (up to 2023-12) and holdout (2024-01+) sets
    contain positive target labels.

    Args:
        conn: Open in-memory SQLite connection with schema applied.
    """
    # Insert series metadata
    for sid, name, cat, freq in _SERIES_META:
        conn.execute(
            "INSERT OR IGNORE INTO series_metadata "
            "(series_id, name, category, frequency) VALUES (?, ?, ?, ?)",
            (sid, name, cat, freq),
        )

    rng = np.random.default_rng(42)
    months = pd.date_range("2017-01-01", periods=108, freq="MS")

    for month in months:
        date_str = month.strftime("%Y-%m-%d")
        month_idx = (month.year - 2017) * 12 + month.month

        # T10Y2Y: daily-like but we insert one per month for simplicity.
        # Spread that goes negative near recession windows.
        if 30 <= month_idx <= 42:
            spread = -0.3 + rng.normal(0, 0.1)
        elif 84 <= month_idx <= 95:
            spread = -0.2 + rng.normal(0, 0.1)
        else:
            spread = 0.8 + rng.normal(0, 0.1)
        _insert_obs(conn, "T10Y2Y", date_str, spread)

        # UNRATE: rises during recession windows
        in_recession = (36 <= month_idx <= 38) or (90 <= month_idx <= 92)
        unrate = 3.5 + (3.0 if in_recession else 0.0)
        unrate += rng.normal(0, 0.1)
        _insert_obs(conn, "UNRATE", date_str, unrate)

        # U6RATE: ~2pp above UNRATE
        u6 = unrate + 2.0 + rng.normal(0, 0.05)
        _insert_obs(conn, "U6RATE", date_str, u6)

        # GDPC1: quarterly only (Jan, Apr, Jul, Oct)
        if month.month in (1, 4, 7, 10):
            gdp = 20000 + month_idx * 50 + rng.normal(0, 20)
            _insert_obs(conn, "GDPC1", date_str, gdp)

        # CPIAUCSL: trending up
        cpi = 250 + month_idx * 0.3 + rng.normal(0, 0.1)
        _insert_obs(conn, "CPIAUCSL", date_str, cpi)

        # CNP16OV: slowly growing population
        pop = 260000 + month_idx * 10 + rng.normal(0, 5)
        _insert_obs(conn, "CNP16OV", date_str, pop)

        # USINFO: employment with dip during recession
        info = 2800 - (200 if in_recession else 0)
        info += rng.normal(0, 10)
        _insert_obs(conn, "USINFO", date_str, info)

        # CES2023800001: trades employment, steadier
        trades = 3000 + month_idx * 2 + rng.normal(0, 10)
        _insert_obs(conn, "CES2023800001", date_str, trades)

        # IPG2211S: electric power
        power = 100 + rng.normal(0, 2)
        _insert_obs(conn, "IPG2211S", date_str, power)

        # USREC: two recession windows (2020-01 to 2020-03 and 2024-07 to 2024-09)
        usrec = 1.0 if in_recession else 0.0
        _insert_obs(conn, "USREC", date_str, usrec)

    conn.commit()


def _insert_obs(
    conn: sqlite3.Connection,
    series_id: str,
    date: str,
    value: float,
) -> None:
    """Insert a single observation with covid_adjusted equal to value.

    Args:
        conn: Open SQLite connection.
        series_id: FRED series identifier.
        date: ISO date string.
        value: Observation value.
    """
    conn.execute(
        "INSERT OR IGNORE INTO observations "
        "(series_id, date, value, value_covid_adjusted) "
        "VALUES (?, ?, ?, ?)",
        (series_id, date, value, value),
    )


@pytest.fixture()
def mem_conn() -> sqlite3.Connection:
    """Return an in-memory DB with schema and synthetic data.

    Returns:
        Open SQLite connection ready for feature matrix building.
    """
    conn = sqlite3.connect(":memory:")
    conn.executescript(SCHEMA_SQL)
    _populate_synthetic_db(conn)
    return conn


@pytest.fixture()
def feature_df(mem_conn: sqlite3.Connection) -> pd.DataFrame:
    """Return a feature matrix built from the synthetic DB.

    Args:
        mem_conn: Fixture providing a populated in-memory database.

    Returns:
        Feature matrix DataFrame.
    """
    return recession_model._build_feature_matrix_from_conn(mem_conn)


@pytest.fixture()
def trained_results(feature_df: pd.DataFrame) -> dict:
    """Return model training results from the synthetic feature matrix.

    Args:
        feature_df: Fixture providing the feature matrix.

    Returns:
        Dictionary of trained model results.
    """
    return recession_model.train_models(feature_df)


# ---------------------------------------------------------------------------
# Feature Engineering Tests
# ---------------------------------------------------------------------------


class TestBuildFeatureMatrixColumns:
    """Tests for build_feature_matrix column structure."""

    def test_returns_expected_columns(
        self, feature_df: pd.DataFrame
    ) -> None:
        """Feature matrix contains all 11 feature columns plus target."""
        for col in recession_model.FEATURE_COLUMNS:
            assert col in feature_df.columns, f"Missing column: {col}"
        assert recession_model.TARGET_COLUMN in feature_df.columns

    def test_index_is_month_string(
        self, feature_df: pd.DataFrame
    ) -> None:
        """Index values are YYYY-MM formatted strings."""
        for month in feature_df.index:
            assert len(month) == 7
            assert month[4] == "-"

    def test_no_feature_nans(self, feature_df: pd.DataFrame) -> None:
        """No NaN values in feature columns after lag-row removal."""
        nans = feature_df[recession_model.FEATURE_COLUMNS].isnull().sum()
        assert nans.sum() == 0, f"Unexpected NaN features:\n{nans[nans > 0]}"


class TestBuildFeatureMatrixLeakage:
    """Tests for target variable construction and leakage prevention."""

    def test_no_future_leakage(
        self, mem_conn: sqlite3.Connection
    ) -> None:
        """Target label only uses USREC values from future months.

        For each month, recession_within_12m should be 1 only if there is a
        recession in the following 12 months, never from the current or past.
        """
        df = recession_model._build_feature_matrix_from_conn(mem_conn)
        usrec = recession_model._load_series_monthly(
            mem_conn, "USREC", use_raw=True
        )
        usrec = usrec.set_index("month")

        for month in df.index:
            target_val = df.loc[month, recession_model.TARGET_COLUMN]
            if pd.isna(target_val):
                continue
            # Check the next 12 months for recession
            month_dt = pd.Timestamp(month + "-01")
            future_months = pd.date_range(
                month_dt + pd.DateOffset(months=1),
                periods=12,
                freq="MS",
            )
            future_keys = [m.strftime("%Y-%m") for m in future_months]
            has_recession = any(
                usrec.loc[k, "value"] > 0
                for k in future_keys
                if k in usrec.index
            )
            assert target_val == float(has_recession), (
                f"Target mismatch at {month}: "
                f"expected {float(has_recession)}, got {target_val}"
            )

    def test_trailing_rows_have_nan_target(
        self, feature_df: pd.DataFrame
    ) -> None:
        """The last 12 months should have NaN targets (incomplete window)."""
        last_12 = feature_df.tail(12)[recession_model.TARGET_COLUMN]
        assert last_12.isna().sum() > 0, "Expected NaN targets in trailing rows"


class TestFeatureMatrixGDP:
    """Tests for GDP forward-fill handling."""

    def test_gdp_forward_filled(self, feature_df: pd.DataFrame) -> None:
        """GDP growth is forward-filled from quarterly to monthly."""
        gdp_col = feature_df["gdp_growth_annualized"]
        # Should have no NaN after the lag warmup
        assert gdp_col.isna().sum() == 0


class TestFeatureMatrixPerCapita:
    """Tests for per-capita normalization."""

    def test_per_capita_normalization_applied(
        self, mem_conn: sqlite3.Connection
    ) -> None:
        """Info/trades divergence uses per-capita normalized values.

        The divergence feature should not equal the raw difference between
        USINFO and CES2023800001.
        """
        df = recession_model._build_feature_matrix_from_conn(mem_conn)

        # Load raw values for a sample month
        sample_month = df.index[5]
        raw_info = mem_conn.execute(
            "SELECT AVG(value_covid_adjusted) FROM observations "
            "WHERE series_id = 'USINFO' AND SUBSTR(date, 1, 7) = ?",
            (sample_month,),
        ).fetchone()[0]
        raw_trades = mem_conn.execute(
            "SELECT AVG(value_covid_adjusted) FROM observations "
            "WHERE series_id = 'CES2023800001' AND SUBSTR(date, 1, 7) = ?",
            (sample_month,),
        ).fetchone()[0]

        raw_diff = raw_trades - raw_info
        actual_div = df.loc[sample_month, "info_trades_divergence"]

        # The indexed, per-capita divergence should differ from raw difference
        assert actual_div != pytest.approx(raw_diff, abs=1.0), (
            "Divergence appears to use raw values instead of per-capita"
        )


# ---------------------------------------------------------------------------
# Model Training Tests
# ---------------------------------------------------------------------------


class TestTrainModels:
    """Tests for train_models function."""

    def test_returns_both_models(self, trained_results: dict) -> None:
        """Result contains both logistic_regression and random_forest."""
        assert "logistic_regression" in trained_results
        assert "random_forest" in trained_results

    def test_logistic_regression_baseline_runs(
        self, trained_results: dict
    ) -> None:
        """Logistic regression model is fitted and has coefficients."""
        lr_result = trained_results["logistic_regression"]
        model = lr_result["model"]
        assert hasattr(model, "coef_")
        assert model.coef_.shape[1] == len(recession_model.FEATURE_COLUMNS)

    def test_random_forest_grid_search_runs(
        self, trained_results: dict
    ) -> None:
        """Random forest model is fitted with feature importances."""
        rf_result = trained_results["random_forest"]
        model = rf_result["model"]
        assert hasattr(model, "feature_importances_")
        assert len(model.feature_importances_) == len(
            recession_model.FEATURE_COLUMNS
        )

    def test_time_series_split_no_future_leakage(
        self, feature_df: pd.DataFrame
    ) -> None:
        """TimeSeriesSplit train indices are always before test indices."""
        from sklearn.model_selection import TimeSeriesSplit

        trainable = feature_df[
            feature_df[recession_model.TARGET_COLUMN].notna()
        ]
        train_data = trainable[
            trainable.index <= recession_model.TRAIN_CUTOFF
        ]
        X = train_data[recession_model.FEATURE_COLUMNS]

        tscv = TimeSeriesSplit(n_splits=5)
        for train_idx, test_idx in tscv.split(X):
            assert train_idx.max() < test_idx.min(), (
                "Train indices must precede test indices"
            )

    def test_class_weight_balanced_used(
        self, trained_results: dict
    ) -> None:
        """Both models use class_weight='balanced'."""
        lr = trained_results["logistic_regression"]["model"]
        rf = trained_results["random_forest"]["model"]
        assert lr.class_weight == "balanced"
        assert rf.class_weight == "balanced"


# ---------------------------------------------------------------------------
# Evaluation Tests
# ---------------------------------------------------------------------------


class TestEvaluateModel:
    """Tests for evaluate_model function."""

    def test_returns_expected_metrics(
        self, trained_results: dict
    ) -> None:
        """Evaluation returns precision, recall, f1, and auc_roc."""
        for name in ("logistic_regression", "random_forest"):
            metrics = trained_results[name]["metrics"]
            assert "precision" in metrics
            assert "recall" in metrics
            assert "f1" in metrics
            assert "auc_roc" in metrics

    def test_metrics_are_numeric(self, trained_results: dict) -> None:
        """All metric values are floats."""
        for name in ("logistic_regression", "random_forest"):
            metrics = trained_results[name]["metrics"]
            for key, val in metrics.items():
                assert isinstance(val, float), f"{name}.{key} is not float"

    def test_handles_single_class(self) -> None:
        """AUC-ROC is NaN when test set has only one class."""
        from sklearn.linear_model import LogisticRegression

        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y_train = pd.Series([0, 1, 0, 1])
        y_test = pd.Series([0, 0])

        model = LogisticRegression(max_iter=1000)
        model.fit(X, y_train)

        metrics = recession_model.evaluate_model(model, X[:2], y_test)
        assert np.isnan(metrics["auc_roc"])


# ---------------------------------------------------------------------------
# Prediction Output Tests
# ---------------------------------------------------------------------------


class TestGeneratePredictions:
    """Tests for generate_predictions function."""

    def test_populates_table(self, tmp_path: Path) -> None:
        """generate_predictions creates rows in recession_predictions."""
        db = self._create_test_db(tmp_path)
        results = self._train_on_db(db)

        recession_model.generate_predictions(db, results)

        conn = sqlite3.connect(str(db))
        count = conn.execute(
            "SELECT COUNT(*) FROM recession_predictions"
        ).fetchone()[0]
        conn.close()
        assert count > 0

    def test_probability_range(self, tmp_path: Path) -> None:
        """All probabilities are between 0 and 1."""
        db = self._create_test_db(tmp_path)
        results = self._train_on_db(db)
        recession_model.generate_predictions(db, results)

        conn = sqlite3.connect(str(db))
        rows = conn.execute(
            "SELECT probability FROM recession_predictions"
        ).fetchall()
        conn.close()

        for (prob,) in rows:
            assert 0.0 <= prob <= 1.0, f"Probability out of range: {prob}"

    def test_features_json_valid(self, tmp_path: Path) -> None:
        """Stored features_json is valid JSON with expected keys."""
        db = self._create_test_db(tmp_path)
        results = self._train_on_db(db)
        recession_model.generate_predictions(db, results)

        conn = sqlite3.connect(str(db))
        rows = conn.execute(
            "SELECT features_json FROM recession_predictions LIMIT 3"
        ).fetchall()
        conn.close()

        for (fj,) in rows:
            parsed = json.loads(fj)
            assert isinstance(parsed, dict)
            for col in recession_model.FEATURE_COLUMNS:
                assert col in parsed, f"Missing feature key: {col}"

    def test_idempotent_rerun(self, tmp_path: Path) -> None:
        """Running twice does not duplicate rows."""
        db = self._create_test_db(tmp_path)
        results = self._train_on_db(db)

        recession_model.generate_predictions(db, results)
        conn = sqlite3.connect(str(db))
        count1 = conn.execute(
            "SELECT COUNT(*) FROM recession_predictions"
        ).fetchone()[0]
        conn.close()

        recession_model.generate_predictions(db, results)
        conn = sqlite3.connect(str(db))
        count2 = conn.execute(
            "SELECT COUNT(*) FROM recession_predictions"
        ).fetchone()[0]
        conn.close()

        assert count1 == count2, "Row count changed on second run"

    # --- Helpers ---

    @staticmethod
    def _create_test_db(tmp_path: Path) -> Path:
        """Create a temporary database with schema and synthetic data.

        Args:
            tmp_path: Pytest temporary directory.

        Returns:
            Path to the test database.
        """
        db = tmp_path / "test.db"
        conn = sqlite3.connect(str(db))
        conn.executescript(SCHEMA_SQL)
        conn.executescript(PREDICTION_SCHEMA_SQL)
        _populate_synthetic_db(conn)
        conn.close()
        return db

    @staticmethod
    def _train_on_db(db: Path) -> dict:
        """Build features and train models on a test database.

        Args:
            db: Path to a populated test database.

        Returns:
            Model training results.
        """
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            df = recession_model.build_feature_matrix(db)
            return recession_model.train_models(df)
