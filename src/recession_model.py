"""Recession probability classifier for the Macro Economic Dashboard.

Builds a monthly feature matrix from existing FRED series in the database,
trains logistic regression and random forest classifiers, evaluates with
time-series-aware cross-validation, and stores predictions in the
``recession_predictions`` table.

Pipeline position:
    data_pull → db_setup → covid_adjustment → export_csv →
    ai_insights → verify_insights → **recession_model** → dashboard
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import sqlite3
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger: logging.Logger = logging.getLogger(__name__)

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
SQL_DIR: Path = PROJECT_ROOT / "sql"
DATA_DIR: Path = PROJECT_ROOT / "data"

FEATURE_COLUMNS: list[str] = [
    "yield_spread",
    "yield_spread_3m_avg",
    "yield_inverted_months",
    "unrate",
    "unrate_12m_change",
    "u6_u3_gap",
    "gdp_growth_annualized",
    "cpi_yoy",
    "info_trades_divergence",
    "info_employment_yoy",
    "power_output_yoy",
    # Phase 14 — Hacker News tech-practitioner sentiment.
    "hn_sentiment_3m_avg",
    "hn_story_volume_yoy",
    "layoff_story_freq",
]

HN_FEATURE_COLUMNS: tuple[str, ...] = (
    "hn_sentiment_3m_avg",
    "hn_story_volume_yoy",
    "layoff_story_freq",
)
"""Subset of ``FEATURE_COLUMNS`` sourced from ``hn_sentiment_monthly``."""

TARGET_COLUMN: str = "recession_within_12m"

TRAIN_CUTOFF: str = "2023-12"
"""Training data ends at this month (inclusive). Holdout starts after."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _db_path(mode: str = "seed") -> Path:
    """Return the database path for the given mode.

    Args:
        mode: Either ``'seed'`` or ``'full'``.

    Returns:
        Absolute path to the SQLite database file.
    """
    filename = "seed.db" if mode == "seed" else "full.db"
    return DATA_DIR / filename


def _ensure_prediction_table(conn: sqlite3.Connection) -> None:
    """Create the ``recession_predictions`` table if it does not exist.

    Args:
        conn: Open SQLite connection.
    """
    schema_sql = (SQL_DIR / "04_prediction_schema.sql").read_text()
    conn.executescript(schema_sql)


def _load_series_monthly(
    conn: sqlite3.Connection,
    series_id: str,
    *,
    use_raw: bool = False,
) -> pd.DataFrame:
    """Load a single series from the database, aggregated to monthly.

    For daily series (T10Y2Y) this computes the monthly mean. For monthly
    and quarterly series the value is taken as-is.

    Args:
        conn: Open SQLite connection.
        series_id: FRED series identifier.
        use_raw: If ``True``, read the ``value`` column instead of
            ``value_covid_adjusted``.

    Returns:
        DataFrame with ``month`` (YYYY-MM string) and ``value`` columns,
        sorted by month.
    """
    value_col = "value" if use_raw else "value_covid_adjusted"
    query = f"""
        SELECT SUBSTR(date, 1, 7) AS month, AVG({value_col}) AS value
        FROM observations
        WHERE series_id = ?
        GROUP BY month
        ORDER BY month
    """
    return pd.read_sql_query(query, conn, params=(series_id,))


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------


def build_feature_matrix(db_path: Path) -> pd.DataFrame:
    """Build a monthly feature matrix from the observations table.

    Queries the database for all 10 series, engineers 11 features and a
    forward-looking binary target (recession within 12 months).

    Args:
        db_path: Path to the SQLite database.

    Returns:
        DataFrame indexed by ``month`` (YYYY-MM string) with 11 feature
        columns plus the ``recession_within_12m`` target.
    """
    conn = sqlite3.connect(str(db_path))
    try:
        df = _build_feature_matrix_from_conn(conn)
    finally:
        conn.close()
    return df


def _build_feature_matrix_from_conn(conn: sqlite3.Connection) -> pd.DataFrame:
    """Core feature-matrix builder operating on an open connection.

    Args:
        conn: Open SQLite connection with populated observations table.

    Returns:
        Feature matrix DataFrame.
    """
    # --- Yield spread features ---
    t10y2y = _load_series_monthly(conn, "T10Y2Y")
    t10y2y = t10y2y.rename(columns={"value": "yield_spread"})
    t10y2y["yield_spread_3m_avg"] = (
        t10y2y["yield_spread"].rolling(3, min_periods=1).mean()
    )
    t10y2y["yield_inverted_months"] = (
        (t10y2y["yield_spread"] < 0)
        .rolling(6, min_periods=1)
        .sum()
        .astype(int)
    )

    # --- Unemployment features ---
    unrate = _load_series_monthly(conn, "UNRATE")
    unrate = unrate.rename(columns={"value": "unrate"})
    unrate["unrate_12m_change"] = unrate["unrate"] - unrate["unrate"].shift(12)

    u6rate = _load_series_monthly(conn, "U6RATE")
    u6rate = u6rate.rename(columns={"value": "u6rate"})

    unemp = unrate.merge(u6rate, on="month", how="inner")
    unemp["u6_u3_gap"] = unemp["u6rate"] - unemp["unrate"]
    unemp = unemp.drop(columns=["u6rate"])

    # --- GDP growth (quarterly → monthly forward-fill) ---
    gdp = _load_series_monthly(conn, "GDPC1")
    gdp["gdp_growth_annualized"] = (
        (gdp["value"] / gdp["value"].shift(1) - 1) * 4 * 100
    )
    gdp = gdp.drop(columns=["value"])
    # Forward-fill quarterly growth rate to monthly
    gdp = gdp.set_index("month")
    all_months = t10y2y[["month"]].set_index("month")
    gdp = all_months.join(gdp, how="left")
    gdp["gdp_growth_annualized"] = gdp["gdp_growth_annualized"].ffill()
    gdp = gdp.reset_index()

    # --- CPI year-over-year ---
    cpi = _load_series_monthly(conn, "CPIAUCSL")
    cpi["cpi_yoy"] = (cpi["value"] / cpi["value"].shift(12) - 1) * 100
    cpi = cpi.drop(columns=["value"]).rename(columns={})

    # --- Per-capita employment: info vs trades divergence ---
    pop = _load_series_monthly(conn, "CNP16OV")
    pop = pop.rename(columns={"value": "pop"})

    info = _load_series_monthly(conn, "USINFO")
    info = info.rename(columns={"value": "info_raw"})

    trades = _load_series_monthly(conn, "CES2023800001")
    trades = trades.rename(columns={"value": "trades_raw"})

    emp = (
        pop.merge(info, on="month", how="inner")
        .merge(trades, on="month", how="inner")
    )
    emp["info_pc"] = emp["info_raw"] / emp["pop"] * 1000
    emp["trades_pc"] = emp["trades_raw"] / emp["pop"] * 1000

    # Index both to 100 at earliest month
    info_base = emp["info_pc"].iloc[0]
    trades_base = emp["trades_pc"].iloc[0]
    emp["info_index"] = emp["info_pc"] / info_base * 100
    emp["trades_index"] = emp["trades_pc"] / trades_base * 100
    emp["info_trades_divergence"] = emp["trades_index"] - emp["info_index"]

    # Info employment YoY growth (per-capita)
    emp["info_employment_yoy"] = (
        (emp["info_pc"] / emp["info_pc"].shift(12) - 1) * 100
    )

    emp = emp[["month", "info_trades_divergence", "info_employment_yoy"]]

    # --- Electric power YoY ---
    power = _load_series_monthly(conn, "IPG2211S")
    power["power_output_yoy"] = (
        (power["value"] / power["value"].shift(12) - 1) * 100
    )
    power = power.drop(columns=["value"])

    # --- Target: recession within 12 months (uses RAW USREC) ---
    usrec = _load_series_monthly(conn, "USREC", use_raw=True)
    usrec = usrec.rename(columns={"value": "usrec"})
    # Forward-looking: is there a recession in the next 12 months?
    usrec[TARGET_COLUMN] = (
        usrec["usrec"]
        .rolling(12, min_periods=1)
        .max()
        .shift(-12)
    )
    # Keep raw USREC for the actual column in predictions
    usrec["usrec_actual"] = usrec["usrec"].astype(int)
    usrec = usrec.drop(columns=["usrec"])

    # --- HN sentiment features (Phase 14) ---
    # hn_sentiment_monthly exists from 2022-01 onward. LEFT JOIN onto the
    # all-months index so pre-window rows surface as NaN rather than
    # dropping 5 years of training data.
    hn = _build_hn_feature_frame(conn, all_months_index=t10y2y[["month"]])

    # --- Merge all features ---
    frames = [t10y2y, unemp, gdp, cpi, emp, power, hn, usrec]
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on="month", how="inner")

    merged = merged.set_index("month").sort_index()

    # --- HN imputation: training-median fill for leading / sparse NaNs ---
    # Pre-2022 rows and any mid-window gap in ``hn_sentiment_monthly``
    # land here as NaN after the LEFT JOIN. We impute with the median
    # computed over the training mask (``month <= TRAIN_CUTOFF``) only --
    # a leak-free constant-impute that mirrors sklearn's
    # ``SimpleImputer`` contract. YoY-shift leading NaNs get 0.0 because
    # "no year-ago comparison" is neutral.
    merged = _impute_hn_features(merged)

    # Drop rows where features aren't available (first 12 months have NaN
    # from 12-month lags)
    feature_cols = [c for c in FEATURE_COLUMNS if c in merged.columns]
    merged = merged.dropna(subset=feature_cols)

    return merged


def _build_hn_feature_frame(
    conn: sqlite3.Connection,
    all_months_index: pd.DataFrame,
) -> pd.DataFrame:
    """Build the HN feature frame indexed on ``month``.

    Pulls ``hn_sentiment_monthly`` as-is, computes the three HN features
    (3-month rolling mean sentiment, YoY story volume, layoff-story
    share), and LEFT JOINs onto ``all_months_index`` so months without
    HN coverage produce NaN rows. The caller handles imputation.

    Args:
        conn: Open SQLite connection.
        all_months_index: DataFrame with a single ``month`` column
            spanning the full feature-matrix window.

    Returns:
        DataFrame with columns ``month``, ``hn_sentiment_3m_avg``,
        ``hn_story_volume_yoy``, ``layoff_story_freq``. Missing months
        carry NaN across all three HN columns.
    """
    hn_query: str = """
        SELECT
            SUBSTR(month, 1, 7) AS month,
            mean_sentiment,
            story_count,
            layoff_story_count
        FROM hn_sentiment_monthly
        ORDER BY month
    """
    hn = pd.read_sql_query(hn_query, conn)

    if hn.empty:
        # Empty HN table (dev DB or future coverage trim) -> all-NaN frame.
        empty = all_months_index.copy()
        for col in HN_FEATURE_COLUMNS:
            empty[col] = np.nan
        return empty

    hn["hn_sentiment_3m_avg"] = (
        hn["mean_sentiment"].rolling(3, min_periods=1).mean()
    )
    hn["hn_story_volume_yoy"] = (
        hn["story_count"] / hn["story_count"].shift(12) - 1
    )
    # Guard division by zero defensively even though story_count > 0 is
    # enforced by the aggregate's GROUP BY.
    hn["layoff_story_freq"] = np.where(
        hn["story_count"] > 0,
        hn["layoff_story_count"] / hn["story_count"].replace(0, np.nan),
        0.0,
    )

    hn = hn[["month", *HN_FEATURE_COLUMNS]]

    # LEFT JOIN onto the full months index so pre-2022 / sparse months
    # carry NaN (imputed later) rather than being dropped.
    return all_months_index.merge(hn, on="month", how="left")


def _impute_hn_features(merged: pd.DataFrame) -> pd.DataFrame:
    """Fill HN-feature NaNs with training-period medians (leak-free).

    ``hn_story_volume_yoy``'s first 12 in-window months carry NaN from
    the YoY shift; those are filled with 0.0 (neutral "no year-ago
    comparison"). Remaining NaNs -- pre-2022 rows and mid-window sparse
    months -- are filled with the median computed over the training
    mask (``month <= TRAIN_CUTOFF`` AND column is not NaN). Applying
    the training-derived median to test rows is the standard
    ``SimpleImputer`` contract: the imputer itself contains no test-set
    information.

    Args:
        merged: Feature matrix indexed by month with HN feature columns
            present (possibly with NaN leading/sparse rows).

    Returns:
        The same DataFrame with HN columns imputed. If an HN column is
        entirely NaN in the training window (impossible with the
        shipping seed), the fallback fill value is 0.0.
    """
    if "hn_story_volume_yoy" in merged.columns:
        merged["hn_story_volume_yoy"] = merged["hn_story_volume_yoy"].fillna(
            0.0
        )

    train_mask = merged.index <= TRAIN_CUTOFF
    for col in HN_FEATURE_COLUMNS:
        if col not in merged.columns:
            continue
        train_series = merged.loc[train_mask, col].dropna()
        median_value: float = (
            float(train_series.median()) if not train_series.empty else 0.0
        )
        merged[col] = merged[col].fillna(median_value)
    return merged


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------


def train_models(df: pd.DataFrame) -> dict:
    """Train logistic regression and random forest classifiers.

    Uses ``TimeSeriesSplit`` cross-validation and evaluates on a holdout
    test set (months after ``TRAIN_CUTOFF``).

    Args:
        df: Feature matrix from ``build_feature_matrix``. Must contain
            all ``FEATURE_COLUMNS`` plus ``recession_within_12m``.

    Returns:
        Dictionary with keys ``'logistic_regression'`` and
        ``'random_forest'``, each mapping to a dict containing
        ``'model'``, ``'scaler'`` (for LR), ``'metrics'``, and
        ``'feature_importance'``.
    """
    # Split train / test by date
    trainable = df[df[TARGET_COLUMN].notna()].copy()
    trainable[TARGET_COLUMN] = trainable[TARGET_COLUMN].astype(int)

    train_mask = trainable.index <= TRAIN_CUTOFF
    test_mask = trainable.index > TRAIN_CUTOFF

    X_train = trainable.loc[train_mask, FEATURE_COLUMNS]
    y_train = trainable.loc[train_mask, TARGET_COLUMN]
    X_test = trainable.loc[test_mask, FEATURE_COLUMNS]
    y_test = trainable.loc[test_mask, TARGET_COLUMN]

    logger.info(
        "Train: %d rows (%d positive). Test: %d rows (%d positive).",
        len(X_train),
        y_train.sum(),
        len(X_test),
        y_test.sum(),
    )

    results: dict = {}

    # --- Logistic Regression (baseline) ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lr = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
    )
    lr.fit(X_train_scaled, y_train)

    lr_metrics = evaluate_model(lr, X_test_scaled, y_test)
    lr_importance = dict(zip(FEATURE_COLUMNS, lr.coef_[0]))

    results["logistic_regression"] = {
        "model": lr,
        "scaler": scaler,
        "metrics": lr_metrics,
        "feature_importance": lr_importance,
    }
    logger.info("Logistic Regression — %s", lr_metrics)

    # --- Random Forest (tuned) ---
    tscv = TimeSeriesSplit(n_splits=5)
    rf_param_grid: dict = {
        "n_estimators": [100, 200, 500],
        "max_depth": [3, 5, 10, None],
        "min_samples_leaf": [5, 10, 20],
    }

    rf_base = RandomForestClassifier(
        class_weight="balanced",
        random_state=42,
    )
    grid_search = GridSearchCV(
        rf_base,
        rf_param_grid,
        cv=tscv,
        scoring="roc_auc",
        n_jobs=-1,
        refit=True,
        error_score=0.0,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        grid_search.fit(X_train, y_train)

    rf = grid_search.best_estimator_
    logger.info("Best RF params: %s", grid_search.best_params_)

    rf_metrics = evaluate_model(rf, X_test, y_test)
    rf_importance = dict(zip(FEATURE_COLUMNS, rf.feature_importances_))

    results["random_forest"] = {
        "model": rf,
        "scaler": None,
        "metrics": rf_metrics,
        "feature_importance": rf_importance,
    }
    logger.info("Random Forest — %s", rf_metrics)

    return results


def evaluate_model(
    model: LogisticRegression | RandomForestClassifier,
    X_test: np.ndarray | pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float]:
    """Compute classification metrics for a fitted model.

    Args:
        model: A fitted sklearn classifier with ``predict`` and
            ``predict_proba`` methods.
        X_test: Test feature matrix.
        y_test: True binary labels.

    Returns:
        Dictionary with ``precision``, ``recall``, ``f1``, and
        ``auc_roc`` values.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    unique_classes = y_test.unique()
    if len(unique_classes) < 2:
        # Edge case: test set has only one class — AUC undefined
        auc = float("nan")
    else:
        auc = roc_auc_score(y_test, y_proba)

    zero_div_val = 0.0
    return {
        "precision": precision_score(
            y_test, y_pred, zero_division=zero_div_val
        ),
        "recall": recall_score(y_test, y_pred, zero_division=zero_div_val),
        "f1": f1_score(y_test, y_pred, zero_division=zero_div_val),
        "auc_roc": auc,
    }


# ---------------------------------------------------------------------------
# Prediction output
# ---------------------------------------------------------------------------


def generate_predictions(db_path: Path, model_results: dict) -> None:
    """Generate predictions for every month and store in the database.

    Selects the model with the higher test AUC-ROC. If tied, prefers
    logistic regression for interpretability.

    Args:
        db_path: Path to the SQLite database.
        model_results: Output from ``train_models``.
    """
    # Pick best model
    lr_auc = model_results["logistic_regression"]["metrics"]["auc_roc"]
    rf_auc = model_results["random_forest"]["metrics"]["auc_roc"]

    if np.isnan(lr_auc) and np.isnan(rf_auc):
        best_name = "logistic_regression"
    elif np.isnan(lr_auc):
        best_name = "random_forest"
    elif np.isnan(rf_auc):
        best_name = "logistic_regression"
    elif rf_auc > lr_auc:
        best_name = "random_forest"
    else:
        best_name = "logistic_regression"

    best = model_results[best_name]
    model = best["model"]
    scaler = best["scaler"]

    logger.info(
        "Selected model: %s (AUC-ROC: %.4f)",
        best_name,
        model_results[best_name]["metrics"]["auc_roc"],
    )

    # Build full feature matrix
    df = build_feature_matrix(db_path)
    X_all = df[FEATURE_COLUMNS]

    if scaler is not None:
        X_all_transformed = scaler.transform(X_all)
    else:
        X_all_transformed = X_all

    probabilities = model.predict_proba(X_all_transformed)[:, 1]
    predictions = model.predict(X_all_transformed)

    generated_at = datetime.now(timezone.utc).isoformat()

    conn = sqlite3.connect(str(db_path))
    try:
        _ensure_prediction_table(conn)

        rows: list[tuple] = []
        for i, month in enumerate(df.index):
            # Use first day of month as date
            date_str = f"{month}-01"
            features_dict = df.loc[month, FEATURE_COLUMNS].to_dict()
            actual = (
                int(df.loc[month, "usrec_actual"])
                if "usrec_actual" in df.columns
                else None
            )

            rows.append((
                date_str,
                float(probabilities[i]),
                int(predictions[i]),
                actual,
                best_name,
                json.dumps(features_dict),
                generated_at,
            ))

        conn.executemany(
            """
            INSERT OR REPLACE INTO recession_predictions
                (date, probability, prediction, actual, model_name,
                 features_json, generated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()
        logger.info("Wrote %d predictions to recession_predictions.", len(rows))
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Scenario grid
# ---------------------------------------------------------------------------

# Coarser grid ranges for the 4 slider features (keeps total under ~12k rows)
SCENARIO_GRID_RANGES: dict[str, dict[str, float]] = {
    "yield_spread": {"min": -1.5, "max": 3.0, "step": 0.5},
    "unrate": {"min": 3.0, "max": 7.0, "step": 0.5},
    "gdp_growth_annualized": {"min": -4.0, "max": 8.0, "step": 1.0},
    "cpi_yoy": {"min": 1.0, "max": 10.0, "step": 1.0},
}

SLIDER_FEATURES: list[str] = list(SCENARIO_GRID_RANGES.keys())


def _ensure_scenario_table(conn: sqlite3.Connection) -> None:
    """Create the ``scenario_grid`` table if it does not exist.

    Args:
        conn: Open SQLite connection.
    """
    schema_sql = (SQL_DIR / "05_scenario_schema.sql").read_text()
    conn.executescript(schema_sql)


def generate_scenario_grid(db_path: Path, model_results: dict) -> None:
    """Generate a pre-computed scenario grid for the What If explorer.

    Varies 4 slider features across defined ranges while holding the
    remaining 7 features at their latest observed values. Stores all
    scenario probabilities in the ``scenario_grid`` table.

    Args:
        db_path: Path to the SQLite database.
        model_results: Output from ``train_models``, containing the
            trained models and scalers.
    """
    # Pick best model (same logic as generate_predictions)
    lr_auc = model_results["logistic_regression"]["metrics"]["auc_roc"]
    rf_auc = model_results["random_forest"]["metrics"]["auc_roc"]

    if np.isnan(lr_auc) and np.isnan(rf_auc):
        best_name = "logistic_regression"
    elif np.isnan(lr_auc):
        best_name = "random_forest"
    elif np.isnan(rf_auc):
        best_name = "logistic_regression"
    elif rf_auc > lr_auc:
        best_name = "random_forest"
    else:
        best_name = "logistic_regression"

    best = model_results[best_name]
    model = best["model"]
    scaler = best["scaler"]

    # Get latest feature values from the most recent prediction row
    conn = sqlite3.connect(str(db_path))
    try:
        latest_row = pd.read_sql_query(
            "SELECT features_json FROM recession_predictions "
            "ORDER BY date DESC LIMIT 1",
            conn,
        )
    finally:
        conn.close()

    if latest_row.empty:
        logger.warning("No predictions found. Run generate_predictions() first.")
        return

    latest_features: dict[str, float] = json.loads(
        latest_row["features_json"].iloc[0]
    )

    # Build grid combinations for 4 slider features
    grid_axes: list[list[float]] = []
    for feat in SLIDER_FEATURES:
        r = SCENARIO_GRID_RANGES[feat]
        values = np.arange(r["min"], r["max"] + r["step"] / 2, r["step"])
        grid_axes.append(values.tolist())

    combinations = list(itertools.product(*grid_axes))
    logger.info("Scenario grid: %d combinations.", len(combinations))

    # Fixed features (the 7 not adjustable via sliders)
    fixed_features: list[str] = [
        f for f in FEATURE_COLUMNS if f not in SLIDER_FEATURES
    ]

    # Build feature matrix for all grid points
    rows_data: list[list[float]] = []
    for combo in combinations:
        row = []
        slider_map = dict(zip(SLIDER_FEATURES, combo))
        for feat in FEATURE_COLUMNS:
            if feat in slider_map:
                row.append(slider_map[feat])
            else:
                row.append(latest_features[feat])
        rows_data.append(row)

    X_grid = np.array(rows_data)

    if scaler is not None:
        X_grid_transformed = scaler.transform(X_grid)
    else:
        X_grid_transformed = X_grid

    probabilities = model.predict_proba(X_grid_transformed)[:, 1]

    generated_at = datetime.now(timezone.utc).isoformat()

    # Write to database atomically
    conn = sqlite3.connect(str(db_path))
    try:
        _ensure_scenario_table(conn)

        conn.execute("DELETE FROM scenario_grid")
        insert_rows: list[tuple] = []
        for i, combo in enumerate(combinations):
            insert_rows.append((
                combo[0],  # yield_spread
                combo[1],  # unrate
                combo[2],  # gdp_growth_annualized
                combo[3],  # cpi_yoy
                float(probabilities[i]),
                best_name,
                generated_at,
            ))

        conn.executemany(
            """
            INSERT INTO scenario_grid
                (yield_spread, unrate, gdp_growth_annualized, cpi_yoy,
                 probability, model_name, generated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            insert_rows,
        )
        conn.commit()
        logger.info(
            "Wrote %d rows to scenario_grid.", len(insert_rows)
        )
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _print_feature_importance(model_results: dict) -> None:
    """Print feature importance rankings for both models.

    Args:
        model_results: Output from ``train_models``.
    """
    for name, result in model_results.items():
        label = name.replace("_", " ").title()
        importance = result["feature_importance"]
        sorted_feats = sorted(
            importance.items(), key=lambda x: abs(x[1]), reverse=True
        )
        print(f"\n{'=' * 50}")
        print(f"  {label} — Feature Importance")
        print(f"{'=' * 50}")
        for feat, val in sorted_feats:
            print(f"  {feat:<30s}  {val:+.4f}")


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Train recession probability classifier and store predictions.",
    )
    parser.add_argument(
        "--db",
        choices=["seed", "full"],
        default="seed",
        help="Which database to use (default: seed).",
    )
    parser.add_argument(
        "--importance-only",
        action="store_true",
        help="Show feature importance without writing predictions.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point: build features, train models, generate predictions."""
    args = _parse_args()
    db = _db_path(args.db)

    logger.info("Building feature matrix from %s ...", db.name)
    df = build_feature_matrix(db)
    logger.info("Feature matrix: %d rows, %d columns.", *df.shape)

    logger.info("Training models ...")
    results = train_models(df)

    _print_feature_importance(results)

    if args.importance_only:
        logger.info("--importance-only set. Skipping prediction storage.")
        return

    logger.info("Generating predictions ...")
    generate_predictions(db, results)

    logger.info("Generating scenario grid ...")
    generate_scenario_grid(db, results)
    logger.info("Done.")


if __name__ == "__main__":
    main()
