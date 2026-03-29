"""Database setup and loading script for the Macro Economic Dashboard.

Creates the SQLite schema and loads FRED data from cached JSON files.
Supports two modes:
  --seed (default): loads all 10 series, last 10 years only.
                    Writes to data/seed.db.
  --full:           loads all 10 series, full history.
                    Writes to data/full.db.
"""

import argparse
import json
import logging
import sqlite3
import sys
from datetime import date
from pathlib import Path

from series_config import SERIES, SERIES_IDS, SeriesInfo

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
SQL_DIR: Path = PROJECT_ROOT / "sql"
RAW_DIR: Path = PROJECT_ROOT / "data" / "raw"
DATA_DIR: Path = PROJECT_ROOT / "data"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger: logging.Logger = logging.getLogger(__name__)

SEED_SERIES: list[str] = SERIES_IDS
"""Series included in seed mode. All 10 series, last SEED_YEARS of history."""

SEED_YEARS: int = 10
"""Number of years of history to include in seed mode."""


def _seed_cutoff_date() -> str:
    """Calculate the cutoff date for seed mode (today minus SEED_YEARS).

    Returns:
        ISO date string for the earliest date to include.
    """
    today: date = date.today()
    cutoff: date = today.replace(year=today.year - SEED_YEARS)
    return cutoff.isoformat()


def _db_path(mode: str) -> Path:
    """Return the database file path for the given mode.

    Args:
        mode: Either 'seed' or 'full'.

    Returns:
        Path to the SQLite database file.
    """
    filename: str = "seed.db" if mode == "seed" else "full.db"
    return DATA_DIR / filename


def _create_schema(conn: sqlite3.Connection) -> None:
    """Read and execute the schema SQL file to create all tables.

    Args:
        conn: Open SQLite connection.
    """
    schema_path: Path = SQL_DIR / "01_schema.sql"
    schema_sql: str = schema_path.read_text()
    conn.executescript(schema_sql)
    logger.info("Schema created from %s", schema_path.name)


def _load_json(series_id: str) -> dict | None:
    """Load and return the cached JSON for a series.

    Args:
        series_id: FRED series identifier.

    Returns:
        Parsed JSON dict, or None if the file is missing.
    """
    path: Path = RAW_DIR / f"{series_id}.json"
    if not path.exists():
        logger.warning("No cached JSON for %s — skipping", series_id)
        return None
    with path.open() as f:
        return json.load(f)


def _insert_metadata(conn: sqlite3.Connection, data: dict) -> None:
    """Insert or update a row in series_metadata.

    Args:
        conn: Open SQLite connection.
        data: Parsed JSON dict from a raw cache file.
    """
    conn.execute(
        """
        INSERT INTO series_metadata (
            series_id, name, category, frequency,
            units, seasonal_adjustment, last_updated
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(series_id) DO UPDATE SET
            name = excluded.name,
            units = excluded.units,
            seasonal_adjustment = excluded.seasonal_adjustment,
            last_updated = excluded.last_updated
        """,
        (
            data["series_id"],
            data["name"],
            data["category"],
            data["frequency"],
            data.get("units"),
            data.get("seasonal_adjustment"),
            data.get("last_updated"),
        ),
    )


def _insert_observations(
    conn: sqlite3.Connection,
    series_id: str,
    observations: list[dict],
    cutoff_date: str | None = None,
) -> dict[str, int]:
    """Insert observations into the observations table.

    Skips rows with null values or invalid types. Uses INSERT OR IGNORE
    to handle the unique constraint on (series_id, date) for idempotent loads.

    Args:
        conn: Open SQLite connection.
        series_id: FRED series identifier.
        observations: List of {"date": str, "value": float|None} dicts.
        cutoff_date: If set, only include observations on or after this date.

    Returns:
        Dict with counts: 'inserted', 'skipped_null', 'skipped_date'.
    """
    inserted: int = 0
    skipped_null: int = 0
    skipped_date: int = 0

    for obs in observations:
        obs_date: str = obs["date"]
        value = obs["value"]

        if cutoff_date and obs_date < cutoff_date:
            skipped_date += 1
            continue

        if value is None:
            skipped_null += 1
            continue

        if not isinstance(value, (int, float)):
            skipped_null += 1
            continue

        float_value: float = float(value)
        conn.execute(
            """
            INSERT OR IGNORE INTO observations (series_id, date, value, value_covid_adjusted)
            VALUES (?, ?, ?, ?)
            """,
            (series_id, obs_date, float_value, float_value),
        )
        inserted += 1

    return {
        "inserted": inserted,
        "skipped_null": skipped_null,
        "skipped_date": skipped_date,
    }


def _load_series(
    conn: sqlite3.Connection,
    series_ids: list[str],
    cutoff_date: str | None = None,
) -> dict[str, dict[str, int]]:
    """Load one or more series from cached JSON into the database.

    Args:
        conn: Open SQLite connection.
        series_ids: List of FRED series IDs to load.
        cutoff_date: If set, only include observations on or after this date.

    Returns:
        Dict mapping series_id to its insertion stats.
    """
    stats: dict[str, dict[str, int]] = {}

    for series_id in series_ids:
        data: dict | None = _load_json(series_id)
        if data is None:
            continue

        _insert_metadata(conn, data)
        result: dict[str, int] = _insert_observations(
            conn, series_id, data["observations"], cutoff_date
        )
        stats[series_id] = result

        logger.info(
            "  %-16s  inserted=%d  skipped_null=%d  skipped_date=%d",
            series_id,
            result["inserted"],
            result["skipped_null"],
            result["skipped_date"],
        )

    conn.commit()
    return stats


def _print_summary(conn: sqlite3.Connection, stats: dict[str, dict[str, int]]) -> None:
    """Print a summary of what was loaded into the database.

    Args:
        conn: Open SQLite connection.
        stats: Dict mapping series_id to insertion stats.
    """
    print("\n" + "=" * 70)
    print("LOAD SUMMARY")
    print("=" * 70)

    # Row counts per table
    for table in ("series_metadata", "observations", "ai_insights"):
        row: tuple = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()  # noqa: S608
        print(f"  {table:25s}  {row[0]:>8,} rows")

    # Date range per series
    print("\n  Date ranges:")
    rows = conn.execute(
        """
        SELECT series_id, MIN(date) AS min_date, MAX(date) AS max_date, COUNT(*) AS n
        FROM observations
        GROUP BY series_id
        ORDER BY series_id
        """
    ).fetchall()
    for series_id, min_date, max_date, n in rows:
        print(f"    {series_id:16s}  {min_date} → {max_date}  ({n:,} obs)")

    # Null/skip totals
    total_inserted: int = sum(s["inserted"] for s in stats.values())
    total_null: int = sum(s["skipped_null"] for s in stats.values())
    total_date: int = sum(s["skipped_date"] for s in stats.values())
    print(f"\n  Total inserted:     {total_inserted:>8,}")
    print(f"  Total skipped null: {total_null:>8,}")
    print(f"  Total skipped date: {total_date:>8,}")
    print("=" * 70 + "\n")


def build_database(mode: str = "seed") -> Path:
    """Create and populate the database.

    Args:
        mode: 'seed' for curated subset, 'full' for everything.

    Returns:
        Path to the created database file.
    """
    db_path: Path = _db_path(mode)

    # Remove existing DB so we get a clean build every time
    if db_path.exists():
        db_path.unlink()
        logger.info("Removed existing %s", db_path.name)

    if mode == "seed":
        series_ids: list[str] = SEED_SERIES
        cutoff_date: str | None = _seed_cutoff_date()
        logger.info(
            "Seed mode: %d series, observations from %s onward",
            len(series_ids),
            cutoff_date,
        )
    else:
        series_ids = SERIES_IDS
        cutoff_date = None
        logger.info("Full mode: all %d series, full history", len(series_ids))

    conn: sqlite3.Connection = sqlite3.connect(db_path)
    try:
        _create_schema(conn)
        stats: dict[str, dict[str, int]] = _load_series(conn, series_ids, cutoff_date)
        _print_summary(conn, stats)
    finally:
        conn.close()

    size_mb: float = db_path.stat().st_size / (1024 * 1024)
    logger.info("Database written to %s (%.2f MB)", db_path, size_mb)
    return db_path


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Create and populate the SQLite database from cached FRED JSON.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Load all series with full history. Default is seed mode (all series, last 10 years).",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for the database setup script."""
    args: argparse.Namespace = _parse_args()
    mode: str = "full" if args.full else "seed"
    build_database(mode)


if __name__ == "__main__":
    main()
