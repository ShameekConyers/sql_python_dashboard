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
SCHOLARLY_DIR: Path = PROJECT_ROOT / "data" / "reference_sources" / "scholarly"
"""Directory holding curated scholarly reference JSON fixtures (Phase 12)."""

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
    """Read and execute the schema SQL files to create all tables.

    Loads `01_schema.sql` (core star schema + ai_insights) followed by
    `06_reference_schema.sql` (reference_docs for Phase 11 RAG citations).

    Args:
        conn: Open SQLite connection.
    """
    for schema_filename in ("01_schema.sql", "06_reference_schema.sql"):
        schema_path: Path = SQL_DIR / schema_filename
        schema_sql: str = schema_path.read_text()
        conn.executescript(schema_sql)
        logger.info("Schema applied from %s", schema_path.name)


def _ensure_citations_column(conn: sqlite3.Connection) -> None:
    """Add citations_json to ai_insights if the column is missing.

    SQLite does not support ``ADD COLUMN IF NOT EXISTS``, so a naked ALTER in
    a schema file would fail on the second run. This helper reads the current
    ai_insights columns via PRAGMA and only performs the ALTER when the
    column is absent. On a fresh DB (schema already includes the column) this
    is a no-op; on a pre-Phase-11 DB it adds the column exactly once.

    Args:
        conn: Open SQLite connection.
    """
    rows = conn.execute("PRAGMA table_info('ai_insights')").fetchall()
    columns: set[str] = {row[1] for row in rows}
    if "citations_json" in columns:
        return
    conn.execute(
        "ALTER TABLE ai_insights "
        "ADD COLUMN citations_json TEXT NOT NULL DEFAULT '[]'"
    )
    conn.commit()
    logger.info("Added citations_json column to ai_insights")


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


def _metadata_json_path(series_id: str) -> Path:
    """Return the metadata JSON cache path for a series.

    Args:
        series_id: FRED series identifier.

    Returns:
        Path to ``data/raw/{series_id}_metadata.json``.
    """
    return RAW_DIR / f"{series_id}_metadata.json"


def _load_reference_docs(conn: sqlite3.Connection, mode: str) -> int:
    """Load reference_docs rows from cached metadata JSON files.

    For each series active in ``mode``, read ``{series_id}_metadata.json``
    from ``data/raw/`` and upsert up to three reference_docs rows:
    ``series_notes``, ``release_info``, and ``category_path``. Rows with
    empty content are skipped. Existing rows are replaced via
    ``INSERT OR REPLACE`` keyed on the unique (series_id, doc_type) pair.

    Args:
        conn: Open SQLite connection.
        mode: Either 'seed' or 'full'. Reserved for future filtering; today
            both modes use the same 10-series set.

    Returns:
        Count of rows inserted or replaced.
    """
    series_ids: list[str] = SEED_SERIES if mode == "seed" else SERIES_IDS
    inserted: int = 0

    for series_id in series_ids:
        metadata_path: Path = _metadata_json_path(series_id)
        if not metadata_path.exists():
            logger.info(
                "No metadata JSON for %s — skipping reference_docs load",
                series_id,
            )
            continue

        with metadata_path.open() as f:
            metadata: dict = json.load(f)

        fetched_at: str = metadata.get("fetched_at", "")

        rows: list[tuple[str, str, str, str, str | None, str]] = []

        notes: str = (metadata.get("series_notes") or "").strip()
        if notes:
            rows.append(
                (
                    series_id,
                    "series_notes",
                    f"FRED Series Notes — {series_id}",
                    notes,
                    f"https://fred.stlouisfed.org/series/{series_id}",
                    fetched_at,
                )
            )

        release_name: str = (metadata.get("release_name") or "").strip()
        release_notes: str = (metadata.get("release_notes") or "").strip()
        release_link: str | None = metadata.get("release_link") or None
        release_content: str = release_notes or release_name
        if release_content:
            rows.append(
                (
                    series_id,
                    "release_info",
                    f"Release — {release_name}" if release_name else f"Release — {series_id}",
                    release_content,
                    release_link,
                    fetched_at,
                )
            )

        category_path: str = (metadata.get("category_path") or "").strip()
        if category_path:
            rows.append(
                (
                    series_id,
                    "category_path",
                    f"Category — {series_id}",
                    category_path,
                    None,
                    fetched_at,
                )
            )

        for row in rows:
            conn.execute(
                """
                INSERT OR REPLACE INTO reference_docs (
                    series_id, doc_type, title, content, source_url, fetched_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                row,
            )
            inserted += 1

    conn.commit()
    logger.info("Loaded %d reference_docs rows", inserted)
    return inserted


_REQUIRED_SCHOLARLY_FIELDS: tuple[str, ...] = (
    "id",
    "series_id",
    "title",
    "content",
    "source_url",
)
"""JSON keys that every scholarly fixture file must provide."""


def _load_scholarly_docs(conn: sqlite3.Connection, mode: str) -> int:
    """Load curated scholarly reference_docs rows from JSON fixtures.

    Iterates ``*.json`` files in ``SCHOLARLY_DIR`` in sorted order, validates
    the schema, and upserts each into ``reference_docs`` with
    ``doc_type = f"scholarly:{id}"``. The unique ``(series_id, doc_type)``
    constraint on ``reference_docs`` enforces at most one row per slug per
    series; ``INSERT OR REPLACE`` lets repeat runs refresh content without
    duplicating rows.

    Invalid fixtures are skipped with a warning. Empty content, malformed
    JSON, missing required fields, an unknown ``series_id``, or a duplicate
    slug within the directory all log and skip rather than raise. A missing
    directory logs an info-level message and returns ``0``.

    Args:
        conn: Open SQLite connection.
        mode: Either ``'seed'`` or ``'full'``. Reserved for future filtering;
            scholarly rows are mode-agnostic in Phase 12.

    Returns:
        Count of rows inserted or replaced.
    """
    if not SCHOLARLY_DIR.exists():
        logger.info(
            "No scholarly fixtures directory at %s - skipping",
            SCHOLARLY_DIR,
        )
        return 0

    known_series: set[str] = {
        row[0] for row in conn.execute("SELECT series_id FROM series_metadata")
    }
    seen_slugs: set[str] = set()
    loaded: int = 0
    _ = mode  # reserved for future filtering

    for path in sorted(SCHOLARLY_DIR.glob("*.json")):
        try:
            with path.open() as f:
                fixture: dict = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(
                "Skipping scholarly fixture %s - unreadable JSON: %s",
                path.name,
                exc,
            )
            continue

        missing: list[str] = [
            key for key in _REQUIRED_SCHOLARLY_FIELDS if not fixture.get(key)
        ]
        if missing:
            logger.warning(
                "Skipping scholarly fixture %s - missing fields: %s",
                path.name,
                ", ".join(missing),
            )
            continue

        slug: str = str(fixture["id"]).strip()
        series_id: str = str(fixture["series_id"]).strip()
        title: str = str(fixture["title"]).strip()
        content: str = str(fixture["content"]).strip()
        source_url: str = str(fixture["source_url"]).strip()

        if not content:
            logger.warning(
                "Skipping scholarly fixture %s - empty content after strip",
                path.name,
            )
            continue

        if slug in seen_slugs:
            logger.warning(
                "Skipping scholarly fixture %s - duplicate slug '%s'",
                path.name,
                slug,
            )
            continue

        if series_id not in known_series:
            logger.warning(
                "Skipping scholarly fixture %s - unknown series_id '%s'",
                path.name,
                series_id,
            )
            continue

        seen_slugs.add(slug)
        doc_type: str = f"scholarly:{slug}"
        fetched_at: str = date.today().isoformat()

        conn.execute(
            """
            INSERT OR REPLACE INTO reference_docs (
                series_id, doc_type, title, content, source_url, fetched_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (series_id, doc_type, title, content, source_url, fetched_at),
        )
        loaded += 1
        logger.info(
            "Loaded scholarly fixture %s (series_id=%s)",
            slug,
            series_id,
        )

    conn.commit()
    logger.info("Loaded %d scholarly reference_docs rows", loaded)
    return loaded


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
    for table in ("series_metadata", "observations", "ai_insights", "reference_docs"):
        row: tuple = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()  # noqa: S608
        print(f"  {table:25s}  {row[0]:>8,} rows")

    # reference_docs breakdown: FRED triple vs scholarly fixtures
    scholarly_count: int = conn.execute(
        "SELECT COUNT(*) FROM reference_docs WHERE doc_type LIKE 'scholarly:%'"
    ).fetchone()[0]
    fred_count: int = conn.execute(
        "SELECT COUNT(*) FROM reference_docs WHERE doc_type NOT LIKE 'scholarly:%'"
    ).fetchone()[0]
    print(f"    reference_docs (FRED):   {fred_count:>6,}")
    print(f"    reference_docs (schol.): {scholarly_count:>6,}")

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
        _ensure_citations_column(conn)
        stats: dict[str, dict[str, int]] = _load_series(conn, series_ids, cutoff_date)
        _load_reference_docs(conn, mode)
        _load_scholarly_docs(conn, mode)
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
