"""Export analysis query results to CSV for notebook and dashboard ingestion.

Reads each labeled query from 03_analysis_queries.sql, executes it against the
specified database, and writes the result to data/exports/<query_name>.csv.

Usage:
    python src/export_csv.py           # uses seed.db
    python src/export_csv.py --full    # uses full.db
"""

import argparse
import csv
import logging
import re
import sqlite3
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
SQL_DIR: Path = PROJECT_ROOT / "sql"
DATA_DIR: Path = PROJECT_ROOT / "data"
EXPORT_DIR: Path = DATA_DIR / "exports"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger: logging.Logger = logging.getLogger(__name__)

# Maps query header labels to output filenames. The keys are matched against
# the "-- Q<n>:" comment lines in 03_analysis_queries.sql.
QUERY_LABELS: dict[str, str] = {
    "Q1": "yield_curve_vs_unemployment",
    "Q2": "info_vs_trades_divergence",
    "Q3": "gdp_growth_recession",
    "Q4": "rolling_12m_employment_growth",
    "Q5": "covid_recovery_comparison",
    "Q6": "u6_vs_u3_gap",
    "Q7": "power_vs_info_employment",
    "Q8": "cpi_inflation",
}


def _db_path(mode: str) -> Path:
    """Return the database file path for the given mode.

    Args:
        mode: Either 'seed' or 'full'.

    Returns:
        Path to the SQLite database file.
    """
    filename: str = "seed.db" if mode == "seed" else "full.db"
    return DATA_DIR / filename


def _parse_labeled_queries(sql_path: Path) -> list[tuple[str, str]]:
    """Split a SQL file into labeled queries based on '-- Q<n>:' headers.

    Each query runs from its header comment through the next header or end
    of file. Only blocks that start with a recognized Q-label are returned.

    Args:
        sql_path: Path to the SQL file containing labeled queries.

    Returns:
        List of (label, sql_text) tuples.
    """
    text: str = sql_path.read_text()
    # Split on the Q-header pattern, keeping the delimiter
    parts: list[str] = re.split(r"(?=^-- Q\d+:)", text, flags=re.MULTILINE)

    queries: list[tuple[str, str]] = []
    for part in parts:
        match = re.match(r"^-- (Q\d+):", part)
        if match:
            label: str = match.group(1)
            queries.append((label, part.strip()))

    return queries


def _export_query(
    conn: sqlite3.Connection, label: str, sql: str, output_path: Path
) -> int:
    """Execute a query and write results to CSV.

    Args:
        conn: Open SQLite connection.
        label: Query label (e.g., 'Q1').
        sql: Full SQL text to execute.
        output_path: Destination CSV file path.

    Returns:
        Number of rows written.
    """
    cursor: sqlite3.Cursor = conn.execute(sql)
    columns: list[str] = [desc[0] for desc in cursor.description]
    rows: list[tuple] = cursor.fetchall()

    with output_path.open("w", newline="") as f:
        writer: csv.writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows(rows)

    return len(rows)


def export_all(mode: str = "seed") -> None:
    """Export all labeled analysis queries to CSV.

    Args:
        mode: 'seed' or 'full', determines which database to query.
    """
    db: Path = _db_path(mode)
    if not db.exists():
        logger.error("Database not found: %s — run db_setup.py first", db)
        return

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    sql_path: Path = SQL_DIR / "03_analysis_queries.sql"
    queries: list[tuple[str, str]] = _parse_labeled_queries(sql_path)
    logger.info("Found %d labeled queries in %s", len(queries), sql_path.name)

    conn: sqlite3.Connection = sqlite3.connect(db)
    try:
        for label, sql in queries:
            filename: str | None = QUERY_LABELS.get(label)
            if filename is None:
                logger.warning("No export mapping for %s — skipping", label)
                continue

            output_path: Path = EXPORT_DIR / f"{filename}.csv"
            row_count: int = _export_query(conn, label, sql, output_path)
            logger.info("  %s → %s (%d rows)", label, output_path.name, row_count)
    finally:
        conn.close()

    logger.info("Exports written to %s", EXPORT_DIR)


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Export analysis query results to CSV.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Export from full.db instead of seed.db.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for the CSV export script."""
    args: argparse.Namespace = _parse_args()
    mode: str = "full" if args.full else "seed"
    export_all(mode)


if __name__ == "__main__":
    main()
