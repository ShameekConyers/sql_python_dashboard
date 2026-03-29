"""Tests for src/export_csv.py.

Uses in-memory SQLite databases and tmp_path for CSV output.
"""

import argparse
import csv
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import export_csv

# ---------------------------------------------------------------------------
# Schema helper
# ---------------------------------------------------------------------------

SCHEMA_SQL: str = (export_csv.PROJECT_ROOT / "sql" / "01_schema.sql").read_text()


# ---------------------------------------------------------------------------
# _db_path
# ---------------------------------------------------------------------------


class TestDbPath:
    """Tests for _db_path."""

    def test_seed_returns_seed_db(self) -> None:
        """Seed mode points to seed.db."""
        assert export_csv._db_path("seed").name == "seed.db"

    def test_full_returns_full_db(self) -> None:
        """Full mode points to full.db."""
        assert export_csv._db_path("full").name == "full.db"


# ---------------------------------------------------------------------------
# _parse_labeled_queries
# ---------------------------------------------------------------------------


class TestParseLabeledQueries:
    """Tests for _parse_labeled_queries."""

    def test_parses_single_query(self, tmp_path: Path) -> None:
        """Extracts one labeled query from a SQL file."""
        sql_file: Path = tmp_path / "queries.sql"
        sql_file.write_text("-- Q1: Yield curve\nSELECT 1;\n")

        result: list[tuple[str, str]] = export_csv._parse_labeled_queries(sql_file)

        assert len(result) == 1
        assert result[0][0] == "Q1"
        assert "SELECT 1" in result[0][1]

    def test_parses_multiple_queries(self, tmp_path: Path) -> None:
        """Splits multiple labeled queries correctly."""
        sql_file: Path = tmp_path / "queries.sql"
        sql_file.write_text(
            "-- Q1: First\nSELECT 1;\n\n-- Q2: Second\nSELECT 2;\n"
        )

        result: list[tuple[str, str]] = export_csv._parse_labeled_queries(sql_file)

        assert len(result) == 2
        assert result[0][0] == "Q1"
        assert result[1][0] == "Q2"

    def test_ignores_non_query_preamble(self, tmp_path: Path) -> None:
        """Text before the first Q-header is ignored."""
        sql_file: Path = tmp_path / "queries.sql"
        sql_file.write_text(
            "-- This is a preamble comment\n-- Another line\n\n"
            "-- Q1: Real query\nSELECT 42;\n"
        )

        result: list[tuple[str, str]] = export_csv._parse_labeled_queries(sql_file)

        assert len(result) == 1
        assert result[0][0] == "Q1"

    def test_empty_file_returns_empty_list(self, tmp_path: Path) -> None:
        """An empty SQL file produces no queries."""
        sql_file: Path = tmp_path / "queries.sql"
        sql_file.write_text("")

        result: list[tuple[str, str]] = export_csv._parse_labeled_queries(sql_file)
        assert result == []

    def test_preserves_full_sql_text(self, tmp_path: Path) -> None:
        """Each query's SQL text includes the header comment and body."""
        sql_file: Path = tmp_path / "queries.sql"
        sql_file.write_text(
            "-- Q1: Test\nWITH cte AS (SELECT 1)\nSELECT * FROM cte;\n"
        )

        result: list[tuple[str, str]] = export_csv._parse_labeled_queries(sql_file)
        sql_text: str = result[0][1]
        assert "WITH cte AS" in sql_text
        assert "SELECT * FROM cte" in sql_text


# ---------------------------------------------------------------------------
# _export_query
# ---------------------------------------------------------------------------


class TestExportQuery:
    """Tests for _export_query."""

    def test_writes_csv_with_header_and_rows(self, tmp_path: Path) -> None:
        """Produces a CSV file with column headers and data rows."""
        conn: sqlite3.Connection = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE t (a INTEGER, b TEXT)")
        conn.execute("INSERT INTO t VALUES (1, 'x')")
        conn.execute("INSERT INTO t VALUES (2, 'y')")

        output: Path = tmp_path / "test.csv"
        row_count: int = export_csv._export_query(
            conn, "Q1", "SELECT a, b FROM t", output
        )

        assert row_count == 2
        assert output.exists()

        with output.open() as f:
            reader: csv.reader = csv.reader(f)
            rows: list[list[str]] = list(reader)

        assert rows[0] == ["a", "b"]
        assert rows[1] == ["1", "x"]
        assert rows[2] == ["2", "y"]
        conn.close()

    def test_returns_zero_for_empty_result(self, tmp_path: Path) -> None:
        """Returns 0 when the query produces no rows."""
        conn: sqlite3.Connection = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE t (a INTEGER)")

        output: Path = tmp_path / "empty.csv"
        row_count: int = export_csv._export_query(
            conn, "Q1", "SELECT a FROM t", output
        )

        assert row_count == 0
        # File should still exist with just the header
        assert output.exists()

        with output.open() as f:
            reader: csv.reader = csv.reader(f)
            rows: list[list[str]] = list(reader)
        assert len(rows) == 1  # header only
        conn.close()

    def test_column_names_from_query(self, tmp_path: Path) -> None:
        """CSV header matches the column names from the query."""
        conn: sqlite3.Connection = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE t (foo INTEGER, bar_baz TEXT)")
        conn.execute("INSERT INTO t VALUES (1, 'a')")

        output: Path = tmp_path / "cols.csv"
        export_csv._export_query(conn, "Q1", "SELECT foo, bar_baz FROM t", output)

        with output.open() as f:
            header: list[str] = next(csv.reader(f))
        assert header == ["foo", "bar_baz"]
        conn.close()


# ---------------------------------------------------------------------------
# export_all
# ---------------------------------------------------------------------------


class TestExportAll:
    """Tests for export_all."""

    def test_exports_queries_to_csv_files(self, tmp_path: Path) -> None:
        """Creates CSV files for each mapped query label."""
        # Create a DB with a simple table
        db_file: Path = tmp_path / "seed.db"
        conn: sqlite3.Connection = sqlite3.connect(db_file)
        conn.execute("CREATE TABLE t (val INTEGER)")
        conn.execute("INSERT INTO t VALUES (42)")
        conn.commit()
        conn.close()

        # Create a SQL file with two labeled queries
        sql_dir: Path = tmp_path / "sql"
        sql_dir.mkdir()
        (sql_dir / "03_analysis_queries.sql").write_text(
            "-- Q1: Test\nSELECT val FROM t;\n\n"
            "-- Q2: Test2\nSELECT val * 2 AS doubled FROM t;\n"
        )

        export_dir: Path = tmp_path / "exports"

        with (
            patch.object(export_csv, "DATA_DIR", tmp_path),
            patch.object(export_csv, "SQL_DIR", sql_dir),
            patch.object(export_csv, "EXPORT_DIR", export_dir),
        ):
            export_csv.export_all("seed")

        assert (export_dir / "yield_curve_vs_unemployment.csv").exists()
        assert (export_dir / "info_vs_trades_divergence.csv").exists()

    def test_skips_unmapped_labels(self, tmp_path: Path) -> None:
        """Queries with labels not in QUERY_LABELS are skipped."""
        db_file: Path = tmp_path / "seed.db"
        conn: sqlite3.Connection = sqlite3.connect(db_file)
        conn.execute("CREATE TABLE t (val INTEGER)")
        conn.execute("INSERT INTO t VALUES (1)")
        conn.commit()
        conn.close()

        sql_dir: Path = tmp_path / "sql"
        sql_dir.mkdir()
        (sql_dir / "03_analysis_queries.sql").write_text(
            "-- Q99: Unknown\nSELECT val FROM t;\n"
        )

        export_dir: Path = tmp_path / "exports"

        with (
            patch.object(export_csv, "DATA_DIR", tmp_path),
            patch.object(export_csv, "SQL_DIR", sql_dir),
            patch.object(export_csv, "EXPORT_DIR", export_dir),
        ):
            export_csv.export_all("seed")

        # No files should be created for Q99
        if export_dir.exists():
            assert len(list(export_dir.iterdir())) == 0

    def test_returns_early_when_db_missing(self, tmp_path: Path) -> None:
        """Does not raise when the database file does not exist."""
        with patch.object(export_csv, "DATA_DIR", tmp_path):
            # Should not raise
            export_csv.export_all("seed")


# ---------------------------------------------------------------------------
# QUERY_LABELS
# ---------------------------------------------------------------------------


class TestQueryLabels:
    """Validate the QUERY_LABELS mapping."""

    def test_has_eight_entries(self) -> None:
        """QUERY_LABELS maps all 8 analysis queries."""
        assert len(export_csv.QUERY_LABELS) == 8

    def test_keys_are_q1_through_q8(self) -> None:
        """Keys are Q1 through Q8."""
        expected: set[str] = {f"Q{i}" for i in range(1, 9)}
        assert set(export_csv.QUERY_LABELS.keys()) == expected

    def test_values_are_nonempty_strings(self) -> None:
        """Every label value is a non-empty string."""
        for label, filename in export_csv.QUERY_LABELS.items():
            assert isinstance(filename, str) and len(filename) > 0, (
                f"{label} has invalid filename: {filename!r}"
            )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


class TestMain:
    """Tests for the main() entry point."""

    @patch("export_csv.export_all")
    @patch("export_csv._parse_args")
    def test_default_is_seed(self, mock_args: MagicMock, mock_export: MagicMock) -> None:
        """Without --full, main calls export_all with 'seed'."""
        mock_args.return_value = argparse.Namespace(full=False)
        export_csv.main()
        mock_export.assert_called_once_with("seed")

    @patch("export_csv.export_all")
    @patch("export_csv._parse_args")
    def test_full_flag(self, mock_args: MagicMock, mock_export: MagicMock) -> None:
        """--full flag passes 'full' to export_all."""
        mock_args.return_value = argparse.Namespace(full=True)
        export_csv.main()
        mock_export.assert_called_once_with("full")
