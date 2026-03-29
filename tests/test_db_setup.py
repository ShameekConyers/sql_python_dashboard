"""Tests for src/db_setup.py.

All tests use in-memory or tmp_path SQLite databases. No real data files required.
"""

import argparse
import json
import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest

import db_setup


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SCHEMA_SQL: str = (db_setup.SQL_DIR / "01_schema.sql").read_text()


@pytest.fixture()
def mem_conn() -> sqlite3.Connection:
    """Return an in-memory SQLite connection with the schema already created.

    Returns:
        Open connection with all three tables ready.
    """
    conn: sqlite3.Connection = sqlite3.connect(":memory:")
    conn.executescript(SCHEMA_SQL)
    return conn


@pytest.fixture()
def sample_json_data() -> dict:
    """Return a minimal FRED JSON payload for testing.

    Returns:
        Dict matching the structure written by data_pull.py.
    """
    return {
        "series_id": "UNRATE",
        "name": "Unemployment Rate",
        "category": "labor_market",
        "frequency": "monthly",
        "units": "Percent",
        "seasonal_adjustment": "Seasonally Adjusted",
        "last_updated": "2026-01-15 08:00:00-06",
        "observation_count": 4,
        "observations": [
            {"date": "2015-01-01", "value": 5.7},
            {"date": "2020-06-01", "value": 11.1},
            {"date": "2023-01-01", "value": 3.4},
            {"date": "2024-06-01", "value": None},
        ],
    }


# ---------------------------------------------------------------------------
# _seed_cutoff_date
# ---------------------------------------------------------------------------


class TestSeedCutoffDate:
    """Tests for _seed_cutoff_date."""

    def test_returns_iso_string(self) -> None:
        """Result is an ISO-formatted date string."""
        result: str = db_setup._seed_cutoff_date()
        assert len(result) == 10
        assert result[4] == "-" and result[7] == "-"

    def test_cutoff_is_seed_years_ago(self) -> None:
        """Cutoff year is SEED_YEARS less than the current year."""
        from datetime import date

        result: str = db_setup._seed_cutoff_date()
        cutoff_year: int = int(result[:4])
        assert cutoff_year == date.today().year - db_setup.SEED_YEARS


# ---------------------------------------------------------------------------
# _db_path
# ---------------------------------------------------------------------------


class TestDbPath:
    """Tests for _db_path."""

    def test_seed_mode_returns_seed_db(self) -> None:
        """Seed mode produces a path ending in seed.db."""
        path: Path = db_setup._db_path("seed")
        assert path.name == "seed.db"

    def test_full_mode_returns_full_db(self) -> None:
        """Full mode produces a path ending in full.db."""
        path: Path = db_setup._db_path("full")
        assert path.name == "full.db"

    def test_paths_are_in_data_dir(self) -> None:
        """Both modes point into the DATA_DIR."""
        assert db_setup._db_path("seed").parent == db_setup.DATA_DIR
        assert db_setup._db_path("full").parent == db_setup.DATA_DIR


# ---------------------------------------------------------------------------
# _create_schema
# ---------------------------------------------------------------------------


class TestCreateSchema:
    """Tests for _create_schema."""

    def test_creates_three_tables(self) -> None:
        """Schema creates series_metadata, observations, and ai_insights."""
        conn: sqlite3.Connection = sqlite3.connect(":memory:")
        # Patch SQL_DIR to use the real schema file
        db_setup._create_schema(conn)

        tables: list[str] = [
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            ).fetchall()
        ]
        assert "series_metadata" in tables
        assert "observations" in tables
        assert "ai_insights" in tables
        conn.close()

    def test_schema_is_idempotent(self) -> None:
        """Running _create_schema twice does not raise."""
        conn: sqlite3.Connection = sqlite3.connect(":memory:")
        db_setup._create_schema(conn)
        db_setup._create_schema(conn)  # should not fail
        conn.close()


# ---------------------------------------------------------------------------
# _load_json
# ---------------------------------------------------------------------------


class TestLoadJson:
    """Tests for _load_json."""

    def test_returns_parsed_dict(self, tmp_path: Path) -> None:
        """Reads and parses a cached JSON file."""
        payload: dict = {"series_id": "UNRATE", "observations": []}
        (tmp_path / "UNRATE.json").write_text(json.dumps(payload))

        with patch.object(db_setup, "RAW_DIR", tmp_path):
            result: dict | None = db_setup._load_json("UNRATE")

        assert result is not None
        assert result["series_id"] == "UNRATE"

    def test_returns_none_when_missing(self, tmp_path: Path) -> None:
        """Returns None when the JSON file does not exist."""
        with patch.object(db_setup, "RAW_DIR", tmp_path):
            result: dict | None = db_setup._load_json("NOTREAL")

        assert result is None


# ---------------------------------------------------------------------------
# _insert_metadata
# ---------------------------------------------------------------------------


class TestInsertMetadata:
    """Tests for _insert_metadata."""

    def test_inserts_row(self, mem_conn: sqlite3.Connection, sample_json_data: dict) -> None:
        """Inserts a metadata row into series_metadata."""
        db_setup._insert_metadata(mem_conn, sample_json_data)
        mem_conn.commit()

        row: tuple = mem_conn.execute(
            "SELECT series_id, name, category, frequency FROM series_metadata"
        ).fetchone()
        assert row == ("UNRATE", "Unemployment Rate", "labor_market", "monthly")

    def test_upsert_updates_on_conflict(
        self, mem_conn: sqlite3.Connection, sample_json_data: dict
    ) -> None:
        """Second insert with same series_id updates instead of failing."""
        db_setup._insert_metadata(mem_conn, sample_json_data)
        sample_json_data["name"] = "Updated Name"
        db_setup._insert_metadata(mem_conn, sample_json_data)
        mem_conn.commit()

        row: tuple = mem_conn.execute(
            "SELECT name FROM series_metadata WHERE series_id = 'UNRATE'"
        ).fetchone()
        assert row[0] == "Updated Name"

    def test_count_stays_one_after_upsert(
        self, mem_conn: sqlite3.Connection, sample_json_data: dict
    ) -> None:
        """Upserting the same series does not create a second row."""
        db_setup._insert_metadata(mem_conn, sample_json_data)
        db_setup._insert_metadata(mem_conn, sample_json_data)
        mem_conn.commit()

        count: int = mem_conn.execute("SELECT COUNT(*) FROM series_metadata").fetchone()[0]
        assert count == 1


# ---------------------------------------------------------------------------
# _insert_observations
# ---------------------------------------------------------------------------


class TestInsertObservations:
    """Tests for _insert_observations."""

    def test_inserts_valid_observations(
        self, mem_conn: sqlite3.Connection, sample_json_data: dict
    ) -> None:
        """Inserts rows with non-null numeric values."""
        db_setup._insert_metadata(mem_conn, sample_json_data)
        stats: dict[str, int] = db_setup._insert_observations(
            mem_conn, "UNRATE", sample_json_data["observations"]
        )

        assert stats["inserted"] == 3  # 4 obs, 1 null
        assert stats["skipped_null"] == 1

    def test_skips_null_values(
        self, mem_conn: sqlite3.Connection, sample_json_data: dict
    ) -> None:
        """Observations with value=None are counted as skipped_null."""
        db_setup._insert_metadata(mem_conn, sample_json_data)
        obs: list[dict] = [{"date": "2024-01-01", "value": None}]
        stats: dict[str, int] = db_setup._insert_observations(mem_conn, "UNRATE", obs)

        assert stats["skipped_null"] == 1
        assert stats["inserted"] == 0

    def test_skips_invalid_types(
        self, mem_conn: sqlite3.Connection, sample_json_data: dict
    ) -> None:
        """Non-numeric values are skipped."""
        db_setup._insert_metadata(mem_conn, sample_json_data)
        obs: list[dict] = [{"date": "2024-01-01", "value": "bad"}]
        stats: dict[str, int] = db_setup._insert_observations(mem_conn, "UNRATE", obs)

        assert stats["skipped_null"] == 1
        assert stats["inserted"] == 0

    def test_cutoff_date_filters_old_observations(
        self, mem_conn: sqlite3.Connection, sample_json_data: dict
    ) -> None:
        """Observations before the cutoff date are counted as skipped_date."""
        db_setup._insert_metadata(mem_conn, sample_json_data)
        stats: dict[str, int] = db_setup._insert_observations(
            mem_conn,
            "UNRATE",
            sample_json_data["observations"],
            cutoff_date="2020-01-01",
        )

        # "2015-01-01" is before cutoff → skipped_date
        assert stats["skipped_date"] == 1
        # "2020-06-01" and "2023-01-01" pass, "2024-06-01" is null
        assert stats["inserted"] == 2
        assert stats["skipped_null"] == 1

    def test_no_cutoff_inserts_all_valid(
        self, mem_conn: sqlite3.Connection, sample_json_data: dict
    ) -> None:
        """Without a cutoff, all non-null observations are inserted."""
        db_setup._insert_metadata(mem_conn, sample_json_data)
        stats: dict[str, int] = db_setup._insert_observations(
            mem_conn, "UNRATE", sample_json_data["observations"]
        )

        assert stats["skipped_date"] == 0
        assert stats["inserted"] == 3

    def test_idempotent_insert_or_ignore(
        self, mem_conn: sqlite3.Connection, sample_json_data: dict
    ) -> None:
        """Reinserting the same observations does not raise or duplicate rows."""
        db_setup._insert_metadata(mem_conn, sample_json_data)
        db_setup._insert_observations(mem_conn, "UNRATE", sample_json_data["observations"])
        # Second insert should silently ignore duplicates
        db_setup._insert_observations(mem_conn, "UNRATE", sample_json_data["observations"])
        mem_conn.commit()

        count: int = mem_conn.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
        assert count == 3

    def test_stores_value_covid_adjusted_equal_to_value(
        self, mem_conn: sqlite3.Connection, sample_json_data: dict
    ) -> None:
        """On initial load, value_covid_adjusted equals value."""
        db_setup._insert_metadata(mem_conn, sample_json_data)
        db_setup._insert_observations(mem_conn, "UNRATE", sample_json_data["observations"])
        mem_conn.commit()

        rows: list[tuple] = mem_conn.execute(
            "SELECT value, value_covid_adjusted FROM observations"
        ).fetchall()
        for value, adjusted in rows:
            assert value == adjusted


# ---------------------------------------------------------------------------
# _load_series
# ---------------------------------------------------------------------------


class TestLoadSeries:
    """Tests for _load_series."""

    def test_loads_multiple_series(self, mem_conn: sqlite3.Connection, tmp_path: Path) -> None:
        """Loads two series and returns stats for both."""
        for sid, val in [("UNRATE", 3.5), ("GDPC1", 20000.0)]:
            payload: dict = {
                "series_id": sid,
                "name": sid,
                "category": "test",
                "frequency": "monthly",
                "observations": [{"date": "2024-01-01", "value": val}],
            }
            (tmp_path / f"{sid}.json").write_text(json.dumps(payload))

        with patch.object(db_setup, "RAW_DIR", tmp_path):
            stats: dict = db_setup._load_series(mem_conn, ["UNRATE", "GDPC1"])

        assert "UNRATE" in stats
        assert "GDPC1" in stats
        assert stats["UNRATE"]["inserted"] == 1
        assert stats["GDPC1"]["inserted"] == 1

    def test_skips_missing_json(self, mem_conn: sqlite3.Connection, tmp_path: Path) -> None:
        """Series without cached JSON are silently skipped."""
        with patch.object(db_setup, "RAW_DIR", tmp_path):
            stats: dict = db_setup._load_series(mem_conn, ["NOTREAL"])

        assert "NOTREAL" not in stats


# ---------------------------------------------------------------------------
# build_database
# ---------------------------------------------------------------------------


class TestBuildDatabase:
    """Tests for build_database."""

    def test_creates_db_file(self, tmp_path: Path) -> None:
        """build_database creates a .db file on disk."""
        # Write a minimal JSON cache for one series
        raw_dir: Path = tmp_path / "raw"
        raw_dir.mkdir()
        payload: dict = {
            "series_id": "UNRATE",
            "name": "Unemployment Rate",
            "category": "labor_market",
            "frequency": "monthly",
            "observations": [{"date": "2024-01-01", "value": 3.5}],
        }
        (raw_dir / "UNRATE.json").write_text(json.dumps(payload))

        sql_dir: Path = db_setup.SQL_DIR  # use the real schema

        with (
            patch.object(db_setup, "DATA_DIR", tmp_path),
            patch.object(db_setup, "RAW_DIR", raw_dir),
        ):
            db_path: Path = db_setup.build_database("seed")

        assert db_path.exists()
        assert db_path.name == "seed.db"

    def test_full_mode_creates_full_db(self, tmp_path: Path) -> None:
        """Full mode writes to full.db."""
        raw_dir: Path = tmp_path / "raw"
        raw_dir.mkdir()
        payload: dict = {
            "series_id": "UNRATE",
            "name": "Unemployment Rate",
            "category": "labor_market",
            "frequency": "monthly",
            "observations": [{"date": "2024-01-01", "value": 3.5}],
        }
        (raw_dir / "UNRATE.json").write_text(json.dumps(payload))

        with (
            patch.object(db_setup, "DATA_DIR", tmp_path),
            patch.object(db_setup, "RAW_DIR", raw_dir),
        ):
            db_path: Path = db_setup.build_database("full")

        assert db_path.name == "full.db"

    def test_removes_existing_db(self, tmp_path: Path) -> None:
        """An existing .db file is deleted before rebuilding."""
        raw_dir: Path = tmp_path / "raw"
        raw_dir.mkdir()
        payload: dict = {
            "series_id": "UNRATE",
            "name": "Unemployment Rate",
            "category": "labor_market",
            "frequency": "monthly",
            "observations": [{"date": "2024-01-01", "value": 3.5}],
        }
        (raw_dir / "UNRATE.json").write_text(json.dumps(payload))

        old_db: Path = tmp_path / "seed.db"
        old_db.write_text("stale")

        with (
            patch.object(db_setup, "DATA_DIR", tmp_path),
            patch.object(db_setup, "RAW_DIR", raw_dir),
        ):
            db_setup.build_database("seed")

        # File should exist and not contain the stale placeholder
        assert old_db.exists()
        assert old_db.read_bytes() != b"stale"


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


class TestMain:
    """Tests for the main() entry point."""

    @patch("db_setup.build_database")
    @patch("db_setup._parse_args")
    def test_default_is_seed_mode(
        self, mock_args: patch, mock_build: patch
    ) -> None:
        """Without --full, main calls build_database with 'seed'."""
        mock_args.return_value = argparse.Namespace(full=False)
        db_setup.main()
        mock_build.assert_called_once_with("seed")

    @patch("db_setup.build_database")
    @patch("db_setup._parse_args")
    def test_full_flag(self, mock_args: patch, mock_build: patch) -> None:
        """--full flag passes 'full' to build_database."""
        mock_args.return_value = argparse.Namespace(full=True)
        db_setup.main()
        mock_build.assert_called_once_with("full")
