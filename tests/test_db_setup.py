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
REFERENCE_SCHEMA_SQL: str = (db_setup.SQL_DIR / "06_reference_schema.sql").read_text()
HACKERNEWS_SCHEMA_SQL: str = (
    db_setup.SQL_DIR / "07_hackernews_schema.sql"
).read_text()


@pytest.fixture()
def mem_conn() -> sqlite3.Connection:
    """Return an in-memory SQLite connection with the schema already created.

    Returns:
        Open connection with all tables (core + reference_docs + HN) ready.
    """
    conn: sqlite3.Connection = sqlite3.connect(":memory:")
    conn.executescript(SCHEMA_SQL)
    conn.executescript(REFERENCE_SCHEMA_SQL)
    conn.executescript(HACKERNEWS_SCHEMA_SQL)
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

    def test_creates_all_tables(self) -> None:
        """Schema creates core tables, reference_docs, and HN tables."""
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
        assert "reference_docs" in tables
        assert "hn_stories" in tables
        assert "hn_sentiment_monthly" in tables
        conn.close()

    def test_schema_is_idempotent(self) -> None:
        """Running _create_schema twice does not raise."""
        conn: sqlite3.Connection = sqlite3.connect(":memory:")
        db_setup._create_schema(conn)
        db_setup._create_schema(conn)  # should not fail
        conn.close()

    def test_fresh_build_includes_citations_column(self) -> None:
        """Fresh schema build has citations_json on ai_insights by default."""
        conn: sqlite3.Connection = sqlite3.connect(":memory:")
        db_setup._create_schema(conn)

        rows: list[tuple] = conn.execute(
            "PRAGMA table_info('ai_insights')"
        ).fetchall()
        columns: set[str] = {row[1] for row in rows}
        assert "citations_json" in columns
        conn.close()


# ---------------------------------------------------------------------------
# _ensure_citations_column
# ---------------------------------------------------------------------------


class TestEnsureCitationsColumn:
    """Tests for _ensure_citations_column."""

    def test_adds_column_when_missing(self) -> None:
        """Adds citations_json to a pre-Phase-11 ai_insights table."""
        conn: sqlite3.Connection = sqlite3.connect(":memory:")
        # Create a pre-Phase-11 ai_insights table without citations_json
        conn.execute(
            """
            CREATE TABLE ai_insights (
                id                  INTEGER PRIMARY KEY,
                metric_key          TEXT NOT NULL,
                slice_key           TEXT NOT NULL,
                insight_type        TEXT NOT NULL,
                narrative           TEXT NOT NULL,
                claims_json         TEXT NOT NULL,
                verification_json   TEXT NOT NULL,
                all_verified        BOOLEAN NOT NULL,
                model_used          TEXT NOT NULL,
                generated_at        TIMESTAMP NOT NULL
            )
            """
        )

        db_setup._ensure_citations_column(conn)

        rows: list[tuple] = conn.execute(
            "PRAGMA table_info('ai_insights')"
        ).fetchall()
        columns: set[str] = {row[1] for row in rows}
        assert "citations_json" in columns
        conn.close()

    def test_noop_when_column_present(self) -> None:
        """No error when citations_json already exists."""
        conn: sqlite3.Connection = sqlite3.connect(":memory:")
        db_setup._create_schema(conn)
        # Should no-op
        db_setup._ensure_citations_column(conn)
        db_setup._ensure_citations_column(conn)  # second call also no-ops
        conn.close()

    def test_fresh_vs_migrated_schema_parity(self) -> None:
        """Fresh build and migrated legacy DB produce identical ai_insights schema."""
        # Fresh build: schema already includes citations_json
        fresh: sqlite3.Connection = sqlite3.connect(":memory:")
        db_setup._create_schema(fresh)
        db_setup._ensure_citations_column(fresh)  # no-op on fresh
        fresh_info: list[tuple] = fresh.execute(
            "PRAGMA table_info('ai_insights')"
        ).fetchall()
        fresh.close()

        # Migrated build: start with pre-Phase-11 DDL, then migrate
        legacy: sqlite3.Connection = sqlite3.connect(":memory:")
        legacy.executescript(
            """
            CREATE TABLE ai_insights (
                id                  INTEGER PRIMARY KEY,
                metric_key          TEXT NOT NULL,
                slice_key           TEXT NOT NULL,
                insight_type        TEXT NOT NULL,
                narrative           TEXT NOT NULL,
                claims_json         TEXT NOT NULL,
                verification_json   TEXT NOT NULL,
                all_verified        BOOLEAN NOT NULL,
                model_used          TEXT NOT NULL,
                generated_at        TIMESTAMP NOT NULL,
                UNIQUE (metric_key, slice_key, insight_type)
            );
            """
        )
        db_setup._ensure_citations_column(legacy)
        migrated_info: list[tuple] = legacy.execute(
            "PRAGMA table_info('ai_insights')"
        ).fetchall()
        legacy.close()

        # PRAGMA table_info returns (cid, name, type, notnull, dflt_value, pk).
        # Column order matches and each tuple is identical across both paths.
        fresh_names: list[str] = [row[1] for row in fresh_info]
        migrated_names: list[str] = [row[1] for row in migrated_info]
        assert fresh_names == migrated_names

        # Every column's (name, type, notnull, dflt_value, pk) matches
        for fresh_row, migrated_row in zip(fresh_info, migrated_info):
            # Skip cid — ordering confirmed above
            assert fresh_row[1:] == migrated_row[1:]


# ---------------------------------------------------------------------------
# _load_reference_docs
# ---------------------------------------------------------------------------


class TestLoadReferenceDocs:
    """Tests for _load_reference_docs."""

    def test_loads_three_doc_types(
        self, mem_conn: sqlite3.Connection, tmp_path: Path, sample_json_data: dict
    ) -> None:
        """Loads series_notes, release_info, and category_path rows."""
        # Metadata needs the series_metadata row for FK integrity.
        db_setup._insert_metadata(mem_conn, sample_json_data)
        mem_conn.commit()

        metadata: dict = {
            "series_notes": "The unemployment rate measures the labor force.",
            "release_name": "Employment Situation",
            "release_link": "https://www.bls.gov/news.release/empsit.toc.htm",
            "release_notes": "Monthly BLS release.",
            "category_path": "Labor Market > Unemployment",
            "fetched_at": "2026-04-16T12:00:00Z",
        }
        (tmp_path / "UNRATE_metadata.json").write_text(json.dumps(metadata))

        with patch.object(db_setup, "RAW_DIR", tmp_path), patch.object(
            db_setup, "SEED_SERIES", ["UNRATE"]
        ):
            count: int = db_setup._load_reference_docs(mem_conn, "seed")

        assert count == 3
        rows: list[tuple] = mem_conn.execute(
            "SELECT doc_type, source_url FROM reference_docs ORDER BY doc_type"
        ).fetchall()
        doc_types: list[str] = [r[0] for r in rows]
        assert doc_types == ["category_path", "release_info", "series_notes"]
        # category_path has no source_url
        category_row = next(r for r in rows if r[0] == "category_path")
        assert category_row[1] is None

    def test_skips_empty_content(
        self, mem_conn: sqlite3.Connection, tmp_path: Path, sample_json_data: dict
    ) -> None:
        """Skips doc types whose content is empty or whitespace."""
        db_setup._insert_metadata(mem_conn, sample_json_data)
        mem_conn.commit()

        metadata: dict = {
            "series_notes": "",
            "release_name": "",
            "release_link": None,
            "release_notes": "",
            "category_path": "Labor Market > Unemployment",
            "fetched_at": "2026-04-16T12:00:00Z",
        }
        (tmp_path / "UNRATE_metadata.json").write_text(json.dumps(metadata))

        with patch.object(db_setup, "RAW_DIR", tmp_path), patch.object(
            db_setup, "SEED_SERIES", ["UNRATE"]
        ):
            count: int = db_setup._load_reference_docs(mem_conn, "seed")

        assert count == 1  # only category_path
        rows = mem_conn.execute(
            "SELECT doc_type FROM reference_docs"
        ).fetchall()
        assert rows == [("category_path",)]

    def test_upsert_keyed_on_series_and_doctype(
        self, mem_conn: sqlite3.Connection, tmp_path: Path, sample_json_data: dict
    ) -> None:
        """Running twice does not create duplicate rows."""
        db_setup._insert_metadata(mem_conn, sample_json_data)
        mem_conn.commit()

        metadata: dict = {
            "series_notes": "A" * 100,
            "release_name": "Test Release",
            "release_link": "https://example.com",
            "release_notes": "Test notes",
            "category_path": "Test > Category",
            "fetched_at": "2026-04-16T12:00:00Z",
        }
        (tmp_path / "UNRATE_metadata.json").write_text(json.dumps(metadata))

        with patch.object(db_setup, "RAW_DIR", tmp_path), patch.object(
            db_setup, "SEED_SERIES", ["UNRATE"]
        ):
            db_setup._load_reference_docs(mem_conn, "seed")
            db_setup._load_reference_docs(mem_conn, "seed")

        row_count: int = mem_conn.execute(
            "SELECT COUNT(*) FROM reference_docs"
        ).fetchone()[0]
        assert row_count == 3

    def test_missing_metadata_file_skips_series(
        self, mem_conn: sqlite3.Connection, tmp_path: Path
    ) -> None:
        """Series without a metadata JSON file are silently skipped."""
        with patch.object(db_setup, "RAW_DIR", tmp_path), patch.object(
            db_setup, "SEED_SERIES", ["NONEXISTENT"]
        ):
            count: int = db_setup._load_reference_docs(mem_conn, "seed")

        assert count == 0


# ---------------------------------------------------------------------------
# _load_scholarly_docs
# ---------------------------------------------------------------------------


def _scholarly_fixture(**overrides: object) -> dict:
    """Build a minimal scholarly fixture dict with schema defaults.

    Args:
        **overrides: Field values to replace in the default fixture.

    Returns:
        Dict ready to be serialized as one scholarly JSON fixture.
    """
    base: dict = {
        "id": "test_slug",
        "series_id": "UNRATE",
        "title": "Test Title",
        "content": "Test content about unemployment methodology.",
        "source_url": "https://www.bls.gov/test",
        "publisher": "BLS",
        "year": 2025,
    }
    base.update(overrides)
    return base


class TestLoadScholarlyDocs:
    """Tests for _load_scholarly_docs (Phase 12)."""

    def test_happy_path_loads_multiple_fixtures(
        self,
        mem_conn: sqlite3.Connection,
        tmp_path: Path,
        sample_json_data: dict,
    ) -> None:
        """Two valid fixture files produce two scholarly:<slug> rows."""
        db_setup._insert_metadata(mem_conn, sample_json_data)
        mem_conn.commit()

        f1: dict = _scholarly_fixture(id="cea_labor", title="CEA labor")
        f2: dict = _scholarly_fixture(
            id="bls_methodology",
            title="BLS methodology",
            content="BLS unemployment methodology paragraph.",
        )
        (tmp_path / "a.json").write_text(json.dumps(f1))
        (tmp_path / "b.json").write_text(json.dumps(f2))

        with patch.object(db_setup, "SCHOLARLY_DIR", tmp_path):
            count: int = db_setup._load_scholarly_docs(mem_conn, "seed")

        assert count == 2
        rows: list[tuple] = mem_conn.execute(
            "SELECT doc_type, title FROM reference_docs ORDER BY doc_type"
        ).fetchall()
        doc_types: list[str] = [r[0] for r in rows]
        assert doc_types == ["scholarly:bls_methodology", "scholarly:cea_labor"]

    def test_missing_directory_returns_zero(
        self, mem_conn: sqlite3.Connection, tmp_path: Path
    ) -> None:
        """Loader returns 0 without raising if SCHOLARLY_DIR is missing."""
        missing: Path = tmp_path / "does_not_exist"
        with patch.object(db_setup, "SCHOLARLY_DIR", missing):
            count: int = db_setup._load_scholarly_docs(mem_conn, "seed")

        assert count == 0
        row_count: int = mem_conn.execute(
            "SELECT COUNT(*) FROM reference_docs"
        ).fetchone()[0]
        assert row_count == 0

    def test_missing_required_field_skips(
        self,
        mem_conn: sqlite3.Connection,
        tmp_path: Path,
        sample_json_data: dict,
    ) -> None:
        """Fixture without 'content' is skipped; DB untouched."""
        db_setup._insert_metadata(mem_conn, sample_json_data)
        mem_conn.commit()

        bad: dict = _scholarly_fixture()
        bad.pop("content")
        (tmp_path / "bad.json").write_text(json.dumps(bad))

        with patch.object(db_setup, "SCHOLARLY_DIR", tmp_path):
            count: int = db_setup._load_scholarly_docs(mem_conn, "seed")

        assert count == 0
        row_count: int = mem_conn.execute(
            "SELECT COUNT(*) FROM reference_docs"
        ).fetchone()[0]
        assert row_count == 0

    def test_bad_json_skips_without_raising(
        self,
        mem_conn: sqlite3.Connection,
        tmp_path: Path,
        sample_json_data: dict,
    ) -> None:
        """Malformed JSON file is skipped; loader does not raise."""
        db_setup._insert_metadata(mem_conn, sample_json_data)
        mem_conn.commit()

        (tmp_path / "broken.json").write_text("{not valid json")
        (tmp_path / "good.json").write_text(json.dumps(_scholarly_fixture()))

        with patch.object(db_setup, "SCHOLARLY_DIR", tmp_path):
            count: int = db_setup._load_scholarly_docs(mem_conn, "seed")

        # The good fixture still loads; the bad one is skipped.
        assert count == 1

    def test_empty_content_skips(
        self,
        mem_conn: sqlite3.Connection,
        tmp_path: Path,
        sample_json_data: dict,
    ) -> None:
        """Fixture with only whitespace in content is skipped."""
        db_setup._insert_metadata(mem_conn, sample_json_data)
        mem_conn.commit()

        (tmp_path / "x.json").write_text(
            json.dumps(_scholarly_fixture(content="   \n\t"))
        )

        with patch.object(db_setup, "SCHOLARLY_DIR", tmp_path):
            count: int = db_setup._load_scholarly_docs(mem_conn, "seed")

        assert count == 0

    def test_unknown_series_skips(
        self,
        mem_conn: sqlite3.Connection,
        tmp_path: Path,
        sample_json_data: dict,
    ) -> None:
        """Fixture referencing a series not in series_metadata is skipped."""
        db_setup._insert_metadata(mem_conn, sample_json_data)
        mem_conn.commit()

        (tmp_path / "x.json").write_text(
            json.dumps(_scholarly_fixture(series_id="NOTASERIES"))
        )

        with patch.object(db_setup, "SCHOLARLY_DIR", tmp_path):
            count: int = db_setup._load_scholarly_docs(mem_conn, "seed")

        assert count == 0

    def test_duplicate_slug_skips_second(
        self,
        mem_conn: sqlite3.Connection,
        tmp_path: Path,
        sample_json_data: dict,
    ) -> None:
        """First fixture wins; duplicate slug in a later file is skipped."""
        db_setup._insert_metadata(mem_conn, sample_json_data)
        mem_conn.commit()

        (tmp_path / "a.json").write_text(
            json.dumps(_scholarly_fixture(id="dup", title="first"))
        )
        (tmp_path / "b.json").write_text(
            json.dumps(_scholarly_fixture(id="dup", title="second"))
        )

        with patch.object(db_setup, "SCHOLARLY_DIR", tmp_path):
            count: int = db_setup._load_scholarly_docs(mem_conn, "seed")

        assert count == 1
        titles: list[tuple] = mem_conn.execute(
            "SELECT title FROM reference_docs WHERE doc_type = 'scholarly:dup'"
        ).fetchall()
        # Sorted glob is alphabetical, so a.json wins.
        assert titles == [("first",)]

    def test_idempotent_upsert(
        self,
        mem_conn: sqlite3.Connection,
        tmp_path: Path,
        sample_json_data: dict,
    ) -> None:
        """Running the loader twice does not duplicate rows."""
        db_setup._insert_metadata(mem_conn, sample_json_data)
        mem_conn.commit()

        (tmp_path / "x.json").write_text(json.dumps(_scholarly_fixture()))

        with patch.object(db_setup, "SCHOLARLY_DIR", tmp_path):
            db_setup._load_scholarly_docs(mem_conn, "seed")
            db_setup._load_scholarly_docs(mem_conn, "seed")

        row_count: int = mem_conn.execute(
            "SELECT COUNT(*) FROM reference_docs"
        ).fetchone()[0]
        assert row_count == 1

    def test_coexists_with_fred_reference_docs(
        self,
        mem_conn: sqlite3.Connection,
        tmp_path: Path,
        sample_json_data: dict,
    ) -> None:
        """FRED rows and scholarly rows survive side by side for same series."""
        db_setup._insert_metadata(mem_conn, sample_json_data)
        mem_conn.commit()

        # FRED-style metadata
        raw_dir: Path = tmp_path / "raw"
        raw_dir.mkdir()
        (raw_dir / "UNRATE_metadata.json").write_text(
            json.dumps(
                {
                    "series_notes": "Official BLS methodology paragraph.",
                    "release_name": "Employment Situation",
                    "release_link": "https://www.bls.gov/news.release/empsit.toc.htm",
                    "release_notes": "Monthly BLS release.",
                    "category_path": "Labor Market > Unemployment",
                    "fetched_at": "2026-04-16T12:00:00Z",
                }
            )
        )

        # Scholarly fixture targeting the same series
        scholarly_dir: Path = tmp_path / "scholarly"
        scholarly_dir.mkdir()
        (scholarly_dir / "x.json").write_text(json.dumps(_scholarly_fixture()))

        with (
            patch.object(db_setup, "RAW_DIR", raw_dir),
            patch.object(db_setup, "SEED_SERIES", ["UNRATE"]),
            patch.object(db_setup, "SCHOLARLY_DIR", scholarly_dir),
        ):
            db_setup._load_reference_docs(mem_conn, "seed")
            db_setup._load_scholarly_docs(mem_conn, "seed")

        doc_types: list[str] = [
            r[0]
            for r in mem_conn.execute(
                "SELECT doc_type FROM reference_docs ORDER BY doc_type"
            ).fetchall()
        ]
        assert doc_types == [
            "category_path",
            "release_info",
            "scholarly:test_slug",
            "series_notes",
        ]


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


# ---------------------------------------------------------------------------
# _load_hn_stories (Phase 13)
# ---------------------------------------------------------------------------


def _hn_story(**overrides: object) -> dict:
    """Build a minimal scored HN story dict for fixtures."""
    base: dict = {
        "story_id": 111,
        "created_utc": "2023-03-15T10:30:00+00:00",
        "title": "Example HN title",
        "text_excerpt": "Some excerpt text.",
        "score": 100,
        "num_comments": 25,
        "url": "https://example.com/post",
        "hn_permalink": "https://news.ycombinator.com/item?id=111",
        "sentiment_score": -0.4,
        "sentiment_label": "negative",
        "scored_at": "2026-04-16T12:00:00+00:00",
        "matched_queries": ["layoffs"],
    }
    base.update(overrides)
    return base


class TestLoadHnStories:
    """Tests for ``_load_hn_stories`` (Phase 13)."""

    def test_loads_from_scored_json(
        self, mem_conn: sqlite3.Connection, tmp_path: Path
    ) -> None:
        """A scored JSON file populates hn_stories."""
        cache: Path = tmp_path / "hn_stories.json"
        stories: list[dict] = [
            _hn_story(story_id=1),
            _hn_story(story_id=2, title="Another"),
        ]
        cache.write_text(json.dumps(stories))

        with patch.object(db_setup, "HN_STORIES_PATH", cache):
            count: int = db_setup._load_hn_stories(mem_conn, "seed")

        assert count == 2
        rows: list[tuple] = mem_conn.execute(
            "SELECT story_id, title, sentiment_score, sentiment_label "
            "FROM hn_stories ORDER BY story_id"
        ).fetchall()
        assert rows[0][0] == 1
        assert rows[1][0] == 2
        assert rows[0][2] == pytest.approx(-0.4)
        assert rows[0][3] == "negative"

    def test_skips_missing_file(
        self, mem_conn: sqlite3.Connection, tmp_path: Path
    ) -> None:
        """A missing cache file returns 0 and does not raise."""
        cache: Path = tmp_path / "hn_stories.json"
        with patch.object(db_setup, "HN_STORIES_PATH", cache):
            count: int = db_setup._load_hn_stories(mem_conn, "seed")
        assert count == 0
        row_count: int = mem_conn.execute(
            "SELECT COUNT(*) FROM hn_stories"
        ).fetchone()[0]
        assert row_count == 0

    def test_skips_unscored_file(
        self, mem_conn: sqlite3.Connection, tmp_path: Path
    ) -> None:
        """A file missing sentiment_score on any story loads zero rows."""
        cache: Path = tmp_path / "hn_stories.json"
        stories: list[dict] = [_hn_story(story_id=1)]
        stories[0].pop("sentiment_score")
        cache.write_text(json.dumps(stories))

        with patch.object(db_setup, "HN_STORIES_PATH", cache):
            count: int = db_setup._load_hn_stories(mem_conn, "seed")
        assert count == 0
        row_count: int = mem_conn.execute(
            "SELECT COUNT(*) FROM hn_stories"
        ).fetchone()[0]
        assert row_count == 0

    def test_idempotent_upsert(
        self, mem_conn: sqlite3.Connection, tmp_path: Path
    ) -> None:
        """Running the loader twice does not duplicate rows."""
        cache: Path = tmp_path / "hn_stories.json"
        cache.write_text(json.dumps([_hn_story(story_id=42)]))
        with patch.object(db_setup, "HN_STORIES_PATH", cache):
            db_setup._load_hn_stories(mem_conn, "seed")
            db_setup._load_hn_stories(mem_conn, "seed")

        row_count: int = mem_conn.execute(
            "SELECT COUNT(*) FROM hn_stories"
        ).fetchone()[0]
        assert row_count == 1

    def test_month_column_is_yyyy_mm_01(
        self, mem_conn: sqlite3.Connection, tmp_path: Path
    ) -> None:
        """``month`` is derived as ``YYYY-MM-01`` from created_utc."""
        cache: Path = tmp_path / "hn_stories.json"
        cache.write_text(
            json.dumps(
                [
                    _hn_story(
                        story_id=77,
                        created_utc="2024-07-23T18:15:00+00:00",
                    )
                ]
            )
        )
        with patch.object(db_setup, "HN_STORIES_PATH", cache):
            db_setup._load_hn_stories(mem_conn, "seed")

        month: str = mem_conn.execute(
            "SELECT month FROM hn_stories WHERE story_id = 77"
        ).fetchone()[0]
        assert month == "2024-07-01"

    def test_stores_url_as_null_for_self_posts(
        self, mem_conn: sqlite3.Connection, tmp_path: Path
    ) -> None:
        """Self-post stories (url=None in JSON) become SQL NULL."""
        cache: Path = tmp_path / "hn_stories.json"
        cache.write_text(json.dumps([_hn_story(story_id=9, url=None)]))
        with patch.object(db_setup, "HN_STORIES_PATH", cache):
            db_setup._load_hn_stories(mem_conn, "seed")

        url: object = mem_conn.execute(
            "SELECT url FROM hn_stories WHERE story_id = 9"
        ).fetchone()[0]
        assert url is None


# ---------------------------------------------------------------------------
# _build_hn_monthly_aggregate (Phase 13)
# ---------------------------------------------------------------------------


class TestBuildHnMonthlyAggregate:
    """Tests for ``_build_hn_monthly_aggregate`` (Phase 13)."""

    def _load_stories(
        self,
        mem_conn: sqlite3.Connection,
        tmp_path: Path,
        stories: list[dict],
    ) -> None:
        """Helper: write stories to a cache and call the loader."""
        cache: Path = tmp_path / "hn_stories.json"
        cache.write_text(json.dumps(stories))
        with patch.object(db_setup, "HN_STORIES_PATH", cache):
            db_setup._load_hn_stories(mem_conn, "seed")

    def test_groups_by_month(
        self, mem_conn: sqlite3.Connection, tmp_path: Path
    ) -> None:
        """Stories from two months produce two aggregate rows."""
        self._load_stories(
            mem_conn,
            tmp_path,
            [
                _hn_story(
                    story_id=1,
                    created_utc="2023-03-01T00:00:00+00:00",
                    sentiment_score=-0.5,
                    title="tech layoffs again",
                ),
                _hn_story(
                    story_id=2,
                    created_utc="2023-03-15T00:00:00+00:00",
                    sentiment_score=-0.3,
                    title="Something ordinary",
                ),
                _hn_story(
                    story_id=3,
                    created_utc="2023-04-02T00:00:00+00:00",
                    sentiment_score=0.1,
                    title="Hiring surge",
                ),
            ],
        )
        count: int = db_setup._build_hn_monthly_aggregate(mem_conn)
        assert count == 2

        rows: list[tuple] = mem_conn.execute(
            "SELECT month, story_count FROM hn_sentiment_monthly "
            "ORDER BY month"
        ).fetchall()
        assert rows == [("2023-03-01", 2), ("2023-04-01", 1)]

    def test_layoff_story_count_matches_manual(
        self, mem_conn: sqlite3.Connection, tmp_path: Path
    ) -> None:
        """Layoff-keyword matches count toward layoff_story_count."""
        self._load_stories(
            mem_conn,
            tmp_path,
            [
                _hn_story(
                    story_id=1,
                    title="Big Co layoffs continue",
                    text_excerpt="",
                    created_utc="2023-06-01T00:00:00+00:00",
                ),
                _hn_story(
                    story_id=2,
                    title="New AI lab launches product",
                    text_excerpt="",
                    created_utc="2023-06-02T00:00:00+00:00",
                ),
                _hn_story(
                    story_id=3,
                    title="Innocuous title",
                    text_excerpt="Severance packages offered to the team.",
                    created_utc="2023-06-03T00:00:00+00:00",
                ),
            ],
        )
        db_setup._build_hn_monthly_aggregate(mem_conn)

        row: tuple = mem_conn.execute(
            "SELECT story_count, layoff_story_count "
            "FROM hn_sentiment_monthly WHERE month = '2023-06-01'"
        ).fetchone()
        assert row == (3, 2)

    def test_mean_sentiment_is_unweighted_mean(
        self, mem_conn: sqlite3.Connection, tmp_path: Path
    ) -> None:
        """mean_sentiment is the arithmetic mean of sentiment_score."""
        self._load_stories(
            mem_conn,
            tmp_path,
            [
                _hn_story(
                    story_id=1,
                    sentiment_score=-0.5,
                    created_utc="2023-08-01T00:00:00+00:00",
                ),
                _hn_story(
                    story_id=2,
                    sentiment_score=0.1,
                    created_utc="2023-08-02T00:00:00+00:00",
                ),
                _hn_story(
                    story_id=3,
                    sentiment_score=0.4,
                    created_utc="2023-08-03T00:00:00+00:00",
                ),
            ],
        )
        db_setup._build_hn_monthly_aggregate(mem_conn)
        mean: float = mem_conn.execute(
            "SELECT mean_sentiment FROM hn_sentiment_monthly "
            "WHERE month = '2023-08-01'"
        ).fetchone()[0]
        assert mean == pytest.approx(0.0, abs=1e-9)

    def test_idempotent(
        self, mem_conn: sqlite3.Connection, tmp_path: Path
    ) -> None:
        """Running the aggregate twice yields the same row count."""
        self._load_stories(
            mem_conn,
            tmp_path,
            [
                _hn_story(
                    story_id=1,
                    created_utc="2023-09-01T00:00:00+00:00",
                )
            ],
        )
        db_setup._build_hn_monthly_aggregate(mem_conn)
        first: int = mem_conn.execute(
            "SELECT COUNT(*) FROM hn_sentiment_monthly"
        ).fetchone()[0]

        db_setup._build_hn_monthly_aggregate(mem_conn)
        second: int = mem_conn.execute(
            "SELECT COUNT(*) FROM hn_sentiment_monthly"
        ).fetchone()[0]

        assert first == second == 1


# ---------------------------------------------------------------------------
# build_database includes HN tables (Phase 13)
# ---------------------------------------------------------------------------


class TestBuildDatabaseIncludesHn:
    """Smoke test that ``build_database`` populates the HN tables."""

    def test_populates_hn_tables(self, tmp_path: Path) -> None:
        """End-to-end build loads FRED + HN data when both JSONs exist."""
        raw_dir: Path = tmp_path / "raw"
        raw_dir.mkdir()

        fred_payload: dict = {
            "series_id": "UNRATE",
            "name": "Unemployment Rate",
            "category": "labor_market",
            "frequency": "monthly",
            "observations": [{"date": "2024-01-01", "value": 3.5}],
        }
        (raw_dir / "UNRATE.json").write_text(json.dumps(fred_payload))

        hn_cache: Path = raw_dir / "hn_stories.json"
        hn_cache.write_text(
            json.dumps(
                [
                    _hn_story(
                        story_id=1,
                        created_utc="2023-06-01T00:00:00+00:00",
                    ),
                    _hn_story(
                        story_id=2,
                        created_utc="2023-07-01T00:00:00+00:00",
                    ),
                ]
            )
        )

        with (
            patch.object(db_setup, "DATA_DIR", tmp_path),
            patch.object(db_setup, "RAW_DIR", raw_dir),
            patch.object(db_setup, "HN_STORIES_PATH", hn_cache),
        ):
            db_path: Path = db_setup.build_database("seed")

        conn: sqlite3.Connection = sqlite3.connect(db_path)
        try:
            story_count: int = conn.execute(
                "SELECT COUNT(*) FROM hn_stories"
            ).fetchone()[0]
            monthly_count: int = conn.execute(
                "SELECT COUNT(*) FROM hn_sentiment_monthly"
            ).fetchone()[0]
            social_count: int = conn.execute(
                "SELECT COUNT(*) FROM reference_docs "
                "WHERE doc_type LIKE 'social:hn:%'"
            ).fetchone()[0]
        finally:
            conn.close()

        assert story_count == 2
        assert monthly_count == 2
        # Phase 14: HN reference_docs are loaded alongside the HN tables.
        assert social_count == 2


# ---------------------------------------------------------------------------
# _load_hn_reference_docs (Phase 14)
# ---------------------------------------------------------------------------


def _insert_hn_story_row(
    conn: sqlite3.Connection,
    *,
    story_id: int,
    month: str = "2023-06-01",
    created_utc: str | None = None,
    score: int = 100,
    title: str = "Example HN title",
    text_excerpt: str = "Some excerpt text.",
    hn_permalink: str | None = None,
    sentiment_score: float = 0.0,
    sentiment_label: str = "neutral",
) -> None:
    """Insert a single row into ``hn_stories`` for reference_docs tests."""
    conn.execute(
        """
        INSERT INTO hn_stories (
            story_id, created_utc, month, title, text_excerpt,
            score, num_comments, url, hn_permalink,
            sentiment_score, sentiment_label
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            story_id,
            created_utc or f"{month[:7]}-15T12:00:00+00:00",
            month,
            title,
            text_excerpt,
            score,
            0,
            None,
            hn_permalink or f"https://news.ycombinator.com/item?id={story_id}",
            sentiment_score,
            sentiment_label,
        ),
    )


class TestLoadHnReferenceDocs:
    """Tests for ``_load_hn_reference_docs`` (Phase 14)."""

    def test_loads_top_5_per_month(self, mem_conn: sqlite3.Connection) -> None:
        """Exactly 5 reference_docs rows per month, ordered by score desc."""
        # Seed 10 stories in one month with distinct descending scores.
        scores: list[int] = [90, 85, 80, 75, 70, 65, 60, 55, 50, 45]
        for idx, score in enumerate(scores, start=1):
            _insert_hn_story_row(
                mem_conn,
                story_id=1000 + idx,
                month="2023-06-01",
                score=score,
            )

        count: int = db_setup._load_hn_reference_docs(mem_conn, "seed")
        assert count == 5

        # The top-5 are the ones with scores 90, 85, 80, 75, 70 → story ids
        # 1001..1005.
        selected_ids: list[int] = [
            int(row[0].split(":")[-1])
            for row in mem_conn.execute(
                """
                SELECT doc_type FROM reference_docs
                WHERE doc_type LIKE 'social:hn:%'
                ORDER BY id ASC
                """
            ).fetchall()
        ]
        assert sorted(selected_ids) == [1001, 1002, 1003, 1004, 1005]

    def test_tie_break_by_story_id_ascending(
        self, mem_conn: sqlite3.Connection
    ) -> None:
        """Ties on score resolve deterministically by story_id ASC."""
        # Four stories all tied at score=100. Top-5 cap means all four get
        # picked (not limited by ties), but ordering matters for determinism.
        for story_id in (205, 103, 407, 301, 509, 111):
            _insert_hn_story_row(
                mem_conn,
                story_id=story_id,
                month="2023-08-01",
                score=100,
            )

        db_setup._load_hn_reference_docs(mem_conn, "seed")

        selected: list[int] = sorted(
            int(row[0].split(":")[-1])
            for row in mem_conn.execute(
                "SELECT doc_type FROM reference_docs "
                "WHERE doc_type LIKE 'social:hn:%'"
            ).fetchall()
        )
        # Top-5 from {103, 111, 205, 301, 407, 509} sorted ASC = the first 5.
        assert selected == [103, 111, 205, 301, 407]

    def test_assigns_series_id_usinfo_and_doc_type_prefix(
        self, mem_conn: sqlite3.Connection
    ) -> None:
        """Every row uses series_id='USINFO' and doc_type='social:hn:<id>'."""
        _insert_hn_story_row(mem_conn, story_id=42, month="2023-01-01")
        db_setup._load_hn_reference_docs(mem_conn, "seed")

        rows: list[tuple[str, str]] = mem_conn.execute(
            "SELECT series_id, doc_type FROM reference_docs "
            "WHERE doc_type LIKE 'social:hn:%'"
        ).fetchall()
        assert len(rows) == 1
        assert rows[0][0] == "USINFO"
        assert rows[0][1] == "social:hn:42"

    def test_content_concatenates_title_and_excerpt(
        self, mem_conn: sqlite3.Connection
    ) -> None:
        """Story with title + excerpt produces joined content."""
        _insert_hn_story_row(
            mem_conn,
            story_id=7,
            month="2023-02-01",
            title="Layoffs announced",
            text_excerpt="Details in the filing.",
        )
        db_setup._load_hn_reference_docs(mem_conn, "seed")

        content: str = mem_conn.execute(
            "SELECT content FROM reference_docs "
            "WHERE doc_type = 'social:hn:7'"
        ).fetchone()[0]
        assert content == "Layoffs announced\n\nDetails in the filing."

    def test_content_is_title_only_when_excerpt_empty(
        self, mem_conn: sqlite3.Connection
    ) -> None:
        """Link-post story (empty excerpt) stores title only."""
        _insert_hn_story_row(
            mem_conn,
            story_id=8,
            month="2023-02-01",
            title="Tech hiring freeze",
            text_excerpt="",
        )
        db_setup._load_hn_reference_docs(mem_conn, "seed")

        content: str = mem_conn.execute(
            "SELECT content FROM reference_docs "
            "WHERE doc_type = 'social:hn:8'"
        ).fetchone()[0]
        assert content == "Tech hiring freeze"

    def test_uses_hn_permalink_as_source_url(
        self, mem_conn: sqlite3.Connection
    ) -> None:
        """``source_url`` is the raw hn_permalink."""
        _insert_hn_story_row(
            mem_conn,
            story_id=99,
            month="2023-03-01",
            hn_permalink="https://news.ycombinator.com/item?id=99",
        )
        db_setup._load_hn_reference_docs(mem_conn, "seed")

        url: str = mem_conn.execute(
            "SELECT source_url FROM reference_docs "
            "WHERE doc_type = 'social:hn:99'"
        ).fetchone()[0]
        assert url == "https://news.ycombinator.com/item?id=99"

    def test_idempotent_upsert(self, mem_conn: sqlite3.Connection) -> None:
        """Re-running the loader produces the same rows, same count."""
        _insert_hn_story_row(mem_conn, story_id=1, month="2023-04-01")
        _insert_hn_story_row(mem_conn, story_id=2, month="2023-04-01")

        first: int = db_setup._load_hn_reference_docs(mem_conn, "seed")
        second: int = db_setup._load_hn_reference_docs(mem_conn, "seed")

        assert first == second == 2

    def test_prunes_orphan_social_rows(
        self, mem_conn: sqlite3.Connection
    ) -> None:
        """Pre-existing social rows not in the new top-5 are removed."""
        # Pre-seed an orphan: a social row whose story_id is NOT in hn_stories.
        mem_conn.execute(
            """
            INSERT INTO reference_docs (
                series_id, doc_type, title, content, source_url, fetched_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                "USINFO",
                "social:hn:9999999",
                "Stale title",
                "Stale content",
                "https://news.ycombinator.com/item?id=9999999",
                "2020-01-01T00:00:00+00:00",
            ),
        )
        mem_conn.commit()

        _insert_hn_story_row(mem_conn, story_id=1, month="2023-05-01")
        db_setup._load_hn_reference_docs(mem_conn, "seed")

        remaining: list[tuple[str]] = mem_conn.execute(
            "SELECT doc_type FROM reference_docs "
            "WHERE doc_type LIKE 'social:hn:%' ORDER BY doc_type"
        ).fetchall()
        doc_types: list[str] = [row[0] for row in remaining]
        assert "social:hn:9999999" not in doc_types
        assert doc_types == ["social:hn:1"]

    def test_skips_when_hn_stories_empty(
        self, mem_conn: sqlite3.Connection
    ) -> None:
        """Empty ``hn_stories`` returns 0 and writes no rows."""
        count: int = db_setup._load_hn_reference_docs(mem_conn, "seed")
        assert count == 0
        remaining: int = mem_conn.execute(
            "SELECT COUNT(*) FROM reference_docs "
            "WHERE doc_type LIKE 'social:hn:%'"
        ).fetchone()[0]
        assert remaining == 0


# ---------------------------------------------------------------------------
# _load_concept_docs (Phase 18)
# ---------------------------------------------------------------------------


class TestLoadConceptDocs:
    """Tests for ``_load_concept_docs`` (Phase 18)."""

    def test_loads_concept_files(
        self, mem_conn: sqlite3.Connection, tmp_path: Path
    ) -> None:
        """Markdown files produce concept:<slug> reference_docs rows."""
        concepts_dir: Path = tmp_path / "concepts"
        concepts_dir.mkdir()
        (concepts_dir / "yield_curve.md").write_text(
            "# The Yield Curve\n\nContent about yield curves."
        )
        (concepts_dir / "cpi_inflation.md").write_text(
            "# CPI and Inflation\n\nContent about CPI."
        )

        with patch.object(db_setup, "CONCEPTS_DIR", concepts_dir):
            count: int = db_setup._load_concept_docs(mem_conn, "seed")

        assert count == 2
        rows: list[tuple] = mem_conn.execute(
            "SELECT series_id, doc_type, title FROM reference_docs "
            "WHERE doc_type LIKE 'concept:%' ORDER BY doc_type"
        ).fetchall()
        assert rows[0] == ("_CROSS_SERIES", "concept:cpi_inflation", "CPI and Inflation")
        assert rows[1] == ("_CROSS_SERIES", "concept:yield_curve", "The Yield Curve")

    def test_missing_directory_returns_zero(
        self, mem_conn: sqlite3.Connection, tmp_path: Path
    ) -> None:
        """Missing concepts directory returns 0 without raising."""
        missing: Path = tmp_path / "does_not_exist"
        with patch.object(db_setup, "CONCEPTS_DIR", missing):
            count: int = db_setup._load_concept_docs(mem_conn, "seed")

        assert count == 0

    def test_empty_file_skipped(
        self, mem_conn: sqlite3.Connection, tmp_path: Path
    ) -> None:
        """Empty or whitespace-only markdown files are skipped."""
        concepts_dir: Path = tmp_path / "concepts"
        concepts_dir.mkdir()
        (concepts_dir / "empty.md").write_text("   \n  ")

        with patch.object(db_setup, "CONCEPTS_DIR", concepts_dir):
            count: int = db_setup._load_concept_docs(mem_conn, "seed")

        assert count == 0

    def test_title_fallback_to_filename(
        self, mem_conn: sqlite3.Connection, tmp_path: Path
    ) -> None:
        """File without a heading uses the filename stem as title."""
        concepts_dir: Path = tmp_path / "concepts"
        concepts_dir.mkdir()
        (concepts_dir / "no_heading.md").write_text(
            "Just content with no heading line."
        )

        with patch.object(db_setup, "CONCEPTS_DIR", concepts_dir):
            db_setup._load_concept_docs(mem_conn, "seed")

        title: str = mem_conn.execute(
            "SELECT title FROM reference_docs WHERE doc_type = 'concept:no_heading'"
        ).fetchone()[0]
        assert title == "no_heading"

    def test_idempotent_upsert(
        self, mem_conn: sqlite3.Connection, tmp_path: Path
    ) -> None:
        """Running the loader twice does not duplicate rows."""
        concepts_dir: Path = tmp_path / "concepts"
        concepts_dir.mkdir()
        (concepts_dir / "test.md").write_text("# Test\n\nContent.")

        with patch.object(db_setup, "CONCEPTS_DIR", concepts_dir):
            db_setup._load_concept_docs(mem_conn, "seed")
            db_setup._load_concept_docs(mem_conn, "seed")

        row_count: int = mem_conn.execute(
            "SELECT COUNT(*) FROM reference_docs WHERE doc_type LIKE 'concept:%'"
        ).fetchone()[0]
        assert row_count == 1

    def test_source_url_is_empty_string(
        self, mem_conn: sqlite3.Connection, tmp_path: Path
    ) -> None:
        """Concept docs have empty string as source_url (developer-written)."""
        concepts_dir: Path = tmp_path / "concepts"
        concepts_dir.mkdir()
        (concepts_dir / "test.md").write_text("# Test\n\nContent.")

        with patch.object(db_setup, "CONCEPTS_DIR", concepts_dir):
            db_setup._load_concept_docs(mem_conn, "seed")

        url: str = mem_conn.execute(
            "SELECT source_url FROM reference_docs WHERE doc_type = 'concept:test'"
        ).fetchone()[0]
        assert url == ""

    def test_series_id_is_cross_series(
        self, mem_conn: sqlite3.Connection, tmp_path: Path
    ) -> None:
        """All concept docs use the _CROSS_SERIES sentinel series_id."""
        concepts_dir: Path = tmp_path / "concepts"
        concepts_dir.mkdir()
        (concepts_dir / "a.md").write_text("# A\n\nContent A.")
        (concepts_dir / "b.md").write_text("# B\n\nContent B.")

        with patch.object(db_setup, "CONCEPTS_DIR", concepts_dir):
            db_setup._load_concept_docs(mem_conn, "seed")

        series_ids: list[str] = [
            row[0]
            for row in mem_conn.execute(
                "SELECT DISTINCT series_id FROM reference_docs "
                "WHERE doc_type LIKE 'concept:%'"
            ).fetchall()
        ]
        assert series_ids == ["_CROSS_SERIES"]
