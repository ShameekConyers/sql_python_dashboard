"""Tests for src/embed_references.py.

The chunker is a pure function and is tested directly. The embedding and
ChromaDB persistence layers are tested with a mocked sentence-transformers
model and an in-memory ChromaDB client so we never download the real model
or hit the filesystem during test runs.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import embed_references


# ---------------------------------------------------------------------------
# chunk_document
# ---------------------------------------------------------------------------


class TestChunkDocument:
    """Tests for the pure-function chunker."""

    def test_empty_content_returns_empty_list(self) -> None:
        """Empty or whitespace-only input yields no chunks."""
        assert embed_references.chunk_document("") == []
        assert embed_references.chunk_document("   \n  \n") == []

    def test_short_content_single_chunk(self) -> None:
        """Content below max_tokens is returned as a single chunk."""
        text: str = "Short methodology note."
        chunks: list[str] = embed_references.chunk_document(text, max_tokens=200)
        assert chunks == [text]

    def test_long_content_splits_on_sentences(self) -> None:
        """Long content is split across multiple chunks."""
        sentence: str = "This is a methodology sentence. "
        long_text: str = sentence * 200  # ~5400 chars, well over 200 tokens
        chunks: list[str] = embed_references.chunk_document(
            long_text, max_tokens=50, overlap_sentences=1
        )
        assert len(chunks) > 1
        # Every chunk is non-empty
        for c in chunks:
            assert c.strip()

    def test_paragraph_boundaries_respected(self) -> None:
        """Chunks prefer paragraph boundaries over mid-sentence splits."""
        para_a: str = "Alpha " * 40
        para_b: str = "Beta " * 40
        text: str = f"{para_a}\n\n{para_b}"
        chunks: list[str] = embed_references.chunk_document(
            text, max_tokens=60, overlap_sentences=0
        )
        # At small max_tokens, the two paragraphs land in separate chunks.
        assert len(chunks) >= 2

    def test_oversized_sentence_still_emitted(self) -> None:
        """A single sentence exceeding max_tokens is emitted as its own chunk."""
        giant: str = "word " * 500  # one giant sentence (no terminal period)
        chunks: list[str] = embed_references.chunk_document(
            giant, max_tokens=50
        )
        assert len(chunks) >= 1
        assert all(c.strip() for c in chunks)

    def test_overlap_carries_trailing_sentences(self) -> None:
        """Adjacent chunks share the overlap sentences for continuity."""
        sentences: list[str] = [f"Sentence {i}." for i in range(40)]
        text: str = " ".join(sentences)
        chunks: list[str] = embed_references.chunk_document(
            text, max_tokens=15, overlap_sentences=1
        )
        # At least two chunks; last sentence of chunk[n] should appear in
        # the start of chunk[n+1] when overlap is enabled.
        assert len(chunks) >= 2
        first_last: str = chunks[0].split(".")[-2].strip() + "."
        assert first_last in chunks[1]


# ---------------------------------------------------------------------------
# embed_chunks
# ---------------------------------------------------------------------------


class TestEmbedChunks:
    """Tests for embed_chunks with a mocked model."""

    def test_empty_list_returns_empty(self) -> None:
        """No chunks returns no vectors; model is not invoked."""
        model = MagicMock()
        result: list[list[float]] = embed_references.embed_chunks([], model)
        assert result == []
        model.encode.assert_not_called()

    def test_returns_list_of_float_lists(self) -> None:
        """Each chunk becomes one float list of the model's dim."""
        import numpy as np

        model = MagicMock()
        model.encode.return_value = np.array(
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        )

        result: list[list[float]] = embed_references.embed_chunks(
            ["alpha", "beta"], model
        )

        assert len(result) == 2
        assert all(isinstance(v, list) for v in result)
        assert all(isinstance(x, float) for v in result for x in v)
        assert result[0] == [0.1, 0.2, 0.3]

    def test_passes_through_chunks(self) -> None:
        """The model receives the chunk list verbatim."""
        import numpy as np

        model = MagicMock()
        model.encode.return_value = np.array([[0.0]])
        embed_references.embed_chunks(["one"], model)

        args, kwargs = model.encode.call_args
        assert args[0] == ["one"]


# ---------------------------------------------------------------------------
# build_index (smoke test with mocked model + mocked chroma client)
# ---------------------------------------------------------------------------


class TestBuildIndex:
    """Integration-style tests for build_index with heavy deps mocked."""

    def test_missing_db_raises_file_not_found(self, tmp_path: Path) -> None:
        """build_index raises when the target DB does not exist."""
        with patch.object(embed_references, "DATA_DIR", tmp_path):
            with pytest.raises(FileNotFoundError):
                embed_references.build_index(mode="seed")

    def test_empty_reference_docs_returns_zero_counts(
        self, tmp_path: Path
    ) -> None:
        """With reference_docs empty, no model load or embed is attempted."""
        db_path: Path = tmp_path / "seed.db"
        conn: sqlite3.Connection = sqlite3.connect(db_path)
        conn.executescript(
            """
            CREATE TABLE reference_docs (
                id INTEGER PRIMARY KEY,
                series_id TEXT, doc_type TEXT, title TEXT,
                content TEXT, source_url TEXT, fetched_at TEXT
            );
            """
        )
        conn.close()

        with patch.object(embed_references, "DATA_DIR", tmp_path), patch.object(
            embed_references, "_load_model"
        ) as mock_load:
            result: dict[str, int] = embed_references.build_index(mode="seed")

        assert result == {"docs": 0, "chunks": 0}
        mock_load.assert_not_called()

    def test_chunks_and_upserts(self, tmp_path: Path) -> None:
        """With reference_docs present, upsert is called with chunk records."""
        import numpy as np

        db_path: Path = tmp_path / "seed.db"
        conn: sqlite3.Connection = sqlite3.connect(db_path)
        conn.executescript(
            """
            CREATE TABLE reference_docs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                series_id TEXT, doc_type TEXT, title TEXT,
                content TEXT, source_url TEXT, fetched_at TEXT
            );
            INSERT INTO reference_docs
              (series_id, doc_type, title, content, source_url, fetched_at)
              VALUES
              ('UNRATE', 'series_notes', 'FRED Series Notes — UNRATE',
               'Short methodology.', 'https://fred.stlouisfed.org/series/UNRATE',
               '2026-04-16T00:00:00Z'),
              ('GDPC1', 'category_path', 'Category — GDPC1',
               'National Accounts > GDP', NULL, '2026-04-16T00:00:00Z');
            """
        )
        conn.commit()
        conn.close()

        fake_model = MagicMock()
        fake_model.encode.return_value = np.zeros((2, 4))

        fake_collection = MagicMock()

        with (
            patch.object(embed_references, "DATA_DIR", tmp_path),
            patch.object(embed_references, "CHROMA_DIR", tmp_path / ".chroma"),
            patch.object(embed_references, "_load_model", return_value=fake_model),
            patch.object(
                embed_references,
                "_get_or_create_collection",
                return_value=fake_collection,
            ),
        ):
            result: dict[str, int] = embed_references.build_index(mode="seed")

        assert result == {"docs": 2, "chunks": 2}
        fake_collection.upsert.assert_called_once()
        kwargs = fake_collection.upsert.call_args.kwargs
        assert kwargs["ids"] == ["1::0", "2::0"]
        assert len(kwargs["embeddings"]) == 2
        assert kwargs["metadatas"][0]["doc_id"] == 1
        assert kwargs["metadatas"][0]["series_id"] == "UNRATE"
        assert kwargs["metadatas"][1]["doc_type"] == "category_path"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestCli:
    """Tests for _parse_args and main."""

    def test_default_args(self) -> None:
        """Default CLI args are seed + no rebuild."""
        ns = embed_references._parse_args([])
        assert ns.db == "seed"
        assert ns.rebuild is False

    def test_rebuild_flag(self) -> None:
        """--rebuild is parsed as True."""
        ns = embed_references._parse_args(["--rebuild"])
        assert ns.rebuild is True

    def test_full_flag(self) -> None:
        """--db full chooses the full DB."""
        ns = embed_references._parse_args(["--db", "full"])
        assert ns.db == "full"

    @patch("embed_references.build_index")
    def test_main_returns_zero_on_success(self, mock_build: MagicMock) -> None:
        """main returns 0 when build_index completes."""
        mock_build.return_value = {"docs": 1, "chunks": 1}
        assert embed_references.main([]) == 0

    @patch("embed_references.build_index")
    def test_main_returns_one_on_missing_db(self, mock_build: MagicMock) -> None:
        """main returns 1 when build_index raises FileNotFoundError."""
        mock_build.side_effect = FileNotFoundError("missing")
        assert embed_references.main([]) == 1


# ---------------------------------------------------------------------------
# Orphan collection-dir sweep (Phase 14)
# ---------------------------------------------------------------------------


class TestSweepOrphanCollectionDirs:
    """Tests for ``_sweep_orphan_collection_dirs`` (Phase 14)."""

    def test_sweeps_orphan_collection_dirs_on_rebuild(
        self, tmp_path: Path
    ) -> None:
        """A real chromadb rebuild plus a leftover uuid dir triggers sweep."""
        import chromadb

        chroma_dir: Path = tmp_path / ".chroma"
        chroma_dir.mkdir()

        # Create a real collection so chroma.sqlite3 has the expected shape.
        client = chromadb.PersistentClient(path=str(chroma_dir))
        client.get_or_create_collection(name=embed_references.COLLECTION_NAME)

        # Plant a bogus orphan uuid dir alongside the live one.
        orphan_dir: Path = chroma_dir / "00000000-dead-beef-cafe-000000000000"
        orphan_dir.mkdir()
        (orphan_dir / "data_level0.bin").write_bytes(b"stale bytes")

        with patch.object(embed_references, "CHROMA_DIR", chroma_dir):
            # rebuild=True drops + recreates the collection and sweeps.
            embed_references._get_or_create_collection(rebuild=True)

        assert not orphan_dir.exists(), "orphan dir should be removed"
        # Live collection still opens cleanly after the sweep.
        client2 = chromadb.PersistentClient(path=str(chroma_dir))
        assert (
            client2.get_collection(name=embed_references.COLLECTION_NAME)
            is not None
        )

    def test_sweep_skips_when_chroma_sqlite3_missing(
        self, tmp_path: Path
    ) -> None:
        """No sqlite file present → no-op, no exception."""
        chroma_dir: Path = tmp_path / ".chroma"
        chroma_dir.mkdir()
        # Plant an innocent directory; without chroma.sqlite3 the sweep
        # must not touch it.
        (chroma_dir / "some-uuid").mkdir()

        with patch.object(embed_references, "CHROMA_DIR", chroma_dir):
            removed: int = embed_references._sweep_orphan_collection_dirs()

        assert removed == 0
        assert (chroma_dir / "some-uuid").exists()

    def test_sweep_does_nothing_on_non_rebuild(self, tmp_path: Path) -> None:
        """``_get_or_create_collection(rebuild=False)`` leaves orphans alone."""
        import chromadb

        chroma_dir: Path = tmp_path / ".chroma"
        chroma_dir.mkdir()

        client = chromadb.PersistentClient(path=str(chroma_dir))
        client.get_or_create_collection(name=embed_references.COLLECTION_NAME)

        orphan_dir: Path = chroma_dir / "11111111-dead-beef-cafe-111111111111"
        orphan_dir.mkdir()

        with patch.object(embed_references, "CHROMA_DIR", chroma_dir):
            embed_references._get_or_create_collection(rebuild=False)

        assert orphan_dir.exists(), "non-rebuild path must not sweep"


# ---------------------------------------------------------------------------
# build_index picks up social doc_type rows (Phase 14)
# ---------------------------------------------------------------------------


class TestBuildIndexIncludesSocial:
    """Social reference_docs rows flow through chunking into upsert metadata."""

    def test_build_index_includes_social_doc_type(
        self, tmp_path: Path
    ) -> None:
        """Upsert metadata preserves ``social:hn:<id>`` doc_type values."""
        import numpy as np

        db_path: Path = tmp_path / "seed.db"
        conn: sqlite3.Connection = sqlite3.connect(db_path)
        conn.executescript(
            """
            CREATE TABLE reference_docs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                series_id TEXT, doc_type TEXT, title TEXT,
                content TEXT, source_url TEXT, fetched_at TEXT
            );
            INSERT INTO reference_docs
              (series_id, doc_type, title, content, source_url, fetched_at)
              VALUES
              ('USINFO', 'social:hn:12345', 'Layoffs at BigCo',
               'Layoffs at BigCo: details in the filing.',
               'https://news.ycombinator.com/item?id=12345',
               '2024-03-01T00:00:00+00:00');
            """
        )
        conn.commit()
        conn.close()

        fake_model = MagicMock()
        fake_model.encode.return_value = np.zeros((1, 4))
        fake_collection = MagicMock()

        with (
            patch.object(embed_references, "DATA_DIR", tmp_path),
            patch.object(embed_references, "CHROMA_DIR", tmp_path / ".chroma"),
            patch.object(
                embed_references, "_load_model", return_value=fake_model
            ),
            patch.object(
                embed_references,
                "_get_or_create_collection",
                return_value=fake_collection,
            ),
        ):
            embed_references.build_index(mode="seed")

        fake_collection.upsert.assert_called_once()
        kwargs = fake_collection.upsert.call_args.kwargs
        assert kwargs["metadatas"][0]["doc_type"] == "social:hn:12345"
        assert kwargs["metadatas"][0]["series_id"] == "USINFO"


# ---------------------------------------------------------------------------
# build_index picks up concept doc_type rows (Phase 18)
# ---------------------------------------------------------------------------


class TestBuildIndexIncludesConcepts:
    """Concept reference_docs rows flow through chunking into upsert metadata."""

    def test_build_index_includes_concept_doc_type(
        self, tmp_path: Path
    ) -> None:
        """Upsert metadata preserves ``concept:<slug>`` doc_type values."""
        import numpy as np

        db_path: Path = tmp_path / "seed.db"
        conn: sqlite3.Connection = sqlite3.connect(db_path)
        conn.executescript(
            """
            CREATE TABLE reference_docs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                series_id TEXT, doc_type TEXT, title TEXT,
                content TEXT, source_url TEXT, fetched_at TEXT
            );
            INSERT INTO reference_docs
              (series_id, doc_type, title, content, source_url, fetched_at)
              VALUES
              ('_CROSS_SERIES', 'concept:yield_curve',
               'The Yield Curve and Economic Forecasting',
               'The yield curve plots U.S. Treasury bond yields.',
               '', '2026-04-18');
            """
        )
        conn.commit()
        conn.close()

        fake_model = MagicMock()
        fake_model.encode.return_value = np.zeros((1, 4))
        fake_collection = MagicMock()

        with (
            patch.object(embed_references, "DATA_DIR", tmp_path),
            patch.object(embed_references, "CHROMA_DIR", tmp_path / ".chroma"),
            patch.object(
                embed_references, "_load_model", return_value=fake_model
            ),
            patch.object(
                embed_references,
                "_get_or_create_collection",
                return_value=fake_collection,
            ),
        ):
            embed_references.build_index(mode="seed")

        fake_collection.upsert.assert_called_once()
        kwargs = fake_collection.upsert.call_args.kwargs
        assert kwargs["metadatas"][0]["doc_type"] == "concept:yield_curve"
        assert kwargs["metadatas"][0]["series_id"] == "_CROSS_SERIES"
