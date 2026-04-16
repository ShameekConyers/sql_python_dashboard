"""Tests for src/rag_retrieval.py.

The retrieval wrapper must never crash the generation pipeline. These tests
cover the fallback path (missing persist dir, missing chromadb import,
runtime failure during query) as well as happy-path flattening of the
ChromaDB response shape.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import rag_retrieval


@pytest.fixture(autouse=True)
def _clear_caches() -> None:
    """Ensure module-level caches are clean before every test."""
    rag_retrieval._reset_cache()


# ---------------------------------------------------------------------------
# _get_collection — graceful fallback
# ---------------------------------------------------------------------------


class TestGetCollectionFallback:
    """Tests for _get_collection's degenerate paths."""

    def test_returns_none_when_dir_missing(self, tmp_path: Path) -> None:
        """Missing persist directory returns None and emits one warning."""
        missing: Path = tmp_path / "does-not-exist"
        with patch.object(rag_retrieval, "CHROMA_DIR", missing):
            assert rag_retrieval._get_collection() is None
            # Cached None on repeat call
            assert rag_retrieval._get_collection() is None

    def test_returns_none_when_import_fails(self, tmp_path: Path) -> None:
        """A failing chromadb import falls back to None."""
        # Create the dir so we reach the import step
        (tmp_path / ".chroma").mkdir()

        with patch.object(rag_retrieval, "CHROMA_DIR", tmp_path / ".chroma"):
            with patch.dict("sys.modules", {"chromadb": None}):
                assert rag_retrieval._get_collection() is None

    def test_returns_none_when_collection_open_fails(self, tmp_path: Path) -> None:
        """Collection errors fall back to None."""
        (tmp_path / ".chroma").mkdir()

        fake_client = MagicMock()
        fake_client.get_collection.side_effect = Exception("no such collection")
        fake_chromadb = MagicMock()
        fake_chromadb.PersistentClient.return_value = fake_client

        with patch.object(rag_retrieval, "CHROMA_DIR", tmp_path / ".chroma"):
            with patch.dict("sys.modules", {"chromadb": fake_chromadb}):
                assert rag_retrieval._get_collection() is None


# ---------------------------------------------------------------------------
# retrieve — fallback path
# ---------------------------------------------------------------------------


class TestRetrieveFallback:
    """Tests for retrieve() when the vector store is unavailable."""

    def test_empty_when_collection_missing(self, tmp_path: Path) -> None:
        """retrieve returns an empty list when the store is missing."""
        missing: Path = tmp_path / "does-not-exist"
        with patch.object(rag_retrieval, "CHROMA_DIR", missing):
            result = rag_retrieval.retrieve("unemployment rate")

        assert result == []

    def test_empty_query_returns_empty(self) -> None:
        """Blank queries short-circuit before touching the collection."""
        assert rag_retrieval.retrieve("") == []
        assert rag_retrieval.retrieve("   ") == []

    def test_query_exception_returns_empty(self) -> None:
        """Any runtime failure during query returns an empty list."""
        fake_collection = MagicMock()
        fake_collection.query.side_effect = Exception("chroma blew up")

        with patch.object(
            rag_retrieval, "_get_collection", return_value=fake_collection
        ):
            result = rag_retrieval.retrieve("unemployment rate", k=3)

        assert result == []


# ---------------------------------------------------------------------------
# retrieve — happy path
# ---------------------------------------------------------------------------


def _make_query_response() -> dict:
    """Return a canned ChromaDB query response for two hits."""
    return {
        "ids": [["12::0", "15::1"]],
        "metadatas": [
            [
                {
                    "doc_id": 12,
                    "series_id": "UNRATE",
                    "doc_type": "series_notes",
                    "title": "FRED Series Notes — UNRATE",
                    "source_url": "https://fred.stlouisfed.org/series/UNRATE",
                    "chunk_index": 0,
                },
                {
                    "doc_id": 15,
                    "series_id": "UNRATE",
                    "doc_type": "release_info",
                    "title": "Release — Employment Situation",
                    "source_url": "https://www.bls.gov/empsit",
                    "chunk_index": 1,
                },
            ]
        ],
        "documents": [
            [
                "The unemployment rate represents the unemployed as a percentage...",
                "The Employment Situation report is published monthly by BLS...",
            ]
        ],
        "distances": [[0.12, 0.23]],
    }


class TestRetrieveHappyPath:
    """Tests for retrieve()'s normal response flattening."""

    def test_returns_typed_chunks(self) -> None:
        """Query response is converted into RetrievedChunk records."""
        fake_collection = MagicMock()
        fake_collection.query.return_value = _make_query_response()

        with patch.object(
            rag_retrieval, "_get_collection", return_value=fake_collection
        ):
            result = rag_retrieval.retrieve(
                "unemployment", k=2, series_hint="UNRATE"
            )

        assert len(result) == 2
        first: rag_retrieval.RetrievedChunk = result[0]
        assert first.doc_id == 12
        assert first.series_id == "UNRATE"
        assert first.doc_type == "series_notes"
        assert first.source_url.startswith("https://fred")
        assert first.score == pytest.approx(0.12)
        assert "unemployment rate" in first.content.lower()

    def test_series_hint_passed_as_where_filter(self) -> None:
        """series_hint forwards a where={'series_id': ...} filter."""
        fake_collection = MagicMock()
        fake_collection.query.return_value = _make_query_response()

        with patch.object(
            rag_retrieval, "_get_collection", return_value=fake_collection
        ):
            rag_retrieval.retrieve("gdp", k=3, series_hint="GDPC1")

        kwargs = fake_collection.query.call_args.kwargs
        assert kwargs["where"] == {"series_id": "GDPC1"}

    def test_no_hint_omits_where(self) -> None:
        """Without a hint, where is None."""
        fake_collection = MagicMock()
        fake_collection.query.return_value = _make_query_response()

        with patch.object(
            rag_retrieval, "_get_collection", return_value=fake_collection
        ):
            rag_retrieval.retrieve("gdp", k=3)

        kwargs = fake_collection.query.call_args.kwargs
        assert kwargs["where"] is None

    def test_fallback_when_filter_returns_empty(self) -> None:
        """If the filtered query yields nothing, retry without the filter."""
        empty_response: dict = {
            "ids": [[]],
            "metadatas": [[]],
            "documents": [[]],
            "distances": [[]],
        }
        populated: dict = _make_query_response()

        fake_collection = MagicMock()
        # First call (with filter) returns empty; second (no filter) returns hits.
        fake_collection.query.side_effect = [empty_response, populated]

        with patch.object(
            rag_retrieval, "_get_collection", return_value=fake_collection
        ):
            result = rag_retrieval.retrieve(
                "obscure query", k=2, series_hint="UNRATE"
            )

        assert fake_collection.query.call_count == 2
        second_call = fake_collection.query.call_args_list[1]
        assert "where" not in second_call.kwargs or second_call.kwargs.get("where") is None
        assert len(result) == 2

    def test_empty_response_stays_empty(self) -> None:
        """When the store returns nothing, retrieve returns an empty list."""
        empty_response: dict = {
            "ids": [[]],
            "metadatas": [[]],
            "documents": [[]],
            "distances": [[]],
        }
        fake_collection = MagicMock()
        fake_collection.query.return_value = empty_response

        with patch.object(
            rag_retrieval, "_get_collection", return_value=fake_collection
        ):
            result = rag_retrieval.retrieve("anything", k=2)

        assert result == []


# ---------------------------------------------------------------------------
# _rows_from_query_result
# ---------------------------------------------------------------------------


class TestRowsFromQueryResult:
    """Tests for the response-flattening helper."""

    def test_handles_missing_fields(self) -> None:
        """A response with no hits converts to an empty list."""
        assert rag_retrieval._rows_from_query_result({}) == []
        assert (
            rag_retrieval._rows_from_query_result({"ids": [[]]}) == []
        )

    def test_null_source_url_becomes_none(self) -> None:
        """Missing/empty source_url in metadata maps to None (category_path)."""
        response: dict = {
            "ids": [["20::0"]],
            "metadatas": [
                [
                    {
                        "doc_id": 20,
                        "series_id": "GDPC1",
                        "doc_type": "category_path",
                        "title": "Category — GDPC1",
                        "source_url": "",  # stored as empty string
                        "chunk_index": 0,
                    }
                ]
            ],
            "documents": [["National Accounts > GDP"]],
            "distances": [[0.05]],
        }
        chunks = rag_retrieval._rows_from_query_result(response)
        assert len(chunks) == 1
        assert chunks[0].source_url is None


# ---------------------------------------------------------------------------
# SOCIAL_DOC_TYPE_PREFIX + scholarly floor (Phase 14)
# ---------------------------------------------------------------------------


class TestSocialDocTypePrefix:
    """Tests for the social doc_type prefix constant (Phase 14)."""

    def test_social_doc_type_prefix_constant_value(self) -> None:
        """The constant exposes the agreed prefix verbatim."""
        assert rag_retrieval.SOCIAL_DOC_TYPE_PREFIX == "social:"


class TestMinScholarlyIgnoresSocial:
    """Social chunks must not be counted toward the scholarly floor."""

    def test_min_scholarly_does_not_count_social_chunks(self) -> None:
        """A first-query result of all-social triggers a scholarly pad query."""
        social_metadata = [
            {
                "doc_id": 100 + i,
                "series_id": "USINFO",
                "doc_type": f"social:hn:{9000 + i}",
                "title": f"Social title {i}",
                "source_url": f"https://news.ycombinator.com/item?id={9000 + i}",
                "chunk_index": 0,
            }
            for i in range(5)
        ]
        first_response: dict = {
            "ids": [[f"social-{i}" for i in range(5)]],
            "metadatas": [social_metadata],
            "documents": [[f"social body {i}" for i in range(5)]],
            "distances": [[0.1 + 0.01 * i for i in range(5)]],
        }

        scholarly_metadata = [
            {
                "doc_id": 500,
                "series_id": "USINFO",
                "doc_type": "scholarly:bea_labor",
                "title": "BEA Labor Market",
                "source_url": "https://example.gov/bea",
                "chunk_index": 0,
            },
            {
                "doc_id": 501,
                "series_id": "USINFO",
                "doc_type": "scholarly:cea_erp",
                "title": "CEA Economic Report",
                "source_url": "https://example.gov/cea",
                "chunk_index": 0,
            },
            {
                "doc_id": 502,
                "series_id": "USINFO",
                "doc_type": "scholarly:eia_power",
                "title": "EIA Power",
                "source_url": "https://example.gov/eia",
                "chunk_index": 0,
            },
        ]
        pad_response: dict = {
            "ids": [[f"schol-{i}" for i in range(3)]],
            "metadatas": [scholarly_metadata],
            "documents": [[f"scholarly body {i}" for i in range(3)]],
            "distances": [[0.2, 0.21, 0.22]],
        }

        fake_collection = MagicMock()
        fake_collection.query.side_effect = [first_response, pad_response]

        with patch.object(
            rag_retrieval, "_get_collection", return_value=fake_collection
        ):
            result = rag_retrieval.retrieve(
                "labor market", k=5, series_hint="USINFO", min_scholarly=3
            )

        # Two queries: the primary (series-hinted) and the scholarly pad.
        assert fake_collection.query.call_count == 2
        pad_call = fake_collection.query.call_args_list[1]
        # Pad query should use the $nin scholarly filter.
        assert "where" in pad_call.kwargs
        assert pad_call.kwargs["where"] == {
            "doc_type": {"$nin": list(rag_retrieval.FRED_DOC_TYPES)}
        }
        # Final result contains the original 5 social chunks + 3 scholarly.
        scholarly_ids = [
            c.doc_id for c in result if c.doc_type.startswith("scholarly")
        ]
        assert set(scholarly_ids) == {500, 501, 502}
