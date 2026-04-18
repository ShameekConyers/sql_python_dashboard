"""Tests for src/agent/tools/rag_tool.py.

All tests mock ``rag_retrieval.retrieve`` so they run without ChromaDB
or the sentence-transformers model.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.agent.tools.rag_tool import (
    _SNIPPET_MAX_CHARS,
    _chunk_to_dict,
    _format_reference_block,
    make_rag_tool,
)
from src.rag_retrieval import RetrievedChunk


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_chunk(
    *,
    doc_id: int = 1,
    series_id: str = "UNRATE",
    doc_type: str = "concept:yield_curve",
    title: str = "The Yield Curve and Economic Forecasting",
    content: str = "The yield curve plots U.S. Treasury bond yields.",
    source_url: str | None = None,
    score: float = 0.25,
) -> RetrievedChunk:
    """Build a ``RetrievedChunk`` for testing."""
    return RetrievedChunk(
        doc_id=doc_id,
        series_id=series_id,
        doc_type=doc_type,
        title=title,
        content=content,
        source_url=source_url,
        score=score,
    )


# ---------------------------------------------------------------------------
# make_rag_tool
# ---------------------------------------------------------------------------


class TestMakeRagTool:
    """Tests for the ``make_rag_tool`` factory."""

    def test_returns_tool_with_invoke(self) -> None:
        """Factory returns a tool object with an invoke method."""
        tool_fn = make_rag_tool()
        assert hasattr(tool_fn, "invoke")

    def test_tool_has_correct_name(self) -> None:
        """The tool is named ``retrieve_context``."""
        tool_fn = make_rag_tool()
        assert tool_fn.name == "retrieve_context"

    def test_tool_has_description(self) -> None:
        """The tool has a non-empty description."""
        tool_fn = make_rag_tool()
        assert tool_fn.description
        assert "knowledge base" in tool_fn.description.lower()


# ---------------------------------------------------------------------------
# _format_reference_block
# ---------------------------------------------------------------------------


class TestFormatReferenceBlock:
    """Tests for the reference block formatter."""

    def test_empty_chunks_returns_no_sources(self) -> None:
        """No chunks produces a 'no sources' message."""
        result: str = _format_reference_block([])
        assert result == "No reference sources available."

    def test_single_chunk_formatting(self) -> None:
        """A single chunk produces a correctly formatted block."""
        chunk = _make_chunk(title="Test Title", content="Short content.")
        result: str = _format_reference_block([chunk])
        assert result.startswith("REFERENCE CONTEXT:")
        assert '[ref:1] Test Title — "Short content."' in result

    def test_multiple_chunks_numbered_sequentially(self) -> None:
        """Multiple chunks get sequential ref_id numbers."""
        chunks: list[RetrievedChunk] = [
            _make_chunk(doc_id=1, title="Alpha"),
            _make_chunk(doc_id=2, title="Beta"),
            _make_chunk(doc_id=3, title="Gamma"),
        ]
        result: str = _format_reference_block(chunks)
        assert "[ref:1] Alpha" in result
        assert "[ref:2] Beta" in result
        assert "[ref:3] Gamma" in result

    def test_long_content_truncated(self) -> None:
        """Content longer than _SNIPPET_MAX_CHARS is truncated with ellipsis."""
        long_content: str = "A" * (_SNIPPET_MAX_CHARS + 50)
        chunk = _make_chunk(content=long_content)
        result: str = _format_reference_block([chunk])
        assert "..." in result
        # The snippet should be exactly _SNIPPET_MAX_CHARS + "..."
        lines: list[str] = result.split("\n")
        assert len(lines[1]) < len(long_content) + 100

    def test_short_content_no_ellipsis(self) -> None:
        """Content within the limit has no trailing ellipsis."""
        chunk = _make_chunk(content="Short.")
        result: str = _format_reference_block([chunk])
        assert "..." not in result


# ---------------------------------------------------------------------------
# _chunk_to_dict
# ---------------------------------------------------------------------------


class TestChunkToDict:
    """Tests for the chunk-to-dict converter."""

    def test_includes_all_fields(self) -> None:
        """Dict has all expected keys."""
        chunk = _make_chunk(
            doc_id=42,
            title="Test",
            doc_type="concept:test",
            content="Content",
            source_url="https://example.com",
            score=0.35,
        )
        result: dict = _chunk_to_dict(chunk, ref_id=3)
        assert result["ref_id"] == 3
        assert result["doc_id"] == 42
        assert result["title"] == "Test"
        assert result["doc_type"] == "concept:test"
        assert result["content"] == "Content"
        assert result["source_url"] == "https://example.com"
        assert result["score"] == 0.35

    def test_none_source_url_becomes_empty_string(self) -> None:
        """A None source_url is converted to an empty string."""
        chunk = _make_chunk(source_url=None)
        result: dict = _chunk_to_dict(chunk, ref_id=1)
        assert result["source_url"] == ""


# ---------------------------------------------------------------------------
# retrieve_context tool invocation
# ---------------------------------------------------------------------------


class TestRetrieveContextTool:
    """Tests for the actual tool invocation (mocked retrieval)."""

    @patch("src.agent.tools.rag_tool.retrieve")
    def test_returns_chunks_and_reference_block(
        self, mock_retrieve: object
    ) -> None:
        """Tool returns dict with chunks list and reference_block string."""
        mock_retrieve.return_value = [
            _make_chunk(doc_id=1, title="Alpha"),
            _make_chunk(doc_id=2, title="Beta"),
        ]
        tool_fn = make_rag_tool()
        result: dict = tool_fn.invoke({"query": "yield curve"})

        assert "chunks" in result
        assert "reference_block" in result
        assert len(result["chunks"]) == 2
        assert result["chunks"][0]["ref_id"] == 1
        assert result["chunks"][1]["ref_id"] == 2
        assert "REFERENCE CONTEXT:" in result["reference_block"]

    @patch("src.agent.tools.rag_tool.retrieve")
    def test_empty_retrieval_returns_graceful_message(
        self, mock_retrieve: object
    ) -> None:
        """Empty retrieval returns no chunks and a graceful message."""
        mock_retrieve.return_value = []
        tool_fn = make_rag_tool()
        result: dict = tool_fn.invoke({"query": "something obscure"})

        assert result["chunks"] == []
        assert result["reference_block"] == "No reference sources available."

    @patch("src.agent.tools.rag_tool.retrieve")
    def test_passes_query_to_retrieve(self, mock_retrieve: object) -> None:
        """The query string is forwarded to rag_retrieval.retrieve."""
        mock_retrieve.return_value = []
        tool_fn = make_rag_tool()
        tool_fn.invoke({"query": "CPI inflation methodology"})

        mock_retrieve.assert_called_once_with("CPI inflation methodology", k=5)

    @patch("src.agent.tools.rag_tool.retrieve")
    def test_ref_ids_are_one_indexed(self, mock_retrieve: object) -> None:
        """ref_id values start at 1, not 0."""
        mock_retrieve.return_value = [
            _make_chunk(doc_id=10),
            _make_chunk(doc_id=20),
            _make_chunk(doc_id=30),
        ]
        tool_fn = make_rag_tool()
        result: dict = tool_fn.invoke({"query": "test"})

        ref_ids: list[int] = [c["ref_id"] for c in result["chunks"]]
        assert ref_ids == [1, 2, 3]

    @patch("src.agent.tools.rag_tool.retrieve")
    def test_reference_block_matches_chunk_count(
        self, mock_retrieve: object
    ) -> None:
        """reference_block has one [ref:N] line per chunk."""
        mock_retrieve.return_value = [
            _make_chunk(doc_id=i, title=f"Doc {i}") for i in range(1, 6)
        ]
        tool_fn = make_rag_tool()
        result: dict = tool_fn.invoke({"query": "test"})

        block: str = result["reference_block"]
        for i in range(1, 6):
            assert f"[ref:{i}]" in block

    @patch("src.agent.tools.rag_tool.retrieve")
    def test_preserves_source_url_in_chunks(
        self, mock_retrieve: object
    ) -> None:
        """Source URLs from chunks are preserved in the output dicts."""
        mock_retrieve.return_value = [
            _make_chunk(source_url="https://fred.stlouisfed.org/series/UNRATE"),
        ]
        tool_fn = make_rag_tool()
        result: dict = tool_fn.invoke({"query": "UNRATE"})

        assert (
            result["chunks"][0]["source_url"]
            == "https://fred.stlouisfed.org/series/UNRATE"
        )

    @patch("src.agent.tools.rag_tool.retrieve")
    def test_preserves_doc_type_in_chunks(
        self, mock_retrieve: object
    ) -> None:
        """doc_type from chunks is preserved in the output dicts."""
        mock_retrieve.return_value = [
            _make_chunk(doc_type="scholarly:cea_labor_2024"),
        ]
        tool_fn = make_rag_tool()
        result: dict = tool_fn.invoke({"query": "labor economics"})

        assert result["chunks"][0]["doc_type"] == "scholarly:cea_labor_2024"

    @patch("src.agent.tools.rag_tool.retrieve")
    def test_preserves_score_in_chunks(self, mock_retrieve: object) -> None:
        """Cosine distance score is preserved in chunk dicts."""
        mock_retrieve.return_value = [_make_chunk(score=0.42)]
        tool_fn = make_rag_tool()
        result: dict = tool_fn.invoke({"query": "test"})

        assert result["chunks"][0]["score"] == pytest.approx(0.42)
