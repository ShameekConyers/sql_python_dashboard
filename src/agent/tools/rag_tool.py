"""RAG retrieval tool for the economic dashboard agent.

Wraps the existing ``rag_retrieval.retrieve()`` function as a LangGraph-
compatible tool. The agent uses this to answer conceptual questions about
economic indicators, FRED series documentation, and scholarly sources
without needing SQL.
"""

from __future__ import annotations

from langchain_core.tools import tool

from src.rag_retrieval import RetrievedChunk, retrieve

_SNIPPET_MAX_CHARS: int = 150
"""Maximum characters of chunk content to include in the reference_block summary."""


def _format_reference_block(chunks: list[RetrievedChunk]) -> str:
    """Format retrieved chunks as a ``REFERENCE CONTEXT`` block.

    Each chunk gets a positional ``[ref:N]`` tag (1-indexed) that the agent
    can cite in its narrative. The content is truncated to the first
    ``_SNIPPET_MAX_CHARS`` characters for the summary line.

    Args:
        chunks: Retrieved chunks from the vector store.

    Returns:
        A multi-line string starting with ``REFERENCE CONTEXT:`` followed
        by one summary line per chunk.
    """
    if not chunks:
        return "No reference sources available."

    lines: list[str] = ["REFERENCE CONTEXT:"]
    for idx, chunk in enumerate(chunks, start=1):
        snippet: str = chunk.content.strip()[:_SNIPPET_MAX_CHARS]
        if len(chunk.content.strip()) > _SNIPPET_MAX_CHARS:
            snippet += "..."
        lines.append(f'[ref:{idx}] {chunk.title} — "{snippet}"')
    return "\n".join(lines)


def _chunk_to_dict(chunk: RetrievedChunk, ref_id: int) -> dict:
    """Convert a ``RetrievedChunk`` to a serializable dict.

    Args:
        chunk: A single retrieval hit.
        ref_id: Positional reference ID (1-indexed).

    Returns:
        Dict with ref_id, doc_id, title, doc_type, content, source_url,
        and score.
    """
    return {
        "ref_id": ref_id,
        "doc_id": chunk.doc_id,
        "title": chunk.title,
        "doc_type": chunk.doc_type,
        "content": chunk.content,
        "source_url": chunk.source_url or "",
        "score": chunk.score,
    }


def make_rag_tool():
    """Create a LangGraph-compatible RAG retrieval tool.

    Returns:
        A ``@tool``-decorated function suitable for binding to a
        LangGraph agent.
    """

    @tool
    def retrieve_context(query: str) -> dict:
        """Search the reference knowledge base for economic concepts, FRED series documentation, and scholarly sources.

        Use this tool to answer conceptual questions like "What is the yield
        curve?" or "How is CPI measured?" without needing SQL. The result
        includes formatted reference context with [ref:N] citation tags.
        """
        chunks: list[RetrievedChunk] = retrieve(query, k=5)

        chunk_dicts: list[dict] = [
            _chunk_to_dict(c, idx) for idx, c in enumerate(chunks, start=1)
        ]
        reference_block: str = _format_reference_block(chunks)

        return {
            "chunks": chunk_dicts,
            "reference_block": reference_block,
        }

    return retrieve_context
