"""Thin ChromaDB retrieval wrapper for RAG citations.

Isolates the vector store API from callers so ``ai_insights.py`` does not
directly depend on ChromaDB. Provides graceful degradation: when the persist
directory is missing, the collection is empty, or the chromadb import fails,
``retrieve()`` returns an empty list and logs a single warning per process
lifetime. Callers must handle the empty case.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
CHROMA_DIR: Path = PROJECT_ROOT / "data" / ".chroma"
COLLECTION_NAME: str = "fred_references"

logger: logging.Logger = logging.getLogger(__name__)

_COLLECTION_CACHE: dict[str, Any] = {}
"""Process-wide memoization cell for the Chroma collection handle."""

_WARNED: dict[str, bool] = {"missing": False}
"""One-shot guard so we emit the fallback warning at most once per process."""


@dataclass
class RetrievedChunk:
    """A retrieval hit attached to its source reference document.

    Attributes:
        doc_id: Foreign key into ``reference_docs.id``.
        series_id: The FRED series the reference applies to.
        doc_type: One of 'series_notes', 'release_info', 'category_path'.
        title: Human-readable title, carried over from the source doc.
        content: The chunk text (not the full source doc).
        source_url: FRED or BLS URL; None for category_path rows.
        score: Cosine distance. Lower = more similar.
    """

    doc_id: int
    series_id: str
    doc_type: str
    title: str
    content: str
    source_url: str | None
    score: float


def _get_collection() -> Any | None:
    """Open (or return the cached) ChromaDB collection.

    Returns:
        The collection handle, or None when ChromaDB cannot be used (missing
        import, missing persist directory, or no collection present). On the
        first miss this emits one warning; subsequent misses are silent.
    """
    if "collection" in _COLLECTION_CACHE:
        return _COLLECTION_CACHE["collection"]

    if not CHROMA_DIR.exists():
        if not _WARNED["missing"]:
            logger.warning(
                "ChromaDB persist dir %s is missing; retrieval disabled.",
                CHROMA_DIR,
            )
            _WARNED["missing"] = True
        _COLLECTION_CACHE["collection"] = None
        return None

    try:
        import chromadb  # noqa: PLC0415
    except Exception as e:
        if not _WARNED["missing"]:
            logger.warning(
                "chromadb import failed (%s); retrieval disabled.", e
            )
            _WARNED["missing"] = True
        _COLLECTION_CACHE["collection"] = None
        return None

    try:
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        collection = client.get_collection(name=COLLECTION_NAME)
    except Exception as e:
        if not _WARNED["missing"]:
            logger.warning(
                "Failed to open ChromaDB collection %s (%s); "
                "retrieval disabled.",
                COLLECTION_NAME,
                e,
            )
            _WARNED["missing"] = True
        _COLLECTION_CACHE["collection"] = None
        return None

    _COLLECTION_CACHE["collection"] = collection
    return collection


def _reset_cache() -> None:
    """Reset the memoized collection and warning guard. Test-only."""
    _COLLECTION_CACHE.clear()
    _WARNED["missing"] = False


def _rows_from_query_result(result: dict[str, Any]) -> list[RetrievedChunk]:
    """Flatten a ChromaDB query response into ``RetrievedChunk``s.

    ChromaDB returns each field as a list-of-lists (one inner list per query
    string). Phase 11 always queries with a single string, so we read the
    zeroth inner list for every field.

    Args:
        result: The dict returned by ``collection.query(...)``.

    Returns:
        A list of typed chunks. Empty when no hits or shape is malformed.
    """
    ids: list[list[str]] = result.get("ids") or [[]]
    metadatas: list[list[dict[str, Any]]] = result.get("metadatas") or [[]]
    documents: list[list[str]] = result.get("documents") or [[]]
    distances: list[list[float]] = result.get("distances") or [[]]

    if not ids or not ids[0]:
        return []

    chunks: list[RetrievedChunk] = []
    for idx, _chunk_id in enumerate(ids[0]):
        meta: dict[str, Any] = (
            metadatas[0][idx] if idx < len(metadatas[0]) else {}
        )
        doc: str = documents[0][idx] if idx < len(documents[0]) else ""
        distance: float = (
            float(distances[0][idx]) if idx < len(distances[0]) else 1.0
        )
        source_url: str | None = meta.get("source_url") or None
        chunks.append(
            RetrievedChunk(
                doc_id=int(meta.get("doc_id", 0)),
                series_id=str(meta.get("series_id", "")),
                doc_type=str(meta.get("doc_type", "")),
                title=str(meta.get("title", "")),
                content=doc,
                source_url=source_url,
                score=distance,
            )
        )
    return chunks


def retrieve(
    query: str,
    *,
    k: int = 5,
    series_hint: str | None = None,
) -> list[RetrievedChunk]:
    """Return the top-k most relevant reference chunks for ``query``.

    When ``series_hint`` is provided, filter the query to chunks whose
    metadata ``series_id`` matches. If that strict filter returns nothing,
    fall back to an unfiltered query so the caller still gets some context.

    Args:
        query: Free-form search string. Typically the metric name plus the
            slice's analysis prompt.
        k: Maximum chunks to return.
        series_hint: Optional FRED series ID to up-weight matches.

    Returns:
        A list of ``RetrievedChunk`` (possibly empty). Never raises; any
        underlying failure causes an empty return and a one-shot warning.
    """
    if not query.strip():
        return []

    collection = _get_collection()
    if collection is None:
        return []

    try:
        where: dict[str, Any] | None = (
            {"series_id": series_hint} if series_hint else None
        )
        result = collection.query(
            query_texts=[query],
            n_results=k,
            where=where,
        )
        chunks: list[RetrievedChunk] = _rows_from_query_result(result)
        if not chunks and series_hint:
            # Fall back to unfiltered query so a too-narrow hint does not
            # produce a hard empty — similarity alone is often good enough.
            result = collection.query(query_texts=[query], n_results=k)
            chunks = _rows_from_query_result(result)
        return chunks
    except Exception as e:
        if not _WARNED["missing"]:
            logger.warning(
                "ChromaDB query failed (%s); returning empty retrieval.",
                e,
            )
            _WARNED["missing"] = True
        return []
