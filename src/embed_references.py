"""Embed reference_docs rows into a local ChromaDB vector store.

Chunks long documents at paragraph/sentence boundaries, embeds each chunk
with ``sentence-transformers/all-MiniLM-L6-v2``, and persists the collection
to ``data/.chroma/``. Runs after ``db_setup.py`` (which populates
``reference_docs``) and before ``ai_insights.py`` (which retrieves from the
persisted store).

Usage:
    .venv/bin/python src/embed_references.py [--db seed|full] [--rebuild]
"""

from __future__ import annotations

import argparse
import logging
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
CHROMA_DIR: Path = DATA_DIR / ".chroma"
COLLECTION_NAME: str = "fred_references"
EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger: logging.Logger = logging.getLogger(__name__)

DEFAULT_MAX_TOKENS: int = 200
"""Approximate token cap per chunk. 200 tokens ≈ 800 characters for English."""

CHARS_PER_TOKEN: int = 4
"""Cheap character-to-token heuristic used by the chunker (English text)."""

DEFAULT_OVERLAP_SENTENCES: int = 1
"""Number of trailing sentences carried into the next chunk for continuity."""

_SENTENCE_SPLIT_RE: re.Pattern[str] = re.compile(r"(?<=[\.!?])\s+")
"""Split on sentence-terminal punctuation followed by whitespace."""


def _split_sentences(text: str) -> list[str]:
    """Split ``text`` into sentences using a punctuation regex.

    Args:
        text: Arbitrary text, possibly with multiple paragraphs.

    Returns:
        Non-empty, whitespace-trimmed sentence strings.
    """
    pieces: list[str] = _SENTENCE_SPLIT_RE.split(text)
    return [p.strip() for p in pieces if p.strip()]


def _approx_tokens(text: str) -> int:
    """Return an approximate token count for ``text``.

    Uses a simple character-based heuristic rather than pulling in a tokenizer
    dependency. This is accurate enough for the chunker's "should I split?"
    decision since the exact boundary does not matter.

    Args:
        text: Text to measure.

    Returns:
        Approximate token count, rounded up.
    """
    return max(1, (len(text) + CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN)


def chunk_document(
    content: str,
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    overlap_sentences: int = DEFAULT_OVERLAP_SENTENCES,
) -> list[str]:
    """Split ``content`` into overlapping, bounded chunks.

    Rules:
      1. If the whole document fits under ``max_tokens``, return it as a
         single chunk.
      2. Otherwise split on blank-line paragraph boundaries. If any paragraph
         still exceeds ``max_tokens``, split it at sentence boundaries.
      3. Adjacent chunks of the same document overlap by
         ``overlap_sentences`` trailing sentences from the previous chunk to
         preserve continuity across the boundary.

    Args:
        content: Full document text.
        max_tokens: Approximate maximum token count per chunk.
        overlap_sentences: Trailing sentences carried into the next chunk.

    Returns:
        List of chunk strings. Always returns at least one entry for any
        non-empty input. An empty input returns an empty list.
    """
    text: str = content.strip()
    if not text:
        return []

    # Case 1: short enough to emit as one chunk
    if _approx_tokens(text) <= max_tokens:
        return [text]

    # Case 2: split on blank-line paragraph boundaries, then on sentences
    paragraphs: list[str] = [
        p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()
    ]
    # If paragraph splitting did not meaningfully help (1 paragraph),
    # collapse to sentences directly.
    units: list[str]
    if len(paragraphs) <= 1:
        units = _split_sentences(text)
    else:
        # Further break any paragraph still over max_tokens by sentences
        units = []
        for paragraph in paragraphs:
            if _approx_tokens(paragraph) <= max_tokens:
                units.append(paragraph)
            else:
                units.extend(_split_sentences(paragraph))

    chunks: list[str] = []
    current: list[str] = []
    current_tokens: int = 0

    def _flush() -> None:
        """Emit the accumulated ``current`` buffer as a chunk."""
        nonlocal current, current_tokens
        if current:
            chunks.append(" ".join(current).strip())
            # Keep overlap sentences for continuity with the next chunk
            if overlap_sentences > 0 and len(current) > overlap_sentences:
                current = current[-overlap_sentences:]
                current_tokens = sum(_approx_tokens(s) for s in current)
            else:
                current = []
                current_tokens = 0

    for unit in units:
        unit_tokens: int = _approx_tokens(unit)
        # If a single unit exceeds the cap, still emit it as its own chunk
        # rather than dropping content. The retrieval layer tolerates long
        # chunks; better to over-include than truncate methodology text.
        if unit_tokens > max_tokens and not current:
            chunks.append(unit.strip())
            continue

        if current_tokens + unit_tokens > max_tokens and current:
            _flush()
        current.append(unit)
        current_tokens += unit_tokens

    _flush()
    # Overlap buffer carried past the last unit should not re-emit a
    # duplicate tail chunk; drop anything still in ``current`` after flush.
    return [c for c in chunks if c]


def embed_chunks(chunks: list[str], model: Any) -> list[list[float]]:
    """Embed ``chunks`` using the given sentence-transformers model.

    Args:
        chunks: Strings to embed.
        model: Object with an ``encode(list[str]) -> ndarray`` method
            compatible with sentence-transformers.

    Returns:
        One embedding vector per chunk, as a list of float lists.
    """
    if not chunks:
        return []
    vectors = model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
    return [list(map(float, v)) for v in vectors]


def _fetch_reference_rows(
    conn: sqlite3.Connection,
) -> list[dict[str, Any]]:
    """Return all reference_docs rows as dicts.

    Args:
        conn: Open SQLite connection on the target DB.

    Returns:
        List of dicts keyed by ``id``, ``series_id``, ``doc_type``, ``title``,
        ``content``, and ``source_url``.
    """
    rows = conn.execute(
        """
        SELECT id, series_id, doc_type, title, content, source_url
        FROM reference_docs
        ORDER BY id
        """
    ).fetchall()
    return [
        {
            "id": row[0],
            "series_id": row[1],
            "doc_type": row[2],
            "title": row[3],
            "content": row[4],
            "source_url": row[5] or "",
        }
        for row in rows
    ]


def _db_path(mode: str) -> Path:
    """Return the path to the SQLite DB for the given mode.

    Args:
        mode: Either 'seed' or 'full'.

    Returns:
        Absolute path to the DB file.
    """
    filename: str = "seed.db" if mode == "seed" else "full.db"
    return DATA_DIR / filename


def _load_model() -> Any:
    """Load the sentence-transformers embedding model.

    Imported lazily so the module import does not trigger the ~700 MB torch
    download at test collection time.

    Returns:
        A ``SentenceTransformer`` instance ready for ``.encode``.
    """
    from sentence_transformers import SentenceTransformer  # noqa: PLC0415

    logger.info("Loading embedding model %s ...", EMBEDDING_MODEL_NAME)
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def _get_or_create_collection(rebuild: bool) -> Any:
    """Open (or recreate) the ChromaDB collection at ``CHROMA_DIR``.

    Args:
        rebuild: If True, delete and recreate the collection.

    Returns:
        A ChromaDB collection handle.
    """
    import chromadb  # noqa: PLC0415

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    if rebuild:
        try:
            client.delete_collection(COLLECTION_NAME)
            logger.info("Dropped existing collection %s", COLLECTION_NAME)
        except Exception:
            # Collection may not exist yet on first run
            pass

    return client.get_or_create_collection(name=COLLECTION_NAME)


def build_index(
    mode: str = "seed",
    rebuild: bool = False,
) -> dict[str, int]:
    """Build the ChromaDB index from reference_docs rows.

    Args:
        mode: Either 'seed' or 'full'.
        rebuild: If True, drop and recreate the collection before indexing.

    Returns:
        Dict with keys 'docs' and 'chunks' giving input doc count and emitted
        chunk count.

    Raises:
        FileNotFoundError: If the target DB file does not exist.
    """
    db_path: Path = _db_path(mode)
    if not db_path.exists():
        raise FileNotFoundError(
            f"Database not found at {db_path}. Run db_setup.py first."
        )

    conn: sqlite3.Connection = sqlite3.connect(db_path)
    try:
        docs: list[dict[str, Any]] = _fetch_reference_rows(conn)
    finally:
        conn.close()

    if not docs:
        logger.warning("reference_docs is empty — nothing to embed")
        return {"docs": 0, "chunks": 0}

    # Chunk all docs first so we can batch-embed for speed
    chunk_records: list[dict[str, Any]] = []
    for doc in docs:
        chunks: list[str] = chunk_document(doc["content"])
        for idx, chunk in enumerate(chunks):
            chunk_records.append(
                {
                    "doc_id": doc["id"],
                    "series_id": doc["series_id"],
                    "doc_type": doc["doc_type"],
                    "title": doc["title"],
                    "source_url": doc["source_url"],
                    "chunk_index": idx,
                    "text": chunk,
                }
            )

    logger.info(
        "Chunked %d reference docs into %d chunks.",
        len(docs),
        len(chunk_records),
    )

    model: Any = _load_model()
    vectors: list[list[float]] = embed_chunks(
        [r["text"] for r in chunk_records], model
    )
    logger.info(
        "Embedded %d chunks using %s (%d dim).",
        len(vectors),
        EMBEDDING_MODEL_NAME,
        len(vectors[0]) if vectors else 0,
    )

    collection = _get_or_create_collection(rebuild=rebuild)

    # Upsert every chunk. Chroma expects str ids.
    ids: list[str] = [
        f"{r['doc_id']}::{r['chunk_index']}" for r in chunk_records
    ]
    metadatas: list[dict[str, Any]] = [
        {
            "doc_id": r["doc_id"],
            "series_id": r["series_id"],
            "doc_type": r["doc_type"],
            "title": r["title"],
            "source_url": r["source_url"],
            "chunk_index": r["chunk_index"],
        }
        for r in chunk_records
    ]
    documents: list[str] = [r["text"] for r in chunk_records]

    collection.upsert(
        ids=ids,
        embeddings=vectors,
        metadatas=metadatas,
        documents=documents,
    )

    try:
        display_path: str = str(CHROMA_DIR.relative_to(PROJECT_ROOT))
    except ValueError:
        display_path = str(CHROMA_DIR)
    logger.info(
        "Persisted to %s (collection: %s).",
        display_path,
        COLLECTION_NAME,
    )

    return {"docs": len(docs), "chunks": len(chunk_records)}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments.

    Args:
        argv: Optional argv list for testing.

    Returns:
        Parsed namespace with ``db`` and ``rebuild`` attributes.
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description=(
            "Chunk and embed reference_docs into a persistent ChromaDB store."
        ),
    )
    parser.add_argument(
        "--db",
        choices=("seed", "full"),
        default="seed",
        help="Which database to index (default: seed).",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Drop and recreate the collection before indexing.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Optional argv list for tests.

    Returns:
        Process exit code (0 on success).
    """
    args: argparse.Namespace = _parse_args(argv)
    try:
        build_index(mode=args.db, rebuild=args.rebuild)
    except FileNotFoundError as e:
        logger.error("%s", e)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
