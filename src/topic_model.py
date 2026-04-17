"""NMF topic model pipeline for Hacker News stories.

Reads HN story titles and excerpts from ``hn_stories``, fits a
TF-IDF + NMF topic model via sklearn, and writes results to
``hn_topics``, ``hn_topic_assignments``, and ``hn_ngram_monthly``.

Usage:
    .venv/bin/python src/topic_model.py [--db seed|full] [--n-topics 8]
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
from pathlib import Path
from typing import Any

from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger: logging.Logger = logging.getLogger(__name__)

# After inspecting NMF output for n_topics in {6, 7, 8, 9, 10} on the
# 1,547-story seed corpus, n_topics=8 produces the most interpretable,
# non-overlapping clusters. Values 6-7 merge distinct AI-automation and
# layoff themes; 9-10 split career-advice into near-duplicate fragments.
DEFAULT_N_TOPICS: int = 8

TOPIC_LABEL_OVERRIDES: dict[int, str] = {
    0: "Software Engineering Careers",
    1: "AI Automation & Job Displacement",
    2: "Severance & Layoff Experiences",
    3: "Tech Layoffs & Market Conditions",
    4: "Hiring Freezes",
    5: "Executive Firings & Restructuring",
    6: "Remote Work & Job Search",
    7: "Mass Layoffs at Big Tech",
}
"""Human-readable overrides keyed by topic_id.

Labels assigned after inspecting NMF output on 1,547-story seed corpus
with n_topics=8. Auto-labels (top-3 TF-IDF terms joined by ' / ') are
the fallback; these overrides produce cleaner dashboard display.
"""

# HTML entity fragments and URL tokens that leak through the HN
# text-excerpt HTML stripping. Appended to sklearn's English stop list
# so they don't dominate TF-IDF components.
_EXTRA_STOP_WORDS: list[str] = [
    # HTML entity fragments leaking through excerpt stripping
    "x27", "x2f", "quot", "href", "https", "http", "www", "com",
    "amp", "gt", "lt", "ve", "pre", "don", "didn", "doesn", "isn",
    "wasn", "weren", "wouldn", "couldn", "shouldn", "hasn", "haven",
    "ll", "re",
    # HN platform artifacts
    "hn", "ask", "nofollow", "rel", "built", "org", "html", "io",
    # Non-AI proper nouns: people and companies that create personality-
    # or brand-driven topics instead of labor-theme-driven ones. The
    # thesis cares about *what* happened (firing, freeze, automation)
    # not *who*. AI companies/CEOs (OpenAI, Altman) are kept because
    # they are thesis-relevant signal.
    "musk", "musk's", "elon", "zuckerberg", "bezos",
    "twitter", "ex", "tesla", "spacex",
    "meta", "facebook",
]


def _load_documents(conn: sqlite3.Connection) -> list[tuple[int, str]]:
    """Load story documents from ``hn_stories``.

    Concatenates ``title`` and ``text_excerpt`` (stripped) for each row.

    Args:
        conn: Open SQLite connection.

    Returns:
        List of ``(story_id, text)`` tuples.
    """
    rows: list[tuple[int, str, str]] = conn.execute(
        "SELECT story_id, title, text_excerpt FROM hn_stories"
    ).fetchall()
    docs: list[tuple[int, str]] = []
    for story_id, title, excerpt in rows:
        text: str = f"{(title or '').strip()} {(excerpt or '').strip()}".strip()
        if text:
            docs.append((story_id, text))
    logger.info("Loaded %d documents from hn_stories", len(docs))
    return docs


def _build_tfidf(
    texts: list[str],
    max_features: int,
    min_df: int,
    max_df: float,
) -> tuple[Any, TfidfVectorizer]:
    """Fit a TF-IDF vectorizer on the corpus.

    Args:
        texts: List of document strings.
        max_features: Maximum vocabulary size.
        min_df: Minimum document frequency for a term.
        max_df: Maximum document frequency fraction for a term.

    Returns:
        Tuple of ``(tfidf_matrix, vectorizer)``.
    """
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    stop_words: list[str] = list(ENGLISH_STOP_WORDS) + _EXTRA_STOP_WORDS
    vectorizer: TfidfVectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        stop_words=stop_words,
    )
    tfidf_matrix = vectorizer.fit_transform(texts)
    logger.info(
        "TF-IDF matrix: %d docs x %d terms",
        tfidf_matrix.shape[0],
        tfidf_matrix.shape[1],
    )
    return tfidf_matrix, vectorizer


def _fit_nmf(tfidf_matrix: Any, n_topics: int) -> NMF:
    """Fit an NMF model on the TF-IDF matrix.

    Args:
        tfidf_matrix: Sparse TF-IDF matrix from ``_build_tfidf``.
        n_topics: Number of topics to extract.

    Returns:
        Fitted NMF model.
    """
    model: NMF = NMF(
        n_components=n_topics,
        random_state=42,
        max_iter=300,
    )
    model.fit(tfidf_matrix)
    logger.info("NMF fitted with %d topics", n_topics)
    return model


def _extract_topic_labels(
    model: NMF,
    vectorizer: TfidfVectorizer,
    n_top_terms: int = 5,
) -> list[tuple[str, list[str]]]:
    """Extract topic labels from the NMF model.

    For each component, extracts the top terms by weight. The auto-label
    is the top-3 terms joined by ' / '. Entries in ``TOPIC_LABEL_OVERRIDES``
    replace the auto-label for dashboard display.

    Args:
        model: Fitted NMF model.
        vectorizer: Fitted TF-IDF vectorizer.
        n_top_terms: Number of top terms to extract per topic.

    Returns:
        List of ``(label, top_terms_list)`` tuples, one per topic.
    """
    feature_names: list[str] = vectorizer.get_feature_names_out().tolist()
    labels: list[tuple[str, list[str]]] = []

    for topic_id, component in enumerate(model.components_):
        top_indices = component.argsort()[-n_top_terms:][::-1]
        top_terms: list[str] = [feature_names[i] for i in top_indices]
        auto_label: str = " / ".join(top_terms[:3])
        label: str = TOPIC_LABEL_OVERRIDES.get(topic_id, auto_label)
        labels.append((label, top_terms))

    return labels


def _assign_topics(
    nmf_model: NMF,
    tfidf_matrix: Any,
    story_ids: list[int],
) -> list[tuple[int, int, float]]:
    """Assign each document to its dominant topic.

    Uses argmax of the NMF coefficient row for hard assignment.

    Args:
        nmf_model: Fitted NMF model.
        tfidf_matrix: Sparse TF-IDF matrix.
        story_ids: Parallel list of story IDs matching matrix rows.

    Returns:
        List of ``(story_id, topic_id, score)`` tuples.
    """
    doc_topic_matrix = nmf_model.transform(tfidf_matrix)
    assignments: list[tuple[int, int, float]] = []

    for i, story_id in enumerate(story_ids):
        row = doc_topic_matrix[i]
        topic_id: int = int(row.argmax())
        score: float = float(row[topic_id])
        assignments.append((story_id, topic_id, score))

    return assignments


def _write_results(
    conn: sqlite3.Connection,
    topics: list[tuple[str, list[str]]],
    assignments: list[tuple[int, int, float]],
) -> None:
    """Write topic model results to the database.

    Clears both target tables and re-inserts in a single transaction.

    Args:
        conn: Open SQLite connection.
        topics: List of ``(label, top_terms_list)`` from ``_extract_topic_labels``.
        assignments: List of ``(story_id, topic_id, score)`` from ``_assign_topics``.
    """
    conn.execute("DELETE FROM hn_topic_assignments")
    conn.execute("DELETE FROM hn_topics")

    # Count stories per topic
    topic_counts: dict[int, int] = {}
    for _, topic_id, _ in assignments:
        topic_counts[topic_id] = topic_counts.get(topic_id, 0) + 1

    for topic_id, (label, top_terms) in enumerate(topics):
        conn.execute(
            """
            INSERT INTO hn_topics (topic_id, label, top_terms, story_count)
            VALUES (?, ?, ?, ?)
            """,
            (topic_id, label, json.dumps(top_terms), topic_counts.get(topic_id, 0)),
        )

    for story_id, topic_id, score in assignments:
        conn.execute(
            """
            INSERT INTO hn_topic_assignments (story_id, topic_id, score)
            VALUES (?, ?, ?)
            """,
            (story_id, topic_id, score),
        )

    conn.commit()
    logger.info(
        "Wrote %d topics and %d assignments", len(topics), len(assignments)
    )


def _compute_ngrams(conn: sqlite3.Connection, top_n: int = 10) -> int:
    """Compute monthly bigram frequencies from HN story titles.

    Per month, fits a ``CountVectorizer`` with ``ngram_range=(2, 2)``
    on that month's titles and keeps the top-N bigrams by count.

    Args:
        conn: Open SQLite connection with ``hn_stories`` populated.
        top_n: Number of top bigrams to keep per month.

    Returns:
        Total number of rows inserted into ``hn_ngram_monthly``.
    """
    conn.execute("DELETE FROM hn_ngram_monthly")

    months: list[tuple[str,]] = conn.execute(
        "SELECT DISTINCT month FROM hn_stories ORDER BY month"
    ).fetchall()

    total_inserted: int = 0

    for (month,) in months:
        titles: list[tuple[str,]] = conn.execute(
            "SELECT title FROM hn_stories WHERE month = ?", (month,)
        ).fetchall()
        texts: list[str] = [t[0] for t in titles if t[0]]

        if len(texts) < 2:
            continue

        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

        stop_words_ngram: list[str] = list(ENGLISH_STOP_WORDS) + _EXTRA_STOP_WORDS
        cv: CountVectorizer = CountVectorizer(
            ngram_range=(2, 2),
            stop_words=stop_words_ngram,
            min_df=1,
        )
        try:
            matrix = cv.fit_transform(texts)
        except ValueError:
            # Empty vocabulary after stop-word removal
            continue

        feature_names: list[str] = cv.get_feature_names_out().tolist()
        counts = matrix.sum(axis=0).A1

        # Sort by count descending, keep top_n
        ranked = sorted(
            zip(feature_names, counts), key=lambda x: x[1], reverse=True
        )[:top_n]

        for ngram, count in ranked:
            conn.execute(
                """
                INSERT OR REPLACE INTO hn_ngram_monthly (month, ngram, count)
                VALUES (?, ?, ?)
                """,
                (month, ngram, int(count)),
            )
            total_inserted += 1

    conn.commit()
    logger.info("Wrote %d ngram rows across %d months", total_inserted, len(months))
    return total_inserted


def fit_topics(
    db_path: Path,
    *,
    n_topics: int = DEFAULT_N_TOPICS,
    max_features: int = 2000,
    min_df: int = 3,
    max_df: float = 0.85,
) -> None:
    """Fit NMF topic model on hn_stories and write results to DB.

    Runs the full pipeline: load documents, build TF-IDF, fit NMF,
    extract labels, assign topics, write results, and compute ngrams.

    Args:
        db_path: Path to the SQLite database.
        n_topics: Number of topics to extract.
        max_features: Maximum TF-IDF vocabulary size.
        min_df: Minimum document frequency for a term.
        max_df: Maximum document frequency fraction for a term.
    """
    # Ensure schema exists
    schema_path: Path = Path(__file__).resolve().parent.parent / "sql" / "08_topic_schema.sql"
    conn: sqlite3.Connection = sqlite3.connect(db_path)

    try:
        if schema_path.exists():
            conn.executescript(schema_path.read_text())

        docs: list[tuple[int, str]] = _load_documents(conn)
        if not docs:
            logger.warning("No documents found in hn_stories — nothing to fit")
            return

        story_ids: list[int] = [d[0] for d in docs]
        texts: list[str] = [d[1] for d in docs]

        tfidf_matrix, vectorizer = _build_tfidf(texts, max_features, min_df, max_df)
        nmf_model: NMF = _fit_nmf(tfidf_matrix, n_topics)
        topics: list[tuple[str, list[str]]] = _extract_topic_labels(
            nmf_model, vectorizer
        )
        assignments: list[tuple[int, int, float]] = _assign_topics(
            nmf_model, tfidf_matrix, story_ids
        )

        _write_results(conn, topics, assignments)
        _compute_ngrams(conn)

        # Log topic summary
        for tid, (label, top_terms) in enumerate(topics):
            count: int = sum(1 for _, t, _ in assignments if t == tid)
            logger.info(
                "  Topic %d: %-40s  (%d stories)  terms=%s",
                tid,
                label,
                count,
                top_terms,
            )

    finally:
        conn.close()

    logger.info("Topic model pipeline complete for %s", db_path.name)


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Fit NMF topic model on HN stories.",
    )
    parser.add_argument(
        "--db",
        choices=["seed", "full"],
        default="seed",
        help="Which database to use (default: seed).",
    )
    parser.add_argument(
        "--n-topics",
        type=int,
        default=DEFAULT_N_TOPICS,
        help=f"Number of topics (default: {DEFAULT_N_TOPICS}).",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for the topic model script."""
    args: argparse.Namespace = _parse_args()
    db_filename: str = "seed.db" if args.db == "seed" else "full.db"
    db_path: Path = DATA_DIR / db_filename
    fit_topics(db_path, n_topics=args.n_topics)


if __name__ == "__main__":
    main()
