"""Hacker News ingestion via the Algolia HN Search API.

Pulls story hits matching the AI-labor-impact keyword themes defined in
``hackernews_config.SEARCH_QUERIES``, deduplicates by story id, buckets by
month, keeps the top ``MONTHLY_STORY_LIMIT`` stories per month by points,
and atomically writes the result to ``data/raw/hn_stories.json``.

The endpoint is public and unauthenticated. Default behavior is
idempotent: if the cache file exists and ``--refresh`` was not passed,
the script logs and exits without making HTTP calls. Empty pulls
(zero matching hits across every query) still write ``[]`` to the cache;
only an all-queries-failed case exits non-zero.

See ``docs/private_docs/claude_phase_13.md`` for the full design.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import httpx

from hackernews_config import (
    ALGOLIA_BASE_URL,
    ALGOLIA_HITS_PER_PAGE,
    ALGOLIA_REQUEST_DELAY,
    COVERAGE_START_ISO,
    COVERAGE_START_UNIX,
    MONTHLY_STORY_LIMIT,
    SEARCH_QUERIES,
    STORY_TEXT_CHAR_LIMIT,
)

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
RAW_DIR: Path = PROJECT_ROOT / "data" / "raw"
CACHE_PATH: Path = RAW_DIR / "hn_stories.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger: logging.Logger = logging.getLogger(__name__)

MAX_PAGES_PER_QUERY: int = 10
"""Safety cap on pagination per query.

Algolia caps ``hitsPerPage`` at 1,000; at 10 pages we have a
10,000-hit ceiling per query, well above any reasonable thematic match
count. A query that reports more pages is too broad and should be
tightened in config rather than silently truncated without logging.
"""


# ---------------------------------------------------------------------------
# Text normalization helpers
# ---------------------------------------------------------------------------


_SMART_PUNCT_MAP: dict[str, str] = {
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2013": "-",
    "\u2014": "-",
    "\u2026": "...",
    "\u00a0": " ",
}
"""Smart-punctuation substitution table.

Same rule Phase 12 enforced on scholarly content: Phase 14's eventual
RAG embedding + citation verification is brittle on smart punctuation,
so we normalize at ingestion.
"""


def truncate_and_normalize(text: str, limit: int) -> str:
    """Strip smart punctuation, collapse whitespace, enforce char cap.

    Runs Unicode NFKC normalization, replaces curly quotes / en / em
    dashes / ellipsis / non-breaking space with ASCII equivalents,
    collapses runs of whitespace to single spaces, and truncates to
    ``limit`` characters on a whitespace boundary where possible.

    Args:
        text: Raw text from an Algolia hit.
        limit: Maximum characters to keep.

    Returns:
        Normalized, length-capped ASCII-ish text. Returns the empty
        string when the input is empty or only whitespace.
    """
    if not text:
        return ""
    normalized: str = unicodedata.normalize("NFKC", text)
    for bad, good in _SMART_PUNCT_MAP.items():
        normalized = normalized.replace(bad, good)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if len(normalized) <= limit:
        return normalized
    # Prefer a whitespace boundary inside the cap window.
    candidate: str = normalized[:limit]
    last_space: int = candidate.rfind(" ")
    if last_space >= limit // 2:
        return candidate[:last_space].rstrip()
    return candidate.rstrip()


# ---------------------------------------------------------------------------
# Pure per-hit transforms
# ---------------------------------------------------------------------------


def build_story_dict(hit: dict, query: str) -> dict | None:
    """Convert one Algolia hit into our per-story dict.

    Args:
        hit: One element of the ``hits`` array returned by Algolia.
        query: The configured query string that surfaced this hit.

    Returns:
        A per-story dict matching the raw JSON schema, or ``None`` if
        the hit is malformed (missing title, non-integer ``objectID``,
        missing ``created_at_i``, etc.).
    """
    raw_object_id = hit.get("objectID")
    if raw_object_id is None:
        return None
    try:
        story_id: int = int(str(raw_object_id))
    except (TypeError, ValueError):
        logger.warning(
            "Skipping hit: objectID is not an integer (%r)", raw_object_id
        )
        return None

    created_at_i = hit.get("created_at_i")
    if created_at_i is None:
        return None
    try:
        created_unix: int = int(created_at_i)
    except (TypeError, ValueError):
        return None
    created_utc: str = datetime.fromtimestamp(
        created_unix, tz=timezone.utc
    ).isoformat()

    raw_title = hit.get("title")
    title_source: str = str(raw_title) if raw_title is not None else ""
    title: str = truncate_and_normalize(title_source, 500)
    if not title:
        return None

    raw_text = hit.get("story_text") or ""
    text_excerpt: str = truncate_and_normalize(
        str(raw_text), STORY_TEXT_CHAR_LIMIT
    )

    score: int = int(hit.get("points") or 0)
    num_comments: int = int(hit.get("num_comments") or 0)

    url_raw = hit.get("url")
    url: str | None = url_raw if url_raw else None

    hn_permalink: str = (
        f"https://news.ycombinator.com/item?id={story_id}"
    )

    return {
        "story_id": story_id,
        "created_utc": created_utc,
        "title": title,
        "text_excerpt": text_excerpt,
        "score": score,
        "num_comments": num_comments,
        "url": url,
        "hn_permalink": hn_permalink,
        "matched_queries": [query],
    }


def dedupe_by_story_id(stories: list[dict]) -> list[dict]:
    """Union-merge stories by ``story_id``, combining matched queries.

    When the same story appears from N different query results, keep
    the first occurrence's fields and replace ``matched_queries`` with
    the ordered union of all sources' ``matched_queries``.

    Args:
        stories: List of per-story dicts from ``build_story_dict``,
            each carrying a ``matched_queries`` list of length 1 (or
            more, to be tolerant).

    Returns:
        Deduplicated list in first-seen order.
    """
    by_id: dict[int, dict] = {}
    order: list[int] = []
    for story in stories:
        sid: int = int(story["story_id"])
        if sid not in by_id:
            merged: dict = dict(story)
            merged["matched_queries"] = list(story.get("matched_queries", []))
            by_id[sid] = merged
            order.append(sid)
        else:
            existing: dict = by_id[sid]
            for q in story.get("matched_queries", []):
                if q not in existing["matched_queries"]:
                    existing["matched_queries"].append(q)
    return [by_id[sid] for sid in order]


def bucket_by_month(
    stories: list[dict],
    start_iso: str,
) -> dict[str, list[dict]]:
    """Group stories by ``YYYY-MM`` bucket, dropping pre-window entries.

    Inclusivity: a story whose ``created_utc`` date portion is greater
    than or equal to ``start_iso`` is kept. Matches Algolia's
    ``numericFilters=created_at_i>=...`` convention so the window
    boundary is consistent across the pipeline.

    Args:
        stories: List of per-story dicts.
        start_iso: Earliest ``YYYY-MM-DD`` to keep (inclusive).

    Returns:
        Mapping from ``YYYY-MM`` month key to list of stories in that
        month (unsorted within a bucket).
    """
    buckets: dict[str, list[dict]] = {}
    for story in stories:
        created_utc: str = str(story["created_utc"])
        # The first 10 chars are always YYYY-MM-DD for ISO-8601 UTC.
        created_date: str = created_utc[:10]
        if created_date < start_iso:
            continue
        month_key: str = created_utc[:7]  # YYYY-MM
        buckets.setdefault(month_key, []).append(story)
    return buckets


def cap_monthly_top_n(
    buckets: dict[str, list[dict]],
    limit: int,
) -> list[dict]:
    """Flatten monthly buckets keeping the top-N stories per month.

    Within each bucket, stories are sorted by ``score`` descending with
    ``story_id`` ascending as a stable tiebreaker. The final list is
    sorted by month ascending, then by score descending within each
    month.

    Args:
        buckets: Mapping from ``YYYY-MM`` to list of per-story dicts.
        limit: Maximum stories to retain per bucket.

    Returns:
        Flat list of kept stories in (month asc, score desc) order.
    """
    result: list[dict] = []
    for month in sorted(buckets.keys()):
        month_stories: list[dict] = list(buckets[month])
        month_stories.sort(
            key=lambda s: (-int(s.get("score") or 0), int(s["story_id"]))
        )
        result.extend(month_stories[:limit])
    return result


# ---------------------------------------------------------------------------
# Network layer
# ---------------------------------------------------------------------------


def fetch_query_page(
    client: httpx.Client,
    query: str,
    page: int = 0,
) -> dict:
    """GET one Algolia ``search_by_date`` page and return parsed JSON.

    Args:
        client: An open ``httpx.Client``.
        query: Configured query string. Must be non-empty after strip.
        page: Zero-indexed Algolia page number.

    Returns:
        The parsed JSON payload (dict with ``hits``, ``nbPages``, etc.).

    Raises:
        ValueError: If ``query`` is empty or whitespace only.
        httpx.HTTPStatusError: On non-2xx responses.
        httpx.HTTPError: On transport-level failures.
    """
    if not query.strip():
        raise ValueError(
            "Empty query would match all hits and blow past the safety cap."
        )

    params: dict[str, str | int] = {
        "query": query,
        "tags": "story",
        "numericFilters": f"created_at_i>={COVERAGE_START_UNIX}",
        "hitsPerPage": ALGOLIA_HITS_PER_PAGE,
        "page": page,
    }
    response: httpx.Response = client.get(
        ALGOLIA_BASE_URL, params=params, timeout=20.0
    )
    response.raise_for_status()
    payload: dict = response.json()
    return payload


def _collect_query(
    client: httpx.Client,
    query: str,
) -> tuple[list[dict], bool]:
    """Pull every page for one query, up to the page-cap.

    Args:
        client: An open ``httpx.Client``.
        query: Configured query string.

    Returns:
        Tuple of (list of per-story dicts, ``ok`` flag). ``ok`` is
        ``False`` if all pages for this query failed — used by
        ``pull_all`` to detect the all-queries-failed case.
    """
    collected: list[dict] = []
    any_success: bool = False
    page: int = 0
    while page < MAX_PAGES_PER_QUERY:
        try:
            payload: dict = fetch_query_page(client, query, page=page)
        except httpx.TimeoutException:
            logger.warning(
                "Timeout on query=%r page=%d - single retry", query, page
            )
            time.sleep(ALGOLIA_REQUEST_DELAY * 2)
            try:
                payload = fetch_query_page(client, query, page=page)
            except Exception as exc:
                logger.warning(
                    "Retry failed for query=%r page=%d: %s",
                    query,
                    page,
                    exc,
                )
                break
        except httpx.HTTPStatusError as exc:
            status: int = exc.response.status_code
            if status == 429:
                logger.warning(
                    "Rate limited on query=%r - sleeping 60s", query
                )
                time.sleep(60.0)
                try:
                    payload = fetch_query_page(client, query, page=page)
                except Exception as retry_exc:
                    logger.warning(
                        "Retry after 429 failed for query=%r page=%d: %s",
                        query,
                        page,
                        retry_exc,
                    )
                    break
            else:
                logger.warning(
                    "HTTP %d on query=%r page=%d - skipping query",
                    status,
                    query,
                    page,
                )
                break
        except httpx.HTTPError as exc:
            logger.warning(
                "HTTP error on query=%r page=%d: %s - skipping query",
                query,
                page,
                exc,
            )
            break

        any_success = True
        hits = payload.get("hits")
        nb_pages = payload.get("nbPages")
        if hits is None or nb_pages is None:
            logger.warning(
                "Malformed Algolia response on query=%r page=%d",
                query,
                page,
            )
            break

        for hit in hits:
            story: dict | None = build_story_dict(hit, query)
            if story is not None:
                collected.append(story)

        if page + 1 >= int(nb_pages):
            break
        page += 1
        time.sleep(ALGOLIA_REQUEST_DELAY)

    # If Algolia reported more pages than we will fetch, log once.
    if page >= MAX_PAGES_PER_QUERY - 1 and payload.get("nbPages", 0) > MAX_PAGES_PER_QUERY:
        logger.warning(
            "Query %r reported %d pages; truncated at %d. Tighten the "
            "keyword in hackernews_config.SEARCH_QUERIES.",
            query,
            payload.get("nbPages", 0),
            MAX_PAGES_PER_QUERY,
        )

    return collected, any_success


def _atomic_write_json(path: Path, payload: list[dict]) -> None:
    """Write JSON atomically via a sibling ``.tmp`` file and rename.

    Args:
        path: Target output path.
        payload: JSON-serializable content.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2))
    tmp_path.replace(path)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def pull_all(refresh: bool = False) -> dict[str, int]:
    """Run every configured query and cache the curated result.

    If the cache file exists and ``refresh`` is False, logs and returns
    early without making HTTP calls. On a live pull, iterates every
    ``SEARCH_QUERIES`` entry, deduplicates by ``story_id``, buckets by
    month, keeps the top ``MONTHLY_STORY_LIMIT`` per bucket, and writes
    the result atomically to ``CACHE_PATH``.

    Args:
        refresh: If True, re-pull even if a cached file exists.

    Returns:
        Stats dict with keys ``queries_run``, ``queries_failed``,
        ``hits_total``, ``unique_stories``, ``kept_after_monthly_cap``.

    Raises:
        SystemExit: If every query failed (status code 1).
    """
    if CACHE_PATH.exists() and not refresh:
        logger.info(
            "hn_stories.json cached - skipping Algolia calls "
            "(use --refresh to re-pull)"
        )
        return {
            "queries_run": 0,
            "queries_failed": 0,
            "hits_total": 0,
            "unique_stories": 0,
            "kept_after_monthly_cap": 0,
        }

    all_hits: list[dict] = []
    queries_failed: int = 0
    queries_run: int = 0

    with httpx.Client() as client:
        for i, query in enumerate(SEARCH_QUERIES, start=1):
            queries_run += 1
            logger.info(
                "[%d/%d] Pulling HN stories for query=%r",
                i,
                len(SEARCH_QUERIES),
                query,
            )
            hits, ok = _collect_query(client, query)
            all_hits.extend(hits)
            if not ok:
                queries_failed += 1
            if i < len(SEARCH_QUERIES):
                time.sleep(ALGOLIA_REQUEST_DELAY)

    if queries_run > 0 and queries_failed == queries_run:
        logger.error(
            "Every Algolia query failed (%d/%d). Algolia unreachable.",
            queries_failed,
            queries_run,
        )
        sys.exit(1)

    unique_stories: list[dict] = dedupe_by_story_id(all_hits)
    buckets: dict[str, list[dict]] = bucket_by_month(
        unique_stories, COVERAGE_START_ISO
    )
    kept: list[dict] = cap_monthly_top_n(buckets, MONTHLY_STORY_LIMIT)

    _atomic_write_json(CACHE_PATH, kept)

    stats: dict[str, int] = {
        "queries_run": queries_run,
        "queries_failed": queries_failed,
        "hits_total": len(all_hits),
        "unique_stories": len(unique_stories),
        "kept_after_monthly_cap": len(kept),
    }
    logger.info("HN pull stats: %s", stats)
    return stats


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description=(
            "Pull Hacker News stories via the Algolia HN search API "
            "and cache as JSON."
        ),
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help=(
            "Re-pull even if data/raw/hn_stories.json exists. "
            "Default: skip the live pull when the cache is present."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for the Hacker News ingestion script."""
    args: argparse.Namespace = _parse_args()
    pull_all(refresh=args.refresh)


if __name__ == "__main__":
    main()
