"""Tests for src/hackernews_pull.py.

All HTTP interactions are mocked via ``httpx.MockTransport`` or
``monkeypatch`` of ``httpx.Client.get``. No live Algolia calls.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable
from unittest.mock import MagicMock, patch

import httpx
import pytest

import hackernews_pull as pull


_REAL_HTTPX_CLIENT = httpx.Client
"""Reference to the unpatched ``httpx.Client`` class.

``_install_mock_client`` monkeypatches ``pull.httpx.Client``, so building
the returned client through ``httpx.Client(...)`` would recurse into the
patched factory. Capturing the class at import time avoids that loop.
"""


def _install_mock_client(
    monkeypatch: pytest.MonkeyPatch,
    handler: Callable[[httpx.Request], httpx.Response],
) -> None:
    """Monkeypatch ``pull.httpx.Client`` to a factory backed by MockTransport.

    Returns a real ``httpx.Client`` (so ``with`` blocks work), but every
    call goes through the handler instead of the network.
    """

    def factory(*_args, **_kwargs):
        return _REAL_HTTPX_CLIENT(transport=httpx.MockTransport(handler))

    monkeypatch.setattr(pull.httpx, "Client", factory)


# ---------------------------------------------------------------------------
# build_story_dict
# ---------------------------------------------------------------------------


def _base_hit() -> dict:
    """Build a minimal well-formed Algolia hit fixture."""
    return {
        "objectID": "12345678",
        "created_at_i": 1678000000,  # 2023-03-05 08:26:40 UTC
        "title": "Big Co announces layoffs",
        "story_text": "Some self-post text.",
        "points": 423,
        "num_comments": 187,
        "url": "https://example.com/article",
    }


class TestBuildStoryDict:
    """Tests for ``build_story_dict``."""

    def test_from_algolia_hit(self) -> None:
        """Canned Algolia hit maps to the documented per-story schema."""
        hit: dict = _base_hit()
        story: dict | None = pull.build_story_dict(hit, "layoffs")
        assert story is not None
        assert story["story_id"] == 12345678
        assert story["title"] == "Big Co announces layoffs"
        assert story["text_excerpt"] == "Some self-post text."
        assert story["score"] == 423
        assert story["num_comments"] == 187
        assert story["url"] == "https://example.com/article"
        assert story["hn_permalink"] == (
            "https://news.ycombinator.com/item?id=12345678"
        )
        assert story["matched_queries"] == ["layoffs"]
        # created_utc should be a valid ISO timestamp starting with date
        assert story["created_utc"].startswith("2023-03-05")

    def test_returns_none_on_missing_title(self) -> None:
        """A hit with no title is dropped."""
        hit: dict = _base_hit()
        hit["title"] = None
        assert pull.build_story_dict(hit, "layoffs") is None

    def test_returns_none_on_empty_title_after_normalize(self) -> None:
        """A whitespace-only title normalizes to empty and is dropped."""
        hit: dict = _base_hit()
        hit["title"] = "   "
        assert pull.build_story_dict(hit, "layoffs") is None

    def test_returns_none_on_non_int_story_id(self) -> None:
        """A non-integer objectID is dropped with a warning."""
        hit: dict = _base_hit()
        hit["objectID"] = "abc-not-an-int"
        assert pull.build_story_dict(hit, "layoffs") is None

    def test_stores_url_as_none_for_self_post(self) -> None:
        """Self-post hits (url=null) are stored with url=None."""
        hit: dict = _base_hit()
        hit["url"] = None
        story: dict | None = pull.build_story_dict(hit, "layoffs")
        assert story is not None
        assert story["url"] is None

    def test_normalizes_smart_punctuation(self) -> None:
        """Smart quotes / em-dashes in title and excerpt become ASCII."""
        hit: dict = _base_hit()
        hit["title"] = "\u201cQuoted\u201d \u2014 em-dash"
        hit["story_text"] = "\u2018single\u2019 \u2013 en-dash \u2026"
        story: dict | None = pull.build_story_dict(hit, "layoffs")
        assert story is not None
        assert story["title"] == '"Quoted" - em-dash'
        assert story["text_excerpt"] == "'single' - en-dash ..."

    def test_missing_created_at_returns_none(self) -> None:
        """A hit without ``created_at_i`` is dropped."""
        hit: dict = _base_hit()
        hit.pop("created_at_i")
        assert pull.build_story_dict(hit, "layoffs") is None

    def test_missing_points_defaults_to_zero(self) -> None:
        """A hit without ``points`` scores 0 rather than crashing."""
        hit: dict = _base_hit()
        hit.pop("points")
        story: dict | None = pull.build_story_dict(hit, "layoffs")
        assert story is not None
        assert story["score"] == 0


# ---------------------------------------------------------------------------
# dedupe_by_story_id
# ---------------------------------------------------------------------------


class TestDedupeByStoryId:
    """Tests for ``dedupe_by_story_id``."""

    def test_union_merges_matched_queries(self) -> None:
        """A story surfaced by two queries merges into one row."""
        s1: dict = {
            "story_id": 1,
            "title": "A",
            "score": 10,
            "matched_queries": ["layoffs"],
            "created_utc": "2023-01-15T00:00:00+00:00",
        }
        s2: dict = {
            "story_id": 1,
            "title": "A",
            "score": 10,
            "matched_queries": ["severance"],
            "created_utc": "2023-01-15T00:00:00+00:00",
        }
        result: list[dict] = pull.dedupe_by_story_id([s1, s2])
        assert len(result) == 1
        assert result[0]["matched_queries"] == ["layoffs", "severance"]

    def test_preserves_first_seen_order(self) -> None:
        """Ordering matches first-seen, not merged-id order."""
        a: dict = {"story_id": 2, "matched_queries": ["q1"]}
        b: dict = {"story_id": 1, "matched_queries": ["q2"]}
        c: dict = {"story_id": 2, "matched_queries": ["q3"]}
        result: list[dict] = pull.dedupe_by_story_id([a, b, c])
        ids: list[int] = [s["story_id"] for s in result]
        assert ids == [2, 1]
        # story 2 picks up both matched queries, in order
        assert result[0]["matched_queries"] == ["q1", "q3"]

    def test_does_not_mutate_input(self) -> None:
        """Callers keep their original dicts untouched."""
        s: dict = {"story_id": 1, "matched_queries": ["layoffs"]}
        pull.dedupe_by_story_id([s, s])
        # Input list item still has its original single-query list
        assert s["matched_queries"] == ["layoffs"]


# ---------------------------------------------------------------------------
# bucket_by_month
# ---------------------------------------------------------------------------


class TestBucketByMonth:
    """Tests for ``bucket_by_month``."""

    def test_filters_pre_window(self) -> None:
        """Stories before ``start_iso`` are dropped."""
        stories: list[dict] = [
            {"story_id": 1, "created_utc": "2022-10-31T12:00:00+00:00"},
            {"story_id": 2, "created_utc": "2022-11-01T00:00:00+00:00"},
        ]
        buckets: dict = pull.bucket_by_month(stories, "2022-11-01")
        assert "2022-10" not in buckets
        assert "2022-11" in buckets
        assert [s["story_id"] for s in buckets["2022-11"]] == [2]

    def test_groups_by_yyyy_mm(self) -> None:
        """Stories are grouped by YYYY-MM of ``created_utc``."""
        stories: list[dict] = [
            {"story_id": 1, "created_utc": "2023-01-05T10:00:00+00:00"},
            {"story_id": 2, "created_utc": "2023-01-20T10:00:00+00:00"},
            {"story_id": 3, "created_utc": "2023-02-02T10:00:00+00:00"},
        ]
        buckets: dict = pull.bucket_by_month(stories, "2022-11-01")
        assert sorted(buckets.keys()) == ["2023-01", "2023-02"]
        assert len(buckets["2023-01"]) == 2
        assert len(buckets["2023-02"]) == 1


# ---------------------------------------------------------------------------
# cap_monthly_top_n
# ---------------------------------------------------------------------------


class TestCapMonthlyTopN:
    """Tests for ``cap_monthly_top_n``."""

    def test_keeps_top_n_by_score(self) -> None:
        """Per month, top ``limit`` stories by points are retained."""
        buckets: dict = {
            "2023-01": [
                {"story_id": 1, "score": 5},
                {"story_id": 2, "score": 100},
                {"story_id": 3, "score": 50},
            ]
        }
        kept: list[dict] = pull.cap_monthly_top_n(buckets, limit=2)
        ids: list[int] = [s["story_id"] for s in kept]
        assert ids == [2, 3]

    def test_sort_stability_ties_break_by_story_id(self) -> None:
        """Equal scores tie-break by ``story_id`` ascending."""
        buckets: dict = {
            "2023-01": [
                {"story_id": 9, "score": 10},
                {"story_id": 3, "score": 10},
                {"story_id": 7, "score": 10},
            ]
        }
        kept: list[dict] = pull.cap_monthly_top_n(buckets, limit=3)
        assert [s["story_id"] for s in kept] == [3, 7, 9]

    def test_sorts_by_month_ascending(self) -> None:
        """Output is ordered by month ascending."""
        buckets: dict = {
            "2023-02": [{"story_id": 1, "score": 5}],
            "2023-01": [{"story_id": 2, "score": 5}],
        }
        kept: list[dict] = pull.cap_monthly_top_n(buckets, limit=5)
        assert [s["story_id"] for s in kept] == [2, 1]


# ---------------------------------------------------------------------------
# truncate_and_normalize
# ---------------------------------------------------------------------------


class TestTruncateAndNormalize:
    """Tests for ``truncate_and_normalize``."""

    def test_collapses_whitespace(self) -> None:
        """Runs of whitespace collapse to single spaces."""
        assert pull.truncate_and_normalize("a   b\n\tc", 100) == "a b c"

    def test_strips_smart_quotes_and_dashes(self) -> None:
        """Smart punctuation is replaced with ASCII equivalents."""
        out: str = pull.truncate_and_normalize(
            "\u201choly\u201d \u2014 cow \u2013 ok\u2026", 100
        )
        assert out == '"holy" - cow - ok...'

    def test_truncates_on_whitespace_boundary(self) -> None:
        """Truncation prefers a whitespace boundary when possible."""
        raw: str = "alpha beta gamma delta epsilon zeta"
        out: str = pull.truncate_and_normalize(raw, 15)
        assert len(out) <= 15
        # Does not end mid-word when a space boundary exists in range
        assert not out.endswith("gam")

    def test_empty_input_returns_empty(self) -> None:
        """Empty or None-like strings return the empty string."""
        assert pull.truncate_and_normalize("", 100) == ""
        assert pull.truncate_and_normalize("   ", 100) == ""

    def test_under_limit_passthrough(self) -> None:
        """Text shorter than limit passes through (after normalization)."""
        assert pull.truncate_and_normalize("short text", 100) == "short text"


# ---------------------------------------------------------------------------
# fetch_query_page
# ---------------------------------------------------------------------------


def _mock_transport(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.MockTransport:
    """Shim so tests can build an httpx.Client with a canned handler."""
    return httpx.MockTransport(handler)


class TestFetchQueryPage:
    """Tests for ``fetch_query_page``."""

    def test_happy_path(self) -> None:
        """A 200 response is parsed and returned as a dict."""
        payload: dict = {"hits": [_base_hit()], "nbPages": 1, "nbHits": 1}

        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.host == "hn.algolia.com"
            return httpx.Response(200, json=payload)

        with httpx.Client(transport=_mock_transport(handler)) as client:
            result: dict = pull.fetch_query_page(client, "layoffs", page=0)
        assert result == payload

    def test_propagates_http_error(self) -> None:
        """A 500 response raises ``HTTPStatusError``."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="server down")

        with httpx.Client(transport=_mock_transport(handler)) as client:
            with pytest.raises(httpx.HTTPStatusError):
                pull.fetch_query_page(client, "layoffs", page=0)

    def test_raises_on_empty_query(self) -> None:
        """Empty query string raises ValueError before any HTTP call."""
        with httpx.Client() as client:
            with pytest.raises(ValueError):
                pull.fetch_query_page(client, "", page=0)

    def test_raises_on_whitespace_query(self) -> None:
        """Whitespace-only query raises ValueError."""
        with httpx.Client() as client:
            with pytest.raises(ValueError):
                pull.fetch_query_page(client, "   ", page=0)


# ---------------------------------------------------------------------------
# pull_all integration
# ---------------------------------------------------------------------------


def _install_pull_all_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    cache_name: str = "hn_stories.json",
) -> Path:
    """Redirect pull_all's cache + speed up the rate-limit sleep."""
    cache: Path = tmp_path / cache_name
    monkeypatch.setattr(pull, "RAW_DIR", tmp_path, raising=True)
    monkeypatch.setattr(pull, "CACHE_PATH", cache, raising=True)
    monkeypatch.setattr(pull, "ALGOLIA_REQUEST_DELAY", 0.0, raising=True)
    # speed up _collect_query's time.sleep calls too
    monkeypatch.setattr(pull.time, "sleep", lambda *_a, **_k: None)
    return cache


def _canned_page(hits: list[dict], nb_pages: int = 1) -> dict:
    """Build a minimal Algolia response payload."""
    return {"hits": hits, "nbPages": nb_pages, "nbHits": len(hits)}


class TestPullAll:
    """Integration tests for ``pull_all``."""

    def test_caches_atomically(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """On success, the cache file exists and the .tmp file does not."""
        cache: Path = _install_pull_all_env(monkeypatch, tmp_path)

        monkeypatch.setattr(
            pull, "SEARCH_QUERIES", ("layoffs",), raising=True
        )

        fake_hit: dict = _base_hit()

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=_canned_page([fake_hit]))

        _install_mock_client(monkeypatch, handler)
        stats: dict = pull.pull_all(refresh=True)

        assert cache.exists()
        # .tmp sibling must not persist
        assert not cache.with_suffix(cache.suffix + ".tmp").exists()
        loaded: list[dict] = json.loads(cache.read_text())
        assert len(loaded) == 1
        assert loaded[0]["story_id"] == 12345678
        assert stats["queries_run"] == 1
        assert stats["kept_after_monthly_cap"] == 1

    def test_skips_when_cache_exists_and_no_refresh(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Existing cache + no refresh flag returns early without HTTP."""
        cache: Path = _install_pull_all_env(monkeypatch, tmp_path)
        cache.write_text("[]")

        # If pull_all tried to hit the network, Client() would raise
        # because we patch it to fail.
        def boom(*_a, **_k):
            raise AssertionError("pull_all should not make HTTP calls")

        monkeypatch.setattr(pull.httpx, "Client", boom)

        stats: dict = pull.pull_all(refresh=False)
        assert stats["queries_run"] == 0

    def test_refreshes_when_flag_set(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """--refresh overrides an existing cache and re-pulls."""
        cache: Path = _install_pull_all_env(monkeypatch, tmp_path)
        cache.write_text("[]")

        monkeypatch.setattr(pull, "SEARCH_QUERIES", ("layoffs",), raising=True)

        hit: dict = _base_hit()

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=_canned_page([hit]))

        _install_mock_client(monkeypatch, handler)
        stats: dict = pull.pull_all(refresh=True)

        assert stats["queries_run"] == 1
        loaded: list[dict] = json.loads(cache.read_text())
        assert len(loaded) == 1

    def test_continues_on_single_query_failure(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """One failed query does not zero the whole pull."""
        cache: Path = _install_pull_all_env(monkeypatch, tmp_path)
        monkeypatch.setattr(
            pull, "SEARCH_QUERIES", ("good", "bad"), raising=True
        )

        def handler(request: httpx.Request) -> httpx.Response:
            q: str = request.url.params.get("query", "")
            if q == "bad":
                return httpx.Response(503, text="service unavailable")
            return httpx.Response(200, json=_canned_page([_base_hit()]))

        _install_mock_client(monkeypatch, handler)
        stats: dict = pull.pull_all(refresh=True)

        assert stats["queries_run"] == 2
        assert stats["queries_failed"] == 1
        loaded: list[dict] = json.loads(cache.read_text())
        assert len(loaded) == 1

    def test_exits_nonzero_if_all_queries_fail(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """All queries failing exits with status code 1."""
        _install_pull_all_env(monkeypatch, tmp_path)
        monkeypatch.setattr(pull, "SEARCH_QUERIES", ("a", "b"), raising=True)

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="boom")

        _install_mock_client(monkeypatch, handler)
        with pytest.raises(SystemExit) as excinfo:
            pull.pull_all(refresh=True)
        assert excinfo.value.code == 1

    def test_caps_pagination_at_10_pages(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A query reporting >10 pages is truncated at the safety cap."""
        cache: Path = _install_pull_all_env(monkeypatch, tmp_path)
        monkeypatch.setattr(
            pull, "SEARCH_QUERIES", ("layoffs",), raising=True
        )

        call_count: dict[str, int] = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            call_count["n"] += 1
            # Every page reports 20 pages worth of hits
            return httpx.Response(
                200, json=_canned_page([_base_hit()], nb_pages=20)
            )

        _install_mock_client(monkeypatch, handler)
        pull.pull_all(refresh=True)

        assert call_count["n"] == pull.MAX_PAGES_PER_QUERY
        loaded: list[dict] = json.loads(cache.read_text())
        # Only 1 unique story (same hit repeated across pages), deduped
        assert len(loaded) == 1

    def test_empty_pull_writes_empty_list(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Zero hits across every query still writes [] rather than erroring."""
        cache: Path = _install_pull_all_env(monkeypatch, tmp_path)
        monkeypatch.setattr(pull, "SEARCH_QUERIES", ("layoffs",), raising=True)

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=_canned_page([]))

        _install_mock_client(monkeypatch, handler)
        stats: dict = pull.pull_all(refresh=True)

        assert cache.exists()
        assert json.loads(cache.read_text()) == []
        assert stats["queries_failed"] == 0
