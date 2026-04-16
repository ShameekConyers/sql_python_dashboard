"""Tests for src/sentiment_score.py.

All tests use a stub pipeline callable. The real transformers pipeline
is never loaded during the suite; one explicit subprocess test asserts
importing the module does not pull ``transformers`` into
``sys.modules``.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Callable
from unittest.mock import MagicMock

import pytest

import sentiment_score as scorer


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


class TestCompoundScore:
    """Tests for ``compound_score``."""

    def test_matches_positive_minus_negative(self) -> None:
        """Compound score is positive minus negative probability."""
        probs: dict[str, float] = {
            "positive": 0.7,
            "neutral": 0.2,
            "negative": 0.1,
        }
        assert scorer.compound_score(probs) == pytest.approx(0.6)

    def test_defaults_missing_keys_to_zero(self) -> None:
        """Missing label keys are treated as 0.0."""
        assert scorer.compound_score({"positive": 0.9}) == pytest.approx(0.9)
        assert scorer.compound_score({"negative": 0.4}) == pytest.approx(-0.4)

    def test_neutral_only_scores_zero(self) -> None:
        """Neutral-heavy probabilities produce ~0 compound."""
        probs: dict[str, float] = {
            "positive": 0.1,
            "neutral": 0.8,
            "negative": 0.1,
        }
        assert scorer.compound_score(probs) == pytest.approx(0.0)


class TestArgmaxLabel:
    """Tests for ``argmax_label``."""

    def test_picks_highest(self) -> None:
        """Returns the label with the max probability."""
        probs: dict[str, float] = {
            "positive": 0.2,
            "neutral": 0.3,
            "negative": 0.5,
        }
        assert scorer.argmax_label(probs) == "negative"

    def test_raises_on_empty(self) -> None:
        """Empty probs dict raises ValueError."""
        with pytest.raises(ValueError):
            scorer.argmax_label({})


# ---------------------------------------------------------------------------
# Stub pipeline
# ---------------------------------------------------------------------------


def _stub_pipeline_factory(positive: float = 0.8, neutral: float = 0.15, negative: float = 0.05) -> Callable[..., list]:
    """Return a pipeline stub that always returns the same distribution."""

    def _stub(texts: list[str], batch_size: int = 16) -> list[list[dict]]:  # noqa: ARG001
        prediction: list[dict] = [
            {"label": "positive", "score": positive},
            {"label": "neutral", "score": neutral},
            {"label": "negative", "score": negative},
        ]
        return [prediction for _ in texts]

    return _stub


# ---------------------------------------------------------------------------
# score_stories
# ---------------------------------------------------------------------------


def _story(**overrides: object) -> dict:
    """Build a minimal per-story dict."""
    base: dict = {
        "story_id": 1,
        "title": "Some title about tech layoffs",
        "text_excerpt": "",
        "created_utc": "2023-05-01T10:00:00+00:00",
        "score": 100,
    }
    base.update(overrides)
    return base


class TestScoreStories:
    """Tests for ``score_stories``."""

    def test_writes_score_and_label(self) -> None:
        """Every story gains sentiment_score + sentiment_label + scored_at."""
        stories: list[dict] = [_story(story_id=1), _story(story_id=2)]
        out: list[dict] = scorer.score_stories(stories, _stub_pipeline_factory())
        for s in out:
            assert s["sentiment_label"] == "positive"
            assert s["sentiment_score"] == pytest.approx(0.75)
            assert "scored_at" in s

    def test_handles_deleted_placeholder(self) -> None:
        """Deleted stories bypass the pipeline with neutral assignment."""
        stub: MagicMock = MagicMock()
        stories: list[dict] = [_story(title="[deleted]", text_excerpt="")]
        out: list[dict] = scorer.score_stories(stories, stub)
        assert out[0]["sentiment_score"] == 0.0
        assert out[0]["sentiment_label"] == "neutral"
        assert stub.call_count == 0

    def test_is_pure(self) -> None:
        """Input list is not mutated in place."""
        stories: list[dict] = [_story()]
        scorer.score_stories(stories, _stub_pipeline_factory())
        assert "sentiment_score" not in stories[0]

    def test_skips_prescored(self) -> None:
        """Default rescore=False means the pipeline is never called for
        already-scored stories."""
        stub: MagicMock = MagicMock()
        prescored: dict = _story()
        prescored["sentiment_score"] = -0.3
        prescored["sentiment_label"] = "negative"
        out: list[dict] = scorer.score_stories([prescored], stub)
        assert stub.call_count == 0
        assert out[0]["sentiment_score"] == -0.3
        assert out[0]["sentiment_label"] == "negative"

    def test_rescores_when_flag_set(self) -> None:
        """rescore=True overwrites existing sentiment fields."""
        prescored: dict = _story()
        prescored["sentiment_score"] = -0.3
        prescored["sentiment_label"] = "negative"
        out: list[dict] = scorer.score_stories(
            [prescored], _stub_pipeline_factory(), rescore=True
        )
        assert out[0]["sentiment_label"] == "positive"
        assert out[0]["sentiment_score"] == pytest.approx(0.75)

    def test_batches_respect_size(self) -> None:
        """Pipeline is called with batches no larger than batch_size."""
        calls: list[int] = []

        def recording_stub(texts: list[str], batch_size: int = 16) -> list[list[dict]]:
            calls.append(len(texts))
            return [
                [
                    {"label": "positive", "score": 0.5},
                    {"label": "neutral", "score": 0.3},
                    {"label": "negative", "score": 0.2},
                ]
                for _ in texts
            ]

        stories: list[dict] = [_story(story_id=i) for i in range(5)]
        scorer.score_stories(stories, recording_stub, batch_size=2)
        assert max(calls) <= 2
        assert sum(calls) == len(stories)


# ---------------------------------------------------------------------------
# score_all
# ---------------------------------------------------------------------------


class TestScoreAll:
    """Tests for ``score_all``."""

    def test_roundtrips_via_tmp_path(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """score_all writes sentiment fields back to the cache file."""
        cache: Path = tmp_path / "hn_stories.json"
        raw: list[dict] = [_story(story_id=1), _story(story_id=2), _story(story_id=3)]
        cache.write_text(json.dumps(raw))
        monkeypatch.setattr(scorer, "RAW_DIR", tmp_path, raising=True)
        monkeypatch.setattr(scorer, "CACHE_PATH", cache, raising=True)

        stats: dict[str, int] = scorer.score_all(
            pipeline_fn=_stub_pipeline_factory()
        )
        assert stats["scored"] == 3
        assert stats["skipped_prescored"] == 0

        loaded: list[dict] = json.loads(cache.read_text())
        for s in loaded:
            assert "sentiment_score" in s
            assert "sentiment_label" in s

    def test_missing_cache_is_noop(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """A missing cache file returns zero counts without raising."""
        cache: Path = tmp_path / "hn_stories.json"
        monkeypatch.setattr(scorer, "RAW_DIR", tmp_path, raising=True)
        monkeypatch.setattr(scorer, "CACHE_PATH", cache, raising=True)

        stats: dict[str, int] = scorer.score_all(
            pipeline_fn=_stub_pipeline_factory()
        )
        assert stats == {
            "scored": 0,
            "skipped_deleted": 0,
            "skipped_prescored": 0,
        }

    def test_load_pipeline_not_called_when_stub_passed(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Passing pipeline_fn bypasses the real pipeline loader."""
        cache: Path = tmp_path / "hn_stories.json"
        cache.write_text(json.dumps([_story()]))
        monkeypatch.setattr(scorer, "RAW_DIR", tmp_path, raising=True)
        monkeypatch.setattr(scorer, "CACHE_PATH", cache, raising=True)

        def explode() -> Callable[..., list]:
            raise AssertionError(
                "_load_pipeline should not be called when a stub is provided"
            )

        monkeypatch.setattr(scorer, "_load_pipeline", explode)

        # Should not raise
        scorer.score_all(pipeline_fn=_stub_pipeline_factory())

    def test_skips_pipeline_load_when_all_prescored(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """All-prescored cache does not trigger a pipeline load."""
        cache: Path = tmp_path / "hn_stories.json"
        prescored: dict = _story()
        prescored["sentiment_score"] = 0.5
        prescored["sentiment_label"] = "positive"
        cache.write_text(json.dumps([prescored]))
        monkeypatch.setattr(scorer, "RAW_DIR", tmp_path, raising=True)
        monkeypatch.setattr(scorer, "CACHE_PATH", cache, raising=True)

        def explode() -> Callable[..., list]:
            raise AssertionError("_load_pipeline should not be called")

        monkeypatch.setattr(scorer, "_load_pipeline", explode)
        stats: dict[str, int] = scorer.score_all()
        assert stats["scored"] == 0
        assert stats["skipped_prescored"] == 1


# ---------------------------------------------------------------------------
# Import-time discipline
# ---------------------------------------------------------------------------


class TestImportTimeDiscipline:
    """Guarantee sentiment_score does not load transformers at import."""

    def test_module_imports_no_transformers_at_top(self) -> None:
        """Freshly importing sentiment_score must not load transformers."""
        src_dir: Path = Path(__file__).resolve().parent.parent / "src"
        script: str = (
            "import sys, pathlib; "
            f"sys.path.insert(0, r'{src_dir}'); "
            "import sentiment_score; "
            "assert 'transformers' not in sys.modules, "
            "'transformers was imported at top of sentiment_score'"
        )
        result: subprocess.CompletedProcess = subprocess.run(
            [sys.executable, "-c", script],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            f"Subprocess failed.\nstdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )
