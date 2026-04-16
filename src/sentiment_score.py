"""Transformer sentiment scoring for Hacker News stories (Phase 13).

Reads the curated list at ``data/raw/hn_stories.json``, scores each
story with ``cardiffnlp/twitter-roberta-base-sentiment-latest`` (3-class
positive / neutral / negative + softmax), computes a compound score
(positive minus negative) in ``[-1, 1]``, and writes the augmented list
back to the same file.

The transformer pipeline is **lazy-imported** inside ``_load_pipeline``
so that importing this module does not pull ``transformers`` or
``torch`` into ``sys.modules``. Tests monkeypatch ``_load_pipeline`` to
return a stub callable with the same signature as the Hugging Face
pipeline; the test suite therefore never downloads a model or loads
torch.

Idempotent by default: stories that already carry ``sentiment_score``
are skipped unless ``--rescore`` is passed.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
RAW_DIR: Path = PROJECT_ROOT / "data" / "raw"
CACHE_PATH: Path = RAW_DIR / "hn_stories.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger: logging.Logger = logging.getLogger(__name__)

MODEL_NAME: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
"""Hugging Face model used for sentiment classification.

3-class (positive / neutral / negative), softmax scores, CPU-friendly.
"""

DELETED_PLACEHOLDER: str = "[deleted]"
"""Title marker used by HN (and Algolia) for deleted stories.

Combined with an empty or placeholder excerpt, deleted stories get the
neutral-sentiment assignment instead of a tokenizer pass.
"""

BATCH_SIZE: int = 16
"""Default batch size for the transformer pipeline."""


# ---------------------------------------------------------------------------
# Pure scoring helpers
# ---------------------------------------------------------------------------


def compound_score(probs: dict[str, float]) -> float:
    """Compute a VADER-style compound score in ``[-1, 1]``.

    Args:
        probs: Mapping from the three labels ``positive`` / ``neutral``
            / ``negative`` to softmax probabilities.

    Returns:
        ``probs['positive'] - probs['negative']``. Missing keys default
        to 0.0 so a stub pipeline can omit classes without crashing.
    """
    positive: float = float(probs.get("positive", 0.0))
    negative: float = float(probs.get("negative", 0.0))
    return positive - negative


def argmax_label(probs: dict[str, float]) -> str:
    """Return the label with the highest probability.

    Args:
        probs: Mapping from label to probability.

    Returns:
        Label with the highest value. Ties broken by Python ``max``
        iteration order (fine for our 3-label case).

    Raises:
        ValueError: If ``probs`` is empty.
    """
    if not probs:
        raise ValueError("argmax_label requires a non-empty probs dict.")
    return max(probs.items(), key=lambda kv: kv[1])[0]


def _normalize_pipeline_output(raw: object) -> dict[str, float]:
    """Convert one pipeline prediction into a ``{label: prob}`` dict.

    Hugging Face pipelines with ``top_k=None`` return a list of
    ``{"label": ..., "score": ...}`` dicts. This helper tolerates a
    single-dict prediction as well so that test stubs can return
    minimal shapes.

    Args:
        raw: The per-story output from the pipeline.

    Returns:
        Dict from lowercase label to float probability.
    """
    if isinstance(raw, dict):
        return {str(raw["label"]).lower(): float(raw["score"])}
    probs: dict[str, float] = {}
    for entry in raw:  # type: ignore[union-attr]
        probs[str(entry["label"]).lower()] = float(entry["score"])
    return probs


def _build_input_text(story: dict) -> str:
    """Assemble the model input string for one story.

    Titles carry most of the sentiment signal on HN link posts, so we
    concatenate the title and (possibly-empty) excerpt with a newline.

    Args:
        story: Per-story dict from the Algolia cache.

    Returns:
        String passed to the transformer pipeline.
    """
    title: str = str(story.get("title") or "")
    excerpt: str = str(story.get("text_excerpt") or "")
    if excerpt:
        return f"{title} \n {excerpt}"
    return title


def _is_deleted(story: dict) -> bool:
    """Return True if the story looks like a deleted HN item."""
    title: str = str(story.get("title") or "").strip()
    excerpt: str = str(story.get("text_excerpt") or "").strip()
    return title == DELETED_PLACEHOLDER and not excerpt


# ---------------------------------------------------------------------------
# Scoring orchestration
# ---------------------------------------------------------------------------


def score_stories(
    stories: list[dict],
    pipeline_fn: Callable[..., list],
    batch_size: int = BATCH_SIZE,
    rescore: bool = False,
) -> list[dict]:
    """Score stories in batches and return a new list with sentiment fields.

    Does not mutate the input list. Each returned story gains
    ``sentiment_score`` (float in ``[-1, 1]``), ``sentiment_label``
    (``positive`` / ``neutral`` / ``negative``), and ``scored_at``
    (UTC ISO8601 timestamp).

    Args:
        stories: List of per-story dicts from ``hackernews_pull``.
        pipeline_fn: Callable matching ``transformers.pipeline`` —
            called as ``pipeline_fn(list[str], batch_size=...)`` and
            expected to return a list of pipeline predictions.
        batch_size: Rows passed to the pipeline per call.
        rescore: If True, re-score stories that already carry
            ``sentiment_score``. Default skips them.

    Returns:
        New list with sentiment fields populated.
    """
    scored_at: str = datetime.now(timezone.utc).isoformat()

    pending: list[tuple[int, dict]] = []
    output: list[dict] = []
    for idx, story in enumerate(stories):
        new_story: dict = dict(story)
        output.append(new_story)

        if _is_deleted(new_story):
            new_story["sentiment_score"] = 0.0
            new_story["sentiment_label"] = "neutral"
            new_story["scored_at"] = scored_at
            continue

        if not rescore and "sentiment_score" in new_story:
            continue

        pending.append((idx, new_story))

    if not pending:
        return output

    texts: list[str] = [_build_input_text(item[1]) for item in pending]

    for start in range(0, len(texts), batch_size):
        batch_texts: list[str] = texts[start : start + batch_size]
        batch_items: list[tuple[int, dict]] = pending[start : start + batch_size]
        predictions = pipeline_fn(batch_texts, batch_size=batch_size)
        for (_, story), raw in zip(batch_items, predictions):
            probs: dict[str, float] = _normalize_pipeline_output(raw)
            story["sentiment_score"] = compound_score(probs)
            story["sentiment_label"] = argmax_label(probs)
            story["scored_at"] = scored_at

    return output


def _load_pipeline() -> Callable[..., list]:
    """Lazy-load the transformers sentiment pipeline.

    Only imported when scoring actually runs. Tests monkeypatch this
    function or pass ``pipeline_fn`` directly to ``score_all`` so
    ``transformers`` and ``torch`` never load in the suite.

    Returns:
        A callable with the Hugging Face pipeline signature.
    """
    from transformers import pipeline

    return pipeline(
        "sentiment-analysis",
        model=MODEL_NAME,
        tokenizer=MODEL_NAME,
        top_k=None,
        truncation=True,
        padding=True,
    )


def score_all(
    pipeline_fn: Callable[..., list] | None = None,
    rescore: bool = False,
) -> dict[str, int]:
    """Read ``hn_stories.json``, score pending stories, write back.

    Idempotent: by default, pre-scored stories are skipped. An empty or
    missing cache is a no-op that returns zero counts.

    Args:
        pipeline_fn: Callable matching the Hugging Face pipeline API.
            Tests pass a stub here. Production code leaves this as
            None and ``_load_pipeline`` is called lazily.
        rescore: If True, re-score every story including the ones
            already carrying ``sentiment_score``.

    Returns:
        Stats dict with keys ``scored``, ``skipped_deleted``,
        ``skipped_prescored``.
    """
    if not CACHE_PATH.exists():
        logger.warning(
            "No HN cache at %s - nothing to score", CACHE_PATH
        )
        return {
            "scored": 0,
            "skipped_deleted": 0,
            "skipped_prescored": 0,
        }

    stories: list[dict] = json.loads(CACHE_PATH.read_text())

    skipped_deleted: int = 0
    skipped_prescored: int = 0
    needs_scoring: int = 0
    for story in stories:
        if _is_deleted(story):
            skipped_deleted += 1
            continue
        if not rescore and "sentiment_score" in story:
            skipped_prescored += 1
            continue
        needs_scoring += 1

    if needs_scoring == 0 and not any(_is_deleted(s) for s in stories):
        logger.info(
            "All %d stories already scored - skipping pipeline load",
            len(stories),
        )
        return {
            "scored": 0,
            "skipped_deleted": skipped_deleted,
            "skipped_prescored": skipped_prescored,
        }

    pipeline_callable: Callable[..., list]
    if pipeline_fn is not None:
        pipeline_callable = pipeline_fn
    elif needs_scoring > 0:
        logger.info("Loading %s pipeline (CPU)", MODEL_NAME)
        pipeline_callable = _load_pipeline()
    else:
        pipeline_callable = lambda *_a, **_k: []  # unreachable safety  # noqa: E731

    updated: list[dict] = score_stories(
        stories, pipeline_callable, rescore=rescore
    )

    CACHE_PATH.write_text(json.dumps(updated, indent=2))
    stats: dict[str, int] = {
        "scored": needs_scoring,
        "skipped_deleted": skipped_deleted,
        "skipped_prescored": skipped_prescored,
    }
    logger.info("Sentiment scoring stats: %s", stats)
    return stats


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description=(
            "Score Hacker News stories with a transformer sentiment "
            "classifier. Reads and writes data/raw/hn_stories.json."
        ),
    )
    parser.add_argument(
        "--rescore",
        action="store_true",
        help=(
            "Re-score every story, including those already carrying "
            "a sentiment_score. Default: skip already-scored stories."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for the sentiment scoring script."""
    args: argparse.Namespace = _parse_args()
    score_all(rescore=args.rescore)


if __name__ == "__main__":
    main()
