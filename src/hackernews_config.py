"""Configuration constants for Hacker News ingestion (Phase 13).

Pure stdlib module. No runtime dependencies — importing this file must
not load ``httpx``, ``transformers``, or any other heavy third-party
package. The ingestion script and test fixtures import from here.

The coverage window starts at 2022-01-01 to match the default x-axis
start of the dashboard's time-series charts, so HN months line up with
visible FRED data. That also captures ~10 months of pre-ChatGPT
tech-labor signal (Twitter acquisition aftermath, Meta's Nov 2022
layoff wave, big-tech hiring slowdown) alongside the post-2022-11
AI-labor-impact era. Keyword queries are curated to over-retrieve
within the AI-labor-impact theme; monthly top-N-by-points selection
filters the corpus down to a seed-friendly size.
"""

from __future__ import annotations


COVERAGE_START_UNIX: int = 1640995200
"""Earliest ``created_at_i`` (inclusive) to keep.

Corresponds to 2022-01-01T00:00:00Z. Matches the default x-axis
start of the dashboard's time-series charts so HN months line up
with visible FRED data. Captures the ~10 months of pre-ChatGPT
tech-labor signal (Twitter acquisition aftermath, Meta's first big
layoff wave in Nov 2022, broader big-tech hiring slowdown) before
the post-2022-11 AI-impact era.
"""

COVERAGE_START_ISO: str = "2022-01-01"
"""Human-readable form of ``COVERAGE_START_UNIX``."""

MONTHLY_STORY_LIMIT: int = 30
"""Max stories to keep per month after top-points selection."""

STORY_TEXT_CHAR_LIMIT: int = 500
"""Hard cap on characters of HN self-post text stored per story.

Truncates on whitespace where possible, else at the character boundary.
"""

SEARCH_QUERIES: tuple[str, ...] = (
    # layoffs theme
    "layoffs",
    "fired tech",
    "severance",
    # ai_jobs theme
    "ai replaced",
    "ai jobs",
    "ai hiring",
    # career theme
    "hiring freeze",
    "tech job market",
    "software engineer job",
)
"""Algolia ``query`` parameter values.

One HTTP request per query. Results are union-merged and deduplicated
by ``story_id``. Curated to over-retrieve within the AI-labor-impact
theme; quality is filtered later by per-month top-N selection and by
the fact that non-thematic stories rarely score high on the monthly
leaderboard even if Algolia surfaces them.
"""

ALGOLIA_BASE_URL: str = "https://hn.algolia.com/api/v1/search_by_date"
"""Algolia HN search endpoint.

``search_by_date`` (not ``search``) sorts by recency and supports
``numericFilters`` on ``created_at_i``, which we use for the
2022-01-01-onward window.
"""

ALGOLIA_HITS_PER_PAGE: int = 1000
"""Max ``hitsPerPage`` Algolia allows.

Set to the cap so a typical query returns a single page and pagination
can be skipped unless ``nbPages > 1``.
"""

ALGOLIA_REQUEST_DELAY: float = 0.4
"""Seconds between Algolia calls.

Rate limit is 10,000/hr per IP. With ~9 queries * up to 3 pages each,
0.4s keeps us polite and nowhere near the cap.
"""

LAYOFF_KEYWORDS: tuple[str, ...] = (
    "layoff",
    "laid off",
    "layoffs",
    "fired",
    "let go",
    "downsized",
    "riffed",
    "rif",
    "reduction in force",
    "restructuring",
    "severance",
    "terminated",
)
"""Case-insensitive substrings matched against (title + " " + text_excerpt).

Used at load time to compute ``hn_sentiment_monthly.layoff_story_count``.
Curated for precision over recall.
"""
