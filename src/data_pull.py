"""FRED API data ingestion script.

Pulls economic time series from the FRED API and caches raw responses
as JSON files in data/raw/. Idempotent by default: skips series that
already have a cached file unless --refresh is passed.
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
from dotenv import load_dotenv
from fredapi import Fred

from series_config import SERIES, SeriesInfo

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
RAW_DIR: Path = PROJECT_ROOT / "data" / "raw"
ENV_PATH: Path = PROJECT_ROOT / ".env"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger: logging.Logger = logging.getLogger(__name__)

RATE_LIMIT_DELAY: float = 0.6
"""Seconds to pause between API calls. FRED allows 120 req/min, so
0.5s per request stays under the limit. 0.6s adds a small margin."""


def _load_api_key() -> str:
    """Load the FRED API key from the .env file.

    Returns:
        The FRED_API_KEY value.

    Raises:
        SystemExit: If the key is missing or empty.
    """
    load_dotenv(ENV_PATH)
    import os

    key: str | None = os.getenv("FRED_API_KEY")
    if not key or key == "your_key_here":
        logger.error(
            "FRED_API_KEY not set. Copy .env.example to .env and add your key."
        )
        sys.exit(1)
    return key


def _output_path(series_id: str) -> Path:
    """Return the expected JSON cache path for a series.

    Args:
        series_id: FRED series identifier.

    Returns:
        Path to the JSON file in data/raw/.
    """
    return RAW_DIR / f"{series_id}.json"


def _metadata_path(series_id: str) -> Path:
    """Return the expected metadata JSON cache path for a series.

    Args:
        series_id: FRED series identifier.

    Returns:
        Path to ``data/raw/{series_id}_metadata.json``.
    """
    return RAW_DIR / f"{series_id}_metadata.json"


FRED_API_BASE: str = "https://api.stlouisfed.org/fred"
"""Base URL for FRED API endpoints not wrapped by ``fredapi``."""


def _fetch_release_info(
    client: httpx.Client, api_key: str, series_id: str
) -> dict[str, str]:
    """Fetch the release metadata (name, link, notes) for a FRED series.

    Args:
        client: An open ``httpx.Client`` used for the API call.
        api_key: FRED API key.
        series_id: FRED series identifier.

    Returns:
        Dict with keys 'release_name', 'release_link', 'release_notes'. Values
        may be empty strings when the field is absent from the response.
    """
    url: str = f"{FRED_API_BASE}/series/release"
    params: dict[str, str] = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
    }
    response = client.get(url, params=params, timeout=15.0)
    response.raise_for_status()
    payload: dict = response.json()
    releases: list[dict] = payload.get("releases") or []
    if not releases:
        return {"release_name": "", "release_link": "", "release_notes": ""}
    release: dict = releases[0]
    return {
        "release_name": str(release.get("name", "") or ""),
        "release_link": str(release.get("link", "") or ""),
        "release_notes": str(release.get("notes", "") or ""),
    }


def _fetch_category_path(
    client: httpx.Client, api_key: str, series_id: str
) -> str:
    """Fetch the flattened category hierarchy for a FRED series.

    Walks the ``parent_id`` chain for the first category attached to the
    series and returns a string like ``"Labor Market > Unemployment >
    Monthly Rates"``. Falls back to the leaf category name when no parents
    resolve.

    Args:
        client: An open ``httpx.Client`` used for the API calls.
        api_key: FRED API key.
        series_id: FRED series identifier.

    Returns:
        A ``" > "``-joined category path, or an empty string if FRED
        returns no categories.
    """
    leaf_url: str = f"{FRED_API_BASE}/series/categories"
    params: dict[str, str] = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
    }
    leaf_response = client.get(leaf_url, params=params, timeout=15.0)
    leaf_response.raise_for_status()
    categories: list[dict] = leaf_response.json().get("categories") or []
    if not categories:
        return ""

    category: dict = categories[0]
    names: list[str] = [str(category.get("name", "") or "")]
    parent_id = category.get("parent_id")

    # Walk parents up to a sensible depth to avoid runaway queries
    parent_url: str = f"{FRED_API_BASE}/category"
    max_depth: int = 6
    depth: int = 0
    while parent_id and parent_id != 0 and depth < max_depth:
        parent_response = client.get(
            parent_url,
            params={
                "category_id": str(parent_id),
                "api_key": api_key,
                "file_type": "json",
            },
            timeout=15.0,
        )
        parent_response.raise_for_status()
        parents: list[dict] = parent_response.json().get("categories") or []
        if not parents:
            break
        parent: dict = parents[0]
        names.append(str(parent.get("name", "") or ""))
        parent_id = parent.get("parent_id")
        depth += 1
        time.sleep(RATE_LIMIT_DELAY)

    return " > ".join(reversed([n for n in names if n]))


def _pull_series_metadata(
    fred: Fred, api_key: str, series_id: str
) -> dict:
    """Pull series-level metadata for RAG citations (Phase 11).

    Combines three FRED sources:
      1. ``fred.get_series_info(sid)`` — methodology notes, units, etc.
      2. ``/fred/series/release?series_id=...`` — parent release metadata.
      3. ``/fred/series/categories?series_id=...`` with parent walk —
         flattened category hierarchy.

    Individual sub-fetches are wrapped in try/except; a failure on any one
    leaves that field blank rather than losing the entire record.

    Args:
        fred: Authenticated ``fredapi.Fred`` client.
        api_key: FRED API key (required for the httpx-direct endpoints).
        series_id: FRED series identifier.

    Returns:
        Dict with keys: 'series_id', 'series_notes', 'title', 'units',
        'seasonal_adjustment', 'release_name', 'release_link',
        'release_notes', 'category_path', 'fetched_at'.
    """
    metadata: dict = {
        "series_id": series_id,
        "series_notes": "",
        "title": "",
        "units": "",
        "seasonal_adjustment": "",
        "release_name": "",
        "release_link": "",
        "release_notes": "",
        "category_path": "",
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }

    # Series-level info via fredapi (wraps /fred/series)
    try:
        info = fred.get_series_info(series_id)
        metadata["series_notes"] = str(info.get("notes", "") or "")
        metadata["title"] = str(info.get("title", "") or "")
        metadata["units"] = str(info.get("units", "") or "")
        metadata["seasonal_adjustment"] = str(
            info.get("seasonal_adjustment", "") or ""
        )
    except Exception:
        logger.exception("Failed to fetch series_info for %s", series_id)

    # Release + category via httpx (endpoints fredapi does not wrap)
    with httpx.Client() as client:
        try:
            release: dict[str, str] = _fetch_release_info(
                client, api_key, series_id
            )
            metadata.update(release)
        except Exception:
            logger.exception("Failed to fetch release info for %s", series_id)
        time.sleep(RATE_LIMIT_DELAY)

        try:
            metadata["category_path"] = _fetch_category_path(
                client, api_key, series_id
            )
        except Exception:
            logger.exception(
                "Failed to fetch category path for %s", series_id
            )

    return metadata


def pull_metadata_list(
    series_list: list[SeriesInfo],
    refresh: bool = False,
) -> None:
    """Pull series metadata (notes, release, category) and cache as JSON.

    Idempotent: skips any series whose metadata JSON already exists unless
    ``refresh`` is True. Missing metadata for a single series logs a warning
    and is NOT fatal — downstream code treats missing files as empty.

    Args:
        series_list: Series to pull.
        refresh: If True, re-pull even if a cached metadata file exists.
    """
    api_key: str = _load_api_key()
    fred: Fred = Fred(api_key=api_key)
    total: int = len(series_list)

    for i, series_info in enumerate(series_list, start=1):
        series_id: str = series_info["id"]
        out_path: Path = _metadata_path(series_id)

        if out_path.exists() and not refresh:
            logger.info(
                "[%d/%d] %s metadata — cached, skipping",
                i,
                total,
                series_id,
            )
            continue

        logger.info("[%d/%d] Pulling %s metadata ...", i, total, series_id)
        try:
            metadata: dict = _pull_series_metadata(fred, api_key, series_id)
            out_path.write_text(json.dumps(metadata, indent=2))
            logger.info(
                "[%d/%d] %s metadata — saved (category=%s)",
                i,
                total,
                series_id,
                metadata.get("category_path") or "(none)",
            )
        except Exception:
            logger.exception(
                "[%d/%d] %s metadata — FAILED (non-fatal, continuing)",
                i,
                total,
                series_id,
            )

        if i < total:
            time.sleep(RATE_LIMIT_DELAY)


def _pull_series(fred: Fred, series_info: SeriesInfo) -> dict:
    """Pull a single series from FRED and return a serializable dict.

    Args:
        fred: Authenticated fredapi client.
        series_info: Metadata for the series to pull.

    Returns:
        Dict with metadata and observations ready for JSON serialization.

    Raises:
        Exception: Propagates any fredapi or network errors.
    """
    series_id: str = series_info["id"]

    # Get the observations as a pandas Series (index=dates, values=floats)
    observations = fred.get_series(series_id)

    # Get series info from FRED for extra metadata
    info = fred.get_series_info(series_id)

    records: list[dict] = []
    for ts, value in observations.items():
        record: dict = {"date": str(ts)[:10]}
        if value != value:  # IEEE 754: NaN != NaN, avoids importing math
            record["value"] = None
        else:
            record["value"] = float(value)
        records.append(record)

    return {
        "series_id": series_id,
        "name": series_info["name"],
        "category": series_info["category"],
        "frequency": series_info["frequency"],
        "units": str(info.get("units", "")),
        "seasonal_adjustment": str(info.get("seasonal_adjustment", "")),
        "last_updated": str(info.get("last_updated", "")),
        "observation_count": len(records),
        "observations": records,
    }


def pull_series_list(
    series_list: list[SeriesInfo],
    refresh: bool = False,
) -> None:
    """Pull one or more FRED series and save as JSON.

    Args:
        series_list: Series to pull.
        refresh: If True, re-pull even if a cached file exists.
    """
    fred: Fred = Fred(api_key=_load_api_key())
    total: int = len(series_list)

    for i, series_info in enumerate(series_list, start=1):
        series_id: str = series_info["id"]
        out_path: Path = _output_path(series_id)

        if out_path.exists() and not refresh:
            logger.info(
                "[%d/%d] %s — cached, skipping (use --refresh to re-pull)",
                i,
                total,
                series_id,
            )
            continue

        logger.info("[%d/%d] Pulling %s ...", i, total, series_id)
        try:
            data: dict = _pull_series(fred, series_info)
            out_path.write_text(json.dumps(data, indent=2))
            logger.info(
                "[%d/%d] %s — saved %d observations",
                i,
                total,
                series_id,
                data["observation_count"],
            )
        except Exception:
            logger.exception("[%d/%d] %s — FAILED", i, total, series_id)

        # Rate limit pause (skip after the last request)
        if i < total:
            time.sleep(RATE_LIMIT_DELAY)


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Pull FRED series data and cache as JSON.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Re-pull series even if a cached JSON file exists.",
    )
    parser.add_argument(
        "--series",
        type=str,
        default=None,
        help="Pull a single series by FRED ID (e.g. UNRATE). Useful for debugging.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for the FRED data pull script."""
    args: argparse.Namespace = _parse_args()

    if args.series:
        matches: list[SeriesInfo] = [
            s for s in SERIES if s["id"] == args.series.upper()
        ]
        if not matches:
            logger.error(
                "Series '%s' not found in series_config.py. Available: %s",
                args.series,
                ", ".join(s["id"] for s in SERIES),
            )
            sys.exit(1)
        target_list: list[SeriesInfo] = matches
    else:
        target_list = SERIES

    logger.info(
        "Starting FRED data pull — %d series, refresh=%s",
        len(target_list),
        args.refresh,
    )
    pull_series_list(target_list, refresh=args.refresh)

    logger.info("Starting FRED metadata pull (RAG sources) ...")
    pull_metadata_list(target_list, refresh=args.refresh)

    logger.info("Done.")


if __name__ == "__main__":
    main()
