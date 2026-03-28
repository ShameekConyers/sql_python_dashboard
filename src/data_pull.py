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
from pathlib import Path

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
        if value != value:  # NaN check without importing math
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
    logger.info("Done.")


if __name__ == "__main__":
    main()
