"""Shared fixtures for data_pull tests."""

import sys
from pathlib import Path

import pytest

# Add src/ to the import path so tests can import project modules.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


@pytest.fixture()
def sample_series_info() -> dict:
    """Return a single SeriesInfo dict for testing.

    Returns:
        A minimal SeriesInfo matching the UNRATE series.
    """
    return {
        "id": "UNRATE",
        "name": "Unemployment Rate",
        "category": "labor_market",
        "frequency": "monthly",
    }


@pytest.fixture()
def two_series_list() -> list[dict]:
    """Return two SeriesInfo dicts for multi-series tests.

    Returns:
        A list with UNRATE and GDPC1 series info.
    """
    return [
        {
            "id": "UNRATE",
            "name": "Unemployment Rate",
            "category": "labor_market",
            "frequency": "monthly",
        },
        {
            "id": "GDPC1",
            "name": "Real Gross Domestic Product",
            "category": "output_growth",
            "frequency": "quarterly",
        },
    ]
