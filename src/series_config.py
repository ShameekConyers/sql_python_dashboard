"""FRED series configuration for the Macro Economic Dashboard.

Defines the 10 series used in the recession risk + AI impact analysis.
Each entry contains the FRED series ID, display name, category for grouping,
and expected frequency for validation during ingestion.
"""

from typing import TypedDict


class SeriesInfo(TypedDict):
    """Metadata for a single FRED series.

    Attributes:
        id: FRED series identifier.
        name: Human-readable display name.
        category: Grouping category for dashboard organization.
        frequency: Expected data frequency (monthly, quarterly, daily).
    """

    id: str
    name: str
    category: str
    frequency: str


SERIES: list[SeriesInfo] = [
    # Labor market
    {
        "id": "UNRATE",
        "name": "Unemployment Rate",
        "category": "labor_market",
        "frequency": "monthly",
    },
    {
        "id": "U6RATE",
        "name": "Total Underemployment Rate (U-6)",
        "category": "labor_market",
        "frequency": "monthly",
    },
    # Output and growth
    {
        "id": "GDPC1",
        "name": "Real Gross Domestic Product",
        "category": "output_growth",
        "frequency": "quarterly",
    },
    # Yield curve
    {
        "id": "T10Y2Y",
        "name": "10-Year Minus 2-Year Treasury Spread",
        "category": "yield_curve",
        "frequency": "daily",
    },
    # Prices
    {
        "id": "CPIAUCSL",
        "name": "Consumer Price Index (All Urban Consumers)",
        "category": "prices",
        "frequency": "monthly",
    },
    # Population normalizer
    {
        "id": "CNP16OV",
        "name": "Civilian Noninstitutional Population (16+)",
        "category": "population",
        "frequency": "monthly",
    },
    # AI impact: labor
    {
        "id": "USINFO",
        "name": "Information Sector Employment",
        "category": "ai_labor",
        "frequency": "monthly",
    },
    {
        "id": "CES2023800001",
        "name": "Specialty Trade Contractors Employment",
        "category": "ai_labor",
        "frequency": "monthly",
    },
    # AI impact: energy
    {
        "id": "IPG2211S",
        "name": "Electric Power Generation, Transmission & Distribution",
        "category": "ai_energy",
        "frequency": "monthly",
    },
    # Recession benchmark
    {
        "id": "USREC",
        "name": "NBER Recession Indicator",
        "category": "recession",
        "frequency": "monthly",
    },
]

SERIES_IDS: list[str] = [s["id"] for s in SERIES]
"""Flat list of all FRED series IDs for quick iteration."""
