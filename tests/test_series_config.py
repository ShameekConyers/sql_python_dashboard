"""Tests for src/series_config.py.

Validates the FRED series configuration: structure, uniqueness, and completeness.
"""

from series_config import SERIES, SERIES_IDS, SeriesInfo


EXPECTED_SERIES_COUNT: int = 10

VALID_FREQUENCIES: set[str] = {"daily", "monthly", "quarterly"}

EXPECTED_CATEGORIES: set[str] = {
    "labor_market",
    "output_growth",
    "yield_curve",
    "prices",
    "population",
    "ai_labor",
    "ai_energy",
    "recession",
}

REQUIRED_KEYS: set[str] = {"id", "name", "category", "frequency"}


class TestSeriesStructure:
    """Validate each SeriesInfo dict has the right shape."""

    def test_series_count(self) -> None:
        """SERIES list contains exactly 10 entries."""
        assert len(SERIES) == EXPECTED_SERIES_COUNT

    def test_each_entry_has_required_keys(self) -> None:
        """Every entry has id, name, category, and frequency."""
        for entry in SERIES:
            missing: set[str] = REQUIRED_KEYS - set(entry.keys())
            assert not missing, f"{entry.get('id', '??')} missing keys: {missing}"

    def test_ids_are_nonempty_strings(self) -> None:
        """Every series ID is a non-empty string."""
        for entry in SERIES:
            assert isinstance(entry["id"], str) and len(entry["id"]) > 0

    def test_names_are_nonempty_strings(self) -> None:
        """Every series name is a non-empty string."""
        for entry in SERIES:
            assert isinstance(entry["name"], str) and len(entry["name"]) > 0

    def test_frequencies_are_valid(self) -> None:
        """Every frequency is one of daily, monthly, or quarterly."""
        for entry in SERIES:
            assert entry["frequency"] in VALID_FREQUENCIES, (
                f"{entry['id']} has invalid frequency: {entry['frequency']}"
            )

    def test_categories_are_valid(self) -> None:
        """Every category matches the expected set."""
        for entry in SERIES:
            assert entry["category"] in EXPECTED_CATEGORIES, (
                f"{entry['id']} has unexpected category: {entry['category']}"
            )


class TestSeriesUniqueness:
    """Ensure no duplicate IDs or names."""

    def test_ids_are_unique(self) -> None:
        """No two series share the same ID."""
        ids: list[str] = [s["id"] for s in SERIES]
        assert len(ids) == len(set(ids))

    def test_names_are_unique(self) -> None:
        """No two series share the same display name."""
        names: list[str] = [s["name"] for s in SERIES]
        assert len(names) == len(set(names))


class TestSeriesIdsList:
    """Validate the SERIES_IDS convenience list."""

    def test_series_ids_matches_series(self) -> None:
        """SERIES_IDS is derived from SERIES in the same order."""
        expected: list[str] = [s["id"] for s in SERIES]
        assert SERIES_IDS == expected

    def test_series_ids_length(self) -> None:
        """SERIES_IDS has the same count as SERIES."""
        assert len(SERIES_IDS) == len(SERIES)
