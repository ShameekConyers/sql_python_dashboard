"""Tests for src/data_pull.py.

All tests mock the FRED API so no network calls or API key is required.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

import data_pull


# ---------------------------------------------------------------------------
# _load_api_key
# ---------------------------------------------------------------------------


class TestLoadApiKey:
    """Tests for _load_api_key."""

    @patch.dict("os.environ", {"FRED_API_KEY": "abc123"})
    @patch("data_pull.load_dotenv")
    def test_returns_key_when_set(self, mock_dotenv: MagicMock) -> None:
        """Returns the key value when FRED_API_KEY is present in env."""
        assert data_pull._load_api_key() == "abc123"

    @patch.dict("os.environ", {"FRED_API_KEY": ""}, clear=False)
    @patch("data_pull.load_dotenv")
    def test_exits_when_key_empty(self, mock_dotenv: MagicMock) -> None:
        """Exits with code 1 when FRED_API_KEY is empty."""
        with pytest.raises(SystemExit) as exc_info:
            data_pull._load_api_key()
        assert exc_info.value.code == 1

    @patch.dict("os.environ", {"FRED_API_KEY": "your_key_here"}, clear=False)
    @patch("data_pull.load_dotenv")
    def test_exits_when_key_is_placeholder(self, mock_dotenv: MagicMock) -> None:
        """Exits with code 1 when FRED_API_KEY is the .env.example placeholder."""
        with pytest.raises(SystemExit) as exc_info:
            data_pull._load_api_key()
        assert exc_info.value.code == 1

    @patch.dict("os.environ", {}, clear=True)
    @patch("data_pull.load_dotenv")
    def test_exits_when_key_missing(self, mock_dotenv: MagicMock) -> None:
        """Exits with code 1 when FRED_API_KEY is not in env at all."""
        with pytest.raises(SystemExit) as exc_info:
            data_pull._load_api_key()
        assert exc_info.value.code == 1


# ---------------------------------------------------------------------------
# _output_path
# ---------------------------------------------------------------------------


class TestOutputPath:
    """Tests for _output_path."""

    def test_returns_json_path_in_raw_dir(self) -> None:
        """Returns a path ending in data/raw/<series_id>.json."""
        result: Path = data_pull._output_path("UNRATE")
        assert result.name == "UNRATE.json"
        assert result.parent == data_pull.RAW_DIR

    def test_different_series_produce_different_paths(self) -> None:
        """Each series ID maps to a unique file path."""
        assert data_pull._output_path("UNRATE") != data_pull._output_path("GDPC1")


# ---------------------------------------------------------------------------
# _metadata_path
# ---------------------------------------------------------------------------


class TestMetadataPath:
    """Tests for _metadata_path."""

    def test_returns_metadata_json_path(self) -> None:
        """Returns a path ending in data/raw/<series_id>_metadata.json."""
        result: Path = data_pull._metadata_path("UNRATE")
        assert result.name == "UNRATE_metadata.json"
        assert result.parent == data_pull.RAW_DIR

    def test_distinct_from_observation_path(self) -> None:
        """Metadata path differs from the observations cache path."""
        obs: Path = data_pull._output_path("UNRATE")
        meta: Path = data_pull._metadata_path("UNRATE")
        assert obs != meta


# ---------------------------------------------------------------------------
# _pull_series
# ---------------------------------------------------------------------------


def _make_mock_fred(
    values: list[float],
    dates: list[str] | None = None,
    info_extras: dict | None = None,
) -> MagicMock:
    """Build a mock Fred client that returns the given observations.

    Args:
        values: List of float values (use float('nan') for missing).
        dates: Optional date strings. Defaults to 2020-01-01 through 2020-01-N.
        info_extras: Extra key/value pairs for get_series_info.

    Returns:
        A MagicMock mimicking fredapi.Fred.
    """
    if dates is None:
        dates = [f"2020-01-{i + 1:02d}" for i in range(len(values))]

    index = pd.to_datetime(dates)
    series = pd.Series(values, index=index)

    info_dict: dict = {
        "units": "Percent",
        "seasonal_adjustment": "Seasonally Adjusted",
        "last_updated": "2026-01-01 08:00:00-06",
    }
    if info_extras:
        info_dict.update(info_extras)

    mock_fred = MagicMock()
    mock_fred.get_series.return_value = series
    mock_fred.get_series_info.return_value = info_dict
    return mock_fred


class TestPullSeries:
    """Tests for _pull_series."""

    def test_returns_correct_structure(self, sample_series_info: dict) -> None:
        """Returned dict contains all expected top-level keys."""
        fred = _make_mock_fred([3.5, 3.6, 3.7])
        result: dict = data_pull._pull_series(fred, sample_series_info)

        assert result["series_id"] == "UNRATE"
        assert result["name"] == "Unemployment Rate"
        assert result["category"] == "labor_market"
        assert result["frequency"] == "monthly"
        assert result["units"] == "Percent"
        assert result["observation_count"] == 3
        assert len(result["observations"]) == 3

    def test_observations_have_date_and_value(self, sample_series_info: dict) -> None:
        """Each observation record has 'date' and 'value' keys."""
        fred = _make_mock_fred([3.5])
        result: dict = data_pull._pull_series(fred, sample_series_info)
        obs: dict = result["observations"][0]

        assert obs["date"] == "2020-01-01"
        assert obs["value"] == 3.5

    def test_nan_values_become_none(self, sample_series_info: dict) -> None:
        """NaN observations are stored as None (JSON null)."""
        fred = _make_mock_fred([float("nan")])
        result: dict = data_pull._pull_series(fred, sample_series_info)

        assert result["observations"][0]["value"] is None

    def test_result_is_json_serializable(self, sample_series_info: dict) -> None:
        """The returned dict can be serialized to JSON without errors."""
        fred = _make_mock_fred([1.0, float("nan"), 3.0])
        result: dict = data_pull._pull_series(fred, sample_series_info)

        serialized: str = json.dumps(result)
        assert isinstance(serialized, str)

    def test_calls_fred_api_with_correct_series_id(
        self, sample_series_info: dict
    ) -> None:
        """Passes the series ID from series_info to the Fred client."""
        fred = _make_mock_fred([1.0])
        data_pull._pull_series(fred, sample_series_info)

        fred.get_series.assert_called_once_with("UNRATE")
        fred.get_series_info.assert_called_once_with("UNRATE")


# ---------------------------------------------------------------------------
# pull_series_list
# ---------------------------------------------------------------------------


class TestPullSeriesList:
    """Tests for pull_series_list."""

    @patch("data_pull.time.sleep")
    @patch("data_pull._load_api_key", return_value="fake_key")
    @patch("data_pull.Fred")
    def test_writes_json_files(
        self,
        mock_fred_cls: MagicMock,
        mock_key: MagicMock,
        mock_sleep: MagicMock,
        sample_series_info: dict,
        tmp_path: Path,
    ) -> None:
        """Creates a JSON file for each series in the output directory."""
        mock_fred_cls.return_value = _make_mock_fred([3.5, 3.6])

        with patch.object(data_pull, "RAW_DIR", tmp_path):
            data_pull.pull_series_list([sample_series_info])

        out_file: Path = tmp_path / "UNRATE.json"
        assert out_file.exists()

        data: dict = json.loads(out_file.read_text())
        assert data["series_id"] == "UNRATE"
        assert data["observation_count"] == 2

    @patch("data_pull.time.sleep")
    @patch("data_pull._load_api_key", return_value="fake_key")
    @patch("data_pull.Fred")
    def test_skips_cached_series_without_refresh(
        self,
        mock_fred_cls: MagicMock,
        mock_key: MagicMock,
        mock_sleep: MagicMock,
        sample_series_info: dict,
        tmp_path: Path,
    ) -> None:
        """Skips pulling a series when its JSON file already exists and refresh is False."""
        cached_file: Path = tmp_path / "UNRATE.json"
        cached_file.write_text('{"old": true}')

        mock_fred_cls.return_value = _make_mock_fred([3.5])

        with patch.object(data_pull, "RAW_DIR", tmp_path):
            data_pull.pull_series_list([sample_series_info], refresh=False)

        # File should still contain old content (not overwritten).
        assert json.loads(cached_file.read_text()) == {"old": True}
        mock_fred_cls.return_value.get_series.assert_not_called()

    @patch("data_pull.time.sleep")
    @patch("data_pull._load_api_key", return_value="fake_key")
    @patch("data_pull.Fred")
    def test_refresh_overwrites_cached_series(
        self,
        mock_fred_cls: MagicMock,
        mock_key: MagicMock,
        mock_sleep: MagicMock,
        sample_series_info: dict,
        tmp_path: Path,
    ) -> None:
        """Re-pulls and overwrites when refresh=True even if cached."""
        cached_file: Path = tmp_path / "UNRATE.json"
        cached_file.write_text('{"old": true}')

        mock_fred_cls.return_value = _make_mock_fred([3.5])

        with patch.object(data_pull, "RAW_DIR", tmp_path):
            data_pull.pull_series_list([sample_series_info], refresh=True)

        data: dict = json.loads(cached_file.read_text())
        assert data["series_id"] == "UNRATE"

    @patch("data_pull.time.sleep")
    @patch("data_pull._load_api_key", return_value="fake_key")
    @patch("data_pull.Fred")
    def test_rate_limit_sleep_between_series(
        self,
        mock_fred_cls: MagicMock,
        mock_key: MagicMock,
        mock_sleep: MagicMock,
        two_series_list: list[dict],
        tmp_path: Path,
    ) -> None:
        """Sleeps between series pulls to respect the rate limit."""
        mock_fred_cls.return_value = _make_mock_fred([1.0])

        with patch.object(data_pull, "RAW_DIR", tmp_path):
            data_pull.pull_series_list(two_series_list)

        # Should sleep once (between series 1 and 2, not after the last).
        mock_sleep.assert_called_once_with(data_pull.RATE_LIMIT_DELAY)

    @patch("data_pull.time.sleep")
    @patch("data_pull._load_api_key", return_value="fake_key")
    @patch("data_pull.Fred")
    def test_no_sleep_for_single_series(
        self,
        mock_fred_cls: MagicMock,
        mock_key: MagicMock,
        mock_sleep: MagicMock,
        sample_series_info: dict,
        tmp_path: Path,
    ) -> None:
        """No rate limit sleep when pulling only one series."""
        mock_fred_cls.return_value = _make_mock_fred([1.0])

        with patch.object(data_pull, "RAW_DIR", tmp_path):
            data_pull.pull_series_list([sample_series_info])

        mock_sleep.assert_not_called()

    @patch("data_pull.time.sleep")
    @patch("data_pull._load_api_key", return_value="fake_key")
    @patch("data_pull.Fred")
    def test_continues_after_api_error(
        self,
        mock_fred_cls: MagicMock,
        mock_key: MagicMock,
        mock_sleep: MagicMock,
        two_series_list: list[dict],
        tmp_path: Path,
    ) -> None:
        """Continues pulling remaining series after one fails."""
        mock_fred = MagicMock()
        # First call raises, second succeeds.
        mock_fred.get_series.side_effect = [
            Exception("API error"),
            pd.Series([1.0], index=pd.to_datetime(["2020-01-01"])),
        ]
        mock_fred.get_series_info.return_value = {
            "units": "Percent",
            "seasonal_adjustment": "SA",
            "last_updated": "2026-01-01",
        }
        mock_fred_cls.return_value = mock_fred

        with patch.object(data_pull, "RAW_DIR", tmp_path):
            data_pull.pull_series_list(two_series_list)

        # First series failed, so no file. Second succeeded.
        assert not (tmp_path / "UNRATE.json").exists()
        assert (tmp_path / "GDPC1.json").exists()


# ---------------------------------------------------------------------------
# _pull_series_metadata
# ---------------------------------------------------------------------------


def _make_mock_http_response(payload: dict) -> MagicMock:
    """Build a mock httpx Response returning the given JSON payload.

    Args:
        payload: Dict to return from .json().

    Returns:
        A MagicMock configured to act like an httpx.Response.
    """
    response = MagicMock()
    response.json.return_value = payload
    response.raise_for_status.return_value = None
    return response


class TestPullSeriesMetadata:
    """Tests for _pull_series_metadata."""

    @patch("data_pull.time.sleep")
    @patch("data_pull.httpx.Client")
    def test_assembles_full_metadata_dict(
        self,
        mock_client_cls: MagicMock,
        mock_sleep: MagicMock,
    ) -> None:
        """Combines series info + release + category into one dict."""
        fred = MagicMock()
        fred.get_series_info.return_value = {
            "notes": "Measures unemployment...",
            "title": "Unemployment Rate",
            "units": "Percent",
            "seasonal_adjustment": "Seasonally Adjusted",
        }

        client_instance = MagicMock()
        # First .get() is release, second is leaf category, subsequent are parents.
        client_instance.get.side_effect = [
            _make_mock_http_response(
                {
                    "releases": [
                        {
                            "name": "Employment Situation",
                            "link": "https://www.bls.gov/empsit",
                            "notes": "Monthly BLS release.",
                        }
                    ]
                }
            ),
            _make_mock_http_response(
                {
                    "categories": [
                        {"id": 32, "name": "Unemployment Rate", "parent_id": 10}
                    ]
                }
            ),
            _make_mock_http_response(
                {
                    "categories": [
                        {"id": 10, "name": "Labor Market", "parent_id": 0}
                    ]
                }
            ),
        ]
        mock_client_cls.return_value.__enter__.return_value = client_instance

        metadata: dict = data_pull._pull_series_metadata(
            fred, "fake_key", "UNRATE"
        )

        assert metadata["series_id"] == "UNRATE"
        assert metadata["series_notes"] == "Measures unemployment..."
        assert metadata["title"] == "Unemployment Rate"
        assert metadata["release_name"] == "Employment Situation"
        assert metadata["release_link"] == "https://www.bls.gov/empsit"
        # Walked parent: "Labor Market > Unemployment Rate"
        assert metadata["category_path"] == "Labor Market > Unemployment Rate"
        # fetched_at is set to an ISO timestamp
        assert "T" in metadata["fetched_at"]

    @patch("data_pull.time.sleep")
    @patch("data_pull.httpx.Client")
    def test_series_info_failure_is_non_fatal(
        self,
        mock_client_cls: MagicMock,
        mock_sleep: MagicMock,
    ) -> None:
        """If get_series_info raises, the metadata dict still returns."""
        fred = MagicMock()
        fred.get_series_info.side_effect = Exception("API down")

        client_instance = MagicMock()
        client_instance.get.side_effect = [
            _make_mock_http_response({"releases": []}),
            _make_mock_http_response({"categories": []}),
        ]
        mock_client_cls.return_value.__enter__.return_value = client_instance

        metadata: dict = data_pull._pull_series_metadata(
            fred, "fake_key", "UNRATE"
        )

        assert metadata["series_id"] == "UNRATE"
        assert metadata["series_notes"] == ""
        assert metadata["release_name"] == ""
        assert metadata["category_path"] == ""

    @patch("data_pull.time.sleep")
    @patch("data_pull.httpx.Client")
    def test_release_failure_leaves_release_fields_blank(
        self,
        mock_client_cls: MagicMock,
        mock_sleep: MagicMock,
    ) -> None:
        """Release HTTP failure does not prevent category fetch."""
        fred = MagicMock()
        fred.get_series_info.return_value = {
            "notes": "notes",
            "title": "t",
            "units": "u",
            "seasonal_adjustment": "sa",
        }

        client_instance = MagicMock()
        # Release raises, category succeeds (leaf has no parent).
        client_instance.get.side_effect = [
            Exception("release down"),
            _make_mock_http_response(
                {"categories": [{"id": 1, "name": "Only", "parent_id": 0}]}
            ),
        ]
        mock_client_cls.return_value.__enter__.return_value = client_instance

        metadata: dict = data_pull._pull_series_metadata(
            fred, "fake_key", "UNRATE"
        )

        assert metadata["release_name"] == ""
        assert metadata["category_path"] == "Only"

    @patch("data_pull.time.sleep")
    @patch("data_pull.httpx.Client")
    def test_no_categories_returns_empty_path(
        self,
        mock_client_cls: MagicMock,
        mock_sleep: MagicMock,
    ) -> None:
        """When FRED returns no categories, category_path is blank."""
        fred = MagicMock()
        fred.get_series_info.return_value = {
            "notes": "",
            "title": "",
            "units": "",
            "seasonal_adjustment": "",
        }

        client_instance = MagicMock()
        client_instance.get.side_effect = [
            _make_mock_http_response({"releases": []}),
            _make_mock_http_response({"categories": []}),
        ]
        mock_client_cls.return_value.__enter__.return_value = client_instance

        metadata: dict = data_pull._pull_series_metadata(
            fred, "fake_key", "UNRATE"
        )

        assert metadata["category_path"] == ""


# ---------------------------------------------------------------------------
# pull_metadata_list
# ---------------------------------------------------------------------------


class TestPullMetadataList:
    """Tests for pull_metadata_list."""

    @patch("data_pull.time.sleep")
    @patch("data_pull._pull_series_metadata")
    @patch("data_pull._load_api_key", return_value="fake_key")
    @patch("data_pull.Fred")
    def test_writes_metadata_json(
        self,
        mock_fred_cls: MagicMock,
        mock_key: MagicMock,
        mock_pull: MagicMock,
        mock_sleep: MagicMock,
        sample_series_info: dict,
        tmp_path: Path,
    ) -> None:
        """Writes one metadata file per series."""
        mock_pull.return_value = {
            "series_id": "UNRATE",
            "series_notes": "notes",
            "title": "t",
            "units": "u",
            "seasonal_adjustment": "sa",
            "release_name": "rn",
            "release_link": "rl",
            "release_notes": "rnotes",
            "category_path": "A > B",
            "fetched_at": "2026-04-16T00:00:00+00:00",
        }

        with patch.object(data_pull, "RAW_DIR", tmp_path):
            data_pull.pull_metadata_list([sample_series_info])

        out_file: Path = tmp_path / "UNRATE_metadata.json"
        assert out_file.exists()
        data: dict = json.loads(out_file.read_text())
        assert data["series_id"] == "UNRATE"
        assert data["category_path"] == "A > B"

    @patch("data_pull.time.sleep")
    @patch("data_pull._pull_series_metadata")
    @patch("data_pull._load_api_key", return_value="fake_key")
    @patch("data_pull.Fred")
    def test_skips_cached_without_refresh(
        self,
        mock_fred_cls: MagicMock,
        mock_key: MagicMock,
        mock_pull: MagicMock,
        mock_sleep: MagicMock,
        sample_series_info: dict,
        tmp_path: Path,
    ) -> None:
        """Cached metadata is not re-pulled unless refresh=True."""
        cached: Path = tmp_path / "UNRATE_metadata.json"
        cached.write_text('{"cached": true}')

        with patch.object(data_pull, "RAW_DIR", tmp_path):
            data_pull.pull_metadata_list([sample_series_info], refresh=False)

        mock_pull.assert_not_called()
        assert json.loads(cached.read_text()) == {"cached": True}

    @patch("data_pull.time.sleep")
    @patch("data_pull._pull_series_metadata")
    @patch("data_pull._load_api_key", return_value="fake_key")
    @patch("data_pull.Fred")
    def test_continues_after_failure(
        self,
        mock_fred_cls: MagicMock,
        mock_key: MagicMock,
        mock_pull: MagicMock,
        mock_sleep: MagicMock,
        two_series_list: list[dict],
        tmp_path: Path,
    ) -> None:
        """One series failing does not halt the metadata loop."""
        mock_pull.side_effect = [
            Exception("boom"),
            {
                "series_id": "GDPC1",
                "series_notes": "",
                "title": "",
                "units": "",
                "seasonal_adjustment": "",
                "release_name": "",
                "release_link": "",
                "release_notes": "",
                "category_path": "",
                "fetched_at": "2026-04-16T00:00:00+00:00",
            },
        ]

        with patch.object(data_pull, "RAW_DIR", tmp_path):
            data_pull.pull_metadata_list(two_series_list)

        assert not (tmp_path / "UNRATE_metadata.json").exists()
        assert (tmp_path / "GDPC1_metadata.json").exists()


# ---------------------------------------------------------------------------
# main (integration-style with mocks)
# ---------------------------------------------------------------------------


class TestMain:
    """Tests for the main() entry point."""

    @patch("data_pull.pull_metadata_list")
    @patch("data_pull.pull_series_list")
    @patch("data_pull._parse_args")
    def test_filters_to_single_series(
        self,
        mock_args: MagicMock,
        mock_pull: MagicMock,
        mock_pull_meta: MagicMock,
    ) -> None:
        """--series flag filters to a single matching series."""
        mock_args.return_value = MagicMock(series="unrate", refresh=False)
        data_pull.main()

        called_list: list = mock_pull.call_args[0][0]
        assert len(called_list) == 1
        assert called_list[0]["id"] == "UNRATE"
        # Metadata pull also receives the filtered list
        meta_list: list = mock_pull_meta.call_args[0][0]
        assert len(meta_list) == 1

    @patch("data_pull.pull_metadata_list")
    @patch("data_pull.pull_series_list")
    @patch("data_pull._parse_args")
    def test_unknown_series_exits(
        self,
        mock_args: MagicMock,
        mock_pull: MagicMock,
        mock_pull_meta: MagicMock,
    ) -> None:
        """Exits with code 1 when --series doesn't match any configured series."""
        mock_args.return_value = MagicMock(series="NOTREAL", refresh=False)

        with pytest.raises(SystemExit) as exc_info:
            data_pull.main()
        assert exc_info.value.code == 1
        mock_pull.assert_not_called()
        mock_pull_meta.assert_not_called()

    @patch("data_pull.pull_metadata_list")
    @patch("data_pull.pull_series_list")
    @patch("data_pull._parse_args")
    def test_no_flags_pulls_all_series(
        self,
        mock_args: MagicMock,
        mock_pull: MagicMock,
        mock_pull_meta: MagicMock,
    ) -> None:
        """Without --series, passes the full SERIES list."""
        mock_args.return_value = MagicMock(series=None, refresh=False)
        data_pull.main()

        called_list: list = mock_pull.call_args[0][0]
        assert len(called_list) == len(data_pull.SERIES)
        mock_pull_meta.assert_called_once()

    @patch("data_pull.pull_metadata_list")
    @patch("data_pull.pull_series_list")
    @patch("data_pull._parse_args")
    def test_refresh_flag_passed_through(
        self,
        mock_args: MagicMock,
        mock_pull: MagicMock,
        mock_pull_meta: MagicMock,
    ) -> None:
        """The --refresh flag is forwarded to both pull functions."""
        mock_args.return_value = MagicMock(series=None, refresh=True)
        data_pull.main()

        assert (
            mock_pull.call_args[1].get("refresh") is True
            or (len(mock_pull.call_args[0]) > 1 and mock_pull.call_args[0][1] is True)
        )
        assert (
            mock_pull_meta.call_args[1].get("refresh") is True
            or (len(mock_pull_meta.call_args[0]) > 1 and mock_pull_meta.call_args[0][1] is True)
        )
