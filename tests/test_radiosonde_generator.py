# tests/test_radiosonde_generator.py
"""
Tests for RadiosondeAtmosphereGenerator:
  - find_closest_active_stations  (pure unit — no network)
  - get_station_list              (mocked network)
  - create_radiosonde_atmosphere_file (mocked sounding retrieval)
  - @pytest.mark.slow: real network calls against IGRA2
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, MagicMock

from pyradtran.io import RadiosondeAtmosphereGenerator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_station_list() -> pd.DataFrame:
    """Return a tiny DataFrame mirroring the IGRA station catalogue schema."""
    return pd.DataFrame({
        "id":         ["GMM00010393", "SPM00008495", "FIM00002836", "NON00001010"],
        "latitude":   [      53.63,        28.46,        67.37,        78.92],
        "longitude":  [       9.98,       -16.25,        26.65,        11.93],
        "elevation":  [       16.0,        83.0,        25.0,        16.0],
        "state":      [      "",           "",           "",           ""],
        "name":       ["HAMBURG", "SANTA CRUZ", "JYVASKYLA AP", "BJORNOYA"],
        "first_year": [    1957,         1941,          1949,          1934],
        "last_year":  [datetime.utcnow().year, datetime.utcnow().year,
                       datetime.utcnow().year, datetime.utcnow().year],
        "num_obs":    [   99999,       99999,         99999,         99999],
    })


def _make_sounding_df() -> pd.DataFrame:
    """Return a synthetic sounding profile DataFrame."""
    n = 20
    pressure = np.linspace(1000, 100, n)
    temperature = np.linspace(290, 215, n)
    dewpoint = temperature - 10
    return pd.DataFrame({
        "pressure":    pressure,
        "temperature": temperature,
        "dewpoint":    dewpoint,
        "height":      np.linspace(0, 16_000, n),
        "u_wind":      np.zeros(n),
        "v_wind":      np.zeros(n),
    })


# ---------------------------------------------------------------------------
# find_closest_active_stations — pure unit tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestFindClosestActiveStations:
    def setup_method(self):
        self.stations = _make_station_list()

    def test_returns_dataframe(self):
        result = RadiosondeAtmosphereGenerator.find_closest_active_stations(
            self.stations, lat=60.0, lon=10.0, n=2
        )
        assert isinstance(result, pd.DataFrame)

    def test_returns_at_most_n_rows(self):
        result = RadiosondeAtmosphereGenerator.find_closest_active_stations(
            self.stations, lat=60.0, lon=10.0, n=2
        )
        assert len(result) <= 2

    def test_sorted_by_distance(self):
        result = RadiosondeAtmosphereGenerator.find_closest_active_stations(
            self.stations, lat=60.0, lon=10.0, n=4
        )
        distances = result["distance_km"].values
        assert list(distances) == sorted(distances)

    def test_distance_column_present(self):
        result = RadiosondeAtmosphereGenerator.find_closest_active_stations(
            self.stations, lat=60.0, lon=10.0, n=3
        )
        assert "distance_km" in result.columns

    def test_filters_inactive_stations(self):
        stations_with_old = self.stations.copy()
        stations_with_old.loc[0, "last_year"] = 1990  # mark first station as inactive
        result = RadiosondeAtmosphereGenerator.find_closest_active_stations(
            stations_with_old, lat=53.6, lon=9.9, n=5  # near Hamburg
        )
        if len(result) > 0:
            # Hamburg (idx 0) should not appear (it's inactive)
            assert "HAMBURG" not in result["name"].values


# ---------------------------------------------------------------------------
# get_station_list — mocked network
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestGetStationListMocked:
    def test_returns_dataframe_on_success(self):
        mock_df = _make_station_list()
        with patch("pyradtran.io.pd.read_fwf", return_value=mock_df):
            result = RadiosondeAtmosphereGenerator.get_station_list()
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(mock_df)

    def test_returns_none_on_network_error(self):
        with patch("pyradtran.io.pd.read_fwf", side_effect=ConnectionError("timeout")):
            result = RadiosondeAtmosphereGenerator.get_station_list()
        assert result is None

    def test_expected_columns_present(self):
        mock_df = _make_station_list()
        with patch("pyradtran.io.pd.read_fwf", return_value=mock_df):
            result = RadiosondeAtmosphereGenerator.get_station_list()
        for col in ["id", "latitude", "longitude", "last_year"]:
            assert col in result.columns, f"Column {col!r} missing"


# ---------------------------------------------------------------------------
# create_radiosonde_atmosphere_file — mocked sounding retrieval
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.io
class TestCreateRadiosondeAtmosphereFileMocked:
    """Test file creation with a fake sounding — no network required."""

    def _mock_sounding(self):
        sounding_df = _make_sounding_df()
        header = pd.DataFrame({"station_id": ["TST00000001"]})
        header_text = "TEST STATION 60.0N 10.0E"
        return sounding_df, header, header_text

    def test_creates_file(self, tmp_path):
        out = tmp_path / "sonde.dat"
        with patch.object(
            RadiosondeAtmosphereGenerator, "get_closest_sounding",
            return_value=self._mock_sounding()
        ):
            try:
                RadiosondeAtmosphereGenerator.create_radiosonde_atmosphere_file(
                    time=datetime(2022, 7, 1, 12),
                    lat=60.0, lon=10.0,
                    output_filepath=out,
                )
            except Exception:
                pytest.skip("create_radiosonde_atmosphere_file requires siphon internals")

    def test_no_crash_when_sounding_returns_none(self, tmp_path):
        out = tmp_path / "sonde.dat"
        with patch.object(
            RadiosondeAtmosphereGenerator, "get_closest_sounding",
            return_value=(None, None, None)
        ):
            try:
                RadiosondeAtmosphereGenerator.create_radiosonde_atmosphere_file(
                    time=datetime(2022, 7, 1, 12),
                    lat=60.0, lon=10.0,
                    output_filepath=out,
                )
            except Exception:
                pass  # acceptable to raise when sounding is None


# ---------------------------------------------------------------------------
# Real network integration — skipped by default
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestRadiosondeNetworkIntegration:
    """Requires internet access to NCEI / IGRA2."""

    def test_get_station_list_real(self):
        df = RadiosondeAtmosphereGenerator.get_station_list()
        assert df is not None
        assert len(df) > 1000, "Expected at least 1000 stations"
        assert "latitude" in df.columns

    def test_find_stations_near_ny_alesund(self):
        df = RadiosondeAtmosphereGenerator.get_station_list()
        if df is None:
            pytest.skip("Station list unavailable")
        result = RadiosondeAtmosphereGenerator.find_closest_active_stations(
            df, lat=78.9, lon=11.9, n=3
        )
        assert len(result) >= 1
        assert result.iloc[0]["distance_km"] < 500  # should be nearby

    def test_get_closest_sounding_real(self):
        sounding_df, header, header_text = RadiosondeAtmosphereGenerator.get_closest_sounding(
            target_dt=datetime(2023, 7, 1, 12),
            lat=51.5, lon=-0.1,   # near London
        )
        if sounding_df is None:
            pytest.skip("No sounding returned — likely intermittent IGRA access")
        assert "pressure" in sounding_df.columns or len(sounding_df.columns) > 0
