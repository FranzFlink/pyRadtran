# tests/test_era5_atmosphere.py
"""
Tests for ERA5AtmosphereGenerator.create_era5_atmosphere_file.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pyradtran.io import ERA5AtmosphereGenerator

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_era5_profile(n_levels: int = 13) -> xr.Dataset:
    """Return a minimal synthetic ERA5 profile dataset."""
    pressure_hpa = np.linspace(1000, 100, n_levels)
    geopotential = np.linspace(0, 160_000, n_levels)  # m²/s²
    temperature = np.linspace(290, 215, n_levels)  # K
    q = np.linspace(1e-2, 1e-5, n_levels)  # kg/kg

    return xr.Dataset(
        {
            "z": (["pressure_level"], geopotential, {"units": "m2 s-2"}),
            "t": (["pressure_level"], temperature, {"units": "K"}),
            "q": (["pressure_level"], q, {"units": "kg kg-1"}),
        },
        coords={
            "pressure_level": (["pressure_level"], pressure_hpa, {"units": "hPa"}),
            "valid_time": pd.Timestamp("2022-03-15T12:00"),
            "latitude": 70.0,
            "longitude": 25.0,
        },
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.io
class TestERA5AtmosphereGenerator:
    def test_creates_file(self, tmp_path):
        ds = _make_era5_profile()
        out = tmp_path / "atm.dat"
        result = ERA5AtmosphereGenerator.create_era5_atmosphere_file(
            ds,
            latitude=70.0,
            longitude=25.0,
            time="2022-03-15T12:00",
            output_filepath=out,
        )
        assert out.exists()

    def test_returns_path(self, tmp_path):
        ds = _make_era5_profile()
        out = tmp_path / "atm.dat"
        result = ERA5AtmosphereGenerator.create_era5_atmosphere_file(
            ds,
            latitude=70.0,
            longitude=25.0,
            time="2022-03-15T12:00",
            output_filepath=out,
        )
        assert Path(result) == out

    def test_file_has_header(self, tmp_path):
        ds = _make_era5_profile()
        out = tmp_path / "atm.dat"
        ERA5AtmosphereGenerator.create_era5_atmosphere_file(
            ds, 70.0, 25.0, "2022-03-15T12:00", out
        )
        content = out.read_text()
        assert "#" in content, "Expected at least one comment/header line"

    def test_file_has_data_rows(self, tmp_path):
        ds = _make_era5_profile()
        out = tmp_path / "atm.dat"
        ERA5AtmosphereGenerator.create_era5_atmosphere_file(
            ds, 70.0, 25.0, "2022-03-15T12:00", out
        )
        data_lines = [
            l
            for l in out.read_text().splitlines()
            if l.strip() and not l.startswith("#")
        ]
        assert len(data_lines) > 0

    def test_three_columns_per_data_row(self, tmp_path):
        ds = _make_era5_profile()
        out = tmp_path / "atm.dat"
        ERA5AtmosphereGenerator.create_era5_atmosphere_file(
            ds, 70.0, 25.0, "2022-03-15T12:00", out
        )
        data_lines = [
            l
            for l in out.read_text().splitlines()
            if l.strip() and not l.startswith("#")
        ]
        for line in data_lines:
            cols = line.split()
            assert len(cols) == 3, f"Expected 3 columns, got: {line!r}"

    def test_pressure_values_positive(self, tmp_path):
        ds = _make_era5_profile()
        out = tmp_path / "atm.dat"
        ERA5AtmosphereGenerator.create_era5_atmosphere_file(
            ds, 70.0, 25.0, "2022-03-15T12:00", out
        )
        data_lines = [
            l
            for l in out.read_text().splitlines()
            if l.strip() and not l.startswith("#")
        ]
        for line in data_lines:
            p = float(line.split()[0])
            assert p > 0, f"Pressure must be positive, got {p}"

    def test_temperatures_physically_reasonable(self, tmp_path):
        ds = _make_era5_profile()
        out = tmp_path / "atm.dat"
        ERA5AtmosphereGenerator.create_era5_atmosphere_file(
            ds, 70.0, 25.0, "2022-03-15T12:00", out
        )
        data_lines = [
            l
            for l in out.read_text().splitlines()
            if l.strip() and not l.startswith("#")
        ]
        for line in data_lines:
            t = float(line.split()[1])
            assert 100 <= t <= 400, f"Temperature {t} K outside physical range"

    def test_creates_parent_directory(self, tmp_path):
        ds = _make_era5_profile()
        out = tmp_path / "new_subdir" / "atm.dat"
        ERA5AtmosphereGenerator.create_era5_atmosphere_file(
            ds, 70.0, 25.0, "2022-03-15T12:00", out
        )
        assert out.exists()

    def test_missing_variable_raises(self, tmp_path):
        ds = _make_era5_profile().drop_vars("q")
        out = tmp_path / "atm.dat"
        with pytest.raises(Exception, match="q"):
            ERA5AtmosphereGenerator.create_era5_atmosphere_file(
                ds, 70.0, 25.0, "2022-03-15T12:00", out
            )


from pyradtran.exceptions import InputGenerationError


class TestUnitHandling:
    def test_unknown_pressure_unit_raises_clear_error(
        self, synthetic_era5_ds, tmp_path
    ):
        # "millibars" (ARCO-ERA5) is accepted as hPa; a truly unknown
        # unit must still fail loudly.
        ds = synthetic_era5_ds.copy(deep=True)
        ds["pressure_level"].attrs["units"] = "furlongs"
        with pytest.raises(InputGenerationError, match="furlongs"):
            ERA5AtmosphereGenerator.create_era5_atmosphere_file(
                ds, 70.0, 25.0, "2022-07-01T12:00", tmp_path / "atm.dat"
            )

    def test_missing_q_units_defaults_to_kg_kg(self, synthetic_era5_ds, tmp_path):
        ds = synthetic_era5_ds.copy(deep=True)
        ds["q"].attrs.pop("units", None)
        out = ERA5AtmosphereGenerator.create_era5_atmosphere_file(
            ds, 70.0, 25.0, "2022-07-01T12:00", tmp_path / "atm.dat"
        )
        assert out.exists()
