# tests/test_era5.py
"""Tests for pyradtran.era5: normalisation, atmosphere files, cloud profiles."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pyradtran.era5 import (
    cloud_profiles,
    era5_atmosphere_file,
    normalize_era5,
    recommend_atmosphere,
    select_profile,
    write_atmosphere_file,
)
from pyradtran.exceptions import InputGenerationError


N = 13
P_HPA = np.linspace(1000, 100, N)


def _column(**extra_vars):
    """Canonical single-column dataset (CDS short names)."""
    data = {
        "t": (["pressure_level"], np.linspace(290, 215, N), {"units": "K"}),
        "q": (["pressure_level"], np.linspace(1e-2, 1e-5, N), {"units": "kg kg-1"}),
        "z": (
            ["pressure_level"],
            np.linspace(0, 160_000, N),
            {"units": "m2 s-2"},
        ),
    }
    data.update(extra_vars)
    return xr.Dataset(
        data,
        coords={
            "pressure_level": (["pressure_level"], P_HPA, {"units": "hPa"}),
            "valid_time": pd.Timestamp("2022-03-15T12:00"),
            "latitude": 70.0,
            "longitude": 25.0,
        },
    )


def _arco_style():
    """ARCO-ERA5 long names, `level`/`time` coords, no unit attrs."""
    return xr.Dataset(
        {
            "temperature": (["time", "level", "latitude", "longitude"],
                            np.full((2, N, 3, 3), 250.0)),
            "specific_humidity": (["time", "level", "latitude", "longitude"],
                                  np.full((2, N, 3, 3), 1e-3)),
            "geopotential": (["time", "level", "latitude", "longitude"],
                             np.tile(np.linspace(0, 160_000, N)[None, :, None, None],
                                     (2, 1, 3, 3))),
            "ozone_mass_mixing_ratio": (["time", "level", "latitude", "longitude"],
                                        np.full((2, N, 3, 3), 5e-6)),
            "specific_cloud_liquid_water_content": (
                ["time", "level", "latitude", "longitude"],
                np.full((2, N, 3, 3), 0.0)),
        },
        coords={
            "time": pd.date_range("2022-03-15", periods=2, freq="6h"),
            "level": ("level", P_HPA, {"units": "millibars"}),
            "latitude": [69.0, 70.0, 71.0],
            "longitude": [24.0, 25.0, 26.0],
        },
    )


@pytest.mark.unit
class TestNormalize:
    def test_arco_names_mapped(self):
        ds = normalize_era5(_arco_style())
        for var in ("t", "q", "z", "o3", "clwc"):
            assert var in ds.variables
        assert "pressure_level" in ds.coords
        assert "valid_time" in ds.coords

    def test_idempotent(self):
        ds = normalize_era5(normalize_era5(_arco_style()))
        assert "t" in ds.variables

    def test_canonical_passthrough(self):
        ds = normalize_era5(_column())
        assert "t" in ds.variables and "q" in ds.variables

    def test_pa_pressure_converted(self):
        ds = _column()
        ds = ds.assign_coords(pressure_level=P_HPA * 100)
        ds["pressure_level"].attrs["units"] = "Pa"
        out = normalize_era5(ds)
        assert out["pressure_level"].values.max() == pytest.approx(1000.0)
        assert out["pressure_level"].attrs["units"] == "hPa"

    def test_unitless_pa_heuristic(self):
        ds = _column()
        ds = ds.assign_coords(pressure_level=P_HPA * 100)
        ds["pressure_level"].attrs.pop("units", None)
        out = normalize_era5(ds)
        assert out["pressure_level"].values.max() == pytest.approx(1000.0)

    def test_millibars_accepted_as_hpa(self):
        ds = _column()
        ds["pressure_level"].attrs["units"] = "millibars"
        out = normalize_era5(ds)
        assert out["pressure_level"].values.max() == pytest.approx(1000.0)

    def test_bogus_pressure_unit_raises(self):
        ds = _column()
        ds["pressure_level"].attrs["units"] = "furlongs"
        with pytest.raises(InputGenerationError, match="furlongs"):
            normalize_era5(ds)

    def test_units_restored(self):
        ds = normalize_era5(_arco_style())
        assert ds["t"].attrs["units"] == "K"
        assert ds["q"].attrs["units"] == "kg kg-1"


@pytest.mark.unit
class TestSelectProfile:
    def test_full_grid_selection(self):
        ds = normalize_era5(_arco_style())
        prof = select_profile(ds, 70.0, 25.0, "2022-03-15T06:00")
        assert list(prof.dims) == ["pressure_level"]

    def test_pre_selected_column(self):
        prof = select_profile(normalize_era5(_column()))
        assert list(prof.dims) == ["pressure_level"]

    def test_missing_coords_raise(self):
        ds = normalize_era5(_arco_style())
        with pytest.raises(ValueError, match="latitude"):
            select_profile(ds)


@pytest.mark.unit
@pytest.mark.io
class TestAtmosphereFile:
    def test_three_columns_without_ozone(self, tmp_path):
        out = write_atmosphere_file(_column(), tmp_path / "atm.dat")
        rows = [l.split() for l in out.read_text().splitlines()
                if l.strip() and not l.startswith("#")]
        assert rows and all(len(r) == 3 for r in rows)

    def test_four_columns_with_ozone(self, tmp_path):
        ds = _column(o3=(["pressure_level"], np.full(N, 5e-6), {"units": "kg kg-1"}))
        out = write_atmosphere_file(ds, tmp_path / "atm.dat")
        content = out.read_text()
        assert "# columns: H2O MMR O3 MMR" in content
        rows = [l.split() for l in content.splitlines()
                if l.strip() and not l.startswith("#")]
        assert all(len(r) == 4 for r in rows)

    def test_column_header_without_ozone(self, tmp_path):
        out = write_atmosphere_file(_column(), tmp_path / "atm.dat")
        assert "# columns: H2O MMR" in out.read_text()

    def test_toa_first_strictly_monotonic(self, tmp_path):
        out = write_atmosphere_file(_column(), tmp_path / "atm.dat")
        p = [float(l.split()[0]) for l in out.read_text().splitlines()
             if l.strip() and not l.startswith("#")]
        assert all(b > a for a, b in zip(p, p[1:]))

    def test_nan_levels_dropped(self, tmp_path):
        ds = _column()
        t = ds["t"].values.copy()
        t[0] = np.nan
        ds["t"].values = t
        out = write_atmosphere_file(ds, tmp_path / "atm.dat")
        rows = [l for l in out.read_text().splitlines()
                if l.strip() and not l.startswith("#")]
        assert len(rows) == N - 1

    def test_nan_ozone_becomes_minus_one(self, tmp_path):
        o3 = np.full(N, 5e-6)
        o3[3] = np.nan
        ds = _column(o3=(["pressure_level"], o3, {"units": "kg kg-1"}))
        out = write_atmosphere_file(ds, tmp_path / "atm.dat")
        vals = [float(l.split()[3]) for l in out.read_text().splitlines()
                if l.strip() and not l.startswith("#")]
        assert -1.0 in vals

    def test_missing_q_raises(self, tmp_path):
        with pytest.raises(InputGenerationError, match="q"):
            write_atmosphere_file(_column().drop_vars("q"), tmp_path / "atm.dat")

    def test_one_step_from_arco(self, tmp_path):
        out = era5_atmosphere_file(
            _arco_style(), 70.0, 25.0, "2022-03-15T00:00", tmp_path / "atm.dat"
        )
        assert Path(out).exists()
        assert "O3 MMR" in out.read_text()


@pytest.mark.unit
class TestCloudProfiles:
    def _cloudy_column(self):
        clwc = np.where((P_HPA > 700) & (P_HPA < 950), 1e-4, 0.0)
        ciwc = np.where((P_HPA > 300) & (P_HPA < 500), 5e-5, 0.0)
        return _column(
            clwc=(["pressure_level"], clwc, {"units": "kg kg-1"}),
            ciwc=(["pressure_level"], ciwc, {"units": "kg kg-1"}),
        )

    def test_both_phases_extracted(self):
        wc, ic = cloud_profiles(self._cloudy_column())
        assert wc is not None and ic is not None
        assert "lwc" in wc and "iwc" in ic

    def test_no_cloud_returns_none(self):
        wc, ic = cloud_profiles(_column())
        assert wc is None and ic is None

    def test_z_descending_top_row_zero(self):
        wc, _ = cloud_profiles(self._cloudy_column())
        z = wc["z"]
        assert all(b < a for a, b in zip(z, z[1:]))
        assert wc["lwc"][0] == 0.0
        assert any(v > 0 for v in wc["lwc"][1:])

    def test_mmr_converted_to_g_m3(self):
        wc, _ = cloud_profiles(self._cloudy_column())
        # 1e-4 kg/kg at ~900 hPa, ~285 K: rho ~ 1.1 kg/m3 -> ~0.11 g/m3
        assert 0.05 < max(wc["lwc"]) < 0.5

    def test_reff_defaults(self):
        wc, ic = cloud_profiles(self._cloudy_column())
        assert set(wc["reff"]) == {10.0}
        assert set(ic["reff"]) == {20.0}

    def test_matching_lengths(self):
        wc, ic = cloud_profiles(self._cloudy_column())
        assert len(wc["z"]) == len(wc["lwc"]) == len(wc["reff"])
        assert len(ic["z"]) == len(ic["iwc"]) == len(ic["reff"])

    def test_low_top_profile_uses_z_units_not_magnitude(self):
        # Profile top at ~9 km: max |z| ≈ 88 000 m2 s-2, below the 100 000
        # magnitude threshold — the units attribute must still make it be
        # read as geopotential, not as height in metres.
        n = 5
        ds = xr.Dataset(
            {
                "t": (
                    ["pressure_level"],
                    np.array([285.0, 275.0, 255.0, 245.0, 230.0]),
                    {"units": "K"},
                ),
                "clwc": (
                    ["pressure_level"],
                    np.array([5e-4, 0.0, 0.0, 0.0, 0.0]),
                    {"units": "kg kg-1"},
                ),
                "z": (
                    ["pressure_level"],
                    np.array([100.0, 2000.0, 5500.0, 7200.0, 9000.0]) * 9.80665,
                    {"units": "m**2 s**-2"},
                ),
            },
            coords={
                "pressure_level": (
                    ["pressure_level"],
                    np.array([1000.0, 800.0, 500.0, 400.0, 300.0]),
                    {"units": "hPa"},
                )
            },
        )
        wc, _ = cloud_profiles(ds)
        assert wc is not None
        # near-surface cloud must stay near the surface
        assert max(wc["z"]) < 3.0


@pytest.mark.unit
class TestRecommendAtmosphere:
    @pytest.mark.parametrize(
        "lat,time,expected",
        [
            (0.0, "2022-01-01", "afglt"),
            (78.0, "2022-01-15", "afglsw"),
            (78.0, "2022-07-15", "afglss"),
            (45.0, "2022-07-15", "afglms"),
            (45.0, "2022-12-15", "afglmw"),
            (-45.0, "2022-12-15", "afglms"),  # SH summer
            (-70.0, "2022-07-15", "afglsw"),  # SH winter
        ],
    )
    def test_bands_and_seasons(self, lat, time, expected):
        assert recommend_atmosphere(lat, time) == expected
