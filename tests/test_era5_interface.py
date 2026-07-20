# tests/test_era5_interface.py
"""End-to-end: raw ERA5 dataset drives atmosphere, ozone, and clouds."""

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from helpers import has_libradtran

import pyradtran  # noqa: F401  (registers the accessor)
from pyradtran.config import load_config
from pyradtran.exceptions import PyRadtranError

N = 13
P_HPA = np.linspace(1000, 100, N)


def _era5_ds(with_o3=True, with_clouds=True):
    """Raw CDS-style ERA5 dataset with time/lat/lon dims."""
    times = pd.date_range("2022-01-01 12:00", periods=1, freq="h")
    lats = [77.0, 78.0, 79.0]
    lons = [14.0, 15.0, 16.0]
    shape = (len(times), N, len(lats), len(lons))

    def full(profile):
        return np.tile(
            np.asarray(profile)[None, :, None, None],
            (len(times), 1, len(lats), len(lons)),
        )

    data = {
        "t": (
            ["valid_time", "pressure_level", "latitude", "longitude"],
            full(np.linspace(260, 215, N)),
        ),
        "q": (
            ["valid_time", "pressure_level", "latitude", "longitude"],
            full(np.linspace(2e-3, 1e-6, N)),
        ),
        "z": (
            ["valid_time", "pressure_level", "latitude", "longitude"],
            full(np.linspace(0, 160_000, N)),
        ),
    }
    if with_o3:
        data["o3"] = (
            ["valid_time", "pressure_level", "latitude", "longitude"],
            full(np.linspace(5e-8, 8e-6, N)),
        )
    if with_clouds:
        clwc = np.where((P_HPA > 700) & (P_HPA < 950), 1e-4, 0.0)
        data["clwc"] = (
            ["valid_time", "pressure_level", "latitude", "longitude"],
            full(clwc),
        )
    assert all(v[1].shape == shape for v in data.values())
    return xr.Dataset(
        data,
        coords={
            "valid_time": times,
            "pressure_level": ("pressure_level", P_HPA, {"units": "hPa"}),
            "latitude": lats,
            "longitude": lons,
        },
    )


def _sim_ds():
    return xr.Dataset(
        coords={
            "time": pd.date_range("2022-01-01 12:00", periods=1, freq="h"),
            "latitude": ("time", [78.0]),
            "longitude": ("time", [15.0]),
        }
    )


@pytest.mark.integration
@pytest.mark.skipif(not has_libradtran(), reason="LibRadtran not available")
class TestEra5EndToEnd:
    @pytest.fixture()
    def config(self, tmp_path):
        cfg = load_config()
        cfg.paths.working_dir = tmp_path
        cfg.paths.output_dir = tmp_path
        cfg.execution.cleanup_temp_files = False
        cfg.execution.max_workers = 1
        cfg.simulation_defaults.source = "thermal"
        cfg.simulation_defaults.wavelength_nm = [10000, 10001]
        cfg.simulation_defaults.mol_abs_param = "lowtran"
        cfg.simulation_defaults.surface_temperature_k = 260.0
        cfg.simulation_defaults.output_columns = ["edir", "edn", "eup"]
        return cfg

    def _input_files(self, tmp_path):
        return list(Path(tmp_path).glob("*.inp"))

    def test_raw_era5_with_ozone_and_clouds(self, config, tmp_path):
        result = _sim_ds().pyradtran.run(
            config=config,
            era5_atmosphere=_era5_ds(),
            era5_clouds=True,
            save_to_file=False,
            show_progress=False,
        )
        assert int(result["status"].values.ravel()[0]) == 0

        atm_files = list((tmp_path / "era5_atmospheres").glob("*.dat"))
        assert len(atm_files) == 1
        content = atm_files[0].read_text()
        assert "# columns: H2O MMR O3 MMR" in content

        inputs = self._input_files(tmp_path)
        assert inputs, "expected kept .inp files (cleanup disabled)"
        inp = inputs[0].read_text()
        assert "H2O MMR O3 MMR" in inp
        assert "mol_modify O3" not in inp
        assert "mol_modify H2O" not in inp
        assert "wc_file 1D" in inp

    def test_arco_names_accepted(self, config, tmp_path):
        arco = _era5_ds().rename(
            {
                "t": "temperature",
                "q": "specific_humidity",
                "z": "geopotential",
                "o3": "ozone_mass_mixing_ratio",
                "clwc": "specific_cloud_liquid_water_content",
                "pressure_level": "level",
                "valid_time": "time",
            }
        )
        result = _sim_ds().pyradtran.run(
            config=config,
            era5_atmosphere=arco,
            era5_clouds=True,
            save_to_file=False,
            show_progress=False,
        )
        assert int(result["status"].values.ravel()[0]) == 0

    def test_explicit_cloud_vars_win_over_era5(self, config, tmp_path):
        ds = _sim_ds()
        ds["lwc"] = ("time", [0.3])
        ds["reff"] = ("time", [12.0])
        ds["cth"] = ("time", [2.0])
        ds["cbh"] = ("time", [1.0])
        result = ds.pyradtran.run(
            config=config,
            era5_atmosphere=_era5_ds(),
            era5_clouds=True,
            cloud_wc_var="lwc",
            cloud_reff_var="reff",
            cloud_top_var="cth",
            cloud_bottom_var="cbh",
            save_to_file=False,
            show_progress=False,
        )
        assert int(result["status"].values.ravel()[0]) == 0
        inp = self._input_files(tmp_path)[0].read_text()
        # explicit slab cloud (0.3 g/m3) used, not the ERA5 profile
        wc_lines = [l for l in inp.splitlines() if l.startswith("wc_file")]
        assert len(wc_lines) == 1
        wc_path = wc_lines[0].split()[-1]
        assert "0.3" in Path(wc_path).read_text()

    def test_era5_clouds_without_atmosphere_raises(self, config):
        with pytest.raises(PyRadtranError, match="era5_clouds"):
            _sim_ds().pyradtran.run(
                config=config,
                era5_clouds=True,
                save_to_file=False,
                show_progress=False,
            )
