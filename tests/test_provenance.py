"""Result datasets must record what produced them (attrs provenance)."""

import json
from unittest.mock import patch

import pytest
import xarray as xr

import pyradtran
from pyradtran.params import Var


@pytest.fixture
def ds_in():
    return xr.Dataset(
        data_vars={
            "latitude": (["time"], [78.0, 78.1]),
            "longitude": (["time"], [15.0, 15.1]),
            "surface_albedo": (["time"], [0.3, 0.4]),
        },
        coords={"time": [0, 1]},
    )


@pytest.fixture
def fake_result():
    return xr.Dataset({"eglo": (("time",), [500.0, 510.0])}, coords={"time": [0, 1]})


def _run_mocked(ds_in, fake_result, minimal_config, **kwargs):
    import pyradtran.interface as interface
    from pyradtran.interface import PointOutcome

    with (
        patch.object(interface, "execute_simulation_batch") as mock_batch,
        patch.object(
            interface.OutputToXarray, "convert_batch", return_value=fake_result
        ),
    ):
        mock_batch.return_value = [PointOutcome(object(), 0), PointOutcome(object(), 0)]
        return ds_in.pyradtran.run(
            config=minimal_config,
            save_to_file=False,
            show_progress=False,
            **kwargs,
        )


class TestProvenanceAttrs:
    def test_version_and_history(self, ds_in, fake_result, minimal_config):
        out = _run_mocked(ds_in, fake_result, minimal_config)
        assert out.attrs["pyradtran_version"] == pyradtran.__version__
        assert "pyradtran" in out.attrs["history"]

    def test_params_serialized(self, ds_in, fake_result, minimal_config):
        out = _run_mocked(
            ds_in,
            fake_result,
            minimal_config,
            params={"albedo": Var("surface_albedo"), "mol_modify O3": 320.0},
        )
        recorded = json.loads(out.attrs["pyradtran_params"])
        assert recorded["albedo"] == "Var(name='surface_albedo')"
        assert recorded["mol_modify O3"] == 320.0

    def test_config_yaml_attr(self, ds_in, fake_result, minimal_config):
        import yaml

        out = _run_mocked(ds_in, fake_result, minimal_config)
        cfg = yaml.safe_load(out.attrs["pyradtran_config"])
        assert cfg["simulation_defaults"]["rte_solver"] == "disort"

    def test_input_example_annotated(self, ds_in, fake_result, minimal_config):
        out = _run_mocked(
            ds_in,
            fake_result,
            minimal_config,
            params={"albedo": Var("surface_albedo")},
        )
        example = out.attrs["pyradtran_input_example"]
        assert "rte_solver disort" in example
        assert "# dataset-var" in example  # per-point albedo annotated
        assert "albedo 0.3" in example  # first point's value

    def test_channels_recorded(self, ds_in, minimal_config):
        import numpy as np

        wl = np.linspace(400.0, 700.0, 31)
        spectral = xr.Dataset(
            {"uu": (("time", "wavelength"), np.ones((2, 31)))},
            coords={"wavelength": wl, "time": [0, 1]},
        )
        phi = np.zeros((1, wl.size))
        phi[0, 10:20] = 1.0
        srf = xr.DataArray(
            phi,
            dims=("channel", "wavelength"),
            coords={"channel": ["ch1"], "wavelength": wl},
        )
        out = _run_mocked(ds_in, spectral, minimal_config, channels=srf)
        assert out.attrs["pyradtran_channels"] == "ch1"

    def test_attrs_survive_netcdf_roundtrip(
        self, ds_in, fake_result, minimal_config, tmp_path
    ):
        out = _run_mocked(
            ds_in,
            fake_result,
            minimal_config,
            params={"albedo": Var("surface_albedo")},
        )
        path = tmp_path / "prov.nc"
        from pyradtran.io import NetCDFSaver

        NetCDFSaver.save_results_to_netcdf(out, path, ds_in, minimal_config)
        back = xr.open_dataset(path)
        assert back.attrs["pyradtran_version"] == pyradtran.__version__
        assert "albedo" in back.attrs["pyradtran_params"]
        back.close()


class TestCoordinateUnits:
    def test_wavelength_and_altitude_units(self, minimal_config, tmp_path):
        import pandas as pd

        from pyradtran.io import OutputParser, OutputToXarray

        minimal_config.simulation_defaults.output_columns = ["eglo"]
        minimal_config.simulation_defaults.output_altitudes_km = [0.0, 1.0]
        out = tmp_path / "s.out"
        out.write_text(
            "  0.0 500.0 100.0\n  1.0 500.0 90.0\n"
            "  0.0 600.0 110.0\n  1.0 600.0 95.0\n"
        )
        parsed = OutputParser(minimal_config).parse_output_file(out)
        input_ds = xr.Dataset(
            coords={
                "time": [pd.Timestamp("2023-05-01")],
                "latitude": ("time", [60.0]),
                "longitude": ("time", [10.0]),
            }
        )
        res = OutputToXarray.convert_batch([parsed], input_ds)
        assert res["wavelength"].attrs["units"] == "nm"
        assert res["altitude"].attrs["units"] == "km"
