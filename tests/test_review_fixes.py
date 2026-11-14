"""Regressions found in the 0.2 release review."""

from datetime import datetime, timezone
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pyradtran.clouds import CloudFileWriter, CloudLayer
from pyradtran.exceptions import ValidationError
from pyradtran.io import OutputToXarray, OutputType, ParsedOutput
from pyradtran.params import ParamResolver


def _track(n=3):
    return xr.Dataset(
        coords={
            "time": pd.date_range("2022-07-01 12:00", periods=n, freq="30min"),
            "latitude": ("time", [78.0] * n),
            "longitude": ("time", [15.0] * n),
        }
    )


def _point(value=1.0, zout=None, **extra):
    return ParsedOutput(
        OutputType.INTEGRATED_SINGLE_ALTITUDE,
        {"eglo": np.array([value]), **{k: np.array([v]) for k, v in extra.items()}},
        metadata={} if zout is None else {"zout": zout},
    )


def test_scalar_input_with_scalar_altitude_coordinate():
    ds = _track(1).isel(time=0).assign_coords(altitude=5.0)
    out = OutputToXarray.convert_batch([_point(2.0, zout=5.0)], ds)
    assert out.eglo.item() == 2.0
    assert out.altitude.values.tolist() == [5.0]


def test_symbolic_level_is_recorded():
    out = OutputToXarray.convert_batch([_point(zout="toa")] * 3, _track())
    assert out.altitude.attrs["zout"] == "toa"


def test_coordless_sweep_dim_named_like_output_column():
    t = pd.Timestamp("2022-07-01 12:00")
    sweep = xr.Dataset(  # dimension "sza" without a coordinate variable
        coords={
            "time": ("sza", [t, t]),
            "latitude": ("sza", [78.0, 78.0]),
            "longitude": ("sza", [15.0, 15.0]),
        }
    )
    outs = [_point(1.0, sza=30.0), _point(2.0, sza=60.0)]
    out = OutputToXarray.convert_batch(outs, sweep)
    assert out.sza_out.squeeze().values.tolist() == [30.0, 60.0]


def test_explain_point_may_name_altitude(minimal_config):
    ds = _track().assign_coords(altitude=[0.0, 5.0])
    text = ds.pyradtran.explain(
        config=minimal_config, point={"time": "2022-07-01 12:30", "altitude": 5.0}
    )
    assert "zout 0.0000 5.0000" in text


def test_dotted_paths_override_becomes_path(minimal_config, tmp_path):
    cfg = minimal_config.copy()
    ParamResolver(cfg, {"paths.working_dir": str(tmp_path / "w")}, schema=None)
    assert isinstance(cfg.paths.working_dir, Path) and cfg.paths.working_dir.is_dir()


def test_wrong_type_override_is_a_validation_error(minimal_config):
    with pytest.raises(ValidationError):
        ParamResolver(
            minimal_config.copy(),
            {"simulation_defaults.output_altitudes_km": 5.0},
            schema=None,
        )


def test_ignored_cloud_writer_args_warn(tmp_path):
    layer = [CloudLayer(1.0, 1.5, lwc_g_m3=0.2)]
    with pytest.warns(DeprecationWarning):
        CloudFileWriter.write_water_cloud_file(
            layer, tmp_path / "a.dat", altitude_resolution_km=0.05
        )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        CloudFileWriter.write_water_cloud_file(layer, tmp_path / "b.dat")


def test_brdf_type_keeps_user_albedo_library(minimal_config):
    from pyradtran.input_builder import InputFileBuilder
    from pyradtran.params import PROV_LITERAL

    b = InputFileBuilder(minimal_config)
    dt = datetime(2022, 7, 1, 12)
    own = b.render(
        b.build(
            dt,
            78.0,
            15.0,
            resolved={
                "albedo_library": ("IGBP", PROV_LITERAL),
                "brdf_rpv_type": (19, PROV_LITERAL),
            },
        )
    )
    assert "albedo_library IGBP" in own and "brdf_rpv_library" not in own
    assert "\nalbedo " not in own
    default = b.render(
        b.build(dt, 78.0, 15.0, resolved={"brdf_rpv_type": (7, PROV_LITERAL)})
    )
    assert "brdf_rpv_library IGBP" in default


def test_explain_point_with_string_coordinate(minimal_config):
    ds = _track(1).expand_dims(surface=["open water", "thin ice"])
    ds["alb"] = ("surface", [0.06, 0.4])
    text = ds.pyradtran.explain(
        config=minimal_config,
        params={"albedo": __import__("pyradtran").Var("alb")},
        point={"surface": "thin ice", "time": "2022-07-01 12:00"},
    )
    assert "albedo 0.4" in text


def test_normalize_era5_arco_level_units():
    from pyradtran.era5 import normalize_era5

    ds = xr.Dataset(
        {"temperature": (("level",), [250.0, 280.0])},
        coords={"level": ("level", [500, 1000], {"units": "Hectopascal(hPa)"})},
    )
    out = normalize_era5(ds)
    assert out.pressure_level.values.tolist() == [500.0, 1000.0]
