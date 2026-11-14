"""Regression tests for the interface / io / input_builder bug-fix pass."""

import inspect
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from helpers import has_libradtran

import pyradtran.io as pio
from pyradtran.config import CloudParameters
from pyradtran.core import Simulation
from pyradtran.exceptions import ValidationError
from pyradtran.input_builder import InputFileBuilder, calculate_solar_zenith_angle
from pyradtran.interface import (
    PointOutcome,
    execute_simulation_batch,
    run_pyradtran_simulation,
)
from pyradtran.io import (
    NetCDFSaver,
    OutputParser,
    OutputToXarray,
    OutputType,
    ParsedOutput,
)
from pyradtran.params import PROV_LITERAL, Var

DT = datetime(2022, 7, 1, 12, 0)


def _track(n=3):
    return xr.Dataset(
        coords={
            "time": pd.date_range("2022-07-01 12:00", periods=n, freq="30min"),
            "latitude": ("time", [78.0] * n),
            "longitude": ("time", [15.0] * n),
        }
    )


def _point(value=1.0, zout=None, **extra):
    """Integrated single-altitude ParsedOutput, as the parser builds it."""
    return ParsedOutput(
        OutputType.INTEGRATED_SINGLE_ALTITUDE,
        {"eglo": np.array([value]), **{k: np.array([v]) for k, v in extra.items()}},
        metadata={} if zout is None else {"zout": zout},
    )


def _fake_run(config, point):
    """Stand-in for a uvspec run: eglo = 100, zout as the parser reports it."""
    overrides = {k: v for k, (v, _p) in point.resolved.items()}
    zout = OutputParser(config, overrides).output_altitudes
    return PointOutcome(_point(100.0, zout=zout[0]), 0, None, point.point_id)


def _build(config, resolved=None):
    b = InputFileBuilder(config)
    return b.render(b.build(DT, 78.0, 15.0, resolved=resolved))


# -- 1. altitude coordinate of single-level output ---------------------------


class TestAltitudeCoordinate:
    def test_shared_level_is_the_coordinate(self):
        ds = OutputToXarray.convert_batch([_point(zout=2.0)] * 3, _track())
        assert ds.altitude.values.tolist() == [2.0]
        assert ds.altitude.dtype == float
        assert "zout" not in ds

    def test_per_point_levels_keep_placeholder_and_return_zout(self):
        outs = [_point(zout=0.0), None, _point(zout=3.0)]
        ds = OutputToXarray.convert_batch(outs, _track())
        assert ds.altitude.values.tolist() == [0.0]
        assert ds.zout.dims == ("time",)
        np.testing.assert_array_equal(ds.zout.values, [0.0, np.nan, 3.0])
        assert ds.eglo.isel(altitude=0).shape == (3,)

    def test_unknown_level_is_float_zero_and_sel_works(self):
        ds = OutputToXarray.convert_batch([_point()] * 3, _track())
        assert ds.altitude.dtype == float
        assert ds.eglo.sel(altitude=0).shape == (3,)

    def test_parser_records_single_level(self, minimal_config, tmp_path):
        minimal_config.simulation_defaults.output_altitudes_km = [2.0]
        minimal_config.simulation_defaults.integrate_wavelength = True
        out = tmp_path / "p.out"
        out.write_text("55.3 420.1 357.1 0.85\n")
        assert OutputParser(minimal_config).parse_output_file(out).metadata == {
            "zout": 2.0
        }
        parsed = OutputParser(minimal_config, {"zout": 7}).parse_output_file(out)
        assert parsed.metadata["zout"] == 7.0
        ds = OutputToXarray.convert(parsed, _track(1))
        assert float(ds.altitude) == 7.0 and ds.eglo.dims == ("time",)

    def test_multi_level_columns_untouched(self, minimal_config):
        # the zout column is still injected only for multi-altitude runs
        assert OutputParser(minimal_config).output_columns[0] == "zout"
        assert "zout" not in OutputParser(minimal_config, {"zout": 2}).output_columns

    def test_run_reports_config_and_per_point_levels(self, minimal_config):
        minimal_config.simulation_defaults.output_altitudes_km = [2.0]
        with patch("pyradtran.interface._run_single_simulation_unified", _fake_run):
            res = _track().pyradtran.run(
                config=minimal_config, save_to_file=False, show_progress=False
            )
            flight = _track().assign(altitude=("time", [0.0, 1.5, 3.0]))
            res_f = flight.pyradtran.run(
                config=minimal_config, save_to_file=False, show_progress=False
            )
        assert res.altitude.values.tolist() == [2.0]
        assert res_f.altitude.values.tolist() == [0.0]
        assert res_f.zout.values.tolist() == [0.0, 1.5, 3.0]


# -- 2. config-driven parametric clouds --------------------------------------


class TestParametricClouds:
    @pytest.fixture
    def cloudy(self, minimal_config):
        minimal_config.simulation_defaults.clouds = CloudParameters(
            enabled=True,
            cloud_type="mixed",
            cloud_source="parametric",
            layer_bottom_km=1.0,
            layer_top_km=2.0,
            water_content_g_m3=0.1,
            ice_content_g_m3=0.02,
            effective_radius_um=10.0,
        )
        return minimal_config

    def test_no_invalid_layer_keywords(self, cloudy):
        text = _build(cloudy)
        assert "wc_layer" not in text and "ic_layer" not in text
        assert "wc_file 1D" in text and "ic_file 1D" in text

    def test_run_writes_two_row_profile(self, cloudy):
        seen = {}

        def fake_uvspec(self, input_path, output_path):
            for line in Path(input_path).read_text().splitlines():
                if line.startswith(("wc_file", "ic_file")):
                    key, kind, path = line.split()
                    seen[key] = (kind, Path(path).read_text())
            return False

        with patch.object(Simulation, "_run_uvspec", fake_uvspec):
            Simulation(cloudy).run_simulation(DT, 78.0, 15.0)
        assert seen["wc_file"] == (
            "1D",
            "2.000000 0.100000 10.000000\n1.000000 0.100000 10.000000\n",
        )
        assert seen["ic_file"][1].splitlines()[1] == "1.000000 0.020000 10.000000"

    def test_explicit_params_win(self, cloudy):
        text = _build(cloudy, {"wc_file": ("1D /my/cloud.dat", PROV_LITERAL)})
        assert "wc_file 1D /my/cloud.dat" in text
        assert text.count("wc_file") == 1
        assert "wc_file" not in _build(cloudy, {"wc_file": (False, PROV_LITERAL)})

    def test_explain_shows_profile(self, cloudy):
        text = _track().pyradtran.explain(config=cloudy)
        (line,) = [ln for ln in text.splitlines() if ln.startswith("wc_file")]
        assert "1D" in line and "{" not in line and line.endswith("# config")

    @pytest.mark.integration
    @pytest.mark.skipif(not has_libradtran(), reason="needs libRadtran")
    def test_real_run_is_cloudy(self, tmp_path):
        from pyradtran import load_config

        cfg = load_config()
        cfg.paths.working_dir = cfg.paths.output_dir = tmp_path
        params = {
            "rte_solver": "disort",
            "wavelength": [550, 550],
            "sza": 40.0,
            "output_user": "lambda edir",
            "execution.max_workers": 1,
        }
        kw = dict(params=params, save_to_file=False, show_progress=False)
        clear = _track(1).pyradtran.run(config=cfg, **kw)
        cfg.simulation_defaults.clouds = CloudParameters(
            enabled=True, cloud_source="parametric", water_content_g_m3=0.3
        )
        cloudy = _track(1).pyradtran.run(config=cfg, **kw)
        assert int(cloudy.status) == 0
        assert float(cloudy.edir.squeeze()) < 0.01 * float(clear.edir.squeeze())


# -- 3. sweep dimension named like an output column --------------------------


def test_sweep_dim_named_like_output_column_is_kept(caplog):
    sweep = _track(1).expand_dims(albedo=[0.1, 0.9])
    outs = [_point(1.0, albedo=0.1), _point(2.0, albedo=0.9)]
    with caplog.at_level(logging.WARNING, logger="pyradtran.io"):
        ds = OutputToXarray.convert_batch(outs, sweep)
    assert ds.albedo.values.tolist() == [0.1, 0.9]  # the sweep coordinate
    assert ds.albedo_out.squeeze().values.tolist() == [0.1, 0.9]
    assert ds.eglo.squeeze().values.tolist() == [1.0, 2.0]
    assert len([r for r in caplog.records if "albedo_out" in r.message]) == 1


# -- 4. explain ---------------------------------------------------------------


class TestExplain:
    def test_does_not_mutate_callers_config(self, minimal_config):
        _track().pyradtran.explain(
            config=minimal_config,
            params={"simulation_defaults.wavelength_nm": [500, 600]},
        )
        assert minimal_config.simulation_defaults.wavelength_nm == [400, 3200]

    def test_altitude_coordinate_becomes_zout_levels(self, minimal_config):
        ds = _track().assign_coords(altitude=[10.0, 0.0, 2.0])
        text = ds.pyradtran.explain(config=minimal_config)
        assert "zout 0.0000 2.0000 10.0000" in text
        assert minimal_config.simulation_defaults.output_altitudes_km == [5, 6, 7]

    def test_altitude_data_variable_is_per_point_zout(self, minimal_config):
        ds = _track().assign(altitude=("time", [0.0, 1.5, 3.0]))
        text = ds.pyradtran.explain(
            config=minimal_config, point={"time": "2022-07-01 12:30"}
        )
        (line,) = [ln for ln in text.splitlines() if ln.startswith("zout")]
        assert line.split()[:2] == ["zout", "1.5"] and "dataset-var" in line


# -- 5. atmosphere / solar short names ---------------------------------------


class TestShortNames:
    def test_config_attribute_assignment(self, minimal_config):
        data = minimal_config.paths.libradtran_data
        minimal_config.paths.atmosphere_profile = "afglss"
        minimal_config.paths.solar_spectrum = "kurudz_1.0nm"
        text = _build(minimal_config)
        assert f"atmosphere_file {data}/atmmod/afglss.dat" in text
        assert f"source solar {data}/solar_flux/kurudz_1.0nm.dat" in text

    def test_params_values(self, minimal_config):
        data = minimal_config.paths.libradtran_data
        text = _build(
            minimal_config,
            {
                "atmosphere_file": ("afglss", PROV_LITERAL),
                "source": ("solar kurudz_1.0nm per_nm", PROV_LITERAL),
            },
        )
        assert f"atmosphere_file {data}/atmmod/afglss.dat" in text
        assert f"source solar {data}/solar_flux/kurudz_1.0nm.dat per_nm" in text

    def test_full_paths_pass_through(self, minimal_config):
        text = _build(
            minimal_config, {"atmosphere_file": ("/x/my_atm.dat", PROV_LITERAL)}
        )
        assert "atmosphere_file /x/my_atm.dat" in text
        assert f"source solar {minimal_config.paths.solar_spectrum}\n" in text

    def test_dotted_override_via_explain(self, minimal_config):
        text = _track().pyradtran.explain(
            config=minimal_config, params={"paths.atmosphere_profile": "afglsw"}
        )
        assert "/atmmod/afglsw.dat" in text


# -- 6. solar zenith angle -----------------------------------------------------


class TestSolarZenithAngle:
    # Reference: NOAA Solar Calculator algorithm (Meeus), geometric zenith
    # angle, evaluated independently (.scratch/fix-interface/sza_ref.py).
    @pytest.mark.parametrize(
        "when, lat, lon, expected",
        [
            (datetime(2024, 6, 21, 12), 23.45, 0.0, 0.44),
            (datetime(2024, 6, 21, 12), 78.9, 11.9, 55.71),
            (datetime(2024, 12, 21, 12), 78.9, 11.9, 102.58),
            (datetime(2024, 3, 20, 12), 51.5, 0.0, 51.38),
            (datetime(2024, 11, 3, 12), 0.0, 0.0, 15.83),  # EoT +16.5 min
            (datetime(2024, 2, 11, 12), 40.0, 0.0, 54.20),  # EoT -14.2 min
            (datetime(2025, 6, 21, 16), 61.0, 22.0, 65.58),
            (datetime(2025, 9, 22, 18, 19), 0.0, -94.75, 1.86),
        ],
    )
    def test_matches_noaa_reference(self, when, lat, lon, expected):
        assert calculate_solar_zenith_angle(when, lat, lon) == pytest.approx(
            expected, abs=0.3
        )

    @pytest.mark.parametrize(
        "equinox", [datetime(2024, 3, 20, 3, 6), datetime(2025, 9, 22, 18, 19)]
    )
    def test_equinox_noon_zenith_equals_latitude(self, equinox):
        # declination 0 at the equinox instant: at true solar noon (the
        # minimum over longitude) the zenith angle is the latitude
        noon = min(
            calculate_solar_zenith_angle(equinox, 50.0, lon)
            for lon in np.arange(-180, 180, 0.25)
        )
        assert noon == pytest.approx(50.0, abs=0.1)

    def test_solstice_declination(self):
        solstice = datetime(2024, 6, 20, 20, 51)
        noon = min(
            calculate_solar_zenith_angle(solstice, 0.0, lon)
            for lon in np.arange(-180, 180, 0.25)
        )
        assert noon == pytest.approx(23.44, abs=0.1)

    def test_timezones(self):
        naive = calculate_solar_zenith_angle(datetime(2024, 6, 21, 12), 50.0, 10.0)
        utc = datetime(2024, 6, 21, 12, tzinfo=timezone.utc)
        cest = datetime(2024, 6, 21, 14, tzinfo=timezone(timedelta(hours=2)))
        assert calculate_solar_zenith_angle(utc, 50.0, 10.0) == naive
        assert calculate_solar_zenith_angle(cest, 50.0, 10.0) == naive


# -- 7. scalar (0-d) dataset ----------------------------------------------------


def test_scalar_dataset_runs_and_stays_scalar(minimal_config):
    ds = xr.Dataset(
        coords={
            "time": pd.Timestamp("2022-07-01 12:00"),
            "latitude": 78.0,
            "longitude": 15.0,
        }
    )
    minimal_config.simulation_defaults.output_altitudes_km = [0.0]
    with patch("pyradtran.interface._run_single_simulation_unified", _fake_run):
        outs = execute_simulation_batch(minimal_config, ds, show_progress=False)
        res = ds.pyradtran.run(
            config=minimal_config, save_to_file=False, show_progress=False
        )
    assert len(outs) == 1
    assert res.eglo.dims == ("altitude",) and res.eglo.item() == 100.0
    assert res.status.ndim == 0 and int(res.status) == 0
    assert float(res.latitude) == 78.0 and "time" in res.coords


# -- 8. custom coordinate names reach the provenance example --------------------


def test_input_example_with_custom_coordinate_names(minimal_config):
    ds = _track().rename(time="t", latitude="lat", longitude="lon")
    with patch("pyradtran.interface._run_single_simulation_unified", _fake_run):
        res = ds.pyradtran.run(
            config=minimal_config,
            time_var="t",
            lat_var="lat",
            lon_var="lon",
            save_to_file=False,
            show_progress=False,
        )
    assert "rte_solver disort" in res.attrs["pyradtran_input_example"]


# -- 9. jacobian argument checks -------------------------------------------------


class TestJacobianArguments:
    def test_return_dataset_false_rejected(self, minimal_config):
        with pytest.raises(ValidationError, match="return_dataset"):
            _track().pyradtran.jacobian(
                "albedo", 0.01, config=minimal_config, return_dataset=False
            )

    def test_non_numeric_base_value(self, minimal_config):
        with pytest.raises(ValidationError, match="rte_solver"):
            _track().pyradtran.jacobian(
                "rte_solver",
                0.01,
                config=minimal_config,
                params={"rte_solver": "disort"},
            )


# -- 10. inspect_cloud_file validates variable names -----------------------------


def test_inspect_cloud_file_missing_variable():
    ds = _track().assign(top=("time", [2.0] * 3), base=("time", [1.0] * 3))
    with pytest.raises(ValueError, match="Missing cloud variables.*lwcc"):
        ds.pyradtran.inspect_cloud_file(
            cloud_wc_var="lwcc", cloud_top_var="top", cloud_bottom_var="base"
        )


# -- 11. run_pyradtran_simulation(show_progress=) ----------------------------------


def test_run_pyradtran_simulation_show_progress(minimal_config, sample_netcdf_file):
    assert (
        inspect.signature(run_pyradtran_simulation).parameters["show_progress"].default
        is True
    )
    with (
        patch("pyradtran.interface.load_config", return_value=minimal_config),
        patch(
            "pyradtran.interface.execute_simulation_batch",
            return_value=[_point()] * 3,
        ) as batch,
    ):
        run_pyradtran_simulation(
            sample_netcdf_file,
            output_path=minimal_config.paths.output_dir / "o.nc",
            show_progress=False,
        )
    assert batch.call_args.kwargs["show_progress"] is False


# -- 12. NetCDFSaver / __all__ -----------------------------------------------------


def test_saver_leaves_callers_attrs_alone(minimal_config, tmp_path):
    ds = xr.Dataset({"eglo": ("time", [1.0])}, attrs={"mine": 1, "empty": None})
    NetCDFSaver.save_results_to_netcdf(ds, tmp_path / "o.nc", ds, minimal_config)
    assert ds.attrs == {"mine": 1, "empty": None}
    with xr.open_dataset(tmp_path / "o.nc") as saved:
        assert saved.attrs["created_by"] == "pyradtran" and saved.attrs["mine"] == 1


def test_io_all_exports_radiosonde_generator():
    assert "RadiosondeAtmosphereGenerator" in pio.__all__
