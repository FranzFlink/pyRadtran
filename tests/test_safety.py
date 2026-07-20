# tests/test_safety.py
"""Error-safety regression tests: config immutability, scratch cleanup,
exception chaining, sentinel handling, and input validation."""

import stat
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import pyradtran  # noqa: F401 — registers the accessor
from pyradtran.exceptions import InputGenerationError
from pyradtran.interface import (
    execute_simulation_batch,
    run_pyradtran_simulation,
)
from pyradtran.params import ParamResolver


def _make_succeeding_uvspec(config):
    bin_path = config.paths.libradtran_bin
    bin_path.write_text(
        "#!/bin/bash\ncat > /dev/null\necho ' 30.0  100.0  50.0  0.1'\n"
    )
    bin_path.chmod(bin_path.stat().st_mode | stat.S_IEXEC)


@pytest.fixture
def run_config(minimal_config):
    """Single-altitude integrated config with a succeeding mock uvspec."""
    minimal_config.simulation_defaults.output_altitudes_km = [0.0]
    minimal_config.simulation_defaults.integrate_wavelength = True
    minimal_config.simulation_defaults.output_columns = [
        "sza", "eglo", "eup", "albedo",
    ]
    _make_succeeding_uvspec(minimal_config)
    return minimal_config


@pytest.fixture
def trajectory_ds():
    times = pd.date_range("2022-07-01 12:00", periods=2, freq="30min")
    return xr.Dataset(
        data_vars={
            "latitude": (["time"], [78.0, 78.1]),
            "longitude": (["time"], [15.0, 15.1]),
            "altitude": (["time"], [0.0, 0.0]),
        },
        coords={"time": times},
    )


class TestConfigImmutability:
    def test_run_does_not_mutate_caller_config(self, run_config, trajectory_ds):
        """Dotted config overrides apply to a per-run copy only."""
        before = run_config.simulation_defaults.albedo_value
        trajectory_ds.pyradtran.run(
            config=run_config,
            params={"simulation_defaults.albedo_value": 0.9},
            save_to_file=False,
            show_progress=False,
        )
        assert run_config.simulation_defaults.albedo_value == before

    def test_copy_is_independent_except_era5_dataset(self, minimal_config):
        marker = object()
        minimal_config.simulation_defaults.clouds.era5_dataset = marker
        cfg2 = minimal_config.copy()
        cfg2.simulation_defaults.albedo_value = 0.99
        cfg2.simulation_defaults.output_altitudes_km.append(42.0)
        assert minimal_config.simulation_defaults.albedo_value != 0.99
        assert 42.0 not in minimal_config.simulation_defaults.output_altitudes_km
        # large ERA5 datasets are shared by reference, not copied
        assert cfg2.simulation_defaults.clouds.era5_dataset is marker
        assert minimal_config.simulation_defaults.clouds.era5_dataset is marker


class TestNoSpuriousDeprecation:
    def test_altitude_data_var_does_not_warn(
        self, run_config, trajectory_ds, recwarn
    ):
        """`altitude` as a data variable is a documented layout; it must
        not trigger the deprecated-kwargs warning."""
        trajectory_ds.pyradtran.run(
            config=run_config, save_to_file=False, show_progress=False
        )
        assert not [
            w for w in recwarn if issubclass(w.category, DeprecationWarning)
        ]


class TestScratchCleanup:
    def test_no_inp_or_out_files_left(self, run_config, trajectory_ds):
        """cleanup_temp_files=True must remove both .inp and .out files."""
        assert run_config.execution.cleanup_temp_files is True
        trajectory_ds.pyradtran.run(
            config=run_config, save_to_file=False, show_progress=False
        )
        work = run_config.paths.working_dir
        assert list(work.glob("*.inp")) == []
        assert list(work.glob("*.out")) == []


class TestEra5CloudsSentinel:
    def test_empty_dict_means_enabled(self, run_config, trajectory_ds):
        """era5_clouds={} is 'enabled with defaults', not disabled."""
        with pytest.raises(ValueError, match="era5_clouds requires"):
            execute_simulation_batch(
                config=run_config,
                input_ds=trajectory_ds,
                era5_clouds={},
                show_progress=False,
            )


class TestConvolveGuards:
    def test_non_overlapping_srf_raises(self):
        from pyradtran.channels import convolve_channels

        result = xr.Dataset(
            {"eglo": (("wavelength",), np.ones(4))},
            coords={"wavelength": [400.0, 500.0, 600.0, 700.0]},
        )
        srf = xr.DataArray(
            np.ones((1, 3)),
            dims=("channel", "wavelength"),
            coords={"channel": ["ch1"], "wavelength": [1000.0, 1050.0, 1100.0]},
        )
        with pytest.raises(ValueError, match="no overlap"):
            convolve_channels(result, srf)


class TestPathsConfigCoercion:
    def test_string_paths_accepted(self, tmp_path):
        from pyradtran.config import PathsConfig

        bin_path = tmp_path / "uvspec"
        bin_path.touch()
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        solar = tmp_path / "solar.dat"
        solar.touch()
        paths = PathsConfig(
            libradtran_bin=str(bin_path),
            libradtran_data=str(data_dir),
            solar_spectrum=str(solar),
            output_dir=str(tmp_path / "out"),
            working_dir=str(tmp_path / "work"),
        )
        assert paths.libradtran_bin == bin_path
        assert paths.working_dir.is_dir()


class TestFalseOmitsOption:
    def test_resolver_passes_false_through_unvalidated(self, minimal_config):
        resolver = ParamResolver(
            minimal_config, {"aerosol_default": False}, schema=None
        )
        value, _prov = resolver.static_params()["aerosol_default"]
        assert value is False

    def test_builder_emits_no_line_for_false(self, minimal_config):
        from pyradtran.input_builder import InputFileBuilder
        from pyradtran.params import PROV_UNVALIDATED
        from datetime import datetime

        b = InputFileBuilder(minimal_config)
        lines = b.build(
            datetime(2022, 7, 1, 12), 78.0, 15.0,
            resolved={"aerosol_default": (False, PROV_UNVALIDATED)},
        )
        assert not any(ln.keyword == "aerosol_default" for ln in lines)

    def test_false_switches_off_config_supplied_option(self, minimal_config):
        """params={'albedo': False} must suppress the config albedo line."""
        from datetime import datetime

        from pyradtran.input_builder import InputFileBuilder

        resolver = ParamResolver(
            minimal_config, {"albedo": False}, schema=None
        )
        b = InputFileBuilder(minimal_config)
        lines = b.build(
            datetime(2022, 7, 1, 12), 78.0, 15.0,
            resolved=resolver.static_params(),
        )
        assert not any(ln.keyword == "albedo" for ln in lines)


class TestInputLoaderErrors:
    def test_csv_without_time_column_clear_error(self, tmp_path):
        from pyradtran.io import InputDataLoader

        csv = tmp_path / "input.csv"
        csv.write_text("latitude,longitude\n10.0,20.0\n")
        with pytest.raises(InputGenerationError, match="'time'"):
            InputDataLoader.load_simulation_input_data(csv)

    def test_missing_file_error_passes_through_unwrapped(
        self, minimal_config, tmp_path
    ):
        """run_pyradtran_simulation must not double-wrap its own
        exception types."""
        with patch(
            "pyradtran.interface.load_config", return_value=minimal_config
        ):
            with pytest.raises(InputGenerationError) as exc_info:
                run_pyradtran_simulation(tmp_path / "does_not_exist.csv")
        assert "Simulation failed" not in str(exc_info.value)
