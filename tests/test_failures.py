# tests/test_failures.py
"""Failure reporting: status codes, stderr log, kept temp files."""

import stat
from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr

from pyradtran.interface import PointOutcome, execute_simulation_batch
from pyradtran.io import OutputType, ParsedOutput


def _make_failing_uvspec(config):
    """Turn the mock uvspec binary into a script that fails loudly."""
    bin_path = config.paths.libradtran_bin
    bin_path.write_text("#!/bin/bash\necho 'uvspec exploded' >&2\nexit 1\n")
    bin_path.chmod(bin_path.stat().st_mode | stat.S_IEXEC)


def _make_succeeding_uvspec(config):
    """Mock uvspec that emits one integrated single-altitude row."""
    bin_path = config.paths.libradtran_bin
    bin_path.write_text(
        "#!/bin/bash\ncat > /dev/null\necho ' 30.0  100.0  50.0  0.1'\n"
    )
    bin_path.chmod(bin_path.stat().st_mode | stat.S_IEXEC)


@pytest.fixture
def single_alt_config(minimal_config):
    minimal_config.simulation_defaults.output_altitudes_km = [0.0]
    minimal_config.simulation_defaults.integrate_wavelength = True
    minimal_config.simulation_defaults.output_columns = [
        "sza", "eglo", "eup", "albedo",
    ]
    return minimal_config


class TestOutcomes:
    def test_failed_run_yields_status_1_with_stderr(
        self, single_alt_config, simple_input_dataset
    ):
        _make_failing_uvspec(single_alt_config)
        with pytest.raises(Exception):
            # all fail -> batch raises, but outcomes requested first
            execute_simulation_batch(
                config=single_alt_config,
                input_ds=simple_input_dataset,
                show_progress=False,
                return_outcomes=True,
            )
        # failure log written
        logs = list(single_alt_config.paths.working_dir.glob("failures_*.log"))
        assert logs, "expected a failures_*.log file"
        assert "uvspec exploded" in logs[0].read_text()

    def test_success_yields_status_0(self, single_alt_config, simple_input_dataset):
        _make_succeeding_uvspec(single_alt_config)
        outcomes = execute_simulation_batch(
            config=single_alt_config,
            input_ds=simple_input_dataset,
            show_progress=False,
            return_outcomes=True,
        )
        assert all(isinstance(o, PointOutcome) for o in outcomes)
        assert all(o.status == 0 for o in outcomes)
        assert all(o.parsed is not None for o in outcomes)

    def test_default_return_shape_unchanged(
        self, single_alt_config, simple_input_dataset
    ):
        _make_succeeding_uvspec(single_alt_config)
        results = execute_simulation_batch(
            config=single_alt_config,
            input_ds=simple_input_dataset,
            show_progress=False,
        )
        assert all(isinstance(r, ParsedOutput) for r in results)


class TestFailedInputKept:
    def test_input_file_kept_on_failure_despite_cleanup_flag(
        self, single_alt_config, simple_input_dataset
    ):
        _make_failing_uvspec(single_alt_config)
        assert single_alt_config.execution.cleanup_temp_files is True
        with pytest.raises(Exception):
            execute_simulation_batch(
                config=single_alt_config,
                input_ds=simple_input_dataset,
                show_progress=False,
            )
        kept = list(single_alt_config.paths.working_dir.glob("*.inp"))
        assert kept, "failed-run input files must be kept for post-mortem"
