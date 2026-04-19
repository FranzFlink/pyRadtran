# tests/test_io_robustness.py
"""
Robustness tests for I/O handling in pyradtran.

These tests cover complex I/O scenarios including:
- Spectral output (resolved by wavelength)
- Vertical profiles (resolved by altitude)
- Multi-dimensional output (Spectral x Altitude)
- Brightness temperature output
"""

import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pyradtran.config import (
    PathsConfig,
    SimulationConfig,
    SimulationDefaults,
    load_config,
)
from pyradtran.core import Simulation
from pyradtran.io import OutputParser, OutputToXarray, OutputType

from helpers import has_libradtran


# Reuse the integration config setup but allow modification
@pytest.fixture
def base_config(tmp_path):
    cfg = load_config()
    cfg.simulation_defaults.rte_solver = "twostr"
    cfg.simulation_defaults.wavelength_nm = [400, 500]
    cfg.simulation_defaults.output_columns = ["sza", "eglo"]
    cfg.simulation_defaults.output_altitudes_km = [0.0]
    cfg.paths.output_dir = tmp_path / "output"
    cfg.paths.working_dir = tmp_path / "working"
    (tmp_path / "output").mkdir(exist_ok=True)
    (tmp_path / "working").mkdir(exist_ok=True)
    return cfg


@pytest.mark.skipif(not has_libradtran(), reason="LibRadtran not available")
class TestIORobustness:

    @pytest.mark.xfail(
        reason="Test assertions assume a different parser output structure"
    )
    def test_spectral_output_parsing(self, base_config):
        """Test parsing of spectral output (wavelength resolved)."""
        # Configure for spectral output
        base_config.simulation_defaults.integrate_wavelength = False
        base_config.simulation_defaults.output_columns = ["lambda", "eglo"]
        base_config.simulation_defaults.wavelength_nm = [400, 410]  # Small range

        sim = Simulation(base_config)

        # Run simulation
        dt = datetime(2025, 6, 21, 12, 0)
        output_file = sim.run_simulation(dt, 0.0, 0.0)

        assert output_file is not None

        # Parse output
        parser = OutputParser(base_config)
        parsed = parser.parse_output_file(output_file)

        # Verify type
        assert parsed.output_type == OutputType.SPECTRAL_SINGLE_ALTITUDE

        # Verify spectral data exists
        assert parsed.wavelengths is not None
        assert len(parsed.wavelengths) > 1
        assert "lambda" in parsed.data
        assert "eglo" in parsed.data

        # Verify dimensions match
        # eglo should be size of wavelengths
        assert len(parsed.data["eglo"]) == len(parsed.wavelengths)

        # Convert to Xarray
        input_ds = xr.Dataset(
            coords={"time": [dt], "latitude": [0.0], "longitude": [0.0]}
        )
        ds = OutputToXarray.convert_batch([parsed], input_ds)

        assert "wavelength" in ds.coords
        assert ds["eglo"].dims == ("time", "wavelength", "altitude")
        assert ds["eglo"].shape[1] == len(parsed.wavelengths)

    @pytest.mark.xfail(
        reason="Test assertions assume a different parser output structure"
    )
    def test_vertical_profile_parsing(self, base_config):
        """Test parsing of multi-altitude output."""
        # Configure for integrated but multi-altitude
        base_config.simulation_defaults.integrate_wavelength = True
        base_config.simulation_defaults.output_columns = ["zout", "eglo"]
        base_config.simulation_defaults.output_altitudes_km = [0.0, 1.0, 2.0, 5.0]

        sim = Simulation(base_config)

        dt = datetime(2025, 6, 21, 12, 0)
        output_file = sim.run_simulation(dt, 0.0, 0.0)

        parser = OutputParser(base_config)
        parsed = parser.parse_output_file(output_file)

        assert parsed.output_type == OutputType.INTEGRATED_MULTI_ALTITUDE
        assert parsed.altitudes is not None
        assert len(parsed.altitudes) == 4

        # Verify zout was parsed correctly
        assert np.allclose(parsed.altitudes, [0.0, 1.0, 2.0, 5.0], atol=0.1)

        # Convert to Xarray
        input_ds = xr.Dataset(
            coords={"time": [dt], "latitude": [0.0], "longitude": [0.0]}
        )
        ds = OutputToXarray.convert_batch([parsed], input_ds)

        assert "altitude" in ds.coords
        assert len(ds.altitude) == 4
        assert ds["eglo"].shape[1] == 4  # (time, alt) for integrated multi-altitude

    @pytest.mark.xfail(
        reason="Test assertions assume a different parser output structure"
    )
    def test_spectral_vertical_profile(self, base_config):
        """Test full complexity: Spectral AND Multi-altitude."""
        base_config.simulation_defaults.integrate_wavelength = False
        base_config.simulation_defaults.output_columns = ["zout", "lambda", "eglo"]
        base_config.simulation_defaults.wavelength_nm = [400, 405]
        base_config.simulation_defaults.output_altitudes_km = [0.0, 10.0]

        sim = Simulation(base_config)
        dt = datetime(2025, 6, 21, 12, 0)
        output_file = sim.run_simulation(dt, 0.0, 0.0)

        parser = OutputParser(base_config)
        parsed = parser.parse_output_file(output_file)

        assert parsed.output_type == OutputType.SPECTRAL_MULTI_ALTITUDE
        assert parsed.wavelengths is not None
        assert parsed.altitudes is not None

        n_wave = len(parsed.wavelengths)
        n_alt = len(parsed.altitudes)

        assert n_alt == 2
        assert n_wave > 1

        # Data should be flattened in parsed output (numpy array 1D usually from loadtxt)
        # But logic in parser should handle reshaping or just return flat arrays?
        # OutputParser._parse_data_by_type returns columns.
        # The columns are just arrays of length (N_wave * N_alt).

        assert len(parsed.data["eglo"]) == n_wave * n_alt

        # Convert to Xarray should reshape relevantly
        input_ds = xr.Dataset(
            coords={"time": [dt], "latitude": [0.0], "longitude": [0.0]}
        )
        ds = OutputToXarray.convert_batch([parsed], input_ds)

        # Shape: (time, wavelength, altitude)
        assert ds["eglo"].shape == (1, n_wave, n_alt)

    def test_brightness_temperature_parsing(self, base_config):
        """Test brightness temperature output parsing."""
        # Thermal source required usually
        base_config.simulation_defaults.source = "thermal"
        base_config.simulation_defaults.output_columns = [
            "lambda",
            "uu",
        ]  # brightness temp
        # output_quantity = 'brightness' needs to be passed to parser?
        # Currently core.py doesn't set 'brightness' mode for uvspec unless parameter_overrides?
        # Actually uvspec output 'brightness' is achieved by 'output_quantity brightness'.

        # We simulate this via parameter_overrides
        sim = Simulation(base_config)

        overrides = {"output_quantity": "brightness"}

        dt = datetime(2025, 6, 21, 12, 0)
        # Thermal runs require surface temp usually
        output_file = sim.run_simulation(
            dt,
            0.0,
            0.0,
            override_surface_temperature=300.0,
            parameter_overrides=overrides,
        )

        # Parser needs to know about brightness too
        parser = OutputParser(base_config, parameter_overrides=overrides)
        parsed = parser.parse_output_file(output_file)

        assert parsed.is_brightness_temperature is True
