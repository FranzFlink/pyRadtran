# tests/test_integration.py
"""
Integration tests for pyradtran with actual libradtran installation.

These tests verify that pyradtran can properly interact with a real
libradtran (uvspec) installation and produce valid results.
"""

import os
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
from pyradtran.interface import run_pyradtran_simulation
from pyradtran.io import OutputParser

from helpers import has_libradtran


# --- Fixtures ---


@pytest.fixture
def integration_config():
    """Create a config using actual LibRadtran paths from master config."""
    cfg = load_config()
    cfg.simulation_defaults.rte_solver = "disort"
    cfg.simulation_defaults.mol_abs_param = "reptran medium"
    cfg.simulation_defaults.wavelength_nm = [400, 700]
    cfg.simulation_defaults.integrate_wavelength = True
    cfg.simulation_defaults.output_columns = ["sza", "eglo", "eup", "albedo"]
    cfg.simulation_defaults.output_altitudes_km = [0.0]
    cfg.simulation_defaults.albedo_value = 0.3
    cfg.paths.output_dir = Path(tempfile.gettempdir())
    cfg.paths.working_dir = Path(tempfile.gettempdir())
    return cfg


@pytest.fixture
def test_dataset():
    """Create a test dataset with a single time point"""
    # Single time point
    time = pd.to_datetime(["2025-05-05 12:00:00"])
    lat = np.array([75.0])  # Arctic
    lon = np.array([0.0])  # Prime meridian

    # Create dataset
    ds = xr.Dataset(
        coords={"time": time, "latitude": ("time", lat), "longitude": ("time", lon)}
    )
    return ds


# --- Integration Tests ---


@pytest.mark.skipif(not has_libradtran(), reason="LibRadtran not available")
def test_simulation_initialization(integration_config):
    """Test that a Simulation can be initialized with the actual LibRadtran paths"""

    sim = Simulation(integration_config)

    # Verify paths were set correctly
    assert Path(sim.config.paths.libradtran_bin).is_file()
    assert Path(sim.config.paths.libradtran_data).is_dir()


@pytest.mark.skipif(not has_libradtran(), reason="LibRadtran not available")
def test_simple_simulation_run(integration_config):
    """Test running a simple simulation with LibRadtran"""

    # Create a simulation instance
    sim = Simulation(integration_config)

    # Generate input content parameters
    dt = datetime(2025, 5, 5, 12, 0, 0)  # Noon on May 5, 2025
    lat = 75.0  # Arctic
    lon = 0.0  # Prime meridian

    # Use internal method to generate input and run
    # Simulation.run_simulation is the main entry point now
    try:
        # We need to construct a proper input file path or let run_simulation do it
        # run_simulation(dt, lat, lon, ...)
        output_file = sim.run_simulation(dt=dt, latitude=lat, longitude=lon)

        # Verify output file exists
        assert output_file.exists(), f"Output file {output_file} does not exist"
        assert output_file.stat().st_size > 0, "Output file is empty"

        # Parse the output using OutputParser
        parser = OutputParser(integration_config)
        parsed_output = parser.parse_output_file(output_file)

        # Verify output contains expected columns
        for column in integration_config.simulation_defaults.output_columns:
            assert column in parsed_output.data, f"Column {column} missing from output"

        # Verify SZA is reasonable (not NaN)
        assert not np.isnan(parsed_output.data["sza"][0]), "SZA is NaN"

        # Verify irradiance is reasonable (positive value)
        assert parsed_output.data["eglo"][0] > 0, "Global irradiance should be positive"

    finally:
        # Clean up temporary files if needed
        pass


@pytest.mark.skipif(not has_libradtran(), reason="LibRadtran not available")
def test_xarray_integration(integration_config):
    """Test xarray integration with PyRadtran"""

    # Create dataset
    time = pd.to_datetime(["2025-05-05 12:00:00"])
    lat = np.array([75.0])
    lon = np.array([0.0])

    ds = xr.Dataset(
        coords={"time": time, "latitude": ("time", lat), "longitude": ("time", lon)}
    )

    # Get a temporary output path
    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
        output_path = Path(tmp.name)

    try:
        # Run simulation using xarray accessor (this will register the accessor)
        import pyradtran  # Import to register accessor

        # Run with the accessor
        result = ds.pyradtran.run(
            config=integration_config,  # Pass config directly
            output_path=output_path,
            return_dataset=True,
            save_to_file=True,
        )

        # Verify result is an xarray Dataset
        assert isinstance(result, xr.Dataset), "Result should be an xarray Dataset"

        # Verify it contains the expected variables
        for column in integration_config.simulation_defaults.output_columns:
            assert column in result, f"Variable {column} missing from result"

        # Verify data values are reasonable
        assert not np.isnan(result.sza.values).any(), "SZA contains NaN values"
        assert (result.eglo.values > 0).all(), "Global irradiance should be positive"

        # Verify output file was created
        assert output_path.exists(), f"Output file {output_path} was not created"

    finally:
        # Clean up
        if output_path.exists():
            os.unlink(output_path)


@pytest.mark.skipif(not has_libradtran(), reason="LibRadtran not available")
def test_disort_vs_twostr_comparison(integration_config):
    """Test comparing disort and twostr radiative transfer solvers"""

    # Create dataset
    time = pd.to_datetime(["2025-05-05 12:00:00"])
    lat = np.array([75.0])
    lon = np.array([0.0])

    ds = xr.Dataset(
        coords={"time": time, "latitude": ("time", lat), "longitude": ("time", lon)}
    )

    # Run with disort (already set in config)
    disort_result = ds.pyradtran.run(
        config=integration_config, return_dataset=True, save_to_file=False
    )

    # Update config to use twostr
    import copy

    twostr_config = copy.deepcopy(integration_config)
    twostr_config.simulation_defaults.rte_solver = "twostr"

    # Run with twostr
    twostr_result = ds.pyradtran.run(
        config=twostr_config, return_dataset=True, save_to_file=False
    )

    # Verify both have expected columns
    for column in integration_config.simulation_defaults.output_columns:
        assert column in disort_result, f"Column {column} missing from disort result"
        assert column in twostr_result, f"Column {column} missing from twostr result"

    # Verify SZA is the same (should be identical for same time/location)
    np.testing.assert_allclose(
        disort_result.sza.values,
        twostr_result.sza.values,
        rtol=1e-5,
        err_msg="SZA should be identical between solvers",
    )

    # Verify irradiance differences are within reasonable bounds
    # twostr is typically less accurate than disort but faster
    # Differences should not exceed 10%
    eglo_diff_pct = (
        abs(disort_result.eglo.values - twostr_result.eglo.values)
        / disort_result.eglo.values
        * 100
    )
    assert eglo_diff_pct.max() < 10, "Irradiance difference between solvers exceeds 10%"
