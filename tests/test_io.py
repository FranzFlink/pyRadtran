# tests/test_io.py
"""
Tests for the IO module
"""

import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

_has_netcdf_backend = True
try:
    import netCDF4  # noqa: F401
except ImportError:
    try:
        import scipy.io.netcdf  # noqa: F401
    except ImportError:
        _has_netcdf_backend = False

from pyradtran.config import (
    CloudParameters,
    ExecutionConfig,
    OutputConfig,
    PathsConfig,
    SimulationConfig,
    SimulationDefaults,
)
from pyradtran.core import Simulation
from pyradtran.io import InputDataLoader, NetCDFSaver, OutputParser


# Fixture for a minimal simulation config
@pytest.fixture
def minimal_config(tmp_path):
    """Create a minimal simulation config for testing"""
    # Create dummy files
    bin_path = tmp_path / "uvspec"
    bin_path.touch()

    data_dir = tmp_path / "data"
    data_dir.mkdir()

    atm_path = tmp_path / "atmosphere.dat"
    atm_path.touch()

    solar_path = tmp_path / "solar.dat"
    solar_path.touch()

    return SimulationConfig(
        paths=PathsConfig(
            libradtran_bin=bin_path,
            libradtran_data=data_dir,
            atmosphere_profile=atm_path,
            solar_spectrum=solar_path,
            output_dir=tmp_path / "output",
            working_dir=tmp_path / "working",
        ),
        simulation_defaults=SimulationDefaults(
            rte_solver="disort",
            wavelength_nm=[400, 700],
            output_columns=["sza", "eglo", "eup", "albedo"],
            output_altitudes_km=[0.0, 1.0],
            clouds=CloudParameters(enabled=False),
        ),
        execution=ExecutionConfig(max_workers=2),
        output=OutputConfig(filename_prefix="test"),
    )


# Fixture for a temporary CSV input file
@pytest.fixture
def temp_csv_input():
    """Create a temporary CSV input file with test data"""
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        # Create test data
        dates = [datetime(2023, 5, 1) + timedelta(hours=i) for i in range(5)]
        data = {
            "time": dates,
            "latitude": [60.0 + 0.1 * i for i in range(5)],
            "longitude": [10.0 + 0.1 * i for i in range(5)],
            "temperature": [273.15 + i for i in range(5)],
        }
        pd.DataFrame(data).to_csv(tmp.name, index=False)
        tmp_path = tmp.name

    yield tmp_path
    # Clean up
    os.unlink(tmp_path)


# Fixture for a temporary NetCDF input file
@pytest.fixture
def temp_nc_input():
    """Create a temporary NetCDF input file with test data"""
    if not _has_netcdf_backend:
        pytest.skip("netCDF4 or scipy required for NetCDF I/O")
    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
        # Create test data
        times = pd.date_range("2023-05-01", periods=5, freq="1h")
        lats = np.array([60.0 + 0.1 * i for i in range(5)])
        lons = np.array([10.0 + 0.1 * i for i in range(5)])
        temp = np.array([273.15 + i for i in range(5)])

        # Create dataset
        ds = xr.Dataset(
            data_vars={"temperature": ("time", temp)},
            coords={
                "time": times,
                "latitude": ("time", lats),
                "longitude": ("time", lons),
            },
        )
        ds.to_netcdf(tmp.name)
        tmp_path = tmp.name

    yield tmp_path
    # Clean up
    os.unlink(tmp_path)


# Tests for data loading
def test_load_simulation_input_data_csv(temp_csv_input):
    """Test loading simulation input data from CSV"""
    loader = InputDataLoader()
    ds = loader.load_simulation_input_data(temp_csv_input)

    assert isinstance(ds, xr.Dataset)
    assert "time" in ds
    assert "latitude" in ds
    assert "longitude" in ds
    assert "temperature" in ds
    assert len(ds.time) == 5


def test_load_simulation_input_data_nc(temp_nc_input):
    """Test loading simulation input data from NetCDF"""
    loader = InputDataLoader()
    ds = loader.load_simulation_input_data(temp_nc_input)

    assert isinstance(ds, xr.Dataset)
    assert "time" in ds
    assert "latitude" in ds
    assert "longitude" in ds
    assert "temperature" in ds
    assert len(ds.time) == 5


# Tests for input generation
def test_generate_uvspec_input_content(minimal_config):
    """Test generating uvspec input content"""
    dt = datetime(2023, 5, 1, 12, 0, 0)
    lat = 60.0
    lon = 10.0

    # Initialize simulation
    sim = Simulation(minimal_config)

    # Generate input content - accessing private method for testing purpose
    content = sim._generate_input_content(dt=dt, latitude=lat, longitude=lon)

    # Verify basic content
    assert isinstance(content, str)
    assert "data_files_path" in content
    assert "atmosphere_file" in content
    assert "rte_solver disort" in content
    assert "wavelength 400 700" in content
    # zout and lambda are auto-injected (multi-altitude, spectral run)
    assert "output_user zout lambda sza eglo eup albedo" in content
    assert "zout 0.0000 1.0000" in content


# Test with clouds enabled
def test_generate_uvspec_input_with_clouds(minimal_config):
    """Test generating uvspec input content with clouds"""
    # Modify the config to include clouds
    minimal_config.simulation_defaults.clouds = CloudParameters(
        enabled=True,
        cloud_type="wc",
        cloud_source="parametric",
        layer_bottom_km=1.0,
        layer_top_km=2.0,
        water_content_g_m3=0.1,
        effective_radius_um=10.0,
    )

    dt = datetime(2023, 5, 1, 12, 0, 0)
    lat = 60.0
    lon = 10.0

    sim = Simulation(minimal_config)

    # Generate input content
    content = sim._generate_input_content(dt=dt, latitude=lat, longitude=lon)

    # Verify cloud content
    assert "wc_layer 1.0 2.0 0.1 10.0" in content
