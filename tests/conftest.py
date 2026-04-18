# tests/conftest.py
"""
Shared test fixtures and configuration for pyradtran tests.
"""

import pytest
import tempfile
from pathlib import Path
import pandas as pd
import xarray as xr
import numpy as np

from pyradtran.config import SimulationConfig, PathsConfig, SimulationDefaults, ExecutionConfig, OutputConfig


def _make_mock_paths(tmp_path: Path) -> PathsConfig:
    """Helper: create a PathsConfig backed by real (empty) files in tmp_path."""
    bin_path = tmp_path / "uvspec"
    bin_path.touch(exist_ok=True)
    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)
    atm_file = tmp_path / "atm.dat"
    atm_file.touch(exist_ok=True)
    solar_file = tmp_path / "solar.dat"
    solar_file.touch(exist_ok=True)
    work_dir = tmp_path / "work"
    work_dir.mkdir(exist_ok=True)

    return PathsConfig(
        libradtran_bin=bin_path,
        libradtran_data=data_dir,
        atmosphere_profile=atm_file,
        solar_spectrum=solar_file,
        output_dir=work_dir,
        working_dir=work_dir,
    )


@pytest.fixture
def minimal_config(tmp_path):
    """Create a minimal simulation config for testing."""
    return SimulationConfig(
        paths=_make_mock_paths(tmp_path),
        simulation_defaults=SimulationDefaults(
            wavelength_nm=[400, 3200],
            output_altitudes_km=[5, 6, 7],
            rte_solver='disort',
            mol_abs_param='lowtran per_nm',
            albedo_value=0.1
        ),
        execution=ExecutionConfig(
            max_workers=1,
            timeout_seconds=60,
            cleanup_temp_files=True
        ),
        output=OutputConfig(
            filename_prefix="test_sim",
            filename_suffix="_test.nc"
        )
    )


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def spectral_config(tmp_path):
    """Create a spectral simulation config for testing."""
    return SimulationConfig(
        paths=_make_mock_paths(tmp_path),
        simulation_defaults=SimulationDefaults(
            wavelength_nm=[400, 600],
            output_altitudes_km=[5, 6, 7],
            rte_solver='disort',
            mol_abs_param='lowtran per_nm',
            albedo_value=0.1
        ),
        execution=ExecutionConfig(
            max_workers=1,
            timeout_seconds=60,
            cleanup_temp_files=True
        ),
        output=OutputConfig(
            filename_prefix="test_spectral",
            filename_suffix="_spectral.nc"
        )
    )


@pytest.fixture
def sample_uvspec_output_spectral():
    """Sample uvspec output for spectral multi-altitude testing."""
    return """  400.000    5.000   100.000    50.000    10.000     5.000
  400.000    6.000   110.000    55.000    11.000     5.500
  400.000    7.000   120.000    60.000    12.000     6.000
  500.000    5.000   150.000    75.000    15.000     7.500
  500.000    6.000   160.000    80.000    16.000     8.000
  500.000    7.000   170.000    85.000    17.000     8.500
  600.000    5.000   200.000   100.000    20.000    10.000
  600.000    6.000   210.000   105.000    21.000    10.500
  600.000    7.000   220.000   110.000    22.000    11.000
"""


@pytest.fixture
def sample_uvspec_output_integrated():
    """Sample uvspec output for integrated multi-altitude testing."""
    return """    5.000   100.000    50.000    10.000     5.000    25.000
    6.000   110.000    55.000    11.000     5.500    27.500
    7.000   120.000    60.000    12.000     6.000    30.000
"""


@pytest.fixture
def sample_netcdf_file(tmp_path):
    """Create a sample NetCDF file for testing."""
    file_path = tmp_path / "sample.nc"
    
    # Create sample dataset with the required variables for load_simulation_input_data
    times = pd.date_range('2023-05-01', periods=3, freq='1h')
    lats = [60.0, 60.1, 60.2]
    lons = [10.0, 10.1, 10.2]
    
    ds = xr.Dataset(
        data_vars={
            'latitude': (['time'], lats),
            'longitude': (['time'], lons),
            'altitude': (['time'], [0.0, 0.0, 0.0]),  # Surface altitude
        },
        coords={
            'time': times,
        }
    )
    
    ds.to_netcdf(file_path)
    return file_path


@pytest.fixture
def synthetic_era5_ds():
    """Synthetic ERA5-style dataset for unit tests (no network required)."""
    n = 13
    pressure_hpa = np.linspace(1000, 100, n)
    geopotential = np.linspace(0, 160_000, n)   # m/s^2
    temperature = np.linspace(290, 215, n)        # K
    q = np.linspace(1e-2, 1e-5, n)               # kg/kg

    return xr.Dataset(
        {
            "z": (["pressure_level"], geopotential, {"units": "m2 s-2"}),
            "t": (["pressure_level"], temperature,  {"units": "K"}),
            "q": (["pressure_level"], q,             {"units": "kg kg-1"}),
            "clwc": (["pressure_level"], np.where(
                (pressure_hpa > 300) & (pressure_hpa < 800), 1e-4, 0.0
            ), {"units": "kg kg-1"}),
            "ciwc": (["pressure_level"], np.where(
                pressure_hpa < 400, 5e-5, 0.0
            ), {"units": "kg kg-1"}),
            "cc": (["pressure_level"], np.where(
                (pressure_hpa > 300) & (pressure_hpa < 800), 0.5, 0.0
            ), {"units": "1"}),
        },
        coords={
            "pressure_level": (
                ["pressure_level"], pressure_hpa, {"units": "hPa"}
            ),
            "valid_time": pd.Timestamp("2022-07-01T12:00"),
            "latitude":  70.0,
            "longitude": 25.0,
        },
    )


@pytest.fixture
def simple_input_dataset():
    """Three-point trajectory dataset for interface tests."""
    times = pd.date_range("2022-07-01 12:00", periods=3, freq="30min")
    return xr.Dataset(
        data_vars={
            "latitude":  (["time"], [78.0, 78.1, 78.2]),
            "longitude": (["time"], [15.0, 15.1, 15.2]),
            "altitude":  (["time"], [0.0, 0.0, 0.0]),
        },
        coords={"time": times},
    )
