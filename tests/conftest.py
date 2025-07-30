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


@pytest.fixture
def minimal_config():
    """Create a minimal simulation config for testing."""
    return SimulationConfig(
        paths=PathsConfig(
            libradtran_bin=Path('/opt/libradtran/2.0.4/bin/uvspec'),
            libradtran_data=Path('/opt/libradtran/2.0.4/share/libRadtran/data'),
            atmosphere_profile=Path('/projekt_agmwend/data/HALO-AC3/05_VELOX_Tools/add_data/afglsw.dat'),
            solar_spectrum=Path('/projekt_agmwend/home_rad/sophie/libradtran/solar_flux/NewGuey2003.dat'),
            output_dir=Path('./work')
        ),
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
def spectral_config():
    """Create a spectral simulation config for testing."""
    return SimulationConfig(
        paths=PathsConfig(
            libradtran_bin=Path('/opt/libradtran/2.0.4/bin/uvspec'),
            libradtran_data=Path('/opt/libradtran/2.0.4/share/libRadtran/data'),
            atmosphere_profile=Path('/projekt_agmwend/data/HALO-AC3/05_VELOX_Tools/add_data/afglsw.dat'),
            solar_spectrum=Path('/projekt_agmwend/home_rad/sophie/libradtran/solar_flux/NewGuey2003.dat'),
            output_dir=Path('./work')
        ),
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
