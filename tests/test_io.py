# tests/test_io.py
"""
Tests for the IO module
"""
import tempfile
import pytest
import os
import pandas as pd
import xarray as xr
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

from pyradtran.io import (
    load_simulation_input_data,
    generate_uvspec_input_content,
    parse_uvspec_output,
    save_results_to_netcdf
)
from pyradtran.config import SimulationConfig, PathsConfig, SimulationDefaults, ExecutionConfig, OutputConfig, CloudParameters, AerosolParameters

# Fixture for a minimal simulation config
@pytest.fixture
def minimal_config():
    """Create a minimal simulation config for testing"""
    return SimulationConfig(
        paths=PathsConfig(
            libradtran_bin=Path('/path/to/uvspec'),
            libradtran_data=Path('/path/to/data'),
            atmosphere_profile=Path('/path/to/atmosphere.dat'),
            solar_spectrum=Path('/path/to/solar.dat'),
            output_dir=Path('/path/to/output'),
            working_dir=Path('/path/to/working')
        ),
        simulation_defaults=SimulationDefaults(
            rte_solver='disort',
            wavelength_nm=[400, 700],
            output_columns=['sza', 'eglo', 'eup', 'albedo'],
            output_altitudes_km=[0.0, 1.0],
            clouds=CloudParameters(enabled=False),
            aerosols=AerosolParameters(enabled=False)
        ),
        execution=ExecutionConfig(max_workers=2),
        output=OutputConfig(filename_prefix='test')
    )

# Fixture for a temporary CSV input file
@pytest.fixture
def temp_csv_input():
    """Create a temporary CSV input file with test data"""
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
        # Create test data
        dates = [datetime(2023, 5, 1) + timedelta(hours=i) for i in range(5)]
        data = {
            'time': dates,
            'latitude': [60.0 + 0.1*i for i in range(5)],
            'longitude': [10.0 + 0.1*i for i in range(5)],
            'temperature': [273.15 + i for i in range(5)]
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
    with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as tmp:
        # Create test data
        times = pd.date_range('2023-05-01', periods=5, freq='1H')
        lats = np.array([60.0 + 0.1*i for i in range(5)])
        lons = np.array([10.0 + 0.1*i for i in range(5)])
        temp = np.array([273.15 + i for i in range(5)])
        
        # Create dataset
        ds = xr.Dataset(
            data_vars={
                'temperature': ('time', temp)
            },
            coords={
                'time': times,
                'latitude': ('time', lats),
                'longitude': ('time', lons)
            }
        )
        ds.to_netcdf(tmp.name)
        tmp_path = tmp.name
    
    yield tmp_path
    # Clean up
    os.unlink(tmp_path)

# Tests for data loading
def test_load_simulation_input_data_csv(temp_csv_input):
    """Test loading simulation input data from CSV"""
    ds = load_simulation_input_data(temp_csv_input)
    
    assert isinstance(ds, xr.Dataset)
    assert 'time' in ds
    assert 'latitude' in ds
    assert 'longitude' in ds
    assert 'temperature' in ds
    assert len(ds.time) == 5

def test_load_simulation_input_data_nc(temp_nc_input):
    """Test loading simulation input data from NetCDF"""
    ds = load_simulation_input_data(temp_nc_input)
    
    assert isinstance(ds, xr.Dataset)
    assert 'time' in ds
    assert 'latitude' in ds
    assert 'longitude' in ds
    assert 'temperature' in ds
    assert len(ds.time) == 5

# Tests for input generation
def test_generate_uvspec_input_content(minimal_config):
    """Test generating uvspec input content"""
    dt = datetime(2023, 5, 1, 12, 0, 0)
    lat = 60.0
    lon = 10.0
    
    # Generate input content
    content = generate_uvspec_input_content(
        config=minimal_config,
        dt=dt,
        latitude=lat,
        longitude=lon
    )
    
    # Verify basic content
    assert isinstance(content, str)
    assert "data_files_path" in content
    assert "atmosphere_file" in content
    assert "time 2023 5 1" in content
    assert "disort" in content  # Check RTE solver
    assert "wavelength 400 700" in content
    assert "output_user sza eglo eup albedo" in content
    assert "zout 0.0 1.0" in content

# Test with clouds enabled
def test_generate_uvspec_input_with_clouds(minimal_config):
    """Test generating uvspec input content with clouds"""
    # Modify the config to include clouds
    minimal_config.simulation_defaults.clouds = CloudParameters(
        enabled=True,
        cloud_optical_properties="mie",
        cloud_overlap="max-random",
        layer_heights_km=[(1.0, 2.0)],
        layer_water_content=[0.1],
        layer_effective_radius_um=[10.0]
    )
    
    dt = datetime(2023, 5, 1, 12, 0, 0)
    lat = 60.0
    lon = 10.0
    
    # Generate input content
    content = generate_uvspec_input_content(
        config=minimal_config,
        dt=dt,
        latitude=lat,
        longitude=lon
    )
    
    # Verify cloud content
    assert "cloud_optical_properties mie" in content
    assert "cloud_overlap max-random" in content
    assert "wc_file 1 0.1 10.0 1.0 2.0" in content

# Test with aerosols enabled
def test_generate_uvspec_input_with_aerosols(minimal_config):
    """Test generating uvspec input content with aerosols"""
    # Modify the config to include aerosols
    minimal_config.simulation_defaults.aerosols = AerosolParameters(
        enabled=True,
        aerosol_type="rural",
        aerosol_visibility_km=23.0,
        aerosol_optical_properties="default"
    )
    
    dt = datetime(2023, 5, 1, 12, 0, 0)
    lat = 60.0
    lon = 10.0
    
    # Generate input content
    content = generate_uvspec_input_content(
        config=minimal_config,
        dt=dt,
        latitude=lat,
        longitude=lon
    )
    
    # Verify aerosol content
    assert "aerosol_optical_properties default" in content
    assert "aerosol_default rural" in content
    assert "aerosol_visibility 23.0" in content