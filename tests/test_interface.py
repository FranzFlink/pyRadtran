# tests/test_interface.py
"""
Tests for the interface module
"""
import pytest
import xarray as xr
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import tempfile
import os
from unittest.mock import patch, MagicMock

from pyradtran.interface import (
    run_pyradtran_simulation,
    execute_simulation_batch,
    PyRadtranAccessor,
    _run_single_simulation
)
from pyradtran.config import SimulationConfig, PathsConfig, SimulationDefaults, ExecutionConfig, OutputConfig

# Fixture for a test dataset
@pytest.fixture
def test_dataset():
    """Create a test xarray dataset with time, lat, lon coordinates"""
    # Create test data
    times = pd.date_range('2023-05-01', periods=3, freq='1H')
    lats = np.array([60.0, 60.1, 60.2])
    lons = np.array([10.0, 10.1, 10.2])
    
    # Create dataset
    ds = xr.Dataset(
        coords={
            'time': times,
            'latitude': ('time', lats),
            'longitude': ('time', lons)
        }
    )
    return ds

# Fixture for a minimal config
@pytest.fixture
def minimal_config():
    """Create a minimal configuration for testing"""
    return SimulationConfig(
        paths=PathsConfig(
            libradtran_bin=Path('/path/to/uvspec'),
            libradtran_data=Path('/path/to/data'),
            atmosphere_profile=Path('/path/to/atmosphere.dat'),
            solar_spectrum=Path('/path/to/solar.dat'),
            output_dir=Path(tempfile.gettempdir()),
            working_dir=Path(tempfile.gettempdir())
        ),
        simulation_defaults=SimulationDefaults(
            rte_solver='disort',
            wavelength_nm=[400, 700],
            output_columns=['sza', 'eglo', 'eup', 'albedo'],
            output_altitudes_km=[0.0]
        ),
        execution=ExecutionConfig(max_workers=1),
        output=OutputConfig(filename_prefix='test')
    )

# Tests with mocking to avoid actual execution
@patch('pyradtran.interface.Simulation')
@patch('pyradtran.interface.parse_uvspec_output')
@patch('pyradtran.interface.save_results_to_netcdf')
def test_execute_simulation_batch(mock_save, mock_parse, mock_simulation_class, minimal_config, test_dataset):
    """Test execute_simulation_batch with mocks"""
    # Setup mocks
    mock_simulation = MagicMock()
    mock_simulation_class.return_value = mock_simulation
    
    # Mock output file
    mock_output_file = Path('/tmp/mock_output.dat')
    mock_simulation.run.return_value = mock_output_file
    
    # Mock parsed output
    mock_parse.return_value = {
        'sza': [30.0, 31.0, 32.0],
        'eglo': [1000.0, 1010.0, 1020.0],
        'eup': [100.0, 101.0, 102.0],
        'albedo': [0.1, 0.11, 0.12]
    }
    
    # Call the function
    results = execute_simulation_batch(
        config=minimal_config,
        input_ds=test_dataset
    )
    
    # Verify the results
    assert isinstance(results, dict)
    assert 'sza' in results
    assert 'eglo' in results
    assert 'eup' in results
    assert 'albedo' in results
    
    # Verify simulation was called for each time/lat/lon combination
    assert mock_simulation.run.call_count == 3

@patch('pyradtran.interface.load_simulation_input_data')
@patch('pyradtran.interface.execute_simulation_batch')
@patch('pyradtran.interface.save_results_to_netcdf')
@patch('pyradtran.interface.load_config')
def test_run_pyradtran_simulation(mock_load_config, mock_save, mock_execute, mock_load_data, minimal_config, test_dataset):
    """Test run_pyradtran_simulation with mocks"""
    # Setup mocks
    mock_load_config.return_value = minimal_config
    mock_load_data.return_value = test_dataset
    
    # Mock execution results
    mock_execute.return_value = {
        'sza': [30.0, 31.0, 32.0],
        'eglo': [1000.0, 1010.0, 1020.0],
        'eup': [100.0, 101.0, 102.0],
        'albedo': [0.1, 0.11, 0.12]
    }
    
    # Mock save result
    output_path = Path('/tmp/output.nc')
    mock_save.return_value = output_path
    
    # Call the function
    result = run_pyradtran_simulation(
        input_file='test_input.csv',
        output_path=output_path
    )
    
    # Verify the result
    assert result == output_path
    mock_load_config.assert_called_once()
    mock_load_data.assert_called_once()
    mock_execute.assert_called_once()
    mock_save.assert_called_once()

# Test xarray accessor
@patch('pyradtran.interface.execute_simulation_batch')
@patch('pyradtran.interface.save_results_to_netcdf')
@patch('pyradtran.interface.load_config')
def test_pyradtran_accessor(mock_load_config, mock_save, mock_execute, minimal_config, test_dataset):
    """Test the PyRadtranAccessor"""
    # Setup mocks
    mock_load_config.return_value = minimal_config
    
    # Mock execution results
    mock_execute.return_value = {
        'sza': [30.0, 31.0, 32.0],
        'eglo': [1000.0, 1010.0, 1020.0],
        'eup': [100.0, 101.0, 102.0],
        'albedo': [0.1, 0.11, 0.12]
    }
    
    # Mock save result
    output_path = Path('/tmp/output.nc')
    mock_save.return_value = output_path
    
    # Call the accessor method
    result_ds = test_dataset.pyradtran.run_uvspec(
        config_path=None,
        output_path=output_path,
        return_dataset=True
    )
    
    # Verify the result
    assert isinstance(result_ds, xr.Dataset)
    assert 'sza' in result_ds
    assert 'eglo' in result_ds
    assert 'eup' in result_ds
    assert 'albedo' in result_ds
    mock_load_config.assert_called_once()
    mock_execute.assert_called_once()
    mock_save.assert_called_once()