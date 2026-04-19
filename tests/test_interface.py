# tests/test_interface.py
"""
Tests for the interface module
"""

import os
import tempfile
from concurrent.futures import Future
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pyradtran.config import (
    ExecutionConfig,
    OutputConfig,
    PathsConfig,
    SimulationConfig,
    SimulationDefaults,
)
from pyradtran.interface import (
    PyRadtranAccessor,
    _run_single_simulation_unified,
    execute_simulation_batch,
    run_pyradtran_simulation,
)
from pyradtran.io import OutputToXarray, OutputType, ParsedOutput


# Fixture for a test dataset
@pytest.fixture
def test_dataset():
    """Create a test xarray dataset with time, lat, lon coordinates"""
    # Create test data
    times = pd.date_range("2023-05-01", periods=3, freq="1H")
    lats = np.array([60.0, 60.1, 60.2])
    lons = np.array([10.0, 10.1, 10.2])

    # Create dataset
    ds = xr.Dataset(
        coords={"time": times, "latitude": ("time", lats), "longitude": ("time", lons)}
    )
    return ds


# Fixture for a minimal config
@pytest.fixture
def minimal_config(tmp_path):
    """Create a minimal configuration for testing"""
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
            output_altitudes_km=[0.0],
        ),
        execution=ExecutionConfig(max_workers=1),
        output=OutputConfig(filename_prefix="test"),
    )


# Tests with mocking
@patch("pyradtran.interface.ProcessPoolExecutor")
@patch("pyradtran.interface.as_completed")
def test_execute_simulation_batch(
    mock_as_completed, mock_executor_class, minimal_config, test_dataset
):
    """Test execute_simulation_batch with mocks"""
    # Setup mock executor
    mock_executor = MagicMock()
    mock_executor_class.return_value.__enter__.return_value = mock_executor

    # Create mock futures
    mock_futures = [MagicMock(spec=Future) for _ in range(3)]

    # Define the output before using it
    mock_parsed_output = ParsedOutput(
        output_type=OutputType.INTEGRATED_SINGLE_ALTITUDE,
        data={
            "sza": np.array([30.0]),
            "eglo": np.array([1000.0]),
            "eup": np.array([100.0]),
            "albedo": np.array([0.1]),
        },
    )

    for f in mock_futures:
        f.result.return_value = mock_parsed_output

    # Configure submit to return a new future each time
    mock_executor.submit.side_effect = mock_futures

    # Configure as_completed to yield the futures
    mock_as_completed.return_value = iter(mock_futures)

    # Call the function
    results = execute_simulation_batch(config=minimal_config, input_ds=test_dataset)

    # Verify the results
    assert isinstance(results, list)
    assert len(results) == 3
    assert isinstance(results[0], ParsedOutput)

    # Verify submit was called 3 times
    assert mock_executor.submit.call_count == 3


@patch("pyradtran.interface.Simulation")
@patch("pyradtran.interface.OutputParser")
def test_run_single_simulation_unified(
    mock_parser_class, mock_simulation_class, minimal_config
):
    """Test _run_single_simulation_unified"""
    # Setup mocks
    mock_simulation = MagicMock()
    mock_simulation_class.return_value = mock_simulation

    mock_parser = MagicMock()
    mock_parser_class.return_value = mock_parser

    mock_output_file = Path("/tmp/mock.out")
    mock_output_file.touch()
    mock_simulation.run_simulation.return_value = mock_output_file

    mock_parsed_output = ParsedOutput(
        output_type=OutputType.INTEGRATED_SINGLE_ALTITUDE, data={}
    )
    mock_parser.parse_output_file.return_value = mock_parsed_output

    # Test data point: (time, lat, lon, albedo, surf_temp, surf_type, altitude, era5_file, point_id)
    dt = datetime(2023, 5, 1, 12, 0)
    point_data = (dt, 60.0, 10.0, 0.2, 290.0, 1.0, 0.0, None, "test_point")

    # Call function
    result = _run_single_simulation_unified(minimal_config, point_data)

    # Verify
    assert result == mock_parsed_output
    mock_simulation.run_simulation.assert_called_once()
    mock_parser.parse_output_file.assert_called_once()


@patch("pyradtran.interface.InputDataLoader")
@patch("pyradtran.interface.execute_simulation_batch")
@patch("pyradtran.interface.NetCDFSaver")
@patch("pyradtran.interface.load_config")
@patch("pyradtran.interface.OutputToXarray")
def test_run_pyradtran_simulation(
    mock_to_xarray,
    mock_load_config,
    mock_saver_class,
    mock_execute,
    mock_loader_class,
    minimal_config,
    test_dataset,
):
    """Test run_pyradtran_simulation with mocks"""
    # Setup mocks
    mock_load_config.return_value = minimal_config

    # Mock loader
    mock_loader = MagicMock()
    mock_loader.load_simulation_input_data.return_value = test_dataset
    mock_loader_class.return_value = mock_loader

    # Mock execution results
    mock_parsed_output = MagicMock(spec=ParsedOutput)
    mock_execute.return_value = [mock_parsed_output] * 3

    # Mock conversion
    mock_result_ds = xr.Dataset()  # Define a dummy result
    mock_to_xarray.return_value.convert_batch.return_value = mock_result_ds

    # Mock save result
    output_path = Path("/tmp/output.nc")
    mock_saver_class.return_value.save_results_to_netcdf.return_value = output_path

    # Call the function
    result = run_pyradtran_simulation(
        input_file="test_input.csv", output_path=output_path
    )

    # Verify the result
    assert result == output_path
    mock_load_config.assert_called_once()
    mock_loader.load_simulation_input_data.assert_called_once()
    mock_execute.assert_called_once()
    mock_saver_class.return_value.save_results_to_netcdf.assert_called_once()


# Test xarray accessor
@patch("pyradtran.interface.execute_simulation_batch")
@patch("pyradtran.interface.NetCDFSaver")
@patch("pyradtran.interface.load_config")
@patch("pyradtran.interface.OutputToXarray")
def test_pyradtran_accessor(
    mock_to_xarray,
    mock_load_config,
    mock_saver_class,
    mock_execute,
    minimal_config,
    test_dataset,
):
    """Test the PyRadtranAccessor"""
    # Setup mocks
    mock_load_config.return_value = minimal_config

    # Mock execution results
    mock_parsed_output = MagicMock(spec=ParsedOutput)
    mock_execute.return_value = [mock_parsed_output] * 3

    # Mock conversion

    # Create a proper mock dataset structure
    mock_result_ds = xr.Dataset(
        data_vars={
            "sza": (("time", "altitude"), np.zeros((3, 1))),
            "eglo": (("time", "altitude"), np.zeros((3, 1))),
            "eup": (("time", "altitude"), np.zeros((3, 1))),
            "albedo": (("time", "altitude"), np.zeros((3, 1))),
        },
        coords={"time": test_dataset.time, "altitude": [0.0]},
    )
    mock_to_xarray.return_value.convert_batch.return_value = mock_result_ds

    # Mock save result
    output_path = Path("/tmp/output.nc")
    mock_saver_class.return_value.save_results_to_netcdf.return_value = output_path

    # Call the accessor method
    result_ds = test_dataset.pyradtran.run(
        config_path=None,
        output_path=output_path,
        return_dataset=True,
        save_to_file=True,
    )

    # Verify the result
    assert isinstance(result_ds, xr.Dataset)
    assert "sza" in result_ds
    assert "eglo" in result_ds
    assert "eup" in result_ds
    assert "albedo" in result_ds
    mock_load_config.assert_called_once()
    mock_execute.assert_called_once()
    mock_saver_class.return_value.save_results_to_netcdf.assert_called_once()
