# pyradtran/interface.py
"""
High-level interface for pyradtran:
- PyRadtranAccessor: xarray accessor registered as ds.pyradtran
- execute_simulation_batch: Parallel execution of multiple uvspec runs
- run_pyradtran_simulation: Standalone function for running simulations from a file
"""

import logging
import os
import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from datetime import datetime

from .config import SimulationConfig, load_config
from .exceptions import PyRadtranError
from .core import Simulation
from .io import (
    load_simulation_input_data, 
    parse_uvspec_output, 
    save_results_to_netcdf
)

logger = logging.getLogger(__name__)

# --- High-level standalone function ---

def run_pyradtran_simulation(
    input_file: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    config_path: Optional[Union[str, Path]] = None,
    parameter_overrides: Dict[str, Any] = None,
    max_workers: Optional[int] = None
) -> Path:
    """
    Run a complete pyradtran simulation based on input data from a file.
    
    Args:
        input_file: Path to input data file (CSV/NetCDF) with time, lat, lon
        output_path: Path for output NetCDF file (auto-generated if None)
        config_path: Path to YAML configuration file (uses default if None)
        parameter_overrides: Dictionary of simulation parameters to override
        max_workers: Number of parallel workers (overrides config)
        
    Returns:
        Path to the output NetCDF file
        
    Raises:
        PyRadtranError: If simulation fails
    """
    try:
        # Load configuration
        config = load_config(config_path)
        
        # Override max_workers if specified
        if max_workers is not None:
            config.execution.max_workers = max_workers
        
        # Apply parameter overrides if provided
        if parameter_overrides:
            # Apply parameter overrides to config
            # This is a simplified approach - a more complete implementation
            # would handle nested attributes
            for key, value in parameter_overrides.items():
                parts = key.split('.')
                if len(parts) == 2:
                    section, param = parts
                    if hasattr(config, section) and hasattr(getattr(config, section), param):
                        setattr(getattr(config, section), param, value)
                        logger.info(f"Overriding config: {section}.{param} = {value}")
        
        # Load input data
        input_ds = load_simulation_input_data(input_file)
        
        # Generate output path if not provided
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = Path(config.paths.output_dir) / f"{config.output.filename_prefix}_{timestamp}{config.output.filename_suffix}"
        else:
            output_path = Path(output_path)
        
        # Run the simulation batch
        results = execute_simulation_batch(
            config=config,
            input_ds=input_ds
        )
        
        # Save results
        if results:
            return save_results_to_netcdf(
                data=results,
                output_path=output_path,
                input_ds=input_ds,
                config=config,
                simulation_params=parameter_overrides
            )
        else:
            raise PyRadtranError("Simulation produced no valid results")
    
    except Exception as e:
        logger.error(f"Error running pyradtran simulation: {e}")
        raise PyRadtranError(f"Failed to run simulation: {e}")

# --- Batch execution ---

def execute_simulation_batch(
    config: SimulationConfig,
    input_ds: xr.Dataset,
    time_var: str = 'time',
    lat_var: str = 'latitude',
    lon_var: str = 'longitude',
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> Dict[str, Any]:
    """
    Execute a batch of uvspec simulations based on an input dataset.
    
    Args:
        config: SimulationConfig object
        input_ds: xarray Dataset with time, latitude, longitude coords
        time_var: Name of time dimension/coordinate in the dataset
        lat_var: Name of latitude dimension/coordinate in the dataset
        lon_var: Name of longitude dimension/coordinate in the dataset
        progress_callback: Optional callback function(current, total) for progress updates
        
    Returns:
        Dictionary with parsed output data
        
    Raises:
        PyRadtranError: If all simulations fail
    """
    # Initialize simulation runner
    runner = Simulation(config)
    
    # Extract coordinates
    times = input_ds[time_var].values
    
    # Handle different dataset structures
    if lat_var in input_ds.dims:
        # Lat/lon are dimensions
        latitudes = input_ds[lat_var].values
        longitudes = input_ds[lon_var].values
        # Create combinations
        points = []
        for t in times:
            for lat in latitudes:
                for lon in longitudes:
                    points.append((t, lat, lon))
    else:
        # Lat/lon are per timestamp (coordinates)
        points = [
            (t, input_ds[lat_var].sel({time_var: t}).item(), 
             input_ds[lon_var].sel({time_var: t}).item())
            for t in times
        ]
    
    # Prepare for parallel execution
    total_points = len(points)
    completed = 0
    logger.info(f"Running {total_points} simulations...")
    
    # Store results
    results_dict: Dict[str, Dict[str, List[float]]] = {}
    
    # Run simulations (parallel if configured)
    with ProcessPoolExecutor(max_workers=config.execution.max_workers) as executor:
        # Submit all tasks
        future_to_point = {
            executor.submit(
                _run_single_simulation, runner, t, lat, lon
            ): (t, lat, lon) for t, lat, lon in points
        }
        
        # Process results as they complete
        for future in as_completed(future_to_point):
            t, lat, lon = future_to_point[future]
            try:
                result = future.result()
                if result:
                    # First result initializes the dictionary structure
                    if not results_dict:
                        results_dict = {
                            col: {} if isinstance(result[col], dict) else []
                            for col in result.keys() if not col.startswith('_')
                        }
                    
                    # Add results
                    for col, val in result.items():
                        if col.startswith('_'):
                            # Skip special fields
                            continue
                        
                        if isinstance(val, dict):
                            # Multi-level output
                            for level, level_val in val.items():
                                if level not in results_dict[col]:
                                    results_dict[col][level] = []
                                results_dict[col][level].append(level_val.item() if hasattr(level_val, 'item') else level_val)
                        else:
                            # Single-level output
                            results_dict[col].append(val.item() if hasattr(val, 'item') else val)
                
                completed += 1
                if progress_callback:
                    progress_callback(completed, total_points)
                
                logger.debug(f"Completed {completed}/{total_points} simulations")
            
            except Exception as e:
                logger.error(f"Error in simulation for t={t}, lat={lat}, lon={lon}: {e}")
                completed += 1
    
    logger.info(f"Completed {completed}/{total_points} simulations with {len(results_dict)} output variables")
    
    if not results_dict:
        raise PyRadtranError("All simulations failed, no valid results")
    
    return results_dict

def _run_single_simulation(
    runner: Simulation,
    dt: np.datetime64,
    latitude: float,
    longitude: float
) -> Optional[Dict[str, Any]]:
    """
    Run a single simulation using the Simulation object (used by execute_simulation_batch).
    Converts numpy datetime to Python datetime and handles exceptions.
    """
    try:
        # Convert numpy datetime64 to Python datetime
        py_dt = pd.to_datetime(dt).to_pydatetime()
        
        # Run uvspec
        output_file = runner.run(py_dt, latitude, longitude)
        if output_file and output_file.exists():
            # Parse output
            result = parse_uvspec_output(output_file, runner.config)
            
            # Clean up output file if needed
            if runner.config.execution.cleanup_temp_files:
                try:
                    output_file.unlink()
                except OSError:
                    pass
            
            return result
        else:
            return None
    except Exception as e:
        logger.exception(f"Error in individual simulation: {e}")
        return None

# --- xarray accessor ---

@xr.register_dataset_accessor("pyradtran")
class PyRadtranAccessor:
    """
    xarray accessor for pyradtran functionality.
    Usage: ds.pyradtran.run_uvspec(config_path="config.yaml")
    """
    
    def __init__(self, xarray_obj):
        """Initialize the accessor with an xarray Dataset."""
        self._obj = xarray_obj
        self._config = None
    
    def run_uvspec(
        self,
        config_path: Optional[Union[str, Path]] = None,
        output_path: Optional[Union[str, Path]] = None,
        parameter_overrides: Dict[str, Any] = None,
        time_var: str = 'time',
        lat_var: str = 'latitude',
        lon_var: str = 'longitude',
        return_dataset: bool = True,
        save_to_file: bool = True,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Union[xr.Dataset, Path]:
        """
        Run uvspec for each time/location coordinate in the dataset.
        
        Args:
            config_path: Path to YAML configuration file (uses default if None)
            output_path: Path for output NetCDF file (auto-generated if None)
            parameter_overrides: Dictionary of simulation parameters to override
            time_var: Name of time dimension/coordinate in the dataset
            lat_var: Name of latitude dimension/coordinate in the dataset
            lon_var: Name of longitude dimension/coordinate in the dataset
            return_dataset: If True, return the results as an xarray Dataset
            save_to_file: If True, save results to a NetCDF file
            progress_callback: Optional callback function(current, total) for progress updates
            
        Returns:
            If return_dataset is True, return an xarray Dataset with results
            If save_to_file is True and return_dataset is False, return the output file path
            
        Raises:
            PyRadtranError: If simulation fails
        """
        # Load configuration
        self._config = load_config(config_path)
        
        # Apply parameter overrides if provided
        if parameter_overrides:
            # Apply parameter overrides to config
            for key, value in parameter_overrides.items():
                parts = key.split('.')
                if len(parts) == 2:
                    section, param = parts
                    if hasattr(self._config, section) and hasattr(getattr(self._config, section), param):
                        setattr(getattr(self._config, section), param, value)
                        logger.info(f"Overriding config: {section}.{param} = {value}")
        
        # Validate input dataset
        if time_var not in self._obj.dims and time_var not in self._obj.coords:
            raise PyRadtranError(f"Time variable '{time_var}' not found in dataset")
        
        if lat_var not in self._obj.dims and lat_var not in self._obj.coords and lat_var not in self._obj.data_vars:
            raise PyRadtranError(f"Latitude variable '{lat_var}' not found in dataset")
            
        if lon_var not in self._obj.dims and lon_var not in self._obj.coords and lon_var not in self._obj.data_vars:
            raise PyRadtranError(f"Longitude variable '{lon_var}' not found in dataset")
        
        # Run the simulation batch
        results = execute_simulation_batch(
            config=self._config,
            input_ds=self._obj,
            time_var=time_var,
            lat_var=lat_var,
            lon_var=lon_var,
            progress_callback=progress_callback
        )
        
        # Generate output path if saving and not provided
        if save_to_file:
            if output_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = Path(self._config.paths.output_dir) / f"{self._config.output.filename_prefix}_{timestamp}{self._config.output.filename_suffix}"
            else:
                output_path = Path(output_path)
        
        # Create result dataset
        if results:
            result_path = None
            if save_to_file:
                result_path = save_results_to_netcdf(
                    data=results,
                    output_path=output_path,
                    input_ds=self._obj,
                    config=self._config,
                    simulation_params=parameter_overrides
                )
            
            if return_dataset:
                # Create a dataset from results to return
                ds = xr.Dataset()
                
                # Add coordinates from input dataset
                for coord_name in [time_var, lat_var, lon_var]:
                    if coord_name in self._obj:
                        ds[coord_name] = self._obj[coord_name]
                
                # Add altitude coordinate if we have multi-level data
                has_multi_level = False
                for col_name, values in results.items():
                    if isinstance(values, dict) and not col_name.startswith('_'):  # Multi-level data
                        has_multi_level = True
                        break
                
                if has_multi_level:
                    # Extract altitude levels
                    altitude_levels = []
                    for col_name, level_data in results.items():
                        if isinstance(level_data, dict) and not col_name.startswith('_'):
                            altitude_levels.extend(float(alt) for alt in level_data.keys())
                    
                    # Sort and deduplicate altitude levels
                    altitude_levels = sorted(set(altitude_levels))
                    ds['altitude'] = xr.DataArray(
                        altitude_levels,
                        dims=('altitude',),
                        attrs={'units': 'km', 'long_name': 'Altitude above sea level'}
                    )
                
                # Add variables
                for col_name, values in results.items():
                    if col_name.startswith('_'):
                        continue  # Skip metadata fields
                        
                    if isinstance(values, dict):  # Multi-level data
                        # Create variable with altitude dimension
                        var_data = np.zeros((len(ds[time_var]), len(ds['altitude'])))
                        var_data.fill(np.nan)  # Initialize with NaN values
                        
                        # Fill in data for each altitude level
                        for i, alt in enumerate(ds['altitude'].values):
                            if alt in values and len(values[alt]) > 0:  # Check if we have data
                                # Handle potential size mismatch
                                data_length = min(len(values[alt]), len(ds[time_var]))
                                var_data[:data_length, i] = values[alt][:data_length]
                        
                        ds[col_name] = xr.DataArray(
                            var_data,
                            dims=(time_var, 'altitude'),
                            coords={time_var: ds[time_var], 'altitude': ds['altitude']},
                            attrs={'units': self._get_variable_units(col_name)}
                        )
                    else:  # Single-level data
                        # Convert to numpy array
                        if not isinstance(values, np.ndarray):
                            values = np.array(values, dtype=float)
                        
                        # Ensure 1D shape
                        if values.ndim > 1:
                            values = values.flatten()
                            
                        # Handle potential size mismatch
                        if len(values) == len(ds[time_var]):
                            ds[col_name] = xr.DataArray(
                                values,
                                dims=(time_var,),
                                coords={time_var: ds[time_var]},
                                attrs={'units': self._get_variable_units(col_name)}
                            )
                        else:
                            logger.warning(f"Size mismatch for {col_name}: expected {len(ds[time_var])}, got {len(values)}")
                            # Create with proper length, padding with NaN if needed
                            temp_data = np.full(len(ds[time_var]), np.nan)
                            if len(values) > 0:
                                temp_data[:min(len(values), len(ds[time_var]))] = values[:min(len(values), len(ds[time_var]))]
                            ds[col_name] = xr.DataArray(
                                temp_data,
                                dims=(time_var,),
                                coords={time_var: ds[time_var]},
                                attrs={'units': self._get_variable_units(col_name)}
                            )
                
                # Add metadata
                ds.attrs['generated_by'] = 'pyradtran'
                
                return ds
            else:
                # Return output file path
                return result_path
        else:
            raise PyRadtranError("Simulation produced no valid results")
    
    def _get_variable_units(self, variable_name: str) -> str:
        """Helper function to assign units based on variable name"""
        units_dict = {
            'wavelength': 'nm',
            'sza': 'degrees',
            'eglo': 'W m-2 nm-1',
            'eup': 'W m-2 nm-1',
            'edir': 'W m-2 nm-1',
            'edn': 'W m-2 nm-1',
            'enet': 'W m-2 nm-1',
            'esum': 'W m-2',
            'uavgdir': 'W m-2 nm-1',
            'uavgglo': 'W m-2 nm-1',
            'uavgdn': 'W m-2 nm-1',
            'uavgup': 'W m-2 nm-1',
            'heat': 'K day-1',
            'albedo': 'dimensionless',
            'transmittance': 'dimensionless',
            'reflectivity': 'dimensionless',
            'pressure': 'hPa',
            'temperature': 'K',
            'altitude': 'km'
        }
        
        # Generic name matching for common prefixes
        for prefix, unit in [
            ('e', 'W m-2 nm-1'),       # Irradiance
            ('u', 'W m-2 nm-1 sr-1'),  # Radiance
            ('tau', 'dimensionless'),  # Optical depth
            ('z', 'km')                # Altitude
        ]:
            if variable_name.startswith(prefix):
                return unit
        
        return units_dict.get(variable_name.lower(), 'unknown')