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
    save_results_to_netcdf,
    create_era5_atmosphere_file
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
    albedo_var: Optional[str] = None,
    surface_temperature_var: Optional[str] = None,
    altitude_var: Optional[str] = None,
    era5_atmosphere: Optional[xr.Dataset] = None,
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
        albedo_var: Optional name of albedo data_var in the dataset
        surface_temperature_var: Optional name of surface temperature data_var in the dataset
        altitude_var: Optional name of altitude data_var in the dataset (treated as scalar)
        era5_atmosphere: Optional ERA5 dataset for custom atmosphere profiles
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
    albedos = None
    if albedo_var and albedo_var in input_ds:
        albedos = input_ds[albedo_var].values
    
    surface_temperatures = None
    if surface_temperature_var and surface_temperature_var in input_ds:
        surface_temperatures = input_ds[surface_temperature_var].values
    
    altitudes = None
    if altitude_var and altitude_var in input_ds:
        altitudes = input_ds[altitude_var].values
    
    # Handle ERA5 atmosphere files if provided
    era5_atmosphere_files = {}
    if era5_atmosphere is not None:
        logger.info("Creating ERA5 atmosphere files for simulation points...")
        # Create working directory for atmosphere files
        atm_dir = config.paths.working_dir / "era5_atmospheres"
        atm_dir.mkdir(parents=True, exist_ok=True)

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
                    point_id = None
                    # Create ERA5 atmosphere file if needed
                    if era5_atmosphere is not None:
                        try:
                            point_id = f"{pd.to_datetime(t).strftime('%Y%m%d_%H%M%S')}_{lat:.2f}_{lon:.2f}"
                            atm_file = atm_dir / f"era5_atm_{point_id}.dat"
                            era5_atmosphere_files[point_id] = create_era5_atmosphere_file(
                                era5_atmosphere, lat, lon, t, atm_file
                            )
                            logger.debug(f"Created ERA5 atmosphere file for {point_id}: {atm_file}")
                        except Exception as e:
                            logger.error(f"Failed to create ERA5 atmosphere file for lat={lat}, lon={lon}, time={t}: {e}")
                            era5_atmosphere_files[point_id] = None
                    
                    points.append((t, lat, lon, None, None, None, point_id))
    else:
        # Lat/lon are per timestamp (coordinates)
        points = []
        for i, t in enumerate(times):
            lat = input_ds[lat_var].sel({time_var: t}).item()
            lon = input_ds[lon_var].sel({time_var: t}).item()
            alb = albedos[i] if albedos is not None and i < len(albedos) else None
            surf_temp = surface_temperatures[i] if surface_temperatures is not None and i < len(surface_temperatures) else None
            alt = altitudes[i] if altitudes is not None and i < len(altitudes) else None
            point_id = None
            
            # Create ERA5 atmosphere file if needed
            if era5_atmosphere is not None:
                try:
                    point_id = f"{pd.to_datetime(t).strftime('%Y%m%d_%H%M%S')}_{lat:.2f}_{lon:.2f}"
                    atm_file = atm_dir / f"era5_atm_{point_id}.dat"
                    era5_atmosphere_files[point_id] = create_era5_atmosphere_file(
                        era5_atmosphere, lat, lon, t, atm_file
                    )
                    logger.debug(f"Created ERA5 atmosphere file for {point_id}: {atm_file}")
                except Exception as e:
                    logger.error(f"Failed to create ERA5 atmosphere file for lat={lat}, lon={lon}, time={t}: {e}")
                    era5_atmosphere_files[point_id] = None
            
            points.append((t, lat, lon, alb, surf_temp, alt, point_id))
    
    # Important: We're not creating separate simulation points for each altitude level,
    # as altitude levels are handled by a single libRadtran run
    
    # Prepare for parallel execution
    total_points = len(points)
    completed = 0
    logger.info(f"Running {total_points} simulations across time/location points...")
    
    # Store results
    results_dict: Dict[str, Dict[str, List[float]]] = {}
    
    # Store metadata about result structure
    metadata = {
        '_simulation_type': None,
        '_wavelength_values': [],
        '_altitude_values': []
    }
    
    # Check if we're using scalar altitude (altitude_var is provided)
    using_scalar_altitude = altitude_var is not None
    
    # Run simulations (parallel if configured)
    with ProcessPoolExecutor(max_workers=config.execution.max_workers) as executor:
        # Submit all tasks
        future_to_point = {}
        for t, lat, lon, alb, surf_temp, alt, point_id in points:
            era5_atm_file = era5_atmosphere_files.get(point_id) if point_id else None
            future_to_point[
                executor.submit(
                    _run_single_simulation, runner, t, lat, lon, alb, surf_temp, alt, era5_atm_file
                )
            ] = (t, lat, lon)
        
        # Process results as they complete
        for future in as_completed(future_to_point):
            t, lat, lon = future_to_point[future]
            try:
                result = future.result()
                if result:
                    # Store simulation type if we don't have it yet
                    if '_simulation_type' in result and metadata['_simulation_type'] is None:
                        # Modify simulation type for scalar altitude cases
                        if using_scalar_altitude and result['_simulation_type'] == 'spectral_multi_altitude':
                            metadata['_simulation_type'] = 'spectral_scalar_altitude'
                        elif using_scalar_altitude and result['_simulation_type'] == 'multi_altitude':
                            metadata['_simulation_type'] = 'scalar_altitude'
                        else:
                            metadata['_simulation_type'] = result['_simulation_type']
                    
                    # Store wavelength values if spectral simulation
                    if '_wavelength_values' in result and not metadata['_wavelength_values']:
                        metadata['_wavelength_values'] = result['_wavelength_values']
                    
                    # Store altitude values - handle scalar altitude differently
                    if not using_scalar_altitude and '_unique_altitudes' in result and not metadata['_altitude_values']:
                        metadata['_altitude_values'] = result['_unique_altitudes']
                    
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
    
    # Clean up ERA5 atmosphere files if configured to do so
    if era5_atmosphere is not None and config.execution.cleanup_temp_files:
        logger.debug("Cleaning up ERA5 atmosphere files...")
        for point_id, atm_file in era5_atmosphere_files.items():
            if atm_file and atm_file.exists():
                try:
                    atm_file.unlink()
                    logger.debug(f"Cleaned up ERA5 atmosphere file: {atm_file}")
                except OSError as e:
                    logger.warning(f"Failed to clean up ERA5 atmosphere file {atm_file}: {e}")
    
    if not results_dict:
        raise PyRadtranError("All simulations failed, no valid results")
    
    # Add metadata to the results
    for key, value in metadata.items():
        if value is not None and (not isinstance(value, list) or len(value) > 0):
            results_dict[key] = value
    
    return results_dict

def _run_single_simulation(
    runner: Simulation,
    dt: np.datetime64,
    latitude: float,
    longitude: float,
    albedo: Optional[float] = None,
    surface_temperature: Optional[float] = None,
    altitude: Optional[float] = None,
    era5_atmosphere_file: Optional[Path] = None
) -> Optional[Dict[str, Any]]:
    """
    Run a single simulation using the Simulation object (used by execute_simulation_batch).
    Converts numpy datetime to Python datetime and handles exceptions.
    """
    try:
        # Convert numpy datetime64 to Python datetime
        py_dt = pd.to_datetime(dt).to_pydatetime()
        
        # Run uvspec
        output_file = runner.run(py_dt, latitude, longitude, override_albedo=albedo, override_surface_temperature=surface_temperature, override_altitude_km=altitude, era5_atmosphere_file=era5_atmosphere_file)
        if output_file and output_file.exists():
            # Parse output - create a modified config for scalar altitude cases
            parse_config = runner.config
            if altitude is not None:
                # For scalar altitude cases, modify the config to indicate single altitude level
                # Create a copy of the config to avoid modifying the original
                import copy
                parse_config = copy.deepcopy(runner.config)
                parse_config.simulation_defaults.output_altitudes_km = [altitude]
                logger.debug(f"Modified config for scalar altitude parsing: using single altitude {altitude} km")
            
            result = parse_uvspec_output(output_file, parse_config)
            
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
        albedo_var: Optional[str] = None,
        surface_temperature_var: Optional[str] = None,
        era5_atmosphere: Optional[xr.Dataset] = None,
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
            albedo_var: Optional name of albedo data_var in the dataset
            surface_temperature_var: Optional name of surface temperature data_var in the dataset
            era5_atmosphere: Optional ERA5 xarray Dataset for custom atmosphere profiles
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
        
        # Validate albedo variable if provided
        if albedo_var and albedo_var not in self._obj.data_vars:
            raise PyRadtranError(f"Albedo variable '{albedo_var}' not found in dataset data_vars")
        
        # Validate surface temperature variable if provided
        if surface_temperature_var and surface_temperature_var not in self._obj.data_vars:
            raise PyRadtranError(f"Surface temperature variable '{surface_temperature_var}' not found in dataset data_vars")
        
        # Validate ERA5 atmosphere dataset if provided
        if era5_atmosphere is not None:
            required_era5_vars = ['z', 't', 'o3', 'q']
            required_era5_coords = ['pressure_level', 'latitude', 'longitude', 'valid_time']
            
            for var in required_era5_vars:
                if var not in era5_atmosphere.variables:
                    raise PyRadtranError(f"Required variable '{var}' not found in ERA5 atmosphere dataset")
            
            for coord in required_era5_coords:
                if coord not in era5_atmosphere.coords:
                    raise PyRadtranError(f"Required coordinate '{coord}' not found in ERA5 atmosphere dataset")
                    
            logger.info(f"ERA5 atmosphere dataset validated with {len(era5_atmosphere.pressure_level)} pressure levels")

        # Check if we have altitude information in the input dataset
        alt_var = 'altitude'
        altitude_as_coordinate = False
        altitude_as_data_var = False
        
        if alt_var in self._obj.dims or alt_var in self._obj.coords:
            # Altitude is a coordinate - use as list of zout levels
            altitude_as_coordinate = True
            dataset_altitudes = self._obj[alt_var].values
            
            # Override configuration with dataset altitudes if any are provided
            if len(dataset_altitudes) > 0:
                logger.info(f"Altitude found as coordinate - using {len(dataset_altitudes)} levels for zout: {dataset_altitudes}")
                self._config.simulation_defaults.output_altitudes_km = [float(alt) for alt in dataset_altitudes]
                
        elif alt_var in self._obj.data_vars:
            # Altitude is a data variable - treat as scalar per time step
            altitude_as_data_var = True
            logger.info(f"Altitude found as data variable - will be treated as scalar altitude for each time step")
        
        # Generate output path if saving and not provided
        if save_to_file:
            if output_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = Path(self._config.paths.output_dir) / f"{self._config.output.filename_prefix}_{timestamp}{self._config.output.filename_suffix}"
                # Make sure output directory exists
                output_path.parent.mkdir(exist_ok=True, parents=True)
                logger.info(f"Auto-generating output path: {output_path}")
            else:
                output_path = Path(output_path)
                # Make sure the parent directory exists
                output_path.parent.mkdir(exist_ok=True, parents=True)
        
        # Run the simulation batch
        results = execute_simulation_batch(
            config=self._config,
            input_ds=self._obj,
            time_var=time_var,
            lat_var=lat_var,
            lon_var=lon_var,
            albedo_var=albedo_var,
            surface_temperature_var=surface_temperature_var,
            altitude_var=alt_var if altitude_as_data_var else None,
            era5_atmosphere=era5_atmosphere,
            progress_callback=progress_callback
        )
        
        # Create result dataset
        if results:
            # For scalar altitude cases, convert dictionary results to lists
            if altitude_as_data_var:
                # Check if we have scalar altitude results that need flattening
                simulation_type = results.get('_simulation_type', 'standard')
                if simulation_type in ['scalar_altitude', 'spectral_scalar_altitude']:
                    # Convert dictionaries to simple lists for scalar altitude variables
                    for col_name, values in list(results.items()):
                        if not col_name.startswith('_') and isinstance(values, dict):
                            # For scalar altitude, flatten the dictionary values into a simple list
                            # The keys should be time indices and values should be scalars
                            flattened_values = []
                            for time_idx in sorted(values.keys()):
                                if isinstance(values[time_idx], (list, tuple)) and len(values[time_idx]) == 1:
                                    # Single value wrapped in list/tuple
                                    flattened_values.append(values[time_idx][0])
                                elif isinstance(values[time_idx], (int, float)):
                                    # Scalar value
                                    flattened_values.append(values[time_idx])
                                else:
                                    # Keep as is for more complex structures
                                    flattened_values.append(values[time_idx])
                            results[col_name] = flattened_values
                            logger.debug(f"Converted scalar altitude variable {col_name} from dict to list with {len(flattened_values)} values")
            
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
                
                # Check if we have spectral data
                has_spectral_data = False
                wavelength_values = None
                
                # Check for spectral data metadata from results
                if '_simulation_type' in results and results['_simulation_type'] in ['spectral', 'spectral_multi_altitude', 'spectral_scalar_altitude']:
                    has_spectral_data = True
                    if '_wavelength_values' in results:
                        wavelength_values = results['_wavelength_values']
                        logger.info(f"Found spectral data with {len(wavelength_values)} wavelength points")
                
                # Check if we have multi-level altitude data
                has_multi_level = False
                unique_altitudes = []
                
                # Check if we have scalar altitude data
                has_scalar_altitude = False
                simulation_type = results.get('_simulation_type', 'standard')
                
                if simulation_type in ['scalar_altitude', 'spectral_scalar_altitude']:
                    has_scalar_altitude = True
                    logger.info(f"Detected scalar altitude simulation type: {simulation_type}")
                elif simulation_type in ['multi_altitude', 'spectral_multi_altitude']:
                    # First check if config specifies multiple altitude levels
                    # This ensures dimensions are preserved even if not explicitly in results
                    if hasattr(self._config.simulation_defaults, 'output_altitudes_km'):
                        if len(self._config.simulation_defaults.output_altitudes_km) > 1:
                            has_multi_level = True
                            unique_altitudes = sorted(set(float(alt) for alt in self._config.simulation_defaults.output_altitudes_km))
                            logger.debug(f"Using altitude levels from config: {unique_altitudes}")
                    
                    # Then check for the special _unique_altitudes metadata field from results
                    if '_unique_altitudes' in results:
                        has_multi_level = True
                        unique_altitudes = results['_unique_altitudes']
                        logger.debug(f"Found '_unique_altitudes' with {len(unique_altitudes)} levels in results")
                    
                    # Fallback: check if values are stored in altitude-indexed dictionaries
                    if not has_multi_level and not has_spectral_data:
                        for col_name, values in results.items():
                            if isinstance(values, dict) and not col_name.startswith('_'):
                                # This could be multi-level data
                                # Check if the keys look like altitude values (can be converted to float)
                                try:
                                    # Convert a sample key to float to check if it's a number (altitude)
                                    sample_key = next(iter(values.keys()))
                                    float(sample_key)  # Just to check if it raises an error
                                    has_multi_level = True
                                    unique_altitudes.extend(float(alt) for alt in values.keys())
                                    logger.debug(f"Detected multi-level data in variable {col_name} with keys: {list(values.keys())}")
                                    break
                                except (ValueError, StopIteration):
                                    # Not numeric keys or empty dict
                                    continue
                
                # If we have spectral data, add the wavelength dimension
                if has_spectral_data and wavelength_values is not None:
                    # Define wavelength variable name
                    wl_var = 'wavelength'
                    # Sort and deduplicate wavelength values
                    unique_wavelengths = sorted(set(float(wl) for wl in wavelength_values))
                    ds[wl_var] = xr.DataArray(
                        unique_wavelengths,
                        dims=(wl_var,),
                        attrs={'units': 'nm', 'long_name': 'Wavelength'}
                    )
                    logger.info(f"Created wavelength dimension with {len(unique_wavelengths)} points: {min(unique_wavelengths)} to {max(unique_wavelengths)} nm")
                
                # If we have altitude levels, create the altitude dimension
                if has_multi_level:
                    # Define altitude variable name
                    alt_var = 'altitude'
                    # Sort and deduplicate altitude levels
                    unique_altitudes = sorted(set(float(alt) for alt in unique_altitudes))
                    ds[alt_var] = xr.DataArray(
                        unique_altitudes,
                        dims=(alt_var,),
                        attrs={'units': 'km', 'long_name': 'Altitude above sea level'}
                    )
                    logger.info(f"Created altitude dimension with {len(unique_altitudes)} levels: {unique_altitudes}")
                
                # Add variables based on data type
                for col_name, values in results.items():
                    if col_name.startswith('_'):
                        continue  # Skip metadata fields
                        
                    # Handle scalar altitude cases
                    if has_scalar_altitude:
                        # For scalar altitude, values should be simple lists
                        if isinstance(values, list):
                            self._add_simple_variable(ds, col_name, values, time_var)
                        else:
                            logger.warning(f"Expected list for scalar altitude variable {col_name}, got {type(values)}")
                    
                    # Handle multi-level cases (original logic)
                    elif isinstance(values, dict):
                        if has_spectral_data and not has_multi_level:
                            # Spectral data without altitude dimension
                            unique_wavelengths = sorted(set(float(wl) for wl in wavelength_values)) if wavelength_values else []
                            self._add_spectral_variable(ds, col_name, values, unique_wavelengths, time_var, 'wavelength')
                        elif has_multi_level and not has_spectral_data:
                            # Altitude data without spectral dimension
                            self._add_altitude_variable(ds, col_name, values, unique_altitudes, time_var, alt_var)
                        elif has_spectral_data and has_multi_level:
                            # Both spectral and altitude dimensions
                            unique_wavelengths = sorted(set(float(wl) for wl in wavelength_values)) if wavelength_values else []
                            self._add_spectral_altitude_variable(ds, col_name, values, unique_wavelengths, unique_altitudes, time_var, 'wavelength', alt_var)
                        else:
                            # Dictionary format but not recognized structure
                            logger.warning(f"Unrecognized dictionary structure for variable {col_name}, skipping")
                    else:
                        # Regular 1D array - no altitude or wavelength dimension
                        self._add_simple_variable(ds, col_name, values, time_var)
                
                # Add metadata
                ds.attrs['generated_by'] = 'pyradtran'
                if has_spectral_data and wavelength_values and len(wavelength_values) > 0:
                    ds.attrs['wavelength_range'] = f"{min(wavelength_values)} to {max(wavelength_values)} nm"
                if has_multi_level:
                    ds.attrs['altitude_levels'] = f"{len(unique_altitudes)} levels (km): {', '.join(str(alt) for alt in unique_altitudes)}"
                
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
    
    def _add_simple_variable(self, ds, col_name, values, time_var):
        """Add a simple variable with only time dimension"""
        try:
            # Convert to numpy array if needed
            if not isinstance(values, np.ndarray):
                values = np.array(values, dtype=float)
            
            # Ensure 1D shape if needed
            if values.ndim > 1:
                values = values.flatten()
                logger.debug(f"Flattened array for {col_name} from shape {values.shape}")
                
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
        except Exception as e:
            logger.error(f"Error adding simple variable {col_name}: {e}")
    
    def _add_spectral_variable(self, ds, col_name, values, wavelengths, time_var, wl_var):
        """Add a variable with spectral (wavelength) dimension"""
        try:
            # Create a 2D array [time, wavelength]
            time_len = len(ds[time_var])
            wl_len = len(wavelengths)
            
            # Create a data array to hold values for all times and all wavelengths
            data_array = np.full((time_len, wl_len), np.nan)
            
            # Fill in the data - For spectral data, values is a dict with wavelength keys
            for t_idx in range(time_len):
                for wl_idx, wl in enumerate(wavelengths):
                    wl_key = float(wl)
                    # Try to get the value from dict using both float and string keys
                    val = None
                    if wl_key in values:
                        val = values[wl_key]
                    elif str(wl_key) in values:
                        val = values[str(wl_key)]
                    
                    if val is not None:
                        # Get the value - might be list or scalar
                        if isinstance(val, (list, tuple, np.ndarray)):
                            # If val is a sequence, check if it has enough elements
                            if len(val) > t_idx:
                                try:
                                    # Get scalar value for this specific time and wavelength
                                    data_array[t_idx, wl_idx] = float(val[t_idx])
                                except (ValueError, TypeError):
                                    logger.warning(f"Could not convert value to float for {col_name} at time {t_idx}, wavelength {wl}")
                        else:
                            # Same value for all times (scalar)
                            try:
                                data_array[t_idx, wl_idx] = float(val)
                            except (ValueError, TypeError):
                                logger.warning(f"Could not convert scalar value to float for {col_name} at wavelength {wl}")
            
            # Create the data array with proper dimensions
            ds[col_name] = xr.DataArray(
                data_array,
                dims=(time_var, wl_var),
                coords={
                    time_var: ds[time_var],
                    wl_var: wavelengths
                },
                attrs={'units': self._get_variable_units(col_name)}
            )
            logger.debug(f"Added spectral variable {col_name} with shape {data_array.shape}")
        except Exception as e:
            logger.error(f"Error adding spectral variable {col_name}: {e}")
    
    def _add_altitude_variable(self, ds, col_name, values, altitudes, time_var, alt_var):
        """Add a variable with altitude dimension"""
        try:
            # Create a 2D array [time, altitude]
            time_len = len(ds[time_var])
            alt_len = len(altitudes)
            
            # Create a data array to hold values for all times and all altitudes
            data_array = np.full((time_len, alt_len), np.nan)
            
            # Fill in the data - For multi-altitude data, values is a dict with altitude keys
            for t_idx in range(time_len):
                for alt_idx, alt in enumerate(altitudes):
                    alt_key = float(alt)
                    # Try to get the value from dict using both float and string keys
                    val = None
                    if alt_key in values:
                        val = values[alt_key]
                    elif str(alt_key) in values:
                        val = values[str(alt_key)]
                    
                    if val is not None:
                        # Get the value - might be list or scalar
                        if isinstance(val, (list, tuple, np.ndarray)):
                            # If val is a sequence, check if it has enough elements
                            if len(val) > t_idx:
                                try:
                                    # Get scalar value for this specific time and altitude
                                    data_array[t_idx, alt_idx] = float(val[t_idx])
                                except (ValueError, TypeError):
                                    logger.warning(f"Could not convert value to float for {col_name} at time {t_idx}, altitude {alt}")
                        else:
                            # Same value for all times (scalar)
                            try:
                                data_array[t_idx, alt_idx] = float(val)
                            except (ValueError, TypeError):
                                logger.warning(f"Could not convert scalar value to float for {col_name} at altitude {alt}")
            
            # Create the data array with proper dimensions
            ds[col_name] = xr.DataArray(
                data_array,
                dims=(time_var, alt_var),
                coords={
                    time_var: ds[time_var],
                    alt_var: altitudes
                },
                attrs={'units': self._get_variable_units(col_name)}
            )
            logger.debug(f"Added altitude variable {col_name} with shape {data_array.shape}")
        except Exception as e:
            logger.error(f"Error adding altitude variable {col_name}: {e}")
    
    def _add_spectral_altitude_variable(self, ds, col_name, values, wavelengths, altitudes, time_var, wl_var, alt_var):
        """Add a variable with both wavelength and altitude dimensions"""
        try:
            # Create a 3D array [time, altitude, wavelength]
            time_len = len(ds[time_var])
            alt_len = len(altitudes)
            wl_len = len(wavelengths)
            
            # Create a data array to hold values for all times, altitudes, and wavelengths
            data_array = np.full((time_len, alt_len, wl_len), np.nan)
            
            # Fill in the data - For spectral multi-altitude data, values is a nested dict: altitude -> wavelength -> value
            for t_idx in range(time_len):
                for alt_idx, alt in enumerate(altitudes):
                    alt_key = float(alt)
                    # Get the wavelength dict for this altitude
                    wl_dict = None
                    if alt_key in values:
                        wl_dict = values[alt_key]
                    elif str(alt_key) in values:
                        wl_dict = values[str(alt_key)]
                    
                    if wl_dict is not None and isinstance(wl_dict, dict):
                        # Process each wavelength in this altitude's dict
                        for wl_idx, wl in enumerate(wavelengths):
                            wl_key = float(wl)
                            # Try to get the value from dict using both float and string keys
                            val = None
                            if wl_key in wl_dict:
                                val = wl_dict[wl_key]
                            elif str(wl_key) in wl_dict:
                                val = wl_dict[str(wl_key)]
                            
                            if val is not None:
                                # Get the value - might be list or scalar
                                if isinstance(val, (list, tuple, np.ndarray)):
                                    # If val is a sequence, check if it has enough elements
                                    if len(val) > t_idx:
                                        try:
                                            # Get scalar value for this specific time, altitude, and wavelength
                                            data_array[t_idx, alt_idx, wl_idx] = float(val[t_idx])
                                        except (ValueError, TypeError):
                                            logger.warning(f"Could not convert value to float for {col_name} at time {t_idx}, altitude {alt}, wavelength {wl}")
                                else:
                                    # Same value for all times (scalar)
                                    try:
                                        data_array[t_idx, alt_idx, wl_idx] = float(val)
                                    except (ValueError, TypeError):
                                        logger.warning(f"Could not convert scalar value to float for {col_name} at altitude {alt}, wavelength {wl}")
            
            # Create the data array with proper dimensions
            ds[col_name] = xr.DataArray(
                data_array,
                dims=(time_var, alt_var, wl_var),
                coords={
                    time_var: ds[time_var],
                    alt_var: altitudes,
                    wl_var: wavelengths
                },
                attrs={'units': self._get_variable_units(col_name)}
            )
            logger.debug(f"Added spectral-altitude variable {col_name} with shape {data_array.shape}")
        except Exception as e:
            logger.error(f"Error adding spectral-altitude variable {col_name}: {e}")


# --- Thermal Simulation Helper Functions ---

def create_thermal_simulation_config(
    surface_temperature_k: float = 248.4,
    wavelength_range_nm: List[float] = None,
    mol_abs_param: str = "reptran medium",
    rte_solver: str = "disort", 
    output_altitudes_km: List[float] = None,
    output_columns: List[str] = None,
    config_template_path: Optional[Union[str, Path]] = None,
    **kwargs
) -> SimulationConfig:
    """
    Create a SimulationConfig optimized for thermal infrared simulations.
    
    This helper function sets up default parameters commonly used for thermal
    simulations and allows easy customization of key parameters.
    
    Args:
        surface_temperature_k: Surface temperature in Kelvin (default: 248.4 K)
        wavelength_range_nm: [min, max] wavelength in nm (default: [4500, 42000])
        mol_abs_param: Molecular absorption parameterization (default: "reptran medium")
        rte_solver: Radiative transfer solver (default: "disort")
        output_altitudes_km: List of output altitudes in km
        output_columns: List of output quantities 
        config_template_path: Path to base configuration file to modify
        **kwargs: Additional configuration parameters to override
        
    Returns:
        SimulationConfig: Configuration object for thermal simulations
        
    Example:
        >>> config = create_thermal_simulation_config(
        ...     surface_temperature_k=260.0,
        ...     wavelength_range_nm=[5000, 25000],
        ...     output_altitudes_km=[0, 1, 2, 3, 5, 10]
        ... )
        >>> pyradtran = PyRadtran(config)
    """
    
    # Default thermal simulation parameters
    if wavelength_range_nm is None:
        wavelength_range_nm = [4500, 42000]  # 4.5 to 42 microns
        
    if output_altitudes_km is None:
        output_altitudes_km = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 15.0]
        
    if output_columns is None:
        output_columns = ["zout", "edir", "edn", "eup"]
    
    # Load base config if provided, otherwise use defaults
    if config_template_path:
        config = SimulationConfig.from_yaml(config_template_path)
    else:
        # Create minimal config - user must provide paths
        config_dict = {
            "paths": {
                "libradtran_bin": "/opt/libradtran/current/bin/uvspec",
                "libradtran_data": "/opt/libradtran/current/share/libRadtran/data",
                "atmosphere_profile": "/opt/libradtran/current/share/libRadtran/data/atmmod/afglsw.dat",
                "solar_spectrum": None,  # Not needed for thermal
                "output_dir": "./thermal_output",
                "working_dir": "./thermal_work"
            },
            "simulation_defaults": {},
            "execution": {"max_workers": 4, "timeout_seconds": 120},
            "output": {"filename_prefix": "thermal", "filename_suffix": ".nc"}
        }
        config = SimulationConfig.from_dict(config_dict)
    
    # Override with thermal-specific settings
    config.simulation_defaults.source = "thermal"
    config.simulation_defaults.surface_temperature_k = surface_temperature_k
    config.simulation_defaults.wavelength_nm = wavelength_range_nm
    config.simulation_defaults.mol_abs_param = mol_abs_param
    config.simulation_defaults.rte_solver = rte_solver
    config.simulation_defaults.output_altitudes_km = output_altitudes_km
    config.simulation_defaults.output_columns = output_columns
    config.simulation_defaults.integrate_wavelength = True  # Usually want integrated
    
    # Apply any additional overrides
    for key, value in kwargs.items():
        if hasattr(config.simulation_defaults, key):
            setattr(config.simulation_defaults, key, value)
        else:
            logger.warning(f"Unknown configuration parameter: {key}")
    
    return config


def run_thermal_simulation(
    dt: datetime,
    latitude: float = 0.0,
    longitude: float = 0.0,
    surface_temperature_k: float = 248.4,
    wavelength_range_nm: List[float] = None,
    radiosonde_path: Optional[Union[str, Path]] = None,
    config_path: Optional[Union[str, Path]] = None,
    **kwargs
) -> xr.Dataset:
    """
    Run a single thermal infrared simulation with simplified parameters.
    
    This is a convenience function that handles the common case of running
    a single thermal simulation with minimal setup.
    
    Args:
        dt: Date and time (can be arbitrary for thermal sims)
        latitude: Latitude in degrees (can be arbitrary for thermal sims)  
        longitude: Longitude in degrees (can be arbitrary for thermal sims)
        surface_temperature_k: Surface temperature in Kelvin
        wavelength_range_nm: [min, max] wavelength range in nm
        radiosonde_path: Optional path to radiosonde file
        config_path: Optional path to base configuration file
        **kwargs: Additional configuration parameters
        
    Returns:
        xr.Dataset: Simulation results
        
    Example:
        >>> from datetime import datetime
        >>> result = run_thermal_simulation(
        ...     dt=datetime(2022, 4, 1),
        ...     surface_temperature_k=250.0,
        ...     wavelength_range_nm=[8000, 12000]
        ... )
    """
    
    # Create thermal configuration
    config = create_thermal_simulation_config(
        surface_temperature_k=surface_temperature_k,
        wavelength_range_nm=wavelength_range_nm,
        config_template_path=config_path,
        **kwargs
    )
    
    # Create PyRadtran instance and run simulation
    from .interface import PyRadtran  # Import here to avoid circular import
    pyradtran = PyRadtran(config)
    
    result = pyradtran.run_single_simulation(
        dt=dt,
        latitude=latitude,
        longitude=longitude,
        radiosonde_path=radiosonde_path
    )
    
    return result