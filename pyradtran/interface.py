# pyradtran/interface_new.py
"""
High-level interface for pyradtran using the new IO system:
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
from .io_new import OutputParser, ParsedOutput, OutputToXarray
from .io import (
    load_simulation_input_data, 
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
) -> List[ParsedOutput]:
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
        List of ParsedOutput objects from the new IO system
        
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
    
    # Prepare for parallel execution
    total_points = len(points)
    completed = 0
    logger.info(f"Running {total_points} simulations across time/location points...")
    
    # Store parsed outputs using new IO system
    parsed_outputs: List[ParsedOutput] = []
    
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
                    _run_single_simulation_new_io, runner, t, lat, lon, alb, surf_temp, alt, era5_atm_file
                )
            ] = (t, lat, lon, alb, surf_temp, alt, point_id)
        
        # Collect results
        for future in as_completed(future_to_point):
            t, lat, lon, alb, surf_temp, alt, point_id = future_to_point[future]
            completed += 1
            
            try:
                parsed_output = future.result()
                if parsed_output is not None:
                    # Add metadata about the simulation point
                    parsed_output.metadata['time'] = t
                    parsed_output.metadata['latitude'] = lat
                    parsed_output.metadata['longitude'] = lon
                    if alb is not None:
                        parsed_output.metadata['albedo'] = alb
                    if surf_temp is not None:
                        parsed_output.metadata['surface_temperature'] = surf_temp
                    if alt is not None:
                        parsed_output.metadata['altitude'] = alt
                    
                    parsed_outputs.append(parsed_output)
                    logger.debug(f"Successfully processed simulation {completed}/{total_points} for time {t}")
                else:
                    logger.warning(f"Simulation {completed}/{total_points} returned no output")
                    
            except Exception as e:
                logger.error(f"Error in simulation {completed}/{total_points}: {e}")
            
            # Call progress callback if provided
            if progress_callback:
                progress_callback(completed, total_points)
    
    if not parsed_outputs:
        raise PyRadtranError("All simulations failed - no valid outputs")
    
    logger.info(f"Successfully completed {len(parsed_outputs)}/{total_points} simulations")
    return parsed_outputs


def _run_single_simulation_new_io(
    runner: Simulation,
    time: np.datetime64,
    latitude: float,
    longitude: float,
    albedo: Optional[float] = None,
    surface_temperature: Optional[float] = None,
    altitude: Optional[float] = None,
    era5_atmosphere_file: Optional[Path] = None
) -> Optional[ParsedOutput]:
    """
    Run a single uvspec simulation and return parsed output using new IO system.
    
    Args:
        runner: Simulation object
        time: Timestamp for the simulation
        latitude: Latitude in degrees
        longitude: Longitude in degrees  
        albedo: Optional surface albedo override
        surface_temperature: Optional surface temperature override
        altitude: Optional altitude override (scalar)
        era5_atmosphere_file: Optional path to ERA5 atmosphere file
        
    Returns:
        ParsedOutput object from new IO system, or None if simulation failed
    """
    try:
        # Convert numpy datetime64 to pandas timestamp for better handling
        timestamp = pd.to_datetime(time)
        
        # Run the simulation
        output_file = runner.run(
            dt=timestamp,
            latitude=latitude,
            longitude=longitude,
            override_albedo=albedo,
            override_surface_temperature=surface_temperature,
            override_altitude_km=altitude,
            era5_atmosphere_file=era5_atmosphere_file
        )
        
        if output_file and output_file.exists():
            # Parse using the new IO system - need to pass the config
            parser = OutputParser(runner.config)
            parsed_output = parser.parse(output_file)
            return parsed_output
        else:
            logger.warning(f"No output file generated for time={timestamp}, lat={latitude}, lon={longitude}")
            return None
            
    except Exception as e:
        logger.error(f"Error in simulation for time={time}, lat={latitude}, lon={longitude}: {e}")
        return None


# --- xarray accessor ---

@xr.register_dataset_accessor("pyradtran")
class PyRadtranAccessor:
    """
    xarray accessor for pyradtran functionality using the new IO system.
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
        Run uvspec for each time/location coordinate in the dataset using the new IO system.
        
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
        
        # Run the simulation batch using new IO system
        parsed_outputs = execute_simulation_batch(
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
        
        # Convert to xarray Dataset using new IO system
        if return_dataset and parsed_outputs:
            # Use the new OutputToXarray converter for batch processing
            converter = OutputToXarray()
            
            # Convert parsed outputs to xarray Dataset
            result_ds = converter.convert_batch(parsed_outputs, self._obj, time_var, lat_var, lon_var)
            
            # Add metadata
            result_ds.attrs['generated_by'] = 'pyradtran'
            result_ds.attrs['pyradtran_version'] = 'new_io_system'
            
            # Save to file if requested
            if save_to_file:
                result_ds.to_netcdf(output_path)
                logger.info(f"Results saved to {output_path}")
            
            return result_ds
        
        elif save_to_file and parsed_outputs:
            # Just save to file without returning dataset
            converter = OutputToXarray()
            result_ds = converter.convert_batch(parsed_outputs, self._obj, time_var, lat_var, lon_var)
            result_ds.to_netcdf(output_path)
            logger.info(f"Results saved to {output_path}")
            return output_path
        
        else:
            raise PyRadtranError("Simulation produced no valid results")
