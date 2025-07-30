# pyradtran/interface_unified.py
"""
Unified high-level interface for pyradtran.

This module combines the best features from both interface.py and interface_old.py:
- PyRadtranAccessor: xarray accessor registered as ds.pyradtran
- execute_simulation_batch: Parallel execution of multiple uvspec runs
- run_pyradtran_simulation: Standalone function for running simulations from a file
- Full ERA5 atmosphere support
"""

import logging
import xarray as xr
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime

from .config_clean import SimulationConfig, load_config
from .exceptions import PyRadtranError
from .core_unified import Simulation
from .io_unified import (
    InputDataLoader, ERA5AtmosphereGenerator, OutputParser, 
    OutputToXarray, NetCDFSaver, ParsedOutput
)

logger = logging.getLogger(__name__)


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
            _apply_parameter_overrides(config, parameter_overrides)
        
        # Load input data
        loader = InputDataLoader()
        input_ds = loader.load_simulation_input_data(input_file)
        
        # Generate output path if not provided
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = Path(config.paths.output_dir) / f"{config.output.filename_prefix}_{timestamp}{config.output.filename_suffix}"
        else:
            output_path = Path(output_path)
        
        # Run the simulation batch
        parsed_outputs = execute_simulation_batch(
            config=config,
            input_ds=input_ds,
            parameter_overrides=parameter_overrides
        )
        
        # Convert to xarray and save results
        if parsed_outputs:
            converter = OutputToXarray()
            result_ds = converter.convert_batch(parsed_outputs, input_ds)
            
            saver = NetCDFSaver()
            return saver.save_results_to_netcdf(
                data=result_ds,
                output_path=output_path,
                input_ds=input_ds,
                config=config,
                simulation_params=parameter_overrides
            )
        else:
            raise PyRadtranError("No valid simulation results produced")
            
    except Exception as e:
        logger.error(f"Simulation failed: {str(e)}")
        raise PyRadtranError(f"Simulation failed: {str(e)}")


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
    parameter_overrides: Dict[str, Any] = None,
    progress_callback: Optional[callable] = None
) -> List[ParsedOutput]:
    """
    Execute a batch of simulations in parallel.
    
    Args:
        config: Simulation configuration
        input_ds: Input dataset with time, lat, lon coordinates
        time_var: Name of time dimension/coordinate
        lat_var: Name of latitude dimension/coordinate  
        lon_var: Name of longitude dimension/coordinate
        albedo_var: Optional name of albedo data_var
        surface_temperature_var: Optional name of surface temperature data_var
        altitude_var: Optional name of altitude data_var
        era5_atmosphere: Optional ERA5 dataset for custom atmosphere profiles
        parameter_overrides: Dictionary of simulation parameters to override
        progress_callback: Optional callback function(current, total) for progress updates
        
    Returns:
        List of ParsedOutput objects
        
    Raises:
        PyRadtranError: If all simulations fail
    """
    # Initialize simulation runner
    runner = Simulation(config)
    
    # Extract coordinates
    times = input_ds[time_var].values
    latitudes = input_ds[lat_var].values
    longitudes = input_ds[lon_var].values
    
    # Extract optional data variables
    albedos = input_ds[albedo_var].values if albedo_var and albedo_var in input_ds else None
    surface_temperatures = input_ds[surface_temperature_var].values if surface_temperature_var and surface_temperature_var in input_ds else None
    altitudes = input_ds[altitude_var].values if altitude_var and altitude_var in input_ds else None
    
    # Handle ERA5 atmosphere files if provided
    era5_atmosphere_files = {}
    if era5_atmosphere is not None:
        logger.info("Creating ERA5 atmosphere files for simulation points...")
        # Create working directory for atmosphere files
        atm_dir = config.paths.working_dir / "era5_atmospheres"
        atm_dir.mkdir(parents=True, exist_ok=True)
        
        era5_generator = ERA5AtmosphereGenerator()
        
        for i, (t, lat, lon) in enumerate(zip(times, latitudes, longitudes)):
            try:
                point_id = f"t{i:05d}_lat{lat:.3f}_lon{lon:.3f}"
                atm_file = atm_dir / f"era5_atm_{point_id}.dat"
                
                era5_generator.create_era5_atmosphere_file(
                    era5_atmosphere, lat, lon, t, atm_file
                )
                era5_atmosphere_files[point_id] = atm_file
                logger.debug(f"Created ERA5 atmosphere file for {point_id}: {atm_file}")
            except Exception as e:
                logger.error(f"Failed to create ERA5 atmosphere file for point {i}: {e}")
                era5_atmosphere_files[point_id] = None
    
    # Prepare simulation points
    points = []
    for i, (t, lat, lon) in enumerate(zip(times, latitudes, longitudes)):
        alb = albedos[i] if albedos is not None else None
        surf_temp = surface_temperatures[i] if surface_temperatures is not None else None
        alt = altitudes[i] if altitudes is not None else None
        
        point_id = f"t{i:05d}_lat{lat:.3f}_lon{lon:.3f}"
        era5_atm_file = era5_atmosphere_files.get(point_id) if era5_atmosphere_files else None
        
        points.append((t, lat, lon, alb, surf_temp, alt, era5_atm_file, point_id))
    
    # Run simulations in parallel
    results = []
    total_points = len(points)
    
    with ProcessPoolExecutor(max_workers=config.execution.max_workers) as executor:
        # Submit all simulations
        future_to_point = {}
        for i, point in enumerate(points):
            future = executor.submit(
                _run_single_simulation_unified,
                config, point, parameter_overrides
            )
            future_to_point[future] = (i, point)
        
        # Collect results
        for future in as_completed(future_to_point):
            point_idx, point_data = future_to_point[future]
            
            try:
                result = future.result()
                if result:
                    results.append(result)
                    logger.debug(f"Simulation {point_idx + 1}/{total_points} completed successfully")
                else:
                    logger.warning(f"Simulation {point_idx + 1}/{total_points} produced no output")
            except Exception as e:
                logger.error(f"Simulation {point_idx + 1}/{total_points} failed: {str(e)}")
            
            # Progress callback
            if progress_callback:
                progress_callback(len(results), total_points)
    
    if not results:
        raise PyRadtranError("All simulations failed - no valid results produced")
    
    logger.info(f"Batch execution completed: {len(results)}/{total_points} simulations successful")
    return results


def _run_single_simulation_unified(
    config: SimulationConfig,
    point_data: Tuple,
    parameter_overrides: Dict[str, Any] = None
) -> Optional[ParsedOutput]:
    """
    Run a single simulation (used by execute_simulation_batch).
    
    Args:
        config: Simulation configuration
        point_data: Tuple of (time, lat, lon, albedo, surf_temp, altitude, era5_file, point_id)
        parameter_overrides: Dictionary of simulation parameters to override
        
    Returns:
        ParsedOutput object or None if simulation failed
    """
    try:
        time, lat, lon, albedo, surf_temp, altitude, era5_file, point_id = point_data
        
        # Initialize simulation
        sim = Simulation(config)
        
        # Convert datetime to datetime object if needed
        if isinstance(time, (np.datetime64, str)):
            if isinstance(time, np.datetime64):
                dt = time.astype(datetime)
            else:
                dt = datetime.fromisoformat(time)
        else:
            dt = time
        
        # Run simulation with parameters
        output_file = sim.run_simulation(
            dt=dt,
            latitude=lat,
            longitude=lon,
            override_albedo=albedo,
            override_surface_temperature=surf_temp,
            override_altitude_km=altitude,
            era5_atmosphere_file=era5_file,
            parameter_overrides=parameter_overrides
        )
        
        if output_file and output_file.exists():
            # Parse the output
            parser = OutputParser(config, parameter_overrides)
            parsed_output = parser.parse_output_file(output_file)
            
            # Add point metadata
            parsed_output.metadata.update({
                'point_id': point_id,
                'time': dt.isoformat(),
                'latitude': lat,
                'longitude': lon,
                'albedo': albedo,
                'surface_temperature': surf_temp,
                'altitude': altitude
            })
            
            return parsed_output
        else:
            logger.error(f"No output file produced for point {point_id}")
            return None
            
    except Exception as e:
        logger.error(f"Single simulation failed for point {point_data[-1] if len(point_data) > 7 else 'unknown'}: {str(e)}")
        return None


def _apply_parameter_overrides(config: SimulationConfig, parameter_overrides: Dict[str, Any]):
    """Apply parameter overrides to configuration."""
    for key, value in parameter_overrides.items():
        parts = key.split('.')
        if len(parts) == 2:
            section, param = parts
            if hasattr(config, section) and hasattr(getattr(config, section), param):
                setattr(getattr(config, section), param, value)
                logger.info(f"Overriding config: {section}.{param} = {value}")
        else:
            logger.warning(f"Invalid parameter override format: {key}")


@xr.register_dataset_accessor("pyradtran")
class PyRadtranAccessor:
    """
    xarray accessor for pyradtran functionality.
    
    This accessor provides a convenient interface for running LibRadtran simulations
    directly from xarray datasets containing time, latitude, and longitude information.
    """
    
    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        self._config = None
    
    def run(
        self,
        config_path: Optional[Union[str, Path]] = None,
        parameter_overrides: Dict[str, Any] = None,
        time_var: str = 'time',
        lat_var: str = 'latitude',
        lon_var: str = 'longitude',
        albedo_var: Optional[str] = None,
        surface_temperature_var: Optional[str] = None,
        era5_atmosphere: Optional[xr.Dataset] = None,
        return_dataset: bool = True,
        save_to_file: bool = True,
        output_path: Optional[Union[str, Path]] = None,
        progress_callback: Optional[callable] = None
    ) -> Union[xr.Dataset, Path]:
        """
        Run pyradtran simulations for all points in the dataset.
        
        Args:
            config_path: Path to YAML configuration file (uses default if None)
            parameter_overrides: Dictionary of simulation parameters to override
            time_var: Name of time dimension/coordinate in the dataset
            lat_var: Name of latitude dimension/coordinate in the dataset
            lon_var: Name of longitude dimension/coordinate in the dataset
            albedo_var: Optional name of albedo data_var in the dataset
            surface_temperature_var: Optional name of surface temperature data_var in the dataset
            era5_atmosphere: Optional ERA5 xarray Dataset for custom atmosphere profiles
            return_dataset: If True, return the results as an xarray Dataset
            save_to_file: If True, save results to a NetCDF file
            output_path: Path for output file (auto-generated if None)
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
            _apply_parameter_overrides(self._config, parameter_overrides)
        
        # Validate input dataset
        self._validate_input_dataset(time_var, lat_var, lon_var, albedo_var, 
                                   surface_temperature_var, era5_atmosphere)
        
        # Handle altitude information
        alt_var = 'altitude'
        altitude_as_data_var = False
        
        if alt_var in self._obj.dims or alt_var in self._obj.coords:
            # Altitude is a coordinate - use as list of zout levels
            dataset_altitudes = self._obj[alt_var].values
            if len(dataset_altitudes) > 0:
                logger.info(f"Altitude found as coordinate - using {len(dataset_altitudes)} levels for zout: {dataset_altitudes}")
                self._config.simulation_defaults.output_altitudes_km = [float(alt) for alt in dataset_altitudes]
        elif alt_var in self._obj.data_vars:
            # Altitude is a data variable - treat as scalar per time step
            altitude_as_data_var = True
            logger.info(f"Altitude found as data variable - will be treated as scalar altitude for each time step")
        
        # Generate output path if saving and not provided
        if save_to_file and output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = Path(self._config.paths.output_dir) / f"{self._config.output.filename_prefix}_{timestamp}{self._config.output.filename_suffix}"
            output_path.parent.mkdir(exist_ok=True, parents=True)
            logger.info(f"Auto-generating output path: {output_path}")
        elif output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(exist_ok=True, parents=True)
        
        # Run the simulation batch
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
            parameter_overrides=parameter_overrides,
            progress_callback=progress_callback
        )
        
        # Convert to xarray Dataset
        if return_dataset and parsed_outputs:
            converter = OutputToXarray()
            result_ds = converter.convert_batch(parsed_outputs, self._obj, time_var, lat_var, lon_var)
            
            # Add metadata
            result_ds.attrs['generated_by'] = 'pyradtran'
            result_ds.attrs['pyradtran_version'] = 'unified_system'
            result_ds.attrs['generation_date'] = datetime.now().isoformat()
            
            # Save to file if requested
            if save_to_file and output_path:
                saver = NetCDFSaver()
                saver.save_results_to_netcdf(
                    data=result_ds,
                    output_path=output_path,
                    input_ds=self._obj,
                    config=self._config,
                    simulation_params=parameter_overrides
                )
                logger.info(f"Results saved to {output_path}")
            
            return result_ds
        
        elif save_to_file and parsed_outputs and output_path:
            # Just save to file without returning dataset
            converter = OutputToXarray()
            result_ds = converter.convert_batch(parsed_outputs, self._obj, time_var, lat_var, lon_var)
            
            saver = NetCDFSaver()
            return saver.save_results_to_netcdf(
                data=result_ds,
                output_path=output_path,
                input_ds=self._obj,
                config=self._config,
                simulation_params=parameter_overrides
            )
        else:
            raise PyRadtranError("No valid simulation results to return or save")
    
    def _validate_input_dataset(self, time_var: str, lat_var: str, lon_var: str, 
                              albedo_var: Optional[str], surface_temperature_var: Optional[str],
                              era5_atmosphere: Optional[xr.Dataset]):
        """Validate input dataset variables and coordinates."""
        # Check required variables
        if time_var not in self._obj.dims and time_var not in self._obj.coords:
            raise PyRadtranError(f"Time variable '{time_var}' not found in dataset")
        
        if lat_var not in self._obj.dims and lat_var not in self._obj.coords and lat_var not in self._obj.data_vars:
            raise PyRadtranError(f"Latitude variable '{lat_var}' not found in dataset")
            
        if lon_var not in self._obj.dims and lon_var not in self._obj.coords and lon_var not in self._obj.data_vars:
            raise PyRadtranError(f"Longitude variable '{lon_var}' not found in dataset")
        
        # Check optional variables
        if albedo_var and albedo_var not in self._obj.data_vars:
            raise PyRadtranError(f"Albedo variable '{albedo_var}' not found in dataset data_vars")
        
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


# Expose main functions
__all__ = [
    'run_pyradtran_simulation', 'execute_simulation_batch', 'PyRadtranAccessor'
]
