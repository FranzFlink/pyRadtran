# pyradtran/io.py
"""
Input/output functionality for pyradtran:
- Loading simulation data
- Generating uvspec input files
- Parsing uvspec output
- Saving results to NetCDF
"""

import logging
import os
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

from .config import SimulationConfig
from .exceptions import InputGenerationError, OutputParsingError

logger = logging.getLogger(__name__)

# --- Data Loading ---

def load_simulation_input_data(input_file: Union[str, Path]) -> xr.Dataset:
    """
    Load simulation input data from various file formats.
    
    Args:
        input_file: Path to input file (CSV or NetCDF)
        
    Returns:
        xarray.Dataset containing the simulation input data
        
    Raises:
        ValueError: If the file format is not supported or file doesn't exist
    """
    input_path = Path(input_file)
    if not input_path.exists():
        raise ValueError(f"Input file does not exist: {input_path}")
    
    # Determine file type from extension
    if input_path.suffix.lower() == '.nc':
        logger.info(f"Loading NetCDF input file: {input_path}")
        ds = xr.open_dataset(input_path)
    elif input_path.suffix.lower() == '.csv':
        logger.info(f"Loading CSV input file: {input_path}")
        df = pd.read_csv(input_path)
        
        # Detect and convert datetime columns
        for col in df.columns:
            if 'time' in col.lower() or 'date' in col.lower():
                try:
                    df[col] = pd.to_datetime(df[col])
                except:
                    logger.warning(f"Column '{col}' looks like a datetime but couldn't be converted")
        
        # Convert to xarray dataset
        ds = df.to_xarray()
    else:
        raise ValueError(f"Unsupported input file format: {input_path.suffix}")
    
    # Validate required coordinates/variables
    required_vars = ['time', 'latitude', 'longitude']
    for var in required_vars:
        alt_names = [var, var[0].upper() + var[1:], var.upper(), var.lower()]
        if not any(name in ds for name in alt_names):
            raise ValueError(f"Input data missing required variable: {var} (or alternate name)")
    
    # Rename variables if needed for consistency
    rename_dict = {}
    for std_name, alternates in [
        ('time', ['Time', 'TIME', 'datetime', 'Datetime', 'DATETIME', 'date', 'Date', 'DATE']),
        ('latitude', ['Latitude', 'LATITUDE', 'lat', 'Lat', 'LAT']),
        ('longitude', ['Longitude', 'LONGITUDE', 'lon', 'Lon', 'LON'])
    ]:
        for alt in alternates:
            if alt in ds and std_name not in ds:
                rename_dict[alt] = std_name
    
    if rename_dict:
        ds = ds.rename(rename_dict)
        
    return ds

# --- Input File Generation ---

def generate_uvspec_input_content(
    config: SimulationConfig,
    dt: datetime,
    latitude: float,
    longitude: float,
    radiosonde_path: Optional[Path] = None,
    **kwargs
) -> str:
    """
    Generate uvspec input file content for a specific simulation.
    
    Args:
        config: Simulation configuration
        dt: Datetime for the simulation
        latitude: Latitude for the simulation
        longitude: Longitude for the simulation
        radiosonde_path: Optional path to radiosonde file
        **kwargs: Additional parameters to override defaults
        
    Returns:
        String content for uvspec input file
        
    Raises:
        InputGenerationError: If input generation fails
    """
    try:
        # Use simulation_defaults from config, but allow kwargs to override
        sim_defaults = config.simulation_defaults
        
        # Start building input content
        lines = []
        
        # --- Core simulation parameters ---
        lines.append(f"rte_solver {sim_defaults.rte_solver}")
        lines.append(f"mol_abs_param {sim_defaults.mol_abs_param}")
        
        # --- Data files and directories ---
        # Default atmosphere file from config
        if radiosonde_path:
            # If we have a radiosonde, use it for water vapor
            lines.append(f"atmosphere_file {config.paths.atmosphere_profile} H2O RH")
            lines.append(f"radiosonde {radiosonde_path} H2O RH")
        else:
            # Otherwise just use the atmosphere file
            lines.append(f"atmosphere_file {config.paths.atmosphere_profile}")
            
        # Data files path must be specified
        lines.append(f"data_files_path {config.paths.libradtran_data}")
        
        # --- Molecule modifications ---
        # Handle ozone
        if 'O3' in sim_defaults.mol_modify:
            o3_params = sim_defaults.mol_modify['O3']
            lines.append(f"mol_modify O3 {o3_params['value']} {o3_params['unit']}")
        elif sim_defaults.ozone_du is not None:
            lines.append(f"mol_modify O3 {sim_defaults.ozone_du} DU")
        
        # Handle water vapor
        if 'H2O' in sim_defaults.mol_modify:
            h2o_params = sim_defaults.mol_modify['H2O']
            lines.append(f"mol_modify H2O {h2o_params['value']} {h2o_params['unit']}")
        elif sim_defaults.h2o_mm is not None:
            lines.append(f"mol_modify H2O {sim_defaults.h2o_mm} MM")
        
        # --- Solar spectrum ---
        lines.append(f"source solar {config.paths.solar_spectrum}")
        
        # --- Date/time and location ---
        # Format date: year month day hour min sec
        lines.append(f"time {dt.year} {dt.month} {dt.day} {dt.hour} {dt.minute} {dt.second}")
        
        # Format latitude: N/S degrees
        lat_dir = "N" if latitude >= 0 else "S"
        lat_abs = abs(latitude)
        lines.append(f"latitude {lat_dir} {lat_abs}")
        
        # Format longitude: E/W degrees
        lon_dir = "E" if longitude >= 0 else "W"
        lon_abs = abs(longitude)
        lines.append(f"longitude {lon_dir} {lon_abs}")
        
        # --- Surface properties ---
        # Handle albedo based on type
        if sim_defaults.albedo_type == 'const' and sim_defaults.albedo_value is not None:
            lines.append(f"albedo {sim_defaults.albedo_value}")
        elif sim_defaults.albedo_type == 'file' and sim_defaults.albedo_file:
            lines.append(f"albedo_file {sim_defaults.albedo_file}")
        elif sim_defaults.albedo_type == 'library' and sim_defaults.albedo_library:
            lines.append(f"albedo {sim_defaults.albedo_library}")
        
        # Handle BRDF properties
        # Cox and Munk ocean BRDF
        if hasattr(sim_defaults, 'brdf_cam') and sim_defaults.brdf_cam.enabled:
            # Ensure wind speed is at least 1.0 m/s
            u10 = max(1.0, sim_defaults.brdf_cam.u10)
            
            lines.append(f"brdf_cam pcl {sim_defaults.brdf_cam.pcl}")
            lines.append(f"brdf_cam sal {sim_defaults.brdf_cam.sal}")
            lines.append(f"brdf_cam u10 {u10}")
            lines.append(f"brdf_cam uphi {sim_defaults.brdf_cam.uphi}")
            
            if sim_defaults.brdf_cam.solar_wind:
                lines.append("brdf_cam_solar_wind")
        
        # RPV BRDF
        if hasattr(sim_defaults, 'brdf_rpv') and sim_defaults.brdf_rpv.enabled:
            # File-based RPV parameters
            if sim_defaults.brdf_rpv.rpv_file:
                lines.append(f"brdf_rpv_file {sim_defaults.brdf_rpv.rpv_file}")
            
            # Library-based RPV parameters
            elif sim_defaults.brdf_rpv.rpv_library:
                lines.append(f"brdf_rpv_library {sim_defaults.brdf_rpv.rpv_library}")
                if sim_defaults.brdf_rpv.rpv_type is not None:
                    lines.append(f"brdf_rpv_type {sim_defaults.brdf_rpv.rpv_type}")
            
            # Direct RPV parameters
            else:
                # At least one parameter must be specified
                if any(param is not None for param in [
                    sim_defaults.brdf_rpv.k, 
                    sim_defaults.brdf_rpv.rho0, 
                    sim_defaults.brdf_rpv.theta, 
                    sim_defaults.brdf_rpv.sigma, 
                    sim_defaults.brdf_rpv.t1, 
                    sim_defaults.brdf_rpv.t2, 
                    sim_defaults.brdf_rpv.scale
                ]):
                    if sim_defaults.brdf_rpv.k is not None:
                        lines.append(f"brdf_rpv k {sim_defaults.brdf_rpv.k}")
                    if sim_defaults.brdf_rpv.rho0 is not None:
                        lines.append(f"brdf_rpv rho0 {sim_defaults.brdf_rpv.rho0}")
                    if sim_defaults.brdf_rpv.theta is not None:
                        lines.append(f"brdf_rpv theta {sim_defaults.brdf_rpv.theta}")
                    if sim_defaults.brdf_rpv.sigma is not None:
                        lines.append(f"brdf_rpv sigma {sim_defaults.brdf_rpv.sigma}")
                    if sim_defaults.brdf_rpv.t1 is not None:
                        lines.append(f"brdf_rpv t1 {sim_defaults.brdf_rpv.t1}")
                    if sim_defaults.brdf_rpv.t2 is not None:
                        lines.append(f"brdf_rpv t2 {sim_defaults.brdf_rpv.t2}")
                    if sim_defaults.brdf_rpv.scale is not None:
                        lines.append(f"brdf_rpv scale {sim_defaults.brdf_rpv.scale}")
                else:
                    # Fall back to RPV type 
                    if hasattr(sim_defaults, 'brdf_rpv_type') and sim_defaults.brdf_rpv_type is not None:
                        lines.append(f"brdf_rpv_type {sim_defaults.brdf_rpv_type}")
        
        # Legacy BRDF handling (for backward compatibility)
        elif hasattr(sim_defaults, 'brdf_type') and sim_defaults.brdf_type == 'rpv' and hasattr(sim_defaults, 'brdf_rpv_type') and sim_defaults.brdf_rpv_type is not None:
            lines.append(f"brdf_rpv_type {sim_defaults.brdf_rpv_type}")
        
        # Surface temperature
        if sim_defaults.surface_temperature_k is not None:
            lines.append(f"sur_temperature {sim_defaults.surface_temperature_k}")
        
        # --- Cloud properties ---
        if hasattr(sim_defaults, 'clouds') and sim_defaults.clouds.get('enabled', False):
            clouds = sim_defaults.clouds
            # Set cloud optical properties
            if 'cloud_optical_properties' in clouds:
                lines.append(f"cloud_optical_properties {clouds['cloud_optical_properties']}")
            
            if 'cloud_overlap' in clouds:
                lines.append(f"cloud_overlap {clouds['cloud_overlap']}")
            
            if 'cloud_file' in clouds and clouds['cloud_file']:
                # Use cloud properties file
                lines.append(f"cloud_file {clouds['cloud_file']}")
            elif 'layer_heights_km' in clouds and clouds['layer_heights_km']:
                # Add each cloud layer
                for i, ((bottom, top), lwc, r_eff) in enumerate(
                    zip(clouds['layer_heights_km'], clouds['layer_water_content'], clouds['layer_effective_radius_um'])
                ):
                    lines.append(f"wc_file {i+1} {lwc} {r_eff} {bottom} {top}")
        
        # --- Aerosol properties ---
        if hasattr(sim_defaults, 'aerosols') and sim_defaults.aerosols.get('enabled', False):
            aerosols = sim_defaults.aerosols
            # Set aerosol optical properties
            if 'aerosol_optical_properties' in aerosols:
                lines.append(f"aerosol_optical_properties {aerosols['aerosol_optical_properties']}")
            
            if 'aerosol_file' in aerosols and aerosols['aerosol_file']:
                # Use aerosol properties file
                lines.append(f"aerosol_file {aerosols['aerosol_file']}")
            else:
                # Set basic aerosol properties
                if 'aerosol_type' in aerosols:
                    lines.append(f"aerosol_default {aerosols['aerosol_type']}")
                
                if 'aerosol_visibility_km' in aerosols and aerosols['aerosol_visibility_km']:
                    lines.append(f"aerosol_visibility {aerosols['aerosol_visibility_km']}")
                
                if 'aerosol_angstrom_parameters' in aerosols and aerosols['aerosol_angstrom_parameters']:
                    alpha, beta = aerosols['aerosol_angstrom_parameters']
                    lines.append(f"aerosol_angstrom {alpha} {beta}")
        
        # --- Wavelength range ---
        wl_min, wl_max = sim_defaults.wavelength_nm
        lines.append(f"wavelength {wl_min} {wl_max}")
        
        # --- Viewing geometry ---
        if hasattr(sim_defaults, 'viewing_geometry'):
            if sim_defaults.viewing_geometry == 'nadir':
                lines.append("umu 1.0")  # cos(theta) = 1.0 for nadir (looking down)
                lines.append("phi 0.0")   # Default azimuth angle
            elif sim_defaults.viewing_geometry == 'custom' and hasattr(sim_defaults, 'umu') and sim_defaults.umu is not None:
                umu_str = " ".join(map(str, sim_defaults.umu))
                lines.append(f"umu {umu_str}")
                
                if hasattr(sim_defaults, 'phi') and sim_defaults.phi is not None:
                    phi_str = " ".join(map(str, sim_defaults.phi))
                    lines.append(f"phi {phi_str}")
        
        # --- Output ---
        # Specify output altitudes
        if hasattr(sim_defaults, 'output_altitudes_km') and sim_defaults.output_altitudes_km:
            zout_str = " ".join(map(str, sim_defaults.output_altitudes_km))
            lines.append(f"zout {zout_str}")
        
        # Specify output quantities
        if hasattr(sim_defaults, 'output_columns') and sim_defaults.output_columns:
            output_str = " ".join(sim_defaults.output_columns)
            lines.append(f"output_user {output_str}")
        
        # --- Handle wavelength integration if requested ---
        if hasattr(sim_defaults, 'integrate_wavelength') and sim_defaults.integrate_wavelength:
            lines.append("output_integrated")
        
        # Add any keyword overrides (advanced usage)
        for key, value in kwargs.items():
            # Skip None values and keys that start with "override_" (these are handled separately)
            if value is not None and not key.startswith("override_"):
                lines.append(f"{key} {value}")
        
        # Return the complete input content
        return "\n".join(lines)
    
    except Exception as e:
        logger.error(f"Error generating uvspec input content: {e}")
        raise InputGenerationError(f"Failed to generate uvspec input content: {e}")

# --- Output Parsing ---

def parse_uvspec_output(
    output_file: Path,
    config: SimulationConfig
) -> Dict[str, Any]:
    """
    Parse uvspec output file and extract the results.
    
    Args:
        output_file: Path to the uvspec output file
        config: Simulation configuration (to determine output structure)
        
    Returns:
        Dictionary with parsed output data
        
    Raises:
        OutputParsingError: If output parsing fails
    """
    try:
        # Output structure depends on the output_user columns in the config
        output_columns = config.simulation_defaults.output_columns
        
        # Check if the file exists
        if not output_file.exists():
            raise OutputParsingError(f"Output file does not exist: {output_file}")
        
        # Open and read the file
        with open(output_file, 'r') as f:
            lines = f.readlines()
        
        # Remove any comment or empty lines
        data_lines = [line.strip() for line in lines if line.strip() and not line.startswith('#')]
        
        # Extract column data
        if len(data_lines) == 0:
            raise OutputParsingError(f"No data found in output file: {output_file}")
        
        # Handle multi-level output (for zout with multiple altitudes)
        results = {}
        
        # If we have multiple output altitude levels
        if hasattr(config.simulation_defaults, 'output_altitudes_km') and len(config.simulation_defaults.output_altitudes_km) > 1:
            # Initialize results structure
            for col_name in output_columns:
                results[col_name] = {}
                for altitude in config.simulation_defaults.output_altitudes_km:
                    results[col_name][altitude] = []
            
            # Parse each line as one altitude level
            current_line = 0
            max_lines = len(data_lines)
            
            while current_line < max_lines:
                # Each altitude level is on a separate line
                for i, altitude in enumerate(config.simulation_defaults.output_altitudes_km):
                    if current_line < max_lines:
                        values = data_lines[current_line].split()
                        if len(values) != len(output_columns):
                            logger.warning(f"Line has {len(values)} values but expected {len(output_columns)} columns")
                            # Try to handle misaligned columns gracefully
                            values = values[:len(output_columns)] if len(values) > len(output_columns) else values + ['nan'] * (len(output_columns) - len(values))
                        
                        # Add values to results
                        for j, col_name in enumerate(output_columns):
                            try:
                                results[col_name][altitude].append(float(values[j]))
                            except (ValueError, IndexError):
                                logger.warning(f"Error parsing column {j} ('{col_name}') at altitude {altitude}")
                                results[col_name][altitude].append(np.nan)
                        
                        current_line += 1
                    else:
                        break
        else:
            # Single altitude output (simpler case)
            # Initialize results structure
            for col_name in output_columns:
                results[col_name] = []
            
            # Parse each line as one time point
            for line in data_lines:
                values = line.split()
                if len(values) != len(output_columns):
                    logger.warning(f"Line has {len(values)} values but expected {len(output_columns)} columns")
                    # Try to handle misaligned columns gracefully
                    values = values[:len(output_columns)] if len(values) > len(output_columns) else values + ['nan'] * (len(output_columns) - len(values))
                
                # Add values to results
                for i, col_name in enumerate(output_columns):
                    try:
                        results[col_name].append(float(values[i]))
                    except (ValueError, IndexError):
                        logger.warning(f"Error parsing column {i} ('{col_name}')")
                        results[col_name].append(np.nan)
        
        # Add metadata about the source file
        results['_source_file'] = str(output_file)
        results['_num_data_points'] = len(data_lines)
        
        return results
        
    except Exception as e:
        logger.error(f"Error parsing uvspec output: {e}")
        raise OutputParsingError(f"Failed to parse uvspec output: {e}")

# --- Results Saving ---

def save_results_to_netcdf(
    data: Dict[str, Any],
    output_path: Path,
    input_ds: xr.Dataset,
    config: SimulationConfig,
    simulation_params: Optional[Dict] = None
) -> Path:
    """
    Save simulation results to a NetCDF file.
    
    Args:
        data: Dictionary of simulation results
        output_path: Path for output NetCDF file
        input_ds: Original input dataset with coordinates
        config: Simulation configuration
        simulation_params: Additional simulation parameters for metadata
        
    Returns:
        Path to the output NetCDF file
        
    Raises:
        OutputParsingError: If output cannot be saved
    """
    try:
        # Create a new dataset with input coordinates
        ds = xr.Dataset()
        
        # Add coordinates from input dataset
        time_coords = ['time', 'Time', 'datetime', 'date']
        lat_coords = ['latitude', 'Latitude', 'lat', 'Lat']
        lon_coords = ['longitude', 'Longitude', 'lon', 'Lon']
        
        # Find time coordinate
        time_var = next((c for c in time_coords if c in input_ds), None)
        if time_var:
            ds[time_var] = input_ds[time_var]
        
        # Find lat coordinate
        lat_var = next((c for c in lat_coords if c in input_ds), None)
        if lat_var:
            if lat_var in input_ds.dims:
                # Latitude is a dimension
                ds[lat_var] = input_ds[lat_var]
            else:
                # Latitude is a coordinate
                ds[lat_var] = (time_var, input_ds[lat_var].values)
        
        # Find lon coordinate
        lon_var = next((c for c in lon_coords if c in input_ds), None)
        if lon_var:
            if lon_var in input_ds.dims:
                # Longitude is a dimension
                ds[lon_var] = input_ds[lon_var]
            else:
                # Longitude is a coordinate
                ds[lon_var] = (time_var, input_ds[lon_var].values)
        
        # Add altitude coordinate if we have multi-level data
        has_multi_level = False
        for col_name, values in data.items():
            if isinstance(values, dict) and not col_name.startswith('_'):
                has_multi_level = True
                break
        
        if has_multi_level:
            # Get altitude levels from config or data
            if hasattr(config.simulation_defaults, 'output_altitudes_km'):
                altitude_levels = config.simulation_defaults.output_altitudes_km
            else:
                # Extract from the first multi-level variable
                for col_name, values in data.items():
                    if isinstance(values, dict) and not col_name.startswith('_'):
                        altitude_levels = sorted(float(alt) for alt in values.keys())
                        break
            
            ds['altitude'] = xr.DataArray(
                altitude_levels,
                dims=('altitude',),
                attrs={'units': 'km', 'long_name': 'Altitude above sea level'}
            )
        
        # Add variables from results
        for col_name, values in data.items():
            if col_name.startswith('_'):
                # Skip metadata fields
                continue
            
            if isinstance(values, dict):
                # Multi-level data
                var_dims = (time_var, 'altitude')
                var_shape = (len(ds[time_var]), len(ds['altitude']))
                var_data = np.zeros(var_shape)
                
                # Fill data for each altitude level
                for i, alt in enumerate(ds['altitude'].values):
                    if alt in values:
                        var_data[:, i] = values[alt]
                
                # Add the variable to the dataset
                ds[col_name] = xr.DataArray(
                    var_data,
                    dims=var_dims,
                    coords={time_var: ds[time_var], 'altitude': ds['altitude']},
                    attrs={'units': _get_variable_units(col_name)}
                )
            else:
                # Single-level data
                # Convert to numpy array and ensure it's flattened to 1D
                if not isinstance(values, np.ndarray):
                    values = np.array(values)
                
                # Ensure 1D shape if needed
                if values.ndim > 1:
                    values = values.flatten()
                    logger.debug(f"Flattened values for {col_name} from multi-dimensional to 1D (length {len(values)})")
                
                # Handle potential size mismatch
                if len(values) == len(ds[time_var]):
                    ds[col_name] = xr.DataArray(
                        values,
                        dims=(time_var,),
                        coords={time_var: ds[time_var]},
                        attrs={'units': _get_variable_units(col_name)}
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
                        attrs={'units': _get_variable_units(col_name)}
                    )
        
        # Add metadata
        ds.attrs['title'] = 'libRadtran simulation results'
        ds.attrs['institution'] = 'Generated by pyradtran'
        ds.attrs['source'] = 'libRadtran uvspec'
        ds.attrs['history'] = f'Created {datetime.now().isoformat()}'
        ds.attrs['references'] = 'Mayer and Kylling (2005): Technical note: The libRadtran software package for radiative transfer calculations'
        ds.attrs['comment'] = 'Simulation results from LibRadtran/uvspec'
        
        # Add configuration details to metadata
        ds.attrs['rte_solver'] = config.simulation_defaults.rte_solver
        ds.attrs['mol_abs_param'] = config.simulation_defaults.mol_abs_param
        ds.attrs['wavelength_range'] = f"{config.simulation_defaults.wavelength_nm[0]}-{config.simulation_defaults.wavelength_nm[1]} nm"
        
        # Add any additional simulation parameters
        if simulation_params:
            for key, value in simulation_params.items():
                if isinstance(value, (str, int, float, bool)):
                    ds.attrs[f'param_{key.replace(".", "_")}'] = str(value)
        
        # Save to NetCDF file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        ds.to_netcdf(output_path)
        logger.info(f"Results saved to NetCDF file: {output_path}")
        
        return output_path
        
    except Exception as e:
        logger.error(f"Error saving results to NetCDF: {e}")
        raise OutputParsingError(f"Failed to save results to NetCDF: {e}")


def _get_variable_units(variable_name: str) -> str:
    """Get the standard units for a variable name."""
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