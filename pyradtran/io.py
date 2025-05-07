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
        lines.append(f"source solar {config.paths.solar_spectrum} per_nm")
        
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
        # Collect all output-related directives in the correct order
        output_directives = []
        
        # Specify output altitudes
        if hasattr(sim_defaults, 'output_altitudes_km') and sim_defaults.output_altitudes_km:
            zout_str = " ".join(map(str, sim_defaults.output_altitudes_km))
            output_directives.append(f"zout {zout_str}")
        
        # --- Handle output processing for spectral quantities ---
        # Add per_nm to ensure spectral normalization
        output_directives.append("output_process per_nm")
        
        # --- Handle wavelength integration if requested ---
        if hasattr(sim_defaults, 'integrate_wavelength') and sim_defaults.integrate_wavelength:
            output_directives.append("output_process integrate")
        
        # Add any additional output-related options from the config
        if hasattr(sim_defaults, 'additional_options'):
            for option in sim_defaults.additional_options:
                if option.startswith('output_process'):
                    output_directives.append(option)
        
        # Specify output quantities - CRITICAL: output_user MUST BE LAST in LibRadtran
        if hasattr(sim_defaults, 'output_columns') and sim_defaults.output_columns:
            output_str = " ".join(sim_defaults.output_columns)
            output_directives.append(f"output_user {output_str}")
        
        # Now add all the output directives to the main lines list
        # CRITICAL: The order matters in LibRadtran
        lines.extend(output_directives)
        
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

# --- Input File Analysis ---

def _analyze_uvspec_input_file(input_file: Path) -> Dict[str, Any]:
    """
    Analyzes an input file to extract information about the expected output structure.
    
    This helps determine whether the output will contain wavelength, altitude, or both dimensions.
    
    Args:
        input_file: Path to the uvspec input file
        
    Returns:
        Dictionary with analysis results, including expected dimensions and columns
    """
    results = {
        'has_wavelength_range': False,
        'wavelength_min': None,
        'wavelength_max': None,
        'has_multiple_altitudes': False,
        'altitude_levels': [],
        'output_columns': [],
        'wavelength_integrated': False,
    }
    
    try:
        # Read the input file
        with open(input_file, 'r') as f:
            lines = f.readlines()
        
        # Parse directives that affect output structure
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split()
            
            # Check for wavelength range
            if parts[0] == 'wavelength' and len(parts) >= 3:
                try:
                    results['has_wavelength_range'] = True
                    results['wavelength_min'] = float(parts[1])
                    results['wavelength_max'] = float(parts[2])
                except (ValueError, IndexError):
                    pass
            
            # Check for output altitude levels
            elif parts[0] == 'zout' and len(parts) >= 2:
                try:
                    # Multiple altitude levels
                    altitudes = [float(z) for z in parts[1:]]
                    results['has_multiple_altitudes'] = len(altitudes) > 1
                    results['altitude_levels'] = altitudes
                except (ValueError, IndexError):
                    pass
            
            # Check for output columns
            elif parts[0] == 'output_user' and len(parts) >= 2:
                results['output_columns'] = parts[1:]
            
            # Check for wavelength integration
            elif parts[0] == 'output_process' and len(parts) >= 2:
                if parts[1] == 'integrate':
                    results['wavelength_integrated'] = True
        
        # Lambda in output columns indicates spectral data
        results['has_lambda_column'] = 'lambda' in results['output_columns']
        
        return results
    
    except Exception as e:
        logger.error(f"Error analyzing uvspec input file: {e}")
        return results

# --- Output Parsing ---

def _analyze_uvspec_output_structure(output_file: Path) -> Dict[str, Any]:
    """
    Analyzes the structure of a uvspec output file to determine dimensions.
    
    This is used to help debug parsing issues by understanding the actual 
    output structure when it doesn't match expectations.
    
    Args:
        output_file: Path to uvspec output file
        
    Returns:
        A dictionary with analysis results
    """
    results = {
        'total_rows': 0,
        'columns_per_row': [],
        'unique_column_counts': set(),
        'first_column_values': set(),
        'sample_rows': [],
    }
    
    try:
        with open(output_file, 'r') as f:
            lines = f.readlines()
            
        # Filter out comments and empty lines
        data_lines = [line.strip() for line in lines if line.strip() and not line.startswith('#')]
        results['total_rows'] = len(data_lines)
        
        # Analyze column structure
        for i, line in enumerate(data_lines):
            values = line.split()
            results['columns_per_row'].append(len(values))
            results['unique_column_counts'].add(len(values))
            
            # Collect first column values (often wavelength or altitude)
            if values:
                try:
                    results['first_column_values'].add(float(values[0]))
                except (ValueError, IndexError):
                    pass
            
            # Save sample rows
            if i < 5 or i >= len(data_lines) - 5 or i % (len(data_lines) // 10) == 0:
                results['sample_rows'].append(line)
        
        results['column_stats'] = {
            'min': min(results['columns_per_row']),
            'max': max(results['columns_per_row']),
            'consistent': len(results['unique_column_counts']) == 1
        }
        
        # Clean up for readability
        results['first_column_values'] = sorted(results['first_column_values'])
        if len(results['first_column_values']) > 20:
            # Trim to show range instead of all values
            results['first_column_values'] = f"Range: {min(results['first_column_values'])} to {max(results['first_column_values'])}, Count: {len(results['first_column_values'])}"
        
        return results
        
    except Exception as e:
        logger.error(f"Error analyzing uvspec output structure: {e}")
        return {'error': str(e)}

def parse_uvspec_output(
    output_file: Path,
    config: SimulationConfig,
    input_file: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Parse uvspec output file and extract the results.
    
    Args:
        output_file: Path to the uvspec output file
        config: Simulation configuration (to determine output structure)
        input_file: Optional path to the corresponding input file for improved column detection
        
    Returns:
        Dictionary with parsed output data. Structure depends on the simulation type:
        - For standard output: {col_name: [values...], ...}
        - For multi-altitude output: {col_name: {altitude1: [values...], altitude2: [values...]}, ...}
        - For spectral output: {col_name: {wavelength1: [values...], wavelength2: [values...]}, ...}
        - For both: nested dictionaries with wavelength and altitude dimensions
        
    Raises:
        OutputParsingError: If output parsing fails
    """
    import traceback
    import numpy as np
    
    try:
        # Check if the file exists
        if not output_file.exists():
            raise OutputParsingError(f"Output file does not exist: {output_file}")
        
        # First, analyze the input file if available to get expected column structure
        input_analysis = None
        if input_file and input_file.exists():
            input_analysis = _analyze_uvspec_input_file(input_file)
            logger.debug(f"Input file analysis: {input_analysis}")
        
        # Output structure depends on the output_user columns in the config
        output_columns = config.simulation_defaults.output_columns
        
        # Open and read the file
        with open(output_file, 'r') as f:
            lines = f.readlines()
        
        # Remove any comment or empty lines
        data_lines = [line.strip() for line in lines if line.strip() and not line.startswith('#')]
        
        # Extract column data
        if len(data_lines) == 0:
            raise OutputParsingError("No data found in output file")
        
        # Determine wavelength integration from config or input file analysis
        wavelength_integrated = False
        if hasattr(config.simulation_defaults, 'integrate_wavelength'):
            wavelength_integrated = config.simulation_defaults.integrate_wavelength
        
        # Check if 'output_process integrate' is in additional_options
        if hasattr(config.simulation_defaults, 'additional_options'):
            for opt in config.simulation_defaults.additional_options:
                if opt == 'output_process integrate':
                    wavelength_integrated = True
                    break
        
        # If we have input file analysis, use that information too
        if input_analysis:
            # Input file analysis takes precedence for determining wavelength integration
            if 'wavelength_integrated' in input_analysis:
                wavelength_integrated = input_analysis['wavelength_integrated']
            
            # Use output columns from input file if available
            if input_analysis.get('output_columns'):
                output_columns = input_analysis['output_columns']
        
        logger.debug(f"Wavelength integration enabled: {wavelength_integrated}")
        
        # Check configuration for expected output type
        has_lambda_column = 'lambda' in output_columns
        has_altitude = any(col in output_columns for col in ['zout', 'z'])
        multiple_altitudes = False
        
        # Get altitude levels from config or input file analysis
        altitude_levels = []
        if input_analysis and input_analysis.get('altitude_levels'):
            # Use altitude levels from input file analysis if available
            altitude_levels = input_analysis['altitude_levels']
            multiple_altitudes = len(altitude_levels) > 1
        elif hasattr(config.simulation_defaults, 'output_altitudes_km'):
            altitudes = config.simulation_defaults.output_altitudes_km
            if isinstance(altitudes, list):
                altitude_levels = [float(alt) for alt in altitudes]
                multiple_altitudes = len(altitude_levels) > 1
        
        logger.debug(f"Detected altitude levels: {altitude_levels}")
        
        # Get wavelength range from config or input file analysis
        wavelength_min, wavelength_max = config.simulation_defaults.wavelength_nm
        if input_analysis and input_analysis.get('has_wavelength_range'):
            wavelength_min = input_analysis.get('wavelength_min', wavelength_min)
            wavelength_max = input_analysis.get('wavelength_max', wavelength_max)
        
        # Analyze output structure
        output_analysis = _analyze_uvspec_output_structure(output_file)
        logger.debug(f"Output file analysis: {output_analysis}")
        
        # When wavelength integration is requested, we should only have one line per altitude
        expected_lines = 1  # Default for single altitude, integrated wavelength
        if multiple_altitudes:
            expected_lines = len(altitude_levels)
        
        # Determine if output contains spectral data
        file_has_spectral_data = False
        
        # If we have 'lambda' in the output columns, it's definitely spectral data
        if has_lambda_column:
            file_has_spectral_data = True
            logger.info("Detected spectral data based on 'lambda' column")
        # If we have far more lines than expected, it's likely spectral data
        elif len(data_lines) > expected_lines * 3:  # Using 3 as a threshold (not 10 as before)
            file_has_spectral_data = True
            logger.info(f"Detected potential spectral data based on line count: {len(data_lines)} > {expected_lines}")
        # If wavelength integration was requested but failed, we'll have spectral data
        elif wavelength_integrated and len(data_lines) > expected_lines:
            file_has_spectral_data = True
            logger.warning("Wavelength integration appears to have failed, treating as spectral data")
            
        # Calculate wavelength dimension if spectral data is present
        n_wavelengths = None
        wavelength_values = None
        
        if file_has_spectral_data:
            # Calculate how many wavelength points we have
            if multiple_altitudes:
                # For multi-altitude case:
                # Total output lines = n_altitudes * n_wavelengths
                # So n_wavelengths = len(data_lines) / n_altitudes
                n_wavelengths = len(data_lines) // len(altitude_levels)
                
                # Verify that the division is clean
                if len(data_lines) % len(altitude_levels) != 0:
                    logger.warning(f"Number of data lines ({len(data_lines)}) is not evenly divisible by "
                                  f"number of altitude levels ({len(altitude_levels)}). Using estimated wavelength count.")
            else:
                # For single altitude case, all lines are wavelength points
                n_wavelengths = len(data_lines)
            
            logger.debug(f"Estimated number of wavelength points: {n_wavelengths}")
            
            # Create wavelength values
            # If we have lambda in the columns, extract the values directly from the data
            if has_lambda_column:
                lambda_idx = output_columns.index('lambda')
                try:
                    wavelength_values = []
                    # Extract unique wavelength values from the first 'n_wavelengths' lines
                    for i in range(min(n_wavelengths, len(data_lines))):
                        values = data_lines[i].split()
                        if lambda_idx < len(values):
                            wavelength_values.append(float(values[lambda_idx]))
                    
                    # If we couldn't extract enough values, fall back to estimated range
                    if len(wavelength_values) < n_wavelengths:
                        logger.warning(f"Could only extract {len(wavelength_values)} wavelength values, falling back to estimated range")
                        wavelength_values = np.linspace(wavelength_min, wavelength_max, n_wavelengths)
                except Exception as e:
                    logger.warning(f"Error extracting wavelength values from data: {e}")
                    wavelength_values = np.linspace(wavelength_min, wavelength_max, n_wavelengths)
            else:
                # Otherwise estimate from the wavelength range
                # If exactly matching formula N_out = ((wavelength_max - wavelength_min) + 1)
                # then wavelengths are integers from min to max inclusive
                if n_wavelengths == (wavelength_max - wavelength_min + 1):
                    wavelength_values = np.arange(wavelength_min, wavelength_max + 1, dtype=float)
                    logger.debug(f"Using integer wavelength range: {wavelength_min} to {wavelength_max}")
                else:
                    # Otherwise calculate a reasonable wavelength range
                    wavelength_values = np.linspace(wavelength_min, wavelength_max, n_wavelengths)
                    logger.debug(f"Using linearly spaced wavelength range with {n_wavelengths} points")
        
        # Determine the output structure and create the results dictionary
        if file_has_spectral_data and wavelength_values is not None:
            # Spectral data detected
            if multiple_altitudes:
                # Spectral data with multiple altitudes
                results = {
                    '_simulation_type': 'spectral_multi_altitude',
                    '_wavelength_values': wavelength_values,
                    '_num_wavelengths': n_wavelengths,
                    '_unique_altitudes': altitude_levels,
                    '_source_file': str(output_file),
                    '_num_data_points': len(data_lines)
                }
                
                logger.info(f"Parsing as spectral multi-altitude data with {len(altitude_levels)} levels and {n_wavelengths} wavelength points")
                
                # Initialize result structure with nested dictionaries for altitude and wavelength
                for col_name in output_columns:
                    # Skip the 'lambda' column in the results dict if it's in the output columns
                    if col_name == 'lambda' and has_lambda_column:
                        continue
                        
                    results[col_name] = {}
                    for altitude in altitude_levels:
                        alt_key = float(altitude)
                        results[col_name][alt_key] = {}
                        for wl_idx, wl in enumerate(wavelength_values):
                            results[col_name][alt_key][float(wl)] = None  # Initialize to None, will be filled with data
                
                # Parse data assuming it's organized as blocks per altitude
                # Each altitude has n_wavelengths consecutive lines
                for alt_idx, altitude in enumerate(altitude_levels):
                    alt_key = float(altitude)
                    
                    # Calculate data line range for this altitude
                    start_line = alt_idx * n_wavelengths
                    end_line = start_line + n_wavelengths
                    
                    # Check bounds to prevent index errors
                    if start_line >= len(data_lines):
                        logger.warning(f"Not enough data lines for altitude {altitude}, expected start at line {start_line}")
                        continue
                    
                    end_line = min(end_line, len(data_lines))
                    
                    # Process each wavelength line for this altitude
                    for wl_idx, line_idx in enumerate(range(start_line, end_line)):
                        if wl_idx >= len(wavelength_values):
                            logger.warning(f"Wavelength index {wl_idx} exceeds available wavelength values")
                            break
                            
                        wl_key = float(wavelength_values[wl_idx])
                        line = data_lines[line_idx]
                        values = line.split()
                        
                        # Skip validation of column count if lambda is one of the columns
                        if not has_lambda_column and len(values) != len(output_columns):
                            logger.warning(f"Line {line_idx} has {len(values)} values but expected {len(output_columns)} columns")
                            continue
                            
                        # Store each column value for this altitude and wavelength
                        for col_idx, col_name in enumerate(output_columns):
                            # Skip the 'lambda' column in the results dict
                            if col_name == 'lambda' and has_lambda_column:
                                continue
                                
                            try:
                                if col_idx < len(values):
                                    val = float(values[col_idx])
                                    results[col_name][alt_key][wl_key] = val
                            except (ValueError, IndexError):
                                logger.warning(f"Error parsing value for column {col_name} at altitude {altitude}, wavelength {wl_key}")
                                results[col_name][alt_key][wl_key] = float('nan')
            
            else:
                # Spectral data with single altitude
                results = {
                    '_simulation_type': 'spectral',
                    '_wavelength_values': wavelength_values,
                    '_num_wavelengths': n_wavelengths,
                    '_source_file': str(output_file),
                    '_num_data_points': len(data_lines)
                }
                
                logger.info(f"Parsing as spectral data with {n_wavelengths} wavelength points")
                
                # Initialize result structure with a nested dictionary for wavelength
                for col_name in output_columns:
                    # Skip the 'lambda' column in the results dict if it's in the output columns
                    if col_name == 'lambda' and has_lambda_column:
                        continue
                        
                    results[col_name] = {}
                    for wl_idx, wl in enumerate(wavelength_values):
                        results[col_name][float(wl)] = None  # Initialize to None, will be filled with data
                
                # Parse each line (each representing a different wavelength)
                for wl_idx, line in enumerate(data_lines):
                    if wl_idx >= len(wavelength_values):
                        logger.warning(f"More data lines ({len(data_lines)}) than wavelength values ({len(wavelength_values)})")
                        break
                        
                    wl_key = float(wavelength_values[wl_idx])
                    values = line.split()
                    
                    # Skip validation of column count if lambda is one of the columns
                    if not has_lambda_column and len(values) != len(output_columns):
                        logger.warning(f"Line has {len(values)} values but expected {len(output_columns)} columns")
                        continue
                        
                    # Store each column value for this wavelength
                    for col_idx, col_name in enumerate(output_columns):
                        # Skip the 'lambda' column in the results dict
                        if col_name == 'lambda' and has_lambda_column:
                            continue
                            
                        try:
                            if col_idx < len(values):
                                val = float(values[col_idx])
                                results[col_name][wl_key] = val
                        except (ValueError, IndexError):
                            logger.warning(f"Error parsing value for column {col_name} at wavelength {wl_key}")
                            results[col_name][wl_key] = float('nan')
            
            return results
        
        elif wavelength_integrated and multiple_altitudes:
            # Integrated wavelength with multiple altitudes
            results = {
                '_simulation_type': 'multi_altitude',
                '_unique_altitudes': altitude_levels,
                '_source_file': str(output_file),
                '_num_data_points': len(data_lines)
            }
            
            logger.info(f"Parsing as integrated multi-altitude data with {len(altitude_levels)} altitude levels")
            
            # Initialize result structure for each altitude
            for col_name in output_columns:
                results[col_name] = {}
                for altitude in altitude_levels:
                    alt_key = float(altitude)
                    results[col_name][alt_key] = None  # Will be filled with data
            
            # If we have exactly one line per altitude, this is perfect
            if len(data_lines) == len(altitude_levels):
                # Perfect match - one line per altitude
                for line_idx, altitude in enumerate(sorted(altitude_levels)):
                    alt_key = float(altitude)
                    values = data_lines[line_idx].split()
                    
                    if len(values) != len(output_columns):
                        logger.warning(f"Line has {len(values)} values but expected {len(output_columns)} columns")
                        # Fill with NaN values
                        for col_idx, col_name in enumerate(output_columns):
                            results[col_name][alt_key] = float('nan')
                        continue
                    
                    # Store each column value for this altitude
                    for col_idx, col_name in enumerate(output_columns):
                        try:
                            val = float(values[col_idx])
                            results[col_name][alt_key] = val
                        except (ValueError, IndexError):
                            logger.warning(f"Error parsing column {col_idx} ('{col_name}') at altitude {altitude}")
                            results[col_name][alt_key] = float('nan')
            else:
                # Mismatch between number of lines and altitudes, 
                # but we might still be able to handle it if we have fewer lines than altitudes
                logger.warning(f"Mismatch between number of data lines ({len(data_lines)}) and altitudes ({len(altitude_levels)})")
                
                # Use what data we have
                for line_idx, line in enumerate(data_lines):
                    if line_idx >= len(altitude_levels):
                        break  # Don't process more lines than altitudes
                    
                    alt_key = float(sorted(altitude_levels)[line_idx])
                    values = line.split()
                    
                    for col_idx, col_name in enumerate(output_columns):
                        if col_idx < len(values):
                            try:
                                val = float(values[col_idx])
                                results[col_name][alt_key] = val
                            except (ValueError, IndexError):
                                results[col_name][alt_key] = float('nan')
                        else:
                            results[col_name][alt_key] = float('nan')
            
            return results
        
        else:
            # Standard single-altitude output or properly integrated output
            results = {
                '_simulation_type': 'standard',
                '_source_file': str(output_file),
                '_num_data_points': len(data_lines)
            }
            
            logger.info("Parsing as standard output (single altitude or integrated)")
            
            # Initialize results structure
            for col_name in output_columns:
                results[col_name] = []
            
            # For clean integrated output, we should have just one line
            if wavelength_integrated and len(data_lines) == 1:
                # Parse the single line
                values = data_lines[0].split()
                
                if len(values) != len(output_columns):
                    logger.warning(f"Line has {len(values)} values but expected {len(output_columns)} columns")
                    # Pad with NaN if needed
                    values.extend([float('nan')] * (len(output_columns) - len(values)))
                
                for col_idx, col_name in enumerate(output_columns):
                    try:
                        if col_idx < len(values):
                            results[col_name].append(float(values[col_idx]))
                        else:
                            results[col_name].append(float('nan'))
                    except (ValueError, IndexError):
                        logger.warning(f"Error parsing column {col_idx} ('{col_name}')")
                        results[col_name].append(float('nan'))
            else:
                # Multiple lines despite expected integration, or no integration requested
                # Just use the first line
                if len(data_lines) > 0:
                    values = data_lines[0].split()
                    
                    if len(values) != len(output_columns):
                        logger.warning(f"Line has {len(values)} values but expected {len(output_columns)} columns")
                        # Pad with NaN if needed
                        values.extend([float('nan')] * (len(output_columns) - len(values)))
                    
                    for col_idx, col_name in enumerate(output_columns):
                        try:
                            if col_idx < len(values):
                                results[col_name].append(float(values[col_idx]))
                            else:
                                results[col_name].append(float('nan'))
                        except (ValueError, IndexError):
                            logger.warning(f"Error parsing column {col_idx} ('{col_name}')")
                            results[col_name].append(float('nan'))
                else:
                    # No data lines - fill with NaN
                    for col_name in output_columns:
                        results[col_name].append(float('nan'))
            
            return results
    
    except Exception as e:
        logger.error(f"Error parsing uvspec output: {e}")
        logger.debug(traceback.format_exc())
        raise OutputParsingError(f"Failed to parse uvspec output: {e}")

# --- Netcdf Output ---

def _get_variable_units(var_name: str) -> str:
    """Get standard units for known variables"""
    units_map = {
        'lambda': 'nm',  # Wavelength in nanometers
        'edir': 'W m⁻² nm⁻¹',  # Direct irradiance
        'eglo': 'W m⁻² nm⁻¹',  # Global irradiance
        'edn': 'W m⁻² nm⁻¹',   # Downward irradiance
        'eup': 'W m⁻² nm⁻¹',   # Upward irradiance
        'enet': 'W m⁻² nm⁻¹',  # Net irradiance
        'sza': 'degrees',      # Solar zenith angle
        'albedo': 'dimensionless',  # Albedo
        'zout': 'km',          # Altitude
        'z': 'km',             # Altitude
    }
    return units_map.get(var_name, '')

def save_results_to_netcdf(
    data: Dict[str, Any],
    output_path: Union[str, Path],
    input_ds: xr.Dataset,
    config: SimulationConfig,
    simulation_params: Optional[Dict[str, Any]] = None,
    time_var: str = 'time',
    lat_var: str = 'latitude',
    lon_var: str = 'longitude',
    alt_var: str = 'altitude',
    wl_var: str = 'wavelength'
) -> Path:
    """
    Save simulation results to a NetCDF file.
    
    Args:
        data: Dictionary of simulation results from execute_simulation_batch
        output_path: Path to save the NetCDF file
        input_ds: Input xarray Dataset with time, lat, lon coordinates
        config: Simulation configuration
        simulation_params: Optional parameters used in the simulation
        time_var: Name of time variable in input_ds
        lat_var: Name of latitude variable in input_ds
        lon_var: Name of longitude variable in input_ds
        alt_var: Name of altitude variable in output dataset
        wl_var: Name of wavelength variable in output dataset
        
    Returns:
        Path to the saved NetCDF file
    """
    import numpy as np
    import traceback
    
    logger.info(f"Saving results to NetCDF file: {output_path}")
    
    # Identify the simulation type from the results
    sim_type = data.get('_simulation_type', 'standard')
    logger.debug(f"Detected simulation type: {sim_type}")
    
    # Check if we have spectral data
    has_spectral = sim_type in ['spectral', 'spectral_multi_altitude'] and '_wavelength_values' in data
    
    # Check if we have altitude data
    has_altitude = '_unique_altitudes' in data and data['_unique_altitudes']
    
    # Create a new dataset with same coordinates as input
    # First create without altitude or wavelength, then add them if needed
    output_ds = xr.Dataset(
        coords={
            time_var: input_ds[time_var]
        }
    )
    
    # Copy over lat/lon/etc coordinates from source dataset
    for coord_name in [lat_var, lon_var]:
        if coord_name in input_ds.coords:
            output_ds.coords[coord_name] = input_ds.coords[coord_name]
    
    # Add altitude coordinate if needed
    if has_altitude:
        # Use explicit altitudes from results
        altitude_values = np.array(data['_unique_altitudes'], dtype=float)
        
        # Add the altitude coordinate
        output_ds.coords[alt_var] = (alt_var, altitude_values, {'units': 'km'})
        
        logger.debug(f"Created altitude dimension with {len(altitude_values)} levels: {altitude_values}")
    
    # Add wavelength coordinate if needed
    if has_spectral:
        # Use explicit wavelength values from results
        wavelength_values = np.array(data['_wavelength_values'], dtype=float)
        
        # Add the wavelength coordinate
        output_ds.coords[wl_var] = (wl_var, wavelength_values, {'units': 'nm'})
        
        logger.debug(f"Created wavelength dimension with {len(wavelength_values)} points: {min(wavelength_values)} to {max(wavelength_values)} nm")
    
    # Process each result variable
    for col_name, values in data.items():
        # Skip metadata fields
        if col_name.startswith('_'):
            continue
        
        try:
            # Process differently based on simulation type
            if sim_type == 'spectral_multi_altitude':
                # Spectral data with multiple altitudes - values is a nested dict: altitude -> wavelength -> value
                if isinstance(values, dict) and values:
                    # Create a 3D array [time, altitude, wavelength]
                    time_len = len(output_ds[time_var])
                    alt_len = len(data['_unique_altitudes'])
                    wl_len = len(data['_wavelength_values'])
                    
                    # Create a data array with time, altitude, and wavelength dimensions
                    data_array = np.full((time_len, alt_len, wl_len), np.nan)
                    
                    # Fill array with values
                    for t_idx in range(time_len):
                        for a_idx, alt in enumerate(data['_unique_altitudes']):
                            alt_key = float(alt)
                            if alt_key in values:
                                wl_dict = values[alt_key]
                                for w_idx, wl in enumerate(data['_wavelength_values']):
                                    wl_key = float(wl)
                                    if wl_key in wl_dict:
                                        val = wl_dict[wl_key]
                                        
                                        # Handle scalar or array values
                                        if isinstance(val, (list, tuple, np.ndarray)):
                                            if t_idx < len(val):
                                                data_array[t_idx, a_idx, w_idx] = val[t_idx]
                                        else:
                                            data_array[t_idx, a_idx, w_idx] = val
                            
                    # Create DataArray with proper dimensions
                    output_ds[col_name] = xr.DataArray(
                        data_array,
                        dims=(time_var, alt_var, wl_var),
                        coords={
                            time_var: output_ds[time_var],
                            alt_var: data['_unique_altitudes'],
                            wl_var: data['_wavelength_values']
                        },
                        attrs={'units': _get_variable_units(col_name)}
                    )
                    logger.debug(f"Added spectral multi-altitude variable {col_name} with shape {data_array.shape}")
            
            elif sim_type == 'spectral':
                # Spectral data with single altitude - values is a dict with wavelength keys
                if isinstance(values, dict) and values:
                    # Create a 2D array [time, wavelength]
                    time_len = len(output_ds[time_var])
                    wl_len = len(data['_wavelength_values'])
                    
                    # Create a data array with time and wavelength dimensions
                    data_array = np.full((time_len, wl_len), np.nan)
                    
                    # Fill array with values
                    for t_idx in range(time_len):
                        for w_idx, wl in enumerate(data['_wavelength_values']):
                            wl_key = float(wl)
                            if wl_key in values:
                                val = values[wl_key]
                                
                                # Handle scalar or array values
                                if isinstance(val, (list, tuple, np.ndarray)):
                                    if t_idx < len(val):
                                        data_array[t_idx, w_idx] = val[t_idx]
                                else:
                                    data_array[t_idx, w_idx] = val
                    
                    # Create DataArray with proper dimensions
                    output_ds[col_name] = xr.DataArray(
                        data_array,
                        dims=(time_var, wl_var),
                        coords={
                            time_var: output_ds[time_var],
                            wl_var: data['_wavelength_values']
                        },
                        attrs={'units': _get_variable_units(col_name)}
                    )
                    logger.debug(f"Added spectral variable {col_name} with shape {data_array.shape}")
            
            elif sim_type == 'multi_altitude' or sim_type == 'multi_altitude_structured':
                # Multi-altitude data - values is a dict with altitude keys
                if isinstance(values, dict) and values:
                    # Create a 2D array [time, altitude]
                    time_len = len(output_ds[time_var])
                    alt_len = len(data['_unique_altitudes'])
                    
                    # Create a data array with time and altitude dimensions
                    data_array = np.full((time_len, alt_len), np.nan)
                    
                    # Fill array with values
                    for t_idx in range(time_len):
                        for a_idx, alt in enumerate(data['_unique_altitudes']):
                            alt_key = float(alt)
                            if alt_key in values:
                                val = values[alt_key]
                                
                                # Handle scalar or array values
                                if isinstance(val, (list, tuple, np.ndarray)):
                                    if t_idx < len(val):
                                        data_array[t_idx, a_idx] = val[t_idx]
                                else:
                                    data_array[t_idx, a_idx] = val
                    
                    # Create DataArray with proper dimensions
                    output_ds[col_name] = xr.DataArray(
                        data_array,
                        dims=(time_var, alt_var),
                        coords={
                            time_var: output_ds[time_var],
                            alt_var: data['_unique_altitudes']
                        },
                        attrs={'units': _get_variable_units(col_name)}
                    )
                    logger.debug(f"Added multi-altitude variable {col_name} with shape {data_array.shape}")
            
            else:
                # Standard data - values is a list or array
                try:
                    # Convert to numpy array and ensure it's flattened
                    if not isinstance(values, np.ndarray):
                        values = np.array(values, dtype=float)
                    
                    # Ensure 1D shape if needed
                    if values.ndim > 1:
                        values = values.flatten()
                        logger.debug(f"Flattened values for {col_name} from {values.shape}")
                    
                    # Handle potential size mismatch
                    if len(values) == len(output_ds[time_var]):
                        output_ds[col_name] = xr.DataArray(
                            values,
                            dims=(time_var,),
                            coords={time_var: output_ds[time_var]},
                            attrs={'units': _get_variable_units(col_name)}
                        )
                    else:
                        logger.warning(f"Size mismatch for {col_name}: expected {len(output_ds[time_var])}, got {len(values)}")
                        # Create with proper length, padding with NaN if needed
                        temp_data = np.full(len(output_ds[time_var]), np.nan)
                        if len(values) > 0:
                            temp_data[:min(len(values), len(output_ds[time_var]))] = values[:min(len(values), len(output_ds[time_var]))]
                        output_ds[col_name] = xr.DataArray(
                            temp_data,
                            dims=(time_var,),
                            coords={time_var: output_ds[time_var]},
                            attrs={'units': _get_variable_units(col_name)}
                        )
                except Exception as e:
                    logger.error(f"Error adding standard variable {col_name}: {e}")
                    logger.debug(traceback.format_exc())
        
        except Exception as e:
            logger.error(f"Error adding {col_name} to dataset: {e}")
            logger.debug(traceback.format_exc())
    
    # Add global attributes
    output_ds.attrs['created'] = datetime.now().isoformat()
    output_ds.attrs['source'] = 'PyRadtran'
    output_ds.attrs['simulation_type'] = sim_type
    
    if simulation_params:
        # Add simulation parameters as attributes
        for key, value in simulation_params.items():
            if isinstance(value, (str, int, float, bool)):
                output_ds.attrs[f"param_{key}"] = str(value)
    
    # Save to file
    try:
        # Create directory if it doesn't exist
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Use compression settings if provided
        encoding = {}
        if hasattr(config, 'output') and hasattr(config.output, 'netcdf_encoding'):
            # Make sure netcdf_encoding is actually a dictionary
            if isinstance(config.output.netcdf_encoding, dict):
                for var in output_ds.data_vars:
                    encoding[var] = config.output.netcdf_encoding
            else:
                # Default compression settings if netcdf_encoding is not properly defined
                for var in output_ds.data_vars:
                    encoding[var] = {'zlib': True, 'complevel': 4}
        
        # Save to netCDF
        output_ds.to_netcdf(
            output_path,
            encoding=encoding if encoding else None
        )
        logger.info(f"Results saved to {output_path}")
        return output_path
    
    except Exception as e:
        logger.error(f"Error saving results to {output_path}: {e}")
        logger.debug(traceback.format_exc())
        raise OutputParsingError(f"Failed to save results to NetCDF: {e}")