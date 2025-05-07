#!/usr/bin/env python3
"""
Debug script focused on the PyRadtran xarray accessor functionality.

This script creates a minimal example to test the xarray accessor and conversion
of simulation results to a properly structured xarray dataset.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from datetime import datetime
import traceback
import yaml

# Configure logging
logging.basicConfig(level=logging.DEBUG, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("debug_xarray")

# Add parent directory to path if needed
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))

# Import PyRadtran modules
try:
    from pyradtran.config import (
        SimulationConfig, PathsConfig, SimulationDefaults, 
        ExecutionConfig, OutputConfig, load_config
    )
    from pyradtran.core import Simulation
    from pyradtran.io import parse_uvspec_output
    from pyradtran.interface import PyRadtranAccessor
    from pyradtran.exceptions import PyRadtranError
    print("Successfully imported PyRadtran modules")
except ImportError as e:
    print(f"Error importing PyRadtran: {e}")
    sys.exit(1)

# Define paths
LIBRADTRAN_DATA_PATH = '/opt/libradtran/2.0.4/share/libRadtran/data'
LIBRADTRAN_EXEC_PATH = '/opt/libradtran/2.0.4/bin/uvspec'
ATMOSPHERE_FILE = '/projekt_agmwend/data/HALO-AC3/05_VELOX_Tools/add_data/afglsw.dat'
SOLAR_SPECTRUM_FILE = '/projekt_agmwend/home_rad/sophie/libradtran/solar_flux/NewGuey2003.dat'
RADIOSONDE_BASE_PATH = '/projekt_agmwend/data/HALO-AC3/01_soundings/RS_for_libradtran/Dropsondes_HALO/'
SIMULATION_OUTPUT_DIR = '/projekt_agmwend/home_rad/Joshua/HALO-AC3_Arctic_leads/data/simulation/disort/'
WORKING_DIR = Path(__file__).parent / 'work'

# Fixed simulation parameters
FIXED_OZONE_DU = 300.0
FIXED_IWV_MM = 2.0
FIXED_SURFACE_TEMP_K = 250.0
SEA_ICE_BRDF_TYPE = 20  # Sea ice albedo type

def create_pyradtran_config(wavelength_range=[400, 700], integrate_wavelength=False, spectral_mode=True):
    """
    Create a configuration for PyRadtran simulation.
    
    Parameters:
    -----------
    wavelength_range : list
        Min and max wavelength in nm [min, max]
    integrate_wavelength : bool
        Whether to integrate over wavelength (False provides spectral output)
    spectral_mode : bool
        If True, includes 'lambda' in output columns for proper spectral handling
        
    Returns:
    --------
    SimulationConfig
        Complete configuration for PyRadtran
    """
    # Paths to LibRadtran files
    LIBRADTRAN_EXEC_PATH = "/home/uvlibtmp/libRadtran-2.0.4/bin"
    LIBRADTRAN_DATA_PATH = "/home/uvlibtmp/libRadtran-2.0.4/data"
    ATMOSPHERE_FILE = "/home/uvlibtmp/libRadtran-2.0.4/data/atmmod/afglus.dat"
    SOLAR_SPECTRUM_FILE = "/home/uvlibtmp/libRadtran-2.0.4/data/solar_flux/kurudz_1.0nm.dat"
    
    # Determine output columns based on spectral mode
    if spectral_mode:
        output_columns = ['lambda', 'edir', 'eglo', 'edn', 'eup', 'enet', 'albedo']
    else:
        output_columns = ['sza', 'edir', 'eglo', 'edn', 'eup', 'enet', 'albedo']
    
    # Create the base configuration
    config = SimulationConfig(
        paths=PathsConfig(
            libradtran_bin=Path(LIBRADTRAN_EXEC_PATH),
            libradtran_data=Path(LIBRADTRAN_DATA_PATH),
            atmosphere_profile=Path(ATMOSPHERE_FILE),
            solar_spectrum=Path(SOLAR_SPECTRUM_FILE),
            output_dir=Path(WORKING_DIR),
            working_dir=Path(WORKING_DIR)
        ),
        simulation_defaults=SimulationDefaults(
            rte_solver='disort',
            mol_abs_param='reptran',  # Using reptran instead of lowtran for better spectral resolution
            wavelength_nm=wavelength_range,
            output_columns=output_columns,
            output_altitudes_km=[0.0, 1.0, 2.0, 3.0],  # Multiple altitudes for testing
            albedo_type='const',
            albedo_value=0.3,
            aerosols={'enabled': False},
            clouds={'enabled': False},
            integrate_wavelength=integrate_wavelength,
            additional_options=["output_process per_nm"]  # Start with per_nm normalization
        ),
        execution=ExecutionConfig(
            max_workers=1,
            cleanup_temp_files=False,
            debug_mode=True,
            timeout_seconds=60
        ),
        output=OutputConfig(
            filename_prefix="spectral_example" if spectral_mode else "xarray_example",
            filename_suffix=".nc",
            netcdf_encoding={'zlib': True, 'complevel': 4}  # Add default compression
        )
    )
    
    # Add wavelength integration if requested
    if integrate_wavelength:
        config.simulation_defaults.additional_options.append("output_process integrate")
        
    return config

# Create spectral config (with lambda in output columns and no wavelength integration)
spectral_config = create_pyradtran_config(
    wavelength_range=[400, 700], 
    integrate_wavelength=False,
    spectral_mode=True
)
print(f"Spectral config created with {len(spectral_config.simulation_defaults.output_altitudes_km)} altitude levels")
print(f"Output columns: {spectral_config.simulation_defaults.output_columns}")
print(f"Additional options: {spectral_config.simulation_defaults.additional_options}")
print(f"Wavelength integration: {spectral_config.simulation_defaults.integrate_wavelength}")

# Create integrated config (with wavelength integration)
integrated_config = create_pyradtran_config(
    wavelength_range=[400, 700], 
    integrate_wavelength=True,
    spectral_mode=False
)
print(f"\nIntegrated config created with {len(integrated_config.simulation_defaults.output_altitudes_km)} altitude levels")
print(f"Output columns: {integrated_config.simulation_defaults.output_columns}")
print(f"Additional options: {integrated_config.simulation_defaults.additional_options}")
print(f"Wavelength integration: {integrated_config.simulation_defaults.integrate_wavelength}")

# Convert config to dictionary and handle Path objects
def convert_paths_to_strings(d):
    if isinstance(d, dict):
        return {k: convert_paths_to_strings(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [convert_paths_to_strings(v) for v in d]
    elif isinstance(d, Path):
        return str(d)
    else:
        return d

# Save spectral configuration to YAML file
spectral_config_dict = convert_paths_to_strings(spectral_config.dict())
spectral_config_file_path = os.path.join(WORKING_DIR, 'showcase_spectral_config.yaml')
with open(spectral_config_file_path, 'w') as f:
    yaml.dump(spectral_config_dict, f, default_flow_style=False)

# Save integrated configuration to YAML file
integrated_config_dict = convert_paths_to_strings(integrated_config.dict())
integrated_config_file_path = os.path.join(WORKING_DIR, 'showcase_config.yaml')
with open(integrated_config_file_path, 'w') as f:
    yaml.dump(integrated_config_dict, f, default_flow_style=False)

print(f"\nConfigurations saved to:")
print(f"Spectral: {spectral_config_file_path}")
print(f"Integrated: {integrated_config_file_path}")

def debug_xarray_accessor():
    """Main function to debug the xarray accessor functionality."""
    logger.info("Creating sea ice configuration")
    sea_ice_config = create_pyradtran_config()
    
    # Make sure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)
    
    # Create a test dataset with multiple timestamps
    dates = pd.date_range('2022-04-01 10:00:00', periods=3, freq='1H')
    latitudes = [75.0, 76.0, 77.0]
    longitudes = [0.0, 1.0, 2.0]
    
    logger.info("Creating test xarray dataset")
    ds = xr.Dataset(
        coords={
            'time': dates
        },
        data_vars={
            'latitude': ('time', latitudes),
            'longitude': ('time', longitudes)
        },
        attrs={
            'description': 'PyRadtran xarray debug test dataset',
            'created': datetime.now().isoformat()
        }
    )
    
    # Print dataset structure
    logger.info("Test dataset structure:")
    print(f"Dimensions: {ds.dims}")
    print(f"Coordinates: {list(ds.coords)}")
    print(f"Data variables: {list(ds.data_vars)}")
    print(ds)
    
    # Verify simulation_defaults.to_dict() method works
    try:
        param_dict = sea_ice_config.simulation_defaults.to_dict()
        logger.info(f"to_dict() successful, returned {len(param_dict)} parameters")
    except Exception as e:
        logger.error(f"Error in to_dict(): {e}")
        return
    
    # Try running the xarray accessor
    try:
        logger.info("Running xarray accessor")
        
        # Call the run_uvspec method with explicit variable names
        result_ds = ds.pyradtran.run_uvspec(
            parameter_overrides=param_dict,
            time_var='time',
            lat_var='latitude',
            lon_var='longitude',
            return_dataset=True,
            save_to_file=False
        )
        
        logger.info("Xarray accessor run successful!")
        
        # Print the result dataset
        logger.info("Result dataset structure:")
        print(f"Dimensions: {result_ds.dims}")
        print(f"Coordinates: {list(result_ds.coords)}")
        print(f"Data variables: {list(result_ds.data_vars)}")
        print(result_ds)
        
        # Check for specific variables we expect
        expected_vars = ['sza', 'edir', 'eglo', 'edn', 'eup', 'enet', 'esum', 'albedo']
        for var in expected_vars:
            if var in result_ds:
                print(f"Variable {var} found with shape {result_ds[var].shape}")
            else:
                print(f"Variable {var} NOT found in result dataset")
        
        # Try saving the result to NetCDF
        output_file = WORKING_DIR / 'xarray_debug_result.nc'
        logger.info(f"Saving result to {output_file}")
        result_ds.to_netcdf(output_file)
        logger.info(f"Successfully saved result to {output_file}")
        
    except Exception as e:
        logger.error(f"Error in xarray accessor: {str(e)}")
        print(traceback.format_exc())

if __name__ == "__main__":
    debug_xarray_accessor()