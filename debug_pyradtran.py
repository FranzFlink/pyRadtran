#!/usr/bin/env python3
"""
Debug script for the PyRadtran package.

This script reproduces the error seen in the notebook and helps diagnose the issue.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from datetime import datetime

# Configure logging to see what's happening
logging.basicConfig(level=logging.DEBUG, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s') # Fixed format key

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
    from pyradtran.exceptions import PyRadtranError
    print("Successfully imported PyRadtran modules")
except ImportError as e:
    print(f"Error importing PyRadtran: {e}")
    sys.exit(1)

# Add parent directory to path if needed
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
    print(f"Added {module_path} to sys.path")

# Define paths
# Define LibRadtran and data paths (same as in disort.py)
LIBRADTRAN_DATA_PATH = '/opt/libradtran/2.0.4/share/libRadtran/data'
LIBRADTRAN_EXEC_PATH = '/opt/libradtran/2.0.4/bin/uvspec'
ATMOSPHERE_FILE = '/projekt_agmwend/data/HALO-AC3/05_VELOX_Tools/add_data/afglsw.dat'
SOLAR_SPECTRUM_FILE = '/projekt_agmwend/home_rad/sophie/libradtran/solar_flux/NewGuey2003.dat'
RADIOSONDE_BASE_PATH = '/projekt_agmwend/data/HALO-AC3/01_soundings/RS_for_libradtran/Dropsondes_HALO/'
SIMULATION_OUTPUT_DIR = '/projekt_agmwend/home_rad/Joshua/HALO-AC3_Arctic_leads/data/simulation/disort/'
WORKING_DIR = os.path.join(module_path, 'work')


# Fixed atmospheric properties
FIXED_OZONE_DU = 300.0
FIXED_IWV_MM = 2.0
FIXED_SURFACE_TEMP_K = 250.0
# Fixed surface albedo
SEA_ICE_BRDF_TYPE = 20 # Sea ice albedo type

def create_sea_ice_config():
    """Create a configuration for sea ice simulations similar to disort.py"""
    return SimulationConfig(
        paths=PathsConfig(
            libradtran_bin=Path(LIBRADTRAN_EXEC_PATH),
            libradtran_data=Path(LIBRADTRAN_DATA_PATH),
            atmosphere_profile=Path(ATMOSPHERE_FILE),
            solar_spectrum=Path(SOLAR_SPECTRUM_FILE),
            radiosonde_base=Path(RADIOSONDE_BASE_PATH),
            output_dir=Path(SIMULATION_OUTPUT_DIR),
            working_dir=Path(WORKING_DIR)  # Use dedicated working directory
        ),
        simulation_defaults=SimulationDefaults(
            rte_solver='disort',  # Use disort for accuracy in sea ice simulations
            mol_abs_param='lowtran per_nm',  # Same as disort.py
            wavelength_nm=[400, 3600],  # Same range as disort.py
            output_columns=['sza', 'edir', 'eglo', 'edn', 'eup', 'enet', 'esum', 'albedo'],
            output_altitudes_km=[0.0],  # Surface level only
            
            # Surface properties
            albedo_type='library',
            albedo_library='IGBP',
            brdf_type='rpv',
            brdf_rpv_type=SEA_ICE_BRDF_TYPE,  # Sea ice BRDF
            surface_temperature_k=FIXED_SURFACE_TEMP_K,
            
            # Fixed atmospheric composition
            mol_modify={
                'O3': {'value': FIXED_OZONE_DU, 'unit': 'DU'},
                'H2O': {'value': FIXED_IWV_MM, 'unit': 'MM'}
            },
            
            # Default aerosols
            aerosols={
                'enabled': True,
                'aerosol_type': 'default'
            },
            
            # No clouds by default
            clouds={
                'enabled': False
            }
        ),
        # Add the missing required parameters
        execution=ExecutionConfig(
            max_workers=1,  # Single worker for debugging
            cleanup_temp_files=True,  # Clean up temporary files after simulation
            debug_mode=False,  # Enable debug mode to see what's happening
            timeout_seconds=300  # 5-minute timeout for simulations
        ),
        output=OutputConfig(
            filename_prefix="sea_ice_sim",  # Prefix for output files
            filename_suffix="_results.nc",  # Suffix for output files
            netcdf_encoding={"zlib": True, "complevel": 5}  # Enable compression for NetCDF files
        )
    )

    # Create and display the sea ice configuration
    try:
        sea_ice_config = create_sea_ice_config()
        print("Sea ice configuration created successfully!")
        
        print(f"\nPaths:")
        for key, value in vars(sea_ice_config.paths).items():
            print(f"  {key}: {value}")
        
        print(f"\nSimulation Defaults (selected):")
        print(f"  rte_solver: {sea_ice_config.simulation_defaults.rte_solver}")
        print(f"  mol_abs_param: {sea_ice_config.simulation_defaults.mol_abs_param}")
        print(f"  wavelength_nm: {sea_ice_config.simulation_defaults.wavelength_nm}")
        print(f"  output_columns: {sea_ice_config.simulation_defaults.output_columns}")
        print(f"  surface_temperature_k: {sea_ice_config.simulation_defaults.surface_temperature_k}")
    except Exception as e:
        print(f"Error creating configuration: {e}")

# Create a debug function that runs a simulation with full error info
def debug_pyradtran():
    # Skip this cell if LibRadtran is not available
    sea_ice_config = create_sea_ice_config()
    # Create a test datetime and coordinates
    test_time = pd.Timestamp('2022-04-01 10:00:00').to_pydatetime()  
    test_lat = 75.0  # Arctic location
    test_lon = 0.0   # Prime meridian
    
    print(f"Running manual simulation for {test_time} at lat={test_lat}, lon={test_lon}")
    
    try:
        # Set up detailed logging
        logging.getLogger('pyradtran').setLevel(logging.DEBUG)
        
        # Initialize the simulation runner with our config
        runner = Simulation(sea_ice_config)
        
        # Run the simulation directly
        output_file = runner.run(test_time, test_lat, test_lon)
        
        if output_file and output_file.exists():
            print(f"Simulation succeeded! Output file: {output_file}")
            
            # Parse the output
            result = parse_uvspec_output(output_file, sea_ice_config)
            
            # Display results
            print("\nSimulation Results:")
            for key, value in result.items():
                if not key.startswith('_') and not isinstance(value, dict):
                    if isinstance(value, list) and len(value) > 0:
                        print(f"{key}: {value[0]}")
                    else:
                        print(f"{key}: {value}")
        else:
            print("Simulation failed to produce output file")
            
    except Exception as e:
        print(f"Error in manual simulation: {e}")
        print(traceback.format_exc())
        
    except Exception as e:
        import traceback
        print(f"\nError in manual approach: {e}")
        print(traceback.format_exc())
    
    print("\n=== DIAGNOSTIC APPROACH 2: Use xarray accessor ===")

    ds = xr.Dataset(
        data_vars={
            'lat': (('time',), [test_lat]),
            'lon': (('time',), [test_lon]),
        },
        coords={
            'time': (('time',), [test_time]),
        },
        attrs={
            'description': 'Test dataset for PyRadtran',
            'history': 'Created for debugging purposes'
        }
    )
    
    # Print dataset structure for debugging
    print("\nTest Dataset Structure:")
    print(f"Dataset dimensions: {ds.dims}")
    print(f"Dataset coordinates: {list(ds.coords)}")
    print(f"Dataset data variables: {list(ds.data_vars)}")
    print(ds)
    
    try:
        # Use the xarray accessor
        print("\nUsing xarray accessor...")
        result_ds = ds.pyradtran.run_uvspec(
            parameter_overrides=sea_ice_config.simulation_defaults.to_dict(),
            return_dataset=True,
            save_to_file=False,
            lat_var='lat',  # Specify the correct variable names
            lon_var='lon'   # to match our dataset coordinates
        )
        
        print("\nResult dataset:")
        print(result_ds)
        
    except Exception as e:
        import traceback
        print(f"\nError in xarray accessor approach: {e}")
        print(traceback.format_exc())
    
    print("\n=== Debug complete ===")

if __name__ == "__main__":
    debug_pyradtran()