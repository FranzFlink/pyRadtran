#!/usr/bin/env python3
"""
Sea Ice Solar Simulation Example with PyRadtran

This example demonstrates how to use the PyRadtran package to perform sea ice
solar simulations similar to the original disort.py script, but with a cleaner,
more modular approach.

Usage:
  python sea_ice_simulation.py <input_file.nc> [--alt ALTITUDE_KM] [--debug]
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from datetime import datetime
import tempfile
import logging

# Add parent directory to path if running from examples directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pyradtran
from pyradtran.config import (
    PathsConfig,
    SimulationDefaults,
    SimulationConfig,
    load_config
)
from pyradtran.exceptions import PyRadtranError
from pyradtran.utils import RadiosondeFinder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger('sea_ice_simulation')

# --- Configuration Constants ---
# Use the same paths as in disort.py
LIBRADTRAN_DATA_PATH = '/opt/libradtran/2.0.4/share/libRadtran/data'
LIBRADTRAN_EXEC_PATH = '/opt/libradtran/2.0.4/bin/uvspec'
ATMOSPHERE_FILE = '/projekt_agmwend/data/HALO-AC3/05_VELOX_Tools/add_data/afglsw.dat'
SOLAR_SPECTRUM_FILE = '/projekt_agmwend/home_rad/sophie/libradtran/solar_flux/NewGuey2003.dat'
RADIOSONDE_BASE_PATH = '/projekt_agmwend/data/HALO-AC3/01_soundings/RS_for_libradtran/Dropsondes_HALO/'
SIMULATION_OUTPUT_DIR = '/projekt_agmwend/home_rad/Joshua/HALO-AC3_Arctic_leads/data/simulation/disort/'

# Sea Ice simulation constants
FIXED_OZONE_DU = 300.0  # Fixed total column ozone in Dobson Units
FIXED_IWV_MM = 2.0      # Fixed total column water vapor (precipitable water) in mm
FIXED_SURFACE_TEMP_K = 250.0  # Fixed surface temperature for sea ice conditions (Kelvin)
SEA_ICE_BRDF_TYPE = 20  # RPV BRDF type for sea ice

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
            working_dir=Path(tempfile.gettempdir())
        ),
        simulation_defaults=SimulationDefaults(
            rte_solver='twostr',  # Same as disort.py
            mol_abs_param='lowtran per_nm',  # Same as disort.py
            wavelength_nm=[400, 3600],  # Same range as disort.py
            output_columns=['sza', 'edir', 'eglo', 'edn', 'eup', 'enet', 'esum', 'albedo'],
            output_altitudes_km=[0.0],  # Surface level
            
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
    )

def load_input_data(filepath):
    """Load time, lat, lon, alt data from NetCDF or CSV file."""
    logger.info(f"Loading input data from: {filepath}")
    
    if filepath.lower().endswith('.nc'):
        try:
            ds = xr.open_dataset(filepath)
            
            # Find time coordinate/variable
            time_vars = ['time', 'Time', 'datetime', 'timestamp', 'timestamps']
            time_coord = next((var for var in time_vars if var in ds.coords or var in ds.data_vars), None)
            if not time_coord:
                raise ValueError(f"Time coordinate/variable not found in {list(ds.coords.keys()) + list(ds.data_vars.keys())}.")
            
            # Find latitude coordinate/variable
            lat_vars = ['lat', 'latitude', 'Lat', 'Latitude']
            lat_coord = next((var for var in lat_vars if var in ds.coords or var in ds.data_vars), None)
            if not lat_coord:
                raise ValueError(f"Latitude coordinate/variable not found in {list(ds.coords.keys()) + list(ds.data_vars.keys())}.")
            
            # Find longitude coordinate/variable
            lon_vars = ['lon', 'longitude', 'Lon', 'Longitude']
            lon_coord = next((var for var in lon_vars if var in ds.coords or var in ds.data_vars), None)
            if not lon_coord:
                raise ValueError(f"Longitude coordinate/variable not found in {list(ds.coords.keys()) + list(ds.data_vars.keys())}.")
            
            # Find altitude coordinate/variable (in meters)
            alt_vars = ['alt', 'altitude', 'Alt', 'Altitude', 'z', 'height', 'Height', 'geopotential_height']
            alt_coord = next((var for var in alt_vars if var in ds.coords or var in ds.data_vars), None)
            if not alt_coord:
                logger.warning(f"Altitude coordinate/variable not found, altitude will need to be set manually.")
                alts_m = None
            else:
                alts_m = ds[alt_coord].values
            
            # Create dataset for PyRadtran
            if alt_coord is not None:
                # If altitude is found, ensure it's correctly included
                if time_coord in ds.dims:
                    # Time is a dimension, so ensure latitude and longitude align
                    result_ds = xr.Dataset(
                        coords={
                            'time': ds[time_coord],
                            'latitude': (('time',), ds[lat_coord].values),
                            'longitude': (('time',), ds[lon_coord].values),
                            'altitude': (('time',), alts_m / 1000.0)  # Convert meters to km
                        }
                    )
                else:
                    # Time is not a dimension, need to check other coordinates
                    result_ds = xr.Dataset(
                        coords={
                            'time': ds[time_coord].values,
                            'latitude': ds[lat_coord].values,
                            'longitude': ds[lon_coord].values,
                            'altitude': alts_m / 1000.0  # Convert meters to km
                        }
                    )
            else:
                # No altitude found
                if time_coord in ds.dims:
                    result_ds = xr.Dataset(
                        coords={
                            'time': ds[time_coord],
                            'latitude': (('time',), ds[lat_coord].values),
                            'longitude': (('time',), ds[lon_coord].values)
                        }
                    )
                else:
                    result_ds = xr.Dataset(
                        coords={
                            'time': ds[time_coord].values,
                            'latitude': ds[lat_coord].values,
                            'longitude': ds[lon_coord].values
                        }
                    )
            
            logger.info(f"Successfully loaded {len(result_ds.time)} time points from NetCDF file.")
            return result_ds
            
        except Exception as e:
            logger.error(f"Error loading NetCDF file: {e}")
            raise
    
    elif filepath.lower().endswith('.csv'):
        try:
            # Try to auto-detect time column
            time_cols = ['time', 'Time', 'datetime', 'timestamp', 'timestamps']
            # Read first few rows to find time column
            df_peek = pd.read_csv(filepath, nrows=5)
            time_col_name = next((col for col in time_cols if col in df_peek.columns), None)
            
            if not time_col_name:
                logger.warning(f"Common time column names not found in CSV header. Using first column as time.")
                # Fallback to first column
                time_col_name = df_peek.columns[0]
            
            # Read the data
            df = pd.read_csv(filepath, parse_dates=[time_col_name])
            
            # Standardize column names (case-insensitive) for lat/lon/alt lookup
            df.columns = df.columns.str.lower()
            
            # Find column names
            lat_col = next((col for col in df.columns if col in ['lat', 'latitude']), None)
            lon_col = next((col for col in df.columns if col in ['lon', 'longitude']), None)
            alt_col = next((col for col in df.columns if col in ['alt', 'altitude', 'z', 'height']), None)
            
            if not all([lat_col, lon_col]):
                missing = [name for name, col in zip(['latitude', 'longitude'], [lat_col, lon_col]) if col is None]
                raise ValueError(f"CSV missing required columns: {missing}")
            
            # Ensure time column name is also lower case for lookup
            time_col_lookup = time_col_name.lower()
            
            # Create xarray dataset
            coords = {
                'time': df[time_col_lookup].values,
                'latitude': df[lat_col].values,
                'longitude': df[lon_col].values
            }
            
            # Add altitude if available
            if alt_col:
                coords['altitude'] = df[alt_col].values / 1000.0  # Convert meters to km
            
            result_ds = xr.Dataset(coords=coords)
            logger.info(f"Successfully loaded {len(result_ds.time)} time points from CSV file.")
            return result_ds
            
        except Exception as e:
            logger.error(f"Error loading CSV file: {e}")
            raise
    
    else:
        raise ValueError(f"Unsupported input file format: {filepath}. Please use .nc or .csv")

def find_radiosondes_for_dataset(ds, radiosonde_base_path):
    """
    Find matching radiosondes for each time point in the dataset.
    
    Args:
        ds: xarray dataset with time, latitude, longitude coordinates
        radiosonde_base_path: path to radiosonde directory
    
    Returns:
        List of radiosonde paths (may contain None for time points with no match)
    """
    finder = RadiosondeFinder(radiosonde_base_path)
    radiosonde_paths = []
    
    for i, timestamp in enumerate(ds.time.values):
        # Convert to pandas Timestamp for easier date handling
        dt = pd.Timestamp(timestamp)
        
        # Create date string in the format used for radiosondes
        date_str = dt.strftime('%Y%m%d')
        
        # Try to find a matching radiosonde
        radiosonde_path = finder.find_closest(
            timestamp=dt,
            date_str=date_str,
            max_time_diff_hours=3  # Allow up to 3 hours difference
        )
        
        if radiosonde_path:
            logger.debug(f"Found radiosonde for time point {i}: {radiosonde_path}")
        else:
            logger.debug(f"No matching radiosonde found for time point {i}")
        
        radiosonde_paths.append(radiosonde_path)
    
    return radiosonde_paths

def run_sea_ice_simulation(input_filepath, altitude_override_km=None, debug_mode=False):
    """
    Run sea ice simulation with PyRadtran.
    
    Args:
        input_filepath: Path to input NetCDF or CSV file
        altitude_override_km: Fixed altitude to use (in km)
        debug_mode: Print debug information
    """
    if debug_mode:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled")
    
    logger.info("=== Starting Sea Ice Simulation ===")
    logger.info(f"Using input file: {input_filepath}")
    logger.info(f"Fixed ozone: {FIXED_OZONE_DU} DU")
    logger.info(f"Fixed IWV: {FIXED_IWV_MM} mm")
    logger.info(f"Fixed surface temperature: {FIXED_SURFACE_TEMP_K} K")
    logger.info(f"BRDF type: {SEA_ICE_BRDF_TYPE} (Sea Ice)")
    
    # Create sea ice configuration
    config = create_sea_ice_config()
    
    # Load input data
    try:
        ds = load_input_data(input_filepath)
    except Exception as e:
        logger.error(f"Failed to load input data: {e}")
        return
    
    # Override altitude if requested
    if altitude_override_km is not None:
        logger.info(f"Overriding altitude to {altitude_override_km} km")
        ds = ds.assign_coords(altitude=('time', np.full(len(ds.time), altitude_override_km)))
        alt_str_suffix = f"{altitude_override_km}km"
    elif 'altitude' not in ds.coords:
        logger.error("No altitude found in input file and no override provided")
        return
    else:
        alt_str_suffix = "InputAlt"
    
    # Infer date from first timestamp for output file naming
    first_timestamp = pd.Timestamp(ds.time.values[0])
    fnum = first_timestamp.strftime('%Y%m%d')  # YYYYMMDD format
    date_str = first_timestamp.strftime('%Y-%m-%d')  # YYYY-MM-DD format
    logger.info(f"Inferred date (for radiosondes/output): {date_str}")
    
    # Define output file path
    output_filename = f'PyRadtran_SeaIceSim_{fnum}_{alt_str_suffix}.nc'
    output_path = os.path.join(SIMULATION_OUTPUT_DIR, output_filename)
    
    # Find matching radiosondes
    logger.info("Looking for matching radiosondes...")
    radiosonde_paths = find_radiosondes_for_dataset(ds, RADIOSONDE_BASE_PATH)
    logger.info(f"Found {sum(1 for p in radiosonde_paths if p is not None)} matching radiosondes")
    
    # Run simulation
    logger.info(f"Running simulation with {len(ds.time)} time points...")
    try:
        # Pass radiosonde paths as parameter overrides
        parameter_overrides = []
        for i, rs_path in enumerate(radiosonde_paths):
            if rs_path:
                parameter_overrides.append({'index': i, 'radiosonde_path': rs_path})
        
        # Run the simulation with xarray accessor
        result = ds.pyradtran.run_uvspec(
            config=config,
            parameter_overrides=parameter_overrides,
            output_path=output_path,
            return_dataset=True,
            save_to_file=True
        )
        
        # Add metadata
        result.attrs.update({
            'title': 'Libradtran Sea Ice Solar Simulation Results (PyRadtran)',
            'description': (f"Integrated downward solar radiance (400-3600 nm) simulated using libradtran (uvspec) "
                           f"for sea ice conditions. Date ({date_str}) inferred from input data. "
                           f"Uses fixed ozone ({FIXED_OZONE_DU} DU), fixed IWV ({FIXED_IWV_MM} mm), "
                           f"fixed surface temperature ({FIXED_SURFACE_TEMP_K} K), sea ice BRDF ({SEA_ICE_BRDF_TYPE}), "
                           f"and atmospheric profiles from closest HALO-AC3 dropsondes."),
            'input_file': input_filepath,
            'inferred_simulation_date': date_str,
            'creation_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S UTC'),
            
            # LibRadtran settings
            'libradtran_version_used': '2.0.4',
            'rte_solver': config.simulation_defaults.rte_solver,
            'mol_abs_param': config.simulation_defaults.mol_abs_param,
            'wavelength_range_nm': f"{config.simulation_defaults.wavelength_nm[0]} - {config.simulation_defaults.wavelength_nm[1]}",
            'fixed_ozone_DU': FIXED_OZONE_DU,
            'fixed_iwv_mm': FIXED_IWV_MM,
            'fixed_surface_temp_K': FIXED_SURFACE_TEMP_K,
            'surface_brdf': f'RPV type {SEA_ICE_BRDF_TYPE} (Sea Ice)',
        })
        
        # Save result
        result.to_netcdf(output_path, mode='w')
        logger.info(f"Results saved to: {output_path}")
        
        return result
        
    except PyRadtranError as e:
        logger.error(f"PyRadtran error: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    
    logger.info("=== Simulation Finished ===")

def main():
    """Parse command line arguments and run simulation."""
    parser = argparse.ArgumentParser(description="Run PyRadtran sea ice solar simulations.")
    parser.add_argument("input_file", type=str, help="Path to input NetCDF or CSV file")
    parser.add_argument("-a", "--altitude", type=float, default=None, help="Optional altitude override in km")
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    try:
        run_sea_ice_simulation(
            input_filepath=args.input_file,
            altitude_override_km=args.altitude,
            debug_mode=args.debug
        )
    except Exception as e:
        logger.error(f"Error running simulation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()