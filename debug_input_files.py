#!/usr/bin/env python
"""
Debug script to examine PyRadtran input file generation for various configurations.
This script will create input files for different combinations of altitude and 
wavelength settings and save them to text files for inspection.
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd

# Configure detailed logging
logging.basicConfig(level=logging.DEBUG, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Add parent directory to path if needed
module_path = os.path.abspath(os.path.dirname(__file__))
if module_path not in sys.path:
    sys.path.append(module_path)
    print(f"Added {module_path} to sys.path")

# Import PyRadtran modules
import pyradtran
from pyradtran.config import PathsConfig, SimulationDefaults, SimulationConfig, ExecutionConfig, OutputConfig
from pyradtran.core import generate_input_content

# Define paths
LIBRADTRAN_DATA_PATH = '/opt/libradtran/2.0.4/share/libRadtran/data'
LIBRADTRAN_EXEC_PATH = '/opt/libradtran/2.0.4/bin/uvspec'
ATMOSPHERE_FILE = '/projekt_agmwend/data/HALO-AC3/05_VELOX_Tools/add_data/afglsw.dat'
SOLAR_SPECTRUM_FILE = '/projekt_agmwend/home_rad/sophie/libradtran/solar_flux/NewGuey2003.dat'
RADIOSONDE_BASE_PATH = '/projekt_agmwend/data/HALO-AC3/01_soundings/RS_for_libradtran/Dropsondes_HALO/'
WORKING_DIR = os.path.join(module_path, 'work')
os.makedirs(WORKING_DIR, exist_ok=True)

# Constants
FIXED_OZONE_DU = 300.0
FIXED_IWV_MM = 2.0
FIXED_SURFACE_TEMP_K = 250.0
SEA_ICE_BRDF_TYPE = 20

def create_sea_ice_config():
    """Create a configuration for sea ice simulations"""
    return SimulationConfig(
        paths=PathsConfig(
            libradtran_bin=Path(LIBRADTRAN_EXEC_PATH),
            libradtran_data=Path(LIBRADTRAN_DATA_PATH),
            atmosphere_profile=Path(ATMOSPHERE_FILE),
            solar_spectrum=Path(SOLAR_SPECTRUM_FILE),
            radiosonde_base=Path(RADIOSONDE_BASE_PATH),
            output_dir=Path(WORKING_DIR),
            working_dir=Path(WORKING_DIR)
        ),
        simulation_defaults=SimulationDefaults(
            rte_solver='disort',
            mol_abs_param='lowtran per_nm',
            wavelength_nm=[400, 3600],
            output_columns=['sza', 'edir', 'eglo', 'edn', 'eup', 'enet', 'albedo'],
            output_altitudes_km=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            
            # Surface properties
            albedo_type='library',
            albedo_library='IGBP',
            brdf_type='rpv',
            brdf_rpv_type=SEA_ICE_BRDF_TYPE,
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
        execution=ExecutionConfig(
            max_workers=1,
            cleanup_temp_files=False,  # Keep temp files for debugging
            debug_mode=True,  # Enable debug mode
            timeout_seconds=300
        ),
        output=OutputConfig(
            filename_prefix="debug_output",
            filename_suffix="_results.nc",
            netcdf_encoding={"zlib": True, "complevel": 5}
        )
    )

def generate_test_case(name, altitudes, include_wavelength=False, integrate_wavelength=True):
    """Generate a test case configuration and input file"""
    print(f"\n{'='*80}")
    print(f"GENERATING TEST CASE: {name}")
    print(f"{'='*80}")
    
    # Create test configuration
    test_config = create_sea_ice_config()
    
    # Configure altitude settings
    test_config.simulation_defaults.output_altitudes_km = altitudes
    
    # Configure output columns
    output_cols = ['edir', 'eglo', 'edn', 'eup', 'enet', 'albedo', 'sza']
    
    # Add altitude column if using multiple altitudes
    if len(altitudes) > 1:
        output_cols.insert(0, 'zout')
        print(f"Added 'zout' to output columns for multiple altitudes: {len(altitudes)} levels")
    
    # Add wavelength column if requested
    if include_wavelength:
        output_cols.insert(0, 'lambda')
        print(f"Added 'lambda' to output columns for spectral output")
    
    test_config.simulation_defaults.output_columns = output_cols
    
    # CRITICAL: Clear any existing output_process settings to ensure proper ordering
    if hasattr(test_config.simulation_defaults, 'additional_options'):
        test_config.simulation_defaults.additional_options = [
            opt for opt in test_config.simulation_defaults.additional_options 
            if not opt.startswith('output')
        ]
    else:
        test_config.simulation_defaults.additional_options = []
    
    # Set wavelength integration - make sure these are added in the CORRECT ORDER
    # LibRadtran is extremely picky about ordering - output_user MUST be the last command
    if integrate_wavelength:
        # First add per_nm normalization
        test_config.simulation_defaults.additional_options.append("output_process per_nm")
        # Then add integrate command
        test_config.simulation_defaults.additional_options.append("output_process integrate")
        print("Wavelength integration enabled (output_process per_nm + output_process integrate)")
    else:
        # Just add per_nm normalization
        test_config.simulation_defaults.additional_options.append("output_process per_nm")
        print("Wavelength integration disabled (output_process per_nm only)")
    
    # Print configuration details
    print(f"\nConfiguration:")
    print(f"  Output columns: {test_config.simulation_defaults.output_columns}")
    print(f"  Output altitudes: {test_config.simulation_defaults.output_altitudes_km}")
    print(f"  Integrate wavelength: {test_config.simulation_defaults.integrate_wavelength}")
    print(f"  Wavelength range: {test_config.simulation_defaults.wavelength_nm}")
    print(f"  Additional options: {test_config.simulation_defaults.additional_options}")
    
    # Override generate_input_content to ensure proper ordering of commands
    from pyradtran.core import generate_input_content as original_generate
    
    def custom_generate_input(config, dt, latitude, longitude, **kwargs):
        """Generate input content with corrected order of output commands"""
        # Get the base input content but split into lines
        lines = original_generate(config, dt, latitude, longitude, **kwargs).split('\n')
        
        # Collect all directives by category
        core_lines = []
        output_zout_lines = []
        output_process_lines = []
        output_user_lines = []
        other_lines = []
        
        # Categorize lines
        for line in lines:
            if line.startswith('zout '):
                output_zout_lines.append(line)
            elif line.startswith('output_process '):
                output_process_lines.append(line)
            elif line.startswith('output_user '):
                output_user_lines.append(line)
            elif line.startswith('output'):
                # Any other output-related lines
                other_lines.append(line)
            else:
                core_lines.append(line)
        
        # Reassemble in correct order: core first, then zout, then output_process,
        # then other output directives, and finally output_user LAST
        ordered_lines = core_lines + output_zout_lines + output_process_lines + other_lines + output_user_lines
        
        return '\n'.join(ordered_lines)
    
    # Generate input file content using the custom generator
    test_time = pd.to_datetime('2022-04-01 10:00:00').to_pydatetime()
    test_lat = 75.0
    test_lon = 0.0
    
    input_content = custom_generate_input(
        config=test_config,
        dt=test_time,
        latitude=test_lat,
        longitude=test_lon
    )
    
    # Write to a file
    file_path = os.path.join(WORKING_DIR, f"debug_input_{name.lower().replace(' ', '_')}.inp")
    with open(file_path, 'w') as f:
        f.write(input_content)
    
    # Print the generated input content for inspection
    print(f"\n--- GENERATED INPUT FILE CONTENT ---\n")
    print(input_content)
    print(f"\n--- END OF INPUT FILE CONTENT ---\n")
    
    print(f"Input file written to: {file_path}")
    
    return {
        'name': name,
        'config': test_config,
        'input_file': file_path
    }

def main():
    """Main function to generate all test cases"""
    print("Generating test input files for PyRadtran debugging")
    
    # Generate all test cases
    test_cases = [
        generate_test_case(
            name="Single Altitude Integrated Wavelength", 
            altitudes=[0.0],
            include_wavelength=False,
            integrate_wavelength=True
        ),
        generate_test_case(
            name="Multi Altitude Integrated Wavelength", 
            altitudes=[0.0, 1.0, 2.0, 5.0, 10.0],
            include_wavelength=False,
            integrate_wavelength=True
        ),
        generate_test_case(
            name="Single Altitude Spectral Wavelength", 
            altitudes=[0.0],
            include_wavelength=True,
            integrate_wavelength=False
        ),
        generate_test_case(
            name="Multi Altitude Spectral Wavelength", 
            altitudes=[0.0, 1.0, 2.0, 5.0, 10.0],
            include_wavelength=True,
            integrate_wavelength=False
        )
    ]
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY OF TEST CASES")
    print("="*80)
    for case in test_cases:
        print(f"- {case['name']}: {case['input_file']}")
    
    print("\nPlease examine the input files in the work directory")

if __name__ == "__main__":
    main()