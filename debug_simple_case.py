import os
import sys
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd

# Configure simple logging for the script
logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
script_logger = logging.getLogger(__name__)

# Add parent directory to path if needed
module_path = os.path.abspath(os.path.join('.')) # Assuming script is in pyradtran main dir
if module_path not in sys.path:
    sys.path.append(module_path)
    script_logger.info(f"Added {module_path} to sys.path")

try:
    from pyradtran.config import PathsConfig, SimulationDefaults, SimulationConfig, ExecutionConfig, OutputConfig
    from pyradtran.core import Simulation, generate_input_content 
    from pyradtran.io import parse_uvspec_output
except ImportError as e:
    script_logger.error(f"Failed to import PyRadtran modules: {e}")
    script_logger.error("Please ensure the script is in the root of the pyradtran project or the PYTHONPATH is set correctly.")
    sys.exit(1)

# Define paths 
LIBRADTRAN_DATA_PATH = '/opt/libradtran/2.0.4/share/libRadtran/data'
LIBRADTRAN_EXEC_PATH = '/opt/libradtran/2.0.4/bin/uvspec'
ATMOSPHERE_FILE = '/projekt_agmwend/data/HALO-AC3/05_VELOX_Tools/add_data/afglsw.dat'
SOLAR_SPECTRUM_FILE = '/projekt_agmwend/home_rad/sophie/libradtran/solar_flux/NewGuey2003.dat'
WORKING_DIR = os.path.join(module_path, 'work_debug_simple') 
os.makedirs(WORKING_DIR, exist_ok=True)

script_logger.info(f"Using working directory: {WORKING_DIR}")

def create_simple_debug_config():
    """Create a very simple configuration for debugging."""
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
            mol_abs_param='lowtran per_nm',
            wavelength_nm=[400, 700], 
            output_columns=['sza', 'eglo', 'eup'], 
            output_altitudes_km=[0.0], 
            albedo_type='const',
            albedo_value=0.3, 
            mol_modify={'O3': {'value': 300.0, 'unit': 'DU'}, 'H2O': {'value': 2.0, 'unit': 'MM'}},
            aerosols={'enabled': False},
            clouds={'enabled': False},
            integrate_wavelength=True  # Ensure this is set to True for integration
        ),
        execution=ExecutionConfig(
            max_workers=1,
            cleanup_temp_files=False, 
            debug_mode=True,          
            timeout_seconds=60
        ),
        output=OutputConfig(
            filename_prefix="debug_simple_run",
            filename_suffix="_results.out", 
            netcdf_encoding=None 
        )
    )
    
    # CRITICAL: Add additional_options as an attribute after creation
    # First create an empty list if it doesn't exist
    if not hasattr(config.simulation_defaults, 'additional_options'):
        config.simulation_defaults.additional_options = []
    
    # First add per_nm normalization
    config.simulation_defaults.additional_options.append("output_process per_nm")
    # Then add integrate command if wavelength integration is enabled
    if config.simulation_defaults.integrate_wavelength:
        config.simulation_defaults.additional_options.append("output_process integrate")
        script_logger.info("Wavelength integration enabled (output_process per_nm + output_process integrate)")
    else:
        script_logger.info("Wavelength integration disabled (output_process per_nm only)")
    
    return config

def main():
    script_logger.info("Starting simple debug simulation script.")

    debug_config = create_simple_debug_config()
    script_logger.debug(f"Debug Configuration: {debug_config}")
    
    # Log the additional options to verify they are set correctly
    script_logger.info(f"Additional options: {debug_config.simulation_defaults.additional_options}")

    runner = Simulation(debug_config)
    script_logger.info("Simulation runner instantiated.")

    test_dt = datetime(2022, 4, 1, 12, 0, 0) 
    test_latitude = 75.0  
    test_longitude = 0.0   

    script_logger.info(f"Test point: dt={test_dt}, lat={test_latitude}, lon={test_longitude}")

    script_logger.info("--- STEP 1: Generating input file content ---")
    try:
        input_content_generated = generate_input_content(
            config=debug_config,
            dt=test_dt,
            latitude=test_latitude,
            longitude=test_longitude
        )
        script_logger.info("Input content generated successfully.")
        script_logger.info(f"--- Generated UVSPEC Input Content ---\n{input_content_generated}\n--- End of Input Content ---")
        
        # Save the generated input content to a file for inspection
        input_debug_path = os.path.join(WORKING_DIR, "debug_content_direct.inp")
        with open(input_debug_path, 'w') as f:
            f.write(input_content_generated)
        script_logger.info(f"Saved direct input content to {input_debug_path}")
    except Exception as e:
        script_logger.error(f"Error during generate_input_content: {e}", exc_info=True)
        return

    script_logger.info("--- STEP 2: Calling runner.run() ---")
    output_file_path = None
    try:
        output_file_path = runner.run(
            dt=test_dt,
            latitude=test_latitude,
            longitude=test_longitude
        )

        if output_file_path and output_file_path.exists():
            script_logger.info(f"SUCCESS: Simulation run completed. Output file: {output_file_path}")
            script_logger.info("--- STEP 3: Examining output file ---")
            try:
                with open(output_file_path, 'r') as f:
                    output_content = f.read()
                script_logger.info(f"--- UVSPEC Output File Content ({output_file_path.name}) ---\n{output_content}\n--- End of Output Content ---")
                
                parsed_output = parse_uvspec_output(output_file_path, debug_config)
                script_logger.info(f"Parsed output: {parsed_output}")

            except Exception as e:
                script_logger.error(f"Error reading or parsing output file {output_file_path}: {e}", exc_info=True)
        elif output_file_path:
            script_logger.error(f"FAILURE: Simulation run completed but output file does not exist: {output_file_path}")
        else:
            script_logger.error("FAILURE: Simulation run did not return an output file path.")

    except Exception as e:
        script_logger.error(f"Error during Simulation.run(): {e}", exc_info=True)

    script_logger.info("--- STEP 4: Verifying files in working directory ---")
    script_logger.info(f"Listing files in {WORKING_DIR}:")
    for item in os.listdir(WORKING_DIR):
        script_logger.info(f"  - {item}")
        if item.endswith(".inp") or item.endswith(".out"):
            item_path = os.path.join(WORKING_DIR, item)
            try:
                with open(item_path, 'r') as f_content:
                    content = f_content.read()
                    script_logger.info(f"Content of {item}:\n{content}\n--------------------")
            except Exception as e_read:
                script_logger.error(f"Could not read {item_path}: {e_read}")

    script_logger.info("Simple debug simulation script finished.")

if __name__ == "__main__":
    main()