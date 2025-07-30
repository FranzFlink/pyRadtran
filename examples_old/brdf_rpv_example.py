#!/usr/bin/env python3
"""
Minimal working example demonstrating BRDF functionality in PyRadtran.

This example focuses specifically on the RPV (Rahman, Pinty, Verstraete) BRDF model
for land surfaces like sea ice and snow, with detailed debugging information.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta

# Configure logging to show detailed debug information
logging.basicConfig(level=logging.DEBUG, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Add parent directory to path if running script directly
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
    print(f"Added {parent_dir} to sys.path")

# Import PyRadtran modules
from pyradtran.config import (
    SimulationConfig, PathsConfig, SimulationDefaults, 
    ExecutionConfig, OutputConfig, BRDFRpvParameters, load_config
)
from pyradtran.core import Simulation
from pyradtran.io_old import parse_uvspec_output

# Define paths to LibRadtran and data files (adjust as needed)
LIBRADTRAN_DATA_PATH = '/opt/libradtran/2.0.4/share/libRadtran/data'
LIBRADTRAN_EXEC_PATH = '/opt/libradtran/2.0.4/bin/uvspec'
ATMOSPHERE_FILE = '/projekt_agmwend/data/HALO-AC3/05_VELOX_Tools/add_data/afglsw.dat'
SOLAR_SPECTRUM_FILE = '/projekt_agmwend/home_rad/sophie/libradtran/solar_flux/NewGuey2003.dat'
OUTPUT_DIR = './output'

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

def create_base_config():
    """Create a basic configuration for BRDF simulations."""
    return SimulationConfig(
        paths=PathsConfig(
            libradtran_bin=Path(LIBRADTRAN_EXEC_PATH),
            libradtran_data=Path(LIBRADTRAN_DATA_PATH),
            atmosphere_profile=Path(ATMOSPHERE_FILE),
            solar_spectrum=Path(SOLAR_SPECTRUM_FILE),
            output_dir=Path(OUTPUT_DIR),
            working_dir=Path(OUTPUT_DIR)
        ),
        simulation_defaults=SimulationDefaults(
            rte_solver='disort',  # Use disort for accuracy with BRDF
            mol_abs_param='reptran coarse',  # Faster for testing
            wavelength_nm=[400, 2400],  # Visible to near-IR
            output_columns=['sza', 'edir', 'eglo', 'edn', 'eup', 'enet', 'albedo'],
            output_altitudes_km=[0.0],  # Surface level only
            
            # Basic atmospheric settings
            h2o_mm=2.0,
            h2o_source='fixed',
            ozone_du=300.0,
            
            # Disable clouds and aerosols for simplicity
            clouds={'enabled': False},
            aerosols={'enabled': False},
            
            # Set default surface temperature
            surface_temperature_k=273.15
        ),
        execution=ExecutionConfig(
            max_workers=1,  # Single worker for debugging
            cleanup_temp_files=False,  # Keep temp files for inspection
            debug_mode=True,  # Enable debug mode
            timeout_seconds=30  # Short timeout for quick testing
        ),
        output=OutputConfig(
            filename_prefix="brdf_example"
        )
    )

def create_test_dataset():
    """Create a test dataset with a single point at solar noon."""
    noon_time = pd.Timestamp('2025-05-05 12:00:00')
    return xr.Dataset(
        coords={
            'time': [noon_time],
            'latitude': [75.0],  # Arctic location
            'longitude': [0.0]   # Prime meridian
        }
    )

def run_rpv_comparison():
    """Run comparison between different RPV BRDF settings."""
    # Create base configuration
    config = create_base_config()
    
    # Create test dataset
    ds = create_test_dataset()
    
    # Define RPV BRDF scenarios to test
    rpv_scenarios = {
        'Lambertian Snow': {
            'description': 'Simple constant albedo snow surface (no BRDF)',
            'config': lambda cfg: setattr(cfg.simulation_defaults, 'albedo_value', 0.85) or cfg
        },
        'RPV Sea Ice': {
            'description': 'Sea ice with RPV type 20',
            'config': lambda cfg: (
                setattr(cfg.simulation_defaults.brdf_rpv, 'enabled', True),
                setattr(cfg.simulation_defaults.brdf_rpv, 'rpv_type', 20),
                cfg  # Return the modified config
            )[2]  # Return the last item from the tuple
        },
        'RPV Custom Snow': {
            'description': 'Snow with custom RPV parameters',
            'config': lambda cfg: (
                setattr(cfg.simulation_defaults.brdf_rpv, 'enabled', True),
                setattr(cfg.simulation_defaults.brdf_rpv, 'k', 0.7),
                setattr(cfg.simulation_defaults.brdf_rpv, 'rho0', 0.9),
                setattr(cfg.simulation_defaults.brdf_rpv, 'theta', 0.1),
                setattr(cfg.simulation_defaults.brdf_rpv, 'sigma', 0.1),
                cfg  # Return the modified config
            )[5]  # Return the last item from the tuple
        }
    }
    
    # Run simulations for different viewing angles
    viewing_angles = [-60, -30, 0, 30, 60]  # Negative = toward sun, Positive = away from sun
    results = {scenario: {} for scenario in rpv_scenarios}
    
    # Create runners for each scenario
    for scenario_name, scenario_info in rpv_scenarios.items():
        print(f"\n=== Testing {scenario_name}: {scenario_info['description']} ===")
        
        # Apply scenario-specific configuration
        scenario_config = scenario_info['config'](config)
        
        for angle in viewing_angles:
            print(f"\n--- Viewing angle: {angle}° ---")
            
            # Set viewing angle (mu = cos(angle))
            umu = np.cos(np.radians(angle))
            scenario_config.simulation_defaults.viewing_geometry = 'custom'
            scenario_config.simulation_defaults.umu = [umu]
            scenario_config.simulation_defaults.phi = [0.0]  # Principal plane
            
            # Create simulation runner
            runner = Simulation(scenario_config)
            
            # Get the timestamp from our dataset
            dt = ds.time.values[0].astype('datetime64[s]').item()
            lat = float(ds.latitude.values[0])
            lon = float(ds.longitude.values[0])
            
            # Run simulation
            print(f"Running simulation for {dt} at lat={lat}, lon={lon} with viewing angle={angle}°")
            try:
                output_file = runner.run(
                    dt=dt,
                    latitude=lat, 
                    longitude=lon
                )
                
                if output_file and output_file.exists():
                    # Parse the output
                    result = parse_uvspec_output(output_file, scenario_config)
                    
                    # Store results
                    results[scenario_name][angle] = {
                        'albedo': result['albedo'][0],
                        'eglo': result['eglo'][0],
                        'edir': result['edir'][0],
                        'eup': result['eup'][0]
                    }
                    
                    print(f"Results: Albedo={result['albedo'][0]:.3f}, "
                          f"Global={result['eglo'][0]:.2f} W/m², "
                          f"Direct={result['edir'][0]:.2f} W/m², "
                          f"Up={result['eup'][0]:.2f} W/m²")
                else:
                    print(f"Error: Simulation failed to produce output file")
                    results[scenario_name][angle] = None
            except Exception as e:
                print(f"Error running simulation: {e}")
                results[scenario_name][angle] = None
    
    return results

def plot_results(results):
    """Plot BRDF comparison results."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot albedo vs. viewing angle
    for scenario, angle_results in results.items():
        angles = []
        albedos = []
        for angle, values in angle_results.items():
            if values is not None:
                angles.append(angle)
                albedos.append(values['albedo'])
        ax1.plot(angles, albedos, 'o-', label=scenario)
    
    ax1.set_xlabel('Viewing Angle (degrees)')
    ax1.set_ylabel('Effective Albedo')
    ax1.set_title('BRDF Effect on Effective Albedo')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best')
    
    # Prepare data for comparing scenarios at nadir view
    scenarios = []
    nadir_albedos = []
    nadir_eglos = []
    nadir_eups = []
    
    for scenario, angle_results in results.items():
        if 0 in angle_results and angle_results[0] is not None:
            scenarios.append(scenario)
            nadir_albedos.append(angle_results[0]['albedo'])
            nadir_eglos.append(angle_results[0]['eglo'])
            nadir_eups.append(angle_results[0]['eup'])
    
    x = np.arange(len(scenarios))
    width = 0.35
    
    # Plot nadir-view comparison
    ax2.bar(x, nadir_eglos, width, label='Global', color='blue', alpha=0.7)
    ax2.bar(x + width, nadir_eups, width, label='Upward', color='green', alpha=0.7)
    
    # Add albedo values as text
    for i, albedo in enumerate(nadir_albedos):
        ax2.text(i + width/2, nadir_eglos[i] + 5, f"α={albedo:.2f}", ha='center')
    
    ax2.set_ylabel('Irradiance (W/m²)')
    ax2.set_title('Nadir-View Comparison of BRDF Models')
    ax2.set_xticks(x + width/2)
    ax2.set_xticklabels(scenarios, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'brdf_rpv_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    plt.show()

def display_brdf_analysis(results):
    """Display detailed analysis of BRDF effects."""
    print("\n=== BRDF Effect Analysis ===")
    print("-" * 70)
    
    for scenario in results:
        if all(angle in results[scenario] and results[scenario][angle] for angle in [-60, 0, 60]):
            backward = results[scenario][-60]['albedo']
            nadir = results[scenario][0]['albedo']
            forward = results[scenario][60]['albedo']
            
            print(f"{scenario}:")
            print(f"  Backward scatter (view -60°): albedo = {backward:.3f}")
            print(f"  Nadir view (view 0°): albedo = {nadir:.3f}")
            print(f"  Forward scatter (view 60°): albedo = {forward:.3f}")
            print(f"  Directionality: {(backward-forward)/nadir:.2f}")
            print()

def main():
    """Main function to run the BRDF example."""
    print("=== PyRadtran BRDF Example ===")
    print("This example demonstrates the Rahman, Pinty, and Verstraete (RPV) BRDF model")
    print("for different surface types and viewing angles.\n")
    
    # Check if LibRadtran is available
    if not os.path.isfile(LIBRADTRAN_EXEC_PATH) or not os.path.isdir(LIBRADTRAN_DATA_PATH):
        print("Error: LibRadtran not found at the specified paths.")
        print(f"LIBRADTRAN_EXEC_PATH: {LIBRADTRAN_EXEC_PATH}")
        print(f"LIBRADTRAN_DATA_PATH: {LIBRADTRAN_DATA_PATH}")
        return 1
    
    try:
        # Run the BRDF comparison
        results = run_rpv_comparison()
        
        # Plot the results
        plot_results(results)
        
        # Display detailed analysis
        display_brdf_analysis(results)
        
        print("\nExample completed successfully!")
        return 0
    except Exception as e:
        print(f"Error running example: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())