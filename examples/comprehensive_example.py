#!/usr/bin/env python
"""
Comprehensive PyRadtran Example

This example demonstrates the key features of the pyradtran package:
1. Loading and customizing configuration
2. Using helper functions for common scenarios
3. Running simulations with xarray integration
4. Analyzing and visualizing results
"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

# Add parent directory to path to import pyradtran (when running from examples directory)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pyradtran
from pyradtran import (
    load_config,
    configure_common_scenario,
    configure_spectral_range,
    configure_output_altitudes,
    configure_surface,
    configure_cloud
)

# Create output directories
EXAMPLE_DIR = Path(__file__).parent
OUTPUT_DIR = EXAMPLE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

def example_1_basic_simulation():
    """Basic simulation demonstrating core functionality."""
    print("\n=== Example 1: Basic Simulation ===")
    
    # Load default config and customize with helpers
    config = load_config()
    config = configure_common_scenario(config, "arctic_clear")
    config = configure_spectral_range(config, "broadband", integrate=True)
    
    # Create a simple dataset with time and location
    times = pd.date_range("2023-05-01", periods=24, freq="1H")
    coords = {
        "time": times,
        "latitude": ("time", np.full(len(times), 75.0)),  # Arctic location
        "longitude": ("time", np.full(len(times), 0.0))  # Prime meridian
    }
    ds = xr.Dataset(coords=coords)
    
    print("Running simulation for 24-hour time series in the Arctic...")
    
    # Run the simulation using xarray accessor
    results = ds.pyradtran.run_uvspec(
        return_dataset=True,
        save_to_file=True,
        output_path=OUTPUT_DIR / "example1_arctic_clear.nc"
    )
    
    # Plot results
    fig, ax = plt.subplots(figsize=(10, 6))
    results.eglo.plot(ax=ax, label='Global irradiance')
    results.eup.plot(ax=ax, label='Upward irradiance')
    results.sza.plot(ax=ax, linestyle='--', label='Solar zenith angle')
    
    ax.set_title("Diurnal Cycle of Irradiance in Arctic Conditions")
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Irradiance (W/m²)")
    ax.legend()
    
    fig.savefig(OUTPUT_DIR / "example1_arctic_diurnal.png")
    print(f"Saved plot to {OUTPUT_DIR / 'example1_arctic_diurnal.png'}")
    
    return results

def example_2_parameter_study():
    """Parameter study with multiple albedo values."""
    print("\n=== Example 2: Parameter Study (Albedo) ===")
    
    # Load and configure base simulation
    config = load_config()
    config = configure_spectral_range(config, "visible")
    config = configure_output_altitudes(config, "surface_only")  
    
    # Create dataset with multiple albedo values
    albedo_values = np.linspace(0.0, 1.0, 11)  # 0.0, 0.1, ..., 1.0
    
    # Single time point (noon) to focus on albedo effect
    noon_time = pd.to_datetime("2023-05-01 12:00:00")
    
    # Run simulations for each albedo
    all_results = []
    
    for albedo in albedo_values:
        # Update config for this albedo
        current_config = configure_surface(config, "custom", albedo_value=albedo)
        
        # Create dataset for this simulation
        ds = xr.Dataset(
            coords={
                "time": [noon_time],
                "latitude": 45.0,  # Mid-latitude
                "longitude": 0.0,
                "albedo": albedo  # Store albedo as coordinate for reference
            }
        )
        
        # Run simulation
        print(f"Running simulation with albedo = {albedo:.1f}")
        result = ds.pyradtran.run_uvspec(
            config_path=None,  # Use the config object we already created
            return_dataset=True,
            save_to_file=False  # Don't save individual runs
        )
        all_results.append(result)
    
    # Combine results along albedo dimension
    combined = xr.concat(all_results, dim="albedo")
    
    # Save combined results
    combined.to_netcdf(OUTPUT_DIR / "example2_albedo_study.nc")
    
    # Plot relationship between albedo and irradiance
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(combined.albedo, combined.eglo, 'o-', label='Global irradiance')
    ax.plot(combined.albedo, combined.eup, 'o-', label='Upward irradiance')
    
    ax.set_title("Effect of Surface Albedo on Irradiance")
    ax.set_xlabel("Albedo")
    ax.set_ylabel("Irradiance (W/m²)")
    ax.set_xlim(0, 1)
    ax.grid(True)
    ax.legend()
    
    fig.savefig(OUTPUT_DIR / "example2_albedo_study.png")
    print(f"Saved plot to {OUTPUT_DIR / 'example2_albedo_study.png'}")
    
    return combined

def example_3_cloud_study():
    """Study effects of different cloud types."""
    print("\n=== Example 3: Cloud Study ===")
    
    # Base configuration
    config = load_config()
    config = configure_spectral_range(config, "visible", integrate=True)
    config = configure_surface(config, "ocean")  # Ocean surface
    
    # Create dataset with solar zenith angles
    sza_values = np.arange(0, 85, 5)  # 0°, 5°, ..., 80°
    
    # Reference time (not important since we're varying SZA directly)
    reference_time = pd.to_datetime("2023-05-01 12:00:00")
    
    # Cloud scenarios to compare
    cloud_scenarios = ["clear", "stratus", "cumulus", "cirrus"]
    
    # Run simulations for each cloud type and SZA
    all_scenario_results = {}
    
    for cloud_type in cloud_scenarios:
        # Configure for this cloud type
        print(f"Simulating {cloud_type} clouds...")
        current_config = configure_cloud(config, cloud_type)
        
        # Results for this scenario
        scenario_results = []
        
        for sza in sza_values:
            # Create dataset for this SZA
            ds = xr.Dataset(
                coords={
                    "time": [reference_time],
                    "latitude": 0.0,  # Equator for simplicity
                    "longitude": 0.0,
                    "sza": sza  # Store SZA as coordinate for reference
                }
            )
            
            # Override SZA for this simulation
            parameter_overrides = {
                "custom_sza": sza  # Will be handled in uvspec input generation
            }
            
            # Run simulation
            result = ds.pyradtran.run_uvspec(
                config_path=None,
                parameter_overrides=parameter_overrides,
                return_dataset=True,
                save_to_file=False
            )
            scenario_results.append(result)
        
        # Combine results for this scenario
        combined_scenario = xr.concat(scenario_results, dim="sza")
        all_scenario_results[cloud_type] = combined_scenario
    
    # Plot comparisons
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot global irradiance
    for cloud_type, result in all_scenario_results.items():
        axes[0].plot(result.sza, result.eglo, 'o-', label=f"{cloud_type}")
    
    axes[0].set_title("Global Irradiance vs. Solar Zenith Angle")
    axes[0].set_xlabel("Solar Zenith Angle (degrees)")
    axes[0].set_ylabel("Global Irradiance (W/m²)")
    axes[0].set_xlim(0, 80)
    axes[0].grid(True)
    axes[0].legend()
    
    # Plot albedo
    for cloud_type, result in all_scenario_results.items():
        axes[1].plot(result.sza, result.albedo, 'o-', label=f"{cloud_type}")
    
    axes[1].set_title("Effective Albedo vs. Solar Zenith Angle")
    axes[1].set_xlabel("Solar Zenith Angle (degrees)")
    axes[1].set_ylabel("Effective Albedo")
    axes[1].set_xlim(0, 80)
    axes[1].set_ylim(0, 1)
    axes[1].grid(True)
    axes[1].legend()
    
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "example3_cloud_comparison.png")
    print(f"Saved plot to {OUTPUT_DIR / 'example3_cloud_comparison.png'}")
    
    # Save results
    for cloud_type, result in all_scenario_results.items():
        result.to_netcdf(OUTPUT_DIR / f"example3_{cloud_type}_results.nc")
    
    return all_scenario_results

def example_4_vertical_profile():
    """Study vertical profiles of radiation under different conditions."""
    print("\n=== Example 4: Vertical Profiles ===")
    
    # Base configuration
    config = load_config()
    config = configure_spectral_range(config, "visible", integrate=True)
    config = configure_output_altitudes(config, "troposphere")  # Multiple altitude levels
    
    # Scenarios to compare
    scenarios = ["arctic_clear", "arctic_snow_clouds", "tropical_clear", "tropical_cumulus"]
    
    # Reference time (noon)
    reference_time = pd.to_datetime("2023-05-01 12:00:00")
    
    # Run simulations for each scenario
    all_profile_results = {}
    
    for scenario in scenarios:
        # Configure for this scenario
        print(f"Simulating {scenario} scenario...")
        current_config = configure_common_scenario(config, scenario)
        
        # Create dataset for this simulation
        ds = xr.Dataset(
            coords={
                "time": [reference_time],
                "latitude": 0.0 if "tropical" in scenario else 75.0,  # Location appropriate for scenario
                "longitude": 0.0,
                "scenario": scenario  # Store scenario as coordinate for reference
            }
        )
        
        # Run simulation
        result = ds.pyradtran.run_uvspec(
            config_path=None,
            return_dataset=True,
            save_to_file=True,
            output_path=OUTPUT_DIR / f"example4_{scenario}_profile.nc"
        )
        
        all_profile_results[scenario] = result
    
    # Plot vertical profiles
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))
    
    for scenario, result in all_profile_results.items():
        # Downward (global) irradiance profile
        axes[0].plot(result.eglo.isel(time=0), result.altitude, 'o-', label=scenario)
        
        # Upward irradiance profile
        axes[1].plot(result.eup.isel(time=0), result.altitude, 'o-', label=scenario)
    
    axes[0].set_title("Downward Irradiance Profile")
    axes[0].set_xlabel("Global Irradiance (W/m²)")
    axes[0].set_ylabel("Altitude (km)")
    axes[0].grid(True)
    axes[0].legend()
    
    axes[1].set_title("Upward Irradiance Profile")
    axes[1].set_xlabel("Upward Irradiance (W/m²)")
    axes[1].set_ylabel("Altitude (km)")
    axes[1].grid(True)
    axes[1].legend()
    
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "example4_vertical_profiles.png")
    print(f"Saved plot to {OUTPUT_DIR / 'example4_vertical_profiles.png'}")
    
    return all_profile_results

if __name__ == "__main__":
    print(f"PyRadtran version: {pyradtran.__version__}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    try:
        # Run examples
        example_1_results = example_1_basic_simulation()
        example_2_results = example_2_parameter_study()
        example_3_results = example_3_cloud_study()
        example_4_results = example_4_vertical_profile()
        
        print("\nAll examples completed successfully!")
    except Exception as e:
        print(f"Error running examples: {e}")
        raise