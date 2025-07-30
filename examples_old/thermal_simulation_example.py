#!/usr/bin/env python3
"""
Thermal Simulation Example for PyRadtran

This example demonstrates how to run thermal infrared simulations using the
new thermal source functionality. Thermal simulations are used to model
atmospheric emission and absorption in the infrared spectrum.

Key differences from solar simulations:
- Uses 'source thermal' instead of 'source solar'
- Wavelength range typically 4.5-42 microns (thermal IR)
- Requires surface temperature specification
- Uses 'per_cm' output processing instead of 'per_nm'
- No solar geometry (zenith angle, time, coordinates) needed
- Often uses 'reptran' for molecular absorption parameterization
"""

from pathlib import Path
from datetime import datetime
import logging

from pyradtran import SimulationConfig, PyRadtran

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Run thermal simulation example."""
    
    # Load thermal simulation configuration
    config_path = Path(__file__).parent / "config" / "thermal_simulation_example.yaml"
    
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        return
    
    # Load configuration
    config = SimulationConfig.from_yaml(config_path)
    
    # Create PyRadtran instance
    pyradtran = PyRadtran(config)
    
    # Thermal simulations don't require real coordinates or time
    # since they don't depend on solar geometry
    dt = datetime(2022, 4, 1, 0, 0, 0)  # Can be any time
    latitude = 75.0   # Can be any latitude
    longitude = 10.0  # Can be any longitude
    
    # For thermal simulations, you can optionally specify a radiosonde
    # to get realistic atmospheric profiles
    radiosonde_path = None
    if config.paths.radiosonde_base:
        # Example: use a specific radiosonde file
        radiosonde_file = "20220401_37064SOD.dat"
        radiosonde_path = config.paths.radiosonde_base / radiosonde_file
        
        if radiosonde_path.exists():
            logger.info(f"Using radiosonde: {radiosonde_path}")
        else:
            logger.warning(f"Radiosonde file not found: {radiosonde_path}")
            radiosonde_path = None
    
    try:
        # Run thermal simulation
        logger.info("Running thermal simulation...")
        logger.info(f"Source: {config.simulation_defaults.source}")
        logger.info(f"Wavelength range: {config.simulation_defaults.wavelength_nm} nm")
        logger.info(f"Surface temperature: {config.simulation_defaults.surface_temperature_k} K")
        logger.info(f"Molecular absorption: {config.simulation_defaults.mol_abs_param}")
        logger.info(f"RTE solver: {config.simulation_defaults.rte_solver}")
        
        result = pyradtran.run_single_simulation(
            dt=dt,
            latitude=latitude,
            longitude=longitude,
            radiosonde_path=radiosonde_path
        )
        
        if result is not None:
            logger.info("Thermal simulation completed successfully!")
            logger.info(f"Output saved to: {config.paths.output_dir}")
            
            # Display some basic results info
            if hasattr(result, 'dims'):
                logger.info(f"Result dimensions: {dict(result.dims)}")
            if hasattr(result, 'data_vars'):
                logger.info(f"Output variables: {list(result.data_vars.keys())}")
            
            # Example: Save the result to a NetCDF file
            output_file = config.paths.output_dir / "thermal_simulation_result.nc"
            result.to_netcdf(output_file)
            logger.info(f"Result saved to: {output_file}")
            
        else:
            logger.error("Thermal simulation failed!")
            
    except Exception as e:
        logger.error(f"Error running thermal simulation: {e}")
        raise

def compare_solar_vs_thermal():
    """
    Example function showing how to compare solar vs thermal simulations.
    This demonstrates the key differences in configuration.
    """
    
    logger.info("\\n=== Comparison: Solar vs Thermal Simulation Settings ===")
    
    # Solar simulation settings
    solar_settings = {
        "source": "solar",
        "wavelength_nm": [400, 4000],  # Visible to near-IR
        "mol_abs_param": "lowtran per_nm",
        "output_process": "per_nm",
        "requires_geometry": True,
        "surface_temperature_k": None,  # Not used
    }
    
    # Thermal simulation settings  
    thermal_settings = {
        "source": "thermal", 
        "wavelength_nm": [4500, 42000],  # Thermal IR
        "mol_abs_param": "reptran medium",
        "output_process": "per_cm", 
        "requires_geometry": False,
        "surface_temperature_k": 248.4,  # Required
    }
    
    logger.info("Solar simulation:")
    for key, value in solar_settings.items():
        logger.info(f"  {key}: {value}")
    
    logger.info("\\nThermal simulation:")
    for key, value in thermal_settings.items():
        logger.info(f"  {key}: {value}")

if __name__ == "__main__":
    compare_solar_vs_thermal()
    main()
