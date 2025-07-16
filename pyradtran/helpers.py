# pyradtran/helpers.py
"""
Helper functions to simplify common use cases for pyradtran.

This module provides convenient functions to generate common configurations for:
- Various surface types (snow, ocean, vegetation, etc.)
- Cloud scenarios (cirrus, stratus, cumulus)
- Atmospheric conditions (clear, polluted, arctic, tropical)
"""

import logging
from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path

from .config import (
    SimulationConfig, 
    PathsConfig, 
    SimulationDefaults, 
    ExecutionConfig, 
    OutputConfig,
    CloudParameters,
    AerosolParameters,
    load_config
)

logger = logging.getLogger(__name__)

# --- Surface Type Helpers ---

def configure_surface(
    config: SimulationConfig,
    surface_type: str,
    albedo_value: Optional[float] = None,
    temperature_k: Optional[float] = None
) -> SimulationConfig:
    """
    Configure surface properties for common surface types.
    
    Args:
        config: The simulation configuration to modify
        surface_type: Type of surface ('snow', 'ocean', 'vegetation', 'desert', 'custom')
        albedo_value: Albedo value for 'custom' type or to override defaults
        temperature_k: Surface temperature in K to override defaults
    
    Returns:
        Updated SimulationConfig
    
    Examples:
        >>> config = load_config("config.yaml")
        >>> config = configure_surface(config, "snow")
    """
    # Make a copy of the config to avoid modifying the original
    # In reality we're just changing a reference, but this ensures clear semantics
    config = config
    
    # Set surface properties based on type
    if surface_type.lower() == "snow":
        config.simulation_defaults.albedo_type = "const"
        config.simulation_defaults.albedo_value = albedo_value if albedo_value is not None else 0.85
        config.simulation_defaults.surface_temperature_k = temperature_k if temperature_k is not None else 265.0
        config.simulation_defaults.brdf_type = "lambertian"
    
    elif surface_type.lower() == "fresh_snow":
        config.simulation_defaults.albedo_type = "const"
        config.simulation_defaults.albedo_value = albedo_value if albedo_value is not None else 0.95
        config.simulation_defaults.surface_temperature_k = temperature_k if temperature_k is not None else 265.0
        config.simulation_defaults.brdf_type = "lambertian"
    
    elif surface_type.lower() == "ocean" or surface_type.lower() == "water":
        config.simulation_defaults.albedo_type = "const"
        config.simulation_defaults.albedo_value = albedo_value if albedo_value is not None else 0.07
        config.simulation_defaults.surface_temperature_k = temperature_k if temperature_k is not None else 283.0
        # Ocean is better represented with a BRDF
        config.simulation_defaults.brdf_type = "rpv"
        config.simulation_defaults.brdf_rpv_type = 11  # Open ocean
    
    elif surface_type.lower() == "vegetation" or surface_type.lower() == "forest":
        config.simulation_defaults.albedo_type = "const"
        config.simulation_defaults.albedo_value = albedo_value if albedo_value is not None else 0.15
        config.simulation_defaults.surface_temperature_k = temperature_k if temperature_k is not None else 293.0
        # Vegetation can use library albedo for more accuracy
        # config.simulation_defaults.albedo_type = "library"
        # config.simulation_defaults.albedo_library = "IGBP_VEGETATION"
    
    elif surface_type.lower() == "desert" or surface_type.lower() == "sand":
        config.simulation_defaults.albedo_type = "const"
        config.simulation_defaults.albedo_value = albedo_value if albedo_value is not None else 0.35
        config.simulation_defaults.surface_temperature_k = temperature_k if temperature_k is not None else 310.0
    
    elif surface_type.lower() == "urban" or surface_type.lower() == "city":
        config.simulation_defaults.albedo_type = "const"
        config.simulation_defaults.albedo_value = albedo_value if albedo_value is not None else 0.18
        config.simulation_defaults.surface_temperature_k = temperature_k if temperature_k is not None else 293.0
    
    elif surface_type.lower() == "custom":
        config.simulation_defaults.albedo_type = "const"
        if albedo_value is not None:
            config.simulation_defaults.albedo_value = albedo_value
        if temperature_k is not None:
            config.simulation_defaults.surface_temperature_k = temperature_k
    
    else:
        logger.warning(f"Unknown surface type: {surface_type}, using defaults.")
    
    return config

# --- Cloud Type Helpers ---

def add_cloud_layer(
    config: SimulationConfig,
    bottom_km: float,
    top_km: float,
    water_content: float,
    effective_radius_um: float
) -> SimulationConfig:
    """
    Add a cloud layer to the simulation configuration.
    
    Args:
        config: The simulation configuration to modify
        bottom_km: Bottom height of cloud layer in km
        top_km: Top height of cloud layer in km
        water_content: Liquid/ice water content in g/m³
        effective_radius_um: Effective radius in μm
    
    Returns:
        Updated SimulationConfig
    
    Examples:
        >>> config = load_config("config.yaml")
        >>> config = add_cloud_layer(config, 1.0, 2.0, 0.1, 10.0)
    """
    # Enable clouds if they aren't already
    config.simulation_defaults.clouds.enabled = True
    
    # Add the cloud layer
    config.simulation_defaults.clouds.layer_heights_km.append((bottom_km, top_km))
    config.simulation_defaults.clouds.layer_water_content.append(water_content)
    config.simulation_defaults.clouds.layer_effective_radius_um.append(effective_radius_um)
    
    return config

def configure_cloud(
    config: SimulationConfig,
    cloud_type: str
) -> SimulationConfig:
    """
    Configure cloud properties for common cloud types.
    
    Args:
        config: The simulation configuration to modify
        cloud_type: Type of cloud ('stratus', 'cumulus', 'cirrus', 'clear')
    
    Returns:
        Updated SimulationConfig
    
    Examples:
        >>> config = load_config("config.yaml")
        >>> config = configure_cloud(config, "cirrus")
    """
    # Clear existing cloud layers
    config.simulation_defaults.clouds.layer_heights_km = []
    config.simulation_defaults.clouds.layer_water_content = []
    config.simulation_defaults.clouds.layer_effective_radius_um = []
    
    if cloud_type.lower() == "clear":
        # Disable clouds
        config.simulation_defaults.clouds.enabled = False
        return config
    
    # Set default cloud properties
    config.simulation_defaults.clouds.enabled = True
    config.simulation_defaults.clouds.cloud_optical_properties = "mie"
    config.simulation_defaults.clouds.cloud_overlap = "max-random"
    
    # Configure specific cloud types
    if cloud_type.lower() == "stratus":
        # Low-level stratus cloud
        return add_cloud_layer(
            config=config,
            bottom_km=0.5,
            top_km=1.5,
            water_content=0.3,
            effective_radius_um=10.0
        )
    
    elif cloud_type.lower() == "cumulus":
        # Fair-weather cumulus
        return add_cloud_layer(
            config=config,
            bottom_km=1.0,
            top_km=3.0,
            water_content=0.5,
            effective_radius_um=12.0
        )
    
    elif cloud_type.lower() == "stratocumulus":
        # Stratocumulus deck
        return add_cloud_layer(
            config=config,
            bottom_km=0.7,
            top_km=1.5,
            water_content=0.4,
            effective_radius_um=11.0
        )
    
    elif cloud_type.lower() == "cirrus":
        # High-level ice cloud
        config.simulation_defaults.clouds.cloud_optical_properties = "yang"  # Better for ice clouds
        return add_cloud_layer(
            config=config,
            bottom_km=8.0,
            top_km=10.0,
            water_content=0.005,  # Much lower for ice clouds
            effective_radius_um=40.0  # Larger for ice crystals
        )
    
    elif cloud_type.lower() == "deep_convective":
        # Deep convective cloud with multiple layers
        config = add_cloud_layer(
            config=config,
            bottom_km=1.0,
            top_km=3.0,
            water_content=0.5,
            effective_radius_um=12.0
        )
        config = add_cloud_layer(
            config=config,
            bottom_km=3.0,
            top_km=6.0,
            water_content=0.3,
            effective_radius_um=15.0
        )
        config = add_cloud_layer(
            config=config,
            bottom_km=6.0,
            top_km=10.0,
            water_content=0.05,
            effective_radius_um=30.0  # Ice particles at the top
        )
        return config
    
    else:
        logger.warning(f"Unknown cloud type: {cloud_type}, using defaults.")
        return config

# --- Aerosol Type Helpers ---

def configure_aerosol(
    config: SimulationConfig,
    aerosol_type: str,
    visibility_km: Optional[float] = None
) -> SimulationConfig:
    """
    Configure aerosol properties for common atmospheric conditions.
    
    Args:
        config: The simulation configuration to modify
        aerosol_type: Type of aerosol ('rural', 'urban', 'maritime', 'desert', 'none')
        visibility_km: Visibility in km (optional override)
    
    Returns:
        Updated SimulationConfig
    
    Examples:
        >>> config = load_config("config.yaml")
        >>> config = configure_aerosol(config, "maritime")
    """
    if aerosol_type.lower() == "none" or aerosol_type.lower() == "clear":
        # Disable aerosols
        config.simulation_defaults.aerosols.enabled = False
        return config
    
    # Enable and configure aerosols
    config.simulation_defaults.aerosols.enabled = True
    
    # Set default visibility based on type if not specified
    default_visibility = {
        "rural": 50.0,
        "urban": 15.0,
        "maritime": 40.0,
        "desert": 25.0,
        "arctic": 100.0,
        "tropical": 30.0
    }
    
    # Use provided visibility or default
    vis = visibility_km if visibility_km is not None else default_visibility.get(aerosol_type.lower(), 23.0)
    
    # Configure specific aerosol types
    if aerosol_type.lower() in ["rural", "urban", "maritime", "desert"]:
        config.simulation_defaults.aerosols.aerosol_type = aerosol_type.lower()
        config.simulation_defaults.aerosols.aerosol_visibility_km = vis
        config.simulation_defaults.aerosols.aerosol_optical_properties = "default"
    
    elif aerosol_type.lower() == "arctic":
        # Arctic haze
        config.simulation_defaults.aerosols.aerosol_type = "rural"
        config.simulation_defaults.aerosols.aerosol_visibility_km = vis
        config.simulation_defaults.aerosols.aerosol_optical_properties = "default"
    
    elif aerosol_type.lower() == "tropical":
        # Tropical conditions
        config.simulation_defaults.aerosols.aerosol_type = "maritime"
        config.simulation_defaults.aerosols.aerosol_visibility_km = vis
        config.simulation_defaults.aerosols.aerosol_optical_properties = "default"
    
    elif aerosol_type.lower() == "volcanic":
        # Volcanic aerosols - would typically use a file in practice
        config.simulation_defaults.aerosols.aerosol_type = "user"
        config.simulation_defaults.aerosols.aerosol_visibility_km = 10.0 if visibility_km is None else visibility_km
        config.simulation_defaults.aerosols.aerosol_optical_properties = "mie"
    
    else:
        logger.warning(f"Unknown aerosol type: {aerosol_type}, using defaults.")
    
    return config

# --- Common Scenario Helpers ---

def configure_common_scenario(
    config: SimulationConfig,
    scenario: str
) -> SimulationConfig:
    """
    Configure complete scenarios with surface, clouds, and aerosols.
    
    Args:
        config: The simulation configuration to modify
        scenario: Name of the scenario ('arctic_clear', 'tropical_cumulus', 
                 'marine_stratus', 'urban_haze', 'desert_dust')
    
    Returns:
        Updated SimulationConfig
    
    Examples:
        >>> config = load_config("config.yaml")
        >>> config = configure_common_scenario(config, "arctic_clear")
    """
    if scenario.lower() == "arctic_clear":
        config = configure_surface(config, "snow")
        config = configure_cloud(config, "clear")
        config = configure_aerosol(config, "arctic")
        # Set appropriate RTE solver
        config.simulation_defaults.rte_solver = "disort"
    
    elif scenario.lower() == "arctic_snow_clouds":
        config = configure_surface(config, "snow")
        config = configure_cloud(config, "stratus")
        config = configure_aerosol(config, "arctic")
        config.simulation_defaults.rte_solver = "disort"
    
    elif scenario.lower() == "tropical_clear":
        config = configure_surface(config, "ocean")
        config = configure_cloud(config, "clear")
        config = configure_aerosol(config, "tropical")
        config.simulation_defaults.rte_solver = "disort"
    
    elif scenario.lower() == "tropical_cumulus":
        config = configure_surface(config, "ocean")
        config = configure_cloud(config, "cumulus")
        config = configure_aerosol(config, "tropical")
        config.simulation_defaults.rte_solver = "disort"
    
    elif scenario.lower() == "marine_stratus":
        config = configure_surface(config, "ocean")
        config = configure_cloud(config, "stratus")
        config = configure_aerosol(config, "maritime")
        config.simulation_defaults.rte_solver = "disort"
    
    elif scenario.lower() == "urban_haze":
        config = configure_surface(config, "urban")
        config = configure_cloud(config, "clear")
        config = configure_aerosol(config, "urban", visibility_km=10.0)
        config.simulation_defaults.rte_solver = "disort"
    
    elif scenario.lower() == "desert_dust":
        config = configure_surface(config, "desert")
        config = configure_cloud(config, "clear")
        config = configure_aerosol(config, "desert", visibility_km=15.0)
        config.simulation_defaults.rte_solver = "disort"
    
    elif scenario.lower() == "cirrus_over_ocean":
        config = configure_surface(config, "ocean")
        config = configure_cloud(config, "cirrus")
        config = configure_aerosol(config, "maritime")
        config.simulation_defaults.rte_solver = "disort"
    
    else:
        logger.warning(f"Unknown scenario: {scenario}, using defaults.")
    
    return config

# --- Spectral Range Helpers ---

def configure_spectral_range(
    config: SimulationConfig,
    spectral_range: str,
    integrate: bool = False
) -> SimulationConfig:
    """
    Configure spectral range for common applications.
    
    Args:
        config: The simulation configuration to modify
        spectral_range: Name of spectral range ('uv', 'visible', 'nir', 'swir', 
                        'broadband', 'par')
        integrate: Whether to integrate over the wavelength range
    
    Returns:
        Updated SimulationConfig
    
    Examples:
        >>> config = load_config("config.yaml")
        >>> config = configure_spectral_range(config, "par", integrate=True)
    """
    # Set wavelength range based on requested range
    if spectral_range.lower() == "uv":
        config.simulation_defaults.wavelength_nm = [280, 400]
    elif spectral_range.lower() == "visible":
        config.simulation_defaults.wavelength_nm = [400, 700]
    elif spectral_range.lower() == "nir":
        config.simulation_defaults.wavelength_nm = [700, 1400]
    elif spectral_range.lower() == "swir":
        config.simulation_defaults.wavelength_nm = [1400, 3000]
    elif spectral_range.lower() == "broadband":
        config.simulation_defaults.wavelength_nm = [280, 3000]
    elif spectral_range.lower() == "par":
        # Photosynthetically Active Radiation
        config.simulation_defaults.wavelength_nm = [400, 700]
    else:
        logger.warning(f"Unknown spectral range: {spectral_range}, using defaults.")
    
    # Set integration flag
    config.simulation_defaults.integrate_wavelength = integrate
    
    return config

# --- Output Configuration Helpers ---

def configure_output_altitudes(
    config: SimulationConfig,
    altitude_set: str
) -> SimulationConfig:
    """
    Configure output altitudes for common scenarios.
    
    Args:
        config: The simulation configuration to modify
        altitude_set: Type of altitude set ('surface_only', 'boundary_layer', 'troposphere', 
                      'full_atmosphere')
    
    Returns:
        Updated SimulationConfig
    
    Examples:
        >>> config = load_config("config.yaml")
        >>> config = configure_output_altitudes(config, "troposphere")
    """
    if altitude_set.lower() == "surface_only":
        config.simulation_defaults.output_altitudes_km = [0.0]
    
    elif altitude_set.lower() == "boundary_layer":
        config.simulation_defaults.output_altitudes_km = [0.0, 0.1, 0.5, 1.0, 2.0]
    
    elif altitude_set.lower() == "troposphere":
        config.simulation_defaults.output_altitudes_km = [0.0, 1.0, 2.0, 5.0, 10.0, 15.0]
    
    elif altitude_set.lower() == "full_atmosphere":
        config.simulation_defaults.output_altitudes_km = [0.0, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0, 50.0, 70.0, 100.0]
    
    elif altitude_set.lower() == "cloud_focus":
        # Altitudes focused around typical cloud locations
        config.simulation_defaults.output_altitudes_km = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 9.0, 12.0]
    
    else:
        logger.warning(f"Unknown altitude set: {altitude_set}, using defaults.")
    
    return config
