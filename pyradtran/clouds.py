# pyradtran/clouds.py
"""
Cloud utilities for pyRadtran - Generate cloud files from xarray datasets
"""

import logging
import numpy as np
import xarray as xr
from pathlib import Path
from typing import Optional, Union, Tuple, Dict, Any, List
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class CloudLayer:
    """Represents a single cloud layer with its properties."""
    z_bottom_km: float
    z_top_km: float
    lwc_g_m3: float = 0.0  # Liquid water content
    iwc_g_m3: float = 0.0  # Ice water content
    r_eff_um: float = 10.0  # Effective radius in micrometers
    cloud_fraction: float = 1.0  # Cloud fraction (0-1)
    
    def __post_init__(self):
        """Validate cloud layer parameters."""
        if self.z_bottom_km >= self.z_top_km:
            raise ValueError(f"Bottom altitude ({self.z_bottom_km}) must be less than top altitude ({self.z_top_km})")
        if self.lwc_g_m3 < 0 or self.iwc_g_m3 < 0:
            raise ValueError("Water content cannot be negative")
        if self.r_eff_um <= 0:
            raise ValueError("Effective radius must be positive")
        if not 0 <= self.cloud_fraction <= 1:
            raise ValueError("Cloud fraction must be between 0 and 1")

class CloudGenerator:
    """Generate libRadtran cloud files from various data sources."""
    
    @staticmethod
    def from_era5_dataset(
        ds: xr.Dataset,
        time: Optional[datetime] = None,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
        cloud_variables: Optional[Dict[str, str]] = None,
        altitude_levels_km: Optional[np.ndarray] = None,
        lwc_threshold: float = 1e-6,
        iwc_threshold: float = 1e-6,
        default_r_eff_water: float = 10.0,
        default_r_eff_ice: float = 30.0,
        pressure_levels: Optional[str] = None,
        geopotential_var: Optional[str] = None
    ) -> List[CloudLayer]:
        """
        Extract cloud layers from ERA5 dataset.
        
        Args:
            ds: xarray Dataset containing cloud data
            time: Time to extract (if None, uses first time)
            lat: Latitude to extract (if None, uses spatial mean)
            lon: Longitude to extract (if None, uses spatial mean)
            cloud_variables: Dictionary mapping standard names to dataset variable names
                           e.g., {'lwc': 'clwc', 'iwc': 'ciwc', 'cc': 'cc', 'z': 'z'}
            altitude_levels_km: Altitude levels in km (if None, computed from geopotential or pressure)
            lwc_threshold: Minimum LWC to consider as cloud (g/m³)
            iwc_threshold: Minimum IWC to consider as cloud (g/m³)
            default_r_eff_water: Default effective radius for water droplets (μm)
            default_r_eff_ice: Default effective radius for ice crystals (μm)
            pressure_levels: Name of pressure coordinate (default: auto-detect)
            geopotential_var: Name of geopotential variable (default: auto-detect from cloud_variables or 'z')
            
        Returns:
            List of CloudLayer objects
        """
        logger.info("Generating cloud layers from ERA5 dataset")
        
        # Default variable mappings for ERA5
        if cloud_variables is None:
            cloud_variables = {
                'lwc': 'clwc',  # Cloud liquid water content
                'iwc': 'ciwc',  # Cloud ice water content  
                'cc': 'cc',     # Cloud cover
                'temp': 't',    # Temperature (for altitude calculation)
                'z': 'z'        # Geopotential height
            }
        
        # Select time
        if time is not None:
            try:
                ds_time = ds.sel(time=time, method='nearest')
            except (KeyError, ValueError):
                logger.warning(f"Time {time} not found, using first available time")
                ds_time = ds.isel(time=0)
        else:
            ds_time = ds.isel(time=0)
            
        # Select location or compute spatial mean
        if lat is not None and lon is not None:
            try:
                ds_point = ds_time.sel(latitude=lat, longitude=lon, method='nearest')
                logger.debug(f"Selected point: lat={lat}, lon={lon}")
            except (KeyError, ValueError):
                logger.warning("Lat/lon not found, using spatial mean")
                ds_point = ds_time.mean(dim=['latitude', 'longitude'], skipna=True)
        else:
            ds_point = ds_time.mean(dim=['latitude', 'longitude'], skipna=True)
            logger.debug("Using spatial mean for cloud properties")
        
        # Auto-detect pressure coordinate
        if pressure_levels is None:
            pressure_coords = [coord for coord in ds_point.coords 
                             if any(name in coord.lower() for name in ['pressure', 'level', 'plev'])]
            if pressure_coords:
                pressure_levels = pressure_coords[0]
                logger.debug(f"Auto-detected pressure coordinate: {pressure_levels}")
            else:
                raise ValueError("Could not find pressure coordinate in dataset")
        
        # Get pressure levels (in Pa, convert to hPa if needed)
        pressure = ds_point[pressure_levels].values
        if np.max(pressure) > 10000:  # Likely in Pa, convert to hPa
            pressure = pressure / 100
        
        # Calculate altitude levels - prioritize geopotential height if available
        if altitude_levels_km is None:
            # Try to use geopotential height first
            geopotential_var = geopotential_var or cloud_variables.get('z', 'z')
            
            if geopotential_var in ds_point:
                # Use geopotential height (convert from geopotential to geometric height)
                geopotential = ds_point[geopotential_var].values
                
                # Check if it's 1D (constant across time/space) or needs to be extracted
                if geopotential.ndim > 1:
                    # Take mean over any non-level dimensions
                    non_level_dims = [dim for dim in ds_point[geopotential_var].dims if dim != pressure_levels]
                    if non_level_dims:
                        geopotential = ds_point[geopotential_var].mean(dim=non_level_dims, skipna=True).values
                
                # Convert geopotential to geometric height (m -> km)
                # Geopotential height: h = Φ/g, where g ≈ 9.80665 m/s²
                if np.max(geopotential) > 100000:  # Likely in m²/s² (geopotential)
                    altitude_levels_km = geopotential / 9.80665 / 1000  # Convert to km
                else:  # Already in meters
                    altitude_levels_km = geopotential / 1000  # Convert to km
                
                logger.debug(f"Using geopotential height data with {len(altitude_levels_km)} levels")
                logger.debug(f"Altitude range: {altitude_levels_km.min():.2f} - {altitude_levels_km.max():.2f} km")
                
            else:
                # Fall back to hypsometric equation approximation
                logger.warning(f"Geopotential variable '{geopotential_var}' not found, using pressure-based approximation")
                # h ≈ -7 * ln(p/p0) where p0 = 1013.25 hPa, h in km
                altitude_levels_km = -7.0 * np.log(pressure / 1013.25)
                logger.debug(f"Calculated {len(altitude_levels_km)} altitude levels from pressure")
        
        # Ensure altitudes are in ascending order (surface to top)
        if len(altitude_levels_km) > 1 and altitude_levels_km[0] > altitude_levels_km[1]:
            altitude_levels_km = altitude_levels_km[::-1]
            pressure = pressure[::-1]
            logger.debug("Reversed altitude and pressure arrays to ensure ascending order")
        
        # Extract cloud variables
        cloud_layers = []
        
        try:
            # Get cloud water content variables
            lwc_var = cloud_variables.get('lwc')
            iwc_var = cloud_variables.get('iwc')
            cc_var = cloud_variables.get('cc')
            
            lwc = ds_point[lwc_var].values if lwc_var and lwc_var in ds_point else np.zeros_like(pressure)
            iwc = ds_point[iwc_var].values if iwc_var and iwc_var in ds_point else np.zeros_like(pressure)
            cc = ds_point[cc_var].values if cc_var and cc_var in ds_point else np.ones_like(pressure)
            
            # Convert from kg/kg to g/m³ if needed (typical ERA5 units)
            if lwc_var and lwc_var in ds_point:
                # Estimate air density for conversion (rough approximation)
                # ρ_air ≈ p/(R*T) where R ≈ 287 J/(kg·K)
                temp_var = cloud_variables.get('temp', 't')
                if temp_var in ds_point:
                    temp = ds_point[temp_var].values
                    air_density = (pressure * 100) / (287 * temp)  # kg/m³
                    lwc = lwc * air_density * 1000  # Convert to g/m³
                else:
                    # Use standard atmosphere approximation
                    air_density = pressure / 10  # Rough approximation in kg/m³
                    lwc = lwc * air_density * 1000
                    
            if iwc_var and iwc_var in ds_point:
                temp_var = cloud_variables.get('temp', 't')
                if temp_var in ds_point:
                    temp = ds_point[temp_var].values
                    air_density = (pressure * 100) / (287 * temp)  # kg/m³
                    iwc = iwc * air_density * 1000  # Convert to g/m³
                else:
                    air_density = pressure / 10  # Rough approximation
                    iwc = iwc * air_density * 1000
            
            # Create cloud layers for each level with significant cloud content
            for i, (alt, p, lwc_val, iwc_val, cc_val) in enumerate(zip(
                altitude_levels_km, pressure, lwc, iwc, cc
            )):
                # Check if there's significant cloud content
                has_liquid = lwc_val > lwc_threshold
                has_ice = iwc_val > iwc_threshold
                
                if has_liquid or has_ice:
                    # Determine layer boundaries (midpoint between levels)
                    # Use a minimum layer thickness to avoid zero-thickness layers
                    min_layer_thickness = 0.05  # 50 meters minimum
                    
                    if i == 0:
                        # Top layer
                        if i < len(altitude_levels_km) - 1:
                            z_bottom = (alt + altitude_levels_km[i+1]) / 2
                        else:
                            z_bottom = alt - min_layer_thickness
                        z_top = alt + min_layer_thickness
                    elif i == len(altitude_levels_km) - 1:
                        # Bottom layer
                        z_top = (altitude_levels_km[i-1] + alt) / 2
                        z_bottom = alt - min_layer_thickness
                    else:
                        # Middle layers
                        z_top = (altitude_levels_km[i-1] + alt) / 2
                        z_bottom = (alt + altitude_levels_km[i+1]) / 2
                    
                    # Ensure proper ordering and minimum thickness
                    if z_bottom >= z_top:
                        # Fallback: create a thin layer around the level
                        z_bottom = alt - min_layer_thickness / 2
                        z_top = alt + min_layer_thickness / 2
                    
                    # Ensure minimum layer thickness
                    if (z_top - z_bottom) < min_layer_thickness:
                        layer_center = (z_top + z_bottom) / 2
                        z_bottom = layer_center - min_layer_thickness / 2
                        z_top = layer_center + min_layer_thickness / 2
                    
                    # Ensure non-negative altitudes
                    if z_bottom < 0:
                        z_bottom = 0
                        z_top = max(z_top, min_layer_thickness)
                    
                    # Choose appropriate effective radius
                    if has_liquid and has_ice:
                        # Mixed phase - use temperature to decide dominant phase
                        r_eff = default_r_eff_ice if alt > 8.0 else default_r_eff_water
                    elif has_liquid:
                        r_eff = default_r_eff_water
                    else:
                        r_eff = default_r_eff_ice
                    
                    try:
                        cloud_layer = CloudLayer(
                            z_bottom_km=z_bottom,
                            z_top_km=z_top,
                            lwc_g_m3=lwc_val,
                            iwc_g_m3=iwc_val,
                            r_eff_um=r_eff,
                            cloud_fraction=cc_val
                        )
                        cloud_layers.append(cloud_layer)
                    except ValueError as e:
                        logger.warning(f"Skipping invalid cloud layer at {alt:.3f} km: {e}")
                        continue
                    
        except KeyError as e:
            logger.error(f"Required cloud variable not found in dataset: {e}")
            raise
        except Exception as e:
            logger.error(f"Error processing cloud data: {e}")
            raise
        
        logger.info(f"Generated {len(cloud_layers)} cloud layers from ERA5 data")
        return cloud_layers
    
    @staticmethod
    def from_simple_parameters(
        z_base_km: float,
        z_top_km: float,
        lwc_g_m3: float = 0.1,
        iwc_g_m3: float = 0.0,
        r_eff_um: float = 10.0,
        cloud_fraction: float = 1.0,
        n_layers: int = 1
    ) -> List[CloudLayer]:
        """
        Create simple cloud layers from basic parameters.
        
        Args:
            z_base_km: Cloud base altitude (km)
            z_top_km: Cloud top altitude (km)
            lwc_g_m3: Liquid water content (g/m³)
            iwc_g_m3: Ice water content (g/m³)
            r_eff_um: Effective radius (μm)
            cloud_fraction: Cloud fraction (0-1)
            n_layers: Number of layers to create
            
        Returns:
            List of CloudLayer objects
        """
        if n_layers <= 0:
            raise ValueError("Number of layers must be positive")
        
        layer_thickness = (z_top_km - z_base_km) / n_layers
        layers = []
        
        for i in range(n_layers):
            z_bottom = z_base_km + i * layer_thickness
            z_top = z_bottom + layer_thickness
            
            layer = CloudLayer(
                z_bottom_km=z_bottom,
                z_top_km=z_top,
                lwc_g_m3=lwc_g_m3,
                iwc_g_m3=iwc_g_m3,
                r_eff_um=r_eff_um,
                cloud_fraction=cloud_fraction
            )
            layers.append(layer)
        
        return layers

class CloudFileWriter:
    """Write cloud data to libRadtran-compatible files."""
    
    @staticmethod
    def write_water_cloud_file(
        cloud_layers: List[CloudLayer],
        output_path: Path,
        include_zero_layers: bool = True,
        altitude_resolution_km: float = 0.1
    ) -> Path:
        """
        Write water cloud file in libRadtran format.
        
        Args:
            cloud_layers: List of CloudLayer objects
            output_path: Output file path
            include_zero_layers: Whether to include layers with zero LWC
            altitude_resolution_km: Vertical resolution for output
            
        Returns:
            Path to written file
        """
        logger.info(f"Writing water cloud file to {output_path}")
        
        # Filter for water clouds only
        water_layers = [layer for layer in cloud_layers if layer.lwc_g_m3 > 0]
        
        if not water_layers and not include_zero_layers:
            logger.warning("No water cloud layers found")
            water_layers = cloud_layers  # Include all layers anyway
        
        # Determine altitude range
        if water_layers:
            z_min = min(layer.z_bottom_km for layer in water_layers) 
            z_max = max(layer.z_top_km for layer in water_layers)
        else:
            z_min, z_max = 0.0, 20.0
        
        # Create altitude grid
        z_grid = np.arange(z_max, z_min - altitude_resolution_km, -altitude_resolution_km)
        
        # Interpolate cloud properties to grid
        lwc_profile = np.zeros_like(z_grid)
        r_eff_profile = np.full_like(z_grid, 10.0)  # Default effective radius
        
        for layer in water_layers:
            # Find grid points within this layer
            in_layer = (z_grid >= layer.z_bottom_km) & (z_grid <= layer.z_top_km)
            lwc_profile[in_layer] = layer.lwc_g_m3
            r_eff_profile[in_layer] = layer.r_eff_um
        
        # Write file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write("#      z     LWC    R_eff\n")
            f.write("#     (km)  (g/m^3) (um)\n")
            
            for z, lwc, r_eff in zip(z_grid, lwc_profile, r_eff_profile):
                f.write(f"{z:10.3f} {lwc:8.3f} {r_eff:8.1f}\n")
        
        logger.info(f"Wrote {len(z_grid)} altitude levels to {output_path}")
        return output_path
    
    @staticmethod
    def write_ice_cloud_file(
        cloud_layers: List[CloudLayer],
        output_path: Path,
        include_zero_layers: bool = True,
        altitude_resolution_km: float = 0.1
    ) -> Path:
        """
        Write ice cloud file in libRadtran format.
        
        Args:
            cloud_layers: List of CloudLayer objects
            output_path: Output file path
            include_zero_layers: Whether to include layers with zero IWC
            altitude_resolution_km: Vertical resolution for output
            
        Returns:
            Path to written file
        """
        logger.info(f"Writing ice cloud file to {output_path}")
        
        # Filter for ice clouds only
        ice_layers = [layer for layer in cloud_layers if layer.iwc_g_m3 > 0]
        
        if not ice_layers and not include_zero_layers:
            logger.warning("No ice cloud layers found")
            ice_layers = cloud_layers
        
        # Determine altitude range
        if ice_layers:
            z_min = min(layer.z_bottom_km for layer in ice_layers)
            z_max = max(layer.z_top_km for layer in ice_layers)
        else:
            z_min, z_max = 5.0, 15.0  # Typical ice cloud range
        
        # Create altitude grid
        z_grid = np.arange(z_max, z_min - altitude_resolution_km, -altitude_resolution_km)
        
        # Interpolate cloud properties to grid
        iwc_profile = np.zeros_like(z_grid)
        r_eff_profile = np.full_like(z_grid, 30.0)  # Default ice effective radius
        
        for layer in ice_layers:
            in_layer = (z_grid >= layer.z_bottom_km) & (z_grid <= layer.z_top_km)
            iwc_profile[in_layer] = layer.iwc_g_m3
            r_eff_profile[in_layer] = layer.r_eff_um
        
        # Write file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write("#      z     IWC    R_eff\n")
            f.write("#     (km)  (g/m^3) (um)\n")
            
            for z, iwc, r_eff in zip(z_grid, iwc_profile, r_eff_profile):
                f.write(f"{z:10.3f} {iwc:8.3f} {r_eff:8.1f}\n")
        
        logger.info(f"Wrote {len(z_grid)} altitude levels to {output_path}")
        return output_path

def generate_cloud_file_from_era5(
    era5_dataset: xr.Dataset,
    output_path: Path,
    cloud_type: str = 'wc',
    time: Optional[datetime] = None,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    **kwargs
) -> Path:
    """
    Convenience function to generate cloud file from ERA5 dataset.
    
    Args:
        era5_dataset: xarray Dataset with cloud data
        output_path: Path where to save the cloud file
        cloud_type: Type of cloud file ('wc' for water, 'ic' for ice)
        time: Time to extract
        lat: Latitude to extract  
        lon: Longitude to extract
        **kwargs: Additional arguments for CloudGenerator.from_era5_dataset
        
    Returns:
        Path to generated cloud file
    """
    # Generate cloud layers from ERA5
    cloud_layers = CloudGenerator.from_era5_dataset(
        era5_dataset, 
        time=time, 
        lat=lat, 
        lon=lon, 
        **kwargs
    )
    
    # Write appropriate cloud file
    if cloud_type.lower() == 'wc':
        return CloudFileWriter.write_water_cloud_file(cloud_layers, output_path)
    elif cloud_type.lower() == 'ic':
        return CloudFileWriter.write_ice_cloud_file(cloud_layers, output_path)
    else:
        raise ValueError(f"Unknown cloud type: {cloud_type}. Use 'wc' or 'ic'")
