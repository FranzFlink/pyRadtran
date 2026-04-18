# pyradtran/clouds.py
"""
Cloud utilities for pyRadtran.

This module provides tools to create libRadtran-compatible cloud input
files from various data sources.  The typical workflow is:

1. Build a list of :class:`CloudLayer` objects — either from simple
   parameters (:meth:`CloudGenerator.from_simple_parameters`) or from an
   ERA5 dataset (:meth:`CloudGenerator.from_era5_dataset`).
2. Write the layers to a ``.dat`` file with :class:`CloudFileWriter`.
3. Pass the file path to ``ds.pyradtran.run()`` via
   ``parameter_overrides={'wc_file 1D': 'my_cloud.dat'}``.

Examples
--------
Create a simple water cloud and write it to a file:

>>> from pyradtran.clouds import CloudGenerator, CloudFileWriter
>>> from pathlib import Path
>>> layers = CloudGenerator.from_simple_parameters(
...     z_base_km=1.0, z_top_km=2.0, lwc_g_m3=0.3, r_eff_um=10.0, n_layers=5
... )
>>> CloudFileWriter.write_water_cloud_file(layers, Path("work/wc.dat"))

See Also
--------
pyradtran.config.CloudParameters : Declarative cloud settings in YAML configs.
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
    """A single cloud layer with vertical extent and microphysical properties.

    ``CloudLayer`` is a validated data container.  It is not written to disk
    directly — pass a list of layers to :class:`CloudFileWriter` instead.

    Parameters
    ----------
    z_bottom_km : float
        Bottom altitude of the layer in km above sea level.
    z_top_km : float
        Top altitude of the layer in km above sea level.  Must be greater
        than *z_bottom_km*.
    lwc_g_m3 : float, default ``0.0``
        Liquid water content in g m⁻³.
    iwc_g_m3 : float, default ``0.0``
        Ice water content in g m⁻³.
    r_eff_um : float, default ``10.0``
        Effective droplet / crystal radius in µm.
    cloud_fraction : float, default ``1.0``
        Cloud fraction (0–1).

    Raises
    ------
    ValueError
        If *z_bottom_km* >= *z_top_km*, water content is negative,
        effective radius is non-positive, or cloud fraction is outside [0, 1].

    Examples
    --------
    >>> layer = CloudLayer(z_bottom_km=1.0, z_top_km=2.0, lwc_g_m3=0.3)
    >>> layer.r_eff_um
    10.0

    See Also
    --------
    CloudGenerator.from_simple_parameters : Build layers from basic numbers.
    CloudGenerator.from_era5_dataset : Build layers from ERA5 reanalysis.
    """
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
    """Factory for :class:`CloudLayer` sequences.

    All methods are static — no instantiation needed.  Choose a source
    method according to your input data:

    * :meth:`from_simple_parameters` — uniform slab from a few numbers.
    * :meth:`from_era5_dataset` — vertically resolved layers from ERA5.

    See Also
    --------
    CloudFileWriter : Persist layers to libRadtran ``.dat`` files.
    """
    
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
        """Extract cloud layers from an ERA5 :class:`xarray.Dataset`.

        The method selects a single profile (time, lat, lon), converts
        LWC / IWC from kg kg⁻¹ to g m⁻³, and returns
        one :class:`CloudLayer` per model level that exceeds the water-
        content thresholds.

        Parameters
        ----------
        ds : xarray.Dataset
            ERA5 dataset containing, at minimum, cloud water content on
            pressure levels.
        time : datetime, optional
            Time step to select.  Defaults to the first available.
        lat, lon : float, optional
            Coordinates for nearest-neighbour selection.  If either is
            *None* the spatial mean is used.
        cloud_variables : dict of str, optional
            Mapping ``{'lwc': 'clwc', 'iwc': 'ciwc', 'cc': 'cc',
            'temp': 't', 'z': 'z'}`` that connects internal keys to
            dataset variable names.
        altitude_levels_km : numpy.ndarray, optional
            Pre-computed altitude grid.  When *None*, altitudes are
            derived from geopotential height or the hypsometric equation.
        lwc_threshold : float, default ``1e-6``
            Minimum liquid water content (g m⁻³) to retain.
        iwc_threshold : float, default ``1e-6``
            Minimum ice water content (g m⁻³) to retain.
        default_r_eff_water : float, default ``10.0``
            Effective radius for liquid droplets (µm).
        default_r_eff_ice : float, default ``30.0``
            Effective radius for ice crystals (µm).
        pressure_levels : str, optional
            Name of the pressure coordinate.  Auto-detected when *None*.
        geopotential_var : str, optional
            Name of the geopotential variable.  Auto-detected when *None*.

        Returns
        -------
        list of CloudLayer
            One layer per model level with significant cloud content,
            sorted from lowest to highest altitude.

        Raises
        ------
        ValueError
            If no pressure coordinate can be found.
        KeyError
            If a required cloud variable is missing from *ds*.

        Examples
        --------
        >>> import xarray as xr
        >>> ds = xr.open_dataset("era5_cloud.nc")
        >>> layers = CloudGenerator.from_era5_dataset(
        ...     ds, lat=78.0, lon=15.0, default_r_eff_water=8.0
        ... )

        See Also
        --------
        generate_cloud_file_from_era5 : One-step convenience wrapper.
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
        """Create uniform cloud layers from basic parameters.

        The altitude range *z_base_km* … *z_top_km* is split into
        *n_layers* sub-layers, each receiving the same microphysical
        values.

        Parameters
        ----------
        z_base_km : float
            Cloud base altitude (km above sea level).
        z_top_km : float
            Cloud top altitude (km above sea level).
        lwc_g_m3 : float, default ``0.1``
            Liquid water content (g m⁻³).
        iwc_g_m3 : float, default ``0.0``
            Ice water content (g m⁻³).
        r_eff_um : float, default ``10.0``
            Effective droplet / crystal radius (µm).
        cloud_fraction : float, default ``1.0``
            Cloud fraction (0–1).
        n_layers : int, default ``1``
            Number of sub-layers to create.

        Returns
        -------
        list of CloudLayer
            Layers ordered from bottom to top.

        Raises
        ------
        ValueError
            If *n_layers* < 1 or *z_base_km* >= *z_top_km*.

        Examples
        --------
        >>> layers = CloudGenerator.from_simple_parameters(
        ...     z_base_km=1.0, z_top_km=2.0, lwc_g_m3=0.3, n_layers=5
        ... )
        >>> len(layers)
        5

        See Also
        --------
        CloudFileWriter.write_water_cloud_file : Write layers to disk.
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
    """Persist :class:`CloudLayer` sequences as libRadtran ``.dat`` files.

    Two static methods are provided — one per cloud phase:

    * :meth:`write_water_cloud_file` → ``wc_file 1D``
    * :meth:`write_ice_cloud_file`   → ``ic_file 1D``

    The output format is a three-column ASCII table
    (altitude, water-content, effective-radius) understood by
    libRadtran's ``wc_file`` / ``ic_file`` input options.

    See Also
    --------
    CloudGenerator : Build :class:`CloudLayer` lists from data or parameters.
    """
    
    @staticmethod
    def write_water_cloud_file(
        cloud_layers: List[CloudLayer],
        output_path: Path,
        include_zero_layers: bool = True,
        altitude_resolution_km: float = 0.1
    ) -> Path:
        """Write a water-cloud ``.dat`` file for libRadtran.

        The file contains three columns — altitude (km), liquid water
        content (g m⁻³), and effective radius (µm) — on a
        regular altitude grid.

        Parameters
        ----------
        cloud_layers : list of CloudLayer
            Layers to rasterise.  Only entries with ``lwc_g_m3 > 0``
            contribute unless *include_zero_layers* is set.
        output_path : pathlib.Path
            Destination file.  Parent directories are created
            automatically.
        include_zero_layers : bool, default ``True``
            If *True*, grid points outside any layer are written as
            zero LWC.  Set to *False* to skip them.
        altitude_resolution_km : float, default ``0.1``
            Vertical spacing of the output grid (km).

        Returns
        -------
        pathlib.Path
            *output_path* (echoed back for convenience).

        Examples
        --------
        >>> from pathlib import Path
        >>> layers = CloudGenerator.from_simple_parameters(
        ...     z_base_km=1.0, z_top_km=2.0, lwc_g_m3=0.3
        ... )
        >>> CloudFileWriter.write_water_cloud_file(layers, Path("wc.dat"))
        PosixPath('wc.dat')
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
        """Write an ice-cloud ``.dat`` file for libRadtran.

        Analogous to :meth:`write_water_cloud_file` but uses
        ``iwc_g_m3`` and defaults to a 30 µm effective radius.

        Parameters
        ----------
        cloud_layers : list of CloudLayer
            Layers to rasterise.  Only entries with ``iwc_g_m3 > 0``
            contribute unless *include_zero_layers* is set.
        output_path : pathlib.Path
            Destination file.
        include_zero_layers : bool, default ``True``
            Write zero-IWC grid points when *True*.
        altitude_resolution_km : float, default ``0.1``
            Vertical spacing of the output grid (km).

        Returns
        -------
        pathlib.Path
            *output_path*.

        Examples
        --------
        >>> CloudFileWriter.write_ice_cloud_file(layers, Path("ic.dat"))
        PosixPath('ic.dat')
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
    """One-step cloud-file generation from an ERA5 dataset.

    Combines :meth:`CloudGenerator.from_era5_dataset` and the
    appropriate :class:`CloudFileWriter` method into a single call.

    Parameters
    ----------
    era5_dataset : xarray.Dataset
        ERA5 dataset with cloud variables on pressure levels.
    output_path : pathlib.Path
        Destination ``.dat`` file.
    cloud_type : {"wc", "ic"}, default ``"wc"``
        ``"wc"`` for liquid water cloud, ``"ic"`` for ice cloud.
    time : datetime, optional
        Time step to select from the dataset.
    lat, lon : float, optional
        Coordinates for nearest-neighbour selection.
    **kwargs
        Forwarded to :meth:`CloudGenerator.from_era5_dataset`.

    Returns
    -------
    pathlib.Path
        *output_path*.

    Raises
    ------
    ValueError
        If *cloud_type* is not ``"wc"`` or ``"ic"``.

    Examples
    --------
    >>> path = generate_cloud_file_from_era5(
    ...     ds, Path("work/wc.dat"), cloud_type="wc", lat=78.0, lon=15.0
    ... )

    See Also
    --------
    CloudGenerator.from_era5_dataset : Extract layers only (no I/O).
    CloudFileWriter : Write layers to disk separately.
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
