# pyradtran/core.py - UNIFIED VERSION
"""
Unified core simulation engine for pyradtran - REFACTORED VERSION.

This file provides the main Simulation class with:
- Simplified input file generation (no more confusion about which generator is used!)
- LibRadtran execution
- Basic output handling
- ERA5 atmosphere file support
- Streamlined cloud support

Original version backed up as core.py.backup

Key improvements:
- Single input generation path (no more duplicate functions)
- Works with cleaned configuration
- Better error handling
- Simplified cloud support

For migration guide, see REFACTORING_SUMMARY.md
"""

import logging
import subprocess
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

from .config import SimulationConfig
from .exceptions import UvspecExecutionError, InputGenerationError
from .utils import RadiosondeFinder

logger = logging.getLogger(__name__)


class Simulation:
    """Main simulation class for running LibRadtran simulations."""
    
    def __init__(self, config: SimulationConfig):
        """Initialize simulation with configuration."""
        self.config = config
        self.radiosonde_finder = RadiosondeFinder(config.paths.radiosonde_base) if config.paths.radiosonde_base else None
    
    def run_simulation(
        self,
        dt: datetime,
        latitude: float,
        longitude: float,
        override_albedo: Optional[float] = None,
        override_surface_temperature: Optional[float] = None,
        override_altitude_km: Optional[float] = None,
        era5_atmosphere_file: Optional[Path] = None,
        parameter_overrides: Dict[str, Any] = None
    ) -> Optional[Path]:
        """
        Run a single LibRadtran simulation.
        
        Args:
            dt: Date and time for the simulation
            latitude: Location latitude in degrees (-90 to 90)
            longitude: Location longitude in degrees (-180 to 180)
            override_albedo: Optional override for surface albedo value
            override_surface_temperature: Optional override for surface temperature (K)
            override_altitude_km: Optional override for observation altitude
            era5_atmosphere_file: Optional path to custom ERA5 atmosphere file
            parameter_overrides: Dictionary of additional parameter overrides
            
        Returns:
            Path to output file if successful, None otherwise
            
        Raises:
            UvspecExecutionError: If LibRadtran execution fails
        """
        try:
            # Find radiosonde file if using radiosonde data
            radiosonde_path = None
            if self.config.simulation_defaults.h2o_source == 'radiosonde' and self.radiosonde_finder:
                radiosonde_path = self.radiosonde_finder.find_radiosonde_file(dt, latitude, longitude)
            
            # Generate input file content
            input_content = self._generate_input_content(
                dt=dt,
                latitude=latitude,
                longitude=longitude,
                radiosonde_path=radiosonde_path,
                override_albedo=override_albedo,
                override_surface_temperature=override_surface_temperature,
                override_altitude_km=override_altitude_km,
                era5_atmosphere_file=era5_atmosphere_file,
                parameter_overrides=parameter_overrides
            )
            
            # Create temporary input file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.inp', delete=False, 
                                           dir=self.config.paths.working_dir) as inp_file:
                inp_file.write(input_content)
                input_path = Path(inp_file.name)
            
            # Create output file path
            output_path = input_path.with_suffix('.out')
            
            # Run LibRadtran
            success = self._run_uvspec(input_path, output_path)
            
            if success and output_path.exists():
                logger.debug(f"Simulation completed successfully: {output_path}")
                
                # Clean up input file if requested
                if self.config.execution.cleanup_temp_files:
                    input_path.unlink(missing_ok=True)
                
                return output_path
            else:
                logger.error(f"Simulation failed or produced no output")
                return None
                
        except Exception as e:
            logger.error(f"Simulation failed: {str(e)}")
            raise UvspecExecutionError(f"Simulation failed: {str(e)}")
    
    def _generate_input_content(
        self,
        dt: datetime,
        latitude: float,
        longitude: float,
        radiosonde_path: Optional[Path] = None,
        override_albedo: Optional[float] = None,
        override_surface_temperature: Optional[float] = None,
        override_altitude_km: Optional[float] = None,
        era5_atmosphere_file: Optional[Path] = None,
        parameter_overrides: Dict[str, Any] = None
    ) -> str:
        """Generate LibRadtran input file content."""
        lines = []
        
        # Basic LibRadtran settings
        lines.append(f"rte_solver {self.config.simulation_defaults.rte_solver}")
        lines.append(f"mol_abs_param {self.config.simulation_defaults.mol_abs_param}")
        
        # Data files path
        lines.append(f"data_files_path {self.config.paths.libradtran_data}")
        
        # Atmosphere settings
        if era5_atmosphere_file is not None:
            # Use custom ERA5 atmosphere file
            era5_abs_path = Path(era5_atmosphere_file).resolve()
            lines.append(f"atmosphere_file {era5_abs_path}")
            logger.debug(f"Using ERA5 atmosphere file: {era5_abs_path}")
        else:
            # Use default atmosphere profile
            lines.append(f"atmosphere_file {self.config.paths.atmosphere_profile}")
        
        # Radiosonde data
        if radiosonde_path and self.config.simulation_defaults.h2o_source == 'radiosonde':
            lines.append(f"radiosonde {radiosonde_path} H2O RH")
        
        # Molecular modifications
        if self.config.simulation_defaults.ozone_du is not None:
            lines.append(f"mol_modify O3 {self.config.simulation_defaults.ozone_du} DU")
        
        if (self.config.simulation_defaults.h2o_mm is not None and 
            self.config.simulation_defaults.h2o_source == 'fixed'):
            lines.append(f"mol_modify H2O {self.config.simulation_defaults.h2o_mm} MM")
        
        # Solar/thermal source
        if self.config.simulation_defaults.source == 'solar':
            if self.config.simulation_defaults.integrate_wavelength:
                lines.append(f"source solar {self.config.paths.solar_spectrum} per_nm")
            else:
                lines.append(f"source solar {self.config.paths.solar_spectrum}")
            
            # Solar geometry
            if self.config.simulation_defaults.sza is not None:
                lines.append(f"sza {self.config.simulation_defaults.sza}")
            else:
                # Calculate solar zenith angle from time and location
                sza = self._calculate_solar_zenith_angle(dt, latitude, longitude)
                lines.append(f"sza {sza}")
            
            # Day of year for solar spectrum adjustment
            day_of_year = dt.timetuple().tm_yday
            lines.append(f"day_of_year {day_of_year}")
            
        elif self.config.simulation_defaults.source == 'thermal':
            lines.append("source thermal")
            
            # Surface temperature for thermal calculations
            surf_temp = override_surface_temperature or self.config.simulation_defaults.surface_temperature_k
            if surf_temp is not None:
                lines.append(f"sur_temperature {surf_temp}")
        
        # Spectral settings
        wl_min, wl_max = self.config.simulation_defaults.wavelength_nm
        lines.append(f"wavelength {wl_min} {wl_max}")
        
        if self.config.simulation_defaults.integrate_wavelength:
            lines.append("output_process integrate")
        
        # Surface properties
        albedo = override_albedo or self.config.simulation_defaults.albedo_value
        lines.append(f"albedo {albedo}")
        
        # Output settings
        output_columns = " ".join(self.config.simulation_defaults.output_columns)
        lines.append(f"output_user {output_columns}")
        
        # Output altitudes
        if override_altitude_km is not None:
            # Single altitude override
            lines.append(f"zout {override_altitude_km}")
        else:
            # Use configured altitudes

            alt_str = " ".join(f"{alt:.1f}" for alt in self.config.simulation_defaults.output_altitudes_km)
            lines.append(f"zout {alt_str}")

        # Viewing geometry
        if self.config.simulation_defaults.viewing_geometry == 'nadir':
            lines.append("umu 1.0")  # Nadir viewing
        
        # Cloud settings (simplified)
        if self.config.simulation_defaults.clouds.enabled:
            self._add_cloud_settings(lines)
        
        # Apply parameter overrides
        if parameter_overrides:
            for key, value in parameter_overrides.items():
            # Remove any existing lines that start with the key followed by a space
                lines = [line for line in lines if not line.startswith(f"{key} ")]
                # Append the override line
                lines.append(f"{key} {value}")
        
        # Add quiet mode to reduce output
        lines.append("quiet")
        
        return '\n'.join(lines) + '\n'
    
    def _add_cloud_settings(self, lines: list):
        """Add cloud settings to input file."""
        clouds = self.config.simulation_defaults.clouds
        
        if clouds.cloud_source == 'file':
            # File-based clouds
            if clouds.cloud_type in ['wc', 'mixed'] and clouds.wc_file:
                lines.append(f"wc_file {clouds.wc_file}")
            if clouds.cloud_type in ['ic', 'mixed'] and clouds.ic_file:
                lines.append(f"ic_file {clouds.ic_file}")
                
        elif clouds.cloud_source == 'parametric':
            # Simple parametric cloud
            if clouds.cloud_type in ['wc', 'mixed']:
                lines.append(f"wc_layer {clouds.layer_bottom_km} {clouds.layer_top_km} {clouds.water_content_g_m3} {clouds.effective_radius_um}")
            if clouds.cloud_type in ['ic', 'mixed']:
                lines.append(f"ic_layer {clouds.layer_bottom_km} {clouds.layer_top_km} {clouds.ice_content_g_m3} {clouds.effective_radius_um}")
    
    def _calculate_solar_zenith_angle(self, dt: datetime, latitude: float, longitude: float) -> float:
        """Calculate solar zenith angle for given time and location."""
        try:
            import numpy as np
            
            # Day of year
            day_of_year = dt.timetuple().tm_yday
            
            # Solar declination angle
            declination = 23.45 * np.sin(np.radians((360 * (284 + day_of_year)) / 365))
            
            # Hour angle
            time_decimal = dt.hour + dt.minute / 60.0 + dt.second / 3600.0
            hour_angle = 15 * (time_decimal - 12) + longitude
            
            # Solar zenith angle
            lat_rad = np.radians(latitude)
            decl_rad = np.radians(declination)
            hour_rad = np.radians(hour_angle)
            
            cos_sza = (np.sin(lat_rad) * np.sin(decl_rad) + 
                      np.cos(lat_rad) * np.cos(decl_rad) * np.cos(hour_rad))
            
            sza = np.degrees(np.arccos(np.clip(cos_sza, -1, 1)))
            
            return round(sza, 2)
            
        except Exception as e:
            logger.warning(f"Failed to calculate solar zenith angle: {e}, using default 30°")
            return 30.0
    
    def _run_uvspec(self, input_path: Path, output_path: Path) -> bool:
        """Execute LibRadtran/uvspec."""
        try:
            cmd = [str(self.config.paths.libradtran_bin)]
            
            logger.debug(f"Running LibRadtran: {' '.join(cmd)}")
            logger.debug(f"Input file: {input_path}")
            logger.debug(f"Output file: {output_path}")
            
            with open(input_path, 'r') as inp_file, open(output_path, 'w') as out_file:
                result = subprocess.run(
                    cmd,
                    stdin=inp_file,
                    stdout=out_file,
                    stderr=subprocess.PIPE,
                    timeout=self.config.execution.timeout_seconds,
                    text=True
                )
            
            if result.returncode != 0:
                error_msg = result.stderr.strip() if result.stderr else "Unknown error"
                logger.error(f"LibRadtran failed with return code {result.returncode}: {error_msg}")
                return False
            
            return True
            
        except subprocess.TimeoutExpired:
            logger.error(f"LibRadtran execution timed out after {self.config.execution.timeout_seconds} seconds")
            return False
        except Exception as e:
            logger.error(f"Failed to execute LibRadtran: {str(e)}")
            return False


# Expose main class
__all__ = ['Simulation']
