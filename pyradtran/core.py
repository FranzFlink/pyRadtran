# libradpy/core.py
import logging
import subprocess
import tempfile
import traceback
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, Dict, Any

from .config import SimulationConfig
from .utils import RadiosondeFinder # Import the finder

logger = logging.getLogger(__name__)

# Add generate_input_content to core.py to break the circular dependency
def generate_input_content(
    config: SimulationConfig,
    dt: datetime,
    latitude: float,
    longitude: float,
    radiosonde_path: Optional[Path] = None,
    override_albedo: Optional[float] = None,
    override_sza: Optional[float] = None,
) -> str:
    """
    Generates the content for a LibRadtran/uvspec input file.
    
    Args:
        config: The simulation configuration object
        dt: Date and time for the simulation
        latitude: Location latitude in degrees (-90 to 90)
        longitude: Location longitude in degrees (-180 to 180)
        radiosonde_path: Optional path to a radiosonde file
        override_albedo: Optional override for surface albedo value
        override_sza: Optional override for solar zenith angle
        
    Returns:
        String containing the complete input file content
    """
    lines = []
    
    # --- RTE (Radiative Transfer Equation) solver ---
    lines.append(f"rte_solver {config.simulation_defaults.rte_solver}")
    
    # --- Molecular absorption parameterization ---
    lines.append(f"mol_abs_param {config.simulation_defaults.mol_abs_param}")
    
    # --- Atmosphere and aerosol settings ---
    if radiosonde_path and config.simulation_defaults.h2o_source == 'radiosonde':
        # Use standard atmosphere file without H2O RH
        lines.append(f"atmosphere_file {config.paths.atmosphere_profile}")
        # Add radiosonde with H2O RH parameters
        lines.append(f"radiosonde {radiosonde_path} H2O RH")
    else:
        # Use default atmosphere file only
        lines.append(f"atmosphere_file {config.paths.atmosphere_profile}")
    
    # Add data files path from configuration
    if config.paths.libradtran_data:
        lines.append(f"data_files_path {config.paths.libradtran_data}")
    
    # --- Molecular modifications (ozone, water vapor, etc) ---
    if hasattr(config.simulation_defaults, 'mol_modify') and config.simulation_defaults.mol_modify is not None:
        # Handle mol_modify as a dictionary
        if isinstance(config.simulation_defaults.mol_modify, dict):
            for molecule, mod_info in config.simulation_defaults.mol_modify.items():
                value = mod_info['value']
                unit = mod_info['unit']
                lines.append(f"mol_modify {molecule} {value} {unit}")
        # Handle mol_modify as a string (legacy format)
        elif isinstance(config.simulation_defaults.mol_modify, str):
            lines.append(f"mol_modify {config.simulation_defaults.mol_modify}")
    
    # Add direct molecule modifications if present and mol_modify not used
    if not config.simulation_defaults.mol_modify:
        if hasattr(config.simulation_defaults, 'ozone_du') and config.simulation_defaults.ozone_du is not None:
            lines.append(f"mol_modify O3 {config.simulation_defaults.ozone_du} DU")
        
        if hasattr(config.simulation_defaults, 'h2o_mm') and config.simulation_defaults.h2o_mm is not None and config.simulation_defaults.h2o_source == 'fixed':
            lines.append(f"mol_modify H2O {config.simulation_defaults.h2o_mm} MM")
    
    # --- Aerosol settings ---
    if hasattr(config.simulation_defaults, 'aerosols'):
        # Check if aerosols enabled (could be dictionary or dataclass)
        aerosols_enabled = False
        if isinstance(config.simulation_defaults.aerosols, dict):
            aerosols_enabled = config.simulation_defaults.aerosols.get('enabled', False)
        elif hasattr(config.simulation_defaults.aerosols, 'enabled'):
            aerosols_enabled = config.simulation_defaults.aerosols.enabled
        
        if aerosols_enabled:
            # Get aerosol type (could be in dictionary or as attribute)
            aerosol_type = 'default'
            if isinstance(config.simulation_defaults.aerosols, dict):
                aerosol_type = config.simulation_defaults.aerosols.get('aerosol_type', 'default')
            elif hasattr(config.simulation_defaults.aerosols, 'aerosol_type'):
                aerosol_type = config.simulation_defaults.aerosols.aerosol_type
            
            lines.append(f"aerosol_default")
            
            # Add optional aerosol properties if specified
            if isinstance(config.simulation_defaults.aerosols, dict):
                if 'tau' in config.simulation_defaults.aerosols:
                    tau = config.simulation_defaults.aerosols['tau']
                    lines.append(f"aerosol_set_tau {tau}")
                
                if 'ssa' in config.simulation_defaults.aerosols:
                    ssa = config.simulation_defaults.aerosols['ssa']
                    lines.append(f"aerosol_set_ssa {ssa}")
            elif hasattr(config.simulation_defaults.aerosols, 'aerosol_visibility_km') and config.simulation_defaults.aerosols.aerosol_visibility_km:
                lines.append(f"aerosol_visibility {config.simulation_defaults.aerosols.aerosol_visibility_km}")
            
    # --- Cloud settings ---
    if hasattr(config.simulation_defaults, 'clouds'):
        # Check if clouds enabled (could be dictionary or dataclass)
        clouds_enabled = False
        if isinstance(config.simulation_defaults.clouds, dict):
            clouds_enabled = config.simulation_defaults.clouds.get('enabled', False)
        elif hasattr(config.simulation_defaults.clouds, 'enabled'):
            clouds_enabled = config.simulation_defaults.clouds.enabled
        
        if clouds_enabled:
            cloud_type = 'wc'  # Default to water clouds
            if isinstance(config.simulation_defaults.clouds, dict):
                cloud_type = config.simulation_defaults.clouds.get('cloud_type', 'wc')
            elif hasattr(config.simulation_defaults.clouds, 'cloud_type'):
                cloud_type = config.simulation_defaults.clouds.cloud_type
            
            # Water or ice clouds
            if cloud_type == 'wc':  # Water clouds
                wc_file = None
                if isinstance(config.simulation_defaults.clouds, dict):
                    wc_file = config.simulation_defaults.clouds.get('wc_file')
                elif hasattr(config.simulation_defaults.clouds, 'wc_file'):
                    wc_file = config.simulation_defaults.clouds.wc_file
                
                if wc_file:
                    lines.append(f"wc_file {wc_file}")
                else:
                    # Use parametric water cloud
                    wc_props = {}
                    if isinstance(config.simulation_defaults.clouds, dict):
                        wc_props = config.simulation_defaults.clouds.get('wc_properties', {})
                    elif hasattr(config.simulation_defaults.clouds, 'wc_properties'):
                        wc_props = config.simulation_defaults.clouds.wc_properties
                    
                    thick = wc_props.get('thickness', 1.0)
                    lwc = wc_props.get('lwc', 0.1)
                    r_eff = wc_props.get('r_eff', 10.0)
                    z_base = wc_props.get('z_base', 2.0)
                    
                    lines.append(f"wc_set_tau {wc_props.get('tau', 5.0)}")
                    lines.append(f"wc_properties {thick} {lwc} {r_eff}")
                    lines.append(f"wc_layer {z_base} {z_base + thick}")
            
            elif cloud_type == 'ic':  # Ice clouds
                ic_file = None
                if isinstance(config.simulation_defaults.clouds, dict):
                    ic_file = config.simulation_defaults.clouds.get('ic_file')
                elif hasattr(config.simulation_defaults.clouds, 'ic_file'):
                    ic_file = config.simulation_defaults.clouds.ic_file
                
                if ic_file:
                    lines.append(f"ic_file {ic_file}")
                else:
                    # Use parametric ice cloud
                    ic_props = {}
                    if isinstance(config.simulation_defaults.clouds, dict):
                        ic_props = config.simulation_defaults.clouds.get('ic_properties', {})
                    elif hasattr(config.simulation_defaults.clouds, 'ic_properties'):
                        ic_props = config.simulation_defaults.clouds.ic_properties
                    
                    thick = ic_props.get('thickness', 1.0)
                    iwc = ic_props.get('iwc', 0.01)
                    r_eff = ic_props.get('r_eff', 30.0)
                    z_base = ic_props.get('z_base', 9.0)
                    
                    lines.append(f"ic_set_tau {ic_props.get('tau', 1.0)}")
                    lines.append(f"ic_properties {thick} {iwc} {r_eff}")
                    lines.append(f"ic_layer {z_base} {z_base + thick}")
    
    # --- Solar and Geometry settings ---
    # Use specified solar spectrum file if available
    if config.paths.solar_spectrum:
        lines.append(f"source solar {config.paths.solar_spectrum}")
    else:
        lines.append("source solar")
    
    if override_sza is not None:
        # Use the explicitly provided solar zenith angle
        lines.append(f"sza {override_sza}")
    else:
        # Calculate SZA from time and location
        lines.append(f"time {dt.year} {dt.month} {dt.day} {dt.hour} {dt.minute} {dt.second}")
        
        # Format latitude with N/S indicator
        lat_hemisphere = "N" if latitude >= 0 else "S"
        lat_value = abs(latitude)
        lines.append(f"latitude {lat_hemisphere} {lat_value}")
        
        # Format longitude with E/W indicator
        lon_hemisphere = "E" if longitude >= 0 else "W"
        lon_value = abs(longitude)
        lines.append(f"longitude {lon_hemisphere} {lon_value}")
    
    # --- Surface properties ---
    if override_albedo is not None:
        # Use explicit albedo value
        lines.append(f"albedo {override_albedo}")
    elif config.simulation_defaults.albedo_type == 'library':
        # Use albedo from library
        lines.append(f"albedo_library {config.simulation_defaults.albedo_library}")
    elif config.simulation_defaults.albedo_type == 'file':
        # Use albedo from file
        if hasattr(config.simulation_defaults, 'albedo_file') and config.simulation_defaults.albedo_file:
            lines.append(f"albedo_file {config.simulation_defaults.albedo_file}")
    else:
        # Use default constant albedo
        albedo_value = getattr(config.simulation_defaults, 'albedo_value', 0.3)
        lines.append(f"albedo {albedo_value}")
    
    # BRDF if specified - RPV only (removed brdf_cam as requested)
    if hasattr(config.simulation_defaults, 'brdf_rpv') and config.simulation_defaults.brdf_rpv.enabled:
        # RPV BRDF parameters
        if config.simulation_defaults.brdf_rpv.rpv_type is not None:
            lines.append(f"brdf_rpv_type {config.simulation_defaults.brdf_rpv.rpv_type}")
        elif any([
            config.simulation_defaults.brdf_rpv.k is not None,
            config.simulation_defaults.brdf_rpv.rho0 is not None,
            config.simulation_defaults.brdf_rpv.theta is not None,
            config.simulation_defaults.brdf_rpv.sigma is not None,
            config.simulation_defaults.brdf_rpv.t1 is not None,
            config.simulation_defaults.brdf_rpv.t2 is not None,
            config.simulation_defaults.brdf_rpv.scale is not None
        ]):
            # Custom RPV parameters
            if config.simulation_defaults.brdf_rpv.k is not None:
                lines.append(f"brdf_rpv k {config.simulation_defaults.brdf_rpv.k}")
            if config.simulation_defaults.brdf_rpv.rho0 is not None:
                lines.append(f"brdf_rpv rho0 {config.simulation_defaults.brdf_rpv.rho0}")
            if config.simulation_defaults.brdf_rpv.theta is not None:
                lines.append(f"brdf_rpv theta {config.simulation_defaults.brdf_rpv.theta}")
            if config.simulation_defaults.brdf_rpv.sigma is not None:
                lines.append(f"brdf_rpv sigma {config.simulation_defaults.brdf_rpv.sigma}")
            if config.simulation_defaults.brdf_rpv.t1 is not None:
                lines.append(f"brdf_rpv t1 {config.simulation_defaults.brdf_rpv.t1}")
            if config.simulation_defaults.brdf_rpv.t2 is not None:
                lines.append(f"brdf_rpv t2 {config.simulation_defaults.brdf_rpv.t2}")
            if config.simulation_defaults.brdf_rpv.scale is not None:
                lines.append(f"brdf_rpv scale {config.simulation_defaults.brdf_rpv.scale}")
    # Legacy BRDF handling (for backward compatibility)
    elif hasattr(config.simulation_defaults, 'brdf_type') and config.simulation_defaults.brdf_type == 'rpv':
        if hasattr(config.simulation_defaults, 'brdf_rpv_type') and config.simulation_defaults.brdf_rpv_type is not None:
            lines.append(f"brdf_rpv_type {config.simulation_defaults.brdf_rpv_type}")
        
    # Surface temperature if needed
    if hasattr(config.simulation_defaults, 'surface_temperature_k') and config.simulation_defaults.surface_temperature_k:
        lines.append(f"sur_temperature {config.simulation_defaults.surface_temperature_k}")
    
    # --- Wavelength settings ---
    wavelength = config.simulation_defaults.wavelength_nm
    if isinstance(wavelength, list) and len(wavelength) == 2:
        # Wavelength range
        lines.append(f"wavelength {wavelength[0]} {wavelength[1]}")
    elif isinstance(wavelength, (int, float)):
        # Single wavelength
        lines.append(f"wavelength {wavelength}")
    
    # --- Output settings ---
    
    # Vertical levels for output (altitudes in km)
    altitudes = config.simulation_defaults.output_altitudes_km
    if altitudes:
        if isinstance(altitudes, list):
            # Format each altitude with explicit format
            alt_str = " ".join(f"{alt:.1f}" for alt in altitudes)
            lines.append(f"zout {alt_str}")
        else:
            lines.append(f"zout {altitudes:.1f}")
    
    # Output quantities
    if config.simulation_defaults.output_columns:
        output_format = " ".join(config.simulation_defaults.output_columns)
        lines.append(f"output_user {output_format}")
    else:
        # Default output format if none specified
        lines.append("output_user lambda edir edn eup")
    
    # Any additional options specified in the configuration
    if hasattr(config.simulation_defaults, 'additional_options'):
        for option in config.simulation_defaults.additional_options:
            lines.append(option)
    
    return "\n".join(lines)

class Simulation:
    """
    Represents and executes a single LibRadtran (uvspec) simulation instance.
    """
    def __init__(self, config: SimulationConfig):
        """
        Initializes the Simulation runner with configuration.

        Args:
            config: The loaded SimulationConfig object.
        """
        self.config = config
        # Initialize RadiosondeFinder if base path is set in config
        self.radiosonde_finder = None
        if config.paths.radiosonde_base and config.simulation_defaults.h2o_source == 'radiosonde':
             try:
                 self.radiosonde_finder = RadiosondeFinder(config.paths.radiosonde_base)
             except Exception as e:
                  logger.error(f"Failed to initialize RadiosondeFinder: {e}")
                  # Continue without it, Atmosphere interface will handle fallback

    def _generate_input_file(
        self,
        dt: datetime,
        latitude: float,
        longitude: float,
        # Add specific overrides here if needed, e.g.
        # override_albedo: Optional[float] = None,
        ) -> Tuple[Optional[Path], Optional[Path]]:
        """
        Generates the uvspec input file for a specific run.

        Returns:
            A tuple containing (input_file_path, radiosonde_path_used).
            Returns (None, None) if input generation fails.
        """
        # Find closest radiosonde if configured and finder is available
        radiosonde_path: Optional[Path] = None
        if self.config.simulation_defaults.h2o_source == 'radiosonde' and self.radiosonde_finder:
            radiosonde_path = self.radiosonde_finder.find_closest(dt)
            if radiosonde_path:
                 logger.debug(f"Time {dt} -> Closest Sonde: {radiosonde_path.name}")
            else:
                 logger.debug(f"Time {dt} -> No sonde found for this timestamp.")


        try:
            input_content = generate_input_content(
                config=self.config,
                dt=dt,
                latitude=latitude,
                longitude=longitude,
                radiosonde_path=radiosonde_path,
                # Pass overrides here: override_albedo=override_albedo
            )
        except Exception as e:
             logger.error(f"Failed to generate input content for {dt} @ ({latitude},{longitude}): {e}")
             logger.debug(traceback.format_exc())
             return None, None

        try:
            # Create a temporary file in the working directory
            # Suffix helps identify files if cleanup fails
            with tempfile.NamedTemporaryFile(
                mode='w',
                delete=False, # We manage deletion
                dir=self.config.paths.working_dir,
                prefix=f"uvspec_{dt.strftime('%Y%m%d_%H%M%S')}_",
                suffix=".inp"
            ) as tmp_inp:
                tmp_inp.write(input_content)
                input_file_path = Path(tmp_inp.name)
            logger.debug(f"Generated input file: {input_file_path}")
            if self.config.execution.debug_mode:
                 logger.debug(f"--- Input Content ({input_file_path.name}) ---\n{input_content}\n--- End Input ---")
            return input_file_path, radiosonde_path

        except IOError as e:
            logger.error(f"Error writing temporary input file: {e}")
            return None, None
        except Exception as e:
            logger.exception(f"Unexpected error creating input file: {e}")
            return None, None


    def run(
        self,
        dt: datetime,
        latitude: float,
        longitude: float,
        # Add specific overrides here if needed
        ) -> Optional[Path]:
        """
        Generates input, runs one uvspec instance, and returns the output file path.

        Args:
            dt: Timestamp for the simulation.
            latitude: Latitude for the simulation.
            longitude: Longitude for the simulation.

        Returns:
            Path to the generated output file, or None if the simulation failed
            at any stage (input generation, execution, timeout).
        """
        input_file = None
        output_file = None
        process = None

        try:
            # 1. Generate Input File
            input_file, _ = self._generate_input_file(dt, latitude, longitude) # Radiosonde path used internally
            if not input_file:
                return None # Error already logged

            # Define output file path based on input file name
            output_file = input_file.with_suffix(".out")

            # 2. Run uvspec
            cmd = [str(self.config.paths.libradtran_bin)]
            stderr_pipe = subprocess.PIPE if self.config.execution.debug_mode else subprocess.DEVNULL
            timeout = self.config.execution.timeout_seconds

            logger.debug(f"Running uvspec: {' '.join(cmd)} < {input_file.name} > {output_file.name}")

            with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
                process = subprocess.Popen(
                    cmd,
                    stdin=infile,
                    stdout=outfile,
                    stderr=stderr_pipe,
                    cwd=self.config.paths.working_dir, # Run in working dir
                    encoding='utf-8'
                )

                stdout_data, stderr_data = process.communicate(timeout=timeout)

                # 3. Check Results
                if process.returncode != 0:
                    logger.error(f"uvspec failed for {input_file.name}. Return code: {process.returncode}.")
                    if stderr_data:
                        logger.error(f"--- uvspec stderr ---\n{stderr_data}\n--- end stderr ---")
                    # Keep output file for debugging if cleanup is disabled
                    if self.config.execution.cleanup_temp_files:
                         output_file.unlink(missing_ok=True) # Remove failed output
                    return None # Indicate failure
                else:
                    logger.debug(f"uvspec finished successfully for {input_file.name}.")
                    if self.config.execution.debug_mode and stderr_data:
                        logger.debug(f"--- uvspec stderr (Success) ---\n{stderr_data}\n--- end stderr ---")
                    return output_file # Success! Return output path

        except FileNotFoundError:
             logger.error(f"LibRadtran executable not found at {self.config.paths.libradtran_bin}")
             return None
        except subprocess.TimeoutExpired:
             logger.error(f"uvspec process timed out after {timeout}s for {input_file.name if input_file else 'N/A'}. Killing process.")
             if process:
                 process.kill()
                 process.communicate() # Ensure process is cleaned up
             # Keep output file (likely empty/incomplete) for debugging if cleanup disabled
             if self.config.execution.cleanup_temp_files and output_file:
                  output_file.unlink(missing_ok=True)
             return None
        except Exception as e:
            logger.exception(f"An unexpected error occurred running uvspec for {input_file.name if input_file else 'N/A'}: {e}")
            return None
        finally:
            # 4. Cleanup Input File
            if input_file and input_file.exists() and self.config.execution.cleanup_temp_files:
                try:
                    input_file.unlink()
                    logger.debug(f"Cleaned up input file: {input_file.name}")
                except OSError as e:
                    logger.warning(f"Could not remove temporary input file {input_file}: {e}")
            # Output file cleanup happens based on success/failure/config above
