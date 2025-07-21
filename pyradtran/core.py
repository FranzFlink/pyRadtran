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
    override_albedo: Optional[float] = None, # Added override_albedo
    override_sza: Optional[float] = None,
    override_atmosphere: Optional[str] = None, # Added override_atmosphere
    override_surface_temperature: Optional[float] = None, # Added override_surface_temperature
    override_altitude_km: Optional[float] = None, # Added override_altitude for scalar altitude
    era5_atmosphere_file: Optional[Path] = None, # Added ERA5 atmosphere file
) -> str:
    """
    Generates the content for a LibRadtran/uvspec input file.
    
    Args:
        config: The simulation configuration object
        dt: Date and time for the simulation
        latitude: Location latitude in degrees (-90 to 90)
        longitude: Location longitude in degrees (-180 to 180)
        radiosonde_path: Optional path to a radiosonde file
        override_albedo: Optional override for surface albedo value # Added
        override_sza: Optional override for solar zenith angle
        override_atmosphere: Optional override for atmosphere type (e.g., 'tropical', 'subarctic_winter')
        override_surface_temperature: Optional override for surface temperature (K) # Added
        override_altitude_km: Optional override for observation altitude (treated as scalar) # Added
        era5_atmosphere_file: Optional path to custom ERA5 atmosphere file
        
    Returns:
        String containing the complete input file content
    """
    if config.execution.debug_mode:
        logger.debug(f"Generating input content for dt={dt}, lat={latitude}, lon={longitude}, rs_path={radiosonde_path}")
    
    lines = []
    
    # --- RTE (Radiative Transfer Equation) solver ---
    lines.append(f"rte_solver {config.simulation_defaults.rte_solver}")
    
    # --- Molecular absorption parameterization ---
    lines.append(f"mol_abs_param {config.simulation_defaults.mol_abs_param}")

    # --- Atmosphere settings ---
    if era5_atmosphere_file is not None:
        # Use the custom ERA5 atmosphere file with absolute path
        era5_abs_path = Path(era5_atmosphere_file).resolve()
        lines.append(f"atmosphere_file {era5_abs_path}")
        logger.debug(f"Using ERA5 atmosphere file: {era5_abs_path}")
    elif override_atmosphere is not None:
        # Use the explicitly provided atmosphere type
        lines.append(f"atmosphere {override_atmosphere}")
    else:
        # Use the default atmosphere type from configuration
        lines.append(f"atmosphere_file {config.paths.atmosphere_profile}")
    
    # --- Atmosphere and aerosol settings ---
    if radiosonde_path and config.simulation_defaults.h2o_source == 'radiosonde':
        # Add radiosonde with H2O RH parameters
        lines.append(f"radiosonde {radiosonde_path} H2O RH")
    
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
    # Handle source type (solar or thermal)
    if config.simulation_defaults.source == "thermal":
        lines.append("source thermal")
    else:
        # Solar source
        if config.paths.solar_spectrum:
            lines.append(f"source solar {config.paths.solar_spectrum} per_nm")
        else:
            lines.append("source solar per_nm")
    
    # Solar geometry parameters only apply to solar simulations
    if config.simulation_defaults.source == "solar":
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
    
    # For thermal simulations, add surface temperature
    if config.simulation_defaults.source == "thermal":
        # Use override surface temperature if provided, otherwise use config
        surface_temp = override_surface_temperature if override_surface_temperature is not None else config.simulation_defaults.surface_temperature_k
        if surface_temp is not None:
            lines.append(f"sur_temperature {surface_temp}")


    
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
        
    # Surface temperature if needed (only for solar simulations, thermal already added above)
    if config.simulation_defaults.source != "thermal":
        # Use override surface temperature if provided, otherwise use config
        surface_temp = override_surface_temperature if override_surface_temperature is not None else config.simulation_defaults.surface_temperature_k
        if hasattr(config.simulation_defaults, 'surface_temperature_k') and surface_temp:
            lines.append(f"sur_temperature {surface_temp}")
    
    # --- Wavelength settings ---
    wavelength = config.simulation_defaults.wavelength_nm
    if isinstance(wavelength, list) and len(wavelength) == 2:
        # Wavelength range - format with decimal points for consistency
        lines.append(f"wavelength {wavelength[0]:.0f}. {wavelength[1]:.0f}.")
    elif isinstance(wavelength, (int, float)):
        # Single wavelength - format with decimal point
        lines.append(f"wavelength {wavelength:.0f}.")
    
    # --- Output settings ---
    
    # Collect output directives in the correct order
    output_directives = []
    
    # Vertical levels for output (altitudes in km)
    altitudes = config.simulation_defaults.output_altitudes_km
    
    # Override with scalar altitude if provided (altitude as data variable)
    if override_altitude_km is not None:
        altitudes = [override_altitude_km]
        logger.debug(f"Using override altitude as scalar: {override_altitude_km} km")
    
    if altitudes:
        if isinstance(altitudes, list):
            # Format each altitude with explicit format
            alt_str = " ".join(f"{alt:.1f}" for alt in altitudes)
            output_directives.append(f"zout {alt_str}")
            # Add interpolation for multiple altitudes
            if len(altitudes) > 1:
                output_directives.append("zout_interpolate")
        else:
            output_directives.append(f"zout {altitudes:.1f}")
    
    # Add output_process directives in the correct order
    # Use per_cm for thermal, per_nm for solar simulations
    if config.simulation_defaults.source == "thermal":
        output_directives.append("output_process per_cm")
    else:
        output_directives.append("output_process per_nm")
    
    # Add integrate directive if wavelength integration is requested
    if hasattr(config.simulation_defaults, 'integrate_wavelength') and config.simulation_defaults.integrate_wavelength:
        output_directives.append("output_process integrate")
    
    # Add any additional output-related options from additional_options
    if hasattr(config.simulation_defaults, 'additional_options'):
        for option in config.simulation_defaults.additional_options:
            # Skip output_process directives as we've already handled them above
            if not option.startswith('output_process'):
                output_directives.append(option)
    
    # Output quantities - MUST be the very last directive
    if config.simulation_defaults.output_columns:
        output_format = " ".join(config.simulation_defaults.output_columns)
        output_directives.append(f"output_user {output_format}")
    else:
        # Default output format if none specified
        output_directives.append("output_user lambda edir edn eup")
    
    # Add all output directives in the correct order
    lines.extend(output_directives)
    
    if config.execution.debug_mode:
        logger.debug(f"Final input content before returning:\n{lines}")
    
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
        override_albedo: Optional[float] = None, # Added override_albedo
        override_surface_temperature: Optional[float] = None, # Added override_surface_temperature
        override_altitude_km: Optional[float] = None, # Added override_altitude
        era5_atmosphere_file: Optional[Path] = None, # Added ERA5 atmosphere file
        ) -> Tuple[Optional[Path], Optional[Path]]:
        """
        Generates the uvspec input file for a specific run.

        Returns:
            A tuple containing (input_file_path, radiosonde_path_used).
            Returns (None, None) if input generation fails.
        """
        if self.config.execution.debug_mode:
            logger.debug(f"[_generate_input_file] Called for dt={dt}, lat={latitude}, lon={longitude}")

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
                override_albedo=override_albedo, # Pass override_albedo
                override_surface_temperature=override_surface_temperature, # Pass override_surface_temperature
                override_altitude_km=override_altitude_km, # Pass override_altitude
                era5_atmosphere_file=era5_atmosphere_file, # Pass ERA5 atmosphere file
            )
        except Exception as e:
             logger.error(f"Failed to generate input content for {dt} @ ({latitude},{longitude}): {e}")
             if self.config.execution.debug_mode:
                logger.debug(traceback.format_exc())
             return None, None

        try:
            # Create a temporary file in the working directory
            # Ensure working_dir exists
            self.config.paths.working_dir.mkdir(parents=True, exist_ok=True)
            if self.config.execution.debug_mode:
                logger.debug(f"[_generate_input_file] Working directory for temp file: {self.config.paths.working_dir}")

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
            logger.debug(f"Generated input file: {input_file_path}") # This log is good for all modes
            if self.config.execution.debug_mode:
                 logger.debug(f"--- Input Content ({input_file_path.name}) ---\n{input_content}\n--- End Input ---")
            return input_file_path, radiosonde_path

        except IOError as e:
            logger.error(f"Error writing temporary input file: {e}")
            if self.config.execution.debug_mode:
                logger.debug(traceback.format_exc())
            return None, None
        except Exception as e:
            logger.exception(f"Unexpected error creating input file: {e}")
            if self.config.execution.debug_mode:
                logger.debug(traceback.format_exc())
            return None, None


    def run(
        self,
        dt: datetime,
        latitude: float,
        longitude: float,
        override_albedo: Optional[float] = None, # Added override_albedo
        override_surface_temperature: Optional[float] = None, # Added override_surface_temperature
        override_altitude_km: Optional[float] = None, # Added override_altitude
        era5_atmosphere_file: Optional[Path] = None, # Added ERA5 atmosphere file
        ) -> Optional[Path]:
        """
        Generates input, runs one uvspec instance, and returns the output file path.

        Args:
            dt: Timestamp for the simulation.
            latitude: Latitude for the simulation.
            longitude: Longitude for the simulation.
            override_albedo: Optional albedo value to override config. # Added
            override_surface_temperature: Optional surface temperature (K) to override config. # Added
            override_altitude_km: Optional observation altitude (km) to override config. # Added
            era5_atmosphere_file: Optional path to custom ERA5 atmosphere file.

        Returns:
            Path to the generated output file, or None if the simulation failed
            at any stage (input generation, execution, timeout).
        """
        input_file = None
        output_file = None # Initialize to None
        process = None
        
        if self.config.execution.debug_mode:
            logger.debug(f"[run] Called for dt={dt}, lat={latitude}, lon={longitude}")
            logger.debug(f"[run] Config cleanup_temp_files: {self.config.execution.cleanup_temp_files}")
            logger.debug(f"[run] Config debug_mode: {self.config.execution.debug_mode}")

        try:
            # 1. Generate Input File
            if self.config.execution.debug_mode:
                logger.debug("[run] Step 1: Generating input file...")
            input_file, _ = self._generate_input_file(dt, latitude, longitude, override_albedo=override_albedo, override_surface_temperature=override_surface_temperature, override_altitude_km=override_altitude_km, era5_atmosphere_file=era5_atmosphere_file) # Pass parameters
            if not input_file:
                logger.error("[run] Input file generation failed. Aborting run.")
                return None # Error already logged

            # Define output file path based on input file name
            output_file = input_file.with_suffix(".out")
            if self.config.execution.debug_mode:
                logger.debug(f"[run] Input file: {input_file}, Proposed output file: {output_file}")

            # 2. Run uvspec
            if self.config.execution.debug_mode:
                logger.debug("[run] Step 2: Running uvspec...")
            cmd = [str(self.config.paths.libradtran_bin)]
            # Capture stderr if in debug mode, otherwise discard
            stderr_pipe = subprocess.PIPE if self.config.execution.debug_mode else subprocess.DEVNULL
            timeout = self.config.execution.timeout_seconds

            logger.debug(f"Running uvspec: {' '.join(cmd)} < {input_file.name} > {output_file.name}") # Good for all modes

            # Ensure working_dir exists for uvspec execution
            self.config.paths.working_dir.mkdir(parents=True, exist_ok=True)
            if self.config.execution.debug_mode:
                logger.debug(f"[run] Executing uvspec in CWD: {self.config.paths.working_dir}")

            with open(input_file, 'r') as infile, open(output_file, 'w') as outfile_handle:
                process = subprocess.Popen(
                    cmd,
                    stdin=infile,
                    stdout=outfile_handle, # Write directly to the output file
                    stderr=stderr_pipe,
                    cwd=self.config.paths.working_dir, 
                    encoding='utf-8'
                )
                stdout_data, stderr_data = process.communicate(timeout=timeout)

                # 3. Check Results
                if process.returncode != 0:
                    logger.error(f"uvspec failed for {input_file.name}. Return code: {process.returncode}.")
                    if stderr_data: # Will only have data if debug_mode was true for stderr_pipe
                        logger.error(f"--- uvspec stderr ---\n{stderr_data}\n--- end stderr ---")
                    # Keep output file for debugging if cleanup is disabled, otherwise remove failed output
                    if self.config.execution.cleanup_temp_files and output_file.exists():
                         logger.debug(f"[run] cleanup_temp_files is True, removing failed output: {output_file}")
                         output_file.unlink(missing_ok=True) 
                    return None # Indicate failure
                else:
                    logger.debug(f"uvspec finished successfully for {input_file.name}.") # Good for all modes
                    if self.config.execution.debug_mode and stderr_data:
                        logger.debug(f"--- uvspec stderr (Success) ---\n{stderr_data}\n--- end stderr ---")
                    
                    # Ensure output file actually has content if uvspec succeeded
                    if output_file.exists() and output_file.stat().st_size > 0:
                        if self.config.execution.debug_mode:
                            logger.debug(f"[run] Output file {output_file} exists and is not empty.")
                        return output_file # Success! Return output path
                    else:
                        logger.error(f"[run] uvspec reported success, but output file {output_file} is missing or empty.")
                        if self.config.execution.cleanup_temp_files and output_file.exists():
                            logger.debug(f"[run] cleanup_temp_files is True, removing (unexpectedly) empty/missing output: {output_file}")
                            output_file.unlink(missing_ok=True)
                        return None


        except FileNotFoundError:
             logger.error(f"LibRadtran executable not found at {self.config.paths.libradtran_bin}")
             if self.config.execution.debug_mode:
                logger.debug(traceback.format_exc())
             return None
        except subprocess.TimeoutExpired:
             logger.error(f"uvspec process timed out after {timeout}s for {input_file.name if input_file else 'N/A'}. Killing process.")
             if process:
                 process.kill()
                 process.communicate() 
             if self.config.execution.cleanup_temp_files and output_file and output_file.exists():
                  logger.debug(f"[run] Timeout: cleanup_temp_files is True, removing output file: {output_file}")
                  output_file.unlink(missing_ok=True)
             return None
        except Exception as e:
            logger.exception(f"An unexpected error occurred running uvspec for {input_file.name if input_file else 'N/A'}: {e}")
            if self.config.execution.debug_mode:
                logger.debug(traceback.format_exc())
            # Clean up potentially corrupted output file if it exists and cleanup is on
            if self.config.execution.cleanup_temp_files and output_file and output_file.exists():
                logger.debug(f"[run] Exception: cleanup_temp_files is True, removing output file: {output_file}")
                output_file.unlink(missing_ok=True)
            return None
        finally:
            # 4. Cleanup Input File
            if self.config.execution.debug_mode:
                logger.debug(f"[run] Finally block. Input file: {input_file}")
            if input_file and input_file.exists():
                if self.config.execution.cleanup_temp_files:
                    try:
                        input_file.unlink()
                        logger.debug(f"Cleaned up input file: {input_file.name}")
                    except OSError as e_unlink:
                        logger.warning(f"Could not remove temporary input file {input_file}: {e_unlink}")
                else:
                    if self.config.execution.debug_mode:
                        logger.debug(f"[run] cleanup_temp_files is False. Input file {input_file} preserved.")
            elif input_file: # input_file path was set but it does not exist
                 if self.config.execution.debug_mode:
                    logger.debug(f"[run] Input file {input_file} was defined but does not exist in finally block.")
            else: # input_file was None
                if self.config.execution.debug_mode:
                    logger.debug("[run] Input file was None in finally block (likely failed generation).")
            
            # Output file cleanup is handled based on success/failure/config within the try/except blocks for uvspec run
            if self.config.execution.debug_mode:
                if output_file and output_file.exists():
                    logger.debug(f"[run] End of run. Output file {output_file} exists.")
                elif output_file:
                    logger.debug(f"[run] End of run. Output file {output_file} was defined but does not exist.")
                else:
                    logger.debug(f"[run] End of run. Output file was not defined (None).")
