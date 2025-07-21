# libradpy/config.py
import yaml
from pathlib import Path
from dataclasses import dataclass, field, fields, is_dataclass
from typing import List, Dict, Optional, Any, Union, Tuple
import logging
import os

logger = logging.getLogger(__name__)

# --- Define Dataclasses for structure and validation ---

@dataclass
class PathsConfig:
    libradtran_bin: Path
    libradtran_data: Path
    atmosphere_profile: Path
    solar_spectrum: Path
    radiosonde_base: Optional[Path] = None # Optional if not using sondes
    output_dir: Path = Path("./libradpy_output")
    working_dir: Path = Path("./libradpy_work")

    def __post_init__(self):
        # Basic validation
        if not self.libradtran_bin.is_file():
            raise FileNotFoundError(f"LibRadtran executable not found: {self.libradtran_bin}")
        if not self.libradtran_data.is_dir():
            raise FileNotFoundError(f"LibRadtran data directory not found: {self.libradtran_data}")
        if not self.atmosphere_profile.is_file():
            logger.warning(f"Default atmosphere profile not found: {self.atmosphere_profile}")
        if not self.solar_spectrum.is_file():
            raise FileNotFoundError(f"Solar spectrum file not found: {self.solar_spectrum}")
        if self.radiosonde_base and not self.radiosonde_base.is_dir():
             logger.warning(f"Radiosonde base directory not found: {self.radiosonde_base}")
        # Create output/working dirs
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.working_dir.mkdir(parents=True, exist_ok=True)

@dataclass
class CloudParameters:
    """
    Configuration for cloud properties in uvspec simulations.
    
    This class defines parameters for including clouds in libradtran simulations.
    Multiple cloud layers can be defined with different properties.
    """
    
    enabled: bool = False
    
    # Cloud layer properties (lists for multiple layers)
    # Each element corresponds to one cloud layer
    layer_heights_km: List[Tuple[float, float]] = field(default_factory=list)  # [(bottom1, top1), (bottom2, top2), ...]
    layer_water_content: List[float] = field(default_factory=list)  # [lwc1, lwc2, ...]
    layer_effective_radius_um: List[float] = field(default_factory=list)  # [r_eff1, r_eff2, ...]
    
    # Cloud properties file (alternative to specifying layers)
    cloud_file: Optional[Path] = None
    
    # Cloud optical properties - for more advanced cases
    cloud_optical_properties: str = "mie"  # 'mie', 'hu', 'echam4', etc.
    cloud_overlap: str = "max-random"  # 'max-random', 'maximum', 'random'
    
    def __post_init__(self):
        # Validate cloud configuration
        if self.enabled:
            if self.cloud_file:
                if not self.cloud_file.is_file():
                    logger.warning(f"Cloud file specified but not found: {self.cloud_file}")
            else:
                # Check consistency of layer properties
                lengths = [
                    len(self.layer_heights_km),
                    len(self.layer_water_content),
                    len(self.layer_effective_radius_um)
                ]
                
                if len(set(lengths)) > 1:
                    non_zero_lengths = [l for l in lengths if l > 0]
                    if len(set(non_zero_lengths)) > 1:
                        raise ValueError(
                            f"Inconsistent cloud layer definitions: "
                            f"heights={len(self.layer_heights_km)}, "
                            f"water_content={len(self.layer_water_content)}, "
                            f"effective_radius={len(self.layer_effective_radius_um)}"
                        )

@dataclass
class BRDFCamParameters:
    """
    Configuration for Cox and Munk (1954) ocean BRDF properties.
    """
    enabled: bool = False
    pcl: float = 0.01  # Pigment concentration in mg/m^3 (default 0.01)
    sal: float = 34.3  # Salinity in per mille (default 34.3)
    u10: float = 5.0   # Wind speed in m/s (min 1.0)
    uphi: float = 0.0  # Wind direction in degrees (default 0)
    solar_wind: bool = False  # Use old definition of wind direction
    
    def __post_init__(self):
        # Validate u10 parameter
        if self.enabled and self.u10 < 1.0:
            logger.warning("brdf_cam u10 < 1.0 m/s, will be set to minimum allowed 1.0 m/s")

@dataclass
class BRDFRpvParameters:
    """
    Configuration for Rahman, Pinty, and Verstraete BRDF properties.
    """
    enabled: bool = False
    k: Optional[float] = None      # RPV k parameter
    rho0: Optional[float] = None   # RPV rho0 parameter
    theta: Optional[float] = None  # RPV theta parameter
    sigma: Optional[float] = None  # RPV sigma parameter (for snow)
    t1: Optional[float] = None     # RPV t1 parameter (for snow)
    t2: Optional[float] = None     # RPV t2 parameter (for snow)
    scale: Optional[float] = None  # RPV scaling factor
    
    # File-based RPV parameters
    rpv_file: Optional[Path] = None  # 4-7 column file with RPV parameters
    
    # Library-based RPV parameters
    rpv_library: Optional[str] = None  # Path to RPV library or "IGBP"
    rpv_type: Optional[int] = None     # Surface type number

@dataclass
class AerosolParameters:
    """
    Configuration for aerosol properties in uvspec simulations.
    """
    
    enabled: bool = False
    
    # Simple configuration
    aerosol_type: str = "default"  # 'default', 'rural', 'maritime', 'urban', etc.
    aerosol_visibility_km: Optional[float] = None
    aerosol_angstrom_parameters: Optional[Tuple[float, float]] = None  # (alpha, beta)
    
    # Advanced configuration via file
    aerosol_file: Optional[Path] = None
    
    # Aerosol optical properties
    aerosol_optical_properties: str = "default"  # 'default', 'mie'
    
    def __post_init__(self):
        # Validation
        if self.enabled:
            if self.aerosol_file and not self.aerosol_file.is_file():
                logger.warning(f"Aerosol file specified but not found: {self.aerosol_file}")

@dataclass
class SimulationDefaults:
    rte_solver: str = "twostr"
    mol_abs_param: str = "lowtran per_nm"
    source: str = "solar"  # 'solar' or 'thermal'
    wavelength_nm: List[Union[int, float]] = field(default_factory=lambda: [400, 3600])
    output_columns: List[str] = field(default_factory=lambda: ["sza", "eglo", "eup", "albedo"])
    output_altitudes_km: List[float] = field(default_factory=lambda: [0.0])
    albedo_type: str = "const" # 'const', 'file', 'library'
    albedo_value: Optional[float] = 0.85
    albedo_file: Optional[Path] = None
    albedo_library: Optional[str] = None
    brdf_type: str = "lambertian" # 'lambertian', 'rpv'
    brdf_rpv_type: Optional[int] = None
    surface_temperature_k: Optional[float] = 273.15
    ozone_du: Optional[float] = 300.0
    h2o_mm: Optional[float] = 2.0 # Used if h2o_source is 'fixed'
    h2o_source: str = "radiosonde" # 'fixed' or 'radiosonde'
    aerosol_type: str = "default" # 'default', 'none', 'custom'
    viewing_geometry: str = "nadir"
    mol_modify: Optional[str] = None
    umu: Optional[List[float]] = None
    phi: Optional[List[float]] = None
    
    # New parameters
    clouds: CloudParameters = field(default_factory=CloudParameters)
    aerosols: AerosolParameters = field(default_factory=AerosolParameters)
    brdf_cam: BRDFCamParameters = field(default_factory=BRDFCamParameters)
    brdf_rpv: BRDFRpvParameters = field(default_factory=BRDFRpvParameters)
    
    # Spectral handling
    integrate_wavelength: bool = False  # Whether to integrate over wavelength range
    correlated_k: bool = False  # Enable correlated-k distribution
    mol_tau_file: Optional[Path] = None  # Optional molecular optical depth file
    transmittance_source: str = "default"  # 'default', 'user', 'lowtran', etc.
    
    def __post_init__(self):
        if len(self.wavelength_nm) != 2:
            raise ValueError("wavelength_nm must contain [min, max]")
        if not self.output_altitudes_km:
            raise ValueError("output_altitudes_km cannot be empty.")
        if self.source not in ["solar", "thermal"]:
            raise ValueError(f"source must be 'solar' or 'thermal', got '{self.source}'")
        self.output_altitudes_km = sorted(list(set(self.output_altitudes_km)))
        
        # Handle transition from old BRDF parameters to new ones
        if hasattr(self, 'brdf_type') and self.brdf_type != 'lambertian':
            if self.brdf_type == 'rpv':
                # Handle RPV BRDF
                if not self.brdf_rpv.enabled:
                    self.brdf_rpv.enabled = True
                    
                if hasattr(self, 'brdf_rpv_type') and self.brdf_rpv_type is not None:
                    if self.brdf_rpv.rpv_type is None:
                        self.brdf_rpv.rpv_type = self.brdf_rpv_type
            elif self.brdf_type == 'cam':
                # Handle Cox and Munk BRDF
                if not self.brdf_cam.enabled:
                    self.brdf_cam.enabled = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert the SimulationDefaults object to a dictionary.
        
        This method converts all non-None attributes to a dictionary, 
        handling nested dataclass objects by converting them to dictionaries as well.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the SimulationDefaults object
        """
        result = {}
        for field_obj in fields(self):
            field_name = field_obj.name
            field_value = getattr(self, field_name)
            
            # Skip None values
            if field_value is None:
                continue
                
            # Handle nested dataclasses
            if is_dataclass(field_value):
                # Convert nested dataclass to dict recursively
                field_dict = {}
                for nested_field in fields(field_value):
                    nested_name = nested_field.name
                    nested_value = getattr(field_value, nested_name)
                    if nested_value is not None:  # Skip None values
                        field_dict[nested_name] = nested_value
                result[field_name] = field_dict
            else:
                # Regular attribute
                result[field_name] = field_value
                
        return result

@dataclass
class ExecutionConfig:
    max_workers: Optional[int] = min(8, os.cpu_count() or 1)
    cleanup_temp_files: bool = True
    debug_mode: bool = False
    timeout_seconds: int = 300

@dataclass
class OutputConfig:
    filename_prefix: str = "libradpy_sim"
    filename_suffix: str = "_results.nc"
    netcdf_encoding: Dict[str, Any] = field(default_factory=lambda: {"zlib": True, "complevel": 5})
    variable_encoding: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SimulationConfig:
    paths: PathsConfig
    simulation_defaults: SimulationDefaults
    execution: ExecutionConfig
    output: OutputConfig

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> 'SimulationConfig':
        """Loads configuration from a YAML file."""
        yaml_path = Path(yaml_path)
        if not yaml_path.is_file():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")

        with open(yaml_path, 'r') as f:
            raw_config = yaml.safe_load(f)

        # Recursively instantiate dataclasses
        return cls._dict_to_dataclass(raw_config, cls)

    @classmethod
    def _dict_to_dataclass(cls, data: Dict[str, Any], dataclass_type: type) -> Any:
        """Helper to recursively convert dict to nested dataclasses."""
        field_types = {f.name: f.type for f in fields(dataclass_type)}
        init_args = {}
        for name, value in data.items():
            if name not in field_types:
                logger.warning(f"Ignoring unknown config key: {name} in section {dataclass_type.__name__}")
                continue

            field_type = field_types[name]
            # Handle Optional types
            if hasattr(field_type, '__origin__') and field_type.__origin__ is Union:
                 # Get the actual type, assuming Optional[T] -> T
                 possible_types = [arg for arg in field_type.__args__ if arg is not type(None)]
                 if len(possible_types) == 1:
                     field_type = possible_types[0]
                 else:
                     # Handle more complex Unions if necessary, for now assume Optional
                     pass # Keep original field_type

            if is_dataclass(field_type) and isinstance(value, dict):
                init_args[name] = cls._dict_to_dataclass(value, field_type)
            elif hasattr(field_type, '__origin__') and field_type.__origin__ is list:
                 # Basic list handling, could be more specific
                 init_args[name] = value
            elif field_type is Path:
                 init_args[name] = Path(value) if value is not None else None
            else:
                # Attempt direct type conversion (e.g., str to int/float if needed)
                try:
                    init_args[name] = field_type(value) if value is not None else None
                except (TypeError, ValueError):
                     init_args[name] = value # Keep original if conversion fails

        # Add default values for missing keys
        for f in fields(dataclass_type):
            if f.name not in init_args and f.default is not field(default_factory=lambda: None).default:
                 if callable(f.default_factory):
                     init_args[f.name] = f.default_factory()
                 else:
                     init_args[f.name] = f.default


        try:
             return dataclass_type(**init_args)
        except TypeError as e:
             logger.error(f"Error creating dataclass {dataclass_type.__name__}: {e}")
             logger.error(f"Arguments provided: {init_args}")
             raise


# --- Helper function to get config ---
_DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config" / "default_simulation.yaml"

def load_config(config_path: Optional[Union[str, Path]] = None) -> SimulationConfig:
    """Loads simulation config, using default if path is not provided."""
    if config_path is None:
        config_path = _DEFAULT_CONFIG_PATH
        logger.info(f"No config path provided, using default: {config_path}")
    else:
        config_path = Path(config_path)
        logger.info(f"Loading configuration from: {config_path}")

    try:
        config = SimulationConfig.from_yaml(config_path)
        logger.info("Configuration loaded successfully.")
        # Set logging level based on config after loading
        log_level = logging.DEBUG if config.execution.debug_mode else logging.INFO
        logging.basicConfig(level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        logger.setLevel(log_level) # Ensure logger for this module respects level
        return config
    except Exception as e:
        logger.exception(f"Failed to load or validate configuration from {config_path}: {e}")
        raise

# Example usage within the module (optional)
# if __name__ == "__main__":
#     try:
#         # Test loading default config
#         default_config = load_config()
#         print("Default Config Loaded:")
#         print(default_config)

#         # Create a dummy user config for testing override
#         dummy_path = Path("./dummy_config.yaml")
#         dummy_data = {
#             'paths': {'libradtran_bin': default_config.paths.libradtran_bin}, # Only specify one required path
#             'simulation_defaults': {'rte_solver': 'disort'},
#             'execution': {'max_workers': 4}
#             # Missing sections will use defaults if possible, or raise errors
#         }
#         # This will likely fail without all required paths, demonstrating validation
#         # with open(dummy_path, 'w') as f:
#         #     yaml.dump(dummy_data, f)
#         # user_config = load_config(dummy_path)
#         # print("\nUser Config Loaded:")
#         # print(user_config)
#         # dummy_path.unlink()

#     except Exception as e:
#         print(f"\nError during config loading test: {e}")
