# pyradtran/config.py - UNIFIED VERSION
"""
Configuration system for pyradtran - REFACTORED VERSION.

This file has been refactored to include only parameters that are actually used.
Original version backed up as config.py.backup.

Key changes:
- Removed 75+ unused parameters
- Simplified cloud configuration
- Cleaner validation
- Only essential parameters remain

For migration guide, see REFACTORING_SUMMARY.md
"""

import yaml
from pathlib import Path
from dataclasses import dataclass, field, fields, is_dataclass
from typing import List, Dict, Optional, Any, Union, Tuple
import logging
import os

logger = logging.getLogger(__name__)


@dataclass
class PathsConfig:
    """Essential paths for LibRadtran execution."""
    libradtran_bin: Path  # Path to uvspec executable
    libradtran_data: Path  # Path to LibRadtran data directory
    atmosphere_profile: Path  # Default atmosphere profile file
    solar_spectrum: Path  # Solar spectrum file
    radiosonde_base: Optional[Path] = None  # Optional radiosonde directory
    output_dir: Path = Path("./pyradtran_output")
    working_dir: Path = Path("./pyradtran_work")

    def __post_init__(self):
        # Validate essential paths
        if not self.libradtran_bin.is_file():
            raise FileNotFoundError(f"LibRadtran executable not found: {self.libradtran_bin}")
        if not self.libradtran_data.is_dir():
            raise FileNotFoundError(f"LibRadtran data directory not found: {self.libradtran_data}")
        if not self.atmosphere_profile.is_file():
            logger.warning(f"Default atmosphere profile not found: {self.atmosphere_profile}")
        if not self.solar_spectrum.is_file():
            raise FileNotFoundError(f"Solar spectrum file not found: {self.solar_spectrum}")
        
        # Create output/working directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.working_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class CloudParameters:
    """Simple cloud configuration - only essential parameters."""
    enabled: bool = False
    
    # Cloud type and source
    cloud_type: str = "wc"  # 'wc' (water), 'ic' (ice), 'mixed'
    cloud_source: str = "parametric"  # 'parametric', 'file', 'era5'
    
    # Simple parametric cloud (single layer)
    layer_bottom_km: float = 1.0
    layer_top_km: float = 2.0
    water_content_g_m3: float = 0.1
    ice_content_g_m3: float = 0.0
    effective_radius_um: float = 10.0
    cloud_fraction: float = 1.0
    
    # File-based clouds
    wc_file: Optional[Path] = None
    ic_file: Optional[Path] = None
    
    # ERA5 cloud generation
    era5_dataset: Optional[Any] = None  # xarray Dataset (not serializable in YAML)
    era5_time: Optional[str] = None
    era5_lat: Optional[float] = None
    era5_lon: Optional[float] = None
    
    def __post_init__(self):
        if self.enabled and self.cloud_source == "file":
            if self.cloud_type in ["wc", "mixed"] and self.wc_file and not self.wc_file.exists():
                logger.warning(f"Water cloud file not found: {self.wc_file}")
            if self.cloud_type in ["ic", "mixed"] and self.ic_file and not self.ic_file.exists():
                logger.warning(f"Ice cloud file not found: {self.ic_file}")


@dataclass
class SimulationDefaults:
    """Core simulation parameters - only the ones actually used."""
    
    # Essential LibRadtran parameters
    rte_solver: str = "twostr"  # RTE solver: 'twostr', 'disort', 'fdisort1', 'rodents', etc.
    mol_abs_param: str = "lowtran per_nm"  # Molecular absorption: 'lowtran', 'reptran', 'kato', etc.
    source: str = "solar"  # Radiation source: 'solar' or 'thermal'
    
    # Spectral configuration
    wavelength_nm: List[Union[int, float]] = field(default_factory=lambda: [400, 3600])
    integrate_wavelength: bool = False  # Whether to integrate over wavelength range
    
    # Output configuration
    output_columns: List[str] = field(default_factory=lambda: ["sza", "eglo", "eup", "albedo"])
    output_altitudes_km: List[float] = field(default_factory=lambda: [0.0])
    
    # Surface properties
    albedo_value: float = 0.85  # Surface albedo (0-1)
    surface_temperature_k: Optional[float] = None  # Surface temperature in Kelvin
    
    # Atmospheric composition (commonly used)
    ozone_du: Optional[float] = 300.0  # Total ozone column in Dobson Units
    h2o_mm: Optional[float] = 2.0  # Precipitable water in mm
    h2o_source: str = "fixed"  # H2O source: 'fixed' or 'radiosonde'
    
    # Cloud configuration
    clouds: CloudParameters = field(default_factory=CloudParameters)
    
    # Viewing geometry (simplified)
    viewing_geometry: str = "nadir"  # 'nadir' or 'custom'
    sza: Optional[float] = None  # Solar zenith angle (degrees) - if None, calculated from time/location
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if len(self.wavelength_nm) != 2:
            raise ValueError("wavelength_nm must contain [min, max]")
        if not self.output_altitudes_km:
            raise ValueError("output_altitudes_km cannot be empty")
        if self.source not in ["solar", "thermal"]:
            raise ValueError(f"source must be 'solar' or 'thermal', got '{self.source}'")
        if not (0 <= self.albedo_value <= 1):
            raise ValueError(f"albedo_value must be between 0 and 1, got {self.albedo_value}")
        
        # Sort and deduplicate altitude levels
        self.output_altitudes_km = sorted(list(set(self.output_altitudes_km)))


@dataclass
class ExecutionConfig:
    """Execution parameters."""
    max_workers: Optional[int] = min(8, os.cpu_count() or 1)
    cleanup_temp_files: bool = False  # Keep temp files for debugging
    debug_mode: bool = False
    timeout_seconds: int = 300


@dataclass
class OutputConfig:
    """Output file configuration."""
    filename_prefix: str = "pyradtran_sim"
    filename_suffix: str = "_results.nc"
    netcdf_encoding: Dict[str, Any] = field(default_factory=lambda: {"zlib": True, "complevel": 5})


@dataclass
class SimulationConfig:
    """Main configuration class containing all settings."""
    paths: PathsConfig
    simulation_defaults: SimulationDefaults
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> 'SimulationConfig':
        """Load configuration from a YAML file."""
        yaml_path = Path(yaml_path)
        if not yaml_path.is_file():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")

        with open(yaml_path, 'r') as f:
            raw_config = yaml.safe_load(f)

        return cls._dict_to_dataclass(raw_config, cls)

    @classmethod
    def _dict_to_dataclass(cls, data: Dict[str, Any], dataclass_type: type) -> Any:
        """Convert dictionary to dataclass recursively."""
        field_types = {f.name: f.type for f in fields(dataclass_type)}
        init_args = {}
        
        for name, value in data.items():
            if name not in field_types:
                logger.warning(f"Ignoring unknown config parameter: {name} in {dataclass_type.__name__}")
                continue

            field_type = field_types[name]
            
            # Handle Optional types
            if hasattr(field_type, '__origin__') and field_type.__origin__ is Union:
                possible_types = [arg for arg in field_type.__args__ if arg is not type(None)]
                if len(possible_types) == 1:
                    field_type = possible_types[0]

            if is_dataclass(field_type) and isinstance(value, dict):
                init_args[name] = cls._dict_to_dataclass(value, field_type)
            elif field_type is Path:
                init_args[name] = Path(value) if value is not None else None
            else:
                try:
                    init_args[name] = field_type(value) if value is not None else None
                except (TypeError, ValueError):
                    init_args[name] = value

        # Add default values for missing keys
        for f in fields(dataclass_type):
            if f.name not in init_args:
                if f.default is not dataclass_type.__dataclass_fields__[f.name].default:
                    init_args[f.name] = f.default
                elif f.default_factory is not dataclass_type.__dataclass_fields__[f.name].default_factory:
                    init_args[f.name] = f.default_factory()

        try:
            return dataclass_type(**init_args)
        except TypeError as e:
            logger.error(f"Error creating dataclass {dataclass_type.__name__}: {e}")
            logger.error(f"Arguments provided: {init_args}")
            raise

    def get_used_parameters(self) -> Dict[str, Any]:
        """Get a dictionary of all parameters that are actually used."""
        return {
            'paths': {
                'libradtran_bin': str(self.paths.libradtran_bin),
                'libradtran_data': str(self.paths.libradtran_data),
                'atmosphere_profile': str(self.paths.atmosphere_profile),
                'solar_spectrum': str(self.paths.solar_spectrum),
                'radiosonde_base': str(self.paths.radiosonde_base) if self.paths.radiosonde_base else None,
                'output_dir': str(self.paths.output_dir),
                'working_dir': str(self.paths.working_dir)
            },
            'simulation_defaults': {
                'rte_solver': self.simulation_defaults.rte_solver,
                'mol_abs_param': self.simulation_defaults.mol_abs_param,
                'source': self.simulation_defaults.source,
                'wavelength_nm': self.simulation_defaults.wavelength_nm,
                'integrate_wavelength': self.simulation_defaults.integrate_wavelength,
                'output_columns': self.simulation_defaults.output_columns,
                'output_altitudes_km': self.simulation_defaults.output_altitudes_km,
                'albedo_value': self.simulation_defaults.albedo_value,
                'surface_temperature_k': self.simulation_defaults.surface_temperature_k,
                'ozone_du': self.simulation_defaults.ozone_du,
                'h2o_mm': self.simulation_defaults.h2o_mm,
                'h2o_source': self.simulation_defaults.h2o_source,
                'viewing_geometry': self.simulation_defaults.viewing_geometry,
                'sza': self.simulation_defaults.sza,
                'clouds': {
                    'enabled': self.simulation_defaults.clouds.enabled,
                    'cloud_type': self.simulation_defaults.clouds.cloud_type,
                    'cloud_source': self.simulation_defaults.clouds.cloud_source,
                    'layer_bottom_km': self.simulation_defaults.clouds.layer_bottom_km,
                    'layer_top_km': self.simulation_defaults.clouds.layer_top_km,
                    'water_content_g_m3': self.simulation_defaults.clouds.water_content_g_m3,
                    'ice_content_g_m3': self.simulation_defaults.clouds.ice_content_g_m3,
                    'effective_radius_um': self.simulation_defaults.clouds.effective_radius_um,
                    'cloud_fraction': self.simulation_defaults.clouds.cloud_fraction,
                    'wc_file': str(self.simulation_defaults.clouds.wc_file) if self.simulation_defaults.clouds.wc_file else None,
                    'ic_file': str(self.simulation_defaults.clouds.ic_file) if self.simulation_defaults.clouds.ic_file else None
                }
            },
            'execution': {
                'max_workers': self.execution.max_workers,
                'cleanup_temp_files': self.execution.cleanup_temp_files,
                'debug_mode': self.execution.debug_mode,
                'timeout_seconds': self.execution.timeout_seconds
            },
            'output': {
                'filename_prefix': self.output.filename_prefix,
                'filename_suffix': self.output.filename_suffix,
                'netcdf_encoding': self.output.netcdf_encoding
            }
        }


# Default configuration path
_DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config" / "clean_simulation.yaml"


def load_config(config_path: Optional[Union[str, Path]] = None) -> SimulationConfig:
    """Load simulation configuration."""
    if config_path is None:
        config_path = _DEFAULT_CONFIG_PATH
        logger.debug(f"No config path provided, using default: {config_path}")
    else:
        config_path = Path(config_path)
        logger.debug(f"Loading configuration from: {config_path}")

    try:
        config = SimulationConfig.from_yaml(config_path)
        logger.debug("Configuration loaded successfully.")
        
        # Set logging level based on config
        log_level = logging.DEBUG if config.execution.debug_mode else logging.INFO
        logging.basicConfig(level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        logger.setLevel(log_level)
        
        return config
    except Exception as e:
        logger.exception(f"Failed to load configuration from {config_path}: {e}")
        raise


def create_example_config(output_path: Union[str, Path]):
    """Create an example configuration file."""
    example_config = {
        'paths': {
            'libradtran_bin': '/path/to/libradtran/bin/uvspec',
            'libradtran_data': '/path/to/libradtran/data',
            'atmosphere_profile': '/path/to/libradtran/data/atmmod/afglus.dat',
            'solar_spectrum': '/path/to/libradtran/data/solar_flux/kurudz_1.0nm.dat',
            'radiosonde_base': '/path/to/radiosonde/data',  # Optional
            'output_dir': './pyradtran_output',
            'working_dir': './pyradtran_work'
        },
        'simulation_defaults': {
            'rte_solver': 'twostr',
            'mol_abs_param': 'lowtran per_nm',
            'source': 'solar',
            'wavelength_nm': [400, 3600],
            'integrate_wavelength': False,
            'output_columns': ['sza', 'eglo', 'eup', 'albedo'],
            'output_altitudes_km': [0.0],
            'albedo_value': 0.85,
            'surface_temperature_k': 273.15,
            'ozone_du': None,
            'h2o_mm': None,
            'h2o_source': 'fixed',
            'viewing_geometry': 'nadir',
            'sza': None,
            'clouds': {
                'enabled': False,
                'cloud_type': 'wc',
                'cloud_source': 'parametric',
                'layer_bottom_km': 1.0,
                'layer_top_km': 2.0,
                'water_content_g_m3': 0.1,
                'ice_content_g_m3': 0.0,
                'effective_radius_um': 10.0,
                'cloud_fraction': 1.0,
                'wc_file': None,
                'ic_file': None
            }
        },
        'execution': {
            'max_workers': 4,
            'cleanup_temp_files': False,
            'debug_mode': False,
            'timeout_seconds': 300
        },
        'output': {
            'filename_prefix': 'pyradtran_sim',
            'filename_suffix': '_results.nc',
            'netcdf_encoding': {'zlib': True, 'complevel': 5}
        }
    }
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        yaml.dump(example_config, f, default_flow_style=False, indent=2)
    
    print(f"Example configuration created at: {output_path}")


if __name__ == "__main__":
    # Create example config
    create_example_config("./config/clean_simulation.yaml")
