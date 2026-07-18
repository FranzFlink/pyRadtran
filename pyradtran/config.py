# pyradtran/config.py
"""
Configuration system for pyRadtran.

The configuration is assembled from up to three layers, each overriding
the previous:

1. **Package defaults** — ``config/clean_simulation.yaml`` shipped with
   pyRadtran.
2. **User master config** — ``~/.pyradtran/config.yaml`` (paths to
   libRadtran, preferred solver, etc.).
3. **Simulation config** — the YAML file passed to
   ``ds.pyradtran.run(config_path=...)``.

All settings are represented as :mod:`dataclasses` so they can be
accessed as typed attributes and validated on construction.

Examples
--------
Load the merged configuration and inspect paths:

>>> from pyradtran.config import load_config
>>> cfg = load_config("config/my_simulation.yaml")
>>> cfg.paths.libradtran_bin
PosixPath('/opt/libradtran/2.0.6/bin/uvspec')

See Also
--------
pyradtran.core.Simulation : Consumes a :class:`SimulationConfig`.
pyradtran.interface.PyRadtranAccessor.run : Calls :func:`load_config`
    internally.
"""

import logging
import os
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Catalogs of short names for libRadtran data files
# ---------------------------------------------------------------------------

#: Short-name → (relative path under ``data/``, description)
SOLAR_SPECTRA: Dict[str, Tuple[str, str]] = {
    "kurudz_1.0nm": (
        "solar_flux/kurudz_1.0nm.dat",
        "Kurucz (1985) solar spectrum, 1 nm resolution, 250–10000 nm",
    ),
    "kurudz_0.1nm": (
        "solar_flux/kurudz_0.1nm.dat",
        "Kurucz (1985) solar spectrum, 0.1 nm resolution",
    ),
    "NewGuey2003": (
        "solar_flux/NewGuey2003.dat",
        "Gueymard (2003) high-resolution solar spectrum",
    ),
    "Thekaekara": (
        "solar_flux/Thekaekara.dat",
        "Thekaekara (1974) solar spectrum",
    ),
}

#: Short-name → (relative path under ``data/``, description)
ATMOSPHERE_PROFILES: Dict[str, Tuple[str, str]] = {
    "afglus": ("atmmod/afglus.dat", "US Standard Atmosphere 1976"),
    "afglms": ("atmmod/afglms.dat", "Mid-latitude Summer"),
    "afglmw": ("atmmod/afglmw.dat", "Mid-latitude Winter"),
    "afglt": ("atmmod/afglt.dat", "Tropical"),
    "afglss": ("atmmod/afglss.dat", "Sub-arctic Summer"),
    "afglsw": ("atmmod/afglsw.dat", "Sub-arctic Winter"),
    "mcclams": ("atmmod/mcclams.dat", "McClatchey Mid-latitude Summer (extended)"),
    "mcclamw": ("atmmod/mcclamw.dat", "McClatchey Mid-latitude Winter (extended)"),
    # Trace-gas variants of US Standard
    "afglus_ch4_vmr": ("atmmod/afglus_ch4_vmr.dat", "US Standard – CH4 VMR profile"),
    "afglus_co_vmr": ("atmmod/afglus_co_vmr.dat", "US Standard – CO VMR profile"),
    "afglus_n2o_vmr": ("atmmod/afglus_n2o_vmr.dat", "US Standard – N2O VMR profile"),
    "afglus_n2_vmr": ("atmmod/afglus_n2_vmr.dat", "US Standard – N2 VMR profile"),
    "afglus_no2": ("atmmod/afglus_no2.dat", "US Standard – NO2 profile"),
}


def _resolve_libradtran_shortname(
    value: Optional[Union[str, Path]],
    data_root: Path,
    catalog: Dict[str, Tuple[str, str]],
) -> Optional[Path]:
    """Resolve a short name or path to an absolute :class:`~pathlib.Path`.

    If *value* is a key in *catalog*, the corresponding file inside
    *data_root* is returned.  Otherwise *value* is interpreted as a literal
    path (absolute or relative to CWD).

    Parameters
    ----------
    value : str or Path, optional
        Short name (e.g. ``"kurudz_1.0nm"``) or explicit file path.
    data_root : Path
        The libRadtran ``data/`` directory used as the base for catalog
        resolutions.
    catalog : dict
        One of :data:`SOLAR_SPECTRA` or :data:`ATMOSPHERE_PROFILES`.

    Returns
    -------
    Path or None
    """
    if value is None:
        return None
    s = str(value)
    if s in catalog:
        return data_root / catalog[s][0]
    return Path(s).expanduser()


def list_solar_spectra() -> None:
    """Print a table of available solar spectrum short names."""
    print(f"{'Short name':<20}  Description")
    print("-" * 70)
    for name, (_, desc) in SOLAR_SPECTRA.items():
        print(f"  {name:<18}  {desc}")


def list_atmosphere_profiles() -> None:
    """Print a table of available atmosphere profile short names."""
    print(f"{'Short name':<22}  Description")
    print("-" * 70)
    for name, (_, desc) in ATMOSPHERE_PROFILES.items():
        print(f"  {name:<20}  {desc}")


@dataclass
class PathsConfig:
    """File-system paths required by libRadtran.

    Parameters
    ----------
    libradtran_bin : pathlib.Path
        Absolute path to the ``uvspec`` executable.
    libradtran_data : pathlib.Path
        Absolute path to the libRadtran ``data/`` directory.
    atmosphere_profile : pathlib.Path, optional
        Default atmosphere profile.  Inferred from *libradtran_data* when
        *None*.
    solar_spectrum : pathlib.Path, optional
        Solar spectrum file.  Inferred from *libradtran_data* when *None*.
    radiosonde_base : pathlib.Path, optional
        Root directory for local radiosonde files.
    output_dir : pathlib.Path, default ``"./pyradtran_output"``
        Directory for NetCDF result files.
    working_dir : pathlib.Path, default ``"./pyradtran_work"``
        Scratch directory for temporary ``uvspec`` input/output files.

    Raises
    ------
    FileNotFoundError
        If *libradtran_bin* or *libradtran_data* do not exist.
    """

    libradtran_bin: Path  # Path to uvspec executable
    libradtran_data: Path  # Path to LibRadtran data directory
    atmosphere_profile: Optional[Path] = None  # Default atmosphere profile file
    solar_spectrum: Optional[Path] = None  # Solar spectrum file
    radiosonde_base: Optional[Path] = None  # Optional radiosonde directory
    output_dir: Path = Path("./pyradtran_output")
    working_dir: Path = Path("./pyradtran_work")

    def __post_init__(self):
        # Validate essential paths
        if not self.libradtran_bin.is_file():
            raise FileNotFoundError(
                f"LibRadtran executable not found: {self.libradtran_bin}"
            )
        if not self.libradtran_data.is_dir():
            raise FileNotFoundError(
                f"LibRadtran data directory not found: {self.libradtran_data}"
            )

        # Resolve short names / infer defaults
        self.atmosphere_profile = _resolve_libradtran_shortname(
            self.atmosphere_profile, self.libradtran_data, ATMOSPHERE_PROFILES
        )
        self.solar_spectrum = _resolve_libradtran_shortname(
            self.solar_spectrum, self.libradtran_data, SOLAR_SPECTRA
        )
        if self.atmosphere_profile is None:
            self.atmosphere_profile = self.libradtran_data / "atmmod" / "afglus.dat"
        if self.solar_spectrum is None:
            self.solar_spectrum = (
                self.libradtran_data / "solar_flux" / "kurudz_1.0nm.dat"
            )

        if not self.atmosphere_profile.is_file():
            logger.warning(
                f"Default atmosphere profile not found: {self.atmosphere_profile}"
            )
        if not self.solar_spectrum.is_file():
            raise FileNotFoundError(
                f"Solar spectrum file not found: {self.solar_spectrum}"
            )

        # Create output/working directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.working_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class CloudParameters:
    """Declarative cloud settings used inside :class:`SimulationDefaults`.

    Three *cloud_source* modes are supported:

    ``"parametric"``
        A single homogeneous slab defined by *layer_bottom_km*,
        *layer_top_km*, *water_content_g_m3*, etc.
    ``"file"``
        Pre-computed cloud profile(s) on disk (*wc_file*, *ic_file*).
    ``"era5"``
        Auto-generated from an ERA5 dataset at run time.

    Parameters
    ----------
    enabled : bool, default ``False``
        Enable cloud handling.
    cloud_type : {"wc", "ic", "mixed"}, default ``"wc"``
        Cloud phase.
    cloud_source : {"parametric", "file", "era5"}, default ``"parametric"``
        How the cloud profile is supplied.
    layer_bottom_km, layer_top_km : float
        Vertical extent of the parametric slab (km).
    water_content_g_m3, ice_content_g_m3 : float
        Mass content (g m⁻³).
    effective_radius_um : float, default ``10.0``
        Effective droplet / crystal radius (µm).
    cloud_fraction : float, default ``1.0``
        Cloud fraction (0–1).
    wc_file, ic_file : pathlib.Path, optional
        Paths for file-based clouds.
    era5_dataset : any, optional
        Not serialisable in YAML; set at run time.
    era5_time, era5_lat, era5_lon : str or float, optional
        Selection parameters for the ERA5 source.

    See Also
    --------
    pyradtran.clouds.CloudGenerator : Programmatic cloud-layer creation.
    """

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
            if (
                self.cloud_type in ["wc", "mixed"]
                and self.wc_file
                and not self.wc_file.exists()
            ):
                logger.warning(f"Water cloud file not found: {self.wc_file}")
            if (
                self.cloud_type in ["ic", "mixed"]
                and self.ic_file
                and not self.ic_file.exists()
            ):
                logger.warning(f"Ice cloud file not found: {self.ic_file}")


@dataclass
class SimulationDefaults:
    """Core simulation parameters passed to ``uvspec``.

    Values set here act as defaults and can be overridden per-run via
    *parameter_overrides* or the YAML config cascade.

    Parameters
    ----------
    rte_solver : str, default ``"twostr"``
        Radiative-transfer equation solver (``"twostr"``, ``"disort"``,
        ``"fdisort1"``, ``"rodents"``, …).
    mol_abs_param : str, default ``"lowtran per_nm"``
        Molecular absorption parameterisation.
    source : {"solar", "thermal"}, default ``"solar"``
        Radiation source.
    wavelength_nm : list of float, default ``[400, 3600]``
        Two-element ``[min, max]`` wavelength range (nm).
    integrate_wavelength : bool, default ``False``
        If *True*, ``output_process integrate`` is appended.
    output_columns : list of str
        Column names for ``output_user``.
    output_altitudes_km : list of float
        Altitude levels for ``zout``.
    albedo_value : float, optional
        Surface albedo (0–1).
    surface_temperature_k : float, optional
        Surface temperature (K) for thermal simulations.
    ozone_du : float, optional
        Total ozone column (DU).
    h2o_mm : float, optional
        Precipitable water (mm).
    h2o_source : {"fixed", "radiosonde"}, default ``"fixed"``
        Water-vapour source strategy.
    clouds : CloudParameters
        Nested cloud settings.
    viewing_geometry : str, default ``"nadir"``
        Viewing geometry shortcut.
    sza : float, optional
        Solar zenith angle (°).  Calculated from time/location when
        *None*.
    parameter_overrides : dict
        Raw ``key: value`` pairs appended verbatim to the ``uvspec``
        input file, providing an escape hatch for any libRadtran option
        not covered by the typed fields above.

    Raises
    ------
    ValueError
        On invalid *wavelength_nm* length, unknown *source*, or
        out-of-range *albedo_value*.
    """

    # Essential LibRadtran parameters
    rte_solver: str = (
        "twostr"  # RTE solver: 'twostr', 'disort', 'fdisort1', 'rodents', etc.
    )
    mol_abs_param: str = (
        "lowtran per_nm"  # Molecular absorption: 'lowtran', 'reptran', 'kato', etc.
    )
    source: str = "solar"  # Radiation source: 'solar' or 'thermal'

    # Spectral configuration
    wavelength_nm: List[Union[int, float]] = field(default_factory=lambda: [400, 3600])
    integrate_wavelength: bool = False  # Whether to integrate over wavelength range

    # Output configuration
    output_columns: List[str] = field(
        default_factory=lambda: ["sza", "eglo", "eup", "albedo"]
    )
    output_altitudes_km: List[float] = field(default_factory=lambda: [0.0])

    # Surface properties
    albedo_value: Optional[float] = (
        None  # Surface albedo (0-1). If None, uvspec default is used.
    )
    surface_temperature_k: Optional[float] = None  # Surface temperature in Kelvin

    # Atmospheric composition (commonly used)
    ozone_du: Optional[float] = 300.0  # Total ozone column in Dobson Units
    h2o_mm: Optional[float] = 2.0  # Precipitable water in mm
    h2o_source: str = "fixed"  # H2O source: 'fixed' or 'radiosonde'

    # Cloud configuration
    clouds: CloudParameters = field(default_factory=CloudParameters)

    # Viewing geometry (simplified)
    viewing_geometry: str = "nadir"  # 'nadir' or 'custom'
    sza: Optional[float] = (
        None  # Solar zenith angle (degrees) - if None, calculated from time/location
    )

    # Generic overrides for parameters not covered by strict schema
    parameter_overrides: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.wavelength_nm and len(self.wavelength_nm) != 2:
            raise ValueError("wavelength_nm must contain [min, max]")
        # output_altitudes_km can be empty (defaults to uvspec implicit behavior)
        if self.source not in ["solar", "thermal"]:
            raise ValueError(
                f"source must be 'solar' or 'thermal', got '{self.source}'"
            )
        if self.albedo_value is not None and not (0 <= self.albedo_value <= 1):
            raise ValueError(
                f"albedo_value must be between 0 and 1, got {self.albedo_value}"
            )

        # Sort and deduplicate altitude levels
        self.output_altitudes_km = sorted(list(set(self.output_altitudes_km)))


@dataclass
class ExecutionConfig:
    """Run-time execution settings.

    Parameters
    ----------
    max_workers : int, optional
        Maximum parallel ``uvspec`` processes (capped at CPU count).
    cleanup_temp_files : bool, default ``False``
        Delete scratch files after each run.
    debug_mode : bool, default ``False``
        Enable verbose logging.
    timeout_seconds : int, default ``300``
        Per-simulation ``uvspec`` timeout.
    """

    max_workers: Optional[int] = min(8, os.cpu_count() or 1)
    cleanup_temp_files: bool = False  # Keep temp files for debugging
    debug_mode: bool = False
    timeout_seconds: int = 300


@dataclass
class OutputConfig:
    """NetCDF output-file settings.

    Parameters
    ----------
    filename_prefix : str
        Prefix for auto-generated file names.
    filename_suffix : str
        Suffix (including extension) for auto-generated file names.
    netcdf_encoding : dict
        Passed to :meth:`xarray.Dataset.to_netcdf` as *encoding*.
    """

    filename_prefix: str = "pyradtran_sim"
    filename_suffix: str = "_results.nc"
    netcdf_encoding: Dict[str, Any] = field(
        default_factory=lambda: {"zlib": True, "complevel": 5}
    )


@dataclass
class SimulationConfig:
    """Top-level configuration container.

    Composed of four sections that mirror the YAML structure:

    * :class:`PathsConfig` — file-system paths.
    * :class:`SimulationDefaults` — physics & spectral settings.
    * :class:`ExecutionConfig` — parallelism & debugging.
    * :class:`OutputConfig` — NetCDF output.

    Use :meth:`from_yaml` or :func:`load_config` to construct an
    instance from disk.

    See Also
    --------
    load_config : Recommended entry point (merges three layers).
    """

    paths: PathsConfig
    simulation_defaults: SimulationDefaults
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> "SimulationConfig":
        """Load configuration from a single YAML file.

        Parameters
        ----------
        yaml_path : str or pathlib.Path
            Path to the YAML file.

        Returns
        -------
        SimulationConfig

        Raises
        ------
        FileNotFoundError
            If *yaml_path* does not exist.
        """
        yaml_path = Path(yaml_path)
        if not yaml_path.is_file():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")

        with open(yaml_path, "r") as f:
            raw_config = yaml.safe_load(f)

        return cls._dict_to_dataclass(raw_config, cls)

    @classmethod
    def _dict_to_dataclass(cls, data: Dict[str, Any], dataclass_type: type) -> Any:
        """Recursively convert a nested dictionary to dataclass instances."""
        field_types = {f.name: f.type for f in fields(dataclass_type)}
        init_args = {}

        for name, value in data.items():
            if name not in field_types:
                logger.warning(
                    f"Ignoring unknown config parameter: {name} in {dataclass_type.__name__}"
                )
                continue

            field_type = field_types[name]

            # Handle Optional types
            if hasattr(field_type, "__origin__") and field_type.__origin__ is Union:
                possible_types = [
                    arg for arg in field_type.__args__ if arg is not type(None)
                ]
                if len(possible_types) == 1:
                    field_type = possible_types[0]

            if is_dataclass(field_type) and isinstance(value, dict):
                init_args[name] = cls._dict_to_dataclass(value, field_type)
            elif field_type is Path:
                init_args[name] = (
                    Path(value).expanduser() if value is not None else None
                )
            else:
                try:
                    init_args[name] = field_type(value) if value is not None else None
                except (TypeError, ValueError):
                    init_args[name] = value

        try:
            return dataclass_type(**init_args)
        except TypeError as e:
            logger.error(f"Error creating dataclass {dataclass_type.__name__}: {e}")
            logger.error(f"Arguments provided: {init_args}")
            raise

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the full configuration to a plain nested dict.

        All :class:`~pathlib.Path` objects are converted to strings so
        the result is immediately YAML-serialisable.

        Returns
        -------
        dict
            Nested dictionary mirroring the YAML structure.

        See Also
        --------
        to_yaml : Write the result directly to a file.
        """

        def _convert(obj):
            if is_dataclass(obj):
                return {f.name: _convert(getattr(obj, f.name)) for f in fields(obj)}
            elif isinstance(obj, Path):
                return str(obj)
            elif isinstance(obj, (list, tuple)):
                return [_convert(v) for v in obj]
            elif isinstance(obj, dict):
                return {k: _convert(v) for k, v in obj.items()}
            else:
                return obj

        d = _convert(self)
        # Drop keys that are not YAML-serialisable (e.g. era5_dataset)
        if "simulation_defaults" in d and "clouds" in d["simulation_defaults"]:
            d["simulation_defaults"]["clouds"].pop("era5_dataset", None)
        return d

    def to_yaml(self, path: Union[str, Path]) -> Path:
        """Write the configuration to a YAML file.

        Parameters
        ----------
        path : str or pathlib.Path
            Destination file.  Parent directories are created
            automatically.

        Returns
        -------
        pathlib.Path
            The resolved path to the written file.

        Examples
        --------
        Build a config dict in Python, load it, then persist it:

        >>> cfg = load_config()
        >>> cfg.to_yaml("config/my_simulation.yaml")
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(
                self.to_dict(), f, default_flow_style=False, indent=2, sort_keys=True
            )
        logger.info(f"Configuration written to {path}")
        return path

    def get_used_parameters(self) -> Dict[str, Any]:
        """Return a flat dictionary of all active parameters.

        Useful for logging or embedding in NetCDF attributes.

        Returns
        -------
        dict
        """
        return {
            "paths": {
                "libradtran_bin": str(self.paths.libradtran_bin),
                "libradtran_data": str(self.paths.libradtran_data),
                "atmosphere_profile": str(self.paths.atmosphere_profile),
                "solar_spectrum": str(self.paths.solar_spectrum),
                "radiosonde_base": (
                    str(self.paths.radiosonde_base)
                    if self.paths.radiosonde_base
                    else None
                ),
                "output_dir": str(self.paths.output_dir),
                "working_dir": str(self.paths.working_dir),
            },
            "simulation_defaults": {
                "rte_solver": self.simulation_defaults.rte_solver,
                "mol_abs_param": self.simulation_defaults.mol_abs_param,
                "source": self.simulation_defaults.source,
                "wavelength_nm": self.simulation_defaults.wavelength_nm,
                "integrate_wavelength": self.simulation_defaults.integrate_wavelength,
                "output_columns": self.simulation_defaults.output_columns,
                "output_altitudes_km": self.simulation_defaults.output_altitudes_km,
                "albedo_value": self.simulation_defaults.albedo_value,
                "surface_temperature_k": self.simulation_defaults.surface_temperature_k,
                "ozone_du": self.simulation_defaults.ozone_du,
                "h2o_mm": self.simulation_defaults.h2o_mm,
                "h2o_source": self.simulation_defaults.h2o_source,
                "viewing_geometry": self.simulation_defaults.viewing_geometry,
                "sza": self.simulation_defaults.sza,
                "clouds": {
                    "enabled": self.simulation_defaults.clouds.enabled,
                    "cloud_type": self.simulation_defaults.clouds.cloud_type,
                    "cloud_source": self.simulation_defaults.clouds.cloud_source,
                    "layer_bottom_km": self.simulation_defaults.clouds.layer_bottom_km,
                    "layer_top_km": self.simulation_defaults.clouds.layer_top_km,
                    "water_content_g_m3": self.simulation_defaults.clouds.water_content_g_m3,
                    "ice_content_g_m3": self.simulation_defaults.clouds.ice_content_g_m3,
                    "effective_radius_um": self.simulation_defaults.clouds.effective_radius_um,
                    "cloud_fraction": self.simulation_defaults.clouds.cloud_fraction,
                    "wc_file": (
                        str(self.simulation_defaults.clouds.wc_file)
                        if self.simulation_defaults.clouds.wc_file
                        else None
                    ),
                    "ic_file": (
                        str(self.simulation_defaults.clouds.ic_file)
                        if self.simulation_defaults.clouds.ic_file
                        else None
                    ),
                },
            },
            "execution": {
                "max_workers": self.execution.max_workers,
                "cleanup_temp_files": self.execution.cleanup_temp_files,
                "debug_mode": self.execution.debug_mode,
                "timeout_seconds": self.execution.timeout_seconds,
            },
            "output": {
                "filename_prefix": self.output.filename_prefix,
                "filename_suffix": self.output.filename_suffix,
                "netcdf_encoding": self.output.netcdf_encoding,
            },
        }


# Default configuration path
_DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config" / "clean_simulation.yaml"


def _recursive_update(base: Dict, update: Dict) -> Dict:
    """Recursively merge *update* into *base* (mutates *base*)."""
    for key, value in update.items():
        if isinstance(value, dict) and key in base and isinstance(base[key], dict):
            _recursive_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(config_path: Optional[Union[str, Path]] = None) -> SimulationConfig:
    """Load and merge the three-layer configuration.

    Resolution order (later wins):

    1. Package defaults (``config/clean_simulation.yaml``).
    2. User master config (``~/.pyradtran/config.yaml``).
    3. *config_path* (the simulation-specific YAML).

    Parameters
    ----------
    config_path : str or pathlib.Path, optional
        Simulation-specific YAML.  When *None*, only layers 1 + 2 are
        used.

    Returns
    -------
    SimulationConfig

    Raises
    ------
    FileNotFoundError
        If *config_path* is given but does not exist.
    ConfigurationError
        If the merged dictionary cannot be converted to
        :class:`SimulationConfig`.
    """

    # 1. Start with the default configuration as the base
    # This ensures we have all required fields (like simulation_defaults)
    try:
        with open(_DEFAULT_CONFIG_PATH, "r") as f:
            final_config_dict = yaml.safe_load(f) or {}
    except Exception as e:
        logger.error(
            f"Failed to load default configuration from {_DEFAULT_CONFIG_PATH}: {e}"
        )
        # Fallback to empty dict ? No, this will likely fail later.
        # But let's proceed to try master.
        final_config_dict = {}

    # 2. Update with master configuration (User preferences)
    # This overrides defaults (e.g. paths) with user-specific settings
    master_config_path = Path.home() / ".pyradtran" / "config.yaml"
    if master_config_path.is_file():
        logger.debug(f"Loading master configuration from: {master_config_path}")
        try:
            with open(master_config_path, "r") as f:
                master_config = yaml.safe_load(f) or {}
            _recursive_update(final_config_dict, master_config)
        except Exception as e:
            logger.warning(f"Failed to load master configuration: {e}")

    # 3. Update with specific configuration if provided
    # This overrides everything else for this specific run
    if config_path is not None:
        config_path = Path(config_path)
        logger.debug(f"Loading specific configuration from: {config_path}")
        try:
            with open(config_path, "r") as f:
                specific_config = yaml.safe_load(f) or {}
            _recursive_update(final_config_dict, specific_config)
        except Exception as e:
            logger.error(
                f"Failed to load specific configuration from {config_path}: {e}"
            )
            raise

    try:

        # Now convert to dataclass
        config = SimulationConfig._dict_to_dataclass(
            final_config_dict, SimulationConfig
        )

        logger.debug("Configuration loaded successfully.")

        # Set logging level based on config
        log_level = logging.DEBUG if config.execution.debug_mode else logging.INFO
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        logger.setLevel(log_level)

        return config
    except Exception as e:
        logger.exception(f"Failed to load configuration from {config_path}: {e}")
        raise


def save_master_config(
    libradtran_bin: Union[str, Path],
    libradtran_data: Union[str, Path],
    atmosphere_profile: Optional[Union[str, Path]] = None,
    solar_spectrum: Optional[Union[str, Path]] = None,
    radiosonde_base: Optional[Union[str, Path]] = None,
    output_dir: Union[str, Path] = "./pyradtran_output",
    working_dir: Union[str, Path] = "./pyradtran_work",
    max_workers: Optional[int] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """Write (or update) the user master config at ``~/.pyradtran/config.yaml``.

    The master config is **layer 2** of the three-layer config cascade.
    It is ideal for storing machine-specific paths (libRadtran install
    location, radiosonde archive, …) once so that every individual
    simulation YAML can stay minimal.

    Parameters
    ----------
    libradtran_bin : str or pathlib.Path
        Absolute path to the ``uvspec`` executable.
    libradtran_data : str or pathlib.Path
        Absolute path to the libRadtran ``data/`` directory.
    atmosphere_profile : str or pathlib.Path, optional
        Default atmosphere profile.
    solar_spectrum : str or pathlib.Path, optional
        Solar-spectrum file.
    radiosonde_base : str or pathlib.Path, optional
        Root directory for local radiosonde files.
    output_dir : str or pathlib.Path, default ``"./pyradtran_output"``
    working_dir : str or pathlib.Path, default ``"./pyradtran_work"``
    max_workers : int, optional
        Maximum parallel ``uvspec`` processes.
    extra : dict, optional
        Additional config sections / keys to merge in
        (e.g. ``{'execution': {'debug_mode': True}}``).

    Returns
    -------
    pathlib.Path
        The path to the master config file.

    Examples
    --------
    >>> from pyradtran.config import save_master_config
    >>> save_master_config(
    ...     libradtran_bin="/opt/libradtran/bin/uvspec",
    ...     libradtran_data="/opt/libradtran/share/libRadtran/data",
    ... )
    PosixPath('/home/user/.pyradtran/config.yaml')
    """
    master_dir = Path.home() / ".pyradtran"
    master_dir.mkdir(parents=True, exist_ok=True)
    master_path = master_dir / "config.yaml"

    content: Dict[str, Any] = {
        "paths": {
            "libradtran_bin": str(libradtran_bin),
            "libradtran_data": str(libradtran_data),
        }
    }
    if atmosphere_profile is not None:
        content["paths"]["atmosphere_profile"] = str(atmosphere_profile)
    if solar_spectrum is not None:
        content["paths"]["solar_spectrum"] = str(solar_spectrum)
    if radiosonde_base is not None:
        content["paths"]["radiosonde_base"] = str(radiosonde_base)
    content["paths"]["output_dir"] = str(output_dir)
    content["paths"]["working_dir"] = str(working_dir)

    if max_workers is not None:
        content.setdefault("execution", {})["max_workers"] = max_workers

    if extra:
        _recursive_update(content, extra)

    with open(master_path, "w") as f:
        yaml.dump(content, f, default_flow_style=False, indent=2, sort_keys=True)

    logger.info(f"Master config saved to {master_path}")
    print(f"Master config saved to: {master_path}")
    return master_path


def create_example_config(output_path: Union[str, Path]):
    """Write a commented example YAML config to *output_path*.

    Parameters
    ----------
    output_path : str or pathlib.Path
        Destination file.  Parent directories are created automatically.
    """
    example_config = {
        "paths": {
            "libradtran_bin": "/path/to/libradtran/bin/uvspec",
            "libradtran_data": "/path/to/libradtran/data",
            "atmosphere_profile": "/path/to/libradtran/data/atmmod/afglus.dat",
            "solar_spectrum": "/path/to/libradtran/data/solar_flux/kurudz_1.0nm.dat",
            "radiosonde_base": "/path/to/radiosonde/data",  # Optional
            "output_dir": "./pyradtran_output",
            "working_dir": "./pyradtran_work",
        },
        "simulation_defaults": {
            "rte_solver": "twostr",
            "mol_abs_param": "lowtran per_nm",
            "source": "solar",
            "wavelength_nm": [400, 3600],
            "integrate_wavelength": False,
            "output_columns": ["sza", "eglo", "eup", "albedo"],
            "output_altitudes_km": [0.0],
            "albedo_value": 0.85,
            "surface_temperature_k": 273.15,
            "ozone_du": 300.0,
            "h2o_mm": 2.0,
            "h2o_source": "fixed",
            "viewing_geometry": "nadir",
            "sza": None,
            "clouds": {
                "enabled": False,
                "cloud_type": "wc",
                "cloud_source": "parametric",
                "layer_bottom_km": 1.0,
                "layer_top_km": 2.0,
                "water_content_g_m3": 0.1,
                "ice_content_g_m3": 0.0,
                "effective_radius_um": 10.0,
                "cloud_fraction": 1.0,
                "wc_file": None,
                "ic_file": None,
            },
        },
        "execution": {
            "max_workers": 4,
            "cleanup_temp_files": False,
            "debug_mode": False,
            "timeout_seconds": 300,
        },
        "output": {
            "filename_prefix": "pyradtran_sim",
            "filename_suffix": "_results.nc",
            "netcdf_encoding": {"zlib": True, "complevel": 5},
        },
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        yaml.dump(example_config, f, default_flow_style=False, indent=2)

    print(f"Example configuration created at: {output_path}")


if __name__ == "__main__":
    # Create example config
    create_example_config("./config/clean_simulation.yaml")
