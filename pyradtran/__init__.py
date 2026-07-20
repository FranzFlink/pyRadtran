# pyradtran/__init__.py
"""
pyRadtran — a Python wrapper for libRadtran (``uvspec``).

The canonical workflow is:

1. Prepare an :class:`xarray.Dataset` with ``time``, ``latitude``,
   ``longitude`` coordinates.
2. Call ``ds.pyradtran.run(config_path=...)`` to execute ``uvspec`` in
   parallel for every point.
3. Receive an :class:`xarray.Dataset` of radiative-transfer results.

Configuration is assembled from three layers (later wins):

* Package defaults (``config/clean_simulation.yaml``).
* User master config (``~/.pyradtran/config.yaml``).
* Simulation YAML (passed to :meth:`run`).

Core public API
---------------
:class:`PyRadtranAccessor`
    xarray accessor (``ds.pyradtran``).
:func:`run_pyradtran_simulation`
    Standalone file-to-file pipeline.
:func:`execute_simulation_batch`
    Low-level parallel batch driver.
:func:`load_config`
    Load and merge the three config layers.
"""

__version__ = "0.2.0"

import logging

# Configure basic logging for the package
logging.getLogger(__name__).addHandler(logging.NullHandler())

from .channels import brightness_temperature, convolve_channels  # noqa: E402

# Import cleaned components
from .config import (  # noqa: E402
    ATMOSPHERE_PROFILES,
    SOLAR_SPECTRA,
    PathsConfig,
    SimulationConfig,
    SimulationDefaults,
    create_example_config,
    list_atmosphere_profiles,
    list_solar_spectra,
    load_config,
    save_master_config,
)
from .core import Simulation  # noqa: E402
from .era5 import (  # noqa: E402
    cloud_profiles,
    era5_atmosphere_file,
    normalize_era5,
    recommend_atmosphere,
)
from .exceptions import (  # noqa: E402
    ConfigurationError,
    InputGenerationError,
    OutputParsingError,
    PyRadtranError,
    UvspecExecutionError,
)
from .interface import (  # noqa: E402
    PyRadtranAccessor,
    execute_simulation_batch,
    run_pyradtran_simulation,
)
from .io import (  # noqa: E402
    ERA5AtmosphereGenerator,
    InputDataLoader,
    NetCDFSaver,
    OutputParser,
    OutputToXarray,
    OutputType,
    ParsedOutput,
)
from .params import (  # noqa: E402
    REGISTRY,
    ParamResolver,
    ParamSpec,
    Raw,
    Var,
    describe,
    search_options,
)
from .utils import RadiosondeFinder  # noqa: E402

# Import cloud functionality if available
try:
    from .clouds import (  # noqa: F401
        CloudFileWriter,
        CloudGenerator,
        CloudLayer,
        generate_cloud_file_from_era5,
    )

    _HAS_CLOUDS = True
except ImportError:
    _HAS_CLOUDS = False
    logger = logging.getLogger(__name__)
    logger.warning("Cloud functionality not available")

# Expose main components
__all__ = [
    # Core functionality
    "Simulation",
    "run_pyradtran_simulation",
    "execute_simulation_batch",
    "PyRadtranAccessor",
    # Configuration
    "load_config",
    "SimulationConfig",
    "PathsConfig",
    "SimulationDefaults",
    "create_example_config",
    "save_master_config",
    "SOLAR_SPECTRA",
    "ATMOSPHERE_PROFILES",
    "list_solar_spectra",
    "list_atmosphere_profiles",
    # I/O components
    "OutputParser",
    "OutputToXarray",
    "ParsedOutput",
    "OutputType",
    "InputDataLoader",
    "ERA5AtmosphereGenerator",
    "NetCDFSaver",
    # ERA5 helpers
    "normalize_era5",
    "era5_atmosphere_file",
    "cloud_profiles",
    "recommend_atmosphere",
    # Parameters & channels
    "Var",
    "Raw",
    "describe",
    "search_options",
    "ParamSpec",
    "ParamResolver",
    "REGISTRY",
    "convolve_channels",
    "brightness_temperature",
    # Utilities
    "RadiosondeFinder",
    # Exceptions
    "PyRadtranError",
    "ConfigurationError",
    "InputGenerationError",
    "UvspecExecutionError",
    "OutputParsingError",
]

# Add cloud components if available
if _HAS_CLOUDS:
    __all__.extend(
        [
            "CloudGenerator",
            "CloudFileWriter",
            "CloudLayer",
            "generate_cloud_file_from_era5",
        ]
    )


def get_version():
    """Return the package version string."""
    return __version__


def get_info():
    """Return a summary dict of package capabilities."""
    return {
        "version": __version__,
        "has_clouds": _HAS_CLOUDS,
        "description": "Python wrapper for libRadtran (uvspec)",
    }


def quick_start():
    """Print a short getting-started guide to stdout."""
    print(f"PyRadtran {__version__} - Quick Start")
    print("=" * 40)
    print("0. Save machine-specific paths to the master config (one-time setup):")
    print("   import pyradtran")
    print("   pyradtran.save_master_config(")
    print("       libradtran_bin='/opt/libradtran/bin/uvspec',")
    print("       libradtran_data='/opt/libradtran/share/libRadtran/data',")
    print("   )")
    print()
    print("1. Build a simulation config in Python and save it as YAML:")
    print("   cfg = pyradtran.load_config()  # starts from master + package defaults")
    print("   cfg.simulation_defaults.albedo_value = 0.2")
    print("   cfg.to_yaml('config/my_simulation.yaml')")
    print()
    print("2. Run a simulation:")
    print(
        "   result_ds = dataset.pyradtran.run(config_path='config/my_simulation.yaml')"
    )
    print()
    print("For full documentation, see the notebooks/ directory.")


# Make sure xarray accessor is registered
try:
    import xarray as xr  # noqa: F401

    # The accessor will be registered when interface_unified is imported
    logger = logging.getLogger(__name__)
    logger.debug("xarray accessor 'pyradtran' registered")
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("xarray not available - dataset accessor will not work")
