# pyradtran/__init__.py
"""
PyRadtran: A unified Python wrapper for libradtran (uvspec).

This package provides a clean, simplified interface to the libradtran radiative
transfer model with seamless integration into the Python scientific ecosystem.

Refactored version with:
- Unified IO system supporting ERA5 atmosphere files
- Cleaned configuration with only used parameters
- Simplified interface with better error handling
- Comprehensive testing support
"""

__version__ = "0.2.0"

import logging

# Configure basic logging for the package
logging.getLogger(__name__).addHandler(logging.NullHandler())

# Import cleaned components
from .config_clean import load_config, SimulationConfig, PathsConfig, SimulationDefaults, create_example_config
from .core_unified import Simulation
from .io_unified import (
    OutputParser,
    OutputToXarray,
    ParsedOutput,
    OutputType,
    InputDataLoader,
    ERA5AtmosphereGenerator,
    NetCDFSaver
)
from .interface_unified import (
    run_pyradtran_simulation, 
    execute_simulation_batch, 
    PyRadtranAccessor
)
from .utils import RadiosondeFinder
from .exceptions import (
    PyRadtranError,
    ConfigurationError,
    InputGenerationError,
    UvspecExecutionError,
    OutputParsingError
)

# Import cloud functionality if available
try:
    from .clouds import CloudGenerator, CloudFileWriter, CloudLayer, generate_cloud_file_from_era5
    _HAS_CLOUDS = True
except ImportError:
    _HAS_CLOUDS = False
    logger = logging.getLogger(__name__)
    logger.warning("Cloud functionality not available")

# Import helper functions if available
try:
    from .helpers import (
        configure_surface,
        configure_cloud,
        add_cloud_layer,
        configure_aerosol
    )
    _HAS_HELPERS = True
except ImportError:
    _HAS_HELPERS = False

# Expose main components
__all__ = [
    # Core functionality
    'Simulation',
    'run_pyradtran_simulation',
    'execute_simulation_batch',
    'PyRadtranAccessor',
    
    # Configuration
    'load_config',
    'SimulationConfig',
    'PathsConfig', 
    'SimulationDefaults',
    'create_example_config',
    
    # I/O components
    'OutputParser',
    'OutputToXarray',
    'ParsedOutput',
    'OutputType',
    'InputDataLoader',
    'ERA5AtmosphereGenerator',
    'NetCDFSaver',
    
    # Utilities
    'RadiosondeFinder',
    
    # Exceptions
    'PyRadtranError',
    'ConfigurationError',
    'InputGenerationError',
    'UvspecExecutionError',
    'OutputParsingError',
]

# Add cloud components if available
if _HAS_CLOUDS:
    __all__.extend([
        'CloudGenerator',
        'CloudFileWriter', 
        'CloudLayer',
        'generate_cloud_file_from_era5'
    ])

# Add helper components if available
if _HAS_HELPERS:
    __all__.extend([
        'configure_surface',
        'configure_cloud',
        'add_cloud_layer',
        'configure_aerosol'
    ])

def get_version():
    """Return the package version."""
    return __version__

def get_info():
    """Return package information."""
    return {
        'version': __version__,
        'has_clouds': _HAS_CLOUDS,
        'has_helpers': _HAS_HELPERS,
        'description': 'Unified Python wrapper for libradtran (uvspec)'
    }

def quick_start():
    """Print quick start information."""
    print(f"PyRadtran {__version__} - Quick Start")
    print("=" * 40)
    print("1. Create a configuration file:")
    print("   import pyradtran")
    print("   pyradtran.create_example_config('my_config.yaml')")
    print()
    print("2. Load and run simulation:")
    print("   config = pyradtran.load_config('my_config.yaml')")
    print("   result = pyradtran.run_pyradtran_simulation('input_data.csv')")
    print()
    print("3. Use with xarray datasets:")
    print("   result_ds = dataset.pyradtran.run()")
    print()
    print("For full documentation, see the examples in the notebooks/ directory.")

# Make sure xarray accessor is registered
try:
    import xarray as xr
    # The accessor will be registered when interface_unified is imported
    logger = logging.getLogger(__name__)
    logger.debug("xarray accessor 'pyradtran' registered")
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("xarray not available - dataset accessor will not work")
