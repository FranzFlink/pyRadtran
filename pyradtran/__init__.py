# libradpy/__init__.py
"""
PyRadtran: A flexible Python wrapper for libradtran (uvspec).

This package provides a user-friendly interface to the libradtran radiative
transfer model, with seamless integration into the Python scientific ecosystem,
particularly xarray.
"""

__version__ = "0.1.0"

import logging

# Configure basic logging for the package
logging.getLogger(__name__).addHandler(logging.NullHandler())

# Make key components easily accessible
from .config import load_config, SimulationConfig, PathsConfig, SimulationDefaults
from .core import Simulation
from .io import (
    OutputParser,
    OutputToXarray,
    ParsedOutput,
    OutputType
)
from .io_old import (
    load_simulation_input_data,
    generate_uvspec_input_content,
    parse_uvspec_output,
    save_results_to_netcdf
)
from .interface import (
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
from .helpers import (
    configure_surface,
    configure_cloud,
    add_cloud_layer,
    configure_aerosol,
    configure_common_scenario,
    configure_spectral_range,
    configure_output_altitudes
)

__version__ = "0.1.0"