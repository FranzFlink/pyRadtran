# pyradtran/__init__.py
"""
pyRadtran: radiative transfer with libRadtran, driven from xarray.

pyRadtran wraps the ``uvspec`` program of libRadtran.  Describe where and
when to simulate as an :class:`xarray.Dataset` and how in a small YAML
configuration; the simulated irradiances, radiances or brightness
temperatures come back as an :class:`xarray.Dataset` on the same
dimensions.  pyRadtran writes the ``uvspec`` input file for every point,
runs the points in parallel, parses the output and records how each
result was produced.

Examples
--------
Three steps:

1. Prepare a dataset with ``time``, ``latitude`` and ``longitude``; any
   further variable (here a measured surface albedo) can be fed into the
   simulation point by point:

>>> import pandas as pd
>>> import xarray as xr
>>> import pyradtran
>>> from pyradtran import Var
>>> times = ["2022-03-28 10:00", "2022-03-28 12:00", "2022-03-28 14:00"]
>>> ds = xr.Dataset(
...     {"albedo": ("time", [0.85, 0.80, 0.06])},
...     coords={"time": pd.to_datetime(times), "latitude": 78.9, "longitude": 11.9},
... )

2. Run ``uvspec`` for every point through the ``ds.pyradtran`` accessor.
   The YAML file holds the settings shared by all points (solver,
   spectral range, …); ``params`` adds per-run and per-point options:

>>> result = ds.pyradtran.run(  # doctest: +SKIP
...     config_path="config/my_simulation.yaml",
...     params={"albedo": Var("albedo")},
... )

3. Work with the result, an :class:`xarray.Dataset` with the input
   dimensions plus ``wavelength`` and ``altitude``:

>>> result  # doctest: +SKIP
<xarray.Dataset> Size: 11kB
Dimensions:     (time: 3, wavelength: 101, altitude: 1)
Coordinates:
  * time        (time) datetime64[ns] 24B 2022-03-28T10:00:00 ... 2022-03-28T...
  * wavelength  (wavelength) float64 808B 500.0 501.0 502.0 ... 599.0 600.0
  * altitude    (altitude) int64 8B 0
Data variables:
    sza         (time, wavelength, altitude) float64 2kB 77.04 77.04 ... 79.36
    eglo        (time, wavelength, altitude) float64 2kB 351.5 345.1 ... 221.0
    eup         (time, wavelength, altitude) float64 2kB 298.8 293.3 ... 13.26
    albedo      (time, wavelength, altitude) float64 2kB 0.85 0.85 ... 0.06 0.06
    status      (time) int64 24B 0 0 0
Attributes:
    generated_by:              pyradtran
    pyradtran_version:         0.2.0
    pyradtran_params:          {"albedo": "Var(name='albedo')"}
    ...
>>> result.eup.sel(wavelength=550).squeeze().values.round(1)  # doctest: +SKIP
array([287. , 277.1,  14.7])

Before the first run, store the libRadtran paths once with
:func:`~pyradtran.config.save_master_config`.
``ds.pyradtran.explain()`` previews the ``uvspec`` input file without
running anything; ``pyradtran.quick_start()`` prints these steps.

Notes
-----
The public API, grouped by task.  Everything listed here is importable
from ``pyradtran``.

**Running simulations**

* :class:`~pyradtran.interface.PyRadtranAccessor`: the ``ds.pyradtran``
  accessor (``run``, ``explain``, …).
* :func:`~pyradtran.interface.run_pyradtran_simulation`: file-to-file
  pipeline (CSV / NetCDF in, NetCDF out).
* :func:`~pyradtran.interface.execute_simulation_batch`: low-level
  parallel batch driver.
* :class:`~pyradtran.core.Simulation`: one ``uvspec`` call for one point.

**Configuration**: three YAML layers, later wins (see
:func:`~pyradtran.config.load_config`).

* :func:`~pyradtran.config.load_config`: load and merge the three layers.
* :func:`~pyradtran.config.save_master_config`: store machine-specific
  paths once.
* :class:`~pyradtran.config.SimulationConfig`: the merged configuration,
  with its sections :class:`~pyradtran.config.PathsConfig` and
  :class:`~pyradtran.config.SimulationDefaults`.
* :func:`~pyradtran.config.list_atmosphere_profiles`,
  :func:`~pyradtran.config.list_solar_spectra`: short names for standard
  atmospheres and solar spectra (``ATMOSPHERE_PROFILES``,
  ``SOLAR_SPECTRA``).
* :func:`~pyradtran.config.create_example_config`: write a template YAML.

**Parameters**: any ``uvspec`` option, per run or per point.

* :class:`~pyradtran.params.Var`: take an option's value from a dataset
  variable, point by point.
* :class:`~pyradtran.params.Raw`: pass a value through unvalidated.
* :func:`~pyradtran.params.describe`,
  :func:`~pyradtran.params.search_options`: look up and search libRadtran
  options.
* ``REGISTRY``, :class:`~pyradtran.params.ParamSpec`,
  :class:`~pyradtran.params.ParamResolver`: the curated option registry
  and its resolution machinery.

**Atmosphere and clouds**

* :func:`~pyradtran.era5.normalize_era5`,
  :func:`~pyradtran.era5.era5_atmosphere_file`,
  :func:`~pyradtran.era5.cloud_profiles`,
  :func:`~pyradtran.era5.recommend_atmosphere`: ERA5 profiles as
  atmosphere and cloud input, and a matching background atmosphere.
* :class:`~pyradtran.clouds.CloudLayer`,
  :class:`~pyradtran.clouds.CloudGenerator`,
  :class:`~pyradtran.clouds.CloudFileWriter`,
  :func:`~pyradtran.clouds.generate_cloud_file_from_era5`: build
  libRadtran cloud files.
* :class:`~pyradtran.utils.RadiosondeFinder`: pick the local sounding
  closest in time.
* :class:`~pyradtran.io.ERA5AtmosphereGenerator`: legacy wrapper for ERA5
  atmosphere files.

**Instrument channels**

* :func:`~pyradtran.channels.convolve_channels`: average spectra over
  spectral response functions.
* :func:`~pyradtran.channels.brightness_temperature`: radiance to
  brightness temperature.

**Output handling**

* :class:`~pyradtran.io.OutputParser`, :class:`~pyradtran.io.ParsedOutput`,
  :class:`~pyradtran.io.OutputType`: read ``uvspec`` output tables.
* :class:`~pyradtran.io.OutputToXarray`: reshape parsed output into
  datasets.
* :class:`~pyradtran.io.NetCDFSaver`: write results with provenance.
* :class:`~pyradtran.io.InputDataLoader`: read input points from CSV /
  NetCDF.

**Exceptions**: all derive from
:class:`~pyradtran.exceptions.PyRadtranError`:
:class:`~pyradtran.exceptions.ConfigurationError`,
:class:`~pyradtran.exceptions.InputGenerationError`,
:class:`~pyradtran.exceptions.UvspecExecutionError`,
:class:`~pyradtran.exceptions.OutputParsingError`,
:class:`~pyradtran.exceptions.ValidationError`,
:class:`~pyradtran.exceptions.RadiosondeError`.

See Also
--------
pyradtran.interface : The accessor and the batch driver.
pyradtran.config : Configuration system.
pyradtran.params : Parameter registry and ``Var`` / ``Raw``.
"""

__version__ = "0.2.0"

import logging

# Package logger without output; handlers belong to the application
logging.getLogger(__name__).addHandler(logging.NullHandler())

from .channels import brightness_temperature, convolve_channels  # noqa: E402
from .clouds import (  # noqa: E402
    CloudFileWriter,
    CloudGenerator,
    CloudLayer,
    generate_cloud_file_from_era5,
)
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
    RadiosondeError,
    UvspecExecutionError,
    ValidationError,
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
    RadiosondeAtmosphereGenerator,
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
    "RadiosondeAtmosphereGenerator",
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
    "ValidationError",
    "RadiosondeError",
    # Clouds
    "CloudGenerator",
    "CloudFileWriter",
    "CloudLayer",
    "generate_cloud_file_from_era5",
]


def get_version():
    """Return the package version string (``pyradtran.__version__``).

    Returns
    -------
    str
        The same value as ``pyradtran.__version__``.

    Examples
    --------
    >>> import pyradtran
    >>> pyradtran.get_version() == pyradtran.__version__
    True
    """
    return __version__


def get_info():
    """Return a summary dict of package capabilities.

    Returns
    -------
    dict
        ``version`` (str), ``has_clouds`` (bool, always ``True``) and
        ``description`` (str).

    Examples
    --------
    >>> import pyradtran
    >>> info = pyradtran.get_info()
    >>> sorted(info)
    ['description', 'has_clouds', 'version']
    >>> info["description"]
    'Python wrapper for libRadtran (uvspec)'
    """
    return {
        "version": __version__,
        "has_clouds": True,
        "description": "Python wrapper for libRadtran (uvspec)",
    }


def quick_start():
    """Print a short getting-started guide to stdout.

    Lists the steps of a first simulation as copyable code: store the
    libRadtran paths with :func:`save_master_config`, build and save a
    configuration with :func:`load_config` / :meth:`SimulationConfig.to_yaml`,
    and run it with ``ds.pyradtran.run(config_path=...)``.

    Examples
    --------
    >>> import pyradtran
    >>> pyradtran.quick_start()  # doctest: +ELLIPSIS
    PyRadtran ... - Quick Start
    ========================================
    0. Save machine-specific paths to the master config (one-time setup):
       import pyradtran
       pyradtran.save_master_config(
           libradtran_bin='/opt/libradtran/bin/uvspec',
           libradtran_data='/opt/libradtran/share/libRadtran/data',
       )
    <BLANKLINE>
    1. Build a simulation config in Python and save it as YAML:
       cfg = pyradtran.load_config()  # starts from master + package defaults
       cfg.simulation_defaults.albedo_value = 0.2
       cfg.to_yaml('config/my_simulation.yaml')
    <BLANKLINE>
    2. Run a simulation:
       result_ds = dataset.pyradtran.run(config_path='config/my_simulation.yaml')
    <BLANKLINE>
    For full documentation, see https://franzflink.github.io/pyRadtran/
    """
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
    print("For full documentation, see https://franzflink.github.io/pyRadtran/")
