# pyradtran/config.py
"""
Configuration system for pyRadtran.

Everything a ``uvspec`` run needs that is *not* per-point data lives in a
:class:`SimulationConfig`: where libRadtran is installed, which solver and
spectral range to use, how many processes to start and how results are
written. The configuration is assembled by :func:`load_config` from up to
three YAML layers, each overriding the previous one key by key:

1. **Package defaults** — ``pyradtran/defaults.yaml`` shipped with
   pyRadtran.
2. **User master config** — ``~/.pyradtran/config.yaml`` (machine-specific
   paths to libRadtran, preferred solver, …), written once with
   :func:`save_master_config`.
3. **Simulation config** — the YAML file passed to :func:`load_config` or
   to ``ds.pyradtran.run(config_path=...)``.

All settings are represented as :mod:`dataclasses`, so they are accessed
as typed attributes (``cfg.simulation_defaults.albedo_value``) and
validated on construction.

.. autosummary::

   load_config
   save_master_config
   SimulationConfig
   PathsConfig
   SimulationDefaults
   CloudParameters
   ExecutionConfig
   OutputConfig
   list_atmosphere_profiles
   list_solar_spectra
   create_example_config

Notes
-----
A simulation YAML mirrors the dataclass structure: four top-level sections
(``paths``, ``simulation_defaults``, ``execution``, ``output``). Every key
is optional — anything left out falls through to the master config and
then to the package defaults — so a typical file only lists what differs
from the defaults:

.. code-block:: yaml

    # config/my_simulation.yaml
    paths:                               # -> PathsConfig
      libradtran_bin: /opt/libRadtran-2.0.6/bin/uvspec
      libradtran_data: /opt/libRadtran-2.0.6/data
      atmosphere_profile: afglsw         # short name or path -> atmosphere_file
      solar_spectrum: kurudz_1.0nm       # short name or path -> source solar <file>
      output_dir: ./pyradtran_output     # NetCDF results
      working_dir: ./pyradtran_work      # uvspec .inp/.out scratch files

    simulation_defaults:                 # -> SimulationDefaults
      rte_solver: disort                 # rte_solver disort
      mol_abs_param: reptran coarse      # mol_abs_param reptran coarse
      source: solar                      # solar | thermal
      wavelength_nm: [400, 2500]         # wavelength 400 2500   (nm)
      integrate_wavelength: false        # true -> output_process integrate
      output_columns: [edir, edn, eup]   # output_user zout lambda edir edn eup
      output_altitudes_km: [0.0, 10.0]   # zout 0.0000 10.0000   (km)
      albedo_value: 0.85                 # albedo 0.85           (0-1)
      surface_temperature_k: 273.15      # sur_temperature (thermal only, K)
      ozone_du: 300.0                    # mol_modify O3 300.0 DU
      h2o_mm: 2.0                        # mol_modify H2O 2.0 MM
      sza: null                          # null -> computed from time/lat/lon
      parameter_overrides:               # any other uvspec option, verbatim
        aerosol_default: true            # aerosol_default

    execution:                           # -> ExecutionConfig
      max_workers: 4                     # parallel uvspec processes
      timeout_seconds: 300               # per uvspec call
      cleanup_temp_files: false

    output:                              # -> OutputConfig
      filename_prefix: pyradtran_sim
      filename_suffix: _results.nc
      netcdf_encoding: {zlib: true, complevel: 5}

Examples
--------
Load the merged configuration (package defaults + master config) and
inspect it:

>>> from pyradtran import load_config
>>> cfg = load_config()  # doctest: +SKIP
>>> cfg.paths.libradtran_bin  # doctest: +SKIP
PosixPath('/opt/libRadtran-2.0.6/bin/uvspec')
>>> cfg.simulation_defaults.wavelength_nm  # doctest: +SKIP
[400, 3600]

Adjust it in Python, save it for later and use it for a run:

>>> cfg.simulation_defaults.rte_solver = "disort"  # doctest: +SKIP
>>> cfg.simulation_defaults.wavelength_nm = [500, 600]  # doctest: +SKIP
>>> cfg.to_yaml("config/my_simulation.yaml")  # doctest: +SKIP
PosixPath('config/my_simulation.yaml')
>>> result = ds.pyradtran.run(config_path="config/my_simulation.yaml")  # doctest: +SKIP

See Also
--------
pyradtran.interface.PyRadtranAccessor.run : Calls :func:`load_config`
    internally; ``params=`` overrides settings for a single run.
pyradtran.interface.PyRadtranAccessor.explain : Preview the ``uvspec``
    input file a configuration produces.
pyradtran.input_builder.InputFileBuilder : Translates a
    :class:`SimulationConfig` into ``uvspec`` input lines.
pyradtran.core.Simulation : Consumes a :class:`SimulationConfig`.
"""

import copy
import logging
import os
import shutil
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml

from .exceptions import ConfigurationError

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
    """Resolve a catalog short name or a literal path to a :class:`~pathlib.Path`.

    If *value* is exactly a key of *catalog*, the corresponding file inside
    *data_root* is returned.  Anything else is interpreted as a literal
    path: ``~`` is expanded, a relative path stays relative (to the current
    working directory).  Existence is not checked here.

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
        *None* when *value* is *None*.

    Examples
    --------
    >>> from pathlib import Path
    >>> data = Path("/opt/libRadtran/data")
    >>> _resolve_libradtran_shortname("afglsw", data, ATMOSPHERE_PROFILES)
    PosixPath('/opt/libRadtran/data/atmmod/afglsw.dat')
    >>> _resolve_libradtran_shortname("my/profile.dat", data, ATMOSPHERE_PROFILES)
    PosixPath('my/profile.dat')
    """
    if value is None:
        return None
    s = str(value)
    if s in catalog:
        return data_root / catalog[s][0]
    return Path(s).expanduser()


def _print_catalog(catalog: Dict[str, Tuple[str, str]], width: int) -> None:
    print(f"{'Short name':<{width + 2}}  Description")
    print("-" * 70)
    for name, (_, desc) in catalog.items():
        print(f"  {name:<{width}}  {desc}")


def list_solar_spectra() -> None:
    """Print a table of available solar spectrum short names.

    Each short name can be used as ``paths.solar_spectrum`` in a YAML
    config (or as the *solar_spectrum* argument of
    :class:`PathsConfig` / :func:`save_master_config`) instead of a full
    path.  It is resolved to a file below ``<libradtran_data>/solar_flux/``
    and ends up on the ``source solar <file>`` line of the ``uvspec``
    input.  The catalog itself is the module-level dict
    ``SOLAR_SPECTRA`` (short name → ``(relative path, description)``).

    When no solar spectrum is configured at all, ``kurudz_1.0nm`` is used.

    See Also
    --------
    list_atmosphere_profiles : The same for atmosphere profiles.
    PathsConfig : Where the short name is resolved.

    Notes
    -----
    The extraterrestrial spectrum sets the spectral resolution that makes
    sense for a run: a 1 nm spectrum combined with a band parameterisation
    such as ``mol_abs_param lowtran`` or ``reptran coarse`` is the usual
    choice; the 0.1 nm spectrum only pays off with a correspondingly fine
    ``wavelength`` grid.

    Examples
    --------
    >>> from pyradtran import list_solar_spectra
    >>> list_solar_spectra()
    Short name            Description
    ----------------------------------------------------------------------
      kurudz_1.0nm        Kurucz (1985) solar spectrum, 1 nm resolution, 250–10000 nm
      kurudz_0.1nm        Kurucz (1985) solar spectrum, 0.1 nm resolution
      NewGuey2003         Gueymard (2003) high-resolution solar spectrum
      Thekaekara          Thekaekara (1974) solar spectrum

    The catalog is a plain dict, so it can be inspected programmatically:

    >>> from pyradtran import SOLAR_SPECTRA
    >>> SOLAR_SPECTRA["NewGuey2003"][0]
    'solar_flux/NewGuey2003.dat'
    """
    _print_catalog(SOLAR_SPECTRA, 18)


def list_atmosphere_profiles() -> None:
    r"""Print a table of available atmosphere profile short names.

    Each short name can be used as ``paths.atmosphere_profile`` in a YAML
    config (or as the *atmosphere_profile* argument of
    :class:`PathsConfig` / :func:`save_master_config`) instead of a full
    path.  It is resolved to a file below ``<libradtran_data>/atmmod/`` and
    ends up on the ``atmosphere_file`` line of the ``uvspec`` input.  The
    catalog itself is the module-level dict ``ATMOSPHERE_PROFILES``
    (short name → ``(relative path, description)``).

    When no profile is configured at all, ``afglus`` is used.

    See Also
    --------
    list_solar_spectra : The same for solar spectra.
    PathsConfig : Where the short name is resolved.
    pyradtran.era5.recommend_atmosphere : Pick a standard profile from
        latitude and season.

    Notes
    -----
    The six AFGL reference atmospheres (Anderson et al., 1986) [1]_ are the
    standard background profiles of libRadtran.  They provide pressure,
    temperature, air density and the O\ :sub:`3`, O\ :sub:`2`,
    H\ :sub:`2`\ O, CO\ :sub:`2` and NO\ :sub:`2` profiles from the
    surface to 120 km:

    ==========  ====================  ===========================
    Short name  File (``atmmod/``)    Atmosphere
    ==========  ====================  ===========================
    ``afglus``  ``afglus.dat``        US Standard Atmosphere 1976
    ``afglt``   ``afglt.dat``         Tropical
    ``afglms``  ``afglms.dat``        Mid-latitude Summer
    ``afglmw``  ``afglmw.dat``        Mid-latitude Winter
    ``afglss``  ``afglss.dat``        Sub-arctic Summer
    ``afglsw``  ``afglsw.dat``        Sub-arctic Winter
    ==========  ====================  ===========================

    The background profile matters even when a radiosonde or ERA5 profile
    is supplied: libRadtran uses it above the top of the measured profile
    and for every gas the measured profile does not contain.  Total
    columns of ozone and water vapour are rescaled afterwards by
    ``ozone_du`` / ``h2o_mm`` of :class:`SimulationDefaults`.

    References
    ----------
    .. [1] Anderson, G. P., Clough, S. A., Kneizys, F. X., Chetwynd, J. H.
       and Shettle, E. P. (1986): *AFGL atmospheric constituent profiles
       (0–120 km)*. Tech. Rep. AFGL-TR-86-0110, Air Force Geophysics
       Laboratory, Hanscom AFB, MA.

    Examples
    --------
    >>> from pyradtran import list_atmosphere_profiles
    >>> list_atmosphere_profiles()
    Short name              Description
    ----------------------------------------------------------------------
      afglus                US Standard Atmosphere 1976
      afglms                Mid-latitude Summer
      afglmw                Mid-latitude Winter
      afglt                 Tropical
      afglss                Sub-arctic Summer
      afglsw                Sub-arctic Winter
      mcclams               McClatchey Mid-latitude Summer (extended)
      mcclamw               McClatchey Mid-latitude Winter (extended)
      afglus_ch4_vmr        US Standard – CH4 VMR profile
      afglus_co_vmr         US Standard – CO VMR profile
      afglus_n2o_vmr        US Standard – N2O VMR profile
      afglus_n2_vmr         US Standard – N2 VMR profile
      afglus_no2            US Standard – NO2 profile

    The catalog is a plain dict, so it can be inspected programmatically:

    >>> from pyradtran import ATMOSPHERE_PROFILES
    >>> ATMOSPHERE_PROFILES["afglsw"]
    ('atmmod/afglsw.dat', 'Sub-arctic Winter')

    Select the sub-arctic winter profile for one configuration:

    >>> from pyradtran import load_config
    >>> cfg = load_config("config/arctic.yaml")  # doctest: +SKIP
    >>> cfg.paths.atmosphere_profile  # doctest: +SKIP
    PosixPath('/opt/libRadtran-2.0.6/data/atmmod/afglsw.dat')

    where ``config/arctic.yaml`` contains

    .. code-block:: yaml

        paths:
          atmosphere_profile: afglsw
    """
    _print_catalog(ATMOSPHERE_PROFILES, 20)


@dataclass
class PathsConfig:
    """File-system paths required by libRadtran (YAML section ``paths``).

    Construction validates the libRadtran installation, resolves short
    names and creates the output and working directories, so an instance
    that exists is known to point at a usable ``uvspec``.

    Parameters
    ----------
    libradtran_bin : pathlib.Path or str
        Path to the ``uvspec`` executable.  Must be an existing file.
    libradtran_data : pathlib.Path or str
        Path to the libRadtran ``data/`` directory.  Must be an existing
        directory; written to the input file as ``data_files_path``.
    atmosphere_profile : pathlib.Path or str, optional
        Background atmosphere, written as ``atmosphere_file``.  Either a
        short name from :func:`list_atmosphere_profiles` (``"afglsw"``) or
        a file path.  Defaults to ``<libradtran_data>/atmmod/afglus.dat``
        (US Standard Atmosphere).
    solar_spectrum : pathlib.Path or str, optional
        Extraterrestrial solar spectrum, written as
        ``source solar <file>``.  Either a short name from
        :func:`list_solar_spectra` (``"kurudz_1.0nm"``) or a file path.
        Defaults to ``<libradtran_data>/solar_flux/kurudz_1.0nm.dat``.
    radiosonde_base : pathlib.Path or str, optional
        Root directory of local radiosonde files, scanned recursively by
        :class:`~pyradtran.utils.RadiosondeFinder`.  Only used when
        ``simulation_defaults.h2o_source`` is ``"radiosonde"``; a
        non-existent directory is tolerated.
    output_dir : pathlib.Path or str, default ``"./pyradtran_output"``
        Directory for NetCDF result files.  Created if missing.
    working_dir : pathlib.Path or str, default ``"./pyradtran_work"``
        Scratch directory for the ``uvspec`` input/output files.  Created
        if missing.

    Raises
    ------
    ConfigurationError
        If *libradtran_bin* is not a file, *libradtran_data* is not a
        directory, or the (resolved) *solar_spectrum* file does not exist.
        The exception is also a :class:`FileNotFoundError`.

    See Also
    --------
    save_master_config : Store these paths once per machine.
    list_atmosphere_profiles, list_solar_spectra : Available short names.

    Notes
    -----
    A missing *atmosphere_profile* file is only logged as a warning, not
    raised.


    After construction every field holds a :class:`pathlib.Path` (or
    *None* for an unset *radiosonde_base*): strings are converted and
    ``~`` is expanded.  Relative paths are **not** made absolute — they
    stay relative to the working directory of the Python process, which
    is also where the default ``pyradtran_output`` / ``pyradtran_work``
    directories are created.

    A value for *atmosphere_profile* or *solar_spectrum* is treated as a
    short name only if it matches a catalog key exactly; anything else is
    taken as a literal path.

    Examples
    --------
    Usually the instance is created by :func:`load_config`; direct
    construction looks like this:

    >>> from pyradtran import PathsConfig
    >>> paths = PathsConfig(  # doctest: +SKIP
    ...     libradtran_bin="/opt/libRadtran-2.0.6/bin/uvspec",
    ...     libradtran_data="/opt/libRadtran-2.0.6/data",
    ...     atmosphere_profile="afglss",
    ... )
    >>> paths.atmosphere_profile  # doctest: +SKIP
    PosixPath('/opt/libRadtran-2.0.6/data/atmmod/afglss.dat')
    >>> paths.solar_spectrum  # doctest: +SKIP
    PosixPath('/opt/libRadtran-2.0.6/data/solar_flux/kurudz_1.0nm.dat')

    A wrong installation path fails immediately:

    >>> PathsConfig("/nope/uvspec", "/nope")
    Traceback (most recent call last):
        ...
    pyradtran.exceptions.ConfigurationError: LibRadtran executable not found: /nope/uvspec. Set the libRadtran paths once with pyradtran.save_master_config(libradtran_bin=..., libradtran_data=...) or in the 'paths' section of your configuration file.
    """

    libradtran_bin: Path  # Path to uvspec executable
    libradtran_data: Path  # Path to LibRadtran data directory
    atmosphere_profile: Optional[Path] = None  # Default atmosphere profile file
    solar_spectrum: Optional[Path] = None  # Solar spectrum file
    radiosonde_base: Optional[Path] = None  # Optional radiosonde directory
    output_dir: Path = Path("./pyradtran_output")
    working_dir: Path = Path("./pyradtran_work")

    def __post_init__(self):
        # Accept plain strings for every path field (direct construction)
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, str):
                setattr(self, f.name, Path(value).expanduser())

        # Validate essential paths
        hint = (
            ". Set the libRadtran paths once with pyradtran.save_master_config("
            "libradtran_bin=..., libradtran_data=...) or in the 'paths' "
            "section of your configuration file."
        )
        if not self.libradtran_bin.is_file():
            raise ConfigurationError(
                f"LibRadtran executable not found: {self.libradtran_bin}{hint}"
            )
        if not self.libradtran_data.is_dir():
            raise ConfigurationError(
                f"LibRadtran data directory not found: {self.libradtran_data}{hint}"
            )

        # Resolve short names / infer defaults
        self.atmosphere_profile = (
            _resolve_libradtran_shortname(
                self.atmosphere_profile, self.libradtran_data, ATMOSPHERE_PROFILES
            )
            or self.libradtran_data / "atmmod" / "afglus.dat"
        )
        self.solar_spectrum = (
            _resolve_libradtran_shortname(
                self.solar_spectrum, self.libradtran_data, SOLAR_SPECTRA
            )
            or self.libradtran_data / "solar_flux" / "kurudz_1.0nm.dat"
        )

        if not self.atmosphere_profile.is_file():
            logger.warning(
                f"Default atmosphere profile not found: {self.atmosphere_profile}"
            )
        if not self.solar_spectrum.is_file():
            raise ConfigurationError(
                f"Solar spectrum file not found: {self.solar_spectrum}"
            )

        # Create output/working directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.working_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class CloudParameters:
    """Static cloud settings (YAML section ``simulation_defaults.clouds``).

    These settings describe one cloud that is the same for every
    simulated point.  Nothing is written to the ``uvspec`` input unless
    *enabled* is true.  What is written then depends on *cloud_source*:

    ``"file"``
        Existing libRadtran cloud files: ``wc_file <wc_file>`` for
        *cloud_type* ``"wc"`` / ``"mixed"`` and ``ic_file <ic_file>`` for
        ``"ic"`` / ``"mixed"`` (each only if the path is set).  This is
        the mode that works with libRadtran 2.0.6.
    ``"parametric"``
        A single homogeneous slab between *layer_bottom_km* and
        *layer_top_km*.  It is passed to ``uvspec`` as a generated two-row
        ``wc_file 1D`` / ``ic_file 1D`` profile, the same route a
        dict-valued ``wc_file`` in ``params`` takes.
    ``"era5"``
        Accepted, but currently ignored when the input file is built — no
        cloud line is written.  Use ``era5_clouds=True`` of
        :meth:`~pyradtran.interface.PyRadtranAccessor.run` instead.

    Parameters
    ----------
    enabled : bool, default ``False``
        Master switch for all settings below.
    cloud_type : {"wc", "ic", "mixed"}, default ``"wc"``
        Cloud phase: liquid water, ice, or both.
    cloud_source : {"parametric", "file", "era5"}, default ``"parametric"``
        How the cloud is specified (see above).
    layer_bottom_km : float, default ``1.0``
        Base height of the parametric slab (km).
    layer_top_km : float, default ``2.0``
        Top height of the parametric slab (km).
    water_content_g_m3 : float, default ``0.1``
        Liquid water content of the parametric slab (g m⁻³).
    ice_content_g_m3 : float, default ``0.0``
        Ice water content of the parametric slab (g m⁻³).
    effective_radius_um : float, default ``10.0``
        Effective droplet / crystal radius (µm), used for both phases.
    cloud_fraction : float, default ``1.0``
        Cloud fraction (0–1).  Stored only; not translated into a
        ``uvspec`` option.
    wc_file : pathlib.Path, optional
        Water-cloud file for ``cloud_source="file"``.
    ic_file : pathlib.Path, optional
        Ice-cloud file for ``cloud_source="file"``.
    era5_dataset : any, optional
        Placeholder for an ERA5 dataset.  Never written to YAML and
        shared by reference in :meth:`SimulationConfig.copy`.  Not used
        when the input file is built.
    era5_time : str, optional
        Reserved for the ``"era5"`` source; currently unused.
    era5_lat : float, optional
        Reserved for the ``"era5"`` source; currently unused.
    era5_lon : float, optional
        Reserved for the ``"era5"`` source; currently unused.

    See Also
    --------
    pyradtran.interface.PyRadtranAccessor.run : Per-point clouds via
        ``params={"wc_file": ...}``, the ``cloud_*_var`` arguments or
        ``era5_clouds=True``.
    pyradtran.clouds.CloudGenerator : Programmatic cloud-layer creation.
    pyradtran.era5.cloud_profiles : Cloud profiles from ERA5.

    Notes
    -----
    No value is range-checked.  With ``enabled=True`` and
    ``cloud_source="file"`` a non-existent *wc_file* / *ic_file* is logged
    as a warning.  *wc_file* / *ic_file* may be given as strings; they
    are converted to :class:`pathlib.Path` (with ``~`` expanded).

    Examples
    --------
    >>> from pyradtran.config import CloudParameters
    >>> clouds = CloudParameters(enabled=True, layer_bottom_km=0.5, layer_top_km=1.2)
    >>> clouds.cloud_type, clouds.cloud_source, clouds.effective_radius_um
    ('wc', 'parametric', 10.0)

    In YAML, a cloud read from an existing libRadtran cloud file:

    .. code-block:: yaml

        simulation_defaults:
          clouds:
            enabled: true
            cloud_type: wc
            cloud_source: file
            wc_file: /data/clouds/stratus_wc.dat
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
        for name in ("wc_file", "ic_file"):
            value = getattr(self, name)
            if isinstance(value, str):
                setattr(self, name, Path(value).expanduser())
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
    r"""Physics and spectral settings (YAML section ``simulation_defaults``).

    Each field maps onto one line of the ``uvspec`` input file that
    :class:`~pyradtran.input_builder.InputFileBuilder` writes for every
    simulated point.  The values are *defaults*: anything given in the
    ``params`` mapping of :meth:`~pyradtran.interface.PyRadtranAccessor.run`
    replaces the corresponding line for that run.

    Parameters
    ----------
    rte_solver : str, default ``"twostr"``
        Radiative-transfer solver, written as ``rte_solver <value>``
        (``"twostr"``, ``"disort"``, ``"fdisort1"``, ``"rodents"``, …).
        ``twostr`` is fast but computes irradiances only and can become
        unstable for optically thick clouds; use ``disort`` for radiances
        and cloudy scenes.
    mol_abs_param : str, default ``"lowtran per_nm"``
        Molecular absorption parameterisation, written as
        ``mol_abs_param <value>`` (e.g. ``"reptran coarse"``, ``"kato2"``).
        An empty string omits the line.
    source : {"solar", "thermal"}, default ``"solar"``
        Radiation source.  ``"solar"`` writes
        ``source solar <paths.solar_spectrum>`` plus ``sza`` and
        ``day_of_year`` (from the point's time); ``"thermal"`` writes
        ``source thermal`` and, if set, ``sur_temperature``.
    wavelength_nm : list of float, default ``[400, 3600]``
        Spectral range ``[min, max]`` in nm, written as
        ``wavelength <min> <max>``.  A mapping ``{"start": ..., "end": ...}``
        is accepted and converted to the list form.
    integrate_wavelength : bool, default ``False``
        If true, ``output_process integrate`` is written and (solar
        source) ``per_nm`` is appended to the ``source`` line, so each
        output quantity is one spectrally integrated number per altitude
        instead of a spectrum.
    output_columns : list of str, default ``["sza", "eglo", "eup", "albedo"]``
        Quantities requested with ``output_user``, in file order (e.g.
        ``edir``, ``edn``, ``eup``, ``eglo``, ``enet``, ``uu``).  Each
        becomes a data variable of the result.  ``lambda`` is added
        automatically for spectral runs and ``zout`` for runs with more
        than one output altitude; an empty list omits the line.
    output_altitudes_km : list of float, default ``[0.0]``
        Output levels in km, written as ``zout`` (sorted, duplicates
        removed).  They become the ``altitude`` coordinate of the result.
    albedo_value : float, optional
        Lambertian surface albedo (0–1), written as ``albedo <value>``.
        With *None* the line is omitted and ``uvspec`` uses its own
        default (0).  Not written when a run supplies ``albedo`` or
        ``brdf_rpv_type`` through ``params``.
    surface_temperature_k : float, optional
        Surface temperature in K, written as ``sur_temperature <value>``
        for ``source="thermal"`` only.  With *None*, ``uvspec`` takes the
        temperature of the lowest atmospheric level.
    ozone_du : float, optional, default ``300.0``
        Total ozone column in Dobson units, written as
        ``mol_modify O3 <value> DU``.  *None* keeps the column of the
        atmosphere profile.  Skipped when an ERA5 profile already
        provides ozone.
    h2o_mm : float, optional, default ``2.0``
        Precipitable water in mm (kg m⁻²), written as
        ``mol_modify H2O <value> MM``.  *None* keeps the column of the
        atmosphere profile.  Skipped when a radiosonde or ERA5 profile
        provides the humidity.
    h2o_source : {"fixed", "radiosonde"}, default ``"fixed"``
        ``"radiosonde"`` looks up the local sounding closest in time below
        ``paths.radiosonde_base`` (see
        :class:`~pyradtran.utils.RadiosondeFinder`) and writes
        ``radiosonde <file> H2O RH``.  If no file is found the run falls
        back to the ``"fixed"`` behaviour (*h2o_mm*).
    clouds : CloudParameters
        Static cloud settings; nothing is written unless
        ``clouds.enabled`` is true.
    viewing_geometry : str, default ``"nadir"``
        ``"nadir"`` writes ``umu 1.0`` (sensor looking straight down).
        Any other value writes no viewing-geometry line; set ``umu`` /
        ``phi`` through ``params`` or *parameter_overrides* instead.
    sza : float, optional
        Fixed solar zenith angle in degrees (``sza <value>``, solar source
        only).  With *None* it is computed for every point from its time
        and position by
        :func:`~pyradtran.input_builder.calculate_solar_zenith_angle`.
    parameter_overrides : dict, optional
        Escape hatch for any other libRadtran option:
        ``{keyword: value}`` pairs applied to every run.  A keyword that
        the typed fields above already wrote replaces that line; ``True``
        writes a bare flag (``aerosol_default``), ``False`` removes the
        line, a list writes one line per entry.

    Raises
    ------
    ConfigurationError
        If *wavelength_nm* does not have exactly two elements (or, as a
        mapping, lacks ``start`` / ``end``), *source* is not ``"solar"`` or
        ``"thermal"``, or *albedo_value* lies outside [0, 1].  The
        exception is also a :class:`ValueError`.

    See Also
    --------
    pyradtran.interface.PyRadtranAccessor.explain : Show the input file
        these settings produce, line by line.
    pyradtran.params.describe : Documentation of any ``uvspec`` option.
    CloudParameters : The nested ``clouds`` section.

    Notes
    -----
    The checks and the altitude sorting are done by :meth:`validate`,
    which runs at construction (and therefore in :func:`load_config`).
    Assigning to an attribute afterwards (``sd.albedo_value = 2.0``) is
    not checked at that moment; call :meth:`validate` again to re-check
    and re-sort after such changes.  ``run()``, ``explain()`` and
    ``jacobian()`` of the accessor do so before anything is simulated.

    The total ozone column is given in Dobson units,

    .. math::

        1\ \mathrm{DU} = 10^{-5}\ \mathrm{m\ (STP)}
                       = 2.687 \times 10^{20}\ \mathrm{molecules\ m^{-2}},

    and the water vapour column as precipitable water, the depth of the
    liquid layer the column would form:

    .. math::

        \mathrm{PW} = \frac{1}{\rho_w} \int_0^\infty \rho_v(z)\,\mathrm{d}z,
        \qquad 1\ \mathrm{mm} \mathrel{\widehat{=}} 1\ \mathrm{kg\ m^{-2}} .

    ``mol_modify`` rescales the whole profile of the background atmosphere
    so that its vertical integral matches the requested column; the shape
    of the profile is kept.

    With *integrate_wavelength* the output is the band integral

    .. math::

        E = \int_{\lambda_\mathrm{min}}^{\lambda_\mathrm{max}}
            E_\lambda(\lambda)\,\mathrm{d}\lambda

    in the units of the solar spectrum file times nm (mW m⁻² for the
    Kurucz spectra, which are tabulated in mW m⁻² nm⁻¹).

    Examples
    --------
    The defaults, as used when a YAML file does not mention a key:

    >>> from pyradtran import SimulationDefaults
    >>> sd = SimulationDefaults()
    >>> sd.rte_solver, sd.mol_abs_param, sd.source
    ('twostr', 'lowtran per_nm', 'solar')
    >>> sd.wavelength_nm, sd.output_columns
    ([400, 3600], ['sza', 'eglo', 'eup', 'albedo'])

    Altitudes are sorted and de-duplicated, and the mapping form of the
    wavelength range is normalised:

    >>> sd = SimulationDefaults(
    ...     source="thermal",
    ...     wavelength_nm={"start": 8000, "end": 13000},
    ...     output_altitudes_km=[10.0, 0.0, 10.0, 2.0],
    ... )
    >>> sd.wavelength_nm, sd.output_altitudes_km
    ([8000, 13000], [0.0, 2.0, 10.0])

    Invalid values are rejected:

    >>> SimulationDefaults(albedo_value=1.5)
    Traceback (most recent call last):
        ...
    pyradtran.exceptions.ConfigurationError: albedo_value must be between 0 and 1, got 1.5

    The same settings in a YAML file, annotated with the ``uvspec`` line
    each one produces:

    .. code-block:: yaml

        simulation_defaults:
          rte_solver: twostr              # rte_solver twostr
          mol_abs_param: lowtran per_nm   # mol_abs_param lowtran per_nm
          source: solar                   # source solar <solar_spectrum>
          wavelength_nm: [500, 502]       # wavelength 500 502
          integrate_wavelength: false     # true: output_process integrate
          output_columns: [sza, eglo, eup, albedo]  # output_user lambda sza ...
          output_altitudes_km: [0.0]      # zout 0.0000
          albedo_value: 0.85              # albedo 0.85
          surface_temperature_k: 273.15   # sur_temperature 273.15 (thermal)
          ozone_du: 300.0                 # mol_modify O3 300.0 DU
          h2o_mm: 2.0                     # mol_modify H2O 2.0 MM
          h2o_source: fixed               # radiosonde: radiosonde <file> H2O RH
          viewing_geometry: nadir         # umu 1.0
          sza: null                       # sza <computed>, day_of_year <doy>
          parameter_overrides: {}         # e.g. {aerosol_default: true}

    :meth:`~pyradtran.interface.PyRadtranAccessor.explain` shows the
    result for a concrete point (here 28 March 2022, 12 UTC, Ny-Ålesund):

    >>> cfg = load_config()  # doctest: +SKIP
    >>> cfg.simulation_defaults.wavelength_nm = [500, 502]  # doctest: +SKIP
    >>> print(ds.pyradtran.explain(config=cfg))  # doctest: +SKIP
    rte_solver twostr                                                   # config
    mol_abs_param lowtran per_nm                                        # config
    data_files_path /opt/libRadtran-2.0.6/data                          # config
    atmosphere_file /opt/libRadtran-2.0.6/data/atmmod/afglus.dat        # config
    mol_modify O3 300.0 DU                                              # config
    mol_modify H2O 2.0 MM                                               # config
    source solar /opt/libRadtran-2.0.6/data/solar_flux/NewGuey2003.dat  # config
    sza 76.73                                                           # config
    day_of_year 87                                                      # config
    wavelength 500 502                                                  # config
    albedo 0.85                                                         # config
    output_user lambda sza eglo eup albedo                              # config
    zout 0.0000                                                         # config
    umu 1.0                                                             # config
    quiet                                                               # config
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
        self.validate()

    def validate(self) -> None:
        """Check the settings and normalise them in place.

        Called at construction. Call it again after assigning to
        attributes: it normalises the mapping form of *wavelength_nm*,
        checks *wavelength_nm*, *source* and *albedo_value*, and sorts
        and de-duplicates *output_altitudes_km*.

        Raises
        ------
        ConfigurationError
            See the class docstring. Also a :class:`ValueError`.

        Examples
        --------
        >>> from pyradtran import SimulationDefaults
        >>> sd = SimulationDefaults()
        >>> sd.output_altitudes_km = [5, 0, 5]
        >>> sd.validate()
        >>> sd.output_altitudes_km
        [0, 5]
        >>> sd.albedo_value = 2.0
        >>> sd.validate()
        Traceback (most recent call last):
            ...
        pyradtran.exceptions.ConfigurationError: albedo_value must be between 0 and 1, got 2.0
        """
        if isinstance(self.wavelength_nm, dict):
            # accept the {start: ..., end: ...} YAML spelling
            try:
                self.wavelength_nm = [
                    self.wavelength_nm["start"],
                    self.wavelength_nm["end"],
                ]
            except KeyError:
                raise ConfigurationError(
                    "wavelength_nm mapping needs 'start' and 'end' keys, "
                    f"got {sorted(self.wavelength_nm)}"
                )
        if self.wavelength_nm and len(self.wavelength_nm) != 2:
            raise ConfigurationError("wavelength_nm must contain [min, max]")
        # output_altitudes_km can be empty (defaults to uvspec implicit behavior)
        if self.source not in ["solar", "thermal"]:
            raise ConfigurationError(
                f"source must be 'solar' or 'thermal', got '{self.source}'"
            )
        if self.albedo_value is not None and not (0 <= self.albedo_value <= 1):
            raise ConfigurationError(
                f"albedo_value must be between 0 and 1, got {self.albedo_value}"
            )

        # Sort and deduplicate altitude levels
        self.output_altitudes_km = sorted(set(self.output_altitudes_km))


@dataclass
class ExecutionConfig:
    """Run-time execution settings (YAML section ``execution``).

    Parameters
    ----------
    max_workers : int, optional
        Number of ``uvspec`` processes run in parallel.  The dataclass
        default is ``min(8, os.cpu_count())``, evaluated when the module
        is imported; the package default YAML (and therefore
        :func:`load_config`) sets ``4``.  With ``1`` all points run
        serially inside the calling process (no process pool); *None*
        lets :class:`concurrent.futures.ProcessPoolExecutor` choose.
    cleanup_temp_files : bool, default ``False``
        Delete the ``uvspec`` input/output files and temporary cloud
        files in ``paths.working_dir`` after each *successful* point.
        Files of failed points are always kept for inspection.
    debug_mode : bool, default ``False``
        When true, :func:`load_config` sets the ``"pyradtran"`` logger to
        ``DEBUG``.  Handlers are left to the application.
    timeout_seconds : int, default ``300``
        Wall-clock limit in seconds for a single ``uvspec`` call; a point
        exceeding it counts as failed.

    See Also
    --------
    pyradtran.interface.execute_simulation_batch : The parallel driver
        that reads these settings.

    Examples
    --------
    >>> from pyradtran.config import ExecutionConfig
    >>> ExecutionConfig(max_workers=1).timeout_seconds
    300

    Serial execution is convenient for debugging and inside notebooks
    that are rendered in CI:

    >>> cfg = load_config()  # doctest: +SKIP
    >>> cfg.execution.max_workers = 1  # doctest: +SKIP
    """

    max_workers: Optional[int] = min(8, os.cpu_count() or 1)
    cleanup_temp_files: bool = False  # Keep temp files for debugging
    debug_mode: bool = False
    timeout_seconds: int = 300


@dataclass
class OutputConfig:
    """NetCDF output-file settings (YAML section ``output``).

    Parameters
    ----------
    filename_prefix : str, default ``"pyradtran_sim"``
        Prefix for auto-generated file names.
    filename_suffix : str, default ``"_results.nc"``
        Suffix (including extension) for auto-generated file names.
    netcdf_encoding : dict, default ``{"zlib": True, "complevel": 5}``
        Per-variable encoding.  The same dict is applied to every data
        variable and passed to :meth:`xarray.Dataset.to_netcdf` as
        *encoding*; an empty dict disables it.

    See Also
    --------
    pyradtran.io.NetCDFSaver.save_results_to_netcdf : Writes the file.

    Notes
    -----
    When a run saves its result and no explicit ``output_path`` is given,
    the file is written to

    ``<paths.output_dir>/<filename_prefix>_<YYYYmmdd_HHMMSS><filename_suffix>``

    with the local time at which the run was started.

    Examples
    --------
    >>> from pyradtran.config import OutputConfig
    >>> out = OutputConfig(filename_prefix="halo_ac3")
    >>> out.filename_prefix + "_20220328_120000" + out.filename_suffix
    'halo_ac3_20220328_120000_results.nc'
    >>> out.netcdf_encoding
    {'zlib': True, 'complevel': 5}
    """

    filename_prefix: str = "pyradtran_sim"
    filename_suffix: str = "_results.nc"
    netcdf_encoding: Dict[str, Any] = field(
        default_factory=lambda: {"zlib": True, "complevel": 5}
    )


@dataclass
class SimulationConfig:
    """Top-level configuration container.

    Composed of four sections that mirror the top-level keys of the YAML
    file.  Use :func:`load_config` (three merged layers) or
    :meth:`from_yaml` (one self-contained file) to construct an instance
    from disk.

    Parameters
    ----------
    paths : PathsConfig
        File-system paths: libRadtran installation, atmosphere profile,
        solar spectrum, output and working directories.
    simulation_defaults : SimulationDefaults
        Physics and spectral settings written to the ``uvspec`` input.
    execution : ExecutionConfig, optional
        Parallelism, timeout and debugging.
    output : OutputConfig, optional
        NetCDF file naming and encoding.

    See Also
    --------
    load_config : Recommended entry point (merges three layers).
    pyradtran.interface.PyRadtranAccessor.run : Accepts an instance via
        ``config=``.

    Notes
    -----
    The object is a plain mutable dataclass: change a setting by assigning
    to the attribute.  ``ds.pyradtran.run(config=cfg)`` works on a
    :meth:`copy`, so the per-run adjustments it makes (dataset-derived
    output altitudes, dotted keys in ``params``) never leak back into
    *cfg*.

    Examples
    --------
    Load, modify, run and save:

    >>> from pyradtran import load_config
    >>> cfg = load_config()  # doctest: +SKIP
    >>> cfg.simulation_defaults.rte_solver = "disort"  # doctest: +SKIP
    >>> cfg.simulation_defaults.integrate_wavelength = True  # doctest: +SKIP
    >>> cfg.execution.max_workers = 1  # doctest: +SKIP
    >>> result = ds.pyradtran.run(config=cfg, save_to_file=False)  # doctest: +SKIP
    >>> cfg.to_yaml("config/broadband_disort.yaml")  # doctest: +SKIP
    PosixPath('config/broadband_disort.yaml')
    """

    paths: PathsConfig
    simulation_defaults: SimulationDefaults
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> "SimulationConfig":
        """Load configuration from a single, self-contained YAML file.

        Unlike :func:`load_config` nothing is merged: neither the package
        defaults nor ``~/.pyradtran/config.yaml`` are read.  Keys missing
        from the file take the dataclass defaults, so the file must at
        least contain ``paths`` (with ``libradtran_bin`` and
        ``libradtran_data``) and ``simulation_defaults`` (which may be
        empty, ``{}``).

        Parameters
        ----------
        yaml_path : str or pathlib.Path
            Path to the YAML file.

        Returns
        -------
        SimulationConfig

        Raises
        ------
        ConfigurationError
            If *yaml_path* does not exist or the paths in it fail the
            :class:`PathsConfig` checks (also a
            :class:`FileNotFoundError`), or a value fails the
            :class:`SimulationDefaults` validation (also a
            :class:`ValueError`).
        TypeError
            If a required section or key is missing.

        See Also
        --------
        load_config : Layered loading; the usual entry point.
        to_yaml : Writes a file this method can read back.

        Notes
        -----
        Unknown keys are ignored with a logged warning.  Scalars are
        coerced to the annotated field type where possible (``"0.2"``
        becomes ``0.2`` for a float field) and path strings become
        :class:`pathlib.Path` objects with ``~`` expanded.

        Examples
        --------
        Round-trip a configuration exactly as saved, independent of the
        master config of the current machine:

        >>> from pyradtran import SimulationConfig
        >>> cfg = SimulationConfig.from_yaml("config/my_simulation.yaml")  # doctest: +SKIP
        >>> cfg.simulation_defaults.rte_solver  # doctest: +SKIP
        'disort'
        """
        yaml_path = Path(yaml_path)
        if not yaml_path.is_file():
            raise ConfigurationError(f"Configuration file not found: {yaml_path}")

        with open(yaml_path, "r") as f:
            raw_config = yaml.safe_load(f)

        return cls._dict_to_dataclass(raw_config, cls)

    @classmethod
    def _dict_to_dataclass(cls, data: Dict[str, Any], dataclass_type: type) -> Any:
        """Recursively convert a nested dictionary to dataclass instances.

        Unknown keys are skipped with a warning; ``Optional[X]`` fields are
        coerced with ``X(value)`` (falling back to the raw value if that
        fails), ``Path`` fields get ``expanduser()``, and nested mappings
        become the nested dataclass.

        Examples
        --------
        >>> sd = SimulationConfig._dict_to_dataclass(
        ...     {"albedo_value": "0.2", "clouds": {"enabled": True}},
        ...     SimulationDefaults,
        ... )
        >>> sd.albedo_value, sd.clouds.enabled, sd.rte_solver
        (0.2, True, 'twostr')
        """
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

    def copy(self) -> "SimulationConfig":
        """Return an independent deep copy of this configuration.

        The one exception is ``clouds.era5_dataset`` (potentially a large
        xarray Dataset), which is carried over by reference.

        Used by the accessor so that per-run adjustments (dataset-derived
        ``zout`` levels, dotted config overrides in ``params``) never
        mutate a caller-owned config object.

        Returns
        -------
        SimulationConfig

        Examples
        --------
        Derive variants of one base configuration for a sensitivity study:

        >>> base = load_config()  # doctest: +SKIP
        >>> dark = base.copy()  # doctest: +SKIP
        >>> dark.simulation_defaults.albedo_value = 0.06  # doctest: +SKIP
        >>> base.simulation_defaults.albedo_value  # doctest: +SKIP
        0.85
        """
        # pre-seeding the memo makes deepcopy reuse era5_dataset as it is
        era5 = self.simulation_defaults.clouds.era5_dataset
        return copy.deepcopy(self, {id(era5): era5})

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the full configuration to a plain nested dict.

        All :class:`~pathlib.Path` objects are converted to strings so
        the result is immediately YAML-serialisable.  The non-serialisable
        ``clouds.era5_dataset`` entry is dropped.

        Returns
        -------
        dict
            Nested dictionary mirroring the YAML structure, with the
            top-level keys ``paths``, ``simulation_defaults``,
            ``execution`` and ``output``.  Every field is present,
            including those still at their default.

        See Also
        --------
        to_yaml : Write the result directly to a file.
        get_used_parameters : A similar, hand-picked summary.

        Examples
        --------
        >>> cfg = load_config()  # doctest: +SKIP
        >>> d = cfg.to_dict()  # doctest: +SKIP
        >>> sorted(d)  # doctest: +SKIP
        ['execution', 'output', 'paths', 'simulation_defaults']
        >>> d["paths"]["atmosphere_profile"]  # doctest: +SKIP
        '/opt/libRadtran-2.0.6/data/atmmod/afglus.dat'
        >>> d["execution"]  # doctest: +SKIP
        {'max_workers': 4, 'cleanup_temp_files': False, 'debug_mode': False, 'timeout_seconds': 300}
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
        del d["simulation_defaults"]["clouds"]["era5_dataset"]
        return d

    def to_yaml(self, path: Union[str, Path]) -> Path:
        """Write the configuration to a YAML file.

        The file contains *every* setting (the output of :meth:`to_dict`,
        keys sorted alphabetically), so it documents a run completely and
        can be read back with :func:`load_config` or :meth:`from_yaml`.

        Parameters
        ----------
        path : str or pathlib.Path
            Destination file.  Parent directories are created
            automatically; an existing file is overwritten.

        Returns
        -------
        pathlib.Path
            *path* as a :class:`~pathlib.Path` (not made absolute).

        See Also
        --------
        load_config : Read the file back.
        save_master_config : Store only the machine-specific paths.

        Notes
        -----
        Because the file is complete it also pins the machine-specific
        ``paths`` section (already resolved to full file names) and
        thereby overrides the master config wherever it is loaded.  Delete
        the ``paths`` block from the file to make it portable between
        machines.

        Examples
        --------
        Start from the defaults, change what matters and persist it:

        >>> from pyradtran import load_config
        >>> cfg = load_config()  # doctest: +SKIP
        >>> cfg.simulation_defaults.albedo_value = 0.2  # doctest: +SKIP
        >>> cfg.simulation_defaults.wavelength_nm = [500, 600]  # doctest: +SKIP
        >>> cfg.simulation_defaults.rte_solver = "disort"  # doctest: +SKIP
        >>> cfg.to_yaml("config/my_simulation.yaml")  # doctest: +SKIP
        PosixPath('config/my_simulation.yaml')

        The written file (shortened):

        .. code-block:: yaml

            execution:
              cleanup_temp_files: false
              debug_mode: false
              max_workers: 4
              timeout_seconds: 300
            output:
              filename_prefix: pyradtran_sim
              ...
            paths:
              atmosphere_profile: /opt/libRadtran-2.0.6/data/atmmod/afglus.dat
              libradtran_bin: /opt/libRadtran-2.0.6/bin/uvspec
              libradtran_data: /opt/libRadtran-2.0.6/data
              ...
            simulation_defaults:
              albedo_value: 0.2
              ...
              rte_solver: disort
              source: solar
              ...
              wavelength_nm:
              - 500
              - 600
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
        """Return a nested summary dict of the active parameters.

        Useful for logging.  The result has the same four sections as
        :meth:`to_dict` with all paths as strings, but is a hand-picked
        subset: ``simulation_defaults.parameter_overrides`` and the
        ``clouds.era5_*`` entries are not included.

        Returns
        -------
        dict
            ``{"paths": {...}, "simulation_defaults": {..., "clouds": {...}},
            "execution": {...}, "output": {...}}``.  Lists and dicts inside
            are the config's own objects, not copies.

        See Also
        --------
        to_dict : Complete, YAML-serialisable representation.

        Examples
        --------
        >>> cfg = load_config()  # doctest: +SKIP
        >>> used = cfg.get_used_parameters()  # doctest: +SKIP
        >>> sorted(used)  # doctest: +SKIP
        ['execution', 'output', 'paths', 'simulation_defaults']
        >>> used["simulation_defaults"]["rte_solver"]  # doctest: +SKIP
        'twostr'
        """

        def section(obj, skip=()):
            return {
                f.name: getattr(obj, f.name)
                for f in fields(obj)
                if f.name not in skip and not f.name.startswith("era5_")
            }

        def path_str(p):
            return str(p) if p else None

        sd = self.simulation_defaults
        paths = {k: str(v) for k, v in section(self.paths).items()}
        paths["radiosonde_base"] = path_str(self.paths.radiosonde_base)
        clouds = section(sd.clouds)
        clouds["wc_file"] = path_str(sd.clouds.wc_file)
        clouds["ic_file"] = path_str(sd.clouds.ic_file)
        return {
            "paths": paths,
            "simulation_defaults": {
                **section(sd, skip=("clouds", "parameter_overrides")),
                "clouds": clouds,
            },
            "execution": section(self.execution),
            "output": section(self.output),
        }


# Default configuration path
_DEFAULT_CONFIG_PATH = Path(__file__).parent / "defaults.yaml"


def _recursive_update(base: Dict, update: Dict) -> Dict:
    """Recursively merge *update* into *base* (mutates and returns *base*).

    Nested mappings are merged key by key; every other value (scalars and
    lists alike) replaces the one in *base*.

    Examples
    --------
    >>> base = {"paths": {"a": 1, "b": 2}, "cols": ["eglo", "eup"]}
    >>> _recursive_update(base, {"paths": {"b": 3}, "cols": ["edir"]})
    {'paths': {'a': 1, 'b': 3}, 'cols': ['edir']}
    """
    for key, value in update.items():
        if isinstance(value, dict) and key in base and isinstance(base[key], dict):
            _recursive_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(config_path: Optional[Union[str, Path]] = None) -> SimulationConfig:
    """Load and merge the three-layer configuration.

    This is the standard way to obtain a :class:`SimulationConfig`.  Three
    YAML sources are read in turn and merged key by key, later layers
    winning:

    1. **Package defaults** — ``pyradtran/defaults.yaml`` in the
       pyRadtran installation.  Provides a value for every setting, but
       only placeholder libRadtran paths.
    2. **User master config** — ``~/.pyradtran/config.yaml``.  The place
       for machine-specific paths (see :func:`save_master_config`).
    3. **Simulation config** — *config_path*, describing one experiment.

    The merged mapping is then converted to dataclasses, which validates
    it (see :class:`PathsConfig` and :class:`SimulationDefaults`).

    Parameters
    ----------
    config_path : str or pathlib.Path, optional
        Simulation-specific YAML file.  It only needs to contain the keys
        that differ from layers 1 and 2.  When *None*, only layers 1 and 2
        are used.

    Returns
    -------
    SimulationConfig
        The merged, validated configuration.  A new object on every call;
        modifying it does not affect later calls.

    Raises
    ------
    ConfigurationError
        If *config_path* is given but cannot be read or is not valid
        YAML; if, after merging, ``paths.libradtran_bin``,
        ``paths.libradtran_data`` or the solar spectrum file do not exist
        (typically because no master config has been written yet and the
        package placeholders are still in effect — the message then says
        how to set the paths); or if a merged value fails the
        :class:`SimulationDefaults` validation.
        :class:`~pyradtran.exceptions.ConfigurationError` is also a
        :class:`FileNotFoundError` and a :class:`ValueError`, so code
        catching those keeps working.
    TypeError
        If a required section is missing from all layers.

    See Also
    --------
    save_master_config : Write layer 2.
    SimulationConfig.to_yaml : Write a file usable as layer 3.
    SimulationConfig.from_yaml : Load one file without any merging.
    pyradtran.interface.PyRadtranAccessor.run : Calls this function when
        given ``config_path=``; its ``params=`` argument is the per-run
        override on top of all three layers.

    Notes
    -----
    **Precedence.**  Merging is recursive for mappings and wholesale for
    everything else: a simulation YAML containing only

    .. code-block:: yaml

        simulation_defaults:
          source: thermal
          wavelength_nm: [8000, 13000]

    changes these two settings and inherits every other key of
    ``simulation_defaults`` — and the complete ``paths``, ``execution`` and
    ``output`` sections — from the master config and the package defaults.
    A list (``wavelength_nm``, ``output_columns``, …) is always replaced as
    a whole, never concatenated.  On top of the three file layers, the
    ``params`` argument of
    :meth:`~pyradtran.interface.PyRadtranAccessor.run` overrides settings
    for a single run without touching any file.

    **Short names.**  ``paths.atmosphere_profile`` and
    ``paths.solar_spectrum`` may be given as short names (``afglsw``,
    ``kurudz_1.0nm``; see :func:`list_atmosphere_profiles` and
    :func:`list_solar_spectra`).  They are resolved against the *merged*
    ``paths.libradtran_data``, so a short name in the simulation YAML
    works with whatever installation the master config points to.  The
    returned object always holds full paths.

    **Missing files.**  A missing master config is skipped silently, and
    one that cannot be read or parsed is skipped with a logged warning.
    An unreadable package default file is logged as an error and treated
    as empty.  Only an unreadable or unparsable *config_path* raises.

    **Unknown keys** in any layer are ignored with a logged warning naming
    the key, so a typo such as ``albedo`` instead of ``albedo_value`` does
    not raise — check the log (or the returned object) when a setting
    seems to have no effect.

    **Side effects.**  ``paths.output_dir`` and ``paths.working_dir`` are
    created if they do not exist; relative paths are taken relative to the
    current working directory.  With ``execution.debug_mode: true`` the
    ``"pyradtran"`` logger is set to ``DEBUG``.

    The package defaults differ in a few places from the bare dataclass
    defaults of :class:`SimulationDefaults` and :class:`ExecutionConfig`:
    ``albedo_value: 0.85``, ``surface_temperature_k: 273.15`` and
    ``max_workers: 4``.

    Examples
    --------
    Without an argument the package defaults and the master config are
    combined — here the master config supplies the libRadtran paths and
    the short names ``afglus`` / ``NewGuey2003``:

    >>> from pyradtran import load_config
    >>> cfg = load_config()  # doctest: +SKIP
    >>> cfg.paths.libradtran_bin  # doctest: +SKIP
    PosixPath('/opt/libRadtran-2.0.6/bin/uvspec')
    >>> cfg.paths.atmosphere_profile  # doctest: +SKIP
    PosixPath('/opt/libRadtran-2.0.6/data/atmmod/afglus.dat')
    >>> cfg.paths.solar_spectrum  # doctest: +SKIP
    PosixPath('/opt/libRadtran-2.0.6/data/solar_flux/NewGuey2003.dat')
    >>> sd = cfg.simulation_defaults  # doctest: +SKIP
    >>> sd.rte_solver, sd.wavelength_nm, sd.albedo_value  # doctest: +SKIP
    ('twostr', [400, 3600], 0.85)
    >>> sd.output_columns  # doctest: +SKIP
    ['sza', 'eglo', 'eup', 'albedo']

    Add a simulation YAML as the third layer.  ``thermal.yaml`` holds just
    the lines that matter for the experiment:

    .. code-block:: yaml

        paths:
          atmosphere_profile: afglsw
        simulation_defaults:
          source: thermal
          wavelength_nm: [8000, 13000]

    >>> cfg = load_config("thermal.yaml")  # doctest: +SKIP
    >>> sd = cfg.simulation_defaults  # doctest: +SKIP
    >>> sd.source, sd.wavelength_nm, sd.rte_solver  # doctest: +SKIP
    ('thermal', [8000, 13000], 'twostr')
    >>> cfg.paths.atmosphere_profile  # doctest: +SKIP
    PosixPath('/opt/libRadtran-2.0.6/data/atmmod/afglsw.dat')

    Modify fields in Python and save the result as a new simulation YAML:

    >>> cfg = load_config()  # doctest: +SKIP
    >>> cfg.simulation_defaults.albedo_value = 0.2  # doctest: +SKIP
    >>> cfg.simulation_defaults.wavelength_nm = [500, 600]  # doctest: +SKIP
    >>> cfg.simulation_defaults.rte_solver = "disort"  # doctest: +SKIP
    >>> cfg.to_yaml("config/my_simulation.yaml")  # doctest: +SKIP
    PosixPath('config/my_simulation.yaml')
    >>> load_config("config/my_simulation.yaml").simulation_defaults.rte_solver  # doctest: +SKIP
    'disort'

    Override settings for a single run.  ``params`` accepts ``uvspec``
    options (``sza``) as well as dotted config paths; the configuration
    object itself stays untouched:

    >>> import pandas as pd
    >>> import xarray as xr
    >>> ds = xr.Dataset(
    ...     coords={
    ...         "time": pd.to_datetime(["2022-03-28 12:00"]),
    ...         "latitude": 78.9,
    ...         "longitude": 11.9,
    ...     }
    ... )
    >>> cfg = load_config()  # doctest: +SKIP
    >>> cfg.simulation_defaults.rte_solver = "disort"  # doctest: +SKIP
    >>> cfg.simulation_defaults.integrate_wavelength = True  # doctest: +SKIP
    >>> kwargs = dict(config=cfg, save_to_file=False, show_progress=False)  # doctest: +SKIP
    >>> ref = ds.pyradtran.run(**kwargs)  # doctest: +SKIP
    >>> float(ref.eglo.squeeze()), float(ref.eup.squeeze())  # doctest: +SKIP
    (221116.9, 187949.4)
    >>> low = ds.pyradtran.run(  # doctest: +SKIP
    ...     params={"simulation_defaults.albedo_value": 0.1, "sza": 60.0}, **kwargs
    ... )
    >>> float(low.eglo.squeeze()), float(low.eup.squeeze())  # doctest: +SKIP
    (518429.6, 51842.96)
    >>> cfg.simulation_defaults.albedo_value  # doctest: +SKIP
    0.85

    A missing simulation file is an error:

    >>> load_config("does_not_exist.yaml")
    Traceback (most recent call last):
        ...
    pyradtran.exceptions.ConfigurationError: Cannot read configuration file does_not_exist.yaml: [Errno 2] No such file or directory: 'does_not_exist.yaml'
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
        except (OSError, yaml.YAMLError) as e:
            raise ConfigurationError(
                f"Cannot read configuration file {config_path}: {e}"
            ) from e

    try:
        # Now convert to dataclass
        config = SimulationConfig._dict_to_dataclass(
            final_config_dict, SimulationConfig
        )

        logger.debug("Configuration loaded successfully.")

        # debug_mode raises the package logger's verbosity; the root
        # logger and handlers belong to the application, not to us
        if config.execution.debug_mode:
            logging.getLogger("pyradtran").setLevel(logging.DEBUG)

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
    output_dir: Optional[Union[str, Path]] = None,
    working_dir: Optional[Union[str, Path]] = None,
    max_workers: Optional[int] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """Write (or update) the user master config ``~/.pyradtran/config.yaml``.

    The master config is **layer 2** of the three-layer cascade read by
    :func:`load_config`.  It is the place to store machine-specific paths
    (libRadtran installation, radiosonde archive, …) once, so that every
    simulation YAML can stay minimal and portable.  This is the one-time
    setup step after installing pyRadtran.

    Parameters
    ----------
    libradtran_bin : str or pathlib.Path
        Absolute path to the ``uvspec`` executable.
    libradtran_data : str or pathlib.Path
        Absolute path to the libRadtran ``data/`` directory.
    atmosphere_profile : str or pathlib.Path, optional
        Default atmosphere profile: a short name from
        :func:`list_atmosphere_profiles` or a file path.  Left as it is
        in the file when *None*.
    solar_spectrum : str or pathlib.Path, optional
        Default solar spectrum: a short name from
        :func:`list_solar_spectra` or a file path.  Left as it is in the
        file when *None*.
    radiosonde_base : str or pathlib.Path, optional
        Root directory for local radiosonde files.  Left as it is in the
        file when *None*.
    output_dir : str or pathlib.Path, optional
        Directory for NetCDF results.  When *None*, the value already in
        the file is kept, or ``"./pyradtran_output"`` is written if there
        is none.
    working_dir : str or pathlib.Path, optional
        Scratch directory for ``uvspec`` files.  When *None*, the value
        already in the file is kept, or ``"./pyradtran_work"`` is written
        if there is none.
    max_workers : int, optional
        Number of parallel ``uvspec`` processes
        (``execution.max_workers``).  Left as it is in the file when
        *None*.
    extra : dict, optional
        Additional sections / keys merged into the file content, e.g.
        ``{"simulation_defaults": {"rte_solver": "disort"}}``.  Wins over
        the other arguments.

    Returns
    -------
    pathlib.Path
        Path of the master config file.

    See Also
    --------
    load_config : Reads the file as its second layer.
    PathsConfig : Meaning and validation of the path settings.

    Notes
    -----
    An existing master config is **updated**, not replaced: only the
    settings passed in this call change, everything else in the file
    (including hand-written sections) is kept.  An existing file that is
    not valid YAML is replaced, with a logged warning.  The values are
    written as given — no existence check, no short-name resolution, no
    ``~`` expansion; all of that happens when :func:`load_config` reads
    the file.  The directory ``~/.pyradtran`` is created if necessary and
    a confirmation line is printed.

    Relative *output_dir* / *working_dir* values stay relative in the
    file and are therefore resolved against the working directory of each
    later Python session.

    Examples
    --------
    One-time setup on a new machine:

    >>> from pyradtran import save_master_config
    >>> save_master_config(  # doctest: +SKIP
    ...     libradtran_bin="/opt/libRadtran-2.0.6/bin/uvspec",
    ...     libradtran_data="/opt/libRadtran-2.0.6/data",
    ...     atmosphere_profile="afglsw",
    ...     solar_spectrum="kurudz_1.0nm",
    ...     max_workers=4,
    ...     extra={"simulation_defaults": {"rte_solver": "disort"}},
    ... )
    Master config saved to: /home/user/.pyradtran/config.yaml
    PosixPath('/home/user/.pyradtran/config.yaml')

    which writes

    .. code-block:: yaml

        execution:
          max_workers: 4
        paths:
          atmosphere_profile: afglsw
          libradtran_bin: /opt/libRadtran-2.0.6/bin/uvspec
          libradtran_data: /opt/libRadtran-2.0.6/data
          output_dir: ./pyradtran_output
          solar_spectrum: kurudz_1.0nm
          working_dir: ./pyradtran_work
        simulation_defaults:
          rte_solver: disort

    Every later :func:`load_config` call picks these up:

    >>> cfg = load_config()  # doctest: +SKIP
    >>> cfg.paths.atmosphere_profile  # doctest: +SKIP
    PosixPath('/opt/libRadtran-2.0.6/data/atmmod/afglsw.dat')
    >>> cfg.simulation_defaults.rte_solver, cfg.execution.max_workers  # doctest: +SKIP
    ('disort', 4)
    """
    master_dir = Path.home() / ".pyradtran"
    master_dir.mkdir(parents=True, exist_ok=True)
    master_path = master_dir / "config.yaml"

    # Start from the existing file so earlier settings survive.
    content: Dict[str, Any] = {}
    if master_path.is_file():
        try:
            with open(master_path, "r") as f:
                content = yaml.safe_load(f)
        except (OSError, yaml.YAMLError) as e:
            logger.warning(f"Replacing unreadable master config {master_path}: {e}")
        if not isinstance(content, dict):
            content = {}

    paths = {
        "libradtran_bin": libradtran_bin,
        "libradtran_data": libradtran_data,
        "atmosphere_profile": atmosphere_profile,
        "solar_spectrum": solar_spectrum,
        "radiosonde_base": radiosonde_base,
        "output_dir": output_dir,
        "working_dir": working_dir,
    }
    update: Dict[str, Any] = {
        "paths": {k: str(v) for k, v in paths.items() if v is not None}
    }
    if max_workers is not None:
        update["execution"] = {"max_workers": max_workers}
    _recursive_update(content, update)
    if not isinstance(content.get("paths"), dict):
        content["paths"] = update["paths"]
    content["paths"].setdefault("output_dir", "./pyradtran_output")
    content["paths"].setdefault("working_dir", "./pyradtran_work")

    if extra:
        _recursive_update(content, extra)

    with open(master_path, "w") as f:
        yaml.dump(content, f, default_flow_style=False, indent=2, sort_keys=True)

    logger.info(f"Master config saved to {master_path}")
    print(f"Master config saved to: {master_path}")
    return master_path


def create_example_config(output_path: Union[str, Path]):
    """Write a complete example YAML config to *output_path*.

    The file lists every setting of the four sections (``paths``,
    ``simulation_defaults``, ``execution``, ``output``) with its default
    value and placeholder libRadtran paths (``/path/to/libradtran/...``).
    It is a template to edit by hand; the YAML carries no comments — see
    the module documentation of :mod:`pyradtran.config` for an annotated
    version.  The package default layer of :func:`load_config` is exactly
    this file.

    Parameters
    ----------
    output_path : str or pathlib.Path
        Destination file.  Parent directories are created automatically;
        an existing file is overwritten.

    Returns
    -------
    None
        A confirmation line is printed.

    See Also
    --------
    SimulationConfig.to_yaml : Write the *current* configuration instead,
        with the real paths of this machine.

    Notes
    -----
    Because of the placeholder paths the file cannot be loaded with
    :meth:`SimulationConfig.from_yaml` until ``paths`` has been edited.
    :func:`load_config` reads it as-is only if you remove the ``paths``
    block (so that the master config supplies it).

    Examples
    --------
    >>> import tempfile
    >>> from pathlib import Path
    >>> import yaml
    >>> from pyradtran import create_example_config
    >>> target = Path(tempfile.mkdtemp()) / "config" / "example.yaml"
    >>> create_example_config(target)  # doctest: +ELLIPSIS
    Example configuration created at: ...example.yaml
    >>> example = yaml.safe_load(target.read_text())
    >>> sorted(example)
    ['execution', 'output', 'paths', 'simulation_defaults']
    >>> example["simulation_defaults"]["wavelength_nm"]
    [400, 3600]
    >>> example["paths"]["libradtran_bin"]
    '/path/to/libradtran/bin/uvspec'
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(_DEFAULT_CONFIG_PATH, output_path)

    print(f"Example configuration created at: {output_path}")
