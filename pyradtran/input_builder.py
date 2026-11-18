"""
uvspec input-file generation with per-line provenance.

:class:`InputFileBuilder` turns a :class:`~pyradtran.config.SimulationConfig`
plus a resolved parameter mapping (from
:class:`~pyradtran.params.ParamResolver`) into an ordered list of
:class:`InputLine` objects. Each line records which layer produced it;
:meth:`~pyradtran.interface.PyRadtranAccessor.explain` shows this.

.. autosummary::

   InputFileBuilder
   InputLine
   calculate_solar_zenith_angle
   effective_output_columns
   effective_output_altitudes
   parametric_cloud_profiles

:func:`effective_output_columns` and :func:`effective_output_altitudes`
decide what a run reports. The builder (which writes the ``output_user``
/ ``zout`` lines) and the output parser (which reads the file) share
them, so the two cannot disagree.

Examples
--------
Preview a generated input file through the accessor (``ds`` has
``time``, ``latitude``, ``longitude`` and an ``albedo_sw`` variable):

>>> from pyradtran import Var
>>> print(ds.pyradtran.explain(params={"albedo": Var("albedo_sw")}))  # doctest: +SKIP
rte_solver twostr                                                   # config
mol_abs_param lowtran per_nm                                        # config
...
umu 1.0                                                             # config
albedo 0.82                                                         # dataset-var
quiet                                                               # config

See Also
--------
pyradtran.params : Parameter registry and resolution.
pyradtran.core.Simulation : Consumes the rendered input file.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .config import (
    ATMOSPHERE_PROFILES,
    SOLAR_SPECTRA,
    _resolve_libradtran_shortname,
)
from .params import PROV_CONFIG, REGISTRY, _zout_tokens

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InputLine:
    """One uvspec input line with provenance.

    Immutable record produced by :meth:`InputFileBuilder.build`.

    Parameters
    ----------
    keyword : str
        The (possibly multi-word) uvspec keyword, e.g. ``"mol_modify O3"``.
        Lines are replaced by keyword: a ``params`` entry removes every
        config line with the same keyword before adding its own.
    text : str
        The complete input line, keyword included.
    provenance : str
        Which layer produced the line: one of the ``PROV_*`` constants in
        :mod:`pyradtran.params` (``"config"``, ``"params-literal"``,
        ``"dataset-var"``, ``"unvalidated"``), or ``"era5"`` for cloud
        files derived from an ERA5 dataset.

    Examples
    --------
    >>> from pyradtran.input_builder import InputLine
    >>> line = InputLine("mol_modify O3", "mol_modify O3 320 DU", "params-literal")
    >>> line.text
    'mol_modify O3 320 DU'
    >>> line.provenance
    'params-literal'
    """

    keyword: str
    text: str
    provenance: str


def calculate_solar_zenith_angle(
    dt: datetime, latitude: float, longitude: float
) -> float:
    r"""Solar zenith angle (degrees) from time and location.

    :class:`InputFileBuilder` uses this estimate for solar runs when
    neither the configuration nor ``params`` supplies ``sza``.
    Declination and equation of time follow Spencer's (1971) Fourier
    series; the result is good to about 0.2° (see Notes).

    Parameters
    ----------
    dt : datetime.datetime
        Time of the observation. Naive datetimes are taken as **UTC**;
        aware ones are converted to UTC.
    latitude : float
        Geographic latitude :math:`\varphi` in degrees, north positive.
    longitude : float
        Geographic longitude :math:`\lambda` in degrees, east positive.

    Returns
    -------
    float
        Solar zenith angle in degrees, in :math:`[0, 180]`, rounded to
        two decimals. Above 90° the sun is below the horizon.

    See Also
    --------
    InputFileBuilder.build : Writes the result as the ``sza`` line.
    pyradtran.params.Var : Supply your own per-point ``sza`` instead.

    Notes
    -----
    Spencer's series are functions of the fractional-year angle
    :math:`\gamma = 2\pi (N - 1)/365` with :math:`N` the day of the
    year. The plain day number drifts by up to a day over the leap
    cycle (up to :math:`0.4^\circ` in declination near the equinoxes),
    so the angle is computed from continuous time and the tropical year,

    .. math::

        \gamma = 2\pi\,\frac{d}{365.2422},

    where :math:`d` is the time in days (with fraction) since
    2000-01-01 00:00 UTC. In the year 2000 this is Spencer's angle with
    a fractional day number.

    Spencer [1]_ gives the solar declination (in radians) as

    .. math::

        \delta = 0.006918 - 0.399912\cos\gamma + 0.070257\sin\gamma
                 - 0.006758\cos 2\gamma + 0.000907\sin 2\gamma
                 - 0.002697\cos 3\gamma + 0.00148\sin 3\gamma

    and the equation of time (in minutes) as

    .. math::

        E = 229.18\,(0.000075 + 0.001868\cos\gamma - 0.032077\sin\gamma
            - 0.014615\cos 2\gamma - 0.040849\sin 2\gamma)

    The hour angle follows from *true* solar time,

    .. math::

        h = 15^\circ\,\mathrm{h}^{-1}\,(t_\mathrm{UTC} - 12\,\mathrm{h})
            + \lambda + \frac{E}{4\,\mathrm{min}/^\circ}

    and the zenith angle :math:`\theta_0` from

    .. math::

        \cos\theta_0 = \sin\varphi\,\sin\delta
                     + \cos\varphi\,\cos\delta\,\cos h

    (:math:`\cos\theta_0` is clipped to :math:`[-1, 1]` before taking
    the arc cosine).

    **Accuracy.** Against the algorithm of the NOAA Solar Calculator
    (Meeus' *Astronomical Algorithms*), evaluated hourly over 1990–2050
    for latitudes 80° S – 80° N, the result deviates by at most 0.2°
    (declination within :math:`0.05^\circ`, equation of time within
    0.7 min). The angle is geometric (no refraction, no parallax) and
    refers to the centre of the solar disc.

    The relative error in :math:`\cos\theta_0`, and hence in the direct
    irradiance, is largest for a low sun. If that matters, compute the
    zenith angle with an ephemeris and pass it per point,
    ``params={"sza": Var("sza")}``.

    Scalar arguments only; arrays raise :class:`TypeError`.

    References
    ----------
    .. [1] Spencer, J. W. (1971). Fourier series representation of the
       position of the sun. *Search*, 2(5), 172.

    Examples
    --------
    At the June solstice the sun is nearly overhead at 23.44° N at
    12 UTC on the Greenwich meridian (true solar noon is at 12:02):

    >>> from datetime import datetime
    >>> from pyradtran.input_builder import calculate_solar_zenith_angle
    >>> calculate_solar_zenith_angle(datetime(2024, 6, 21, 12), 23.45, 0.0)
    0.38
    >>> calculate_solar_zenith_angle(datetime(2024, 6, 21, 12), 0.0, 0.0)
    23.46

    Ny-Ålesund (78.9° N, 11.9° E) in midsummer and in the polar night:

    >>> calculate_solar_zenith_angle(datetime(2024, 6, 21, 12), 78.9, 11.9)
    55.69
    >>> calculate_solar_zenith_angle(datetime(2024, 12, 21, 12), 78.9, 11.9)
    102.56

    Longitude shifts local solar time by one hour per 15°: 06 UTC at
    90° E is local noon, like 12 UTC at Greenwich (the small difference
    is the sun's motion during those six hours):

    >>> calculate_solar_zenith_angle(datetime(2024, 3, 20, 6), 51.5, 90.0)
    51.44
    >>> calculate_solar_zenith_angle(datetime(2024, 3, 20, 12), 51.5, 0.0)
    51.34

    Aware datetimes are converted to UTC (14:00 CEST is 12:00 UTC):

    >>> from datetime import timedelta, timezone
    >>> cest = timezone(timedelta(hours=2))
    >>> calculate_solar_zenith_angle(datetime(2024, 6, 21, 14, tzinfo=cest), 0.0, 0.0)
    23.46

    Diurnal cycle at the June solstice (dashed: horizon):

    .. plot::

        from datetime import datetime, timedelta
        from pyradtran.input_builder import calculate_solar_zenith_angle

        start = datetime(2024, 6, 21)
        hours = np.arange(0.0, 24.01, 0.25)
        fig, ax = plt.subplots()
        for lat in (0, 23.44, 50, 66.56, 78.9):
            sza = [
                calculate_solar_zenith_angle(start + timedelta(hours=h), lat, 0.0)
                for h in hours
            ]
            ax.plot(hours, sza, label=f"{lat:g}° N")
        ax.axhline(90.0, color="0.4", ls="--", lw=1)
        ax.set_xlabel("time on 21 June (UTC, longitude 0°)")
        ax.set_ylabel("solar zenith angle (°)")
        ax.set_xlim(0, 24)
        ax.set_xticks(range(0, 25, 3))
        ax.invert_yaxis()
        ax.legend(title="latitude", fontsize=8)
    """
    import numpy as np

    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc)
    dt = dt.replace(tzinfo=None)
    hours = dt.hour + dt.minute / 60.0 + dt.second / 3600.0
    # fractional-year angle from continuous time (no leap-cycle drift)
    days = (dt - datetime(2000, 1, 1)).total_seconds() / 86400.0
    g = 2 * np.pi * days / 365.2422
    # Spencer (1971): declination (rad) and equation of time (minutes)
    decl = (
        0.006918
        - 0.399912 * np.cos(g)
        + 0.070257 * np.sin(g)
        - 0.006758 * np.cos(2 * g)
        + 0.000907 * np.sin(2 * g)
        - 0.002697 * np.cos(3 * g)
        + 0.00148 * np.sin(3 * g)
    )
    eot = 229.18 * (
        0.000075
        + 0.001868 * np.cos(g)
        - 0.032077 * np.sin(g)
        - 0.014615 * np.cos(2 * g)
        - 0.040849 * np.sin(2 * g)
    )
    hour_rad = np.radians(15 * (hours - 12) + longitude + eot / 4)
    lat_rad = np.radians(latitude)

    cos_sza = np.sin(lat_rad) * np.sin(decl) + np.cos(lat_rad) * np.cos(decl) * np.cos(
        hour_rad
    )
    return round(float(np.degrees(np.arccos(np.clip(cos_sza, -1, 1)))), 2)


def effective_output_altitudes(
    config, overrides: Optional[Dict[str, Any]] = None
) -> List[float]:
    """Output altitudes of a run: the ``zout`` override, else the config.

    Parameters
    ----------
    config : SimulationConfig
        Only ``config.simulation_defaults.output_altitudes_km`` is read.
    overrides : dict, optional
        Plain ``{key: value}`` parameters of the run (no provenance
        tags). A ``"zout"`` entry (number, sequence or space-separated
        string) replaces the config altitudes.

    Returns
    -------
    list
        Levels in km as sorted, de-duplicated floats, matching the
        ``zout`` line the builder writes (uvspec rejects unsorted or
        repeated levels). If any level is symbolic (``toa``, ``sur``,
        ``cpt``), all tokens are returned as strings in their original
        order. An empty or missing config list gives ``[0.0]`` (the
        surface, uvspec's default).

    See Also
    --------
    effective_output_columns : Injects ``zout`` for multi-level runs.

    Examples
    --------
    Only ``simulation_defaults`` is read, so a stand-in for
    :class:`~pyradtran.config.SimulationConfig` is enough:

    >>> from types import SimpleNamespace
    >>> from pyradtran.config import SimulationDefaults
    >>> from pyradtran.input_builder import effective_output_altitudes
    >>> config = SimpleNamespace(simulation_defaults=SimulationDefaults())
    >>> effective_output_altitudes(config)
    [0.0]

    A ``zout`` override is sorted and de-duplicated:

    >>> effective_output_altitudes(config, {"zout": [10, 0, 2, 2]})
    [0.0, 2.0, 10.0]

    Symbolic levels are kept verbatim:

    >>> effective_output_altitudes(config, {"zout": "0 toa"})
    ['0', 'toa']
    """
    overrides = overrides or {}
    zout = overrides.get("zout")
    if zout is not None:
        return _zout_tokens(zout)
    return sorted(set(config.simulation_defaults.output_altitudes_km or [0.0]))


def effective_output_columns(
    config, overrides: Optional[Dict[str, Any]] = None
) -> List[str]:
    """Columns the ``output_user`` line requests, in file order.

    A per-run ``output_user`` override replaces the config columns.
    Spectral runs get ``lambda`` and multi-altitude runs get ``zout``;
    without them the batch converter cannot reshape the rows onto the
    wavelength/altitude axes and every value becomes NaN.
    :class:`~pyradtran.io.OutputParser` uses the same function, so the
    file and the parser cannot disagree.

    Parameters
    ----------
    config : SimulationConfig
        ``simulation_defaults.output_columns``, ``integrate_wavelength``
        and ``output_altitudes_km`` are read.
    overrides : dict, optional
        Plain ``{key: value}`` parameters of the run (no provenance
        tags). Only ``"output_user"`` (string or sequence of column
        names), ``"zout"`` and ``"output_quantity"`` are read.

    Returns
    -------
    list of str
        Column names in the order uvspec writes them. Rules, in order:

        1. ``overrides["output_user"]`` if given, else the config's
           ``output_columns``. An empty list is returned as is: no
           ``output_user`` line, uvspec's default output.
        2. With ``output_quantity brightness`` the ``albedo`` column is
           dropped (meaningless for brightness temperatures).
        3. Unless ``integrate_wavelength`` is set, ``lambda`` is put
           first if missing.
        4. With more than one output altitude
           (:func:`effective_output_altitudes`), ``zout`` is put first if
           missing.

    See Also
    --------
    effective_output_altitudes : The altitude list used for rule 4.
    InputFileBuilder.build : Writes the ``output_user`` line.

    Examples
    --------
    The default configuration is spectral with one output level:

    >>> from types import SimpleNamespace
    >>> from pyradtran.config import SimulationDefaults
    >>> from pyradtran.input_builder import effective_output_columns
    >>> config = SimpleNamespace(simulation_defaults=SimulationDefaults())
    >>> effective_output_columns(config)
    ['lambda', 'sza', 'eglo', 'eup', 'albedo']

    Several output levels add the ``zout`` column:

    >>> effective_output_columns(config, {"zout": [0, 5, 10]})
    ['zout', 'lambda', 'sza', 'eglo', 'eup', 'albedo']

    A per-run ``output_user`` replaces the configured columns;
    ``lambda`` is still added:

    >>> effective_output_columns(config, {"output_user": "edir edn eup"})
    ['lambda', 'edir', 'edn', 'eup']

    Brightness-temperature output drops ``albedo``:

    >>> effective_output_columns(config, {"output_quantity": "brightness"})
    ['lambda', 'sza', 'eglo', 'eup']

    Wavelength-integrated runs have no ``lambda`` column:

    >>> config.simulation_defaults.integrate_wavelength = True
    >>> effective_output_columns(config)
    ['sza', 'eglo', 'eup', 'albedo']
    """
    overrides = overrides or {}
    sd = config.simulation_defaults
    output_user = overrides.get("output_user")
    if output_user is not None:
        cols = (
            output_user.split() if isinstance(output_user, str) else list(output_user)
        )
    else:
        cols = list(sd.output_columns or [])
    if not cols:
        return cols  # empty means uvspec's default output
    if str(overrides.get("output_quantity", "")).strip() == "brightness":
        # albedo is meaningless for brightness temperatures; filtering it
        # here keeps the output_user line and the parser consistent
        cols = [c for c in cols if c != "albedo"]
    if not getattr(sd, "integrate_wavelength", False) and "lambda" not in cols:
        cols.insert(0, "lambda")
    if len(effective_output_altitudes(config, overrides)) > 1 and "zout" not in cols:
        cols.insert(0, "zout")
    return cols


def parametric_cloud_profiles(config) -> Dict[str, Dict[str, List[float]]]:
    r"""Cloud profiles of a ``cloud_source: parametric`` configuration.

    ``uvspec`` has no option for a parametric cloud layer. The layer in
    ``simulation_defaults.clouds`` is therefore expressed as a
    dict-valued ``wc_file`` / ``ic_file`` parameter, as in
    ``params={"wc_file": {...}}``, which
    :meth:`~pyradtran.core.Simulation.run_simulation` writes to a
    temporary cloud file.

    Parameters
    ----------
    config : SimulationConfig
        Only ``config.simulation_defaults.clouds`` is read.

    Returns
    -------
    dict
        ``{"wc_file": profile}`` and/or ``{"ic_file": profile}``
        depending on ``cloud_type`` (``wc``, ``ic``, ``mixed``). Empty
        unless clouds are enabled with ``cloud_source: parametric``.
        Each profile is a homogeneous layer in two rows, top first:
        ``{"z": [top, bottom], "lwc" | "iwc": [w, w], "reff": [r, r]}``
        (km, g m\ :sup:`-3`, µm). libRadtran reads the value at a level
        as the content of the layer above it, so the bottom row carries
        the cloud.

    See Also
    --------
    pyradtran.core.Simulation.format_cloud_profile : Writes the profile.
    pyradtran.config.CloudParameters : The configuration section.

    Examples
    --------
    >>> from types import SimpleNamespace
    >>> from pyradtran.config import CloudParameters, SimulationDefaults
    >>> from pyradtran.input_builder import parametric_cloud_profiles
    >>> clouds = CloudParameters(
    ...     enabled=True, cloud_source="parametric", water_content_g_m3=0.3
    ... )
    >>> config = SimpleNamespace(simulation_defaults=SimulationDefaults(clouds=clouds))
    >>> parametric_cloud_profiles(config)
    {'wc_file': {'z': [2.0, 1.0], 'lwc': [0.3, 0.3], 'reff': [10.0, 10.0]}}
    """
    clouds = config.simulation_defaults.clouds
    if not (clouds.enabled and clouds.cloud_source == "parametric"):
        return {}
    z = [float(clouds.layer_top_km), float(clouds.layer_bottom_km)]
    reff = [float(clouds.effective_radius_um)] * 2
    profiles = {}
    if clouds.cloud_type in ("wc", "mixed"):
        lwc = [float(clouds.water_content_g_m3)] * 2
        profiles["wc_file"] = {"z": z, "lwc": lwc, "reff": reff}
    if clouds.cloud_type in ("ic", "mixed"):
        iwc = [float(clouds.ice_content_g_m3)] * 2
        profiles["ic_file"] = {"z": z, "iwc": iwc, "reff": reff}
    return profiles


class InputFileBuilder:
    """Build provenance-tagged uvspec input lines.

    The merged configuration provides the baseline input file. The
    resolved ``params`` of one point (from
    :meth:`~pyradtran.params.ParamResolver.resolve_point`) replace or
    extend it. Every line records its origin.

    Parameters
    ----------
    config : SimulationConfig
        Merged configuration (package defaults, master config and
        project YAML, with dotted ``params`` overrides already applied).

    Attributes
    ----------
    config : SimulationConfig
        The configuration passed in; read on every :meth:`build` call.

    See Also
    --------
    pyradtran.params.ParamResolver : Produces the *resolved* mapping.
    pyradtran.interface.PyRadtranAccessor.explain : User-facing preview
        built on :meth:`build` and :meth:`render_annotated`.

    Examples
    --------
    The input file for one point (needs valid libRadtran paths; output
    from libRadtran 2.0.6):

    >>> from datetime import datetime
    >>> from pyradtran import ParamResolver, load_config
    >>> from pyradtran.input_builder import InputFileBuilder
    >>> config = load_config()  # doctest: +SKIP
    >>> resolver = ParamResolver(  # doctest: +SKIP
    ...     config,
    ...     {
    ...         "albedo": 0.85,
    ...         "wc_modify": ["tau550 set 12"],
    ...         "aerosol_default": True,
    ...         "mol_modify O3": False,
    ...     },
    ... )
    >>> builder = InputFileBuilder(config)  # doctest: +SKIP
    >>> lines = builder.build(  # doctest: +SKIP
    ...     datetime(2024, 6, 21, 12), 78.9, 11.9, resolver.static_params()
    ... )
    >>> print(builder.render_annotated(lines))  # doctest: +SKIP
    rte_solver twostr                                                   # config
    mol_abs_param lowtran per_nm                                        # config
    data_files_path /opt/libRadtran-2.0.6/data                          # config
    atmosphere_file /opt/libRadtran-2.0.6/data/atmmod/afglus.dat        # config
    mol_modify H2O 2.0 MM                                               # config
    source solar /opt/libRadtran-2.0.6/data/solar_flux/NewGuey2003.dat  # config
    sza 55.69                                                           # config
    day_of_year 173                                                     # config
    wavelength 400 3600                                                 # config
    output_user lambda sza eglo eup albedo                              # config
    zout 0.0000                                                         # config
    umu 1.0                                                             # config
    albedo 0.85                                                         # params-literal
    wc_modify tau550 set 12                                             # params-literal
    aerosol_default                                                     # params-literal
    quiet                                                               # config
    <BLANKLINE>

    ``"mol_modify O3": False`` removed the config's ozone line; ``sza``
    was computed from time and position.
    """

    def __init__(self, config):
        self.config = config

    def build(
        self,
        dt: datetime,
        latitude: float,
        longitude: float,
        resolved: Optional[Dict[str, Tuple[Any, str]]] = None,
        radiosonde_path: Optional[Path] = None,
        era5_atmosphere_file: Optional[Path] = None,
    ) -> List[InputLine]:
        """Build the full ordered line list for one simulation point.

        Parameters
        ----------
        dt : datetime.datetime
            Time of the point (UTC). Sets ``day_of_year`` and, when no
            ``sza`` is configured, the solar zenith angle.
        latitude, longitude : float
            Position in degrees (north / east positive). Used only for
            the computed solar zenith angle.
        resolved : dict, optional
            ``{key: (value, provenance)}`` as returned by
            :meth:`~pyradtran.params.ParamResolver.resolve_point`. Not
            modified.
        radiosonde_path : pathlib.Path, optional
            Radiosonde file. Used (as ``radiosonde <path> H2O RH``) only
            when the config has ``h2o_source: radiosonde`` and no ERA5
            file is given.
        era5_atmosphere_file : pathlib.Path, optional
            Atmosphere file written by :mod:`pyradtran.era5`. Takes
            precedence over *radiosonde_path*. Its gas columns are read
            from the file header.

        Returns
        -------
        list of InputLine
            The input file, one entry per line, in output order.

        See Also
        --------
        render, render_annotated : Turn the lines into text.
        calculate_solar_zenith_angle : Fallback for the ``sza`` line.
        effective_output_columns : Decides the ``output_user`` line.

        Notes
        -----
        Lines tagged ``config`` are written first, in a fixed order:

        1. ``rte_solver``, ``mol_abs_param``, ``data_files_path``,
           ``atmosphere_file``;
        2. ``radiosonde`` (ERA5 or radiosonde profile, if any);
        3. ``mol_modify O3 ... DU`` / ``mol_modify H2O ... MM`` from
           ``ozone_du`` / ``h2o_mm`` (both *None* by default), skipped
           for a gas whose profile comes from the radiosonde/ERA5 file,
           whether that file is set in the configuration or passed as
           ``radiosonde`` in *resolved*, so a measured column is never
           rescaled by the config scalar;
        4. ``source``; for solar runs ``sza`` (config value, else
           :func:`calculate_solar_zenith_angle`) and ``day_of_year``; for
           thermal runs ``sur_temperature`` if configured;
        5. ``wavelength`` and, for integrated runs,
           ``output_process integrate``;
        6. ``albedo`` (if configured), ``output_user``, ``zout`` (levels
           in km, sorted and de-duplicated), ``umu 1.0`` for nadir
           viewing, and the ``wc_file`` / ``ic_file`` lines of a
           ``cloud_source: file`` configuration.

        Catalog short names (``afglss``, ``kurudz_1.0nm``; see
        :func:`~pyradtran.config.list_atmosphere_profiles` and
        :func:`~pyradtran.config.list_solar_spectra`) are resolved to
        files in ``paths.libradtran_data`` on the ``atmosphere_file`` and
        ``source solar <file>`` lines, whether they come from the
        configuration or from *resolved*. Anything else is taken as a
        path and written unchanged.

        A ``cloud_source: parametric`` configuration is added to
        *resolved* as a dict-valued ``wc_file`` / ``ic_file`` (see
        :func:`parametric_cloud_profiles`) unless *resolved* already has
        that key. A dict-valued ``wc_file`` / ``ic_file`` is rendered as
        the placeholder ``wc_file 1D <generated: 2 levels, 1-2 km>``.
        Only :meth:`~pyradtran.core.Simulation.run_simulation` writes
        the cloud file, and it passes the path here instead of the dict.

        The *resolved* parameters are then applied in mapping order.
        Each one **removes every existing line with the same keyword**
        and appends its own line(s) at the end:

        * ``False``: remove only (switch a config-supplied option off);
        * a ``list``: one line per element (repeatable options such as
          ``wc_modify``);
        * a key of :data:`~pyradtran.params.REGISTRY`: formatted by its
          :meth:`~pyradtran.params.ParamSpec.format_line`
          (``mol_modify O3 320 DU``);
        * ``True``, ``None`` or ``""`` on any other key: the bare
          keyword (flag options such as ``aerosol_default``);
        * anything else: ``"<key> <value>"``.

        Special cases: ``sza``, ``sur_temperature``, ``albedo`` and
        ``zout`` in *resolved* suppress the corresponding config line;
        ``output_user`` is merged through
        :func:`effective_output_columns`; ``brdf_rpv_type`` drops any
        ``albedo`` line (config or resolved) and adds
        ``brdf_rpv_library IGBP`` unless ``albedo_library``,
        ``brdf_rpv_library`` or ``brdf_rpv_file`` is given. The file always
        ends with ``quiet``.

        Examples
        --------
        :class:`InputFileBuilder` shows a complete annotated input file.
        Per-point values carry the ``dataset-var`` tag:

        >>> lines = builder.build(  # doctest: +SKIP
        ...     datetime(2024, 6, 21, 12), 78.9, 11.9,
        ...     {"albedo": (0.82, "dataset-var")},
        ... )
        >>> [ln for ln in lines if ln.keyword == "albedo"]  # doctest: +SKIP
        [InputLine(keyword='albedo', text='albedo 0.82', provenance='dataset-var')]
        """
        resolved = dict(resolved or {})
        sd = self.config.simulation_defaults
        lines: List[InputLine] = []

        def add(keyword: str, text: str, provenance: str = PROV_CONFIG):
            lines.append(InputLine(keyword, text, provenance))

        add("rte_solver", f"rte_solver {sd.rte_solver}")
        if sd.mol_abs_param:
            add("mol_abs_param", f"mol_abs_param {sd.mol_abs_param}")

        paths = self.config.paths

        def data_file(value, catalog):
            # catalog short name ("afglss") -> file in the data directory;
            # anything else is a path and passes through unchanged
            if str(value) in catalog:
                return _resolve_libradtran_shortname(
                    value, Path(paths.libradtran_data), catalog
                )
            return value

        add("data_files_path", f"data_files_path {paths.libradtran_data}")
        add(
            "atmosphere_file",
            f"atmosphere_file {data_file(paths.atmosphere_profile, ATMOSPHERE_PROFILES)}",
        )

        # Atmosphere: ERA5 file wins over radiosonde
        radiosonde_columns = None
        if era5_atmosphere_file is not None:
            era5_abs_path = Path(era5_atmosphere_file).resolve()
            radiosonde_columns = _sniff_radiosonde_columns(era5_abs_path)
            add("radiosonde", f"radiosonde {era5_abs_path} {radiosonde_columns}")
        elif radiosonde_path and sd.h2o_source == "radiosonde":
            radiosonde_columns = "H2O RH"
            add("radiosonde", f"radiosonde {radiosonde_path} {radiosonde_columns}")

        # Molecular columns, when configured. A gas whose profile comes
        # from the radiosonde file must not also be rescaled by
        # mol_modify: that would overwrite the measured/reanalysis column
        # with the config scalar.
        profile_gases = (
            set(radiosonde_columns.split()[::2]) if radiosonde_columns else set()
        )
        if sd.ozone_du is not None and "O3" not in profile_gases:
            add("mol_modify O3", REGISTRY["mol_modify O3"].format_line(sd.ozone_du))
        if sd.h2o_mm is not None and "H2O" not in profile_gases:
            add("mol_modify H2O", REGISTRY["mol_modify H2O"].format_line(sd.h2o_mm))

        # Source
        if sd.source == "solar":
            solar_file = data_file(paths.solar_spectrum, SOLAR_SPECTRA)
            per_nm = " per_nm" if sd.integrate_wavelength else ""
            add("source", f"source solar {solar_file}{per_nm}")
            if "sza" not in resolved:
                sza = (
                    sd.sza
                    if sd.sza is not None
                    else calculate_solar_zenith_angle(dt, latitude, longitude)
                )
                add("sza", f"sza {sza}")
            add(
                "day_of_year",
                REGISTRY["day_of_year"].format_line(dt.timetuple().tm_yday),
            )
        elif sd.source == "thermal":
            add("source", "source thermal")
            if (
                "sur_temperature" not in resolved
                and sd.surface_temperature_k is not None
            ):
                add(
                    "sur_temperature",
                    f"sur_temperature {sd.surface_temperature_k}",
                )

        # Spectral range
        if sd.wavelength_nm and len(sd.wavelength_nm) == 2:
            wl_min, wl_max = sd.wavelength_nm
            add("wavelength", f"wavelength {wl_min} {wl_max}")
        if sd.integrate_wavelength:
            add("output_process", "output_process integrate")

        # Surface: BRDF type disables plain albedo
        if (
            "brdf_rpv_type" not in resolved
            and "albedo" not in resolved
            and sd.albedo_value is not None
        ):
            add("albedo", f"albedo {sd.albedo_value}")

        # Output columns: per-run output_user override wins; lambda/zout
        # are injected so the output stays parseable (see
        # effective_output_columns). Consumed here, not in the generic
        # override loop below.
        output_user_override = resolved.pop("output_user", None)
        raw_overrides = {k: v for k, (v, _p) in resolved.items()}
        if output_user_override is not None:
            raw_overrides["output_user"] = output_user_override[0]
        columns = effective_output_columns(self.config, raw_overrides)
        if columns:
            add(
                "output_user",
                "output_user " + " ".join(columns),
                output_user_override[1] if output_user_override else PROV_CONFIG,
            )

        # Output altitudes (sorted+deduped: uvspec rejects unsorted zout)
        if "zout" not in resolved and sd.output_altitudes_km:
            alt_str = " ".join(
                f"{alt:.4f}" for alt in effective_output_altitudes(self.config)
            )
            add("zout", f"zout {alt_str}")

        # Viewing geometry
        if sd.viewing_geometry == "nadir":
            add("umu", "umu 1.0")

        # Static cloud settings from config: files become lines, a
        # parametric layer becomes a dict-valued wc_file/ic_file parameter
        # (explicit params win)
        clouds = sd.clouds
        if clouds.enabled:
            if clouds.cloud_source == "file":
                for kind in ("wc", "ic"):
                    cloud_file = getattr(clouds, f"{kind}_file")
                    if clouds.cloud_type in (kind, "mixed") and cloud_file:
                        add(f"{kind}_file", f"{kind}_file {cloud_file}")
            for key, profile in parametric_cloud_profiles(self.config).items():
                resolved.setdefault(key, (profile, PROV_CONFIG))

        # BRDF surface type unconditionally replaces plain albedo,
        # regardless of dict ordering (the config albedo line was already
        # skipped above).
        if "brdf_rpv_type" in resolved:
            resolved.pop("albedo", None)

        # A user-chosen surface library/file keeps brdf_rpv_type meaning
        # "type N of that library" (e.g. albedo_library IGBP).
        own_surface = any(
            k in resolved
            for k in ("albedo_library", "brdf_rpv_library", "brdf_rpv_file")
        )

        # Apply resolved parameters: replace same-keyword lines, then append
        for key, (value, provenance) in resolved.items():
            spec = REGISTRY.get(key)
            keyword = spec.keyword if spec is not None else key
            lines = [ln for ln in lines if ln.keyword != keyword]
            if value is False:
                continue  # flag switched off: emit nothing
            if isinstance(value, list):
                # Repeatable option (non_unique): one line per entry
                for item in value:
                    lines.append(InputLine(keyword, f"{keyword} {item}", provenance))
                continue
            # Short names in params, as in the config
            if key == "atmosphere_file":
                value = data_file(value, ATMOSPHERE_PROFILES)
            elif key == "source":
                toks = str(value).split()
                if len(toks) > 1 and toks[0] == "solar":
                    toks[1] = str(data_file(toks[1], SOLAR_SPECTRA))
                    value = " ".join(toks)
            if isinstance(value, dict) and key in ("wc_file", "ic_file"):
                # Preview only: Simulation.run_simulation replaces the
                # dict by the temporary file before building.
                z = value.get("z", [])
                text = (
                    f"{key} 1D <generated: {len(z)} levels, "
                    f"{min(z):g}-{max(z):g} km>"
                )
            elif spec is not None:
                text = spec.format_line(value)
            elif value is True or value is None or value == "":
                text = keyword  # flag option, e.g. aerosol_default
            else:
                text = f"{key} {value}"
            lines.append(InputLine(keyword, text, provenance))
            # brdf_rpv_type alone implies the IGBP BRDF library
            if keyword == "brdf_rpv_type":
                lines = [ln for ln in lines if ln.keyword != "albedo"]
                if not own_surface:
                    lines = [ln for ln in lines if ln.keyword != "brdf_rpv_library"]
                    lines.append(
                        InputLine(
                            "brdf_rpv_library", "brdf_rpv_library IGBP", provenance
                        )
                    )

        # A radiosonde given through params brings its own gas columns, as
        # one from the config does: drop the config mol_modify scaling of
        # those gases (explicit mol_modify params are kept).
        for ln in lines:
            if ln.keyword == "radiosonde" and ln.provenance != PROV_CONFIG:
                gases = set(ln.text.split()[2::2])
                lines = [
                    x for x in lines
                    if not (x.provenance == PROV_CONFIG
                            and x.keyword in {f"mol_modify {g}" for g in gases})
                ]

        lines.append(InputLine("quiet", "quiet", PROV_CONFIG))
        return lines

    @staticmethod
    def render(lines: List[InputLine]) -> str:
        """Render lines as plain uvspec input file content.

        Parameters
        ----------
        lines : list of InputLine
            Lines from :meth:`build`.

        Returns
        -------
        str
            One ``text`` per line, newline-terminated: the content of
            the ``.inp`` file passed to ``uvspec``.

        Examples
        --------
        >>> from pyradtran.input_builder import InputFileBuilder, InputLine
        >>> lines = [
        ...     InputLine("albedo", "albedo 0.85", "params-literal"),
        ...     InputLine("sza", "sza 55.45", "config"),
        ...     InputLine("quiet", "quiet", "config"),
        ... ]
        >>> print(InputFileBuilder.render(lines), end="")
        albedo 0.85
        sza 55.45
        quiet
        """
        return "\n".join(ln.text for ln in lines) + "\n"

    @staticmethod
    def render_annotated(lines: List[InputLine]) -> str:
        """Render lines with aligned ``# <provenance>`` annotations.

        :meth:`~pyradtran.interface.PyRadtranAccessor.explain` returns
        this text. It is for reading, not for ``uvspec``.

        Parameters
        ----------
        lines : list of InputLine
            Lines from :meth:`build`.

        Returns
        -------
        str
            Each line padded to a common width and followed by
            ``# <provenance>``, newline-terminated. Empty for an empty
            list.

        Examples
        --------
        >>> from pyradtran.input_builder import InputFileBuilder, InputLine
        >>> lines = [
        ...     InputLine("albedo", "albedo 0.85", "params-literal"),
        ...     InputLine("sza", "sza 55.45", "config"),
        ...     InputLine("quiet", "quiet", "config"),
        ... ]
        >>> print(InputFileBuilder.render_annotated(lines), end="")
        albedo 0.85  # params-literal
        sza 55.45    # config
        quiet        # config
        """
        if not lines:
            return ""
        width = max(len(ln.text) for ln in lines) + 2
        return "\n".join(f"{ln.text:<{width}}# {ln.provenance}" for ln in lines) + "\n"


def _sniff_radiosonde_columns(era5_abs_path: Path) -> str:
    """Read the gas-column spec from an atmosphere file header.

    Uses the ``# columns: H2O MMR O3 MMR`` line written by
    :mod:`pyradtran.era5`; without it, ``%`` in the header means
    ``H2O RH``, otherwise ``H2O MMR``.
    """
    try:
        with open(era5_abs_path, "r") as f:
            header = [f.readline().strip() for _ in range(3)]
        for line in header:
            if line.lower().startswith("# columns:"):
                return line.split(":", 1)[1].strip()
        if any("%" in line for line in header[:2]):
            return "H2O RH"
    except OSError:
        logger.warning(
            f"Could not sniff columns from {era5_abs_path}; assuming H2O MMR"
        )
    return "H2O MMR"


__all__ = [
    "InputLine",
    "InputFileBuilder",
    "calculate_solar_zenith_angle",
    "effective_output_columns",
    "effective_output_altitudes",
    "parametric_cloud_profiles",
]
