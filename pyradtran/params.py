"""
The ``params`` system: registry, markers, validation and discovery.

Every per-run setting of
:meth:`~pyradtran.interface.PyRadtranAccessor.run` travels through one
mapping, ``params={key: value}``. This module defines the allowed keys
and values, validates them before any ``uvspec`` process starts, and
lists the options that exist:

.. autosummary::

   Var
   Raw
   describe
   search_options
   ParamResolver
   ParamSpec
   validate_against_schema
   get_schema

Notes
-----
.. rubric:: Keys

A key selects what is set:

``"albedo"``, ``"aerosol_default"``, ...
    A uvspec option name. Any option of the local libRadtran install is
    accepted (see :func:`search_options` / :func:`describe`); a
    misspelt name is rejected with a "did you mean ...?" hint.
``"mol_modify O3"``, ``"wc_modify tau550 set"``, ...
    A multi-word key: the option name followed by its leading
    arguments; the value supplies the remaining ones. Several lines of a
    repeatable option can be set, and replaced, independently this way.
``"simulation_defaults.rte_solver"``, ...
    A dotted configuration path ``<section>.<field>`` (a ``.`` and no
    space in the key). The value is written into the
    :class:`~pyradtran.config.SimulationConfig` of the run and the entry
    is consumed; it never reaches the input file directly. Use it for
    settings that change how the input file is built (``source``,
    ``integrate_wavelength``, ``output_altitudes_km``, ...). An unknown
    path is rejected with a "did you mean ...?" hint.

.. rubric:: Values

A value selects how the option is set:

literal number or string
    Written after the keyword: ``{"albedo": 0.85}`` → ``albedo 0.85``.
:class:`Var`
    Taken per simulation point from a variable of the input dataset.
:class:`Raw`
    Written verbatim without any validation.
``True``
    A flag option without arguments: ``{"aerosol_default": True}`` →
    ``aerosol_default``.
``False``
    Omit the option, removing a line the configuration would otherwise
    write: ``{"mol_modify O3": False}``.
``tuple``
    Space-joined into one line: ``{"wavelength": (400, 700)}``.
``list``
    On a repeatable option (``non_unique`` in the schema, e.g.
    ``wc_modify``, ``crs_model``) one input line per element; on any
    other option space-joined like a tuple.

A plain string that is also the name of a dataset variable is rejected
as ambiguous; write ``Var("name")`` if the variable is meant.

.. rubric:: Provenance

Every resolved value carries a tag saying where it came from. The tags
appear in :meth:`~pyradtran.interface.PyRadtranAccessor.explain`:

========================  ====================  ==========================
Constant                  Tag                   Meaning
========================  ====================  ==========================
:data:`PROV_CONFIG`       ``"config"``          line derived from the
                                                configuration
:data:`PROV_LITERAL`      ``"params-literal"``  validated literal from
                                                ``params``
:data:`PROV_DATASET`      ``"dataset-var"``     per-point value from a
                                                :class:`Var`
:data:`PROV_UNVALIDATED`  ``"unvalidated"``     :class:`Raw` value, or any
                                                non-registry key when no
                                                option schema is available
========================  ====================  ==========================

``params`` entries always win over the configuration.

.. rubric:: Validation

Two layers check a value. The curated :data:`REGISTRY` below knows
units, ranges and special formatting for the most common parameters.
Everything else is checked against the option schema extracted from the
local libRadtran install (:mod:`pyradtran.schema`): option name, number
of arguments, allowed choices, numeric ranges. Without a schema (no
``src_py`` directory next to ``uvspec``) only the registry is applied
and other keys pass through tagged ``unvalidated``.

The curated registry (:data:`REGISTRY`):

===================  =====  =======================  =========================
Key                  Units  Valid range / choices    Notes
===================  =====  =======================  =========================
``albedo``           –      0 … 1                    Lambertian surface albedo
``sur_temperature``  K      0 … 1000                 surface temperature
                                                     (thermal source)
``sza``              deg    0 … 180                  solar zenith angle
``zout``             km     –                        output level(s); numeric
                                                     levels are sorted and
                                                     de-duplicated
``brdf_rpv_type``    –      1 … 20                   IGBP surface type; drops
                                                     ``albedo`` and writes
                                                     ``brdf_rpv_library IGBP``
                                                     unless a surface library
                                                     or file is given
``wavelength``       nm     –                        wavelength range
                                                     ``(min, max)``
``mol_modify O3``    DU     0 … 1000                 total ozone column; a
                                                     bare number gets the
                                                     unit appended, a string
                                                     with its own unit
                                                     (``"320 DU"``) is kept
``mol_modify H2O``   MM     0 … 200                  precipitable water in
                                                     mm; as for O3
``rte_solver``       –      any solver of the local  RTE solver (checked
                            libRadtran               against the option
                                                     schema; without one,
                                                     against the libRadtran
                                                     2.0.6 list)
``mol_abs_param``    –      –                        molecular absorption
                                                     parameterisation
``umu``              –      −1 … 1                   cosine of the viewing
                                                     zenith angle
``output_user``      –      –                        output columns
``source``           –      ``solar``, ``thermal``   radiation source
``day_of_year``      –      1 … 366                  day of year (Sun–Earth
                                                     distance)
===================  =====  =======================  =========================

Ranges are inclusive. For the keys in :data:`CONFIG_FIELD_MAP`
(``albedo``, ``sur_temperature``, ``sza``, ``rte_solver``,
``mol_abs_param``, ``source``, ``mol_modify O3``, ``mol_modify H2O``)
the configuration holds a default that ``params`` overrides.

Examples
--------
A typical mapping (``ds`` has the variables ``albedo_sw`` and ``cod``):

>>> from pyradtran import Var, Raw
>>> params = {
...     "albedo": Var("albedo_sw"),  # per point, from the dataset
...     "mol_modify O3": 320,  # curated: -> "mol_modify O3 320 DU"
...     "wc_modify tau550 set": Var("cod"),  # multi-word key
...     "crs_model": ["rayleigh Bodhaine", "o3 Bass_and_Paur"],  # two lines
...     "aerosol_default": True,  # flag
...     "pseudospherical": Raw(True),  # not validated
...     "simulation_defaults.rte_solver": "disort",  # config override
... }
>>> result = ds.pyradtran.run(params=params)  # doctest: +SKIP

Check before you run: which options exist, what they take and what the
input file will be.

>>> from pyradtran import describe, search_options
>>> search_options("visibility")  # doctest: +SKIP
['aerosol_default', 'aerosol_visibility']
>>> print(describe("aerosol_visibility"))  # doctest: +SKIP
aerosol_visibility <float in [0.0, 1000000.0]>
group: Aerosol
...
>>> print(ds.pyradtran.explain(params=params))  # doctest: +SKIP
rte_solver disort                                                   # config
...
mol_modify O3 320 DU                                                # params-literal
crs_model rayleigh Bodhaine                                         # params-literal
crs_model o3 Bass_and_Paur                                          # params-literal
aerosol_default                                                     # params-literal
pseudospherical                                                     # unvalidated
albedo 0.82                                                         # dataset-var
wc_modify tau550 set 12.0                                           # dataset-var
quiet                                                               # config

See Also
--------
pyradtran.input_builder : Renders resolved parameters to uvspec input lines.
pyradtran.schema : Extraction of the libRadtran option schema.
pyradtran.interface.PyRadtranAccessor.run : Where ``params`` is passed.
"""

import difflib
import logging
import math
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

from .exceptions import ValidationError

logger = logging.getLogger(__name__)

# Provenance tags (exact strings used in explain() output)

#: Provenance tag of input lines derived from the configuration.
PROV_CONFIG = "config"
#: Provenance tag of validated literal values from ``params``.
PROV_LITERAL = "params-literal"
#: Provenance tag of per-point values resolved from a :class:`Var`.
PROV_DATASET = "dataset-var"
#: Provenance tag of values that were not validated: :class:`Raw` values,
#: and any non-registry key when no libRadtran option schema is available.
PROV_UNVALIDATED = "unvalidated"


@dataclass(frozen=True)
class Var:
    """Per-point reference to a variable of the input dataset.

    Used as a value in the ``params`` mapping of
    :meth:`~pyradtran.interface.PyRadtranAccessor.run`: the parameter
    takes, at every simulation point, the value of the named variable at
    that point.

    Parameters
    ----------
    name : str
        Name of a data variable or coordinate of the input dataset.

    See Also
    --------
    Raw : Pass a literal value through without validation.
    ParamResolver.resolve_point : Where references are resolved.
    pyradtran.interface.PyRadtranAccessor.explain : Preview the resolved
        input file of one point.

    Notes
    -----
    * **Missing variable.** If *name* is not in the dataset,
      :class:`~pyradtran.exceptions.ValidationError` is raised before
      any simulation starts.
    * **Validation.** Each per-point value is validated exactly like a
      literal (range, choices, option schema); a bad value raises
      :class:`~pyradtran.exceptions.ValidationError` naming the key.
    * **NaN means "not set here".** Where the variable is NaN the
      parameter is omitted for that point: the configuration's line for
      that option is used if there is one, otherwise uvspec's default.
      The point is still simulated.
    * **Strings are literals.** ``{"albedo": "albedo_sw"}`` is not a
      reference; a bare string that matches a dataset variable name is
      rejected as ambiguous.
    * Lines produced by a reference are tagged ``dataset-var``
      (:data:`PROV_DATASET`).

    The key may be any key that accepts a literal, including multi-word
    keys: ``{"wc_modify tau550 set": Var("cod")}`` writes
    ``wc_modify tau550 set <cod at this point>``.

    Examples
    --------
    Surface albedo and surface temperature from the dataset, everything
    else from the configuration:

    >>> from pyradtran import Var
    >>> result = ds.pyradtran.run(  # doctest: +SKIP
    ...     config_path="thermal.yaml",
    ...     params={
    ...         "albedo": Var("albedo_lw"),
    ...         "sur_temperature": Var("surface_temperature"),
    ...     },
    ... )

    Per point, shown with the resolver that ``run`` uses (``schema=None``
    keeps the example independent of libRadtran; the stand-in ``config``
    is normally a :class:`~pyradtran.config.SimulationConfig`):

    >>> import numpy as np
    >>> import xarray as xr
    >>> from types import SimpleNamespace
    >>> from pyradtran.params import ParamResolver
    >>> ds = xr.Dataset({"albedo_sw": ("time", [0.82, np.nan, 1.7])})
    >>> config = SimpleNamespace()
    >>> resolver = ParamResolver(config, {"albedo": Var("albedo_sw")}, schema=None)
    >>> resolver.resolve_point(ds.isel(time=0))
    ({'albedo': (0.82, 'dataset-var')}, [])

    NaN: the parameter is left out for this point and reported as
    skipped:

    >>> resolver.resolve_point(ds.isel(time=1))
    ({}, ['albedo'])

    An invalid value is caught before uvspec sees it:

    >>> resolver.resolve_point(ds.isel(time=2))
    Traceback (most recent call last):
        ...
    pyradtran.exceptions.ValidationError: Invalid per-point value(s): 'albedo' must be in [0.0, 1.0], got 1.7

    A reference to a variable the dataset does not have:

    >>> ParamResolver(
    ...     config, {"albedo": Var("albedo")}, schema=None
    ... ).validate_var_targets(ds)
    Traceback (most recent call last):
        ...
    pyradtran.exceptions.ValidationError: 'albedo' references dataset variable 'albedo' which is not in the dataset

    In the annotated input file the two points differ only in the albedo
    line (``albedo_sw`` is 0.82 at the first time and NaN at the second;
    the configuration has ``albedo_value: 0.85``):

    >>> params = {"albedo": Var("albedo_sw")}
    >>> print(ds.pyradtran.explain(params=params))  # doctest: +SKIP
    rte_solver twostr                                                   # config
    ...
    albedo 0.82                                                         # dataset-var
    quiet                                                               # config
    >>> print(  # doctest: +SKIP
    ...     ds.pyradtran.explain(point={"time": "2024-06-21 13:00"}, params=params)
    ... )
    rte_solver twostr                                                   # config
    ...
    albedo 0.85                                                         # config
    ...
    quiet                                                               # config
    """

    name: str


@dataclass(frozen=True)
class Raw:
    """Pass a ``params`` value through without any validation.

    For options, or option values, that validation rejects although your
    ``uvspec`` accepts them: an option the local libRadtran schema does
    not know (e.g. a patched build), or a value outside the range of the
    curated :data:`REGISTRY`. The value reaches the input file verbatim,
    tagged ``unvalidated`` (:data:`PROV_UNVALIDATED`).

    Parameters
    ----------
    value : Any
        The literal value to emit after the keyword. ``True`` emits the
        bare keyword (flag option); a ``list`` emits one line per
        element.

    See Also
    --------
    Var : Per-point values from the dataset.
    describe : Check what an option expects before bypassing validation.

    Notes
    -----
    ``Raw`` also switches off the typo protection: a misspelt option name
    is written to the input file as is, and only uvspec will complain.
    A ``Raw`` value is the same for every point; it cannot wrap a
    :class:`Var`.

    Examples
    --------
    >>> from pyradtran import Raw
    >>> params = {
    ...     "my_patched_option": Raw("on"),  # unknown to the local schema
    ...     "pseudospherical": Raw(True),  # flag option
    ... }
    >>> result = ds.pyradtran.run(params=params)  # doctest: +SKIP

    With a stand-in schema that only knows ``albedo``, an unknown option
    name is rejected unless wrapped in ``Raw``:

    >>> from types import SimpleNamespace
    >>> from pyradtran.params import ParamResolver
    >>> schema = {"albedo": {"name": "albedo", "tokens": []}}
    >>> config = SimpleNamespace()
    >>> ParamResolver(config, {"albedoo": 0.5}, schema=schema)
    Traceback (most recent call last):
        ...
    pyradtran.exceptions.ValidationError: Invalid parameter value(s): 'albedoo' is not a known uvspec option of the local libRadtran install; did you mean albedo? (wrap the value in Raw(...) to bypass validation)
    >>> ParamResolver(config, {"albedoo": Raw(0.5)}, schema=schema).static_params()
    {'albedoo': (0.5, 'unvalidated')}
    """

    value: Any


@dataclass(frozen=True)
class ParamSpec:
    """Declarative description of one curated uvspec parameter.

    The entries of :data:`REGISTRY` are instances of this class. A spec
    knows how to check a value (:meth:`validate`) and how to write the
    input line (:meth:`format_line`).

    Parameters
    ----------
    keyword : str
        The uvspec keyword (possibly multi-word, e.g. ``"mol_modify O3"``).
    dtype : type, default ``float``
        Expected scalar type. Values are not converted. When this is
        ``float`` or ``int``, :meth:`validate` rejects booleans and
        applies the range check.
    units : str, optional
        Physical units, for documentation and error messages.
    valid_range : tuple of float, optional
        Inclusive ``(lo, hi)`` range for numeric values.
    choices : tuple, optional
        Allowed values for enumerated parameters (exact, case-sensitive
        match).
    applicability : {"both", "solar", "thermal"}
        Which source modes the parameter applies to. Informational
        only; not enforced.
    formatter : callable, optional
        ``value -> full input line``. Defaults to ``"{keyword} {value}"``.
    doc : str, default ``""``
        One-line description.

    See Also
    --------
    REGISTRY : The curated specs shipped with pyradtran.
    validate_against_schema : Validation of all other uvspec options.

    Examples
    --------
    A spec from the registry:

    >>> from pyradtran import REGISTRY
    >>> spec = REGISTRY["mol_modify O3"]
    >>> spec.units, spec.valid_range
    ('DU', (0.0, 1000.0))
    >>> spec.format_line(320)
    'mol_modify O3 320 DU'

    Defining one (specs are plain frozen dataclasses):

    >>> from pyradtran import ParamSpec
    >>> pressure = ParamSpec("pressure", float, "hPa", (0, 1100), doc="Surface pressure")
    >>> pressure.format_line(950.0)
    'pressure 950.0'
    >>> pressure.validate(2000)
    Traceback (most recent call last):
        ...
    pyradtran.exceptions.ValidationError: 'pressure' must be in [0, 1100] hPa, got 2000.0
    """

    keyword: str
    dtype: type = float
    units: Optional[str] = None
    valid_range: Optional[Tuple[float, float]] = None
    choices: Optional[tuple] = None
    applicability: str = "both"
    formatter: Optional[Callable[[Any], str]] = None
    doc: str = ""

    def validate(self, value: Any) -> None:
        """Raise ``ValidationError`` if *value* is out of spec.

        Parameters
        ----------
        value : Any
            The candidate value.

        Returns
        -------
        None
            The value is not converted or returned.

        Raises
        ------
        ValidationError
            If the spec has ``choices`` and *value* is not one of them;
            or the spec is numeric and *value* is a ``bool``; or the
            spec is numeric with a ``valid_range`` and *value* is not a
            number or lies outside the (inclusive) range.

        Notes
        -----
        The checks, in order:

        1. A spec with ``choices`` accepts exactly those values.
        2. Otherwise a ``str`` is accepted as is, which allows raw uvspec
           syntax such as ``"0 toa"`` for ``zout``.
           (:class:`ParamResolver` also checks such strings against the
           libRadtran option schema when one is available.)
        3. Otherwise a ``float`` / ``int`` spec rejects ``True`` /
           ``False`` (a boolean is a flag, not a number).
        4. Otherwise, for ``float`` / ``int`` specs with a
           ``valid_range``, the value must convert with :class:`float`
           and satisfy ``lo <= value <= hi``.

        Specs without ``choices`` and without ``valid_range`` (e.g.
        ``wavelength``, ``zout``) accept anything but a boolean.

        Examples
        --------
        >>> from pyradtran import REGISTRY
        >>> REGISTRY["albedo"].validate(0.3)  # fine: returns None
        >>> REGISTRY["albedo"].validate(1.5)
        Traceback (most recent call last):
            ...
        pyradtran.exceptions.ValidationError: 'albedo' must be in [0.0, 1.0], got 1.5

        Units are part of the message where the spec has them:

        >>> REGISTRY["sza"].validate(200)
        Traceback (most recent call last):
            ...
        pyradtran.exceptions.ValidationError: 'sza' must be in [0.0, 180.0] deg, got 200.0

        Enumerated parameters:

        >>> REGISTRY["source"].validate("lunar")
        Traceback (most recent call last):
            ...
        pyradtran.exceptions.ValidationError: 'source' must be one of ('solar', 'thermal'), got 'lunar'

        A boolean is not a number:

        >>> REGISTRY["albedo"].validate(True)
        Traceback (most recent call last):
            ...
        pyradtran.exceptions.ValidationError: 'albedo' expects a number, got True
        """
        if self.choices is not None:
            if value not in self.choices:
                raise ValidationError(
                    f"'{self.keyword}' must be one of {self.choices}, got {value!r}"
                )
            return
        if isinstance(value, str):
            return
        if self.dtype in (float, int) and isinstance(value, bool):
            raise ValidationError(f"'{self.keyword}' expects a number, got {value!r}")
        if self.dtype in (float, int) and self.valid_range is not None:
            try:
                v = float(value)
            except (TypeError, ValueError):
                raise ValidationError(
                    f"'{self.keyword}' expects a number, got {value!r}"
                )
            lo, hi = self.valid_range
            if not (lo <= v <= hi):
                unit = f" {self.units}" if self.units else ""
                raise ValidationError(
                    f"'{self.keyword}' must be in [{lo}, {hi}]{unit}, got {v}"
                )

    def format_line(self, value: Any) -> str:
        """Render the full uvspec input line for *value*.

        Parameters
        ----------
        value : Any
            The (already validated) value.

        Returns
        -------
        str
            The spec's ``formatter(value)`` if it has one, otherwise
            ``"<keyword> <value>"``.

        Examples
        --------
        >>> from pyradtran import REGISTRY
        >>> REGISTRY["albedo"].format_line(0.3)
        'albedo 0.3'

        Custom formatters append units, cast to integer, or tidy lists.
        ``zout`` levels are sorted and de-duplicated, because uvspec
        rejects anything else:

        >>> REGISTRY["mol_modify H2O"].format_line(14.5)
        'mol_modify H2O 14.5 MM'
        >>> REGISTRY["mol_modify O3"].format_line("320 DU")  # unit given: kept
        'mol_modify O3 320 DU'
        >>> REGISTRY["brdf_rpv_type"].format_line(7.0)
        'brdf_rpv_type 7'
        >>> REGISTRY["zout"].format_line([10, 0, 2, 2])
        'zout 0 2 10'
        >>> REGISTRY["zout"].format_line("0 toa")
        'zout 0 toa'
        """
        if self.formatter is not None:
            return self.formatter(value)
        return f"{self.keyword} {value}"


def _fmt_mol_modify(gas: str, unit: str) -> Callable[[Any], str]:
    """Return a formatter writing ``mol_modify <gas> <value> <unit>``.

    A string value that already carries a unit (``"320 DU"``) is written
    as given.
    """

    def fmt(value: Any) -> str:
        if isinstance(value, str) and len(value.split()) > 1:
            return f"mol_modify {gas} {value}"
        return f"mol_modify {gas} {value} {unit}"

    return fmt


def _zout_tokens(value: Any) -> list:
    """Normalise a zout value to sorted unique floats.

    uvspec hard-errors on unsorted or duplicated zout levels.  Symbolic
    levels (``toa``, ``sur``, ``cpt``) cannot be ordered numerically and
    are returned untouched, in their original token order.
    """
    if isinstance(value, str):
        toks = value.split()
    elif isinstance(value, (list, tuple)):
        toks = [str(v) for v in value]
    else:
        toks = [str(value)]
    try:
        return sorted({float(t) for t in toks})
    except ValueError:
        return toks


def _fmt_zout(value: Any) -> str:
    """Format the ``zout`` line from sorted, de-duplicated levels."""
    return "zout " + " ".join(
        f"{t:g}" if isinstance(t, float) else str(t) for t in _zout_tokens(value)
    )


#: Curated registry ``{params key: ParamSpec}`` of the most common uvspec
#: parameters, with units, valid ranges and special formatting. Keys not
#: listed here are validated against the libRadtran option schema instead
#: (see :func:`validate_against_schema`). The module docstring of
#: :mod:`pyradtran.params` tabulates the entries.
REGISTRY: Dict[str, ParamSpec] = {
    "albedo": ParamSpec(
        "albedo", float, None, (0.0, 1.0), doc="Lambertian surface albedo"
    ),
    "sur_temperature": ParamSpec(
        "sur_temperature",
        float,
        "K",
        (0.0, 1000.0),
        applicability="thermal",
        doc="Surface temperature for thermal source",
    ),
    "sza": ParamSpec(
        "sza",
        float,
        "deg",
        (0.0, 180.0),
        applicability="solar",
        doc="Solar zenith angle",
    ),
    "zout": ParamSpec(
        "zout",
        float,
        "km",
        None,
        formatter=_fmt_zout,
        doc="Output altitude level(s)",
    ),
    "brdf_rpv_type": ParamSpec(
        "brdf_rpv_type",
        int,
        None,
        (1, 20),
        formatter=lambda v: f"brdf_rpv_type {int(v)}",
        doc="IGBP surface type for RPV BRDF library",
    ),
    "wavelength": ParamSpec(
        "wavelength", float, "nm", None, doc="Wavelength range [min max]"
    ),
    "mol_modify O3": ParamSpec(
        "mol_modify O3",
        float,
        "DU",
        (0.0, 1000.0),
        formatter=_fmt_mol_modify("O3", "DU"),
        doc="Total ozone column",
    ),
    "mol_modify H2O": ParamSpec(
        "mol_modify H2O",
        float,
        "MM",
        (0.0, 200.0),
        formatter=_fmt_mol_modify("H2O", "MM"),
        doc="Precipitable water column",
    ),
    # No curated ``choices``: the local libRadtran schema is authoritative
    # (see ParamResolver._validate_key_value for the no-schema fallback).
    "rte_solver": ParamSpec(
        "rte_solver", str, doc="Radiative transfer equation solver"
    ),
    "mol_abs_param": ParamSpec(
        "mol_abs_param", str, doc="Molecular absorption parameterisation"
    ),
    "umu": ParamSpec(
        "umu", float, None, (-1.0, 1.0), doc="Cosine of viewing zenith angle"
    ),
    "output_user": ParamSpec("output_user", str, doc="Output column specification"),
    "source": ParamSpec(
        "source", str, choices=("solar", "thermal"), doc="Radiation source"
    ),
    "day_of_year": ParamSpec(
        "day_of_year",
        int,
        None,
        (1, 366),
        formatter=lambda v: f"day_of_year {int(v)}",
        doc="Day of year for Sun–Earth distance correction",
    ),
}

#: Registry key -> name of the
#: :class:`~pyradtran.config.SimulationDefaults` attribute that provides the
#: configuration default for that parameter (e.g. ``"albedo"`` ->
#: ``"albedo_value"``). Used by
#: :meth:`~pyradtran.interface.PyRadtranAccessor.jacobian` to find the base
#: value of a parameter that is not in ``params``.
CONFIG_FIELD_MAP: Dict[str, str] = {
    "albedo": "albedo_value",
    "sur_temperature": "surface_temperature_k",
    "sza": "sza",
    "rte_solver": "rte_solver",
    "mol_abs_param": "mol_abs_param",
    "source": "source",
    "mol_modify O3": "ozone_du",
    "mol_modify H2O": "h2o_mm",
}


#: ``rte_solver`` values of libRadtran 2.0.6. Only consulted when no local
#: option schema is available; otherwise the schema decides.
_RTE_SOLVERS_FALLBACK = frozenset(
    "disort twostr mystic rodents sslidar null sdisort spsdisort fdisort1 "
    "fdisort2 sos ftwostr montecarlo tzs sssi sss twostrebe schwarzschild "
    "twomaxrnd twomaxrnd3C dynamic_twostream dynamic_tenstream".split()
)

#: Memo of loaded schemas, keyed by libRadtran root path.
_SCHEMA_MEMO: Dict[str, Optional[Dict[str, Any]]] = {}


def get_schema(config) -> Optional[Dict[str, Any]]:
    """Load (and memoise) the libRadtran option schema for *config*.

    Thin wrapper around :func:`pyradtran.schema.load_schema` that keeps
    the parsed schema in memory, so the JSON cache is read at most once
    per libRadtran installation and process.

    Parameters
    ----------
    config : SimulationConfig
        ``config.paths.libradtran_bin`` / ``libradtran_data`` locate the
        installation (see :func:`pyradtran.schema.find_libradtran_root`).

    Returns
    -------
    dict or None
        ``{option_name: schema_entry}``, or ``None`` when no local
        libRadtran source tree (a directory with ``src_py``) is
        available or the extraction failed. Validation then falls back
        to the curated :data:`REGISTRY` only. The same object is
        returned on every call; treat it as read-only.

    See Also
    --------
    pyradtran.schema : Extraction, on-disk cache and entry structure.
    describe, search_options : Human-readable views of the schema.

    Examples
    --------
    >>> from pyradtran import load_config
    >>> from pyradtran.params import get_schema
    >>> schema = get_schema(load_config())  # doctest: +SKIP
    >>> len(schema)  # doctest: +SKIP
    244
    >>> schema["sza"]["tokens"]  # doctest: +SKIP
    [{'kind': 'value', 'datatype': 'float', 'valid_range': [0.0, 180.0], 'optional': False}]
    """
    from .schema import find_libradtran_root, load_schema

    root = find_libradtran_root(config)
    key = str(root)
    if key not in _SCHEMA_MEMO:
        _SCHEMA_MEMO[key] = load_schema(root=root) if root else None
    return _SCHEMA_MEMO[key]


def _tokenize_value(key: str, value: Any):
    """Split a params key/value pair into uvspec tokens after the option name.

    Returns *None* when the value is not token-like (dict/list cloud
    profiles), meaning schema validation must be skipped.
    """
    extra = key.split()[1:]
    if isinstance(value, dict):
        return None
    if value is True or value is None or value == "":
        return extra
    if isinstance(value, (list, tuple)):
        return extra + [str(v) for v in value]
    if isinstance(value, str):
        return extra + value.split()
    return extra + [str(value)]


def _is_number(tok: str) -> bool:
    """Whether *tok* converts with :class:`float`."""
    try:
        float(tok)
        return True
    except (TypeError, ValueError):
        return False


def validate_against_schema(entry: Dict[str, Any], key: str, value: Any):
    """Validate one ``params`` entry against a schema option definition.

    The key and value are split into the whitespace-separated tokens
    that will follow the option name in the input file, and the tokens
    are matched one by one against the option's declared signature.

    Parameters
    ----------
    entry : dict
        Schema entry of the option (``schema[option_name]``; structure
        in :mod:`pyradtran.schema`).
    key : str
        The ``params`` key. Words after the option name count as leading
        tokens (``"wc_modify tau550"`` contributes ``tau550``).
    value : Any
        The ``params`` value. A string is split on whitespace, a list or
        tuple contributes one token per element, any other scalar one
        token. ``True``, ``None`` and ``""`` contribute no token (flag
        usage). A ``dict`` (cloud-profile machinery) is not validated
        here.

    Returns
    -------
    None

    Raises
    ------
    ValidationError
        On a choice mismatch, an out-of-range or non-numeric value,
        a missing required token, a value given to a flag option, or
        extra tokens on a fixed-arity option.

    See Also
    --------
    ParamSpec.validate : The curated check that takes precedence for
        :data:`REGISTRY` keys.
    describe : Shows the signature this function checks against.

    Notes
    -----
    Matching rules, per declared token:

    * ``choice`` tokens compare case-insensitively. If the token also
      accepts a file name (``file_allowed``), any value is accepted.
    * ``float`` / ``int`` tokens must be numeric and inside the
      inclusive ``valid_range``, if the schema has one.
    * ``str`` / ``file`` tokens accept anything; as the *last* declared
      token they swallow all remaining values (this is how
      ``zout 0 5 10`` passes).
    * The variable-length types ``floats`` / ``ints`` / ``files`` /
      ``lines`` swallow all remaining values; for the numeric ones every
      value must be a number.
    * ``optional`` tokens may be absent.

    Options flagged ``unmodeled`` (signature unknown to the schema, e.g.
    ``radiosonde``) are accepted with any arguments. Only the syntax of
    a single line is checked: neither whether the option makes sense in
    combination with others (``parents`` are not enforced) nor whether a
    file exists.

    Examples
    --------
    A hand-written entry in the format of the real schema
    (``schema["wc_modify"]``):

    >>> from pyradtran.params import validate_against_schema
    >>> entry = {
    ...     "name": "wc_modify",
    ...     "tokens": [
    ...         {"kind": "choice", "choices": ["gg", "ssa", "tau", "tau550"]},
    ...         {"kind": "choice", "choices": ["set", "scale"]},
    ...         {"kind": "value", "datatype": "float"},
    ...     ],
    ... }

    Three equivalent, valid spellings (nothing is returned):

    >>> validate_against_schema(entry, "wc_modify", "tau550 set 12")
    >>> validate_against_schema(entry, "wc_modify", ["tau550", "set", 12])
    >>> validate_against_schema(entry, "wc_modify tau550", "set 12")

    Typical mistakes:

    >>> validate_against_schema(entry, "wc_modify", "tau551 set 12")
    Traceback (most recent call last):
        ...
    pyradtran.exceptions.ValidationError: 'wc_modify': 'tau551' is not one of ['gg', 'ssa', 'tau', 'tau550']
    >>> validate_against_schema(entry, "wc_modify", "tau set")
    Traceback (most recent call last):
        ...
    pyradtran.exceptions.ValidationError: 'wc_modify' is missing a required value (expected value token #3)
    >>> validate_against_schema(entry, "wc_modify", "tau set 1 2")
    Traceback (most recent call last):
        ...
    pyradtran.exceptions.ValidationError: 'wc_modify' got extra value(s) ['2']; expected 3 token(s)

    A flag option takes ``True`` but no value:

    >>> flag = {"name": "aerosol_default", "tokens": []}
    >>> validate_against_schema(flag, "aerosol_default", True)
    >>> validate_against_schema(flag, "aerosol_default", 3)
    Traceback (most recent call last):
        ...
    pyradtran.exceptions.ValidationError: 'aerosol_default' takes no value, got 3
    """
    queue = _tokenize_value(key, value)
    if queue is None:
        return  # dict-valued (cloud machinery) — validated downstream
    if entry.get("unmodeled"):
        return  # option name is valid; its signature isn't in the schema
    spec_tokens = entry.get("tokens", [])

    if not spec_tokens and queue:
        raise ValidationError(f"'{entry['name']}' takes no value, got {value!r}")

    for i, st in enumerate(spec_tokens):
        last = i == len(spec_tokens) - 1
        if not queue:
            if st.get("optional"):
                continue
            raise ValidationError(
                f"'{key}' is missing a required value "
                f"(expected {st.get('kind')} token #{i + 1})"
            )
        if st["kind"] == "choice":
            tok = queue.pop(0)
            choices = st.get("choices", [])
            if tok.lower() not in [c.lower() for c in choices]:
                if st.get("file_allowed"):
                    continue
                if st.get("optional"):
                    queue.insert(0, tok)
                    continue
                raise ValidationError(
                    f"'{key}': '{tok}' is not one of {sorted(choices)}"
                )
        else:
            dtype = st.get("datatype", "str")
            if dtype in ("floats", "ints", "lines", "files"):
                rest, queue = queue, []
                if dtype in ("floats", "ints"):
                    bad = [t for t in rest if not _is_number(t)]
                    if bad:
                        raise ValidationError(f"'{key}' expects numbers, got {bad}")
            elif dtype in ("str", "file"):
                if last:
                    queue = []  # greedy: swallows the rest (e.g. zout list)
                else:
                    queue.pop(0)
            else:  # float / int
                tok = queue.pop(0)
                if not _is_number(tok):
                    raise ValidationError(f"'{key}' expects a number, got {tok!r}")
                vr = st.get("valid_range")
                if vr is not None and not (vr[0] <= float(tok) <= vr[1]):
                    raise ValidationError(
                        f"'{key}' value {tok} outside [{vr[0]}, {vr[1]}]"
                    )
    if queue:
        raise ValidationError(
            f"'{key}' got extra value(s) {queue}; "
            f"expected {len(spec_tokens)} token(s)"
        )


def _unknown_option_error(base: str, key: str, schema: Dict[str, Any]) -> str:
    """Build the "not a known uvspec option" message with close matches."""
    known = set(schema) | {k.split()[0] for k in REGISTRY}
    matches = difflib.get_close_matches(base, sorted(known), n=3, cutoff=0.6)
    hint = f"; did you mean {' / '.join(matches)}?" if matches else ""
    return (
        f"'{key}' is not a known uvspec option of the local libRadtran "
        f"install{hint} (wrap the value in Raw(...) to bypass validation)"
    )


def _unknown_config_key_error(key: str, config) -> str:
    """Message for a dotted key matching no ``<section>.<field>``."""
    section_name, _, field_name = key.partition(".")
    section = getattr(config, section_name, None)
    if hasattr(section, "__dict__"):  # known section: suggest its fields
        matches = [
            f"{section_name}.{m}"
            for m in difflib.get_close_matches(field_name, list(vars(section)), n=3)
        ]
    else:  # suggest sections
        matches = [
            f"{m}.{field_name}"
            for m in difflib.get_close_matches(section_name, list(vars(config)), n=3)
        ]
    hint = f"; did you mean {' / '.join(matches)}?" if matches else ""
    return f"'{key}' is not a configuration field{hint}"


def _clean_latex(text: str) -> str:
    """Light cleanup of libRadtran's LaTeX-flavoured documentation."""
    text = re.sub(r"\\code\{([^}]*)\}", r"`\1`", text)
    text = re.sub(r"\\file\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\fcode\{([^}]*)\}", r"\1", text, flags=re.S)
    text = re.sub(r"\\emph\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\(begin|end)\{[^}]*\}", "", text)
    text = re.sub(r"\\item\[([^\]]*)\]", r"- \1:", text)
    text = re.sub(r"\\item\b", "-", text)
    text = re.sub(r"\\parameter\{([^}]*)\}", r"** \1 **", text)
    text = re.sub(r"\\(ifthreedmystic|ifmystic)\b", "", text)
    text = text.replace(r"\_", "_").replace(r"\#", "#").replace(r"\%", "%")
    text = re.sub(r"~?\\(ref|label)\{[^}]*\}", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _signature(entry: Dict[str, Any]) -> str:
    """Render a usage signature such as ``albedo <float in [0.0, 1.0]>``."""
    parts = [entry["name"]]
    for t in entry.get("tokens", []):
        if t["kind"] == "choice":
            body = "|".join(t.get("choices", []))
            if t.get("file_allowed"):
                body += "|<file>"
        else:
            body = t.get("datatype", "str")
            vr = t.get("valid_range")
            if vr is not None:
                body += f" in [{vr[0]}, {vr[1]}]"
        parts.append(f"[{body}]" if t.get("optional") else f"<{body}>")
    if entry.get("unmodeled"):
        parts.append("<...>")
    return " ".join(parts)


def _resolve_schema_arg(schema, config):
    """Return *schema* if given, else the schema of *config* (or defaults).

    Raises :class:`RuntimeError` when no schema is available.
    """
    if schema is not None:
        return schema
    if config is None:
        from .config import load_config

        config = load_config()
    schema = get_schema(config)
    if schema is None:
        raise RuntimeError(
            "No local libRadtran option schema available; check that "
            "paths.libradtran_bin points inside a libRadtran source tree"
        )
    return schema


def describe(name: str, config=None, schema: Optional[Dict[str, Any]] = None) -> str:
    """Return usage, help, and documentation for one uvspec option.

    The text comes from the locally installed libRadtran, so it describes
    the ``uvspec`` binary you run. Use it to find out what to write as
    the value of a ``params`` entry.

    Parameters
    ----------
    name : str
        The uvspec option name (e.g. ``"wc_modify"``), without
        arguments.
    config : SimulationConfig, optional
        Used to locate the local libRadtran install. Defaults to
        :func:`pyradtran.load_config() <pyradtran.config.load_config>`
        (package defaults merged with ``~/.pyradtran/config.yaml``).
    schema : dict, optional
        Pre-loaded schema; overrides *config* (mainly for tests).

    Returns
    -------
    str
        Multi-line text, meant to be printed:

        * line 1: the usage signature. ``<a|b>`` is a required choice,
          ``<float>`` a required value (with its valid range, if any),
          ``[...]`` an optional token, ``|<file>`` marks a choice that
          also accepts a file name, and a trailing ``<...>`` means the
          schema does not describe the arguments;
        * line 2: the option group, plus ``(repeatable)`` if the option
          may be given several times (pass a ``list`` in ``params``);
        * ``requires: ...``: options this one depends on, if any;
        * the one-line help and the long documentation from the
          libRadtran manual (LaTeX markup lightly cleaned up).

    Raises
    ------
    KeyError
        If the option is unknown; the message lists close matches.
    RuntimeError
        If no local libRadtran schema is available (no ``src_py``
        directory next to the configured ``uvspec``).

    See Also
    --------
    search_options : Find option names by keyword.
    pyradtran.interface.PyRadtranAccessor.explain : Preview the input
        file a ``params`` mapping produces.

    Examples
    --------
    The arguments of ``wc_modify`` (libRadtran 2.0.6; output shortened):

    >>> from pyradtran import describe
    >>> print(describe("wc_modify"))  # doctest: +SKIP
    wc_modify <gg|ssa|tau|tau550> <set|scale> <float>
    group: Water and ice clouds   (repeatable)
    requires: wc_file
    <BLANKLINE>
    Modify water cloud optical properties.
    <BLANKLINE>
        wc_modify variable scale/set value
    <BLANKLINE>
        `variable` can be one of the following parameter:
    ...
        ** tau550 **
        Set the water cloud optical thickness at 550nm.
        The optical thickness defined here
        is the integral from the surface at the user-defined `altitude`
        to TOA (top of atmosphere). Other wavelengths are scaled accordingly.
        Note that this option requires for technical reasons that the
        wavelength interval defined by `wavelength` does contain 550nm.
    ...

    A cloud optical thickness of 12 at 550 nm is therefore
    ``params={"wc_modify": "tau550 set 12"}`` or, because the option is
    repeatable, a list such as
    ``{"wc_modify": ["tau550 set 12", "gg set 0.85"]}``.

    A simple numeric option shows its valid range:

    >>> print(describe("albedo"))  # doctest: +SKIP
    albedo <float in [0.0, 1.0]>
    group: Surface
    <BLANKLINE>
    Lambertian surface albedo
    <BLANKLINE>
    The Lambertian surface albedo
    <BLANKLINE>
            albedo value
    <BLANKLINE>
        where `value` is a number between 0.0 and 1.0,
        constant for all wavelengths. For wavelength dependent surface albedo
        use `albedo_file`. The default albedo is 0.0.

    A misspelt name suggests the nearest options:

    >>> describe("wc_modfy")  # doctest: +SKIP
    Traceback (most recent call last):
        ...
    KeyError: "Unknown uvspec option 'wc_modfy'; close matches: wc_modify, ic_modify, mol_modify"

    On a hand-written schema (runs without libRadtran):

    >>> schema = {
    ...     "wc_modify": {
    ...         "name": "wc_modify",
    ...         "group": "Water and ice clouds",
    ...         "help": "Modify water cloud optical properties.",
    ...         "non_unique": True,
    ...         "parents": ["wc_file"],
    ...         "tokens": [
    ...             {"kind": "choice", "choices": ["gg", "ssa", "tau", "tau550"]},
    ...             {"kind": "choice", "choices": ["set", "scale"]},
    ...             {"kind": "value", "datatype": "float"},
    ...         ],
    ...     }
    ... }
    >>> print(describe("wc_modify", schema=schema))
    wc_modify <gg|ssa|tau|tau550> <set|scale> <float>
    group: Water and ice clouds   (repeatable)
    requires: wc_file
    <BLANKLINE>
    Modify water cloud optical properties.
    """
    schema = _resolve_schema_arg(schema, config)
    entry = schema.get(name)
    if entry is None:
        matches = difflib.get_close_matches(name, sorted(schema), n=3)
        hint = f"; close matches: {', '.join(matches)}" if matches else ""
        raise KeyError(f"Unknown uvspec option '{name}'{hint}")

    lines = [
        _signature(entry),
        f"group: {entry.get('group', '?')}"
        + ("   (repeatable)" if entry.get("non_unique") else ""),
    ]
    if entry.get("parents"):
        lines.append(f"requires: {', '.join(entry['parents'])}")
    help_text = (entry.get("help") or "").strip()
    doc_text = _clean_latex(entry.get("doc") or "")
    if help_text and not doc_text.startswith(help_text[:40]):
        lines.append("")
        lines.append(help_text)
    if doc_text:
        lines.append("")
        lines.append(doc_text)
    return "\n".join(lines)


def search_options(
    text: str, config=None, schema: Optional[Dict[str, Any]] = None
) -> list:
    """Return uvspec option names whose name, group, or help match *text*.

    A plain, case-insensitive substring search over each option's name,
    group, one-line help and long documentation in the local libRadtran
    option schema. Follow up with :func:`describe` on a hit.

    Parameters
    ----------
    text : str
        Search text, e.g. ``"ozone"`` or ``"brdf"``. No wildcards or
        regular expressions.
    config : SimulationConfig, optional
        Used to locate the local libRadtran install. Defaults to
        :func:`pyradtran.load_config() <pyradtran.config.load_config>`.
    schema : dict, optional
        Pre-loaded schema; overrides *config* (mainly for tests).

    Returns
    -------
    list of str
        Matching option names in alphabetical order. Empty when nothing
        matches.

    Raises
    ------
    RuntimeError
        If no local libRadtran schema is available (no ``src_py``
        directory next to the configured ``uvspec``), as for
        :func:`describe`.

    See Also
    --------
    describe : Usage and documentation of one option.

    Notes
    -----
    The long documentation is searched too, so an option is also returned
    when its text only mentions the term in passing. Short search terms
    can return many hits.

    Examples
    --------
    With libRadtran 2.0.6:

    >>> from pyradtran import search_options
    >>> search_options("brdf_rpv")  # doctest: +SKIP
    ['albedo_library', 'brdf_rpv', 'brdf_rpv_file', 'brdf_rpv_library', 'brdf_rpv_type', 'surface_type_map']
    >>> search_options("ozone")  # doctest: +SKIP
    ['ECMWF_ozone_climatology', 'atmosphere_file', 'crs_model', 'mol_file', 'radiosonde']
    >>> search_options("visibility")  # doctest: +SKIP
    ['aerosol_default', 'aerosol_visibility']
    >>> len(search_options("cloud"))  # doctest: +SKIP
    50

    On a hand-written schema (runs without libRadtran):

    >>> schema = {
    ...     "albedo": {"name": "albedo", "group": "Surface", "help": "Lambertian surface albedo"},
    ...     "wc_modify": {"name": "wc_modify", "group": "Water and ice clouds"},
    ... }
    >>> search_options("CLOUD", schema=schema)
    ['wc_modify']
    >>> search_options("a", schema=schema)
    ['albedo', 'wc_modify']
    >>> search_options("aerosol", schema=schema)
    []
    """
    schema = _resolve_schema_arg(schema, config)
    needle = text.lower()
    return [
        name
        for name, entry in sorted(schema.items())
        if needle
        in " ".join(
            [name, entry.get("group", ""), entry.get("help", ""), entry.get("doc", "")]
        ).lower()
    ]


class ParamResolver:
    """Resolve a user ``params`` mapping into validated per-point values.

    The engine behind the ``params`` argument of
    :meth:`~pyradtran.interface.PyRadtranAccessor.run` and
    :meth:`~pyradtran.interface.PyRadtranAccessor.explain`. It

    * applies and consumes dotted config overrides
      (``"simulation_defaults.albedo_value"``) on *config*; they never
      reach the uvspec input file directly;
    * splits the remaining entries into literals (validated in the
      constructor) and :class:`Var` references (validated per point, in
      :meth:`resolve_point`);
    * tags every value with its provenance (``PROV_*`` constants).

    Parameters
    ----------
    config : SimulationConfig
        Configuration of the run. **Mutated in place** by dotted keys.
    params : dict, optional
        The user mapping; see the module docstring of
        :mod:`pyradtran.params` for the accepted keys and values.
    schema : {"auto", None} or dict, default ``"auto"``
        libRadtran option schema used to validate keys outside the
        curated :data:`REGISTRY`. ``"auto"`` discovers it from *config*
        (:func:`get_schema`); ``None`` disables schema validation
        (registry only; unknown keys pass through as ``unvalidated``);
        a ``dict`` is used as given (tests).

    Attributes
    ----------
    config : SimulationConfig
        The (possibly modified) configuration.
    schema : dict or None
        The schema in use.
    var_refs : dict
        ``{key: Var}`` for all per-point references.

    Raises
    ------
    ValidationError
        If any literal value fails validation, a key is not a known
        uvspec option, or a dotted key names no ``<section>.<field>`` of
        the configuration. All offending entries are reported in one
        exception.

    See Also
    --------
    Var, Raw : Value markers.
    pyradtran.input_builder.InputFileBuilder.build : Consumes the
        resolved mapping.

    Notes
    -----
    How one literal entry is validated:

    1. ``False`` is kept as is (it tells the builder to omit the
       option); :class:`Raw` values are unwrapped and tagged
       ``unvalidated``.
    2. A key of :data:`REGISTRY` is checked by its
       :meth:`ParamSpec.validate` (lists and tuples are space-joined
       first). If the value is a string and a schema is available, its
       tokens are additionally checked by
       :func:`validate_against_schema`.
    3. Any other key is looked up in the schema by its first word. An
       unknown option raises, with close-match suggestions; a known one
       is checked by :func:`validate_against_schema`. A ``list`` on a
       repeatable (``non_unique``) option is validated element-wise and
       kept as a list of lines; other lists and tuples are space-joined.
    4. Without a schema, non-registry keys are accepted as they are and
       tagged ``unvalidated``.

    A dotted key that names no existing ``<section>.<field>`` of the
    configuration raises, with close-match suggestions. Afterwards
    ``config.simulation_defaults`` is re-validated
    (:meth:`SimulationDefaults.validate
    <pyradtran.config.SimulationDefaults.validate>`), so an invalid
    value set through a dotted key or by attribute assignment is
    reported here too. ``rte_solver`` is checked against the schema;
    without one, against the solver list of libRadtran 2.0.6.

    Examples
    --------
    The examples use ``schema=None`` and a stand-in configuration so
    that they run without libRadtran; normally *config* comes from
    :func:`~pyradtran.config.load_config` and the schema is discovered
    automatically.

    >>> from types import SimpleNamespace
    >>> from pyradtran import ParamResolver, Raw, Var
    >>> from pyradtran.config import SimulationDefaults
    >>> config = SimpleNamespace(simulation_defaults=SimulationDefaults())
    >>> resolver = ParamResolver(
    ...     config,
    ...     {
    ...         "albedo": 0.85,
    ...         "zout": [0, 5],
    ...         "mol_modify O3": False,
    ...         "pseudospherical": Raw(True),
    ...         "sur_temperature": Var("tsurf"),
    ...         "simulation_defaults.rte_solver": "disort",
    ...     },
    ...     schema=None,
    ... )

    Literals are available immediately, with their provenance:

    >>> resolver.static_params()
    {'albedo': (0.85, 'params-literal'), 'zout': ('0 5', 'params-literal'), 'mol_modify O3': (False, 'params-literal'), 'pseudospherical': (True, 'unvalidated')}

    The reference waits for a dataset, and the dotted key went into the
    configuration:

    >>> resolver.var_refs
    {'sur_temperature': Var(name='tsurf')}
    >>> config.simulation_defaults.rte_solver
    'disort'

    All invalid literals are reported together:

    >>> ParamResolver(config, {"albedo": 1.5, "sza": 999.0}, schema=None)
    Traceback (most recent call last):
        ...
    pyradtran.exceptions.ValidationError: Invalid parameter value(s): 'albedo' must be in [0.0, 1.0], got 1.5; 'sza' must be in [0.0, 180.0] deg, got 999.0

    With a schema, non-registry options are validated too. A list on a
    repeatable option becomes one input line per element:

    >>> schema = {
    ...     "wc_modify": {
    ...         "name": "wc_modify",
    ...         "non_unique": True,
    ...         "tokens": [
    ...             {"kind": "choice", "choices": ["gg", "ssa", "tau", "tau550"]},
    ...             {"kind": "choice", "choices": ["set", "scale"]},
    ...             {"kind": "value", "datatype": "float"},
    ...         ],
    ...     },
    ...     "aerosol_default": {"name": "aerosol_default", "tokens": []},
    ... }
    >>> ParamResolver(
    ...     config,
    ...     {"wc_modify": ["tau550 set 12", "gg set 0.85"], "aerosol_default": True},
    ...     schema=schema,
    ... ).static_params()
    {'wc_modify': (['tau550 set 12', 'gg set 0.85'], 'params-literal'), 'aerosol_default': (True, 'params-literal')}
    >>> ParamResolver(config, {"wc_modfy": "tau550 set 12"}, schema=schema)
    Traceback (most recent call last):
        ...
    pyradtran.exceptions.ValidationError: Invalid parameter value(s): 'wc_modfy' is not a known uvspec option of the local libRadtran install; did you mean wc_modify / mol_modify? (wrap the value in Raw(...) to bypass validation)

    A misspelt configuration path is also an error:

    >>> ParamResolver(config, {"simulation_defaults.rte_solvr": "disort"}, schema=None)
    Traceback (most recent call last):
        ...
    pyradtran.exceptions.ValidationError: Invalid parameter value(s): 'simulation_defaults.rte_solvr' is not a configuration field; did you mean simulation_defaults.rte_solver?
    """

    def __init__(
        self,
        config,
        params: Optional[Dict[str, Any]] = None,
        schema: Any = "auto",
    ):
        self.config = config
        # "auto": discover the local libRadtran schema; None: curated
        # registry only; dict: injected schema (tests).
        self.schema = get_schema(config) if schema == "auto" else schema
        remaining = dict(params or {})

        # 1. Dotted config overrides: apply to config, consume.
        errors = []
        for key in [k for k in remaining if "." in k and " " not in k]:
            value = remaining.pop(key)
            section_name, _, field_name = key.partition(".")
            section = getattr(config, section_name, None)
            if section is not None and hasattr(section, field_name):
                setattr(section, field_name, value)
                logger.info(f"Overriding config: {key} = {value}")
                if section_name == "paths":
                    # str -> Path, short names, validation, directories
                    try:
                        section.__post_init__()
                    except ValueError as e:
                        errors.append(str(e))
            else:
                errors.append(_unknown_config_key_error(key, config))
        # Re-check the configuration: dotted keys above and attribute
        # assignment by the user both bypass the dataclass validation.
        defaults = getattr(config, "simulation_defaults", None)
        if hasattr(defaults, "validate"):
            try:
                defaults.validate()
            except (ValueError, TypeError) as e:
                errors.append(str(e))

        # 2. Split literals vs Var refs.
        self._static: Dict[str, Tuple[Any, str]] = {}
        self.var_refs: Dict[str, Var] = {}
        for key, value in remaining.items():
            if value is False:
                # explicit "omit this option": reaches the builder, which
                # removes any config-supplied line and emits nothing
                self._static[key] = (False, PROV_LITERAL)
                continue
            if isinstance(value, Var):
                self.var_refs[key] = value
                continue
            if isinstance(value, Raw):
                self._static[key] = (value.value, PROV_UNVALIDATED)
                continue
            try:
                value, provenance = self._validate_key_value(key, value)
            except ValidationError as e:
                errors.append(str(e))
                continue
            self._static[key] = (value, provenance)

        if errors:
            raise ValidationError("Invalid parameter value(s): " + "; ".join(errors))

    @staticmethod
    def _join(value: Any) -> str:
        """Space-join a sequence into one uvspec argument string."""
        return " ".join(str(v) for v in value)

    def _validate_key_value(self, key: str, value: Any):
        """Validate and normalise one literal entry.

        :data:`REGISTRY` entries win; otherwise the schema, when
        available, validates tokens and rejects unknown option names.
        Without a schema, unknown keys pass through unvalidated. Tuples
        are space-joined; a list on a repeatable (``non_unique``) option
        becomes a list of line strings, any other list is space-joined.
        Returns ``(value, provenance)``.
        """
        spec = REGISTRY.get(key)
        if spec is not None:
            if isinstance(value, (list, tuple)):
                value = self._join(value)
            spec.validate(value)
            if (
                key == "rte_solver"
                and self.schema is None
                and value not in _RTE_SOLVERS_FALLBACK
            ):
                raise ValidationError(
                    f"'rte_solver' must be one of {sorted(_RTE_SOLVERS_FALLBACK)}, "
                    f"got {value!r} (wrap the value in Raw(...) to bypass validation)"
                )
            # Curated validation passes strings through; the schema can
            # still check their tokens (choices, arity, numbers).
            if isinstance(value, str) and self.schema is not None:
                entry = self.schema.get(key.split()[0])
                if entry is not None:
                    validate_against_schema(entry, key, value)
            return value, PROV_LITERAL
        if self.schema is not None:
            base = key.split()[0]
            entry = self.schema.get(base)
            if entry is None:
                raise ValidationError(_unknown_option_error(base, key, self.schema))
            if entry.get("non_unique") and isinstance(value, list):
                for item in value:
                    validate_against_schema(entry, key, item)
                lines = [
                    self._join(i) if isinstance(i, (list, tuple)) else str(i)
                    for i in value
                ]
                return lines, PROV_LITERAL
            validate_against_schema(entry, key, value)
            if isinstance(value, (list, tuple)):
                value = self._join(value)
            return value, PROV_LITERAL
        if isinstance(value, (list, tuple)):
            value = self._join(value)
        return value, PROV_UNVALIDATED

    def static_params(self) -> Dict[str, Tuple[Any, str]]:
        """Return ``key -> (value, provenance)`` for point-independent values.

        Returns
        -------
        dict
            A copy of the validated literal entries: plain values tagged
            ``params-literal``, :class:`Raw` values (and, without a
            schema, non-registry keys) tagged ``unvalidated``, and
            ``False`` for options to omit. :class:`Var` references are
            not included; see :meth:`resolve_point`.

        Examples
        --------
        >>> from types import SimpleNamespace
        >>> from pyradtran import ParamResolver, Var
        >>> resolver = ParamResolver(
        ...     SimpleNamespace(),
        ...     {"albedo": 0.85, "wavelength": (400, 700), "sza": Var("sza")},
        ...     schema=None,
        ... )
        >>> resolver.static_params()
        {'albedo': (0.85, 'params-literal'), 'wavelength': ('400 700', 'params-literal')}
        """
        return dict(self._static)

    def validate_var_targets(self, ds) -> None:
        """Check :class:`Var` targets and ambiguous strings against *ds*.

        Called once per run, before any point is resolved, so that a
        misspelt variable name fails before any simulation starts.

        Parameters
        ----------
        ds : xarray.Dataset
            The input dataset of the run.

        Raises
        ------
        ValidationError
            If a ``Var`` names a variable missing from *ds*, or if a bare
            string literal matches a dataset variable name (ambiguous;
            the user almost certainly meant ``Var``). All problems are
            reported in one exception.

        Examples
        --------
        >>> import xarray as xr
        >>> from types import SimpleNamespace
        >>> from pyradtran import ParamResolver, Var
        >>> ds = xr.Dataset({"tsurf": ("time", [271.3, 268.9])})
        >>> config = SimpleNamespace()
        >>> ParamResolver(
        ...     config, {"sur_temperature": Var("tsurf")}, schema=None
        ... ).validate_var_targets(ds)

        A forgotten ``Var``; the string would otherwise be written to
        the input file literally:

        >>> ParamResolver(
        ...     config, {"sur_temperature": "tsurf"}, schema=None
        ... ).validate_var_targets(ds)
        Traceback (most recent call last):
            ...
        pyradtran.exceptions.ValidationError: 'sur_temperature' has string value 'tsurf' which matches a dataset variable name; use Var('tsurf') for a per-point reference or change the literal
        """
        errors = []
        for key, ref in self.var_refs.items():
            if ref.name not in ds:
                errors.append(
                    f"'{key}' references dataset variable '{ref.name}' "
                    f"which is not in the dataset"
                )
        for key, (value, _prov) in self._static.items():
            if isinstance(value, str) and value in ds:
                errors.append(
                    f"'{key}' has string value '{value}' which matches a "
                    f"dataset variable name; use Var('{value}') for a "
                    f"per-point reference or change the literal"
                )
        if errors:
            raise ValidationError("; ".join(errors))

    def resolve_point(self, point_ds):
        """Resolve all parameters for one stacked point.

        Parameters
        ----------
        point_ds : xarray.Dataset
            A single point (0-d variables), as produced by
            ``stacked_ds.isel({sample_dim: i})``.

        Returns
        -------
        resolved : dict
            ``key -> (value, provenance)`` merged from literals and
            per-point references. Values from a :class:`Var` are tagged
            ``dataset-var``.
        skipped : list of str
            Keys omitted for this point because the dataset value was NaN.
            They are absent from *resolved*, so the configuration's own
            line for that option (if any) stays in effect.

        Raises
        ------
        ValidationError
            If a referenced variable is missing from *point_ds*, or a
            per-point value fails the same validation a literal would
            get. All problems of the point are reported together.

        Examples
        --------
        >>> import numpy as np
        >>> import xarray as xr
        >>> from types import SimpleNamespace
        >>> from pyradtran import ParamResolver, Var
        >>> ds = xr.Dataset({"tsurf": ("time", [271.3, np.nan])})
        >>> resolver = ParamResolver(
        ...     SimpleNamespace(),
        ...     {"albedo": 0.85, "sur_temperature": Var("tsurf")},
        ...     schema=None,
        ... )
        >>> resolved, skipped = resolver.resolve_point(ds.isel(time=0))
        >>> resolved
        {'albedo': (0.85, 'params-literal'), 'sur_temperature': (271.3, 'dataset-var')}
        >>> skipped
        []

        At the second point the temperature is NaN and is left out:

        >>> resolver.resolve_point(ds.isel(time=1))
        ({'albedo': (0.85, 'params-literal')}, ['sur_temperature'])
        """
        resolved = dict(self._static)
        skipped = []
        errors = []
        for key, ref in self.var_refs.items():
            if ref.name not in point_ds:
                errors.append(
                    f"'{key}' references dataset variable '{ref.name}' "
                    f"which is missing from the point"
                )
                continue
            value = point_ds[ref.name].values
            if hasattr(value, "item") and getattr(value, "size", 1) == 1:
                value = value.item()
            if isinstance(value, float) and math.isnan(value):
                skipped.append(key)
                continue
            try:
                value, _prov = self._validate_key_value(key, value)
            except ValidationError as e:
                errors.append(str(e))
                continue
            resolved[key] = (value, PROV_DATASET)
        if errors:
            raise ValidationError("Invalid per-point value(s): " + "; ".join(errors))
        return resolved, skipped


__all__ = [
    "Var",
    "Raw",
    "ParamSpec",
    "ParamResolver",
    "REGISTRY",
    "CONFIG_FIELD_MAP",
    "get_schema",
    "validate_against_schema",
    "describe",
    "search_options",
    "PROV_CONFIG",
    "PROV_LITERAL",
    "PROV_DATASET",
    "PROV_UNVALIDATED",
]
