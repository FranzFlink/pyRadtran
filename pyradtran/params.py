"""
Parameter registry and resolution for pyRadtran.

Defines the single source of truth for every uvspec parameter the package
understands (:data:`REGISTRY`), the :class:`Var` marker for per-point
dataset references, and :class:`ParamResolver`, which turns a user
``params`` mapping into validated, provenance-tagged per-point values.

See Also
--------
pyradtran.input_builder : Renders resolved parameters to uvspec input lines.
"""

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

from .exceptions import ValidationError

logger = logging.getLogger(__name__)

#: Provenance tags (exact strings used in explain() output)
PROV_CONFIG = "config"
PROV_LITERAL = "params-literal"
PROV_DATASET = "dataset-var"
PROV_UNVALIDATED = "unvalidated"


@dataclass(frozen=True)
class Var:
    """Marker: resolve this parameter per point from a dataset variable.

    Parameters
    ----------
    name : str
        Name of the variable in the input dataset.
    """

    name: str


@dataclass(frozen=True)
class ParamSpec:
    """Declarative description of one uvspec parameter.

    Parameters
    ----------
    keyword : str
        The uvspec keyword (possibly multi-word, e.g. ``"mol_modify O3"``).
    dtype : type, default ``float``
        Expected scalar type; used for coercion before validation.
    units : str, optional
        Physical units, for documentation and error messages.
    valid_range : tuple of float, optional
        Inclusive ``(lo, hi)`` range for numeric values.
    choices : tuple, optional
        Allowed values for enumerated parameters.
    applicability : {"solar", "thermal", "both"}, default ``"both"``
        Which source modes the parameter applies to.
    formatter : callable, optional
        ``value -> full input line``. Defaults to ``"{keyword} {value}"``.
    doc : str
        One-line description.
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
        """Raise :class:`ValidationError` if *value* is out of spec.

        String values are passed through unvalidated (escape hatch for
        raw uvspec syntax), except for ``choices`` parameters.
        """
        if self.choices is not None:
            if value not in self.choices:
                raise ValidationError(
                    f"'{self.keyword}' must be one of {self.choices}, got {value!r}"
                )
            return
        if isinstance(value, str):
            return
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
        """Render the full uvspec input line for *value*."""
        if self.formatter is not None:
            return self.formatter(value)
        return f"{self.keyword} {value}"


def _fmt_mol_modify(gas: str, unit: str) -> Callable[[Any], str]:
    def fmt(value: Any) -> str:
        return f"mol_modify {gas} {value} {unit}"

    return fmt


#: Registry of known uvspec parameters (single source of truth).
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
        "sza", float, "deg", (0.0, 180.0), applicability="solar",
        doc="Solar zenith angle",
    ),
    "zout": ParamSpec(
        "zout", float, "km", None, doc="Output altitude level(s)"
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
    "rte_solver": ParamSpec(
        "rte_solver",
        str,
        choices=("twostr", "disort", "fdisort1", "fdisort2", "rodents",
                 "sslidar", "montecarlo", "mystic", "sdisort"),
        doc="Radiative transfer equation solver",
    ),
    "mol_abs_param": ParamSpec(
        "mol_abs_param", str, doc="Molecular absorption parameterisation"
    ),
    "umu": ParamSpec(
        "umu", float, None, (-1.0, 1.0), doc="Cosine of viewing zenith angle"
    ),
    "output_user": ParamSpec(
        "output_user", str, doc="Output column specification"
    ),
    "source": ParamSpec(
        "source", str, choices=("solar", "thermal"), doc="Radiation source"
    ),
    "day_of_year": ParamSpec(
        "day_of_year", int, None, (1, 366),
        formatter=lambda v: f"day_of_year {int(v)}",
        doc="Day of year for Sun–Earth distance correction",
    ),
}

#: Registry key -> SimulationDefaults attribute providing the config default.
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


class ParamResolver:
    """Resolve a user ``params`` mapping into validated per-point values.

    Responsibilities:

    * Apply and **consume** dotted config overrides
      (``"simulation_defaults.albedo_value"``) onto *config* — they never
      reach the uvspec input file.
    * Split remaining entries into literals (validated immediately) and
      :class:`Var` references (validated per point).
    * Tag every value with its provenance.

    Parameters
    ----------
    config : SimulationConfig
        Mutated in place by dotted keys.
    params : dict, optional
        Mapping of registry keys / raw uvspec keywords to literal values
        or :class:`Var` references.

    Raises
    ------
    ValidationError
        If any literal value fails its registry validation. All offending
        keys are reported in one exception.
    """

    def __init__(self, config, params: Optional[Dict[str, Any]] = None):
        self.config = config
        remaining = dict(params or {})

        # 1. Dotted config overrides: apply to config, consume (B1 fix).
        for key in [k for k in remaining if "." in k and " " not in k]:
            value = remaining.pop(key)
            section_name, _, field_name = key.partition(".")
            section = getattr(config, section_name, None)
            if section is not None and hasattr(section, field_name):
                setattr(section, field_name, value)
                logger.info(f"Overriding config: {key} = {value}")
            else:
                logger.warning(f"Unknown config parameter: {key}")

        # 2. Split literals vs Var refs.
        self._static: Dict[str, Tuple[Any, str]] = {}
        self.var_refs: Dict[str, Var] = {}
        errors = []
        for key, value in remaining.items():
            if isinstance(value, Var):
                self.var_refs[key] = value
                continue
            spec = REGISTRY.get(key)
            if spec is None:
                self._static[key] = (value, PROV_UNVALIDATED)
                continue
            try:
                spec.validate(value)
            except ValidationError as e:
                errors.append(str(e))
                continue
            self._static[key] = (value, PROV_LITERAL)

        if errors:
            raise ValidationError(
                "Invalid parameter value(s): " + "; ".join(errors)
            )

    def static_params(self) -> Dict[str, Tuple[Any, str]]:
        """Return ``key -> (value, provenance)`` for point-independent values."""
        return dict(self._static)


__all__ = [
    "Var",
    "ParamSpec",
    "ParamResolver",
    "REGISTRY",
    "CONFIG_FIELD_MAP",
    "PROV_CONFIG",
    "PROV_LITERAL",
    "PROV_DATASET",
    "PROV_UNVALIDATED",
]
