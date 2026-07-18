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
class Raw:
    """Marker: pass this value through without any validation.

    Escape hatch for options the local libRadtran schema does not know
    (e.g. a patched uvspec build). The value reaches the input file
    verbatim, tagged with ``unvalidated`` provenance.

    Parameters
    ----------
    value : Any
        The literal value to emit.
    """

    value: Any


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


#: Memo of loaded schemas, keyed by libRadtran root path.
_SCHEMA_MEMO: Dict[str, Optional[Dict[str, Any]]] = {}


def get_schema(config) -> Optional[Dict[str, Any]]:
    """Load (and memoise) the libRadtran option schema for *config*.

    Returns *None* when no local libRadtran source tree is available;
    validation then falls back to the curated registry only.
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
    key_parts = key.split()
    extra = key_parts[1:]
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
    try:
        float(tok)
        return True
    except (TypeError, ValueError):
        return False


def validate_against_schema(entry: Dict[str, Any], key: str, value: Any):
    """Validate one params entry against a schema option definition.

    Raises
    ------
    ValidationError
        On a choice mismatch, an out-of-range or non-numeric value,
        a missing required token, or extra tokens on a fixed-arity
        option.
    """
    tokens = _tokenize_value(key, value)
    if tokens is None:
        return  # dict-valued (cloud machinery) — validated downstream
    if entry.get("unmodeled"):
        return  # option name is valid; its signature isn't in the schema
    queue = list(tokens)
    spec_tokens = entry.get("tokens", [])

    if not spec_tokens and queue:
        raise ValidationError(
            f"'{entry['name']}' takes no value, got {value!r}"
        )

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
                        raise ValidationError(
                            f"'{key}' expects numbers, got {bad}"
                        )
            elif dtype in ("str", "file"):
                if last:
                    queue = []  # greedy: swallows the rest (e.g. zout list)
                else:
                    queue.pop(0)
            else:  # float / int
                tok = queue.pop(0)
                if not _is_number(tok):
                    raise ValidationError(
                        f"'{key}' expects a number, got {tok!r}"
                    )
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
    import difflib

    known = set(schema) | {k.split()[0] for k in REGISTRY}
    matches = difflib.get_close_matches(base, sorted(known), n=3, cutoff=0.6)
    hint = f"; did you mean {' / '.join(matches)}?" if matches else ""
    return (
        f"'{key}' is not a known uvspec option of the local libRadtran "
        f"install{hint} (wrap the value in Raw(...) to bypass validation)"
    )


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
            raise ValidationError(
                "Invalid parameter value(s): " + "; ".join(errors)
            )

    @staticmethod
    def _join(value: Any) -> str:
        return " ".join(str(v) for v in value)

    def _validate_key_value(self, key: str, value: Any):
        """Validate and normalise one literal entry.

        Curated :data:`REGISTRY` entries win; otherwise the libRadtran
        schema (when available) validates tokens and rejects unknown
        option names. Without a schema, unknown keys pass through
        unvalidated (legacy behaviour). Tuples become space-joined
        strings; a list on a repeatable (``non_unique``) option becomes
        a list of line strings (the builder emits one line each), while
        a list on a single-line option is space-joined.

        Returns
        -------
        (value, provenance)

        Raises
        ------
        ValidationError
        """
        spec = REGISTRY.get(key)
        if spec is not None:
            if isinstance(value, (list, tuple)):
                value = self._join(value)
            spec.validate(value)
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
                raise ValidationError(
                    _unknown_option_error(base, key, self.schema)
                )
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
        """Return ``key -> (value, provenance)`` for point-independent values."""
        return dict(self._static)

    def validate_var_targets(self, ds) -> None:
        """Check every :class:`Var` target (and hijack-suspects) against *ds*.

        Raises
        ------
        ValidationError
            If a ``Var`` names a variable missing from *ds*, or if a bare
            string literal matches a dataset variable name (ambiguous —
            the user almost certainly meant ``Var``).
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
            per-point references.
        skipped : list of str
            Keys omitted for this point because the dataset value was NaN.
        """
        import numpy as np

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
            if isinstance(value, float) and np.isnan(value):
                skipped.append(key)
                continue
            try:
                value, _prov = self._validate_key_value(key, value)
            except ValidationError as e:
                errors.append(str(e))
                continue
            resolved[key] = (value, PROV_DATASET)
        if errors:
            raise ValidationError(
                "Invalid per-point value(s): " + "; ".join(errors)
            )
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
    "PROV_CONFIG",
    "PROV_LITERAL",
    "PROV_DATASET",
    "PROV_UNVALIDATED",
]
