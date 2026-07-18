# pyRadtran 0.2 Unified Parameter Handling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace pyRadtran's four inconsistent parameter-passing mechanisms with a single registry-based `params` mapping, fix 14 catalogued bugs, and add `explain()`, failure reporting, instrument channels, and jacobian mode.

**Architecture:** New `pyradtran/params.py` holds a declarative `ParamSpec` registry, a `Var` dataset-reference marker, and a `ParamResolver` that produces one validated `(value, provenance)` dict per simulation point. New `pyradtran/input_builder.py` renders provenance-tagged uvspec input lines (extracted from `Simulation._generate_input_content`). `interface.py` gains a `params` kwarg; all old kwargs become deprecation shims. Channels and jacobian are post-processing layers on the unchanged batch machinery.

**Tech Stack:** Python ≥3.10, dataclasses, numpy, pandas, xarray, pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-07-18-unified-params-design.md`

## Global Constraints

- Existing test suite (`pytest tests/`) must stay green after every task — old kwargs (`albedo_var`, `override_albedo`, `parameter_overrides`, …) keep working via shims that emit `DeprecationWarning`.
- Provenance strings are exactly: `"config"`, `"params-literal"`, `"dataset-var"`, `"unvalidated"`.
- Status codes are exactly: `0` = ok, `1` = uvspec failure, `2` = skipped (NaN inputs).
- `Var` and `convolve_channels` are exported from the top-level `pyradtran` namespace (Task 14).
- Follow existing code style: numpydoc docstrings, `logger = logging.getLogger(__name__)` per module.
- Run tests with `python -m pytest` from the repo root. `pytest.ini` already configures testpaths.
- Commit after every task (steps include the commands). Work happens on branch `five-star-release`.

## File Structure

- Create: `pyradtran/params.py` — `Var`, `ParamSpec`, `REGISTRY`, `CONFIG_FIELD_MAP`, `ParamResolver` (Tasks 1–3)
- Create: `pyradtran/input_builder.py` — `InputLine`, `InputFileBuilder` (Task 4)
- Create: `pyradtran/channels.py` — `convolve_channels`, `brightness_temperature` (Task 11)
- Modify: `pyradtran/core.py` — use builder, `dry_run`, stderr capture, keep files on failure (Tasks 5, 9, 10)
- Modify: `pyradtran/interface.py` — `params` kwarg, `SimPoint`, shims, bug fixes B1/B5/B6/B14, `status` var, `explain`, `jacobian` (Tasks 6, 9, 10, 13)
- Modify: `pyradtran/io.py` — B7, B8, B9, B13 (Tasks 7, 8)
- Modify: `pyradtran/config.py` — B10, B11 (Task 8)
- Modify: `pyradtran/__init__.py` — exports, version bump (Task 14)
- Tests: `tests/test_params.py`, `tests/test_input_builder.py`, `tests/test_explain.py`, `tests/test_failures.py`, `tests/test_channels.py`, `tests/test_jacobian.py`

---

### Task 1: `Var`, `ParamSpec`, and the parameter registry

**Files:**
- Create: `pyradtran/params.py`
- Test: `tests/test_params.py`

**Interfaces:**
- Consumes: `pyradtran.exceptions.ValidationError` (already exists)
- Produces:
  - `Var(name: str)` — frozen dataclass marker for per-point dataset references
  - `ParamSpec(keyword, dtype=float, units=None, valid_range=None, choices=None, applicability="both", formatter=None, doc="")` with methods `format_line(value) -> str` and `validate(value) -> None`
  - `REGISTRY: Dict[str, ParamSpec]` — keys: `albedo`, `sur_temperature`, `sza`, `zout`, `brdf_rpv_type`, `wavelength`, `mol_modify O3`, `mol_modify H2O`, `rte_solver`, `mol_abs_param`, `umu`, `output_user`, `source`, `day_of_year`
  - `CONFIG_FIELD_MAP: Dict[str, str]` — registry key → `SimulationDefaults` attribute name

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_params.py
"""Tests for the parameter registry and resolver (pyradtran/params.py)."""

import pytest

from pyradtran.exceptions import ValidationError
from pyradtran.params import CONFIG_FIELD_MAP, REGISTRY, ParamSpec, Var


class TestVar:
    def test_var_holds_name(self):
        v = Var("sfc_albedo")
        assert v.name == "sfc_albedo"

    def test_var_is_frozen(self):
        v = Var("x")
        with pytest.raises(Exception):
            v.name = "y"

    def test_var_equality(self):
        assert Var("a") == Var("a")
        assert Var("a") != Var("b")


class TestRegistry:
    def test_core_parameters_present(self):
        for key in [
            "albedo",
            "sur_temperature",
            "sza",
            "zout",
            "brdf_rpv_type",
            "wavelength",
            "mol_modify O3",
            "mol_modify H2O",
            "rte_solver",
            "mol_abs_param",
            "umu",
            "output_user",
            "source",
            "day_of_year",
        ]:
            assert key in REGISTRY, f"missing registry entry: {key}"

    def test_albedo_range_validation(self):
        spec = REGISTRY["albedo"]
        spec.validate(0.5)  # no raise
        with pytest.raises(ValidationError):
            spec.validate(1.5)
        with pytest.raises(ValidationError):
            spec.validate(-0.1)

    def test_sur_temperature_positive(self):
        with pytest.raises(ValidationError):
            REGISTRY["sur_temperature"].validate(-5.0)

    def test_source_choices(self):
        REGISTRY["source"].validate("solar")
        with pytest.raises(ValidationError):
            REGISTRY["source"].validate("lunar")

    def test_format_line_default(self):
        assert REGISTRY["albedo"].format_line(0.85) == "albedo 0.85"

    def test_format_line_mol_modify(self):
        assert REGISTRY["mol_modify O3"].format_line(300.0) == "mol_modify O3 300.0 DU"

    def test_format_line_brdf_type_is_int(self):
        assert REGISTRY["brdf_rpv_type"].format_line(7.0) == "brdf_rpv_type 7"

    def test_string_values_bypass_numeric_validation(self):
        # Raw pass-through strings (escape hatch) must not be range-checked
        REGISTRY["zout"].validate("0.0 1.0 120.0")


class TestConfigFieldMap:
    def test_known_mappings(self):
        assert CONFIG_FIELD_MAP["albedo"] == "albedo_value"
        assert CONFIG_FIELD_MAP["sur_temperature"] == "surface_temperature_k"
        assert CONFIG_FIELD_MAP["sza"] == "sza"

    def test_all_map_keys_in_registry(self):
        for key in CONFIG_FIELD_MAP:
            assert key in REGISTRY
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_params.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pyradtran.params'`

- [ ] **Step 3: Write the implementation**

```python
# pyradtran/params.py
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


__all__ = [
    "Var",
    "ParamSpec",
    "REGISTRY",
    "CONFIG_FIELD_MAP",
    "PROV_CONFIG",
    "PROV_LITERAL",
    "PROV_DATASET",
    "PROV_UNVALIDATED",
]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_params.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add pyradtran/params.py tests/test_params.py
git commit -m "feat: add parameter registry, ParamSpec, and Var marker"
```

---

### Task 2: `ParamResolver` — static resolution, dotted keys, validation

**Files:**
- Modify: `pyradtran/params.py` (append)
- Test: `tests/test_params.py` (append)

**Interfaces:**
- Consumes: `REGISTRY`, `Var`, `ValidationError`, `SimulationConfig`
- Produces:
  - `ParamResolver(config, params=None)` — constructor applies and **consumes** dotted keys (`"section.field"`) onto *config* (fixes B1), validates all literals immediately
  - `ParamResolver.static_params() -> Dict[str, Tuple[Any, str]]` — literals only, `key -> (value, provenance)`
  - `ParamResolver.var_refs -> Dict[str, Var]` — the per-point references

- [ ] **Step 1: Write the failing tests** (append to `tests/test_params.py`)

```python
# append to tests/test_params.py
from pyradtran.params import (
    PROV_LITERAL,
    PROV_UNVALIDATED,
    ParamResolver,
)


class TestParamResolverStatic:
    def test_literals_become_static_params(self, minimal_config):
        r = ParamResolver(minimal_config, {"albedo": 0.85, "sza": 60.0})
        static = r.static_params()
        assert static["albedo"] == (0.85, PROV_LITERAL)
        assert static["sza"] == (60.0, PROV_LITERAL)

    def test_unknown_key_is_unvalidated_passthrough(self, minimal_config):
        r = ParamResolver(minimal_config, {"crs_model": "rayleigh Bodhaine"})
        assert r.static_params()["crs_model"] == (
            "rayleigh Bodhaine",
            PROV_UNVALIDATED,
        )

    def test_literal_validation_raises_before_run(self, minimal_config):
        with pytest.raises(ValidationError):
            ParamResolver(minimal_config, {"albedo": 1.5})

    def test_validation_error_lists_all_offenders(self, minimal_config):
        with pytest.raises(ValidationError) as exc:
            ParamResolver(minimal_config, {"albedo": 1.5, "sza": 999.0})
        assert "albedo" in str(exc.value)
        assert "sza" in str(exc.value)

    def test_dotted_keys_applied_to_config_and_consumed(self, minimal_config):
        # B1 regression: dotted keys must never reach uvspec params
        r = ParamResolver(
            minimal_config, {"simulation_defaults.albedo_value": 0.3}
        )
        assert minimal_config.simulation_defaults.albedo_value == 0.3
        assert "simulation_defaults.albedo_value" not in r.static_params()

    def test_unknown_dotted_key_warns_and_is_dropped(self, minimal_config, caplog):
        import logging

        with caplog.at_level(logging.WARNING):
            r = ParamResolver(minimal_config, {"simulation_defaults.nope": 1})
        assert "simulation_defaults.nope" not in r.static_params()
        assert "Unknown config parameter" in caplog.text

    def test_var_refs_separated(self, minimal_config):
        r = ParamResolver(minimal_config, {"albedo": Var("alb_col")})
        assert r.var_refs == {"albedo": Var("alb_col")}
        assert "albedo" not in r.static_params()

    def test_none_params_ok(self, minimal_config):
        r = ParamResolver(minimal_config, None)
        assert r.static_params() == {}
        assert r.var_refs == {}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_params.py::TestParamResolverStatic -v`
Expected: FAIL with `ImportError: cannot import name 'ParamResolver'`

- [ ] **Step 3: Write the implementation** (append to `pyradtran/params.py`, and add `ParamResolver` to `__all__`)

```python
# append to pyradtran/params.py


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
```

Also update `__all__` in `pyradtran/params.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_params.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add pyradtran/params.py tests/test_params.py
git commit -m "feat: add ParamResolver with dotted-key consumption and literal validation"
```

---

### Task 3: `ParamResolver.resolve_point` — `Var` resolution and NaN policy

**Files:**
- Modify: `pyradtran/params.py` (append method)
- Test: `tests/test_params.py` (append)

**Interfaces:**
- Consumes: `ParamResolver` from Task 2; a 0-d-per-variable xarray Dataset (one stacked point)
- Produces: `ParamResolver.resolve_point(point_ds) -> Tuple[Dict[str, Tuple[Any, str]], List[str]]` — merged static + per-point values, plus list of keys skipped because the point value was NaN (fixes B4)
- Also: `ParamResolver.validate_var_targets(ds) -> None` — raises `ValidationError` listing every `Var` whose name is missing from *ds* (called once before batch submission)

- [ ] **Step 1: Write the failing tests** (append to `tests/test_params.py`)

```python
# append to tests/test_params.py
import numpy as np
import xarray as xr

from pyradtran.params import PROV_DATASET


def _point_ds(**values):
    """One stacked point: each kwarg becomes a 0-d variable."""
    return xr.Dataset({k: ((), v) for k, v in values.items()})


class TestResolvePoint:
    def test_var_resolved_from_point(self, minimal_config):
        r = ParamResolver(minimal_config, {"albedo": Var("alb")})
        resolved, skipped = r.resolve_point(_point_ds(alb=0.7))
        assert resolved["albedo"] == (0.7, PROV_DATASET)
        assert skipped == []

    def test_nan_skips_parameter_and_records(self, minimal_config):
        # B4 regression: NaN must never reach the input file
        r = ParamResolver(minimal_config, {"albedo": Var("alb")})
        resolved, skipped = r.resolve_point(_point_ds(alb=np.nan))
        assert "albedo" not in resolved
        assert skipped == ["albedo"]

    def test_static_and_var_merge(self, minimal_config):
        r = ParamResolver(
            minimal_config, {"sza": 45.0, "albedo": Var("alb")}
        )
        resolved, _ = r.resolve_point(_point_ds(alb=0.2))
        assert resolved["sza"] == (45.0, PROV_LITERAL)
        assert resolved["albedo"] == (0.2, PROV_DATASET)

    def test_resolved_var_value_is_validated(self, minimal_config):
        r = ParamResolver(minimal_config, {"albedo": Var("alb")})
        with pytest.raises(ValidationError):
            r.resolve_point(_point_ds(alb=3.0))

    def test_validate_var_targets_missing(self, minimal_config):
        r = ParamResolver(minimal_config, {"albedo": Var("nope")})
        ds = xr.Dataset({"alb": (("time",), [0.1])})
        with pytest.raises(ValidationError) as exc:
            r.validate_var_targets(ds)
        assert "nope" in str(exc.value)

    def test_bare_string_matching_dataset_var_raises(self, minimal_config):
        # Replaces the old silent string-hijack magic
        ds = xr.Dataset({"alb": (("time",), [0.1])})
        r = ParamResolver(minimal_config, {"albedo": "alb"})
        with pytest.raises(ValidationError) as exc:
            r.validate_var_targets(ds)
        assert "Var(" in str(exc.value)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_params.py::TestResolvePoint -v`
Expected: FAIL with `AttributeError: ... has no attribute 'resolve_point'`

- [ ] **Step 3: Write the implementation** (append methods to `ParamResolver` in `pyradtran/params.py`)

```python
    # append inside class ParamResolver

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
                skipped.append(key)
                continue
            value = point_ds[ref.name].values
            if hasattr(value, "item") and getattr(value, "size", 1) == 1:
                value = value.item()
            if isinstance(value, float) and np.isnan(value):
                skipped.append(key)
                continue
            spec = REGISTRY.get(key)
            if spec is not None:
                try:
                    spec.validate(value)
                except ValidationError as e:
                    errors.append(str(e))
                    continue
            resolved[key] = (value, PROV_DATASET)
        if errors:
            raise ValidationError(
                "Invalid per-point value(s): " + "; ".join(errors)
            )
        return resolved, skipped
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_params.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add pyradtran/params.py tests/test_params.py
git commit -m "feat: per-point Var resolution with NaN skip policy"
```

---

### Task 4: `InputFileBuilder` with provenance tags

**Files:**
- Create: `pyradtran/input_builder.py`
- Test: `tests/test_input_builder.py`

**Interfaces:**
- Consumes: `SimulationConfig`; resolved dict `key -> (value, provenance)` from Task 3; registry `format_line`
- Produces:
  - `InputLine(keyword: str, text: str, provenance: str)` — frozen dataclass
  - `InputFileBuilder(config)` with:
    - `build(dt, latitude, longitude, resolved=None, radiosonde_path=None, era5_atmosphere_file=None) -> List[InputLine]`
    - `render(lines) -> str` — plain input file (ends with newline)
    - `render_annotated(lines) -> str` — aligned `# <provenance>` comments
  - Fixes: B2 (`mol_modify` emission), B3 (`is not None`), multi-word keyword replacement

**Notes for implementer:** This extracts and replaces `Simulation._generate_input_content` (core.py:182–343). The SZA calculation stays in `core.py` for now; the builder takes an optional precomputed `sza` via *resolved* — when the config source is solar and no `sza` is resolved, the builder calls the callback `sza_calculator(dt, latitude, longitude)` passed at construction (defaulting to `None` meaning "leave to caller"). To keep this task self-contained, move `_calculate_solar_zenith_angle` into `input_builder.py` as module function `calculate_solar_zenith_angle(dt, latitude, longitude)`; `core.py` re-imports it in Task 5.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_input_builder.py
"""Tests for pyradtran/input_builder.py."""

from datetime import datetime

import pytest

from pyradtran.input_builder import (
    InputFileBuilder,
    InputLine,
    calculate_solar_zenith_angle,
)
from pyradtran.params import PROV_CONFIG, PROV_DATASET, PROV_LITERAL

DT = datetime(2022, 7, 1, 12, 0)


def build_text(config, resolved=None, **kwargs):
    b = InputFileBuilder(config)
    lines = b.build(DT, 78.0, 15.0, resolved=resolved, **kwargs)
    return b.render(lines), lines


class TestBuilderBasics:
    def test_config_albedo_emitted_with_config_provenance(self, minimal_config):
        text, lines = build_text(minimal_config)
        assert "albedo 0.1" in text
        alb = [ln for ln in lines if ln.keyword == "albedo"][0]
        assert alb.provenance == PROV_CONFIG

    def test_render_ends_with_newline_and_quiet(self, minimal_config):
        text, _ = build_text(minimal_config)
        assert text.endswith("\n")
        assert "quiet" in text.splitlines()

    def test_mol_modify_emitted_from_config(self, minimal_config):
        # B2 regression: ozone_du and h2o_mm must reach the input file
        minimal_config.simulation_defaults.ozone_du = 300.0
        minimal_config.simulation_defaults.h2o_mm = 2.0
        text, _ = build_text(minimal_config)
        assert "mol_modify O3 300.0 DU" in text
        assert "mol_modify H2O 2.0 MM" in text

    def test_mol_modify_not_emitted_when_none(self, minimal_config):
        minimal_config.simulation_defaults.ozone_du = None
        minimal_config.simulation_defaults.h2o_mm = None
        text, _ = build_text(minimal_config)
        assert "mol_modify" not in text

    def test_thermal_surface_temperature_zero_not_swallowed(self, minimal_config):
        # B3 regression: `or` swallowed falsy values
        minimal_config.simulation_defaults.source = "thermal"
        minimal_config.simulation_defaults.surface_temperature_k = 271.2
        text, _ = build_text(
            minimal_config, resolved={"sur_temperature": (0.0, PROV_DATASET)}
        )
        assert "sur_temperature 0.0" in text
        assert "sur_temperature 271.2" not in text


class TestOverrideReplacement:
    def test_resolved_replaces_config_line(self, minimal_config):
        text, lines = build_text(
            minimal_config, resolved={"albedo": (0.85, PROV_LITERAL)}
        )
        assert "albedo 0.85" in text
        assert "albedo 0.1" not in text
        alb = [ln for ln in lines if ln.keyword == "albedo"][0]
        assert alb.provenance == PROV_LITERAL

    def test_multiword_keyword_replacement(self, minimal_config):
        minimal_config.simulation_defaults.ozone_du = 300.0
        text, _ = build_text(
            minimal_config, resolved={"mol_modify O3": (350.0, PROV_LITERAL)}
        )
        assert "mol_modify O3 350.0 DU" in text
        assert "mol_modify O3 300.0 DU" not in text

    def test_multiword_replacement_does_not_clobber_sibling(self, minimal_config):
        minimal_config.simulation_defaults.ozone_du = 300.0
        minimal_config.simulation_defaults.h2o_mm = 2.0
        text, _ = build_text(
            minimal_config, resolved={"mol_modify O3": (350.0, PROV_LITERAL)}
        )
        assert "mol_modify H2O 2.0 MM" in text

    def test_brdf_type_disables_albedo(self, minimal_config):
        text, _ = build_text(
            minimal_config, resolved={"brdf_rpv_type": (7, PROV_DATASET)}
        )
        assert "brdf_rpv_library IGBP" in text
        assert "brdf_rpv_type 7" in text
        assert "albedo" not in [ln.split()[0] for ln in text.splitlines()]

    def test_unknown_key_passthrough(self, minimal_config):
        from pyradtran.params import PROV_UNVALIDATED

        text, _ = build_text(
            minimal_config,
            resolved={"crs_model": ("rayleigh Bodhaine", PROV_UNVALIDATED)},
        )
        assert "crs_model rayleigh Bodhaine" in text


class TestAnnotatedRender:
    def test_annotations_present(self, minimal_config):
        b = InputFileBuilder(minimal_config)
        lines = b.build(
            DT, 78.0, 15.0, resolved={"albedo": (0.85, PROV_LITERAL)}
        )
        annotated = b.render_annotated(lines)
        assert "# params-literal" in annotated
        assert "# config" in annotated


class TestSza:
    def test_solar_zenith_angle_reasonable(self):
        # Summer noon at 45N, 0E: sun high, SZA well below 45 deg
        sza = calculate_solar_zenith_angle(datetime(2022, 6, 21, 12, 0), 45.0, 0.0)
        assert 15.0 < sza < 30.0

    def test_resolved_sza_wins_over_calculated(self, minimal_config):
        text, _ = build_text(minimal_config, resolved={"sza": (77.7, PROV_LITERAL)})
        assert "sza 77.7" in text
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_input_builder.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pyradtran.input_builder'`

- [ ] **Step 3: Write the implementation**

```python
# pyradtran/input_builder.py
"""
uvspec input-file generation with per-line provenance.

:class:`InputFileBuilder` turns a :class:`~pyradtran.config.SimulationConfig`
plus a resolved parameter mapping (from
:class:`~pyradtran.params.ParamResolver`) into an ordered list of
:class:`InputLine` objects. Each line knows which layer produced it, which
powers ``ds.pyradtran.explain()``.

See Also
--------
pyradtran.params : Parameter registry and resolution.
pyradtran.core.Simulation : Consumes the rendered input file.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .params import PROV_CONFIG, REGISTRY

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InputLine:
    """One uvspec input line with provenance.

    Parameters
    ----------
    keyword : str
        The (possibly multi-word) uvspec keyword, e.g. ``"mol_modify O3"``.
    text : str
        The complete input line.
    provenance : str
        One of the ``PROV_*`` constants in :mod:`pyradtran.params`.
    """

    keyword: str
    text: str
    provenance: str


def calculate_solar_zenith_angle(
    dt: datetime, latitude: float, longitude: float
) -> float:
    """Approximate solar zenith angle (degrees) from time and location."""
    import numpy as np

    day_of_year = dt.timetuple().tm_yday
    declination = 23.45 * np.sin(np.radians((360 * (284 + day_of_year)) / 365))
    time_decimal = dt.hour + dt.minute / 60.0 + dt.second / 3600.0
    hour_angle = 15 * (time_decimal - 12) + longitude

    lat_rad = np.radians(latitude)
    decl_rad = np.radians(declination)
    hour_rad = np.radians(hour_angle)

    cos_sza = np.sin(lat_rad) * np.sin(decl_rad) + np.cos(lat_rad) * np.cos(
        decl_rad
    ) * np.cos(hour_rad)
    return round(float(np.degrees(np.arccos(np.clip(cos_sza, -1, 1)))), 2)


class InputFileBuilder:
    """Build provenance-tagged uvspec input lines.

    Parameters
    ----------
    config : SimulationConfig
        Merged configuration.
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
        """Build the full ordered line list for one simulation point."""
        resolved = dict(resolved or {})
        sd = self.config.simulation_defaults
        lines: List[InputLine] = []

        def add(keyword: str, text: str, provenance: str = PROV_CONFIG):
            lines.append(InputLine(keyword, text, provenance))

        add("rte_solver", f"rte_solver {sd.rte_solver}")
        if sd.mol_abs_param:
            add("mol_abs_param", f"mol_abs_param {sd.mol_abs_param}")

        add("data_files_path", f"data_files_path {self.config.paths.libradtran_data}")
        add("atmosphere_file", f"atmosphere_file {self.config.paths.atmosphere_profile}")

        # Atmosphere: ERA5 file wins over radiosonde
        if era5_atmosphere_file is not None:
            era5_abs_path = Path(era5_atmosphere_file).resolve()
            h2o_unit = _sniff_h2o_unit(era5_abs_path)
            add("radiosonde", f"radiosonde {era5_abs_path} H2O {h2o_unit}")
        elif radiosonde_path and sd.h2o_source == "radiosonde":
            add("radiosonde", f"radiosonde {radiosonde_path} H2O RH")

        # Molecular columns (B2 fix: emit when configured)
        if sd.ozone_du is not None:
            add("mol_modify O3", REGISTRY["mol_modify O3"].format_line(sd.ozone_du))
        if sd.h2o_mm is not None:
            add("mol_modify H2O", REGISTRY["mol_modify H2O"].format_line(sd.h2o_mm))

        # Source
        if sd.source == "solar":
            if sd.integrate_wavelength:
                add("source", f"source solar {self.config.paths.solar_spectrum} per_nm")
            else:
                add("source", f"source solar {self.config.paths.solar_spectrum}")
            if "sza" not in resolved:
                sza = sd.sza if sd.sza is not None else calculate_solar_zenith_angle(
                    dt, latitude, longitude
                )
                add("sza", f"sza {sza}")
            add(
                "day_of_year",
                REGISTRY["day_of_year"].format_line(dt.timetuple().tm_yday),
            )
        elif sd.source == "thermal":
            add("source", "source thermal")
            # B3 fix: explicit None check, no `or`
            if "sur_temperature" not in resolved and sd.surface_temperature_k is not None:
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
        if "brdf_rpv_type" not in resolved:
            if "albedo" not in resolved and sd.albedo_value is not None:
                add("albedo", f"albedo {sd.albedo_value}")

        # Output columns
        output_columns = " ".join(sd.output_columns)
        if output_columns:
            add("output_user", f"output_user {output_columns}")

        # Output altitudes
        if "zout" not in resolved and sd.output_altitudes_km:
            alt_str = " ".join(f"{alt:.4f}" for alt in sd.output_altitudes_km)
            add("zout", f"zout {alt_str}")

        # Viewing geometry
        if sd.viewing_geometry == "nadir":
            add("umu", "umu 1.0")

        # Static cloud settings from config
        if sd.clouds.enabled:
            self._add_cloud_lines(lines)

        # Apply resolved parameters: replace same-keyword lines, then append
        for key, (value, provenance) in resolved.items():
            spec = REGISTRY.get(key)
            if spec is not None:
                text = spec.format_line(value)
                keyword = spec.keyword
            else:
                text = f"{key} {value}"
                keyword = key
            lines = [ln for ln in lines if ln.keyword != keyword]
            lines.append(InputLine(keyword, text, provenance))
            # brdf implies the IGBP library line
            if keyword == "brdf_rpv_type":
                lines = [ln for ln in lines if ln.keyword not in ("albedo", "brdf_rpv_library")]
                lines.append(
                    InputLine("brdf_rpv_library", "brdf_rpv_library IGBP", provenance)
                )

        lines.append(InputLine("quiet", "quiet", PROV_CONFIG))
        return lines

    def _add_cloud_lines(self, lines: List[InputLine]) -> None:
        """Append config-driven cloud lines (file or parametric)."""
        clouds = self.config.simulation_defaults.clouds
        if clouds.cloud_source == "file":
            if clouds.cloud_type in ["wc", "mixed"] and clouds.wc_file:
                lines.append(
                    InputLine("wc_file", f"wc_file {clouds.wc_file}", PROV_CONFIG)
                )
            if clouds.cloud_type in ["ic", "mixed"] and clouds.ic_file:
                lines.append(
                    InputLine("ic_file", f"ic_file {clouds.ic_file}", PROV_CONFIG)
                )
        elif clouds.cloud_source == "parametric":
            if clouds.cloud_type in ["wc", "mixed"]:
                lines.append(
                    InputLine(
                        "wc_layer",
                        f"wc_layer {clouds.layer_bottom_km} {clouds.layer_top_km} "
                        f"{clouds.water_content_g_m3} {clouds.effective_radius_um}",
                        PROV_CONFIG,
                    )
                )
            if clouds.cloud_type in ["ic", "mixed"]:
                lines.append(
                    InputLine(
                        "ic_layer",
                        f"ic_layer {clouds.layer_bottom_km} {clouds.layer_top_km} "
                        f"{clouds.ice_content_g_m3} {clouds.effective_radius_um}",
                        PROV_CONFIG,
                    )
                )

    @staticmethod
    def render(lines: List[InputLine]) -> str:
        """Render lines to the plain uvspec input file content."""
        return "\n".join(ln.text for ln in lines) + "\n"

    @staticmethod
    def render_annotated(lines: List[InputLine]) -> str:
        """Render lines with aligned ``# <provenance>`` annotations."""
        if not lines:
            return ""
        width = max(len(ln.text) for ln in lines) + 2
        return "\n".join(
            f"{ln.text:<{width}}# {ln.provenance}" for ln in lines
        ) + "\n"


def _sniff_h2o_unit(era5_abs_path: Path) -> str:
    """Read the header of an ERA5 atmosphere file to determine the H2O unit."""
    h2o_unit = "MMR"
    try:
        with open(era5_abs_path, "r") as f:
            second_line = f.readlines()[1].strip()
        if "%" in second_line:
            h2o_unit = "RH"
    except (OSError, IndexError):
        logger.warning(
            f"Could not sniff H2O unit from {era5_abs_path}; assuming MMR"
        )
    return h2o_unit


__all__ = ["InputLine", "InputFileBuilder", "calculate_solar_zenith_angle"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_input_builder.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add pyradtran/input_builder.py tests/test_input_builder.py
git commit -m "feat: InputFileBuilder with provenance tags; fix silent ozone/h2o no-op (B2) and falsy surf-temp (B3)"
```

---

### Task 5: Wire `Simulation` to the builder; deprecate `override_*` kwargs

**Files:**
- Modify: `pyradtran/core.py`
- Test: existing `tests/test_interface.py`, `tests/test_clouds.py`, `tests/test_dynamic_clouds.py` (must stay green); new assertions in `tests/test_input_builder.py`

**Interfaces:**
- Consumes: `InputFileBuilder`, `ParamResolver` provenance constants
- Produces:
  - `Simulation.run_simulation(dt, latitude, longitude, resolved_params=None, era5_atmosphere_file=None, override_albedo=None, override_surface_temperature=None, override_altitude_km=None, override_surface_type=None, parameter_overrides=None) -> Optional[Path]` — new preferred arg `resolved_params: Dict[str, Tuple[Any, str]]`; old kwargs emit `DeprecationWarning` and are merged into `resolved_params`
  - `Simulation.build_input_lines(dt, latitude, longitude, resolved_params=None, era5_atmosphere_file=None) -> List[InputLine]` — shared by run and (later) dry_run
  - `Simulation.last_stderr: Optional[str]` — stderr of the most recent failed uvspec run (used by Task 10)

- [ ] **Step 1: Write the failing test** (append to `tests/test_input_builder.py`)

```python
# append to tests/test_input_builder.py
import warnings


class TestSimulationWiring:
    def test_legacy_override_kwargs_warn_and_apply(self, minimal_config):
        from pyradtran.core import Simulation

        sim = Simulation(minimal_config)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            lines = sim.build_input_lines(
                DT, 78.0, 15.0,
                resolved_params=None,
                override_albedo=0.9,
            )
        assert any(issubclass(x.category, DeprecationWarning) for x in w)
        text = "\n".join(ln.text for ln in lines)
        assert "albedo 0.9" in text

    def test_legacy_parameter_overrides_still_work(self, minimal_config):
        from pyradtran.core import Simulation

        sim = Simulation(minimal_config)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            lines = sim.build_input_lines(
                DT, 78.0, 15.0, parameter_overrides={"number_of_streams": 16}
            )
        text = "\n".join(ln.text for ln in lines)
        assert "number_of_streams 16" in text

    def test_resolved_params_reach_input(self, minimal_config):
        from pyradtran.core import Simulation
        from pyradtran.params import PROV_LITERAL

        sim = Simulation(minimal_config)
        lines = sim.build_input_lines(
            DT, 78.0, 15.0, resolved_params={"albedo": (0.42, PROV_LITERAL)}
        )
        assert "albedo 0.42" in "\n".join(ln.text for ln in lines)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_input_builder.py::TestSimulationWiring -v`
Expected: FAIL with `AttributeError: 'Simulation' object has no attribute 'build_input_lines'`

- [ ] **Step 3: Modify `pyradtran/core.py`**

Replace `_generate_input_content` (lines 182–343), `_add_cloud_settings` (441–461), and `_calculate_solar_zenith_angle` (463–497) with builder delegation. Keep `format_cloud_profile`, `_handle_dynamic_clouds`, `_run_uvspec` (modified in Task 10).

New/changed code in `pyradtran/core.py`:

```python
# new imports at top of core.py
import warnings
from .input_builder import InputFileBuilder, calculate_solar_zenith_angle  # noqa: F401
from .params import PROV_LITERAL, PROV_UNVALIDATED
```

```python
# inside class Simulation

    def __init__(self, config: SimulationConfig):
        """Initialise with a merged :class:`SimulationConfig`."""
        self.config = config
        self.builder = InputFileBuilder(config)
        self.last_stderr: Optional[str] = None
        self.radiosonde_finder = (
            RadiosondeFinder(config.paths.radiosonde_base)
            if config.paths.radiosonde_base
            else None
        )

    @staticmethod
    def _legacy_kwargs_to_resolved(
        resolved_params,
        override_albedo=None,
        override_surface_temperature=None,
        override_altitude_km=None,
        override_surface_type=None,
        parameter_overrides=None,
    ):
        """Merge deprecated kwargs into a resolved-params dict."""
        resolved = dict(resolved_params or {})
        legacy = {
            "albedo": override_albedo,
            "sur_temperature": override_surface_temperature,
            "zout": override_altitude_km,
            "brdf_rpv_type": override_surface_type,
        }
        used = [k for k, v in legacy.items() if v is not None]
        if used or parameter_overrides:
            warnings.warn(
                "override_* kwargs and parameter_overrides are deprecated; "
                "use resolved_params / the params mapping instead",
                DeprecationWarning,
                stacklevel=3,
            )
        import numpy as np

        for key, value in legacy.items():
            if value is None or key in resolved:
                continue
            if isinstance(value, float) and np.isnan(value):
                continue  # B4: NaN never reaches the input file
            resolved[key] = (value, PROV_LITERAL)
        for key, value in (parameter_overrides or {}).items():
            if key not in resolved:
                resolved[key] = (value, PROV_UNVALIDATED)
        return resolved

    def build_input_lines(
        self,
        dt: datetime,
        latitude: float,
        longitude: float,
        resolved_params=None,
        era5_atmosphere_file: Optional[Path] = None,
        override_albedo: Optional[float] = None,
        override_surface_temperature: Optional[float] = None,
        override_altitude_km: Optional[float] = None,
        override_surface_type: Optional[int] = None,
        parameter_overrides: Dict[str, Any] = None,
    ):
        """Build the provenance-tagged input lines for one point."""
        resolved = self._legacy_kwargs_to_resolved(
            resolved_params,
            override_albedo,
            override_surface_temperature,
            override_altitude_km,
            override_surface_type,
            parameter_overrides,
        )
        radiosonde_path = None
        if (
            self.config.simulation_defaults.h2o_source == "radiosonde"
            and self.radiosonde_finder
        ):
            radiosonde_path = self.radiosonde_finder.find_radiosonde_file(
                dt, latitude, longitude
            )
        return self.builder.build(
            dt,
            latitude,
            longitude,
            resolved=resolved,
            radiosonde_path=radiosonde_path,
            era5_atmosphere_file=era5_atmosphere_file,
        )
```

Rewrite `run_simulation` body to use the new path. The dynamic-cloud handling now operates on the resolved dict (`wc_file` / `ic_file` values that are dicts/lists are written to temp files, and the resolved value becomes `("1D <path>", same_provenance)`):

```python
    def run_simulation(
        self,
        dt: datetime,
        latitude: float,
        longitude: float,
        resolved_params=None,
        era5_atmosphere_file: Optional[Path] = None,
        override_albedo: Optional[float] = None,
        override_surface_temperature: Optional[float] = None,
        override_altitude_km: Optional[float] = None,
        override_surface_type: Optional[int] = None,
        parameter_overrides: Dict[str, Any] = None,
    ) -> Optional[Path]:
        """Run a single ``uvspec`` simulation.

        Parameters
        ----------
        dt : datetime
            Simulation date/time (UTC).
        latitude, longitude : float
            Location in degrees.
        resolved_params : dict, optional
            ``key -> (value, provenance)`` mapping from
            :meth:`pyradtran.params.ParamResolver.resolve_point`.
        era5_atmosphere_file : pathlib.Path, optional
            Custom ERA5 atmosphere file (radiosonde format).
        override_albedo, override_surface_temperature : float, optional
            Deprecated — use *resolved_params*.
        override_altitude_km : float, optional
            Deprecated — use *resolved_params*.
        override_surface_type : int, optional
            Deprecated — use *resolved_params*.
        parameter_overrides : dict, optional
            Deprecated — use *resolved_params*.

        Returns
        -------
        pathlib.Path or None
            Path to the ``uvspec`` output file, or *None* on failure
            (:attr:`last_stderr` then holds the captured stderr).
        """
        self.last_stderr = None
        temp_cloud_files = []
        try:
            resolved = self._legacy_kwargs_to_resolved(
                resolved_params,
                override_albedo,
                override_surface_temperature,
                override_altitude_km,
                override_surface_type,
                parameter_overrides,
            )

            # Dynamic clouds: dict/list values for wc_file / ic_file
            cloud_overrides = {
                k: v for k, (v, _p) in resolved.items()
                if k in ("wc_file", "ic_file") and isinstance(v, (dict, list))
            }
            if cloud_overrides:
                new_overrides, files_to_clean = self._handle_dynamic_clouds(
                    cloud_overrides
                )
                temp_cloud_files.extend(files_to_clean)
                for k, v in new_overrides.items():
                    resolved[k] = (v, resolved[k][1])

            lines = self.build_input_lines(
                dt, latitude, longitude,
                resolved_params=resolved,
                era5_atmosphere_file=era5_atmosphere_file,
            )
            input_content = self.builder.render(lines)

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".inp", delete=False,
                dir=self.config.paths.working_dir,
            ) as inp_file:
                inp_file.write(input_content)
                input_path = Path(inp_file.name)

            output_path = input_path.with_suffix(".out")
            success = self._run_uvspec(input_path, output_path)

            if success and output_path.exists():
                logger.debug(f"Simulation completed successfully: {output_path}")
                if self.config.execution.cleanup_temp_files:
                    input_path.unlink(missing_ok=True)
                return output_path
            # Failure: keep input and output for post-mortem (B12)
            logger.error(
                f"Simulation failed; input kept at {input_path}"
            )
            return None

        except Exception as e:
            logger.error(f"Simulation failed: {str(e)}")
            raise UvspecExecutionError(f"Simulation failed: {str(e)}")

        finally:
            for p in temp_cloud_files:
                try:
                    if os.path.exists(p):
                        os.unlink(p)
                except Exception as e:
                    logger.warning(f"Failed to cleanup temp file {p}: {e}")
```

Note: `_handle_dynamic_clouds` currently checks `isinstance(overrides["wc_file"], (dict, list))` internally — it can be left unchanged; we pass it a plain dict of raw values.

Delete `_generate_input_content`, `_add_cloud_settings`, and `_calculate_solar_zenith_angle` from `core.py` (all replaced). Keep a module-level alias for backwards compatibility:

```python
# near bottom of core.py, before __all__
# Backwards-compatible alias (was a Simulation method)
Simulation._calculate_solar_zenith_angle = staticmethod(
    lambda dt, latitude, longitude: calculate_solar_zenith_angle(
        dt, latitude, longitude
    )
)
```

- [ ] **Step 4: Run the new and existing tests**

Run: `python -m pytest tests/test_input_builder.py tests/test_interface.py tests/test_clouds.py tests/test_dynamic_clouds.py -v`
Expected: PASS (existing suites go through the deprecation shims)

Run: `python -m pytest tests/ -x -q`
Expected: full suite PASS (integration-marked tests may skip without libRadtran)

- [ ] **Step 5: Commit**

```bash
git add pyradtran/core.py tests/test_input_builder.py
git commit -m "refactor: Simulation delegates input generation to InputFileBuilder; deprecate override_* kwargs"
```

---

### Task 6: `params` in the batch interface; `SimPoint`; kill the 10-tuple

**Files:**
- Modify: `pyradtran/interface.py`
- Test: `tests/test_params.py` (append batch-level tests); existing `tests/test_interface.py`, `tests/test_variational.py` must stay green

**Interfaces:**
- Consumes: `ParamResolver`, `Var`, `Simulation.run_simulation(resolved_params=...)`
- Produces:
  - `execute_simulation_batch(config, input_ds, params=None, time_var="time", lat_var="latitude", lon_var="longitude", ..., <all old kwargs kept>) -> List[Optional[ParsedOutput]]`
  - `SimPoint` dataclass: `index: int`, `time: datetime`, `latitude: float`, `longitude: float`, `resolved: dict`, `skipped: List[str]`, `era5_file: Optional[Path]`, `point_id: str`
  - `PyRadtranAccessor.run(..., params=None, ...)` — same shim behaviour
  - Old kwargs (`albedo_var`, `surface_temperature_var`, `surface_type_var`, `altitude_var`, `parameter_overrides`) translate to `params` entries with `DeprecationWarning`
  - Fixes: B1 (accessor+function), B5 (scalar altitude coord), B6 (monologue comments), B14 (callback `(completed, total)`), string-hijack removal

- [ ] **Step 1: Write the failing tests** (append to `tests/test_params.py`)

```python
# append to tests/test_params.py
import warnings
from unittest.mock import patch

from pyradtran.interface import _translate_legacy_kwargs


class TestLegacyKwargTranslation:
    def test_var_kwargs_become_params(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            params = _translate_legacy_kwargs(
                params=None,
                albedo_var="alb",
                surface_temperature_var="skin_t",
                surface_type_var="igbp",
                altitude_var="flight_alt",
                parameter_overrides={"number_of_streams": 16},
            )
        assert any(issubclass(x.category, DeprecationWarning) for x in w)
        assert params["albedo"] == Var("alb")
        assert params["sur_temperature"] == Var("skin_t")
        assert params["brdf_rpv_type"] == Var("igbp")
        assert params["zout"] == Var("flight_alt")
        assert params["number_of_streams"] == 16

    def test_explicit_params_win_over_legacy(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            params = _translate_legacy_kwargs(
                params={"albedo": 0.5}, albedo_var="alb",
                surface_temperature_var=None, surface_type_var=None,
                altitude_var=None, parameter_overrides=None,
            )
        assert params["albedo"] == 0.5

    def test_no_legacy_no_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            params = _translate_legacy_kwargs(
                params={"albedo": 0.5}, albedo_var=None,
                surface_temperature_var=None, surface_type_var=None,
                altitude_var=None, parameter_overrides=None,
            )
        assert not any(issubclass(x.category, DeprecationWarning) for x in w)
        assert params == {"albedo": 0.5}


class TestBatchParams(object):
    def test_batch_resolves_var_per_point(
        self, minimal_config, simple_input_dataset
    ):
        """Batch must hand per-point resolved params to the worker."""
        captured = []

        def fake_worker(config, point, _resolver_unused=None):
            captured.append(point)
            return None

        from pyradtran import interface

        ds = simple_input_dataset.copy()
        ds["alb"] = (["time"], [0.1, 0.2, 0.3])

        with patch.object(
            interface, "_run_single_simulation_unified", side_effect=fake_worker
        ):
            with pytest.raises(Exception):
                # All workers return None -> batch raises "all failed"
                interface.execute_simulation_batch(
                    config=minimal_config,
                    input_ds=ds,
                    params={"albedo": Var("alb")},
                    show_progress=False,
                )
        albs = sorted(p.resolved["albedo"][0] for p in captured)
        assert albs == [0.1, 0.2, 0.3]

    def test_missing_var_target_raises_before_submission(
        self, minimal_config, simple_input_dataset
    ):
        from pyradtran.interface import execute_simulation_batch

        with pytest.raises(ValidationError):
            execute_simulation_batch(
                config=minimal_config,
                input_ds=simple_input_dataset,
                params={"albedo": Var("does_not_exist")},
                show_progress=False,
            )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_params.py::TestLegacyKwargTranslation tests/test_params.py::TestBatchParams -v`
Expected: FAIL with `ImportError: cannot import name '_translate_legacy_kwargs'`

- [ ] **Step 3: Modify `pyradtran/interface.py`**

Add imports and the translation helper:

```python
# new imports in interface.py
import warnings
from dataclasses import dataclass, field

from .exceptions import PyRadtranError, ValidationError
from .params import ParamResolver, Var
```

```python
@dataclass
class SimPoint:
    """One flattened simulation point, fully resolved."""

    index: int
    time: datetime
    latitude: float
    longitude: float
    resolved: Dict[str, Any]
    skipped: List[str] = field(default_factory=list)
    era5_file: Optional[Path] = None
    point_id: str = ""


def _translate_legacy_kwargs(
    params,
    albedo_var,
    surface_temperature_var,
    surface_type_var,
    altitude_var,
    parameter_overrides,
):
    """Map deprecated kwargs onto the unified ``params`` mapping.

    Explicit ``params`` entries always win. Emits a single
    DeprecationWarning if any legacy kwarg is used.
    """
    params = dict(params or {})
    legacy_vars = {
        "albedo": albedo_var,
        "sur_temperature": surface_temperature_var,
        "brdf_rpv_type": surface_type_var,
        "zout": altitude_var,
    }
    used_legacy = any(v is not None for v in legacy_vars.values()) or bool(
        parameter_overrides
    )
    if used_legacy:
        warnings.warn(
            "albedo_var/surface_temperature_var/surface_type_var/altitude_var "
            "and parameter_overrides are deprecated; use "
            "params={'albedo': Var('...'), ...} instead",
            DeprecationWarning,
            stacklevel=3,
        )
    for key, var_name in legacy_vars.items():
        if var_name is not None and key not in params:
            params[key] = Var(var_name)
    for key, value in (parameter_overrides or {}).items():
        if key not in params:
            params[key] = value
    return params
```

Rewrite the point-preparation section of `execute_simulation_batch` (interface.py:149–551). The new signature adds `params: Optional[Dict[str, Any]] = None` after `input_ds`; all existing kwargs stay. Key replacement of lines 318–506 (point prep + submission), also deleting the B6 monologue block entirely:

```python
    # inside execute_simulation_batch, after era5 file generation
    params = _translate_legacy_kwargs(
        params, albedo_var, surface_temperature_var, surface_type_var,
        altitude_var, parameter_overrides,
    )
    resolver = ParamResolver(config, params)
    resolver.validate_var_targets(input_ds)

    points: List[SimPoint] = []
    for i in range(num_points):
        point_ds = stacked_ds.isel({sample_dim: i})
        t = get_val(point_ds, time_var)
        lat = get_val(point_ds, lat_var)
        lon = get_val(point_ds, lon_var)

        resolved, skipped = resolver.resolve_point(point_ds)

        # Cloud automation: build dict-valued wc_file/ic_file entries
        try:
            if cloud_wc_var or cloud_ic_var:
                cth = get_val(point_ds, cloud_top_var)
                cbh = get_val(point_ds, cloud_bottom_var)
                if (
                    cth is not None and cbh is not None
                    and not np.isnan(cth) and not np.isnan(cbh)
                ):
                    z_layer = [max(cth, cbh), min(cth, cbh)]
                    if cloud_wc_var:
                        lwc = get_val(point_ds, cloud_wc_var)
                        reff = (
                            get_val(point_ds, cloud_reff_var)
                            if cloud_reff_var else 10.0
                        )
                        if lwc is not None and not np.isnan(lwc):
                            r_val = (
                                reff
                                if (reff is not None and not np.isnan(reff))
                                else 10.0
                            )
                            resolved["wc_file"] = (
                                {
                                    "z": z_layer,
                                    "lwc": [float(lwc), float(lwc)],
                                    "reff": [float(r_val), float(r_val)],
                                },
                                "dataset-var",
                            )
                    if cloud_ic_var:
                        iwc = get_val(point_ds, cloud_ic_var)
                        r_key = (
                            cloud_ic_reff_var if cloud_ic_reff_var
                            else cloud_reff_var
                        )
                        reff_ice = get_val(point_ds, r_key) if r_key else 20.0
                        if iwc is not None and not np.isnan(iwc):
                            r_val = (
                                reff_ice
                                if (reff_ice is not None and not np.isnan(reff_ice))
                                else 20.0
                            )
                            resolved["ic_file"] = (
                                {
                                    "z": z_layer,
                                    "iwc": [float(iwc), float(iwc)],
                                    "reff": [float(r_val), float(r_val)],
                                },
                                "dataset-var",
                            )
        except Exception as e:
            logger.warning(f"Failed to generate cloud parameters for point {i}: {e}")

        dt = pd.to_datetime(t).to_pydatetime()
        era5_key = f"{dt.strftime('%Y%m%d_%H%M%S')}_{lat:.2f}_{lon:.2f}"
        points.append(
            SimPoint(
                index=i,
                time=dt,
                latitude=lat,
                longitude=lon,
                resolved=resolved,
                skipped=skipped,
                era5_file=(
                    era5_atmosphere_files.get(era5_key)
                    if era5_atmosphere_files else None
                ),
                point_id=f"{era5_key}_{i}",
            )
        )
```

Submission loop replacement (delete lines 447–506 including all monologue comments):

```python
    with ProcessPoolExecutor(max_workers=config.execution.max_workers) as executor:
        future_to_idx = {
            executor.submit(_run_single_simulation_unified, config, point): point.index
            for point in points
        }

        completed = 0
        success_count = 0
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            completed += 1
            try:
                result = future.result()
                results[idx] = result
                if result:
                    success_count += 1
                else:
                    logger.warning(
                        f"Simulation {idx + 1}/{num_points} produced no output"
                    )
            except Exception as e:
                logger.error(
                    f"Simulation {idx + 1}/{num_points} failed with error: {str(e)}"
                )
            if pbar:
                pbar.update(1)
                pbar.set_postfix({"Success": success_count, "Total": num_points})
            if progress_callback:
                # B14 fix: report completed count, not success count
                progress_callback(completed, num_points)
```

New worker (replaces `_run_single_simulation_unified`, deleting the time-coercion duplication into a helper):

```python
def _coerce_datetime(time) -> datetime:
    """Convert any supported time representation to datetime."""
    if isinstance(time, datetime):
        return time
    if isinstance(time, np.datetime64) or isinstance(time, (int, np.integer)):
        return pd.to_datetime(time).to_pydatetime()
    if isinstance(time, str):
        return datetime.fromisoformat(time)
    return time


def _run_single_simulation_unified(
    config: SimulationConfig,
    point: SimPoint,
) -> Optional[ParsedOutput]:
    """Execute a single ``uvspec`` run (called by the process pool)."""
    try:
        sim = Simulation(config)
        dt = _coerce_datetime(point.time)

        output_file = sim.run_simulation(
            dt=dt,
            latitude=point.latitude,
            longitude=point.longitude,
            resolved_params=point.resolved,
            era5_atmosphere_file=point.era5_file,
        )

        if output_file and output_file.exists():
            raw_overrides = {k: v for k, (v, _p) in point.resolved.items()}
            parser = OutputParser(config, raw_overrides)
            parsed_output = parser.parse_output_file(output_file)
            parsed_output.metadata.update(
                {
                    "point_id": point.point_id,
                    "time": dt.isoformat(),
                    "latitude": point.latitude,
                    "longitude": point.longitude,
                }
            )
            return parsed_output
        logger.error(f"No output file produced for point {point.point_id}")
        return None
    except Exception as e:
        logger.error(f"Single simulation failed for point {point.point_id}: {str(e)}")
        return None
```

In `PyRadtranAccessor.run`: add `params: Optional[Dict[str, Any]] = None` parameter; delete the `_apply_parameter_overrides` call (B1 — dotted keys are now consumed by `ParamResolver` inside the batch); forward `params=params` plus the legacy kwargs to `execute_simulation_batch`. Fix B5 in the altitude block (interface.py:770):

```python
        if alt_var in self._obj.dims or alt_var in self._obj.coords:
            dataset_altitudes = np.atleast_1d(self._obj[alt_var].values)
            if dataset_altitudes.size > 0:
                logger.info(
                    f"Altitude found as coordinate - using "
                    f"{dataset_altitudes.size} levels for zout: {dataset_altitudes}"
                )
                self._config.simulation_defaults.output_altitudes_km = [
                    float(alt) for alt in dataset_altitudes
                ]
```

In `run_pyradtran_simulation`: replace `_apply_parameter_overrides(config, parameter_overrides)` + forwarding with `params=parameter_overrides` passed to `execute_simulation_batch` (the resolver handles both dotted and raw keys now). Delete `_apply_parameter_overrides` entirely.

Fix B11 (interface.py:301–307): change the misleading comment to:

```python
                # Cache: one atmosphere file per unique (time, lat, lon)
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_params.py tests/test_interface.py tests/test_variational.py -v`
Expected: PASS. `test_variational.py` exercises the old string-hijack path — if it fails on the removed magic, update those tests to use `Var(...)` (that is the intended migration; note it in the commit message).

Run: `python -m pytest tests/ -q`
Expected: full suite PASS

- [ ] **Step 5: Commit**

```bash
git add pyradtran/interface.py tests/test_params.py tests/test_variational.py
git commit -m "feat: unified params mapping in batch interface; SimPoint replaces 10-tuple; fix B1/B5/B6/B14"
```

---

### Task 7: `OutputParser` respects per-run altitudes (B7)

**Files:**
- Modify: `pyradtran/io.py:596–727`
- Modify: `pyradtran/interface.py` (worker passes zout)
- Test: `tests/test_libradtran_output_parsing.py` (append)

**Interfaces:**
- Consumes: existing `OutputParser(config, parameter_overrides)`
- Produces: `OutputParser(config, parameter_overrides=None, output_altitudes=None)` — explicit `output_altitudes` (list of float) overrides the config value; the worker derives it from a resolved `zout` entry when present

- [ ] **Step 1: Write the failing test** (append to `tests/test_libradtran_output_parsing.py`)

```python
# append to tests/test_libradtran_output_parsing.py
from pyradtran.io import OutputParser, OutputType


class TestPerRunAltitudes:
    def test_explicit_output_altitudes_override_config(self, minimal_config, tmp_path):
        # Config says 3 altitudes; this run used zout override with 1
        out = tmp_path / "single_alt.out"
        out.write_text("  500.000  100.000   50.000\n  600.000  110.000   55.000\n")
        minimal_config.simulation_defaults.output_columns = ["lambda", "eglo", "eup"]
        parser = OutputParser(minimal_config, output_altitudes=[1.0])
        parsed = parser.parse_output_file(out)
        assert parsed.output_type == OutputType.SPECTRAL_SINGLE_ALTITUDE

    def test_zout_string_in_overrides_parsed(self, minimal_config):
        # Worker passes raw overrides; zout may be "0.0 1.0 120.0" or a float
        parser = OutputParser(minimal_config, {"zout": "0.0 1.0 120.0"})
        assert parser.output_altitudes == [0.0, 1.0, 120.0]

    def test_zout_float_in_overrides_parsed(self, minimal_config):
        parser = OutputParser(minimal_config, {"zout": 3.5})
        assert parser.output_altitudes == [3.5]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_libradtran_output_parsing.py::TestPerRunAltitudes -v`
Expected: FAIL (`unexpected keyword argument 'output_altitudes'` / altitude assertion)

- [ ] **Step 3: Modify `OutputParser.__init__`** (io.py:596–625)

```python
    def __init__(
        self,
        config: SimulationConfig,
        parameter_overrides: Dict[str, Any] = None,
        output_altitudes: Optional[List[float]] = None,
    ):
        self.config = config
        self.parameter_overrides = parameter_overrides or {}

        self.is_brightness_output = (
            self.parameter_overrides.get("output_quantity") == "brightness"
        )

        self.output_columns = []
        original_columns = config.simulation_defaults.output_columns or []
        for col in original_columns:
            if self.is_brightness_output and col == "albedo":
                logger.debug("Skipping albedo column for brightness temperature output")
                continue
            self.output_columns.append(col)

        # B7 fix: per-run altitudes win over config
        if output_altitudes is not None:
            self.output_altitudes = list(output_altitudes)
        elif "zout" in self.parameter_overrides:
            self.output_altitudes = self._parse_zout(
                self.parameter_overrides["zout"]
            )
        else:
            self.output_altitudes = (
                config.simulation_defaults.output_altitudes_km or [0.0]
            )

        self.wavelength_range = config.simulation_defaults.wavelength_nm
        self.is_integrated = getattr(
            config.simulation_defaults, "integrate_wavelength", False
        )
        logger.debug(f"Initialized parser with columns: {self.output_columns}")

    @staticmethod
    def _parse_zout(zout_value) -> List[float]:
        """Parse a zout override (float or whitespace-separated string)."""
        if isinstance(zout_value, str):
            return [float(tok) for tok in zout_value.split()]
        return [float(zout_value)]
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_libradtran_output_parsing.py tests/test_io.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pyradtran/io.py tests/test_libradtran_output_parsing.py
git commit -m "fix: OutputParser honours per-run zout instead of config altitudes (B7)"
```

---

### Task 8: Remaining small bug fixes — B8, B9, B10, B13

**Files:**
- Modify: `pyradtran/io.py:242` (B9), `io.py:276–290` (B8), `io.py:926–930` (B13)
- Modify: `pyradtran/config.py:535–544` (B10)
- Test: `tests/test_era5_atmosphere.py` (append), `tests/test_config.py` (existing must pass)

- [ ] **Step 1: Write the failing tests** (append to `tests/test_era5_atmosphere.py`)

```python
# append to tests/test_era5_atmosphere.py
import pytest

from pyradtran.exceptions import InputGenerationError
from pyradtran.io import ERA5AtmosphereGenerator


class TestUnitHandling:
    def test_unknown_pressure_unit_raises_clear_error(
        self, synthetic_era5_ds, tmp_path
    ):
        ds = synthetic_era5_ds.copy(deep=True)
        ds["pressure_level"].attrs["units"] = "millibars"
        with pytest.raises(InputGenerationError, match="millibars"):
            ERA5AtmosphereGenerator.create_era5_atmosphere_file(
                ds, 70.0, 25.0, "2022-07-01T12:00", tmp_path / "atm.dat"
            )

    def test_missing_q_units_defaults_to_kg_kg(self, synthetic_era5_ds, tmp_path):
        ds = synthetic_era5_ds.copy(deep=True)
        ds["q"].attrs.pop("units", None)
        out = ERA5AtmosphereGenerator.create_era5_atmosphere_file(
            ds, 70.0, 25.0, "2022-07-01T12:00", tmp_path / "atm.dat"
        )
        assert out.exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_era5_atmosphere.py::TestUnitHandling -v`
Expected: FAIL — unknown unit raises `InputGenerationError` wrapping a `NameError` (message lacks "millibars"); missing units raises `AttributeError` wrapped

- [ ] **Step 3: Apply the fixes**

`pyradtran/io.py` — delete stray line 242 (`output_filepath`). Replace unit handling (io.py:276–290):

```python
            # Units: attrs may be stripped by xarray operations; default sanely
            h2o_unit = profile_data["q"].attrs.get("units", "kg kg-1")
            p_unit = profile_data["pressure_level"].attrs.get("units", "hPa")

            if p_unit == "Pa":
                pressure_pa = profile_data["pressure_level"]
            elif p_unit == "hPa":
                pressure_pa = profile_data["pressure_level"] * 100
            else:
                raise InputGenerationError(
                    f"Unsupported pressure unit '{p_unit}' in ERA5 dataset; "
                    f"expected 'Pa' or 'hPa'"
                )
```

Note: the surrounding `except Exception` re-wraps into `InputGenerationError`; ensure the new raise is not double-wrapped by adding at the top of that `except` block:

```python
        except InputGenerationError:
            raise
        except Exception as e:
            raise InputGenerationError(
                f"Failed to create ERA5 atmosphere file: {str(e)}"
            )
```

`pyradtran/io.py` B13 (io.py:926–930) — only copy non-point metadata:

```python
        # Add attributes (skip per-point fields — they only describe the
        # template point, not the whole batch)
        _POINT_KEYS = {"point_id", "time", "latitude", "longitude",
                       "albedo", "surface_temperature", "surface_type",
                       "altitude"}
        if template_output.metadata:
            result_ds.attrs.update(
                {k: v for k, v in template_output.metadata.items()
                 if k not in _POINT_KEYS}
            )
```

`pyradtran/config.py` B10 — delete lines 535–544 (the dead default-backfill block: `# Add default values for missing keys` through `init_args[f.name] = f.default_factory()`). Dataclass constructors apply defaults themselves.

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_era5_atmosphere.py tests/test_config.py tests/test_io.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pyradtran/io.py pyradtran/config.py tests/test_era5_atmosphere.py
git commit -m "fix: ERA5 unit dispatch (B8), stray statement (B9), dead config code (B10), batch attrs (B13)"
```

---

### Task 9: `explain()` and `dry_run()` (Phase 2)

**Files:**
- Modify: `pyradtran/core.py` (add `dry_run`)
- Modify: `pyradtran/interface.py` (add `PyRadtranAccessor.explain`)
- Test: `tests/test_explain.py`

**Interfaces:**
- Consumes: `Simulation.build_input_lines`, `InputFileBuilder.render_annotated`, `ParamResolver`
- Produces:
  - `Simulation.dry_run(dt, latitude, longitude, resolved_params=None, era5_atmosphere_file=None) -> str` — annotated input file, no subprocess
  - `PyRadtranAccessor.explain(point=None, params=None, config_path=None, config=None, time_var="time", lat_var="latitude", lon_var="longitude") -> str` — *point* is a `.sel()`-style dict (nearest match); default first point

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_explain.py
"""Tests for explain()/dry_run() — annotated input preview, no subprocess."""

from datetime import datetime
from unittest.mock import patch

import pandas as pd
import pytest
import xarray as xr

from pyradtran.core import Simulation
from pyradtran.params import PROV_LITERAL, Var


class TestDryRun:
    def test_dry_run_returns_annotated_text(self, minimal_config):
        sim = Simulation(minimal_config)
        text = sim.dry_run(
            datetime(2022, 7, 1, 12), 78.0, 15.0,
            resolved_params={"albedo": (0.85, PROV_LITERAL)},
        )
        assert "albedo 0.85" in text
        assert "# params-literal" in text
        assert "# config" in text

    def test_dry_run_spawns_no_subprocess(self, minimal_config):
        sim = Simulation(minimal_config)
        with patch("subprocess.run") as mock_run:
            sim.dry_run(datetime(2022, 7, 1, 12), 78.0, 15.0)
        mock_run.assert_not_called()


class TestAccessorExplain:
    def test_explain_first_point_default(self, minimal_config, simple_input_dataset):
        text = simple_input_dataset.pyradtran.explain(config=minimal_config)
        assert "rte_solver disort" in text
        assert "# config" in text

    def test_explain_resolves_var_for_point(self, minimal_config, simple_input_dataset):
        ds = simple_input_dataset.copy()
        ds["alb"] = (["time"], [0.11, 0.22, 0.33])
        text = ds.pyradtran.explain(
            config=minimal_config, params={"albedo": Var("alb")}
        )
        assert "albedo 0.11" in text
        assert "# dataset-var" in text

    def test_explain_with_point_selector(self, minimal_config, simple_input_dataset):
        ds = simple_input_dataset.copy()
        ds["alb"] = (["time"], [0.11, 0.22, 0.33])
        text = ds.pyradtran.explain(
            config=minimal_config,
            params={"albedo": Var("alb")},
            point={"time": pd.Timestamp("2022-07-01 12:30")},
        )
        assert "albedo 0.22" in text
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_explain.py -v`
Expected: FAIL with `AttributeError: 'Simulation' object has no attribute 'dry_run'`

- [ ] **Step 3: Implement**

`pyradtran/core.py`, inside `Simulation`:

```python
    def dry_run(
        self,
        dt: datetime,
        latitude: float,
        longitude: float,
        resolved_params=None,
        era5_atmosphere_file: Optional[Path] = None,
    ) -> str:
        """Render the annotated input file for one point without running uvspec.

        Returns
        -------
        str
            Input-file content, one ``# <provenance>`` comment per line.
        """
        lines = self.build_input_lines(
            dt, latitude, longitude,
            resolved_params=resolved_params,
            era5_atmosphere_file=era5_atmosphere_file,
        )
        return self.builder.render_annotated(lines)
```

`pyradtran/interface.py`, inside `PyRadtranAccessor`:

```python
    def explain(
        self,
        point: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        config_path: Optional[Union[str, Path]] = None,
        config: Optional[SimulationConfig] = None,
        time_var: str = "time",
        lat_var: str = "latitude",
        lon_var: str = "longitude",
    ) -> str:
        """Preview the annotated uvspec input file for one point.

        No simulation is run. Each line is tagged with the layer that
        produced it (``config`` / ``params-literal`` / ``dataset-var`` /
        ``unvalidated``).

        Parameters
        ----------
        point : dict, optional
            ``Dataset.sel()``-style selector (nearest match). Defaults to
            the first element along every dimension.
        params : dict, optional
            Same mapping accepted by :meth:`run`.
        config_path, config
            Configuration source, same as :meth:`run`.

        Returns
        -------
        str
        """
        cfg = config if config is not None else load_config(config_path)
        resolver = ParamResolver(cfg, params)
        resolver.validate_var_targets(self._obj)

        if point is None:
            point_ds = self._obj.isel({d: 0 for d in self._obj.dims})
        else:
            point_ds = self._obj.sel(point, method="nearest")

        def scalar(var):
            if var in point_ds:
                v = point_ds[var].values
                return v.item() if hasattr(v, "item") else v
            return None

        resolved, _skipped = resolver.resolve_point(point_ds)
        dt = pd.to_datetime(scalar(time_var)).to_pydatetime()
        sim = Simulation(cfg)
        return sim.dry_run(
            dt, scalar(lat_var), scalar(lon_var), resolved_params=resolved
        )
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_explain.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pyradtran/core.py pyradtran/interface.py tests/test_explain.py
git commit -m "feat: explain()/dry_run() annotated input preview with provenance"
```

---

### Task 10: Failure reporting — `status` variable, stderr log, kept files (Phase 3)

**Files:**
- Modify: `pyradtran/core.py` (`_run_uvspec` captures stderr into `last_stderr`)
- Modify: `pyradtran/interface.py` (worker returns outcome; batch writes log; accessor adds `status`)
- Test: `tests/test_failures.py`

**Interfaces:**
- Consumes: `Simulation.last_stderr` (Task 5), `SimPoint`
- Produces:
  - `PointOutcome` dataclass in `interface.py`: `parsed: Optional[ParsedOutput]`, `status: int`, `detail: Optional[str]`, `point_id: str` (status: 0 ok, 1 uvspec failure, 2 skipped-NaN)
  - `_run_single_simulation_unified(config, point) -> PointOutcome` (return type change is internal — batch adapts)
  - `execute_simulation_batch(..., return_outcomes: bool = False)` — default returns `List[Optional[ParsedOutput]]` (unchanged public behaviour); `True` returns `List[PointOutcome]`
  - Batch side effect: `working_dir/failures_<YYYYmmdd_HHMMSS>.log` written when any point fails, one block per failure (point_id, input path if available, stderr)
  - `PyRadtranAccessor.run` result gains `status` data variable over the point dims

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_failures.py
"""Failure reporting: status codes, stderr log, kept temp files."""

import stat
from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr

from pyradtran.interface import PointOutcome, execute_simulation_batch
from pyradtran.io import OutputType, ParsedOutput


def _make_failing_uvspec(config):
    """Turn the mock uvspec binary into a script that fails loudly."""
    bin_path = config.paths.libradtran_bin
    bin_path.write_text("#!/bin/bash\necho 'uvspec exploded' >&2\nexit 1\n")
    bin_path.chmod(bin_path.stat().st_mode | stat.S_IEXEC)


def _make_succeeding_uvspec(config):
    """Mock uvspec that emits one integrated single-altitude row."""
    bin_path = config.paths.libradtran_bin
    bin_path.write_text(
        "#!/bin/bash\ncat > /dev/null\necho ' 30.0  100.0  50.0  0.1'\n"
    )
    bin_path.chmod(bin_path.stat().st_mode | stat.S_IEXEC)


@pytest.fixture
def single_alt_config(minimal_config):
    minimal_config.simulation_defaults.output_altitudes_km = [0.0]
    minimal_config.simulation_defaults.integrate_wavelength = True
    minimal_config.simulation_defaults.output_columns = [
        "sza", "eglo", "eup", "albedo",
    ]
    return minimal_config


class TestOutcomes:
    def test_failed_run_yields_status_1_with_stderr(
        self, single_alt_config, simple_input_dataset
    ):
        _make_failing_uvspec(single_alt_config)
        with pytest.raises(Exception):
            # all fail -> batch raises, but outcomes requested first
            execute_simulation_batch(
                config=single_alt_config,
                input_ds=simple_input_dataset,
                show_progress=False,
                return_outcomes=True,
            )
        # failure log written
        logs = list(single_alt_config.paths.working_dir.glob("failures_*.log"))
        assert logs, "expected a failures_*.log file"
        assert "uvspec exploded" in logs[0].read_text()

    def test_success_yields_status_0(self, single_alt_config, simple_input_dataset):
        _make_succeeding_uvspec(single_alt_config)
        outcomes = execute_simulation_batch(
            config=single_alt_config,
            input_ds=simple_input_dataset,
            show_progress=False,
            return_outcomes=True,
        )
        assert all(isinstance(o, PointOutcome) for o in outcomes)
        assert all(o.status == 0 for o in outcomes)
        assert all(o.parsed is not None for o in outcomes)

    def test_default_return_shape_unchanged(
        self, single_alt_config, simple_input_dataset
    ):
        _make_succeeding_uvspec(single_alt_config)
        results = execute_simulation_batch(
            config=single_alt_config,
            input_ds=simple_input_dataset,
            show_progress=False,
        )
        assert all(isinstance(r, ParsedOutput) for r in results)


class TestFailedInputKept:
    def test_input_file_kept_on_failure_despite_cleanup_flag(
        self, single_alt_config, simple_input_dataset
    ):
        _make_failing_uvspec(single_alt_config)
        assert single_alt_config.execution.cleanup_temp_files is True
        with pytest.raises(Exception):
            execute_simulation_batch(
                config=single_alt_config,
                input_ds=simple_input_dataset,
                show_progress=False,
            )
        kept = list(single_alt_config.paths.working_dir.glob("*.inp"))
        assert kept, "failed-run input files must be kept for post-mortem"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_failures.py -v`
Expected: FAIL with `ImportError: cannot import name 'PointOutcome'`

- [ ] **Step 3: Implement**

`pyradtran/core.py` — `_run_uvspec` stores stderr:

```python
            if result.returncode != 0:
                error_msg = result.stderr.strip() if result.stderr else "Unknown error"
                self.last_stderr = error_msg
                logger.error(
                    f"LibRadtran failed with return code {result.returncode}: {error_msg}"
                )
                return False
```

and in the timeout/except branches:

```python
        except subprocess.TimeoutExpired:
            self.last_stderr = (
                f"timeout after {self.config.execution.timeout_seconds}s"
            )
            logger.error(
                f"LibRadtran execution timed out after "
                f"{self.config.execution.timeout_seconds} seconds"
            )
            return False
        except Exception as e:
            self.last_stderr = str(e)
            logger.error(f"Failed to execute LibRadtran: {str(e)}")
            return False
```

Also in `run_simulation`, record the failed input path so the log can reference it:

```python
            # Failure: keep input and output for post-mortem (B12)
            self.last_failed_input = input_path
            logger.error(f"Simulation failed; input kept at {input_path}")
            return None
```

(`last_failed_input: Optional[Path] = None` initialised in `__init__` next to `last_stderr`.)

`pyradtran/interface.py`:

```python
@dataclass
class PointOutcome:
    """Result envelope for one point: parsed output + status + failure detail."""

    parsed: Optional[ParsedOutput]
    status: int  # 0 = ok, 1 = uvspec failure, 2 = skipped (NaN inputs)
    detail: Optional[str] = None
    point_id: str = ""
```

Worker returns `PointOutcome`:

```python
def _run_single_simulation_unified(
    config: SimulationConfig,
    point: SimPoint,
) -> PointOutcome:
    """Execute a single ``uvspec`` run (called by the process pool)."""
    try:
        sim = Simulation(config)
        dt = _coerce_datetime(point.time)
        output_file = sim.run_simulation(
            dt=dt,
            latitude=point.latitude,
            longitude=point.longitude,
            resolved_params=point.resolved,
            era5_atmosphere_file=point.era5_file,
        )
        if output_file and output_file.exists():
            raw_overrides = {k: v for k, (v, _p) in point.resolved.items()}
            parser = OutputParser(config, raw_overrides)
            parsed_output = parser.parse_output_file(output_file)
            parsed_output.metadata.update(
                {
                    "point_id": point.point_id,
                    "time": dt.isoformat(),
                    "latitude": point.latitude,
                    "longitude": point.longitude,
                }
            )
            return PointOutcome(parsed_output, 0, None, point.point_id)
        detail = sim.last_stderr or "no output produced"
        if sim.last_failed_input is not None:
            detail = f"input: {sim.last_failed_input}\n{detail}"
        return PointOutcome(None, 1, detail, point.point_id)
    except Exception as e:
        logger.error(f"Single simulation failed for point {point.point_id}: {e}")
        return PointOutcome(None, 1, str(e), point.point_id)
```

Batch collection loop adapts (results list holds outcomes; success test becomes `outcome.status == 0`), writes the failure log after the pool closes, and honours `return_outcomes`:

```python
    outcomes: List[Optional[PointOutcome]] = [None] * num_points
    # ... in the as_completed loop:
            outcome = future.result()
            outcomes[idx] = outcome
            if outcome is not None and outcome.status == 0:
                success_count += 1

    # after pool close / pbar close:
    failures = [o for o in outcomes if o is not None and o.status != 0]
    if failures:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = Path(config.paths.working_dir) / f"failures_{run_id}.log"
        with open(log_path, "w") as f:
            for o in failures:
                f.write(f"=== point {o.point_id} (status {o.status}) ===\n")
                f.write((o.detail or "no detail") + "\n\n")
        logger.warning(
            f"{len(failures)}/{num_points} simulations failed; "
            f"details in {log_path}"
        )

    if success_count == 0:
        # keep raising AFTER the log is written
        raise PyRadtranError("All simulations failed - no valid results produced")

    if return_outcomes:
        return outcomes
    return [o.parsed if o is not None else None for o in outcomes]
```

**Ordering note:** write the failure log *before* the `success_count == 0` raise, exactly as shown, so the log exists even when everything failed (the first test relies on this).

Points whose `skipped` list is non-empty and whose parsed output is missing a required parameter still run (parameter omitted); `status=2` is reserved for points the resolver skipped entirely — set it in the point-prep loop when `time`/`lat`/`lon` themselves are NaN:

```python
        if any(
            v is not None and isinstance(v, float) and np.isnan(v)
            for v in (lat, lon)
        ):
            outcomes_prefill = PointOutcome(None, 2, "NaN coordinates", f"point_{i}")
            # store and skip submission for this index
```

Implementer note: hold pre-filled outcomes in a dict `prefilled: Dict[int, PointOutcome]`, skip those indices at submit time, and merge into `outcomes` before the log-writing step.

`PyRadtranAccessor.run`: call with `return_outcomes=True`, build `parsed_outputs = [o.parsed if o else None for o in outcomes]` for the converter, then attach status by unstacking (same pattern `convert_batch` uses):

```python
        status_flat = np.array(
            [o.status if o is not None else 1 for o in outcomes]
        )
        dims = list(ds_to_execute.sizes.keys())
        if dims:
            stacked = ds_to_execute.stack({"sample_batch_dim": dims})
            status_da = xr.DataArray(
                status_flat,
                coords={"sample_batch_dim": stacked["sample_batch_dim"]},
                dims=["sample_batch_dim"],
            ).unstack("sample_batch_dim")
        else:
            status_da = xr.DataArray(int(status_flat[0]))
        result_ds["status"] = status_da
        result_ds["status"].attrs["flag_values"] = "0: ok, 1: failed, 2: skipped"
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_failures.py tests/test_interface.py -v`
Expected: PASS

Run: `python -m pytest tests/ -q`
Expected: full suite PASS

- [ ] **Step 5: Commit**

```bash
git add pyradtran/core.py pyradtran/interface.py tests/test_failures.py
git commit -m "feat: per-point status codes, stderr failure log, kept inputs on failure"
```

---

### Task 11: `channels.py` — SRF convolution + brightness temperature (Phase 4)

**Files:**
- Create: `pyradtran/channels.py`
- Test: `tests/test_channels.py`

**Interfaces:**
- Consumes: an `xr.Dataset` with a `wavelength` dim (nm); an SRF `xr.DataArray` with dims `(channel, wavelength)`
- Produces:
  - `convolve_channels(result: xr.Dataset, srf: xr.DataArray, keep_spectral: bool = False) -> xr.Dataset` — every data variable with a `wavelength` dim is SRF-averaged onto a `channel` dim: `v_ch = ∫ v(λ) φ_ch(λ) dλ / ∫ φ_ch(λ) dλ` (trapezoidal); non-spectral variables pass through; spectral originals dropped unless `keep_spectral`
  - `brightness_temperature(radiance, wavelength_nm, radiance_units="mW m-2 nm-1 sr-1") -> same shape` — inverse Planck; accepted units: `"mW m-2 nm-1 sr-1"` (uvspec default) and `"W m-2 nm-1 sr-1"`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_channels.py
"""Analytic tests for SRF convolution and inverse Planck."""

import numpy as np
import pytest
import xarray as xr

from pyradtran.channels import brightness_temperature, convolve_channels

H = 6.62607015e-34
C = 2.99792458e8
KB = 1.380649e-23


def planck_radiance_nm(wavelength_nm, T):
    """Planck spectral radiance in W m-2 nm-1 sr-1."""
    lam = wavelength_nm * 1e-9
    B = (2 * H * C**2 / lam**5) / (np.expm1(H * C / (lam * KB * T)))  # W m-3 sr-1
    return B * 1e-9  # per nm


@pytest.fixture
def spectral_result():
    wl = np.linspace(400.0, 700.0, 301)
    rad = np.tile(2.0 + 0.01 * (wl - 400.0), (2, 1))  # linear in wl, 2 time steps
    return xr.Dataset(
        {"uu": (("time", "wavelength"), rad), "sza": (("time",), [30.0, 40.0])},
        coords={"wavelength": wl, "time": [0, 1]},
    )


@pytest.fixture
def boxcar_srf():
    wl = np.linspace(400.0, 700.0, 301)
    phi = np.zeros((2, wl.size))
    phi[0, (wl >= 450) & (wl <= 550)] = 1.0
    phi[1, (wl >= 600) & (wl <= 650)] = 1.0
    return xr.DataArray(
        phi,
        dims=("channel", "wavelength"),
        coords={"channel": ["ch1", "ch2"], "wavelength": wl},
    )


class TestConvolve:
    def test_boxcar_average_of_linear_spectrum(self, spectral_result, boxcar_srf):
        out = convolve_channels(spectral_result, boxcar_srf)
        # Boxcar over linear ramp -> value at band centre
        expected_ch1 = 2.0 + 0.01 * (500.0 - 400.0)  # centre 500 nm
        assert out["uu"].sel(channel="ch1").values == pytest.approx(
            expected_ch1, rel=1e-3
        )

    def test_channel_dim_replaces_wavelength(self, spectral_result, boxcar_srf):
        out = convolve_channels(spectral_result, boxcar_srf)
        assert "channel" in out["uu"].dims
        assert "wavelength" not in out["uu"].dims

    def test_nonspectral_vars_pass_through(self, spectral_result, boxcar_srf):
        out = convolve_channels(spectral_result, boxcar_srf)
        assert "sza" in out
        assert list(out["sza"].values) == [30.0, 40.0]

    def test_keep_spectral(self, spectral_result, boxcar_srf):
        out = convolve_channels(spectral_result, boxcar_srf, keep_spectral=True)
        assert "uu_spectral" in out
        assert "wavelength" in out["uu_spectral"].dims


class TestBrightnessTemperature:
    def test_planck_roundtrip(self):
        T_true = 280.0
        wl_nm = 10500.0  # thermal IR
        L = planck_radiance_nm(wl_nm, T_true)  # W m-2 nm-1 sr-1
        T = brightness_temperature(L, wl_nm, radiance_units="W m-2 nm-1 sr-1")
        assert T == pytest.approx(T_true, abs=0.01)

    def test_uvspec_default_units_mw(self):
        T_true = 280.0
        wl_nm = 10500.0
        L_mw = planck_radiance_nm(wl_nm, T_true) * 1e3  # mW
        T = brightness_temperature(L_mw, wl_nm)
        assert T == pytest.approx(T_true, abs=0.01)

    def test_unknown_units_raise(self):
        with pytest.raises(ValueError, match="radiance_units"):
            brightness_temperature(1.0, 10500.0, radiance_units="furlongs")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_channels.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pyradtran.channels'`

- [ ] **Step 3: Write the implementation**

```python
# pyradtran/channels.py
"""
Instrument-channel convolution and brightness temperature.

Convolve spectral pyRadtran results with instrument spectral response
functions (SRFs) and convert thermal radiances to brightness temperatures.

Examples
--------
>>> channel_ds = convolve_channels(result_ds, srf)
>>> tb = brightness_temperature(channel_ds["uu"], 10500.0)

See Also
--------
pyradtran.interface.PyRadtranAccessor.run : ``channels=`` integration.
"""

import logging

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)

# CODATA 2018
_H = 6.62607015e-34   # J s
_C = 2.99792458e8     # m s-1
_KB = 1.380649e-23    # J K-1

#: radiance unit -> factor converting to W m-2 m-1 sr-1 (SI, per metre)
_UNIT_TO_SI = {
    "mW m-2 nm-1 sr-1": 1e-3 * 1e9,
    "W m-2 nm-1 sr-1": 1e9,
}


def convolve_channels(
    result: xr.Dataset,
    srf: xr.DataArray,
    keep_spectral: bool = False,
) -> xr.Dataset:
    """SRF-average all spectral variables onto a ``channel`` dimension.

    Parameters
    ----------
    result : xarray.Dataset
        Spectral simulation result with a ``wavelength`` dimension (nm).
    srf : xarray.DataArray
        Spectral response, dims ``(channel, wavelength)``. Interpolated
        onto the result's wavelength grid; values outside the SRF grid
        are treated as zero response.
    keep_spectral : bool, default ``False``
        Keep the original spectral variables under ``<name>_spectral``.

    Returns
    -------
    xarray.Dataset
        Channel-space dataset; non-spectral variables pass through.
    """
    if "wavelength" not in result.dims:
        raise ValueError("result has no 'wavelength' dimension to convolve")
    wl = result["wavelength"]
    phi = srf.interp(wavelength=wl, kwargs={"fill_value": 0.0}).fillna(0.0)
    norm = phi.integrate("wavelength")

    out = xr.Dataset(attrs=dict(result.attrs))
    for name, da in result.data_vars.items():
        if "wavelength" in da.dims:
            conv = (da * phi).integrate("wavelength") / norm
            out[name] = conv
            if keep_spectral:
                out[f"{name}_spectral"] = da
        else:
            out[name] = da
    out["channel"] = srf["channel"]
    return out


def brightness_temperature(
    radiance,
    wavelength_nm,
    radiance_units: str = "mW m-2 nm-1 sr-1",
):
    """Convert monochromatic radiance to brightness temperature (K).

    Inverse Planck: ``T = c2 / (lambda * ln(1 + c1 / (lambda^5 * L)))``
    with ``c1 = 2 h c^2`` and ``c2 = h c / k``.

    Parameters
    ----------
    radiance : array-like or xarray.DataArray
        Spectral radiance.
    wavelength_nm : float or array-like
        Wavelength in nanometres.
    radiance_units : str, default ``"mW m-2 nm-1 sr-1"``
        Units of *radiance* (uvspec's default radiance unit).

    Returns
    -------
    Brightness temperature in K, same shape as *radiance*.
    """
    if radiance_units not in _UNIT_TO_SI:
        raise ValueError(
            f"Unsupported radiance_units '{radiance_units}'; "
            f"expected one of {sorted(_UNIT_TO_SI)}"
        )
    L_si = np.asarray(radiance, dtype=float) * _UNIT_TO_SI[radiance_units]
    lam = np.asarray(wavelength_nm, dtype=float) * 1e-9
    c1 = 2 * _H * _C**2
    c2 = _H * _C / _KB
    return c2 / (lam * np.log1p(c1 / (lam**5 * L_si)))


__all__ = ["convolve_channels", "brightness_temperature"]
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_channels.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pyradtran/channels.py tests/test_channels.py
git commit -m "feat: SRF channel convolution and inverse-Planck brightness temperature"
```

---

### Task 12: `run(channels=...)` integration

**Files:**
- Modify: `pyradtran/interface.py` (`PyRadtranAccessor.run`)
- Test: `tests/test_channels.py` (append)

**Interfaces:**
- Consumes: `convolve_channels` (Task 11)
- Produces: `PyRadtranAccessor.run(..., channels: Optional[xr.DataArray] = None, keep_spectral: bool = False)` — when *channels* is given and the result has a `wavelength` dim, the returned (and saved) dataset is channel-space

- [ ] **Step 1: Write the failing test** (append to `tests/test_channels.py`)

```python
# append to tests/test_channels.py
from unittest.mock import patch


class TestRunChannelsIntegration:
    def test_run_applies_convolution(
        self, minimal_config, boxcar_srf, spectral_result
    ):
        """run(channels=srf) post-processes the converted dataset."""
        import pyradtran.interface as interface

        ds_in = xr.Dataset(
            data_vars={
                "latitude": (["time"], [78.0, 78.1]),
                "longitude": (["time"], [15.0, 15.1]),
            },
            coords={"time": [0, 1]},
        )

        with patch.object(
            interface, "execute_simulation_batch"
        ) as mock_batch, patch.object(
            interface.OutputToXarray, "convert_batch",
            return_value=spectral_result,
        ):
            from pyradtran.interface import PointOutcome

            mock_batch.return_value = [PointOutcome(object(), 0), PointOutcome(object(), 0)]
            out = ds_in.pyradtran.run(
                config=minimal_config,
                channels=boxcar_srf,
                save_to_file=False,
                show_progress=False,
            )
        assert "channel" in out.dims
        assert "wavelength" not in out["uu"].dims
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_channels.py::TestRunChannelsIntegration -v`
Expected: FAIL with `TypeError: run() got an unexpected keyword argument 'channels'`

- [ ] **Step 3: Implement**

In `PyRadtranAccessor.run` signature add:

```python
        channels: Optional[xr.DataArray] = None,
        keep_spectral: bool = False,
```

After `result_ds` is built by `convert_batch` (and after the `status` variable is attached), before saving:

```python
            # Instrument-channel convolution
            if channels is not None:
                from .channels import convolve_channels

                if "wavelength" in result_ds.dims:
                    status_var = result_ds.get("status")
                    result_ds = convolve_channels(
                        result_ds.drop_vars("status", errors="ignore"),
                        channels,
                        keep_spectral=keep_spectral,
                    )
                    if status_var is not None:
                        result_ds["status"] = status_var
                else:
                    logger.warning(
                        "channels= given but result has no wavelength dimension; "
                        "skipping convolution"
                    )
```

Docstring additions for both new parameters mirroring existing style.

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_channels.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pyradtran/interface.py tests/test_channels.py
git commit -m "feat: run(channels=srf) applies SRF convolution to spectral results"
```

---

### Task 13: `jacobian()` (Phase 4)

**Files:**
- Modify: `pyradtran/interface.py` (`PyRadtranAccessor.jacobian`)
- Test: `tests/test_jacobian.py`

**Interfaces:**
- Consumes: `PyRadtranAccessor.run`, `REGISTRY`, `CONFIG_FIELD_MAP`
- Produces: `PyRadtranAccessor.jacobian(param: str, delta: float, params=None, config_path=None, config=None, **run_kwargs) -> xr.Dataset` — `(perturbed − base) / delta`; attrs `jacobian_param`, `jacobian_delta`. Raises `ValidationError` if *param* is a `Var` reference or has no resolvable base value.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_jacobian.py
"""Jacobian mode: finite-difference kernels via paired runs."""

from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr

from pyradtran.exceptions import ValidationError
from pyradtran.params import Var


@pytest.fixture
def ds_in():
    return xr.Dataset(
        data_vars={
            "latitude": (["time"], [78.0, 78.1]),
            "longitude": (["time"], [15.0, 15.1]),
        },
        coords={"time": [0, 1]},
    )


class TestJacobian:
    def test_linear_model_gives_exact_kernel(self, minimal_config, ds_in):
        """Mock forward model: eup = 100 * albedo -> dU/dalbedo = 100."""

        def fake_run(self, **kwargs):
            alb = kwargs["params"]["albedo"]
            return xr.Dataset(
                {"eup": (("time",), [100.0 * alb, 100.0 * alb])},
                coords={"time": [0, 1]},
            )

        from pyradtran.interface import PyRadtranAccessor

        with patch.object(PyRadtranAccessor, "run", fake_run):
            jac = ds_in.pyradtran.jacobian(
                "albedo", 0.01, params={"albedo": 0.5}, config=minimal_config
            )
        assert jac["eup"].values == pytest.approx([100.0, 100.0])
        assert jac.attrs["jacobian_param"] == "albedo"
        assert jac.attrs["jacobian_delta"] == 0.01

    def test_base_value_from_config_field(self, minimal_config, ds_in):
        """albedo base value falls back to config albedo_value (0.1)."""
        seen = []

        def fake_run(self, **kwargs):
            seen.append(kwargs["params"]["albedo"])
            return xr.Dataset({"eup": (("time",), [1.0, 1.0])}, coords={"time": [0, 1]})

        from pyradtran.interface import PyRadtranAccessor

        with patch.object(PyRadtranAccessor, "run", fake_run):
            ds_in.pyradtran.jacobian("albedo", 0.01, config=minimal_config)
        assert seen == [pytest.approx(0.1), pytest.approx(0.11)]

    def test_var_param_rejected(self, minimal_config, ds_in):
        with pytest.raises(ValidationError):
            ds_in.pyradtran.jacobian(
                "albedo", 0.01,
                params={"albedo": Var("alb")}, config=minimal_config,
            )

    def test_unresolvable_base_rejected(self, minimal_config, ds_in):
        minimal_config.simulation_defaults.albedo_value = None
        with pytest.raises(ValidationError):
            ds_in.pyradtran.jacobian("albedo", 0.01, config=minimal_config)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_jacobian.py -v`
Expected: FAIL with `AttributeError: ... no attribute 'jacobian'`

- [ ] **Step 3: Implement** (in `PyRadtranAccessor`; add `from .params import CONFIG_FIELD_MAP` to the interface imports)

```python
    def jacobian(
        self,
        param: str,
        delta: float,
        params: Optional[Dict[str, Any]] = None,
        config_path: Optional[Union[str, Path]] = None,
        config: Optional[SimulationConfig] = None,
        **run_kwargs,
    ) -> xr.Dataset:
        """Finite-difference sensitivity kernel for one scalar parameter.

        Runs the batch twice (base and ``param + delta``) and returns
        ``(perturbed - base) / delta`` with the same dimensions.

        Parameters
        ----------
        param : str
            Registry parameter to perturb (must resolve to a scalar:
            a ``params`` literal or a config default — not a ``Var``).
        delta : float
            Perturbation size in the parameter's units.
        params : dict, optional
            Base parameter mapping (same as :meth:`run`).
        config_path, config
            Configuration source, same as :meth:`run`.
        **run_kwargs
            Forwarded to :meth:`run` (e.g. ``show_progress=False``).

        Returns
        -------
        xarray.Dataset
            Kernel dataset; attrs ``jacobian_param``, ``jacobian_delta``.

        Raises
        ------
        ValidationError
            If *param* is a ``Var`` reference or no base value exists.
        """
        from .exceptions import ValidationError
        from .params import CONFIG_FIELD_MAP

        params = dict(params or {})
        base_value = params.get(param)
        if isinstance(base_value, Var):
            raise ValidationError(
                f"jacobian() cannot perturb '{param}': it is a per-point "
                f"Var reference; supply a scalar literal instead"
            )
        cfg = config if config is not None else load_config(config_path)
        if base_value is None:
            field_name = CONFIG_FIELD_MAP.get(param)
            if field_name is not None:
                base_value = getattr(cfg.simulation_defaults, field_name, None)
        if base_value is None:
            raise ValidationError(
                f"jacobian() needs a base value for '{param}': set it in "
                f"params or in the configuration"
            )

        run_kwargs.setdefault("save_to_file", False)
        base_params = {**params, param: float(base_value)}
        pert_params = {**params, param: float(base_value) + float(delta)}

        base = self.run(config=cfg, params=base_params, **run_kwargs)
        perturbed = self.run(config=cfg, params=pert_params, **run_kwargs)

        jac = (perturbed - base) / float(delta)
        jac.attrs["jacobian_param"] = param
        jac.attrs["jacobian_delta"] = float(delta)
        jac.attrs["jacobian_base_value"] = float(base_value)
        return jac
```

Note: `(perturbed - base)` drops the `status` variable meaningfully — xarray subtracts it too. Acceptable: a nonzero `status` difference flags asymmetric failures; do not special-case it.

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_jacobian.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pyradtran/interface.py tests/test_jacobian.py
git commit -m "feat: jacobian() finite-difference kernels via paired batch runs"
```

---

### Task 14: Public exports, version bump, full-suite gate

**Files:**
- Modify: `pyradtran/__init__.py`
- Test: full suite

- [ ] **Step 1: Write the failing test** (append to `tests/test_params.py`)

```python
# append to tests/test_params.py
class TestPublicAPI:
    def test_top_level_exports(self):
        import pyradtran

        assert pyradtran.Var is Var
        assert callable(pyradtran.convolve_channels)
        assert callable(pyradtran.brightness_temperature)
        assert pyradtran.__version__ == "0.2.0"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_params.py::TestPublicAPI -v`
Expected: FAIL with `AttributeError: module 'pyradtran' has no attribute 'Var'`

- [ ] **Step 3: Modify `pyradtran/__init__.py`**

```python
__version__ = "0.2.0"
```

Add imports after the existing `from .utils import RadiosondeFinder` line:

```python
from .channels import brightness_temperature, convolve_channels  # noqa: E402
from .params import REGISTRY, ParamResolver, ParamSpec, Var  # noqa: E402
```

Extend `__all__`:

```python
    # Parameters & channels
    "Var",
    "ParamSpec",
    "ParamResolver",
    "REGISTRY",
    "convolve_channels",
    "brightness_temperature",
```

Also check `pyproject.toml` for a version field; if present, bump it to `0.2.0` too.

- [ ] **Step 4: Run the full suite**

Run: `python -m pytest tests/ -q`
Expected: full suite PASS (integration tests may skip without a real libRadtran install)

Run: `python -m pytest tests/ -q -W error::DeprecationWarning -k "test_params or test_input_builder or test_explain or test_channels or test_jacobian or test_failures"`
Expected: PASS — new-API tests must not themselves trigger deprecation warnings

- [ ] **Step 5: Commit**

```bash
git add pyradtran/__init__.py pyproject.toml tests/test_params.py
git commit -m "feat: export Var/channels API, bump version to 0.2.0"
```

---

## Verification (after all tasks)

- [ ] `python -m pytest tests/ -q` — full suite green
- [ ] `python -c "import pyradtran; print(pyradtran.__version__)"` → `0.2.0`
- [ ] Manual smoke (requires real libRadtran): run one notebook cell of `book/notebooks/thermal.ipynb` with `params={"sur_temperature": 271.0}` and confirm `explain()` shows the value with `# params-literal`
