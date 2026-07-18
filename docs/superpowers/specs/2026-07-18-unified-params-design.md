# pyRadtran 0.2 — Unified Parameter Handling & Five-Star Release

**Date:** 2026-07-18
**Status:** Approved (design), pending implementation
**Scope:** `pyradtran/` package. Notebooks and docs updated only where APIs they use change.

## 1. Problem

The same physical quantity (surface albedo, surface temperature, surface type,
observation altitude, solar zenith angle, …) currently reaches the `uvspec`
input file through four independent mechanisms:

1. **Typed per-point kwargs** — `albedo_var`, `surface_temperature_var`,
   `surface_type_var`, `altitude_var` threaded through
   `PyRadtranAccessor.run` → `execute_simulation_batch` → a 10-element
   positional tuple → `Simulation.run_simulation(override_albedo=…, …)`.
   Adding a parameter requires touching five call sites.
2. **Config defaults** — `SimulationDefaults.albedo_value`,
   `surface_temperature_k`.
3. **`parameter_overrides` raw dict** — appended verbatim as uvspec lines,
   with hidden magic: a string value that happens to match a dataset variable
   name is silently resolved per-point (`interface.py:341`).
4. **Dotted config overrides** — `"simulation_defaults.albedo_value"` handled
   by `_apply_parameter_overrides`.

This inconsistency produces real bugs (Section 2) and makes the package hard
to extend.

## 2. Bugs to fix (all fixed in Phase 1 unless noted)

| # | Location | Defect |
|---|----------|--------|
| B1 | `interface.py:107,125` and `:753,820` | Dotted override keys (`simulation_defaults.X`) are applied to the config but **not removed** from the dict, so they are also appended as literal uvspec lines (`simulation_defaults.albedo_value 0.3`), which uvspec rejects. |
| B2 | `core.py:242–247` | `ozone_du` and `h2o_mm` config fields are parsed, documented, and written to NetCDF attrs, but the `mol_modify` emission lines are commented out — **silent no-ops**. |
| B3 | `core.py:272` | `override_surface_temperature or config…` — `or` swallows falsy values; inconsistent with albedo's `is not None` check. |
| B4 | `interface.py:327–330` + `core.py:294–297` | Per-point NaN values pass the `is not None` check: `albedo nan` / `sur_temperature nan` written to input file; `int(nan)` for surface type raises and the point silently becomes NaN. |
| B5 | `interface.py:770–773` | Scalar (0-d) `altitude` coordinate → `len()` raises `TypeError`. |
| B6 | `interface.py:465–502` | ~40 lines of leftover editing-monologue comments around the fragile 10-tuple. Remove (the tuple itself is replaced by `SimPoint`, Section 3.2). |
| B7 | `io.py:673–687` | `OutputParser._determine_output_type` trusts config altitude count, ignores actual data; per-point altitude override causes reshape mismatch → silent all-NaN results. Parser must infer altitude/wavelength structure from the data and the actual per-run zout. |
| B8 | `io.py:281–284` | ERA5 pressure-unit handling has two bare `if`s and no `else`; unknown unit → `NameError`. Also `profile_data.q.units` raises when attrs are stripped. Add explicit `else: raise InputGenerationError(...)` and attrs fallback. |
| B9 | `io.py:242` | Stray no-op expression statement `output_filepath`. |
| B10 | `config.py:536–544` | Dead default-backfill block (`f.default is not …[f.name].default` compares an object with itself). Delete. |
| B11 | `interface.py:301–307` | Comment claims "always regenerate" above a cache doing the opposite. Keep the cache, fix the comment. |
| B12 | `core.py:157–167` | Failed runs leave `.inp`/`.out` litter and their stderr is only logged. Superseded by failure reporting (Section 3.5): on failure, files are **deliberately kept** and referenced from the failure log. |
| B13 | `io.py:929` | `convert_batch` stamps the first point's metadata (its lat/lon/albedo) as global dataset attrs. Remove per-point fields from global attrs. |
| B14 | `interface.py:539` | Progress callback receives `(success_count, total)`; must be `(completed, total)`. |

## 3. Architecture

### 3.1 `pyradtran/params.py` (new) — parameter registry

```python
@dataclass(frozen=True)
class ParamSpec:
    keyword: str                  # uvspec keyword, e.g. "sur_temperature"
    dtype: type                   # float | int | str
    units: str | None             # "K", "deg", "km", …
    valid_range: tuple | None     # (lo, hi) inclusive, or None
    choices: tuple | None         # for enumerated params (e.g. rte_solver)
    applicability: str            # "solar" | "thermal" | "both"
    formatter: Callable | None    # value -> line body; default str()
    doc: str                      # one-line description
```

- `REGISTRY: dict[str, ParamSpec]` seeded with all parameters the package
  currently types: `albedo`, `sur_temperature`, `sza`, `zout`,
  `brdf_rpv_type` (implies `brdf_rpv_library IGBP`), `wavelength`,
  `mol_modify O3`, `mol_modify H2O`, `rte_solver`, `mol_abs_param`, `umu`,
  `output_user`, `source`, `day_of_year`.
- **Unknown keys are allowed** and passed through verbatim (escape hatch),
  tagged `unvalidated` in provenance.
- `class Var: name: str` — marker for a per-point dataset variable reference.
  A bare string value that matches a dataset variable name raises a
  `PyRadtranError` with a message pointing to `Var` (replaces the silent
  string-hijack magic).

### 3.2 `ParamResolver` — single resolution order

Layered, later wins:

1. Config defaults (typed `SimulationDefaults` fields mapped to registry
   keywords).
2. `params` literals (apply to every point).
3. `params` `Var()` entries (resolved per point from the stacked dataset).

Output per point: `ResolvedParams` — an ordered mapping
`keyword -> (formatted_value, provenance)` where provenance is one of
`package-default | master-config | sim-yaml | params-literal | dataset-var |
unvalidated`.

- Dotted keys (`section.field`) are applied to the config **and consumed** —
  they never reach the uvspec file (fixes B1).
- **NaN policy:** a per-point NaN resolves to *parameter omitted for that
  point*, and the omission is recorded in the point's status (Section 3.5).
  Fixes B4.
- **Validation:** literal and resolved values are checked against
  `valid_range`/`choices` at submit time; a violation raises before any
  uvspec process is spawned.
- The 10-element point tuple is replaced by a `SimPoint` dataclass:
  `(index, time, latitude, longitude, resolved_params, era5_file, point_id)`.

**Backwards compatibility:** `albedo_var=…`, `surface_temperature_var=…`,
`surface_type_var=…`, `altitude_var=…`, `override_albedo=…`, etc. remain as
thin shims that translate to `params` entries and emit `DeprecationWarning`.
The existing test suite must stay green through the shims.

### 3.3 `InputFileBuilder` — extracted from `Simulation._generate_input_content`

- Builds the ordered line list from config + `ResolvedParams`; every line
  carries its provenance tag.
- Re-enables `ozone_du` → `mol_modify O3 <v> DU` and `h2o_mm` →
  `mol_modify H2O <v> MM` when set (fixes B2).
- Replaces the `startswith(f"{key} ")` de-duplication with keyword-aware
  replacement that handles multi-word keywords (`mol_modify O3`).
- `Simulation` keeps its subprocess responsibilities; input generation moves
  to the builder.

### 3.4 `explain()` — dry run with provenance

- `ds.pyradtran.explain(point=None, params=None, config_path=None)` renders
  the exact input file for one point (default: first point) **without**
  running uvspec, annotated:

  ```
  albedo 0.85            # params (literal)
  sur_temperature 271.2  # dataset var 'skin_temp'
  mol_modify O3 300 DU   # sim yaml
  ```

- `Simulation.dry_run(...)` exposes the same at the low level.
- Implementation is a formatting pass over `InputFileBuilder` output — no new
  logic.

### 3.5 Failure reporting

- Result dataset gains a `status` data variable over the point dimensions:
  `0 = ok`, `1 = uvspec failure`, `2 = skipped (NaN inputs)`.
- On failure: `.inp` and `.out` files are kept regardless of
  `cleanup_temp_files`; uvspec stderr is captured into
  `working_dir/failures_<runid>.log` together with the input-file path,
  one block per failed point.
- End-of-run summary: single warning with failed/skipped counts and log path.
- Progress callback fixed to `(completed, total)` (B14).

### 3.6 `pyradtran/channels.py` (new) — instrument channels

- `convolve_channels(result_ds, srf, radiance_var=...) -> xr.Dataset` —
  `srf` is an `xr.Dataset`/`DataArray` with dims `(channel, wavelength)`;
  spectral results are interpolated onto the SRF wavelength grid and
  integrated: `L_ch = ∫ L(λ) φ_ch(λ) dλ / ∫ φ_ch(λ) dλ`.
- Thermal source additionally yields brightness temperature per channel via
  inverse Planck at the SRF-weighted central wavelength.
- `ds.pyradtran.run(channels=srf)` applies the convolution as a
  post-processing step and returns channel-space results (spectral results
  retained under a flag `keep_spectral=True`).

### 3.7 `ds.pyradtran.jacobian(...)` — perturbation kernels

- `ds.pyradtran.jacobian(param, delta, params=None, **run_kwargs)` runs the
  base batch and one perturbed batch (`param += delta`), returns
  `(perturbed − base) / delta` as a dataset with the same dims, attrs noting
  `jacobian_param`, `jacobian_delta`.
- `param` must be a scalar registry parameter (validated); reuses the batch
  machinery unchanged.

## 4. Error handling

- Validation errors (range/choices/unknown `Var` target) raise **before**
  process-pool submission, listing every offending key at once.
- Per-point runtime failures never raise from the batch; they are recorded in
  `status` + failure log (Section 3.5). `PyRadtranError` is still raised when
  *all* points fail.
- ERA5 generator: explicit `else` branch on unit dispatch raising
  `InputGenerationError` with the offending unit string (B8).

## 5. Testing

TDD throughout (test first, then implementation):

- `tests/test_params.py` — registry contents, `Var` semantics, resolver
  layering, NaN policy, dotted-key consumption, validation errors,
  deprecation shims.
- `tests/test_input_builder.py` — line generation, provenance tags,
  multi-word keyword replacement, `mol_modify` emission (B2 regression).
- `tests/test_explain.py` — annotated rendering, no subprocess spawned.
- `tests/test_failures.py` — status variable, kept temp files, failure log,
  callback signature (mock uvspec).
- `tests/test_channels.py` — analytic SRF (boxcar) convolution against a
  hand-computed integral; inverse Planck round-trip.
- `tests/test_jacobian.py` — linear mock forward model gives exact kernel.
- Existing suite (`tests/test_interface.py`, `test_config.py`, …) must pass
  unchanged via shims.

## 6. Implementation phases

1. **Phase 1:** registry + resolver + `InputFileBuilder` + all bug fixes
   (B1–B14) + shims. Largest phase; everything else builds on it.
2. **Phase 2:** `explain()` / dry run.
3. **Phase 3:** failure reporting (`status`, failure log, callback fix).
4. **Phase 4:** channels + jacobian.

Each phase lands with its tests green and the full existing suite passing.

## 7. Out of scope

- Result caching / point deduplication (explicitly deferred by user).
- Pydantic migration of the config system (Approach A chosen without it;
  `_dict_to_dataclass` is only cleaned, not replaced).
- Notebook rewrites beyond what deprecations require.
