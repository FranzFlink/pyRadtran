# Changelog

## Version 0.2.0 (Upcoming)

### Unified Parameter Handling
- New `params` mapping on `run()` / `execute_simulation_batch()`: one place
  for registry parameters (validated), raw uvspec keywords (escape hatch), and
  dotted config overrides (`simulation_defaults.wavelength_nm`)
- `Var("name")` marker resolves a parameter per point from a dataset variable;
  NaN values skip the parameter for that point
- Parameter registry (`pyradtran.params.REGISTRY`) with type, unit, and range
  validation before any simulation runs
- `ds.pyradtran.explain()` / `Simulation.dry_run()`: preview the exact uvspec
  input file with per-line provenance annotations (no simulation run)
- Deprecated (still working, with `DeprecationWarning`): `parameter_overrides=`,
  `albedo_var=`, `surface_temperature_var=`, `surface_type_var=`,
  `altitude_var=`, and the `Simulation.run_simulation(override_*=...)` kwargs

### Schema-Backed Validation (from your libRadtran install)
- pyRadtran extracts libRadtran's own machine-readable option schema
  (~244 options: argument types, valid ranges, choices, dependencies,
  documentation) from the local installation and caches it under
  `~/.pyradtran/` — validation always matches the exact binary in use
- Unknown option names are rejected before any simulation runs, with
  did-you-mean suggestions; enumerated choices and numeric ranges are
  enforced (`ic_properties`, `wc_modify`, `cloudcover`, ...)
- Repeatable options accept list values (one input line per entry);
  flag options accept `True`; `Raw(value)` bypasses validation
- `pyradtran.describe(name)` / `pyradtran.search_options(text)`: the
  libRadtran manual — signatures, choices, docs — at the Python prompt

### Instrument Channels & Jacobians
- `convolve_channels()` / `run(channels=srf)`: SRF-average spectral results
  onto a `channel` dimension (trapezoidal, pure numpy)
- `brightness_temperature()`: inverse-Planck conversion of thermal radiances
- `ds.pyradtran.jacobian(param, delta)`: finite-difference sensitivity kernels
  via paired batch runs

### Robustness & Failure Reporting
- Per-point `status` variable (0 ok / 1 failed / 2 skipped) in every result
- `failures_<timestamp>.log` with captured stderr; failed runs keep their
  `.inp` file for post-mortem
- **Output columns are now self-consistent**: the `output_user` line and the
  output parser derive from the same code path; `lambda` (spectral) and `zout`
  (multi-altitude) columns are injected automatically. Previously, a spectral
  config without an explicit `lambda` column produced silently all-NaN results
- Per-run `output_user` and `zout` overrides are honoured by the output parser
- `OutputParser` honours per-run altitudes over config altitudes
- ERA5 humidity unit (RH vs MMR) detected from the atmosphere file header

### Documentation & Notebooks
- Overhauled all 21 Jupyter Book notebooks: standardized titles, structure, and API usage
- Fixed broken TOC references and added 11 previously orphaned notebooks to the book
- Filled the "Check your input file!" section with detailed debugging guidance
- Expanded installation guide with `~/.pyradtran/config.yaml` master config documentation
- Expanded usage guide with sections on clouds, ERA5, batch processing, and `parameter_overrides`
- Overhauled README with configuration section, CI badge, and corrected examples

### API Standardization
- Standardized all examples to use `ds.pyradtran.run(config_path=...)` as the canonical API
- Made libRadtran paths version-agnostic in all YAML configs

### CI/CD
- Added GitHub Actions workflow for unit tests (Python 3.9, 3.10, 3.11)
- Added Jupyter Book build checks

### Bug Fixes
- Fixed various typos in documentation and notebooks
- Unified Python version requirement to `>= 3.9` across all documentation

## Version 0.1.0 (Development)

- Initial release
- Basic pyRadtran functionality with `uvspec` wrapper
- xarray integration via `.pyradtran` accessor
- YAML configuration system with layered defaults
- Parallel simulation execution
- Jupyter notebook examples
