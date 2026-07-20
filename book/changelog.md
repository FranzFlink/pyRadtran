# Changelog

## Version 0.2.0 (Upcoming)

### ERA5 Integration (`pyradtran.era5`)
- `era5_atmosphere=` accepts raw ERA5 datasets as-is: CDS NetCDF short
  names and ARCO-ERA5 Zarr long names (e.g. straight from gcsfs) are
  normalised automatically (`normalize_era5`) — no manual renaming or
  unit attributes needed; `millibars`/`Pa` pressure coordinates handled
- Ozone from ERA5: an `o3` field is written into the radiosonde-style
  atmosphere file (`radiosonde … H2O MMR O3 MMR`); atmosphere files
  carry a machine-readable `# columns:` header that the input builder
  reads back
- A gas profile provided by the radiosonde file now disables the
  corresponding `mol_modify` scalar (`ozone_du`, `h2o_mm`) — the
  measured/reanalysis column is no longer silently rescaled
- `era5_clouds=True` on `run()`: per-point `wc_file` / `ic_file` cloud
  profiles from ERA5 `clwc`/`ciwc` (kg/kg → g/m³, altitudes from
  geopotential, correct 1D cloud-file layer semantics); pass a dict to
  set effective radii. Explicit `params` clouds or `cloud_*_var`
  kwargs take precedence per point, so variation-axis studies compose
  with ERA5 atmospheres
- `recommend_atmosphere(lat, time)`: AFGL background profile matching
  latitude band and season
- NaN detection for the known twostr instability with optically thick
  clouds, with a warning that points to `rte_solver disort`
- New user-guide page: *Atmosphere & ERA5* (when to use radiosonde vs
  atmosphere profiles, ozone interplay, gcsfs workflow)

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

### Provenance
- Result datasets (and saved NetCDF files) record what produced them:
  package version + history line, the `params` mapping as JSON, the full
  merged config as YAML, the uvspec binary path, and the annotated input
  file of the first point (`pyradtran_input_example`)
- `wavelength`/`altitude` coordinates carry units attrs; channel runs
  record the channel names

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
- **API reference actually renders**: `only_build_toc_files: true` had
  excluded every autosummary-generated page, leaving all API links dead;
  the book now builds the full reference (and the build is warning-free)
- `save_master_config()` documented as the primary setup path; catalogue
  short names (`afglms`, `NewGuey2003`, …) documented for
  `atmosphere_profile` / `solar_spectrum`
- Restructured the book: slim landing page with a "where to go next" map,
  a four-page User Guide (configuration / parameters / results /
  debugging), tutorials separated from campaign case studies
- New executed deep-dive notebook covering the whole parameter system:
  layers and provenance, validation, `describe()`, repeatable options,
  `Raw`, status codes, channels, and jacobians
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

### Bug Fixes (deep audit)
- **`CloudGenerator.from_era5_dataset` array misalignment**: with
  ascending ERA5 pressure levels, altitudes/pressure were reversed but
  the cloud-content, cloud-cover, and temperature arrays were not — a
  surface cloud could come out at stratospheric altitude with a wrong
  density conversion. All per-level arrays are now reordered together
- **Geopotential unit detection**: profiles topping out below ~10 km were
  misread as height-in-metres (10× altitude error). The `z` units
  attribute now decides the conversion; the magnitude heuristic is only a
  fallback (`era5.cloud_profiles` and `clouds.py`)
- **`OutputToXarray.convert`** crashed for any two-dimensional result
  (spectral single-altitude, integrated multi-altitude); reshape now
  follows the present axes. `convert_batch` results are transposed so
  input dims come first: `(time, ..., wavelength, altitude)`
- **`zout` ordering**: uvspec hard-errors on unsorted output altitudes.
  Dataset-provided altitude coordinates, per-run `zout` overrides, and
  the config line are now sorted and de-duplicated on every path
- **NaT / missing coordinates**: a NaT time crashed the whole batch
  (including inside xarray's stacked-index handling); such points are now
  skipped with `status=2` like NaN coordinates, and missing
  time/latitude/longitude variables raise one clear `ValueError` up front
- **Config-level `parameter_overrides` reach the output parser**: an
  `output_user`/`zout`/`output_quantity` set through the config escape
  hatch shaped the input file but not the parser, silently scrambling
  column names
- **Brightness runs**: the `albedo` output column is now filtered in the
  shared column function, so builder and parser stay aligned wherever
  `albedo` sits in the column list
- `inspect_cloud_file()` shares the batch driver's cloud construction:
  NaN inputs return a message instead of crashing, and mixed-phase
  points preview both profiles
- `jacobian()` no longer differentiates the `status` flag (worst status
  of the two runs is kept); `max_workers: null` no longer breaks saving

### Error Safety
- `run(config=...)` copies the config — caller objects are never mutated
  by per-run adjustments or dotted overrides (`SimulationConfig.copy()`)
- `altitude` as a data variable no longer emits a spurious
  `DeprecationWarning`
- `cleanup_temp_files: true` now also removes parsed `.out` files;
  failed runs keep their generated cloud files for post-mortem
- `era5_clouds={}` means "enabled with defaults" instead of silently off
- `convolve_channels()` raises a clear error for SRFs that do not
  overlap the result wavelength grid (previously silent inf/NaN)
- `PathsConfig` accepts plain strings; `wavelength_nm` accepts the
  `{start:, end:}` YAML spelling
- Library no longer calls `logging.basicConfig()` (host logging config
  untouched); exceptions chain their cause (`raise ... from e`) and
  pyRadtran errors are no longer double-wrapped
- Stale libRadtran schema caches under `~/.pyradtran/` are pruned when a
  new one is written

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
