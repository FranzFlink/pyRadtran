# Changelog

## 0.3.0

### Changed (breaking)

- `simulation_defaults.h2o_mm` and `simulation_defaults.ozone_du` now default to
  `null`. Previously every run without its own profile was rescaled to 2.0 mm of
  precipitable water and 300 DU of ozone (`mol_modify H2O 2.0 MM`,
  `mol_modify O3 300.0 DU`), whatever the atmosphere file contained. Runs now keep
  the columns of the atmosphere profile. Set the two keys explicitly to get the
  old behaviour.

### Fixed

- A `radiosonde` profile passed through `params` is no longer rescaled by the
  configured `h2o_mm` / `ozone_du`. Before, the scaling was skipped only for a
  profile set in the configuration (ERA5 file or `h2o_source: radiosonde`), so a
  measured humidity profile given in `params` was overwritten by the scalar
  column (with the old 2.0 mm default this raised the clear-sky direct
  irradiance at the surface by about 30 W m⁻² in a mid-latitude case). An
  explicit `mol_modify` in `params` is still written.

## 0.2.0

First release.
