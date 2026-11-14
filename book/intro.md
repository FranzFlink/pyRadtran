# pyRadtran

A Python interface for the [libRadtran](http://www.libradtran.org)
radiative transfer model: set up, run and analyse solar and thermal
simulations from xarray datasets.

- **xarray in, xarray out**: `ds.pyradtran.run()` runs `uvspec` in
  parallel for every point of a dataset and returns a labelled Dataset
- **One parameter mapping**: literals, per-point `Var()` references and
  config overrides go through `params`, validated against the installed
  libRadtran before anything runs
- **Atmospheres and clouds**: standard profiles, IGRA radiosondes, ERA5
  reanalysis; parametric and file-based clouds
- **Post-processing**: instrument-channel convolution, brightness
  temperatures, finite-difference sensitivity kernels

## A first result

```python
import pyradtran   # registers the .pyradtran accessor
import xarray as xr, pandas as pd

# 1. Where and when: an xarray dataset
ds = xr.Dataset(coords={
    'time': pd.date_range('2025-04-04', periods=24, freq='h'),
    'latitude': ('time', [61.0] * 24),
    'longitude': ('time', [22.0] * 24),
})

# 2. How: the config sets solver, spectral range and outputs
ds_sim = ds.pyradtran.run(config_path='config/spectral_config.yaml')

# 3. The result is an xarray dataset
ds_sim.eglo.sel(wavelength=550, method='nearest').plot()
```

## Contents

| Section | What is in it |
|---|---|
| **Getting Started** | {doc}`installation`, the {doc}`quickstart <notebooks/quickstart>` notebook, and {doc}`how pyRadtran works <guide/concepts>` |
| **User Guide** | one page per topic: {doc}`guide/configuration`, {doc}`guide/parameters`, {doc}`guide/atmosphere`, {doc}`guide/clouds`, {doc}`guide/outputs`, {doc}`guide/debugging`; short recipes in the {doc}`cookbook <examples/index>` |
| **Tutorials** | short notebooks, one concept each ({doc}`overview <notebooks/index>`) |
| **Case Studies** | research workflows: cloud forcing, thermal imaging, campaign validation, reanalysis processing |
| **Reference** | {doc}`API reference <api/index>` with formulas, examples and plots; example config files; contributing |

## Citing and licence

Please cite the archived pyRadtran release (see `CITATION.cff` in the
repository) and libRadtran: Emde et al. (2016),
[doi:10.5194/gmd-9-1647-2016](https://doi.org/10.5194/gmd-9-1647-2016),
and Mayer & Kylling (2005),
[doi:10.5194/acp-5-1855-2005](https://doi.org/10.5194/acp-5-1855-2005).

pyRadtran is MIT-licensed and independent of the libRadtran project.
libRadtran (GPL-2) is not bundled and is installed separately.
