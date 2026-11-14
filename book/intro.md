# pyRadtran

A Python interface for the [libRadtran](http://www.libradtran.org)
radiative transfer model: set up, run, and analyse solar and thermal
simulations directly from xarray datasets.

- **xarray-native** — one `ds.pyradtran.run()` call parallelises `uvspec`
  over every point in your dataset and returns a labelled Dataset
- **One parameter system** — literals, per-point `Var()` references, and
  config overrides in a single `params` mapping, validated against your
  installed libRadtran before anything runs
- **Real atmospheres** — standard profiles, IGRA radiosondes, ERA5
  reanalysis; parametric and file-based clouds
- **Post-processing built in** — instrument-channel convolution,
  brightness temperatures, finite-difference sensitivity kernels

## Three steps to a first result

```python
import pyradtran   # registers the .pyradtran accessor
import xarray as xr, pandas as pd

# 1. Describe *where and when* as an xarray dataset
ds = xr.Dataset(coords={
    'time': pd.date_range('2025-04-04', periods=24, freq='h'),
    'latitude': ('time', [61.0] * 24),
    'longitude': ('time', [22.0] * 24),
})

# 2. Run (config = *how*: solver, spectral range, outputs)
ds_sim = ds.pyradtran.run(config_path='config/spectral_config.yaml')

# 3. Analyse like any xarray dataset
ds_sim.eglo.sel(wavelength=550, method='nearest').plot()
```

## How this documentation is organised

| Section | What it is | Read it when… |
|---|---|---|
| **Getting Started** | installation, a ten-minute quickstart, and how the pieces fit together | you are new |
| **User Guide** | one page per topic — configuration, parameters, atmospheres, clouds, results, debugging — plus a cookbook of short recipes | you want to understand or look up *how* to do something |
| **Tutorials** | short notebooks, one concept each, grouped by topic | you learn best from a worked example |
| **Case Studies** | full research workflows: cloud forcing, thermal imaging, campaign validation, reanalysis processing | you want a template for real science |
| **Reference** | every function with formulas, examples and plots; example config files; changelog | you need the exact signature or behaviour |

## Where to go next

| I want to… | Go to |
|---|---|
| Install and configure paths | {doc}`installation` |
| Run my first simulation | {doc}`notebooks/quickstart` |
| See how the pieces fit together | {doc}`guide/concepts` |
| Understand configs vs params | {doc}`guide/configuration`, {doc}`guide/parameters` |
| Master the whole parameter system | {doc}`notebooks/parameters_deep_dive` |
| Add clouds / real atmospheres | {doc}`guide/clouds`, {doc}`guide/atmosphere` |
| Know what the output columns and units mean | {doc}`guide/outputs` |
| Copy a recipe | {doc}`examples/index` |
| Look up a function | {doc}`api/index` |
| Debug a weird result | {doc}`guide/debugging` |
| Browse all notebooks by topic | {doc}`notebooks/index` |
