# Usage Guide

## Basic Concepts

pyRadtran provides a Pythonic interface to libRadtran through several key components:

- **Configuration**: A layered YAML-based system — built-in defaults → user master config → simulation-specific config
- **Interface**: An xarray accessor (`ds.pyradtran.run(...)`) for seamless integration
- **Simulation**: The core engine that generates libRadtran input files, runs `uvspec`, and parses the output
- **I/O**: Input/output handling for various data formats (ERA5, radiosondes, cloud profiles)

## Quick Example

```python
import pyradtran  # Registers the .pyradtran xarray accessor
import xarray as xr
import pandas as pd
from pathlib import Path

# Create an input dataset describing the simulation geometry
ds = xr.Dataset(
    coords={
        'time': pd.date_range('2025-04-04', periods=24, freq='h'),
        'latitude': ('time', [61.0] * 24),
        'longitude': ('time', [22.0] * 24),
        'altitude': ('altitude', [10]),
    }
)

# Run the simulation
ds_sim = ds.pyradtran.run(
    config_path=Path('config/spectral_config.yaml'),
)
```


## Configuration System

pyRadtran uses a **three-layer configuration system**:

1. **Built-in defaults** — loaded from `config/default_simulation.yaml` inside the package
2. **User master config** — `~/.pyradtran/config.yaml` overrides defaults (e.g., your local libRadtran paths)
3. **Simulation-specific config** — the YAML file you pass to `config_path=` overrides everything else

This means you only need to specify what's *different* from the defaults. A typical simulation config might only set the spectral range and solver:

```yaml
simulation_defaults:
  wavelength_nm:
    start: 400
    end: 770
  rte_solver: disort
  mol_abs_param: lowtran per_nm

execution:
  cleanup_temp_files: false   # Keep .inp files for debugging
  debug_mode: false
  max_workers: 4              # Parallel simulations
  timeout_seconds: 60
```

Paths like `libradtran_bin`, `libradtran_data`, `atmosphere_profile`, and `solar_spectrum` are typically set once in your master config (`~/.pyradtran/config.yaml`) and inherited by all simulations. See {doc}`installation` for setup instructions.

## Unified Parameter Passing: `params`

All runtime parameters go through a single `params` mapping. A key can be:

1. **A registry parameter** (`albedo`, `sza`, `sur_temperature`, `zout`, `brdf_rpv_type`, `mol_modify O3`, …) — the value is validated (type, physical range) before any simulation runs
2. **A raw uvspec keyword** (anything from the libRadtran manual, e.g. `crs_model`) — passed through unvalidated as an escape hatch
3. **A dotted config path** (`simulation_defaults.wavelength_nm`, `execution.max_workers`, …) — applied to the configuration instead of the input file

The value can be:

- **A literal** — the same value for every simulated point
- **`Var("name")`** — resolved per point from the variable `name` in your input dataset

```python
from pyradtran import Var

ds_sim = ds.pyradtran.run(
    config_path=Path('config/solar_config.yaml'),
    params={
        'albedo': Var('surface_albedo'),   # per-point from ds.surface_albedo
        'mol_modify O3': 320.0,            # same for every point, validated in DU
        'crs_model': 'rayleigh Bodhaine',  # raw uvspec keyword, passed through
        'simulation_defaults.wavelength_nm': [400, 700],  # config override
    },
)
```

Points where a `Var` value is NaN simply omit that parameter (the config
default applies); the coordinates themselves being NaN skips the point and
records `status=2` in the result.

### Validation against your libRadtran install

libRadtran ships a machine-readable description of every `uvspec` option
(names, argument types, valid ranges, allowed choices, documentation). On
first use pyRadtran extracts it from your local installation and caches it
under `~/.pyradtran/`, so every `params` entry is checked against the exact
binary you run — *before* any simulation starts:

```python
ds.pyradtran.run(config_path=cfg, params={'albdeo': 0.3})
# ValidationError: 'albdeo' is not a known uvspec option of the local
# libRadtran install; did you mean albedo / albedo_map?

ds.pyradtran.run(config_path=cfg, params={'ic_properties': 'granite'})
# ValidationError: 'ic_properties': 'granite' is not one of
# ['baum', 'baum_v36', 'echam4', 'fu', 'hey', 'key', ...]
```

Repeatable options take a list — one input line per entry:

```python
params={'wc_modify': ['tau550 set 12', 'ssa set 0.99']}
```

Flag options take `True` (`params={'aerosol_default': True}` emits the bare
keyword). To bypass validation (e.g. a patched uvspec build with custom
options), wrap the value in `Raw`:

```python
from pyradtran import Raw
params={'my_custom_option': Raw('anything goes')}
```

If no libRadtran source tree is found next to your `uvspec` binary,
validation falls back to the built-in registry and unknown keys pass
through unvalidated, as before.

### The libRadtran manual, in Python

`describe()` prints the usage signature, valid choices/ranges, option
dependencies, and the full documentation text for any option;
`search_options()` greps all of it:

```python
import pyradtran

print(pyradtran.describe('wc_modify'))
# wc_modify <gg|ssa|tau|tau550> <set|scale> <float>
# group: Water and ice clouds   (repeatable)
# requires: wc_file
# ...

pyradtran.search_options('optical thickness')
# ['aerosol_modify', 'ic_modify', 'wc_modify', ...]
```

```{tip}
The [libRadtran manual](https://www.libradtran.org/doc/libRadtran.pdf) is still worth reading for the physics — but for signatures and spellings, `pyradtran.describe()` shows you exactly what *your* installed version accepts.
```

```{note}
The old `parameter_overrides=`, `albedo_var=`, `surface_temperature_var=`,
`surface_type_var=` and `altitude_var=` keyword arguments still work but are
deprecated — they emit a `DeprecationWarning` and translate onto `params`
internally. Migrate with `albedo_var="x"` → `params={"albedo": Var("x")}` and
`parameter_overrides={...}` → `params={...}`.
```

### Previewing the input file: `explain()`

`ds.pyradtran.explain()` renders the exact uvspec input file for one point —
without running anything — and annotates every line with the layer it came
from (`config`, `params-literal`, `dataset-var`, `unvalidated`):

```python
print(ds.pyradtran.explain(
    point={'time': ds.time[0]},
    params={'albedo': Var('surface_albedo')},
    config_path=config_path,
))
```

### Output columns and axes

The `output_user` line is derived from `output_columns` in your config (or a
per-run `params={'output_user': ...}` override). pyRadtran automatically adds
the `lambda` column for spectral runs and the `zout` column for multi-altitude
runs: the output parser needs them to reconstruct the `wavelength` and
`altitude` axes, so you no longer have to remember to list them yourself.

### Failure reporting

Every result carries a per-point `status` variable (`0` ok, `1` uvspec
failure, `2` skipped due to NaN coordinates). When at least one point fails, a
`failures_<timestamp>.log` file with the captured stderr and the kept `.inp`
path is written to the working directory.

## Instrument Channels & Brightness Temperature

Convolve spectral results with instrument spectral response functions (SRFs),
either as post-processing or directly in `run()`:

```python
import xarray as xr
from pyradtran import convolve_channels, brightness_temperature

# srf: DataArray with dims (channel, wavelength), wavelength in nm
channel_ds = ds.pyradtran.run(config_path=config_path, channels=srf)

# thermal radiances -> brightness temperature (uvspec default units)
tb = brightness_temperature(channel_ds['uu'], wavelength_nm=10500.0)
```

Pass `keep_spectral=True` to retain the original spectral variables as
`<name>_spectral` alongside the channel-averaged ones.

## Sensitivity Kernels: `jacobian()`

`ds.pyradtran.jacobian(param, delta)` runs the batch twice (base and
perturbed) and returns the finite-difference kernel
`(perturbed - base) / delta`:

```python
jac = ds.pyradtran.jacobian(
    'albedo', 0.01,
    params={'albedo': 0.5},   # base value; falls back to the config default
    config_path=config_path,
    show_progress=False,
)
jac['eup']   # d(eup)/d(albedo), same dims as a normal result
```

The perturbed parameter must be a scalar (a `params` literal or a config
default) — perturbing a per-point `Var` is rejected.


## Working with Results

Results are returned as xarray Datasets with full metadata:

```python
# Access simulation results
direct_radiation = ds_sim.edir     # Direct irradiance
diffuse_radiation = ds_sim.edn     # Diffuse downwelling
upwelling = ds_sim.eup              # Upwelling irradiance (if computed)

# Plot results
import matplotlib.pyplot as plt
ds_sim.edir.plot()
plt.show()
```

For spectral simulations, results include a `wavelength` dimension:

```python
# Spectral analysis
ds_sim.edir.sel(wavelength=550, method='nearest').plot()
```

## Cloud Simulations

pyRadtran supports several approaches for including clouds:

1. **Parametric clouds via `params`**: Pass cloud properties directly (see the [Water Cloud](notebooks/water_cloud) notebook)
2. **Cloud files**: Generate cloud profile files and pass them via `wc_file` / `ic_file`
3. **Automated cloud generation**: Pass cloud variables in your dataset using `cloud_wc_var`, `cloud_top_var`, etc.

## ERA5 Atmosphere Profiles

You can replace the standard atmospheric profile with ERA5 reanalysis data:

```python
ds_sim = ds.pyradtran.run(
    config_path=config_path,
    era5_atmosphere=ds_era5,  # xarray Dataset with ERA5 profiles
)
```

See the [ERA5 Atmosphere](notebooks/era5_atmosphere) notebook for a complete example.

## Batch Processing & Parallel Execution

pyRadtran parallelizes naturally over all points in your input dataset. Set `max_workers` in your config:

```yaml
execution:
  max_workers: 8  # Use 8 parallel processes
```

For large batch jobs, you can monitor progress with a callback:

```python
def my_progress(current, total):
    print(f"{current}/{total} simulations complete")

ds_sim = ds.pyradtran.run(
    config_path=config_path,
    progress_callback=my_progress,
)
```

```{tip}
Always check your input file when debugging! See {ref}`check-your-input-file` for guidance
on inspecting the generated `.inp` files.
```