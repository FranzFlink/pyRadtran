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

### Using `parameter_overrides`

You can override individual libRadtran parameters at runtime without changing your YAML config. This is especially useful for parameter sweeps or injecting cloud files:

```python
ds_sim = ds.pyradtran.run(
    config_path=Path('config/solar_config.yaml'),
    parameter_overrides={
        'albedo': 0.8,
        'wc_file': '1D path/to/cloud_file.dat',
    },
)
```

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

1. **Parametric clouds via `parameter_overrides`**: Pass cloud properties directly (see the [Water Cloud](notebooks/water_cloud) notebook)
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