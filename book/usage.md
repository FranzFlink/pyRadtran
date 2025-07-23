# Usage Guide

## Basic Concepts

PyRadtran provides a Pythonic interface to libRadtran through several key components:

- **Configuration**: YAML-based configuration system
- **Interface**: xarray accessor for seamless integration
- **Simulation**: Core simulation engine
- **I/O**: Input/output handling for various data formats

## Quick Example

```python
from pyradtran.interface import PyRadtranAccessor
import xarray as xr
import pandas as pd
from pathlib import Path

# Load configuration
config_path = Path('config/spectral_config.yaml')

# Create input dataset
ds = xr.Dataset(
    coords={
        'time': pd.date_range('2025-04-04', periods=24, freq='h'),
        'latitude': ('time', [61.0] * 24),
        'longitude': ('time', [22.0] * 24),
        'altitude': ('altitude', [10]),
    }
)

# Run simulation
ds_sim = ds.pyradtran.run_uvspec(
    config_path=config_path,
    return_dataset=True,
    save_to_file=True,
)
```

## Configuration System

PyRadtran uses YAML configuration files to define simulation parameters:

```yaml
simulation_defaults:
  wavelength_nm:
    start: 400
    end: 770
  viewing geometry: nadir
  rte_solver: disort
  mol_abs_param: lowtran per_nm
  
paths:
  atmosphere_profile: /opt/libradtran/2.0.4/share/libRadtran/data/atmmod/afglsw.dat
  libradtran_bin: /opt/libradtran/2.0.4/bin/uvspec
  libradtran_data: /opt/libradtran/2.0.4/share/libRadtran/data
  output_dir: work
  radiosonde_base: null
  solar_spectrum: /opt/libradtran/2.0.4/share/libRadtran/data/solar_flux/NewGuey2003.dat
  working_dir: work
  
execution:
  cleanup_temp_files: false
  debug_mode: false
  max_workers: 1 # Number of parallel workers
  timeout_seconds: 60
```

## Working with Results

Results are returned as xarray Datasets with full metadata:

```python
# Access simulation results
direct_radiation = ds_sim.edir
diffuse_radiation = ds_sim.edn
albedo = ds_sim.albedo

# Plot results
import matplotlib.pyplot as plt
ds_sim.edir.plot()
plt.show()
```