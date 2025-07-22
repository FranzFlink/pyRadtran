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
simulation:
  solar_zenith_angle: 45.0
  wavelength:
    start: 400
    end: 3200
    resolution: 10
  
paths:
  libradtran_path: /path/to/libradtran
  data_files_path: /path/to/data
  
execution:
  solver: disort
  streams: 8
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
