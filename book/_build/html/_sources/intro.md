# PyRadtran Documentation

A Python interface for the libRadtran Radiative Transfer Model.

PyRadtran provides a Pythonic interface to the libRadtran radiative transfer model, which is used 
for modeling radiation transport in Earth's atmosphere. This package aims to make it easier to 
set up, run, and analyze radiative transfer simulations with support for both solar and thermal infrared radiation.

## Key Features

- **Solar & Thermal Simulations** : Support for both solar radiation (UV, visible, near-IR) and thermal infrared simulations, spectral and broadband.
- **Flexible Configuration** : YAML-based configuration system
- **xarray Integration** : Native support for xarray datasets with automatic coordinate handling and metadata preservation
- **Atmospheric Profiles** : Support for standard atmospheric profiles, radiosondes, and ERA5 reanalysis data
- **Multi-core Processing** : Support for running simulations in parallel using multiple CPU cores.

## Quick Start

Here's a simple example to get you started:

```python
from pyradtran.interface import PyRadtranAccessor
import xarray as xr
from pathlib import Path
import pandas as pd

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

## Check your input file! 

