# PyRadtran
![pyRadtranlogo](logo.png)


[![Documentation](https://img.shields.io/badge/docs-latest-blue)](https://franzflink.github.io/pyRadtran/)
[![License](https://img.shields.io/github/license/FranzFlink/pyRadtran)](https://github.com/FranzFlink/pyRadtran/blob/main/LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/FranzFlink/pyRadtran)](https://github.com/FranzFlink/pyRadtran)

A flexible and user-friendly Python wrapper for the libradtran radiative transfer model (`uvspec`), with seamless integration into `xarray`.

## 📖 Documentation

**[View the full documentation on GitHub Pages →](https://franzflink.github.io/pyRadtran/)**

- [Installation Guide](https://franzflink.github.io/pyRadtran/installation.html)
- [Usage Examples](https://franzflink.github.io/pyRadtran/usage.html)  
- [API Reference](https://franzflink.github.io/pyRadtran/api.html)
- [Jupyter Notebooks](https://franzflink.github.io/pyRadtran/notebooks.html)

## Features

- **Flexible Configuration**: Configure simulations via Python objects, dictionaries, or YAML files
- **Xarray Integration**: Just run your configured simulation on an xarray dataset, that contains all the parameters of the simulation: `ds.pyradtran.run_uvspec()`
- **Multi-level Output Support**: Simulate spectral solar/thermal simulations on multiple altitudes, for any space-time vector with one line!
- **Parallel Processing**: Run multiple simulations in parallel for different times/locations
- **Intelligent Input/Output**: The wrapper automatically detects the exptected output of the simulation, which is returned as an xarray dataset.
  
## Installation

### Prerequisites

- Python 3.8+
- libradtran (uvspec) installed and accessible on your system
- Common scientific Python packages (numpy, pandas, xarray)

### Installing from Source

```bash
git clone https://github.com/FranzFlink/pyradtran.git
cd pyradtran
pip install -e .
```

## Basic Usage

### Simple Example

```python
from pyradtran.interface import PyRadtranAccessor  # This should register the accessor automatically
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from pathlib import Path
import pandas as pd

# Load the configuration from the YAML file
config_path = Path('../config/spectral_config.yaml')
import logging
# change to "DEBUG" if you want to see more output
logging.getLogger('pyradtran').setLevel(logging.CRITICAL)

N_timesteps = 24

# stationary simulation at constant altitude, latitude, and longitude but with varying time
ds = xr.Dataset(
    coords={
        'time' : pd.date_range('2025-04-04', periods=N_timesteps, freq='h'),
        'latitude' : ('time', [61.0] * N_timesteps),
        'longitude' : ('time', [22.0] * N_timesteps),
        'altitude' : ('altitude', [10]),
    }
)

# Run a spectral simulation for a single point
ds_sim = ds.pyradtran.run_uvspec(
    config_path=config_path,
    return_dataset=True,
    save_to_file=True,
)

# explore the results
print(ds_sim)
```

## Advanced Usage

... under construction ...


## Acknowledgments
* Emde, C., Buras-Schnell, R., Kylling, A., Mayer, B., Gasteiger, J., Hamann, U., Kylling, J., Richter, B., Pause, C., Dowling, T., and Bugliaro, L.: The libRadtran software package for radiative transfer calculations (version 2.0.1), Geosci. Model Dev., 9, 1647–1672, https://doi.org/10.5194/gmd-9-1647-2016, 2016. 
* Hoyer, S. & Hamman, J., (2017). xarray: N-D labeled Arrays and Datasets in Python. Journal of Open Research Software. 5(1), p.10. DOI: https://doi.org/10.5334/jors.148
