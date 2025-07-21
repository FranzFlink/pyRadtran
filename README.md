# PyRadtran
![pyRadtran logo](logo.png)

A flexible and user-friendly Python wrapper for the libradtran radiative transfer model (`uvspec`), with seamless integration into the scientific Python ecosystem, particularly `xarray`.

## Features

- **Flexible Configuration**: Configure simulations via Python objects, dictionaries, or YAML files
- **Xarray Integration**: Run simulations directly from xarray Datasets with `ds.pyradtran.run_uvspec()`
- **Multi-level Output Support**: Handle multi-altitude simulations with proper dimensions/coordinates
- **Parallel Processing**: Run multiple simulations in parallel for different times/locations
- **Intelligent Input/Output**: Read from CSV/NetCDF, save results with metadata

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
import xarray as xr
import numpy as np
import pandas as pd

# Create a dataset with time, latitude, longitude
times = pd.date_range("2023-05-01", periods=24, freq="1H")
lats = np.linspace(60.0, 60.5, 5)
lons = np.linspace(10.0, 10.5, 5)

# Create a time series at fixed locations
coords = {
    "time": times,
    "latitude": ("time", np.full(len(times), 60.2)),
    "longitude": ("time", np.full(len(times), 10.3))
}

# Or create a grid with lat/lon dimensions
# coords = {
#     "time": times,
#     "latitude": lats,
#     "longitude": lons
# }

# Create the dataset
ds = xr.Dataset(coords=coords)

# Run uvspec simulations
result_ds = ds.pyradtran.run_uvspec(
    config_path="config/your_config.yaml",
    output_path="results/your_simulation.nc",
    return_dataset=True
)

# Explore and visualize results
result_ds.eglo.plot()
```

## Advanced Usage

### Override Parameters for Specific Simulations

```python
# Override specific parameters without changing the config file
result_ds = ds.pyradtran.run_uvspec(
    config_path="config/base_config.yaml",
    parameter_overrides={
        "simulation_defaults.albedo_value": 0.8,
        "execution.max_workers": 16
    }
)
```

### Parameter Studies

```python
import xarray as xr
import numpy as np
import pandas as pd
from pyradtran import load_config, execute_simulation_batch

# Load base configuration
config = load_config("config/base_config.yaml")

# Create a dataset with multiple albedo values
albedo_values = np.linspace(0.1, 0.9, 9)  # 0.1, 0.2, ..., 0.9
times = pd.date_range("2023-05-01 12:00", periods=1)

# Create coordinates
ms_ds = xr.Dataset(
    coords={
        "time": times,
        "latitude": 60.0,
        "longitude": 10.0,
        "albedo": albedo_values
    }
)

# Run separate simulations for each albedo
results = {}
for albedo in albedo_values:
    # Update config for this simulation
    config.simulation_defaults.albedo_value = albedo
    
    # Run simulation batch
    result = execute_simulation_batch(
        config=config,
        input_ds=ms_ds.sel(albedo=albedo)
    )
    
    # Store results for this albedo
    results[albedo] = result

# Combine results into a single dataset with albedo dimension
# (Implementation depends on exact structure of your results)
```

## Acknowledgments
* Emde, C., Buras-Schnell, R., Kylling, A., Mayer, B., Gasteiger, J., Hamann, U., Kylling, J., Richter, B., Pause, C., Dowling, T., and Bugliaro, L.: The libRadtran software package for radiative transfer calculations (version 2.0.1), Geosci. Model Dev., 9, 1647–1672, https://doi.org/10.5194/gmd-9-1647-2016, 2016. 
* Hoyer, S. & Hamman, J., (2017). xarray: N-D labeled Arrays and Datasets in Python. Journal of Open Research Software. 5(1), p.10. DOI: https://doi.org/10.5334/jors.148
