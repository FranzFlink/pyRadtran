# PyRadtran

A flexible and user-friendly Python wrapper for the libradtran radiative transfer model (`uvspec`), with seamless integration into the scientific Python ecosystem, particularly `xarray`.

## Features

- **Flexible Configuration**: Configure simulations via Python objects, dictionaries, or YAML files
- **Xarray Integration**: Run simulations directly from xarray Datasets with `ds.pyradtran.run_uvspec()`
- **Multi-level Output Support**: Handle multi-altitude simulations with proper dimensions/coordinates
- **Parallel Processing**: Run multiple simulations in parallel for different times/locations
- **Cloud & Aerosol Support**: Define complex atmospheric conditions with clouds and aerosols
- **Intelligent Input/Output**: Read from CSV/NetCDF, save results with metadata

## Installation

### Prerequisites

- Python 3.8+
- libradtran (uvspec) installed and accessible on your system
- Common scientific Python packages (numpy, pandas, xarray)

### Installing from Source

```bash
git clone https://github.com/yourusername/pyradtran.git
cd pyradtran
pip install -e .
```

## Basic Usage

### Simple Example

```python
import xarray as xr
from pyradtran import run_pyradtran_simulation

# Run from a CSV/NetCDF file with time, latitude, longitude
result_path = run_pyradtran_simulation(
    input_file="your_input_data.csv",
    config_path="config/your_config.yaml"
)
print(f"Results saved to: {result_path}")

# Load and explore results
ds = xr.open_dataset(result_path)
print(ds)
```

### Using the xarray Accessor

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

### Configuration

Create a YAML configuration file:

```yaml
# config/my_simulation.yaml
paths:
  libradtran_bin: /path/to/uvspec
  libradtran_data: /path/to/libRadtran/data
  atmosphere_profile: /path/to/atmosphere.dat
  solar_spectrum: /path/to/solar.dat
  radiosonde_base: /path/to/radiosondes/
  output_dir: ./results
  working_dir: ./temp

simulation_defaults:
  rte_solver: disort
  mol_abs_param: reptran coarse
  wavelength_nm: [280, 2800]
  output_columns:
    - sza
    - edir
    - eglo
    - edn
    - eup
    - enet
    - albedo
  output_altitudes_km: [0.0, 1.0, 2.0, 3.0, 5.0, 10.0]
  
  # Surface properties
  albedo_type: const
  albedo_value: 0.3
  surface_temperature_k: 273.15
  
  # Cloud properties
  clouds:
    enabled: true
    cloud_optical_properties: mie
    cloud_overlap: max-random
    layer_heights_km:
      - [1.0, 2.0]
    layer_water_content:
      - 0.1
    layer_effective_radius_um:
      - 10.0
  
  # Aerosol properties
  aerosols:
    enabled: true
    aerosol_type: rural
    aerosol_visibility_km: 23.0
    aerosol_optical_properties: default

execution:
  max_workers: 8
  cleanup_temp_files: true
  debug_mode: false
  timeout_seconds: 300

output:
  filename_prefix: rtm_sim
  filename_suffix: _results.nc
  netcdf_encoding:
    zlib: true
    complevel: 5
```

## Advanced Usage

### Override Parameters for Specific Simulations

```python
# Override specific parameters without changing the config file
result_ds = ds.pyradtran.run_uvspec(
    config_path="config/base_config.yaml",
    parameter_overrides={
        "simulation_defaults.albedo_value": 0.8,
        "simulation_defaults.clouds.enabled": False,
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

## Contributing

Contributions are welcome! Please feel free to submit pull requests.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

* The libradtran team for their excellent radiative transfer model
* The xarray developers for their powerful data structures