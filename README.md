# PyRadtran
<img src="logo.png" alt="pyRadtran logo" width="300">

[![Documentation](https://img.shields.io/badge/docs-latest-blue)](https://franzflink.github.io/pyRadtran/)
[![License](https://img.shields.io/github/license/FranzFlink/pyRadtran)](https://github.com/FranzFlink/pyRadtran/blob/main/LICENSE)
[![Tests](https://github.com/FranzFlink/pyRadtran/actions/workflows/tests.yml/badge.svg)](https://github.com/FranzFlink/pyRadtran/actions/workflows/tests.yml)

A flexible and user-friendly Python wrapper for the [libRadtran](http://www.libradtran.org) radiative transfer model (`uvspec`), with seamless integration into [xarray](https://xarray.dev).

## Documentation

**[View the full documentation →](https://franzflink.github.io/pyRadtran/)**

- [Installation Guide](https://franzflink.github.io/pyRadtran/installation.html)
- [Configuration](https://franzflink.github.io/pyRadtran/guide/configuration.html) · [Parameters](https://franzflink.github.io/pyRadtran/guide/parameters.html) · [Results](https://franzflink.github.io/pyRadtran/guide/outputs.html) · [Debugging](https://franzflink.github.io/pyRadtran/guide/debugging.html)
- [Interactive Notebooks](https://franzflink.github.io/pyRadtran/notebooks/index.html)
- [API Reference](https://franzflink.github.io/pyRadtran/api/index.html)

## Features

- **Flexible Configuration**: Configure simulations via YAML files, Python objects, or dictionaries — with a layered config system that supports user defaults via `~/.pyradtran/config.yaml`
- **xarray Integration**: Run your configured simulation on an xarray dataset containing all simulation parameters: `ds.pyradtran.run()`
- **Multi-level Output**: Simulate spectral solar/thermal irradiance at multiple altitudes for any space-time vector — in a single call
- **Parallel Processing**: Run multiple simulations in parallel for different times, locations, or parameter sweeps
- **Intelligent I/O**: The wrapper automatically detects the expected output format and returns results as an xarray `Dataset`

## Installation

### Prerequisites

- Python 3.9+
- [libRadtran](http://www.libradtran.org) installed and accessible on your system
- Common scientific Python packages (numpy, pandas, xarray)

### Install from Source

```bash
git clone https://github.com/FranzFlink/pyRadtran.git
cd pyRadtran
pip install -e .
```

## Configuration

pyRadtran uses a layered configuration system:

1. **Built-in defaults** are loaded automatically
2. **User master config** at `~/.pyradtran/config.yaml` overrides defaults (set your local libRadtran paths here once)
3. **Simulation-specific YAML** overrides everything else for a given run

Create your master config to point to your local libRadtran installation:

```bash
mkdir -p ~/.pyradtran
cat > ~/.pyradtran/config.yaml << 'EOF'
paths:
  libradtran_bin: /path/to/your/libradtran/bin/uvspec
  libradtran_data: /path/to/your/libradtran/share/libRadtran/data
  atmosphere_profile: afglms
  solar_spectrum: NewGuey2003
EOF
```

## Quick Start

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

# Explore the results
print(ds_sim)
ds_sim.edir.plot()
```

Runtime parameters go through the unified `params` mapping — literals apply to
every point, `Var("name")` resolves per point from your dataset, and dotted
keys override the configuration:

```python
from pyradtran import Var

ds_sim = ds.pyradtran.run(
    config_path=Path('config/spectral_config.yaml'),
    params={
        'albedo': Var('surface_albedo'),               # per-point value
        'mol_modify O3': 320.0,                        # validated literal
        'simulation_defaults.wavelength_nm': [400, 700],  # config override
    },
)
```

Preview the generated input file (with per-line provenance) without running
anything via `ds.pyradtran.explain()`.

> **Tip:** Always check your input file! Set `cleanup_temp_files: false` in your config to inspect the generated libRadtran `.inp` files in the working directory. See the [documentation](https://franzflink.github.io/pyRadtran/guide/debugging.html) for debugging guidance.

## Acknowledgments

- Emde, C., Buras-Schnell, R., Kylling, A., Mayer, B., Gasteiger, J., Hamann, U., Kylling, J., Richter, B., Pause, C., Dowling, T., and Bugliaro, L.: *The libRadtran software package for radiative transfer calculations (version 2.0.1)*, Geosci. Model Dev., 9, 1647–1672, https://doi.org/10.5194/gmd-9-1647-2016, 2016.
- B. Mayer and A. Kylling. Technical note: The libRadtran software package for radiative transfer calculations - description and examples of use. Atmos. Chem. Phys., 5: 1855-1877, 2005. 
- Hoyer, S. & Hamman, J. (2017). *xarray: N-D labeled Arrays and Datasets in Python*. Journal of Open Research Software, 5(1), p.10. DOI: https://doi.org/10.5334/jors.148

## License

Please checkout libradtrans [license](https://www.libradtran.org/doc/COPYING)!
The pyRadtran project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
