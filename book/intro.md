# PyRadtran Documentation

A Python interface for the [libRadtran](http://www.libradtran.org) radiative transfer model.

pyRadtran provides a Pythonic interface to the libRadtran radiative transfer model, which is used
for modeling radiation transport in Earth's atmosphere. This package makes it easy to set up, run,
and analyze radiative transfer simulations — with native support for both solar and thermal infrared radiation.

## Key Features

- **Solar & Thermal Simulations**: Support for both solar radiation (UV, visible, near-IR) and thermal infrared simulations, spectral and broadband
- **Flexible Configuration**: YAML-based configuration system with layered defaults (built-in → user master config → simulation-specific)
- **xarray Integration**: Native support for xarray datasets with automatic coordinate handling and metadata preservation
- **Atmospheric Profiles**: Support for standard atmospheric profiles, IGRA radiosondes, and ERA5 reanalysis data
- **Multi-core Processing**: Run simulations in parallel using multiple CPU cores

## Quick Start

Here's a simple example to get you started:

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
```

(check-your-input-file)=
## Check Your Input File!

One of the most important debugging skills when working with radiative transfer models is **inspecting the input file** that gets passed to `uvspec`. pyRadtran generates these files automatically, but when something goes wrong, looking at the raw input is often the fastest way to diagnose the issue.

### How to Inspect Generated Input Files

By default, pyRadtran cleans up temporary files after each simulation. To keep them for inspection, set `cleanup_temp_files: false` in your YAML configuration:

```yaml
execution:
  cleanup_temp_files: false
  debug_mode: true  # Also enables verbose logging output
```

After running your simulation, look in the **working directory** (default: `pyradtran_work/`) for `.inp` files. Each file corresponds to one simulation point.

### Reading a `.inp` File

A typical libRadtran input file looks like this:

```
atmosphere_file /opt/libradtran/share/libRadtran/data/atmmod/afglms.dat
source solar /opt/libradtran/share/libRadtran/data/solar_flux/NewGuey2003.dat
wavelength 400 770
albedo 0.3
sza 45.0
phi0 180.0
rte_solver disort
mol_abs_param lowtran
zout 0.01
day_of_year 94
output_quantity brightness
quiet
```

Each line is a libRadtran command. The key things to check:

| Line | What to look for |
|------|-----------------|
| `atmosphere_file` | Does the path exist? Is it the right atmosphere model? |
| `source solar` / `source thermal` | Is the source type correct for your simulation? Does the solar spectrum file exist? |
| `wavelength` | Is the spectral range correct? Common mistake: mixing up nm and cm⁻¹ |
| `albedo` | Is the value reasonable (0–1)? |
| `sza` | Is the solar zenith angle correct? Values > 90° mean the sun is below the horizon |
| `zout` | Are the output altitudes in km? |
| `rte_solver` | Is it appropriate for your simulation (e.g., `disort` for multiple scattering)? |

### Common Mistakes

1. **Wrong paths**: The most frequent error. If `atmosphere_file` or `source solar` paths don't exist on your machine, the simulation will fail silently or produce garbage output. → Set your paths in `~/.pyradtran/config.yaml` (see {doc}`installation`).

2. **Missing atmosphere file**: If you're using ERA5 or radiosonde profiles, make sure the generated atmosphere file exists and contains valid data.

3. **Spectral range mismatch**: If you request a wavelength range outside what `mol_abs_param` supports, you'll get unexpected results.

4. **Solar zenith angle > 90°**: Night-time points have SZA > 90°, which means no direct solar radiation. If you see all-zero output, check your time-of-day and location.

5. **Cloud file issues**: When using `wc_file` or `ic_file`, check that the cloud profile file exists and has the correct format (altitude, LWC/IWC, effective radius).

### Enabling Debug Mode

For maximum verbosity, enable debug mode in your config:

```yaml
execution:
  debug_mode: true
```

Or in Python:

```python
import logging
logging.getLogger('pyradtran').setLevel(logging.DEBUG)
```

This will show you exactly which input files are generated, which `uvspec` commands are executed, and what output is returned.

