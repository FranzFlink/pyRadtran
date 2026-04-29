# Interactive Notebooks

This section contains interactive Jupyter notebooks that walk you through pyRadtran's capabilities — from basic quickstart tutorials to advanced scientific applications.

## How to Use These Notebooks

- **Run locally**: Download and run in Jupyter after installing pyRadtran and libRadtran.
- **Copy-paste ready**: All code cells are designed to work out of the box once your `~/.pyradtran/config.yaml` is set up (see {doc}`/installation`).
- **Real data included**: Many examples use actual atmospheric measurements from scientific campaigns.

## Notebook Overview

### Quickstart Tutorials
Start here if you're new to pyRadtran. These notebooks cover the basic simulation workflow:
- **Surface Albedo** — Run your first simulation with varying albedo values
- **Broadband Thermal** — Thermal infrared simulations with varying surface temperature
- **Brightness Temperature** — Simulating broadband brightness temperatures
- **Solar Spectral** — Spectral solar irradiance simulations
- **ERA5 Atmosphere** — Using ERA5 reanalysis data for atmospheric profiles

### Clouds
- **Water Clouds** — Adding parametric water clouds to simulations
- **Mixed-Phase Clouds** — Building mixed-phase cloud profiles from text files
- **Arctic Cloud Experiment** — Multi-dimensional cloud parameter sweeps

### Radiosondes & Atmosphere
- **Radiosonde Atmosphere Generator** — Creating atmosphere files from IGRA radiosonde data
- **Ny-Ålesund Radiation** — Simulating Arctic radiation with radiosonde profiles
- **HALO-(AC)³ Solar Validation** — Comparing simulations against aircraft measurements

### Advanced Topics
- **Custom Dimension Iteration** — Iterating over dimensions not yet natively supported
- **Shupe & Intrieri Reproduction** — Reproducing figures from the literature
- **Multi-Aircraft Thermal Validation** — Thermal validation against HALO-(AC)³ broadband radiometers
- **Realistic Clouds (Thermal)** — Constructing clouds from lidar/radar observations
- **Realistic Clouds (Solar Spectral)** — Same approach for spectral solar simulations

### Data Processing & Campaigns
- **ERA5 Sea Ice Profiles** — Downloading and processing ERA5 data for the Arctic
- **ERA5 Region Profiles** — Downloading and processing ERA5 data for specific regions
- **Sea Ice ERA5 Plotting** — Visualizing processed ERA5 sea ice data
- **Seasonal Sea Ice Profiles** — Seasonal ERA5 profiles for sea ice regions
- **CARRA Atmosphere** — Working with CARRA reanalysis data
- **Station Nord 2024** — Thermal simulations with radiosonde data from Station Nord

```{tip}
Always check your input file! See {ref}`check-your-input-file` for guidance on inspecting
the generated libRadtran input files and debugging common issues.
```

## Requirements

To run these notebooks locally, you'll need:

```bash
pip install pyradtran matplotlib numpy xarray pandas netcdf4
```

Some notebooks require additional packages (noted at the top of each notebook). Notebooks marked with
**"Campaign data required"** use datasets that are not publicly available — you can adapt the workflow
to your own data by replacing the file paths.
