# Notebook overview

All notebooks run locally once pyRadtran and libRadtran are installed and
`~/.pyradtran/config.yaml` points to your installation (see
{doc}`/installation`).

- **Tutorials** teach one thing at a time and run in seconds to minutes.
- **Case studies** are full research workflows; some need non-public
  campaign data and are meant to be adapted — swap in your own file paths.

Looking for a five-line answer instead of a notebook? Try the
{doc}`/examples/index`.

## Tutorials

### {doc}`Basics </tutorials/basics>`

| Notebook | What it shows |
|---|---|
| {doc}`Surface albedo </notebooks/albedo_test>` | Per-point surface albedo with `Var`; upward and downward irradiance |
| {doc}`Solar spectra </notebooks/solar_spectral>` | Spectral solar irradiance and how it changes over a day |
| {doc}`Broadband thermal </notebooks/thermal>` | Longwave irradiance from a thermal-source run |
| {doc}`Brightness temperature </notebooks/thermal_imager>` | Brightness temperatures for a nadir imager; emissivity via albedo |

### {doc}`Atmospheres </tutorials/atmosphere>`

| Notebook | What it shows |
|---|---|
| {doc}`ERA5 atmosphere </notebooks/era5_atmosphere>` | ERA5 reanalysis as the pressure / temperature / humidity profile |
| {doc}`Radiosondes (IGRA2) </notebooks/get_radiosonde>` | Atmosphere files from the nearest IGRA2 radiosonde |

### {doc}`Clouds </tutorials/clouds>`

| Notebook | What it shows |
|---|---|
| {doc}`Water clouds </notebooks/water_cloud>` | A liquid cloud via `wc_file`; solar and thermal net-flux profiles |
| {doc}`Mixed-phase clouds </notebooks/mixed_phase_cloud>` | Liquid + ice profiles combined with an ERA5 atmosphere |

### {doc}`Instruments & special geometries </tutorials/instruments>`

| Notebook | What it shows |
|---|---|
| {doc}`Horizontal LED extinction </notebooks/led_horizontal_extinction>` | Horizontal light extinction through marine aerosol |

## Case studies

### {doc}`Cloud radiative effect </case_studies/cloud_forcing>`

| Notebook | What it shows |
|---|---|
| {doc}`Multi-dimensional cloud experiment </notebooks/arctic_cloud_experiment>` | Cloud parameter sweeps over sea ice |
| {doc}`Shupe & Intrieri (2004) </notebooks/shupe_and_intrieri_plot>` | Reproducing the Arctic cloud-forcing figures |
| {doc}`Shupe & Intrieri with seasonal ERA5 profiles </notebooks/era5_seasonal_sea_ice_profiles>` | The same analysis with seasonal reanalysis atmospheres |
| {doc}`Cloud forcing with ACLOUD data </notebooks/spectral_albedo_shift_cloud>` | Spectral albedo shifts under clouds, from campaign observations |

### {doc}`Thermal imaging </case_studies/thermal_imaging>`

| Notebook | What it shows |
|---|---|
| {doc}`VELOX channels </notebooks/velox_realistic_channels>` | Six-channel thermal-infrared forward model with instrument effects |
| {doc}`Custom dimension iteration </notebooks/thermal_imager_2>` | Thermal imager sweeps over realistic atmospheres |

### {doc}`Campaign validation </case_studies/campaigns>`

| Notebook | What it shows |
|---|---|
| {doc}`HALO-(AC)³ broadband radiation </notebooks/halo-ac3_bbr_all_aircraft>` | Solar validation against aircraft broadband radiometers |
| {doc}`Observed clouds, thermal </notebooks/radiosonde_thermal>` | Radiosonde-driven thermal runs with realistic clouds |
| {doc}`Observed clouds, solar spectral </notebooks/radiosonde_solar_spectral>` | Radiosonde-driven spectral runs with realistic clouds |
| {doc}`Ny-Ålesund radiation </notebooks/nya_rad>` | Station radiation with radiosonde atmospheres |
| {doc}`Station Nord 2024 </notebooks/stn2024>` | Thermal study for the Station Nord campaign |

### {doc}`Reanalysis processing </case_studies/reanalysis>`

| Notebook | What it shows |
|---|---|
| {doc}`ERA5 sea-ice profiles </notebooks/sea_ice_era5>` | Download and processing of ERA5 profiles over sea ice |
| {doc}`Arctic atmosphere in a warming climate </notebooks/sea_ice_era5_plotting>` | Visualising the processed ERA5 profiles |
| {doc}`CARRA atmosphere </notebooks/carra_atmosphere>` | CARRA regional reanalysis as atmosphere profile |

```{tip}
Something looks off? {ref}`check-your-input-file` — inspecting the
generated `uvspec` input is the fastest route to a diagnosis.
```
