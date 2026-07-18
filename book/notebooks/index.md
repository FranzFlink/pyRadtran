# Notebook Gallery

All notebooks run locally once pyRadtran and libRadtran are installed and
`~/.pyradtran/config.yaml` points to your installation (see
{doc}`/installation`). Case studies using non-public campaign data are
adaptable — swap in your own file paths.

## Tutorials — learn one thing at a time

| Notebook | Teaches |
|---|---|
| {doc}`quickstart` | The full basic workflow in ten minutes |
| {doc}`parameters_deep_dive` | The parameter system end to end: `params`, `Var`, validation, `describe()`, `explain()`, channels, jacobians |
| {doc}`albedo_test` | Per-point surface albedo with `Var` |
| {doc}`thermal` | Broadband thermal irradiance |
| {doc}`thermal_imager` | Brightness temperatures for a nadir imager |
| {doc}`solar_spectral` | Spectral solar irradiance |
| {doc}`era5_atmosphere` | ERA5 reanalysis as the atmosphere profile |
| {doc}`water_cloud` | A parametric water cloud via `wc_file` |
| {doc}`mixed_phase_cloud` | Mixed-phase cloud profiles from files |
| {doc}`get_radiosonde` | Atmosphere files from IGRA radiosondes |

## Case studies — full research workflows

Arctic radiation science built from the tutorial ingredients: campaign
simulations, instrument forward models, sensitivity studies.

| Notebook | Topic |
|---|---|
| {doc}`arctic_cloud_experiment` | Cloud parameter sweeps over sea ice |
| {doc}`velox_realistic_channels` | Six-channel VELOX thermal imager forward model |
| {doc}`thermal_imager_2` | Thermal imager with realistic atmospheres |
| {doc}`spectral_albedo_shift_cloud` | Spectral albedo shifts under clouds |
| {doc}`shupe_and_intrieri_plot` | Shupe & Intrieri cloud-forcing analysis |
| {doc}`era5_seasonal_sea_ice_profiles` | Seasonal ERA5 profiles over sea ice |
| {doc}`halo-ac3_bbr_all_aircraft` | HALO-(AC)³ broadband radiometer validation |
| {doc}`radiosonde_thermal` | Campaign radiosonde-driven thermal runs |
| {doc}`radiosonde_solar_spectral` | Campaign radiosonde-driven spectral runs |
| {doc}`sea_ice_era5` | Sea-ice ERA5 download & processing |
| {doc}`sea_ice_era5_plotting` | Visualising the processed ERA5 data |
| {doc}`carra_atmosphere` | CARRA regional reanalysis profiles |
| {doc}`stn2024` | Station Nord 2024 thermal study |
| {doc}`nya_rad` | Ny-Ålesund radiation with radiosondes |

```{tip}
Something looks off? {ref}`check-your-input-file` — inspecting the
generated `uvspec` input is the fastest route to a diagnosis.
```
