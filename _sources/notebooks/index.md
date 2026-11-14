# Notebook overview

The notebooks run locally once pyRadtran and libRadtran are installed
and `~/.pyradtran/config.yaml` points to the installation (see
{doc}`/installation`). For a five-line answer instead of a notebook, see
the {doc}`cookbook </examples/index>`.

**Tutorials** cover one concept each and run in seconds to minutes.

| Group | Notebooks |
|---|---|
| {doc}`Basics </tutorials/basics>` | surface albedo, solar spectra, broadband thermal, brightness temperature |
| {doc}`Atmospheres </tutorials/atmosphere>` | ERA5 profiles, IGRA2 radiosondes, lapse-rate feedback |
| {doc}`Clouds </tutorials/clouds>` | water clouds, mixed-phase clouds |
| {doc}`Instruments & special geometries </tutorials/instruments>` | horizontal LED extinction |

**Case studies** are research workflows. Some need campaign data that is
not public; replace the file paths with your own.

| Group | Notebooks |
|---|---|
| {doc}`Cloud radiative effect </case_studies/cloud_forcing>` | cloud parameter sweeps, Shupe & Intrieri (2004), ACLOUD |
| {doc}`Thermal imaging </case_studies/thermal_imaging>` | VELOX channels, thermal imager sweeps |
| {doc}`Campaign validation </case_studies/campaigns>` | HALO-(AC)³, Ny-Ålesund, Station Nord |
| {doc}`Reanalysis processing </case_studies/reanalysis>` | ERA5 sea-ice profiles, CARRA |

If a result looks wrong, {ref}`check the input file <check-your-input-file>` first.
