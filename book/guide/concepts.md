# How pyRadtran works

pyRadtran turns **an xarray dataset of places and times** into **an
xarray dataset of radiative-transfer results**. It writes one libRadtran
(`uvspec`) input file per point, runs them in parallel, and puts the
outputs back on the dimensions of the input.

```text
 your Dataset                 pyRadtran                         result Dataset
┌──────────────┐   ┌──────────────────────────────────┐   ┌─────────────────────┐
│ time         │   │ 1. merge config layers           │   │ your dims            │
│ latitude     │──▶│ 2. validate params vs libRadtran │──▶│ + wavelength         │
│ longitude    │   │ 3. one .inp file per point       │   │ + altitude           │
│ (sweep dims) │   │ 4. uvspec × N in parallel        │   │ edir, edn, eup, …    │
│ (variables)  │   │ 5. parse + reshape + provenance  │   │ status, attrs        │
└──────────────┘   └──────────────────────────────────┘   └─────────────────────┘
```

The rest of this guide covers these five steps.

## The input dataset

Three names are required:

| Name | Role |
|---|---|
| `time` | UTC timestamp of each point; sets the solar zenith angle and day of year |
| `latitude`, `longitude` | degrees north / east; used for the solar geometry and to select ERA5 / radiosonde profiles |

```python
import pandas as pd
import xarray as xr
import pyradtran                      # registers ds.pyradtran

ds = xr.Dataset(
    coords={
        "time": pd.date_range("2025-06-21 06:00", periods=5, freq="3h"),
        "latitude": ("time", [78.9] * 5),
        "longitude": ("time", [11.9] * 5),
    }
)
```

For other names pass `time_var=`, `lat_var=`, `lon_var=` to
{meth}`~pyradtran.interface.PyRadtranAccessor.run`.

### Every dimension is a loop

pyRadtran simulates the **outer product of all dimensions** of the
dataset. A dataset with dims `(time: 5)` is 5 simulations; add a
dimension `albedo: 4` and it becomes 20, and the result has dims
`(time, albedo, …)`. A parameter sweep is a dataset dimension:

```python
import numpy as np
from pyradtran import Var

ds = xr.Dataset(
    coords={
        "time": [pd.Timestamp("2025-06-21 12:00")],
        "latitude": ("time", [78.9]),
        "longitude": ("time", [11.9]),
        "albedo": np.linspace(0.05, 0.9, 5),          # sweep axis
    }
)
ds["surface_albedo"] = ds.albedo                      # variable to reference

res = ds.pyradtran.run(
    config_path="config/solar_broadband.yaml",
    params={"albedo": Var("surface_albedo")},         # per-point value
)
res.eup.dims
# ('time', 'albedo', 'altitude')
```

A data variable can live on any subset of the dimensions: a variable on
`time` varies along the track, one on `albedo` varies along the sweep
axis, one on both varies over the full grid.

```{note}
If a sweep dimension has the same name as an output column you request
(`sza`, `albedo`), the column is returned as `sza_out` / `albedo_out`
so it does not clash with the coordinate.
```

```{tip}
The number of uvspec runs is the product of **all** dimension sizes.
Check `np.prod(list(ds.sizes.values()))` before a large sweep, and drop
dimensions you do not want to loop over.
```

### Altitude

`altitude` (km) means different things as a coordinate and as a data
variable:

| Layout | Meaning | Result |
|---|---|---|
| `altitude` is a **dimension / coordinate** | output levels of *one* simulation (`zout 0 1 2 …`) | an `altitude` dimension with those levels: a vertical profile from one uvspec call per point |
| `altitude` is a **data variable** (e.g. on `time`) | the sensor altitude of *each* point (`zout` per point) | one level per point: an aircraft track |

```python
# vertical profile at every time step
ds = ds.assign_coords(altitude=("altitude", np.arange(0, 10.5, 0.5)))

# aircraft track: altitude changes along time
ds["altitude"] = ("time", [0.1, 2.5, 5.0, 5.0, 3.0])
```

Without any `altitude`, the config's `output_altitudes_km` is used
(default: surface).

## From dataset to input file

For each point, three sources are merged into one input file. Later
ones win:

1. **Config**: the merged YAML layers ({doc}`configuration`), i.e. solver,
   spectral range, output columns, atmosphere file, defaults for albedo
   etc.
2. **`params` literals**: the same value for every point
   ({doc}`parameters`).
3. **`params` `Var(...)` references**: a different value per point,
   read from your dataset.

pyRadtran adds what depends on the point: the solar zenith angle from
time and location (unless you set `sza`), the `day_of_year` for the
Sun–Earth distance and, if requested, per-point atmosphere and cloud
files from ERA5 or radiosondes
({doc}`atmosphere`, {doc}`clouds`).

`explain()` shows the result, tagged by source, without running:

```python
print(ds.pyradtran.explain(
    config_path="config/solar_broadband.yaml",
    params={"albedo": Var("surface_albedo"), "mol_modify O3": 320.0},
))
```

```text
rte_solver disort                                                          # config
mol_abs_param reptran coarse                                               # config
data_files_path /opt/libRadtran-2.0.6/data                                 # config
atmosphere_file /opt/libRadtran-2.0.6/data/atmmod/afglus.dat               # config
source solar /opt/libRadtran-2.0.6/data/solar_flux/NewGuey2003.dat per_nm  # config
sza 55.71                                                                  # config
day_of_year 172                                                            # config
wavelength 300 3500                                                        # config
output_process integrate                                                   # config
output_user edir edn eup                                                   # config
zout 0.0000                                                                # config
umu 1.0                                                                    # config
mol_modify O3 320.0 DU                                                     # params-literal
albedo 0.05                                                                # dataset-var
quiet                                                                      # config
```

### The solar zenith angle

Unless `sza` is given (in the config or via `params`), it is computed
per point
({func}`~pyradtran.input_builder.calculate_solar_zenith_angle`):

$$
\cos\theta_0 = \sin\varphi\,\sin\delta + \cos\varphi\,\cos\delta\,\cos h,
\qquad
h = 15^\circ\left(t_\mathrm{UTC} + \frac{E_t}{60} - 12\right) + \lambda
$$

with latitude $\varphi$, longitude $\lambda$ (east positive), UTC time
$t_\mathrm{UTC}$ in hours, and the solar declination $\delta$ and
equation of time $E_t$ (minutes) from the Fourier series of Spencer
(1971). It agrees with the NOAA solar calculator to about 0.2°;
atmospheric refraction is not included. Naive timestamps are read as
UTC, timezone-aware ones are converted. To impose your own geometry,
pass `params={"sza": Var("my_sza")}`.

## Running in parallel

Each point is an independent `uvspec` process. `execution.max_workers`
sets how many run at once; `execution.timeout_seconds` stops runs that
hang. Points fail individually: a failed or skipped point becomes NaN
with a non-zero `status` ({doc}`outputs`) and the batch continues. Only
a batch where *every* point fails raises
{class}`~pyradtran.exceptions.PyRadtranError`.

Measured cost per point on one core (libRadtran 2.0.6). Spectral
resolution and solver dominate, so start coarse:

| Setup | Time per point |
|---|---|
| `twostr`, `lowtran`, 350–2500 nm spectral | ~0.5 s |
| `disort`, `reptran fine`, 8–13 µm spectral radiance | ~1 s |
| `disort`, `reptran coarse`, 300–3500 nm broadband | ~3.5 s |

## Modules

| Step | Module | Key entry points |
|---|---|---|
| Config layers | {mod}`pyradtran.config` | {func}`~pyradtran.config.load_config`, {class}`~pyradtran.config.SimulationConfig` |
| Parameters & validation | {mod}`pyradtran.params`, {mod}`pyradtran.schema` | {class}`~pyradtran.params.Var`, {func}`~pyradtran.params.describe` |
| Input files | {mod}`pyradtran.input_builder` | {class}`~pyradtran.input_builder.InputFileBuilder` |
| Atmospheres | {mod}`pyradtran.era5`, {mod}`pyradtran.io` | {func}`~pyradtran.era5.normalize_era5`, {class}`~pyradtran.io.RadiosondeAtmosphereGenerator` |
| Clouds | {mod}`pyradtran.clouds`, {mod}`pyradtran.era5` | {class}`~pyradtran.clouds.CloudLayer`, {func}`~pyradtran.era5.cloud_profiles` |
| Execution | {mod}`pyradtran.interface`, {mod}`pyradtran.core` | {meth}`~pyradtran.interface.PyRadtranAccessor.run` |
| Parsing & storage | {mod}`pyradtran.io` | {class}`~pyradtran.io.OutputParser`, {class}`~pyradtran.io.NetCDFSaver` |
| Post-processing | {mod}`pyradtran.channels` | {func}`~pyradtran.channels.convolve_channels`, {func}`~pyradtran.channels.brightness_temperature` |
