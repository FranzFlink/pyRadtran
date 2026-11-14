# Clouds

A cloud in libRadtran is a vertical profile of **water content** and
**effective radius**, separately for liquid (`wc_file`) and ice
(`ic_file`). pyRadtran gives you four ways to provide one — pick by
where your cloud information comes from:

| You have… | Use | Per point? |
|---|---|---|
| a cloud profile file on disk | `params={"wc_file 1D": "cloud.dat"}` | same cloud everywhere |
| a profile in Python (lists / arrays) | `params={"wc_file": {"z": …, "lwc": …, "reff": …}}` | same cloud everywhere |
| cloud top / base / water content as dataset variables | `cloud_wc_var=…, cloud_top_var=…, cloud_bottom_var=…` | yes — sweep axes welcome |
| ERA5 `clwc` / `ciwc` fields | `era5_atmosphere=era5, era5_clouds=True` | yes — from the reanalysis |

All four end up as the same thing in the uvspec input: a
`wc_file 1D <path>` and/or `ic_file 1D <path>` line.

```{warning}
Use `rte_solver: disort` for cloudy runs. The fast `twostr` solver is
numerically unstable for some optically thick layers and returns `NaN`
at individual wavelengths, which poisons integrated (broadband) results.
pyRadtran logs a warning when it detects this.
```

## The cloud file format

Three columns, **top of the atmosphere first**:

```text
#   z      LWC      r_eff
#  (km)  (g m-3)    (µm)
   1.5    0.0       10.0
   1.0    0.2       10.0
```

libRadtran reads each row as a *level*; the water content on a row
applies to the layer **between that level and the one above it**. The
file above is therefore one homogeneous cloud layer from 1.0 to 1.5 km
with 0.2 g m⁻³ — the value on the top row is not used. This is why
pyRadtran-generated two-level files simply repeat the value on both
rows.

## Useful cloud physics

The **liquid water path** is the vertical integral of the water content,

$$
\mathrm{LWP} = \int_{z_\mathrm{base}}^{z_\mathrm{top}} \mathrm{LWC}(z)\,\mathrm{d}z
$$

so a 500 m thick layer with 0.2 g m⁻³ has LWP = 100 g m⁻². In the
geometric-optics limit (droplets much larger than the wavelength,
extinction efficiency ≈ 2) the visible optical thickness follows as

$$
\tau \approx \frac{3\,\mathrm{LWP}}{2\,\rho_w\,r_\mathrm{eff}}
$$

with the density of water $\rho_w = 10^6$ g m⁻³: LWP = 100 g m⁻² and
$r_\mathrm{eff}$ = 10 µm give $\tau \approx 15$. This is a rule of thumb
for choosing inputs — the simulation itself uses the full optical
properties selected by `wc_properties` / `ic_properties`.

## 1. A file on disk

```python
from pathlib import Path

Path("work").mkdir(exist_ok=True)
Path("work/water_cloud.dat").write_text(
    "# z(km) LWC(g/m3) reff(um)\n"
    "1.5  0.0  10.0\n"
    "1.0  0.2  10.0\n"
)

res = ds.pyradtran.run(
    config_path="config/cloud_solar.yaml",
    params={"wc_file 1D": "work/water_cloud.dat"},
)
```

Mixed-phase clouds are simply both files at once:

```python
params = {
    "wc_file 1D": "work/water_cloud.dat",
    "ic_file 1D": "work/ice_cloud.dat",
}
```

## 2. A profile from Python

Pass the profile as a dict — pyRadtran writes the temporary file for
you:

```python
import numpy as np

z = np.array([1.5, 1.4, 1.3, 1.2, 1.1, 1.0])            # km, top first
lwc = np.array([0.0, 0.30, 0.24, 0.18, 0.12, 0.06])     # adiabatic-like, g m-3

res = ds.pyradtran.run(
    config_path="config/cloud_solar.yaml",
    params={"wc_file": {"z": z, "lwc": lwc, "reff": np.full(z.size, 10.0)}},
)
```

For ice use `"ic_file": {"z": …, "iwc": …, "reff": …}`.

Cloud altitudes are heights above sea level, as everywhere in
libRadtran.

To keep the cloud as a file — or to build it from named layers — use
{class}`~pyradtran.clouds.CloudLayer` and
{class}`~pyradtran.clouds.CloudFileWriter`; the file is written at the
exact layer boundaries and is radiatively identical to the dict form:

```python
from pyradtran.clouds import CloudFileWriter, CloudLayer

layers = [
    CloudLayer(z_bottom_km=1.0, z_top_km=1.5, lwc_g_m3=0.2, r_eff_um=10.0),
    CloudLayer(z_bottom_km=8.0, z_top_km=9.0, iwc_g_m3=0.02, r_eff_um=30.0),
]
CloudFileWriter.write_water_cloud_file(layers, "work/wc.dat")
CloudFileWriter.write_ice_cloud_file(layers, "work/ic.dat")
```

## 3. Clouds as dataset variables

The workhorse for sensitivity studies. Put the cloud properties in the
dataset — on `time` for a case study, on their own dimensions for a
parameter sweep — and name them:

```python
import numpy as np
import pandas as pd
import xarray as xr
import pyradtran

lwp = np.array([0.0, 5, 10, 20, 40, 80, 150, 300])      # g m-2
thickness_m = 500.0

ds = xr.Dataset(
    coords={
        "time": [pd.Timestamp("2025-06-21 12:00")],
        "latitude": ("time", [78.9]),
        "longitude": ("time", [11.9]),
        "lwp": lwp,                                     # sweep axis
    },
    data_vars={
        "lwc": ("lwp", lwp / thickness_m),              # g m-3
        "cbh": ("lwp", np.full(lwp.size, 1.0)),         # km
        "cth": ("lwp", np.full(lwp.size, 1.5)),         # km
    },
)

cloud = dict(cloud_wc_var="lwc", cloud_top_var="cth", cloud_bottom_var="cbh")

solar = ds.pyradtran.run(
    config_path="config/cloud_solar.yaml", params={"albedo": 0.8}, **cloud
)
thermal = ds.pyradtran.run(
    config_path="config/cloud_thermal.yaml",
    params={"albedo": 0.01, "sur_temperature": 273.15},
    **cloud,
)
```

| Keyword | Dataset variable holds | Default if omitted |
|---|---|---|
| `cloud_top_var`, `cloud_bottom_var` | cloud top / base height (km) — **both required** | no cloud |
| `cloud_wc_var` | liquid water content (g m⁻³) | no liquid cloud |
| `cloud_reff_var` | droplet effective radius (µm) | 10 µm |
| `cloud_ic_var` | ice water content (g m⁻³) | no ice cloud |
| `cloud_ic_reff_var` | ice effective radius (µm) | `cloud_reff_var`, else 20 µm |

Each point gets a homogeneous layer between base and top. A point where
top, base or the water content is NaN simply gets no cloud of that
phase — handy for mixing clear and cloudy points in one dataset.

The surface cloud radiative effect then is three lines
(remember the {ref}`units <units>`):

```python
net_sw = (solar.edir + solar.edn - solar.eup) / 1e3     # mW → W m-2
net_lw = thermal.edn - thermal.eup
cre = (net_sw + net_lw) - (net_sw + net_lw).sel(lwp=0)
```

```{figure} ../img/guide_cloud_forcing.png
:width: 95%

The result of the code above for two surface albedos: over open ocean
the shortwave cooling wins as soon as the cloud is optically thick,
over sea ice the longwave warming dominates — the cloud warms the
surface. Produced by `book/img/make_guide_figures.py`.
```

Preview the cloud file a point would get, without running:

```python
print(ds.pyradtran.inspect_cloud_file(selector={"lwp": 40}, **cloud))
```

```text
# wc_file profile
1.500000 0.080000 10.000000
1.000000 0.080000 10.000000
```

## 4. Clouds from ERA5

With an ERA5 dataset that contains `clwc` / `ciwc`, one switch converts
the reanalysis cloud water into per-point profiles (see
{doc}`atmosphere`):

```python
res = ds.pyradtran.run(
    config_path="config/cloud_thermal.yaml",
    era5_atmosphere=era5,
    era5_clouds=True,                    # or {"reff_water_um": 8.0}
)
```

The mass mixing ratios $q_c$ (kg kg⁻¹) are converted with the ideal-gas
air density, and the level heights come from the geopotential $\Phi$:

$$
\mathrm{LWC} = q_c\,\rho_\mathrm{air}\times 10^{3}\ \mathrm{g\,m^{-3}},
\qquad \rho_\mathrm{air} = \frac{p}{R_d\,T},
\qquad z = \frac{\Phi}{g_0}
$$

ERA5 carries no particle sizes; effective radii default to 10 µm
(liquid) and 20 µm (ice). Details:
{func}`~pyradtran.era5.cloud_profiles`.

**Precedence** when several sources describe a cloud for the same
point: explicit `wc_file`/`ic_file` in `params` → `cloud_*_var`
keywords → ERA5 clouds.

## Tuning a cloud: `wc_modify` / `ic_modify`

Rescale an existing profile to a target optical property without
touching the file — any number of modifications, as a list:

```python
params = {
    "wc_file 1D": "work/water_cloud.dat",
    "wc_modify": ["tau550 set 12"],          # optical thickness 12 at 550 nm
    "ic_properties": "baum_v36",             # ice habit parameterisation
}
```

```python
import pyradtran
print(pyradtran.describe("wc_modify"))
# wc_modify <gg|ssa|tau|tau550> <set|scale> <float>
# group: Water and ice clouds   (repeatable)
# requires: wc_file
```

Other options worth knowing — look them up with
{func}`pyradtran.describe() <pyradtran.params.describe>`:

| Option | Purpose |
|---|---|
| `wc_properties`, `ic_properties` | optical-property parameterisation (`hu`, `mie`, `fu`, `baum_v36`, …) |
| `ic_habit` | ice crystal habit for parameterisations that need one |
| `cloudcover wc` / `cloudcover ic` | fractional cover, independent-pixel approximation |
| `interpret_as_level wc` | treat file rows as level values rather than layer values |

## Where to go next

- {doc}`/notebooks/water_cloud` — a water cloud step by step
- {doc}`/notebooks/mixed_phase_cloud` — liquid + ice + ERA5 atmosphere
- {doc}`/notebooks/arctic_cloud_experiment` — full LWP × albedo × SZA sweep
- {doc}`/notebooks/shupe_and_intrieri_plot` — cloud forcing à la Shupe & Intrieri
