# Cookbook

Short recipes to copy. Each needs only this common setup:

```python
import numpy as np
import pandas as pd
import xarray as xr

import pyradtran                              # registers ds.pyradtran
from pyradtran import Var, load_config

def scene(**extra_coords):
    """One point (Ny-Ålesund, midsummer noon) plus any sweep dimensions."""
    return xr.Dataset(
        coords={
            "time": [pd.Timestamp("2025-06-21 12:00")],
            "latitude": ("time", [78.9]),
            "longitude": ("time", [11.9]),
            **extra_coords,
        }
    )

def make_config(source="solar", wavelength_nm=(300, 3500), integrate=True,
                columns=("edir", "edn", "eup"), solver="disort",
                mol_abs_param="reptran coarse"):
    """A config built in Python instead of YAML."""
    cfg = load_config()                       # package defaults + ~/.pyradtran
    sd = cfg.simulation_defaults
    sd.source, sd.rte_solver, sd.mol_abs_param = source, solver, mol_abs_param
    sd.wavelength_nm = list(wavelength_nm)
    sd.integrate_wavelength = integrate
    sd.output_columns = list(columns)
    return cfg

RUN = dict(save_to_file=False, show_progress=False)   # no file, no progress bar
```

Every recipe passes `config=`; with a YAML file you would pass
`config_path="config/my_simulation.yaml"` instead.

## Running

### Broadband surface irradiance

```python
res = scene().pyradtran.run(config=make_config(), **RUN)

sfc = res.isel(time=0, altitude=0) / 1e3      # solar: mW m-2 → W m-2
print(f"direct {sfc.edir.item():.0f}, diffuse {sfc.edn.item():.0f} W m-2")
```

### A spectrum

```python
cfg = make_config(integrate=False, wavelength_nm=(300, 1100),
                  solver="twostr", mol_abs_param="lowtran")
res = scene().pyradtran.run(config=cfg, **RUN)

res.edir.isel(time=0, altitude=0).plot()      # mW m-2 nm-1 vs wavelength
```

### A time series (diurnal cycle)

```python
times = pd.date_range("2025-06-21", periods=24, freq="h")
ds = xr.Dataset(coords={
    "time": times,
    "latitude": ("time", np.full(24, 52.0)),
    "longitude": ("time", np.full(24, 13.0)),
})
res = ds.pyradtran.run(config=make_config(), **RUN)

(res.edir + res.edn).isel(altitude=0).plot()  # global irradiance over the day
```

`disort` returns zeros at night; `twostr` fails for a sun below the
horizon and marks those points `status == 1`.

### A vertical profile

```python
ds = scene(altitude=("altitude", np.arange(0.0, 12.5, 0.5)))    # km
res = ds.pyradtran.run(config=make_config(columns=("edir", "edn", "eup", "enet")), **RUN)

res.enet.isel(time=0).plot(y="altitude")
```

### Along a flight track

`altitude` as a *data variable* gives each point its own output level:

```python
n = 6
ds = xr.Dataset(
    coords={
        "time": pd.date_range("2022-03-14 10:00", periods=n, freq="10min"),
        "latitude": ("time", np.linspace(78.0, 80.0, n)),
        "longitude": ("time", np.linspace(10.0, 5.0, n)),
    },
    data_vars={"altitude": ("time", [0.1, 3.0, 8.0, 10.0, 10.0, 5.0])},   # km
)
res = ds.pyradtran.run(config=make_config(), **RUN)
```

## Parameters

### The same value everywhere

```python
res = scene().pyradtran.run(
    config=make_config(),
    params={"albedo": 0.85, "mol_modify O3": 320.0, "aerosol_default": True},
    **RUN,
)
```

### A value per point

```python
ds = scene(albedo=np.linspace(0.05, 0.9, 5))
ds["surface_albedo"] = ds.albedo
res = ds.pyradtran.run(
    config=make_config(), params={"albedo": Var("surface_albedo")}, **RUN
)

res.eup.isel(time=0, altitude=0).plot()       # upward irradiance vs albedo
```

### A two-dimensional sweep

Each dimension is a loop; here 4 × 5 = 20 simulations:

```python
ds = scene(sza=[20.0, 40.0, 60.0, 80.0], albedo=np.linspace(0.05, 0.9, 5))
ds["sza_deg"], ds["surface_albedo"] = ds.sza, ds.albedo
res = ds.pyradtran.run(
    config=make_config(),
    params={"sza": Var("sza_deg"), "albedo": Var("surface_albedo")},
    **RUN,
)

res.edn.isel(time=0, altitude=0).plot()       # 2-D map: diffuse irradiance
```

### Change the config for one run

A dotted key addresses a config field instead of a uvspec option:

```python
res = scene().pyradtran.run(
    config=make_config(),
    params={"simulation_defaults.wavelength_nm": [400, 700]},   # PAR only
    **RUN,
)
```

### Look up an option

```python
pyradtran.search_options("aerosol")[:5]
print(pyradtran.describe("aerosol_visibility"))
```

### Check the input before running

```python
print(scene().pyradtran.explain(config=make_config(), params={"albedo": 0.85}))
```

## Thermal & instruments

### Longwave fluxes at the surface

```python
cfg = make_config(source="thermal", wavelength_nm=(4000, 100000),
                  columns=("edn", "eup"))
res = scene().pyradtran.run(
    config=cfg, params={"sur_temperature": 273.15, "albedo": 0.01}, **RUN
)
print(f"downward longwave: {res.edn.item():.0f} W m-2")       # thermal: W m-2
```

### Brightness temperature from uvspec

```python
cfg = make_config(source="thermal", wavelength_nm=(10000, 12000), columns=("uu",))
cfg.simulation_defaults.output_altitudes_km = [10.0]
ds = scene(surface_temperature=[250.0, 270.0, 290.0])
ds["tsurf"] = ds.surface_temperature
res = ds.pyradtran.run(
    config=cfg,
    params={
        "sur_temperature": Var("tsurf"),
        "albedo": 0.0,                        # emissivity 1
        "output_quantity": "brightness",      # uvspec converts to K
    },
    **RUN,
)
res.uu.isel(time=0, altitude=0).values        # band brightness temperatures (K)
```

### Brightness temperature from spectral radiances

```python
from pyradtran import brightness_temperature

cfg = make_config(source="thermal", wavelength_nm=(8000, 13000), integrate=False,
                  columns=("uu",), mol_abs_param="reptran fine")
cfg.simulation_defaults.output_altitudes_km = [10.0]
res = scene().pyradtran.run(
    config=cfg,
    params={"sur_temperature": 280.0, "albedo": 0.0, "output_process": "per_nm"},
    **RUN,
).isel(time=0, altitude=0)

tb = brightness_temperature(res.uu, res.wavelength, "W m-2 nm-1 sr-1")
```

### Instrument channels

```python
wl = xr.DataArray(np.arange(8000.0, 13000.0, 10.0), dims="wavelength")
wl = wl.assign_coords(wavelength=wl.values)
centres = xr.DataArray([8650.0, 10740.0, 12000.0], dims="channel")
srf = np.exp(-0.5 * ((wl - centres) / 150.0) ** 2).transpose("channel", "wavelength")
srf = srf.assign_coords(channel=["8.65um", "10.74um", "12.00um"])

chan = scene().pyradtran.run(
    config=cfg,                               # spectral thermal config from above
    params={"sur_temperature": 280.0, "albedo": 0.0, "output_process": "per_nm"},
    channels=srf,
    **RUN,
).isel(time=0, altitude=0)

tb = brightness_temperature(chan.uu, centres.values, "W m-2 nm-1 sr-1")
```

## Clouds & atmosphere

### A cloud layer per point

```python
lwp = np.array([0.0, 20.0, 80.0, 300.0])                     # g m-2
ds = scene(lwp=lwp)
ds["lwc"] = ("lwp", lwp / 500.0)                              # 500 m thick → g m-3
ds["cbh"] = ("lwp", np.full(lwp.size, 1.0))                   # km
ds["cth"] = ("lwp", np.full(lwp.size, 1.5))
res = ds.pyradtran.run(
    config=make_config(),
    cloud_wc_var="lwc", cloud_top_var="cth", cloud_bottom_var="cbh",
    **RUN,
)

transmission = (res.edir + res.edn) / (res.edir + res.edn).sel(lwp=0)
```

### A cloud profile from arrays

```python
z = np.array([1.5, 1.4, 1.3, 1.2, 1.1, 1.0])                  # km, top first
lwc = np.array([0.0, 0.30, 0.24, 0.18, 0.12, 0.06])           # g m-3
res = scene().pyradtran.run(
    config=make_config(),
    params={"wc_file": {"z": z, "lwc": lwc, "reff": np.full(z.size, 10.0)}},
    **RUN,
)
```

### Fix the cloud optical thickness

```python
res = scene().pyradtran.run(
    config=make_config(),
    params={
        "wc_file": {"z": [1.5, 1.0], "lwc": [0.2, 0.2], "reff": [10.0, 10.0]},
        "wc_modify": ["tau550 set 12"],
    },
    **RUN,
)
```

### Choose the background atmosphere

```python
pyradtran.recommend_atmosphere(latitude=78.9, time="2025-06-21")
# 'afglss'  (sub-arctic summer)

res = scene().pyradtran.run(
    config=make_config(), params={"atmosphere_file": "afglss"}, **RUN
)
```

Short names (`afglss`, `afglsw`, …) and full paths both work.

### Water vapour and ozone columns

```python
ds = scene(pwv=[2.0, 5.0, 10.0, 20.0])                        # mm
ds["h2o"] = ds.pwv
cfg = make_config(source="thermal", wavelength_nm=(4000, 100000), columns=("edn",))
res = ds.pyradtran.run(
    config=cfg,
    params={"mol_modify H2O": Var("h2o"), "sur_temperature": 273.15},
    **RUN,
)
res.edn.isel(time=0, altitude=0).plot(marker="o")             # LW down vs PWV
```

### ERA5 atmosphere and clouds

```python
# era5: any ERA5 dataset on pressure levels (CDS or ARCO naming),
# subset and .load()-ed around your points
res = ds.pyradtran.run(
    config=make_config(source="thermal", wavelength_nm=(4000, 100000),
                       columns=("edn", "eup")),
    era5_atmosphere=era5,
    era5_clouds=True,
    **RUN,
)
```

## Analysis

### Sensitivity to a parameter

```python
jac = scene().pyradtran.jacobian(
    "albedo", 0.01, config=make_config(), params={"albedo": 0.3},
    show_progress=False,
)
jac.eup.item() / 1e3           # d(E_up)/d(albedo) in W m-2 per unit albedo
```

### Keep only successful points

```python
good = res.where(res.status == 0)
print(f"{int((res.status != 0).sum())} of {res.status.size} points failed")
```

### Save, reload and read the provenance

```python
path = scene().pyradtran.run(
    config=make_config(), output_path="output/midsummer.nc",
    return_dataset=False, show_progress=False,
)
back = xr.open_dataset(path)
print(back.attrs["pyradtran_input_example"])   # the annotated uvspec input
```
