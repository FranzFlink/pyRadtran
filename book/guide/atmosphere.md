# Atmosphere & ERA5

Every simulation needs a vertical profile of pressure, temperature and
trace gases. Three sources combine:

1. **Background profile** (`atmosphere_file`): one of the AFGL
   standard atmospheres shipped with libRadtran.
   Always present.
2. **Measured / reanalysis profile** (`radiosonde`): a p/T/H2O
   (and optionally O3) column from an ERA5 dataset or an IGRA2
   radiosonde. Overlays the background.
3. **Scalar column scalings** (`ozone_du`, `h2o_mm` /
   `mol_modify …`): rescale a gas column to a total value.

## How the layers combine

libRadtran always starts from the `atmosphere_file`. When a
`radiosonde` profile is given, it replaces pressure, temperature and
the provided gas profiles from the surface up to the profile top; above
that, and for every gas the profile does not contain, the background
profile fills in (shifted so the two match at the transition). A `-1`
value at any profile level also falls back to the background.

The background therefore still matters with ERA5 or a radiosonde: pick
one that matches season and latitude, so the stratosphere and the
unmeasured gases stay consistent.

```python
import pyradtran

pyradtran.recommend_atmosphere(latitude=78.0, time="2022-03-15")
# 'afglsw'  (sub-arctic winter)
```

Set it in your YAML config (`paths.atmosphere_profile: afglsw`) or per
run via `params={"atmosphere_file": "afglsw"}`, as a short name or a
full path.

| Short name | Atmosphere | Use for |
|---|---|---|
| `afglus` | US standard 1976 | mid-latitude annual mean, generic studies |
| `afglt` | tropical | 15° S – 15° N |
| `afglms` / `afglmw` | mid-latitude summer / winter | 30–60° |
| `afglss` / `afglsw` | sub-arctic summer / winter | poleward of 60° |

## What goes into a profile file

ERA5 gives specific humidity $q$ and geopotential $\Phi$ on pressure
levels. pyRadtran writes pressure, temperature and the gas mixing ratios
as a libRadtran `radiosonde` file; the conversions involved are

$$
z = \frac{\Phi}{g_0}, \qquad
\rho_\mathrm{air} = \frac{p}{R_d\,T}, \qquad
\mathrm{PWV} = \frac{1}{g_0\,\rho_w}\int_0^{p_s} q\,\mathrm{d}p
$$

with $g_0 = 9.80665$ m s⁻², $R_d = 287.058$ J kg⁻¹ K⁻¹. The last
expression checks a profile against the `h2o_mm` value you would
otherwise set: for $q$ in kg kg⁻¹ and $p$ in Pa it gives the
precipitable water in metres. Details and file format:
{func}`~pyradtran.era5.write_atmosphere_file`.

## When to use which source

| Situation | Use |
|---|---|
| Idealised / sensitivity study | background profile only |
| You know the total ozone / water column | background + `ozone_du` / `h2o_mm` |
| Case study with reanalysis data | `era5_atmosphere=` (ERA5 profile) |
| Co-located sounding available | radiosonde file |

## ERA5

`era5_atmosphere=` takes a raw ERA5 dataset with CDS NetCDF short names
(`t`, `q`, `o3`, `pressure_level`, …) or ARCO-ERA5 Zarr long names
(`temperature`, `specific_humidity`, `level`, …). Names, coordinate
conventions and pressure units are normalised
(`pyradtran.normalize_era5`):

```python
import gcsfs
import xarray as xr

gcs = gcsfs.GCSFileSystem(token="anon")
era5 = xr.open_zarr(
    gcs.get_mapper(
        "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
    ),
    chunks="auto",
    consolidated=True,
)
# Subset and .load() region and time first: per-point access into the
# global archive is slow.
era5 = era5.sel(
    time=slice("2022-03-01", "2022-03-31"),
    latitude=slice(80, 75),
    longitude=slice(10, 20),
).load()

result = ds.pyradtran.run(
    config_path="config/thermal.yaml",
    era5_atmosphere=era5,
)
```

One atmosphere file is written per unique (time, lat, lon) of the input
dataset, selected nearest-neighbour from the ERA5 data.

### Ozone

If the ERA5 dataset contains an ozone field (`o3` /
`ozone_mass_mixing_ratio`), the ozone profile is written into the
atmosphere file alongside humidity. In that case the config
`ozone_du` scaling is **not** applied, because rescaling the profile
would discard the reanalysis information.
Without an ozone field, `ozone_du` scales the background ozone column
as usual. The same rule applies to water vapour: a radiosonde/ERA5
profile disables the `h2o_mm` scaling.

To vary ozone as an experiment axis, drop `o3` from the ERA5 dataset
and use `params={"mol_modify O3": Var("ozone_du")}` instead.

### Clouds from ERA5

The same dataset can drive clouds. `era5_clouds=True` converts the
`clwc` / `ciwc` fields (kg/kg → g/m³ via the ideal-gas air density,
altitudes from geopotential) into per-point `wc_file` / `ic_file`
profiles:

```python
result = ds.pyradtran.run(
    config_path="config/thermal.yaml",
    era5_atmosphere=era5,
    era5_clouds=True,                      # or {"reff_water_um": 8.0}
)
```

ERA5 carries no droplet sizes; effective radii default to 10 µm
(liquid) and 20 µm (ice) and can be set via the dict form.

A `wc_file`/`ic_file` in `params` or the `cloud_*_var` kwargs (see
below) take precedence over ERA5 clouds at the points where both exist.
All cloud inputs are described in {doc}`clouds`.

```{warning}
For cloudy runs use `rte_solver: disort`. The fast `twostr` solver is
numerically unstable for some optically thick cloud layers and returns
`NaN` at individual wavelengths, which turns integrated (broadband)
results into `NaN`. pyRadtran logs a warning when it detects this.
```

## Synthetic clouds in an ERA5 atmosphere

For parameter-space studies (e.g. cloud radiative effect over LWP ×
albedo × SZA) put the cloud parameters on their own dataset dimensions
and reference them with the `cloud_*_var` kwargs:

```python
ds = xr.Dataset(
    coords={"time": times, "latitude": ("time", lats),
            "longitude": ("time", lons)},
    data_vars={
        "lwc":  (("lwc",),  lwcs),   # variation axis
        "reff": (("reff",), reffs),  # variation axis
        "cth":  (("lwc",),  cth),
        "cbh":  (("lwc",),  cbh),
    },
)
result = ds.pyradtran.run(
    config_path="config/arctic_cloud_experiment_thermal.yaml",
    cloud_wc_var="lwc", cloud_reff_var="reff",
    cloud_top_var="cth", cloud_bottom_var="cbh",
)
```

Add `era5_atmosphere=` for a realistic atmosphere under the synthetic
cloud grid. The explicit cloud kwargs override ERA5 clouds; with
`era5_clouds` unset the ERA5 data supplies only the gas and temperature
profiles.

## Radiosondes (IGRA2)

`RadiosondeAtmosphereGenerator` finds the closest active IGRA2 station
and sounding in time and writes the same file format
(`H2O RH` columns). Files from both sources carry a `# columns:`
header, from which the input builder writes the matching `radiosonde`
option line.
