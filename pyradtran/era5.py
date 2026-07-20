"""
ERA5 integration: normalisation, atmosphere files, and cloud profiles.

ERA5 data arrives under several naming conventions — the CDS NetCDF short
names (``t``, ``q``, ``o3``, ``clwc``, coordinate ``pressure_level``) and
the ARCO-ERA5 Zarr long names (``temperature``, ``specific_humidity``,
``geopotential``, coordinate ``level``).  :func:`normalize_era5` maps any
of them onto one canonical form so the rest of pyRadtran never has to
care which flavour the user opened.

From a normalised dataset this module can produce, per (time, lat, lon)
point:

* a libRadtran ``radiosonde`` atmosphere file — pressure, temperature,
  humidity and (when available) an ozone profile
  (:func:`write_atmosphere_file`), and
* dict-valued ``wc_file`` / ``ic_file`` cloud profiles from the ERA5
  ``clwc`` / ``ciwc`` fields (:func:`cloud_profiles`).

:func:`recommend_atmosphere` suggests the AFGL background profile
(``atmosphere_file``) that matches a latitude and date.  The background
profile still matters when a radiosonde is used: libRadtran takes every
level above the radiosonde top — and every gas the radiosonde does not
provide — from the ``atmosphere_file``.

See Also
--------
pyradtran.io.ERA5AtmosphereGenerator : Backwards-compatible wrapper.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr

from .exceptions import InputGenerationError

logger = logging.getLogger(__name__)

# Physical constants
G0 = 9.80665          # standard gravity (m s-2)
R_DRY = 287.058       # specific gas constant, dry air (J kg-1 K-1)

#: canonical name -> accepted aliases (first match wins)
VAR_ALIASES: Dict[str, Tuple[str, ...]] = {
    "t": ("t", "temperature"),
    "q": ("q", "specific_humidity"),
    "z": ("z", "geopotential"),
    "o3": ("o3", "ozone_mass_mixing_ratio"),
    "clwc": ("clwc", "specific_cloud_liquid_water_content"),
    "ciwc": ("ciwc", "specific_cloud_ice_water_content"),
    "cc": ("cc", "fraction_of_cloud_cover"),
    "skt": ("skt", "skin_temperature"),
    "sst": ("sst", "sea_surface_temperature"),
}

COORD_ALIASES: Dict[str, Tuple[str, ...]] = {
    "pressure_level": ("pressure_level", "level", "plev", "isobaricInhPa"),
    "valid_time": ("valid_time", "time"),
}


def normalize_era5(ds: xr.Dataset) -> xr.Dataset:
    """Return *ds* with ERA5 variables/coordinates on canonical names.

    Accepts CDS-style short names, ARCO-ERA5 long names, or an already
    canonical dataset (the call is idempotent).  The pressure coordinate
    is converted to hPa when it arrives in Pa.  Unit attributes that
    ERA5 pipelines commonly strip are restored with their known values.

    Parameters
    ----------
    ds : xarray.Dataset
        Any ERA5(-like) dataset on pressure levels, e.g. straight from
        ``xr.open_zarr(gcsfs...)`` or a CDS NetCDF download.

    Returns
    -------
    xarray.Dataset
        Same data, canonical names: variables ``t``, ``q``, ``z``,
        ``o3``, ``clwc``, ``ciwc``, ``cc``, ``skt``, ``sst`` (those
        present), coordinates ``pressure_level`` (hPa) and
        ``valid_time``.
    """
    rename = {}
    for canonical, aliases in VAR_ALIASES.items():
        if canonical in ds.variables:
            continue
        for alias in aliases[1:]:
            if alias in ds.variables:
                rename[alias] = canonical
                break
    for canonical, aliases in COORD_ALIASES.items():
        if canonical in ds.coords or canonical in ds.dims:
            continue
        for alias in aliases[1:]:
            if alias in ds.coords or alias in ds.dims:
                rename[alias] = canonical
                break
    if rename:
        ds = ds.rename(rename)
        logger.debug(f"normalize_era5 renamed: {rename}")

    if "pressure_level" in ds.coords:
        plev = ds["pressure_level"]
        unit = str(plev.attrs.get("units", "")).strip()
        values = np.asarray(plev.values, dtype=float)
        if unit == "Pa" or (unit == "" and values.max() > 2000.0):
            ds = ds.assign_coords(pressure_level=values / 100.0)
        elif unit.lower() in ("", "hpa", "millibars", "millibar", "mb", "mbar"):
            ds = ds.assign_coords(pressure_level=values)
        else:
            raise InputGenerationError(
                f"Unsupported pressure unit '{unit}' in ERA5 dataset; "
                f"expected 'Pa' or 'hPa'"
            )
        ds["pressure_level"].attrs["units"] = "hPa"

    known_units = {
        "t": "K",
        "q": "kg kg-1",
        "o3": "kg kg-1",
        "clwc": "kg kg-1",
        "ciwc": "kg kg-1",
        "z": "m2 s-2",
        "skt": "K",
        "sst": "K",
    }
    for var, unit in known_units.items():
        if var in ds.variables and "units" not in ds[var].attrs:
            ds[var].attrs["units"] = unit
    return ds


def select_profile(
    ds: xr.Dataset,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    time: Union[str, datetime, np.datetime64, None] = None,
) -> xr.Dataset:
    """Select a single vertical profile from a (normalised) ERA5 dataset.

    Handles datasets that still carry spatial/time dimensions as well as
    ones already reduced to a single column.  Selection is
    nearest-neighbour.

    Returns
    -------
    xarray.Dataset
        Profile with ``pressure_level`` as the only remaining dimension,
        loaded into memory.
    """
    if "latitude" in ds.dims or "longitude" in ds.dims:
        if latitude is None or longitude is None:
            raise ValueError(
                "Dataset has spatial dimensions; latitude and longitude required"
            )
        ds = ds.sel(latitude=latitude, longitude=longitude, method="nearest")

    if "valid_time" in ds.dims:
        if time is None:
            raise ValueError("Dataset has a time dimension; time required")
        ds = ds.sel(valid_time=pd.to_datetime(time), method="nearest")

    extra = [d for d in ds.dims if d != "pressure_level"]
    if extra:
        ds = ds.isel({d: 0 for d in extra})
    return ds.load()


def write_atmosphere_file(
    profile: xr.Dataset,
    output_filepath: Union[str, Path],
    include_ozone: bool = True,
) -> Path:
    """Write a profile as a libRadtran ``radiosonde`` atmosphere file.

    Columns are pressure (hPa), temperature (K), specific humidity
    (MMR, kg/kg) and — when ``o3`` is present and *include_ozone* is
    true — ozone MMR.  The column layout is recorded in a
    machine-readable header line (``# columns: H2O MMR O3 MMR``) that
    the input builder reads back to construct the matching
    ``radiosonde`` option line.

    Levels containing NaN pressure/temperature/humidity (e.g. ERA5
    levels below the surface) are dropped; NaN ozone values become
    ``-1``, which tells libRadtran to fall back to the
    ``atmosphere_file`` value at that level.

    Parameters
    ----------
    profile : xarray.Dataset
        Single column, canonical names (see :func:`normalize_era5`),
        with ``pressure_level`` in hPa and variables ``t`` and ``q``.
    output_filepath : str or pathlib.Path
        Destination file.
    include_ozone : bool, default ``True``
        Write the ozone column when available.

    Returns
    -------
    pathlib.Path
    """
    for var in ("t", "q"):
        if var not in profile.variables:
            raise InputGenerationError(
                f"Profile is missing required variable '{var}' "
                f"(have: {sorted(profile.data_vars)})"
            )
    if "pressure_level" not in profile.coords:
        raise InputGenerationError("Profile has no 'pressure_level' coordinate")

    p = np.asarray(profile["pressure_level"].values, dtype=float).ravel()
    t = np.asarray(profile["t"].values, dtype=float).ravel()
    q = np.asarray(profile["q"].values, dtype=float).ravel()

    has_o3 = include_ozone and "o3" in profile.variables
    o3 = (
        np.asarray(profile["o3"].values, dtype=float).ravel()
        if has_o3
        else None
    )

    valid = ~(np.isnan(p) | np.isnan(t) | np.isnan(q))
    if not valid.any():
        raise InputGenerationError("No valid (non-NaN) levels in ERA5 profile")
    p, t, q = p[valid], t[valid], q[valid]
    if o3 is not None:
        o3 = o3[valid]
        o3 = np.where(np.isnan(o3), -1.0, o3)

    # TOA first: ascending pressure; drop duplicate levels (libRadtran
    # requires a strictly monotonic grid).
    order = np.argsort(p)
    p, t, q = p[order], t[order], q[order]
    if o3 is not None:
        o3 = o3[order]
    keep = np.concatenate(([True], np.diff(p) > 0))
    p, t, q = p[keep], t[keep], q[keep]
    if o3 is not None:
        o3 = o3[keep]

    columns = "H2O MMR O3 MMR" if o3 is not None else "H2O MMR"

    output_path = Path(output_filepath)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("# ERA5 atmosphere profile (libRadtran radiosonde format)\n")
        f.write(f"# columns: {columns}\n")
        f.write("# p(hPa)  T(K)  " + columns.replace(" ", "_") + "\n")
        for i in range(len(p)):
            row = f"{p[i]:.2f}  {t[i]:.2f}  {q[i]:.4e}"
            if o3 is not None:
                row += f"  {o3[i]:.4e}"
            f.write(row + "\n")

    logger.info(f"Wrote ERA5 atmosphere file ({columns}): {output_path}")
    return output_path


def era5_atmosphere_file(
    era5_ds: xr.Dataset,
    latitude: float,
    longitude: float,
    time: Union[str, datetime, np.datetime64],
    output_filepath: Union[str, Path],
    include_ozone: bool = True,
) -> Path:
    """One-step: normalise, select a point, write the atmosphere file."""
    try:
        ds = normalize_era5(era5_ds)
        profile = select_profile(ds, latitude, longitude, time)
        return write_atmosphere_file(profile, output_filepath, include_ozone)
    except InputGenerationError:
        raise
    except Exception as e:
        raise InputGenerationError(
            f"Failed to create ERA5 atmosphere file: {e}"
        ) from e


def _z_to_km(z: np.ndarray, units) -> np.ndarray:
    """Convert geopotential / height values to altitude in km.

    The units attribute decides the conversion; the magnitude heuristic is
    only a fallback for data without (or with unrecognised) units, where it
    can misread profiles whose top lies below ~10 km.
    """
    z = np.asarray(z, dtype=float)
    unit = str(units or "").strip().lower().replace("**", "").replace("^", "")
    if unit in ("m2 s-2", "m2/s2", "m2 s2", "m2s-2"):
        return z / G0 / 1000.0
    if unit in ("m", "meter", "metre", "meters", "metres", "gpm"):
        return z / 1000.0
    if unit == "km":
        return z
    if unit:
        logger.warning(f"Unrecognised geopotential unit '{units}'; guessing from magnitude")
    if np.nanmax(np.abs(z)) > 100_000:  # geopotential (m2 s-2)
        return z / G0 / 1000.0
    return z / 1000.0  # geopotential height (m)


def _altitude_km(profile: xr.Dataset, p_hpa: np.ndarray) -> np.ndarray:
    """Geometric altitude per level: geopotential when present, else hypsometric."""
    if "z" in profile.variables:
        z = np.asarray(profile["z"].values, dtype=float).ravel()
        return _z_to_km(z, profile["z"].attrs.get("units"))
    logger.debug("No geopotential in profile; hypsometric altitude estimate")
    return -7.0 * np.log(p_hpa / 1013.25)


def cloud_profiles(
    profile: xr.Dataset,
    reff_water_um: float = 10.0,
    reff_ice_um: float = 20.0,
    threshold_g_m3: float = 1e-4,
) -> Tuple[Optional[dict], Optional[dict]]:
    """Build ``wc_file`` / ``ic_file`` profile dicts from an ERA5 column.

    Converts ``clwc`` / ``ciwc`` (kg/kg) to g m⁻³ using the ideal-gas
    air density, places layer boundaries midway between model levels,
    and writes the values with libRadtran's 1D cloud-file semantics:
    the value at altitude *z* fills the layer **above** *z*, and the top
    row closes the grid with zero content.

    Parameters
    ----------
    profile : xarray.Dataset
        Single column, canonical names, ``pressure_level`` in hPa.
        Needs ``t`` plus at least one of ``clwc`` / ``ciwc``.
    reff_water_um, reff_ice_um : float
        Effective radii to assign (ERA5 has none).
    threshold_g_m3 : float, default ``1e-4``
        Layers below this water content count as cloud-free.

    Returns
    -------
    (wc, ic) : tuple of dict or None
        Each a ``{"z": [...], "lwc"/"iwc": [...], "reff": [...]}``
        mapping ready for a dict-valued ``wc_file`` / ``ic_file``
        parameter, or *None* when that phase has no cloud.
    """
    p = np.asarray(profile["pressure_level"].values, dtype=float).ravel()
    t = np.asarray(profile["t"].values, dtype=float).ravel()

    valid = ~(np.isnan(p) | np.isnan(t))
    results: Dict[str, Optional[dict]] = {"clwc": None, "ciwc": None}

    alt_all = _altitude_km(profile, p)

    for var, key, reff in (
        ("clwc", "lwc", reff_water_um),
        ("ciwc", "iwc", reff_ice_um),
    ):
        if var not in profile.variables:
            continue
        mmr = np.asarray(profile[var].values, dtype=float).ravel()
        ok = valid & ~np.isnan(mmr) & ~np.isnan(alt_all)
        if not ok.any():
            continue
        p_v, t_v, mmr_v, alt_v = p[ok], t[ok], mmr[ok], alt_all[ok]

        # ascending altitude
        order = np.argsort(alt_v)
        p_v, t_v, mmr_v, alt_v = p_v[order], t_v[order], mmr_v[order], alt_v[order]

        air_density = (p_v * 100.0) / (R_DRY * t_v)  # kg m-3
        content = mmr_v * air_density * 1000.0       # g m-3

        cloudy = content > threshold_g_m3
        if not cloudy.any():
            continue

        # contiguous index range spanning all cloudy levels
        idx = np.where(cloudy)[0]
        lo, hi = idx.min(), idx.max()
        levels = np.arange(lo, hi + 1)
        n = len(alt_v)

        # boundaries midway between levels; half-spacing at the ends
        def boundary_below(i):
            if i == 0:
                return max(alt_v[0] - (alt_v[1] - alt_v[0]) / 2 if n > 1
                           else alt_v[0] - 0.05, 0.0)
            return (alt_v[i - 1] + alt_v[i]) / 2

        def boundary_above(i):
            if i == n - 1:
                return alt_v[-1] + (alt_v[-1] - alt_v[-2]) / 2 if n > 1 \
                    else alt_v[-1] + 0.05
            return (alt_v[i] + alt_v[i + 1]) / 2

        # rows from cloud top down: top boundary closes the grid (zero),
        # each lower boundary carries the content of the layer above it.
        z_rows = [boundary_above(int(levels[-1]))]
        c_rows = [0.0]
        r_rows = [float(reff)]
        for i in levels[::-1]:
            z_rows.append(boundary_below(int(i)))
            c_rows.append(float(max(content[i], 0.0)))
            r_rows.append(float(reff))

        results[var] = {"z": z_rows, key: c_rows, "reff": r_rows}

    return results["clwc"], results["ciwc"]


def recommend_atmosphere(
    latitude: float,
    time: Union[str, datetime, np.datetime64],
) -> str:
    """Suggest the AFGL ``atmosphere_file`` shortname for a location/date.

    The background atmosphere still supplies everything a radiosonde
    profile does not: levels above the profile top and all trace gases
    without a dedicated column.  Matching it to the season and latitude
    band keeps those fill-ins consistent with the profile.

    Parameters
    ----------
    latitude : float
        Degrees north.
    time : str, datetime, or numpy.datetime64
        Date (month decides the season).

    Returns
    -------
    str
        One of ``"afglt"``, ``"afglms"``, ``"afglmw"``, ``"afglss"``,
        ``"afglsw"`` — accepted directly by
        ``paths.atmosphere_profile`` and the ``atmosphere_file`` option.
    """
    month = pd.to_datetime(time).month
    northern_summer = month in (4, 5, 6, 7, 8, 9)
    summer = northern_summer if latitude >= 0 else not northern_summer

    abs_lat = abs(latitude)
    if abs_lat < 30.0:
        return "afglt"
    if abs_lat < 60.0:
        return "afglms" if summer else "afglmw"
    return "afglss" if summer else "afglsw"


__all__ = [
    "normalize_era5",
    "select_profile",
    "write_atmosphere_file",
    "era5_atmosphere_file",
    "cloud_profiles",
    "recommend_atmosphere",
]
