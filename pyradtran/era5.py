r"""
ERA5 integration: normalisation, atmosphere files, and cloud profiles.

ERA5 data arrives under several naming conventions: the CDS NetCDF short
names (``t``, ``q``, ``o3``, ``clwc``, coordinate ``pressure_level``) and
the ARCO-ERA5 Zarr long names (``temperature``, ``specific_humidity``,
``geopotential``, coordinate ``level``).  :func:`normalize_era5` maps
either onto one canonical form.

From a normalised dataset this module produces, per (time, lat, lon)
point:

* a libRadtran ``radiosonde`` atmosphere file with pressure, temperature,
  humidity and, when available, ozone (:func:`write_atmosphere_file`);
* dict-valued ``wc_file`` / ``ic_file`` cloud profiles from the ERA5
  ``clwc`` / ``ciwc`` fields (:func:`cloud_profiles`).

:func:`recommend_atmosphere` suggests the AFGL background profile
(``atmosphere_file``) for a latitude and date.  The background still
matters with a radiosonde: libRadtran takes every level above the
radiosonde top, and every gas the radiosonde lacks, from it.

.. autosummary::

   normalize_era5
   select_profile
   write_atmosphere_file
   era5_atmosphere_file
   cloud_profiles
   recommend_atmosphere

Examples
--------
Passing the ERA5 dataset to
:meth:`~pyradtran.interface.PyRadtranAccessor.run` does all of it per
point (``era5`` is any pressure-level ERA5 dataset, ``ds`` the dataset of
points to simulate):

>>> result = ds.pyradtran.run(  # doctest: +SKIP
...     config_path=cfg, era5_atmosphere=era5, era5_clouds=True
... )

The same steps by hand, for one column:

>>> from pyradtran.era5 import (
...     normalize_era5, select_profile, write_atmosphere_file, cloud_profiles
... )
>>> profile = select_profile(  # doctest: +SKIP
...     normalize_era5(era5), latitude=78.2, longitude=15.6, time="2022-03-21T12"
... )
>>> path = write_atmosphere_file(profile, "work/era5_atm.dat")  # doctest: +SKIP
>>> wc, ic = cloud_profiles(profile)  # doctest: +SKIP

See Also
--------
pyradtran.interface.PyRadtranAccessor.run : ``era5_atmosphere=`` /
    ``era5_clouds=`` integration.
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
G0 = 9.80665  # standard gravity (m s-2)
R_DRY = 287.058  # specific gas constant, dry air (J kg-1 K-1)

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
    The input dataset is not modified.

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
        ``valid_time``.  Variables and coordinates that are not in the
        alias tables below (``latitude``, ``longitude``, ``u``, ...)
        pass through untouched.

    Raises
    ------
    InputGenerationError
        If the pressure coordinate carries a ``units`` attribute that is
        neither ``Pa`` / ``Pascal(s)`` nor one of ``hPa`` /
        ``hectopascal(s)`` / ``millibar(s)`` / ``mb`` / ``mbar`` (case
        is ignored).

    See Also
    --------
    select_profile : Pick one column from the normalised dataset.
    era5_atmosphere_file : Normalise, select and write in one call.

    Notes
    -----
    Accepted names (``VAR_ALIASES`` and ``COORD_ALIASES`` of this module):

    ================== ======================================== =========
    Canonical          Also accepted                            Units set
    ================== ======================================== =========
    ``t``              ``temperature``                          K
    ``q``              ``specific_humidity``                    kg kg-1
    ``z``              ``geopotential``                         m2 s-2
    ``o3``             ``ozone_mass_mixing_ratio``              kg kg-1
    ``clwc``           ``specific_cloud_liquid_water_content``  kg kg-1
    ``ciwc``           ``specific_cloud_ice_water_content``     kg kg-1
    ``cc``             ``fraction_of_cloud_cover``              —
    ``skt``            ``skin_temperature``                     K
    ``sst``            ``sea_surface_temperature``              K
    ``pressure_level`` ``level``, ``plev``, ``isobaricInhPa``   hPa
    ``valid_time``     ``time``                                 —
    ================== ======================================== =========

    * A long name is renamed only when the canonical name is absent; if
      both exist the canonical one wins and the alias is left in place.
    * "Units set" is the ``units`` attribute written when the variable
      has none.  An existing attribute is never overwritten and the data
      are never rescaled, except for the pressure coordinate.
    * The pressure coordinate is divided by 100 when its ``units``
      attribute is ``Pa`` (or ``Pascal``, in any case), or when it has
      no ``units`` and its largest value exceeds 2000 (no atmospheric
      pressure level in hPa is that large).  Its ``units`` attribute is
      always set to ``hPa`` and its values are cast to ``float``.

    Examples
    --------
    A two-level ARCO-ERA5-flavoured column (long names, pressure in Pa,
    no unit attributes on the variables):

    >>> import numpy as np
    >>> import xarray as xr
    >>> from pyradtran.era5 import normalize_era5
    >>> arco = xr.Dataset(
    ...     {
    ...         "temperature": ("level", [268.0, 284.0]),
    ...         "specific_humidity": ("level", [1.5e-3, 5.0e-3]),
    ...         "geopotential": ("level", [29500.0, 1000.0]),
    ...     },
    ...     coords={"level": ("level", [70000, 100000], {"units": "Pa"})},
    ... )
    >>> ds = normalize_era5(arco)
    >>> sorted(ds.data_vars)
    ['q', 't', 'z']
    >>> ds.pressure_level.values
    array([ 700., 1000.])
    >>> ds.pressure_level.attrs["units"], ds.t.attrs["units"], ds.z.attrs["units"]
    ('hPa', 'K', 'm2 s-2')

    The same column as a CDS download (short names, ``pressure_level``
    in hPa) ends up identical:

    >>> cds = xr.Dataset(
    ...     {
    ...         "t": ("pressure_level", [268.0, 284.0]),
    ...         "q": ("pressure_level", [1.5e-3, 5.0e-3]),
    ...         "z": ("pressure_level", [29500.0, 1000.0]),
    ...     },
    ...     coords={"pressure_level": [700.0, 1000.0]},
    ... )
    >>> normalize_era5(cds).identical(ds)
    True

    The call is idempotent; an unknown pressure unit is refused, not
    guessed:

    >>> normalize_era5(ds).identical(ds)
    True
    >>> bad = cds.copy()
    >>> bad.pressure_level.attrs["units"] = "bar"
    >>> normalize_era5(bad)
    Traceback (most recent call last):
        ...
    pyradtran.exceptions.InputGenerationError: Unsupported pressure unit 'bar' in ERA5 dataset; expected 'Pa' or 'hPa'

    The alias tables are module-level dicts:

    >>> from pyradtran.era5 import COORD_ALIASES
    >>> COORD_ALIASES["pressure_level"]
    ('pressure_level', 'level', 'plev', 'isobaricInhPa')
    """
    rename = {}
    for table, present in (
        (VAR_ALIASES, ds.variables),
        (COORD_ALIASES, set(ds.coords) | set(ds.dims)),
    ):
        for canonical, aliases in table.items():
            if canonical in present:
                continue
            for alias in aliases[1:]:
                if alias in present:
                    rename[alias] = canonical
                    break
    if rename:
        ds = ds.rename(rename)
        logger.debug(f"normalize_era5 renamed: {rename}")

    if "pressure_level" in ds.coords:
        plev = ds["pressure_level"]
        unit = str(plev.attrs.get("units", "")).strip()
        values = np.asarray(plev.values, dtype=float)
        if unit.lower() in ("pa", "pascal", "pascals") or (
            unit == "" and values.max() > 2000.0
        ):
            ds = ds.assign_coords(pressure_level=values / 100.0)
        elif "hpa" in unit.lower() or unit.lower() in (
            "",
            "hpa",
            "hectopascal",
            "hectopascals",
            "millibars",
            "millibar",
            "mb",
            "mbar",
        ):
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

    Works on datasets with spatial/time dimensions and on ones already
    reduced to a single column.  Selection is nearest-neighbour without
    a distance tolerance: a point outside the domain or time range
    returns the closest edge column.  A warning is logged when that
    column is more than 1° from the requested position.

    Parameters
    ----------
    ds : xarray.Dataset
        ERA5 dataset on canonical names (see :func:`normalize_era5`);
        the function looks for the dimensions ``latitude``,
        ``longitude``, ``valid_time`` and ``pressure_level`` and does
        not normalise names itself.
    latitude, longitude : float, optional
        Target position in degrees north and east.  The longitude may
        be given in either convention (0…360 or −180…180); it is wrapped
        into the one the dataset uses.  Required when *ds* has a
        ``latitude`` or ``longitude`` dimension, ignored otherwise.
    time : str, datetime.datetime or numpy.datetime64, optional
        Target time, parsed with :func:`pandas.to_datetime`.  Required
        when *ds* has a ``valid_time`` dimension, ignored otherwise.

    Returns
    -------
    xarray.Dataset
        Profile with ``pressure_level`` as the only remaining dimension,
        loaded into memory.  The selected ``latitude`` / ``longitude`` /
        ``valid_time`` remain as scalar coordinates.

    Raises
    ------
    ValueError
        If *ds* has spatial dimensions but *latitude* or *longitude* is
        *None*, or has a ``valid_time`` dimension but *time* is *None*.

    See Also
    --------
    normalize_era5 : Bring any ERA5 flavour onto the names expected here.
    write_atmosphere_file, cloud_profiles : Consumers of the profile.

    Notes
    -----
    Any other dimension still present after the selection (an ensemble
    ``number``, ``expver``, ...) is reduced by taking its **first**
    element.

    A dataset whose largest longitude exceeds 180 is taken to use
    0…360, any other −180…180.  The nearest-neighbour search is not
    cyclic: on a global 0…360 grid a request just west of 0° picks the
    last column, not the first (at most one grid cell off).

    Examples
    --------
    >>> import numpy as np
    >>> import xarray as xr
    >>> from pyradtran.era5 import select_profile
    >>> era5 = xr.Dataset(
    ...     {
    ...         "t": (
    ...             ("valid_time", "pressure_level", "latitude", "longitude"),
    ...             250.0 + np.arange(2 * 3 * 3 * 4).reshape(2, 3, 3, 4),
    ...         )
    ...     },
    ...     coords={
    ...         "valid_time": np.array(
    ...             ["2024-07-15T00", "2024-07-15T06"], dtype="datetime64[ns]"
    ...         ),
    ...         "pressure_level": [500.0, 850.0, 1000.0],
    ...         "latitude": [79.0, 78.5, 78.0],
    ...         "longitude": [14.0, 15.0, 16.0, 17.0],
    ...     },
    ... )
    >>> profile = select_profile(
    ...     era5, latitude=78.4, longitude=15.2, time="2024-07-15T05:00"
    ... )
    >>> profile.t.dims
    ('pressure_level',)
    >>> float(profile.latitude), float(profile.longitude)
    (78.5, 15.0)
    >>> str(profile.valid_time.values)[:13]
    '2024-07-15T06'
    >>> profile.t.values
    array([291., 303., 315.])

    The longitude convention of the request does not matter (15.2° E is
    15.2 − 360 = −344.8):

    >>> float(select_profile(era5, 78.4, -344.8, "2024-07-15T05:00").longitude)
    15.0

    A gridded dataset needs a position:

    >>> select_profile(era5, time="2024-07-15T05:00")
    Traceback (most recent call last):
        ...
    ValueError: Dataset has spatial dimensions; latitude and longitude required
    """
    if "latitude" in ds.dims or "longitude" in ds.dims:
        if latitude is None or longitude is None:
            raise ValueError(
                "Dataset has spatial dimensions; latitude and longitude required"
            )
        if "longitude" in ds.coords and ds["longitude"].size:
            # wrap the request into the dataset's convention
            # ponytail: sel is not cyclic across the 0/360 seam (<= 1 cell off)
            if float(ds["longitude"].max()) > 180.0:
                longitude = longitude % 360.0
            else:
                longitude = (longitude + 180.0) % 360.0 - 180.0
        ds = ds.sel(latitude=latitude, longitude=longitude, method="nearest")
        dlat = abs(float(ds["latitude"]) - latitude)
        dlon = abs(float(ds["longitude"]) - longitude) % 360.0
        dlon = min(dlon, 360.0 - dlon)
        if max(dlat, dlon) > 1.0:
            logger.warning(
                f"Nearest ERA5 column ({float(ds['latitude']):.2f}, "
                f"{float(ds['longitude']):.2f}) is more than 1 deg from the "
                f"requested position ({latitude:.2f}, {longitude:.2f})"
            )

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
    r"""Write a profile as a libRadtran ``radiosonde`` atmosphere file.

    Columns are pressure (hPa), temperature (K), specific humidity
    (MMR, kg/kg) and, when ``o3`` is present and *include_ozone* is
    true, ozone MMR.  A header line (``# columns: H2O MMR O3 MMR``)
    records the layout; the input builder reads it back to construct the
    matching ``radiosonde`` option line.

    Levels with NaN pressure/temperature/humidity (e.g. ERA5 levels
    below the surface) are dropped; NaN ozone values become ``-1``,
    which tells libRadtran to fall back to the ``atmosphere_file`` value
    at that level.

    Parameters
    ----------
    profile : xarray.Dataset
        Single column, canonical names (see :func:`normalize_era5`),
        with ``pressure_level`` in hPa and variables ``t`` (K) and ``q``
        (kg kg\ :sup:`-1`); optionally ``o3`` (kg kg\ :sup:`-1`).  All
        other variables are ignored.  Typically the output of
        :func:`select_profile`.
    output_filepath : str or pathlib.Path
        Destination file.  Missing parent directories are created and an
        existing file is overwritten.
    include_ozone : bool, default ``True``
        Write the ozone column when ``o3`` is available.  With ``False``
        (or without ``o3``) the ozone profile comes entirely from the
        background ``atmosphere_file``.

    Returns
    -------
    pathlib.Path
        *output_filepath* as a :class:`~pathlib.Path`.

    Raises
    ------
    InputGenerationError
        If ``t`` or ``q`` is missing, there is no ``pressure_level``
        coordinate, or no level has finite pressure, temperature and
        humidity.

    See Also
    --------
    era5_atmosphere_file : Normalise, select and write in one call.
    recommend_atmosphere : Matching background ``atmosphere_file``.
    cloud_profiles : Cloud profiles from the same column.

    Notes
    -----
    **File format.**  Three comment lines, then one row per level in
    order of increasing pressure (top of atmosphere first), which is the
    order libRadtran's ``radiosonde`` option expects::

        # ERA5 atmosphere profile (libRadtran radiosonde format)
        # columns: H2O MMR O3 MMR
        # p(hPa)  T(K)  H2O_MMR_O3_MMR
        <p>  <T>  <q>  <o3>

    ====== ====================== ================== ==========
    Column Quantity               Unit               Format
    ====== ====================== ================== ==========
    1      pressure               hPa                ``%.2f``
    2      temperature            K                  ``%.2f``
    3      H\ :sub:`2`\ O MMR     kg kg\ :sup:`-1`   ``%.4e``
    4      O\ :sub:`3` MMR        kg kg\ :sup:`-1`   ``%.4e``
    ====== ====================== ================== ==========

    Column 4 is present only when ozone is written; otherwise the second
    header line reads ``# columns: H2O MMR``.  The input builder copies
    that string verbatim into the uvspec input::

        radiosonde <file> H2O MMR O3 MMR

    **Level handling.**  Rows are sorted by pressure regardless of the
    order in *profile*.  Levels whose pressure does not strictly
    increase after sorting (duplicates) are dropped, because libRadtran
    requires a strictly monotonic grid.  The ``%.2f`` format resolves
    pressure to 0.01 hPa.

    **Humidity.**  No conversion is applied: the ERA5 specific humidity
    :math:`q` (mass of water vapour per mass of moist air) is written
    as is and declared to libRadtran as a mass mixing ratio (``MMR``).
    The mixing ratio with respect to dry air would be

    .. math::

        w = \frac{q}{1 - q}

    which differs from :math:`q` by the relative amount
    :math:`q/(1-q)`: below 2 % even for a very moist tropical boundary
    layer (:math:`q \approx 0.02`) and negligible in the upper
    troposphere.  Ozone is treated the same way.

    Examples
    --------
    A three-level column with one missing ozone value, given bottom-up
    on purpose:

    >>> import pathlib, tempfile
    >>> import numpy as np
    >>> import xarray as xr
    >>> from pyradtran.era5 import write_atmosphere_file
    >>> profile = xr.Dataset(
    ...     {
    ...         "t": ("pressure_level", [284.0, 276.0, 252.0]),
    ...         "q": ("pressure_level", [5.0e-3, 3.0e-3, 4.0e-4]),
    ...         "o3": ("pressure_level", [5.0e-8, np.nan, 9.0e-8]),
    ...     },
    ...     coords={"pressure_level": [1000.0, 850.0, 500.0]},
    ... )
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     path = write_atmosphere_file(profile, pathlib.Path(tmp) / "era5_atm.dat")
    ...     print(path.read_text(), end="")
    # ERA5 atmosphere profile (libRadtran radiosonde format)
    # columns: H2O MMR O3 MMR
    # p(hPa)  T(K)  H2O_MMR_O3_MMR
    500.00  252.00  4.0000e-04  9.0000e-08
    850.00  276.00  3.0000e-03  -1.0000e+00
    1000.00  284.00  5.0000e-03  5.0000e-08

    Without the ozone column:

    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     path = write_atmosphere_file(
    ...         profile, pathlib.Path(tmp) / "era5_atm.dat", include_ozone=False
    ...     )
    ...     print(path.read_text(), end="")
    # ERA5 atmosphere profile (libRadtran radiosonde format)
    # columns: H2O MMR
    # p(hPa)  T(K)  H2O_MMR
    500.00  252.00  4.0000e-04
    850.00  276.00  3.0000e-03
    1000.00  284.00  5.0000e-03
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
    o3 = np.asarray(profile["o3"].values, dtype=float).ravel() if has_o3 else None

    valid = ~(np.isnan(p) | np.isnan(t) | np.isnan(q))
    if not valid.any():
        raise InputGenerationError("No valid (non-NaN) levels in ERA5 profile")

    # TOA first: ascending pressure; drop duplicate levels (libRadtran
    # requires a strictly monotonic grid).
    sel = np.flatnonzero(valid)
    sel = sel[np.argsort(p[sel])]
    sel = sel[np.concatenate(([True], np.diff(p[sel]) > 0))]
    p, t, q = p[sel], t[sel], q[sel]
    if o3 is not None:
        o3 = np.where(np.isnan(o3[sel]), -1.0, o3[sel])

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
    """Normalise, select a point and write the atmosphere file.

    Chains :func:`normalize_era5`, :func:`select_profile` and
    :func:`write_atmosphere_file`, as
    :meth:`~pyradtran.interface.PyRadtranAccessor.run` does for every
    point when given ``era5_atmosphere=``.

    Parameters
    ----------
    era5_ds : xarray.Dataset
        ERA5 dataset on pressure levels in any supported naming flavour
        (see :func:`normalize_era5`).  Needs temperature and specific
        humidity; ozone is used when present.
    latitude, longitude : float
        Position of the column (nearest grid point is taken; the
        longitude may be in 0…360 or −180…180, see
        :func:`select_profile`).
    time : str, datetime.datetime or numpy.datetime64
        Time of the column (nearest time step is taken).
    output_filepath : str or pathlib.Path
        Destination file.
    include_ozone : bool, default ``True``
        Write the ozone column when the dataset has ozone.

    Returns
    -------
    pathlib.Path
        Path of the written file.

    Raises
    ------
    InputGenerationError
        On any failure.  Errors of the three steps that are not already
        an :class:`~pyradtran.exceptions.InputGenerationError` (a
        missing coordinate, an unparsable time, an unwritable path, ...)
        are wrapped, with the original exception chained as
        ``__cause__``.

    See Also
    --------
    write_atmosphere_file : File format and level handling.
    recommend_atmosphere : Matching background ``atmosphere_file``.

    Examples
    --------
    An ARCO-ERA5-flavoured cube (long names, ``time`` / ``level``):

    >>> import pathlib, tempfile
    >>> import numpy as np
    >>> import xarray as xr
    >>> from pyradtran.era5 import era5_atmosphere_file
    >>> shape = (2, 3, 2, 2)
    >>> t = np.broadcast_to(np.array([252.0, 276.0, 284.0])[:, None, None], shape)
    >>> q = np.broadcast_to(np.array([4e-4, 3e-3, 5e-3])[:, None, None], shape)
    >>> dims = ("time", "level", "latitude", "longitude")
    >>> era5 = xr.Dataset(
    ...     {"temperature": (dims, t), "specific_humidity": (dims, q)},
    ...     coords={
    ...         "time": np.array(
    ...             ["2024-07-15T00", "2024-07-15T06"], dtype="datetime64[ns]"
    ...         ),
    ...         "level": [500, 850, 1000],
    ...         "latitude": [78.5, 78.0],
    ...         "longitude": [15.0, 16.0],
    ...     },
    ... )
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     path = era5_atmosphere_file(
    ...         era5, 78.2, 15.6, "2024-07-15T05:00", pathlib.Path(tmp) / "atm.dat"
    ...     )
    ...     print(path.read_text(), end="")
    # ERA5 atmosphere profile (libRadtran radiosonde format)
    # columns: H2O MMR
    # p(hPa)  T(K)  H2O_MMR
    500.00  252.00  4.0000e-04
    850.00  276.00  3.0000e-03
    1000.00  284.00  5.0000e-03
    """
    try:
        ds = normalize_era5(era5_ds)
        profile = select_profile(ds, latitude, longitude, time)
        return write_atmosphere_file(profile, output_filepath, include_ozone)
    except InputGenerationError:
        raise
    except Exception as e:
        raise InputGenerationError(f"Failed to create ERA5 atmosphere file: {e}") from e


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
        logger.warning(
            f"Unrecognised geopotential unit '{units}'; guessing from magnitude"
        )
    if np.nanmax(np.abs(z)) > 100_000:  # geopotential (m2 s-2)
        return z / G0 / 1000.0
    return z / 1000.0  # geopotential height (m)


def _altitude_km(profile: xr.Dataset, p_hpa: np.ndarray) -> np.ndarray:
    """Altitude (km) per level: from ``z`` when present, else from pressure.

    With a ``z`` variable the values go through :func:`_z_to_km`.
    Otherwise a constant 7 km scale height is assumed,
    ``-7 * ln(p / 1013.25 hPa)``.
    """
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
    r"""Build ``wc_file`` / ``ic_file`` profile dicts from an ERA5 column.

    Converts ``clwc`` / ``ciwc`` (kg/kg) to g m⁻³ using the ideal-gas
    air density, places layer boundaries midway between model levels,
    and writes the values with libRadtran's 1D cloud-file semantics:
    the value at altitude *z* fills the layer **above** *z*, and the top
    row closes the grid with zero content.

    Parameters
    ----------
    profile : xarray.Dataset
        Single column, canonical names (see :func:`normalize_era5`,
        :func:`select_profile`), ``pressure_level`` in hPa.  Needs ``t``
        (K) plus at least one of ``clwc`` / ``ciwc``
        (kg kg\ :sup:`-1`).  ``z`` (geopotential) is used for the
        altitudes when present.
    reff_water_um : float, default ``10.0``
        Effective radius (µm) assigned to every liquid row; ERA5
        carries no particle size.
    reff_ice_um : float, default ``20.0``
        Effective radius (µm) assigned to every ice row.
    threshold_g_m3 : float, default ``1e-4``
        Levels at or below this water content (g m\ :sup:`-3`) count as
        cloud-free.

    Returns
    -------
    wc : dict or None
        ``{"z": [...], "lwc": [...], "reff": [...]}``: altitude (km),
        liquid water content (g m\ :sup:`-3`) and effective radius (µm)
        as plain lists, ordered **top-down**.  *None* when ``clwc`` is
        absent or no level exceeds the threshold.
    ic : dict or None
        Same with key ``"iwc"`` for ice, from ``ciwc``.

    Raises
    ------
    InputGenerationError
        If *profile* has no ``pressure_level`` or no ``t``.

    See Also
    --------
    pyradtran.interface.PyRadtranAccessor.run : ``era5_clouds=`` calls
        this function per point; a dict passes keyword overrides, e.g.
        ``era5_clouds={"reff_ice_um": 35.0}``.
    pyradtran.core.Simulation.format_cloud_profile : Turns the dict into
        the three-column file libRadtran reads.
    write_atmosphere_file : Thermodynamic profile of the same column.

    Notes
    -----
    **Water content.**  ERA5 gives the cloud water as a mass mixing
    ratio :math:`q_c` (kg of condensate per kg of air).  With the dry-air
    ideal-gas density at pressure :math:`p` (hPa) and temperature
    :math:`T` (K)

    .. math::

        \rho_\mathrm{air} = \frac{100\,p}{R_d\,T},
        \qquad R_d = 287.058\ \mathrm{J\,kg^{-1}\,K^{-1}},

    the volumetric water content in g m\ :sup:`-3` is

    .. math::

        \mathrm{LWC} = 1000\; q_c\; \rho_\mathrm{air}

    (and likewise IWC from ``ciwc``).  No virtual-temperature correction
    is applied, and the grid-box mean is used as is (not divided by the
    cloud fraction ``cc``).  Negative values are clipped to zero.

    **Altitude.**  When *profile* has ``z`` it is converted according to
    its ``units`` attribute: a geopotential :math:`\Phi`
    (``m2 s-2``, ``m**2 s**-2``, ...) gives the geopotential height

    .. math::

        z = \frac{\Phi}{g_0}, \qquad g_0 = 9.80665\ \mathrm{m\,s^{-2}},

    while ``m`` / ``gpm`` / ``km`` are taken as heights.  Without a
    recognised unit, values above 10\ :sup:`5` are assumed to be a
    geopotential, anything else metres.  Without ``z`` a constant 7 km
    scale height is used, :math:`z = -7\,\mathrm{km}\,\ln(p / 1013.25\,
    \mathrm{hPa})`, which is only a rough estimate.  Altitudes are above
    mean sea level in both cases, not above the local surface.

    **Layers.**  Levels with NaN pressure, temperature, altitude or
    water content are discarded and the rest sorted by altitude,
    :math:`z_0 < z_1 < \dots < z_{n-1}`.  Each level :math:`i`
    represents the layer between the midpoints to its neighbours,

    .. math::

        b_{i-1/2} = \tfrac12 (z_{i-1} + z_i), \qquad
        b_{i+1/2} = \tfrac12 (z_i + z_{i+1}),

    with the outermost layers extended by half the adjacent spacing
    (the lowest boundary is not allowed below 0 km; a single-level
    profile gets ±0.05 km).  If :math:`i_\mathrm{lo}` and
    :math:`i_\mathrm{hi}` are the lowest and highest level above the
    threshold, the returned rows are, from the top down,

    .. math::

        (b_{i_\mathrm{hi}+1/2},\ 0), \quad
        (b_{i_\mathrm{hi}-1/2},\ c_{i_\mathrm{hi}}), \quad \dots, \quad
        (b_{i_\mathrm{lo}-1/2},\ c_{i_\mathrm{lo}})

    so that every content value :math:`c_i` sits at the lower boundary
    of its layer and the zero on top closes the grid, which is how
    libRadtran reads a ``wc_file 1D`` / ``ic_file 1D``.  One contiguous
    block is written per phase: levels between two cloud decks keep
    their own (sub-threshold) content instead of being set to zero.

    Examples
    --------
    A five-level column with liquid water at 850 and 925 hPa and a
    little ice at 500 hPa:

    >>> import numpy as np
    >>> import xarray as xr
    >>> from pyradtran.era5 import cloud_profiles
    >>> lev = "pressure_level"
    >>> profile = xr.Dataset(
    ...     {
    ...         "t": (lev, [252.0, 268.0, 276.0, 280.0, 284.0]),
    ...         "z": (
    ...             lev,
    ...             [54000.0, 29500.0, 14500.0, 7500.0, 1000.0],
    ...             {"units": "m2 s-2"},
    ...         ),
    ...         "clwc": (lev, [0.0, 0.0, 2.0e-4, 3.0e-4, 0.0]),
    ...         "ciwc": (lev, [2.0e-5, 0.0, 0.0, 0.0, 0.0]),
    ...     },
    ...     coords={lev: [500.0, 700.0, 850.0, 925.0, 1000.0]},
    ... )
    >>> wc, ic = cloud_profiles(profile)
    >>> np.round(wc["z"], 3).tolist()
    [2.243, 1.122, 0.433]
    >>> np.round(wc["lwc"], 3).tolist()
    [0.0, 0.215, 0.345]
    >>> wc["reff"]
    [10.0, 10.0, 10.0]

    Read from the bottom: 0.345 g m\ :sup:`-3` between 0.433 and
    1.122 km (the 925 hPa level at 0.765 km), 0.215 g m\ :sup:`-3` up to
    2.243 km (the 850 hPa level at 1.479 km), nothing above.  The ice
    cloud is a single layer around the 500 hPa level:

    >>> np.round(ic["z"], 3).tolist(), np.round(ic["iwc"], 4).tolist()
    ([6.756, 4.257], [0.0, 0.0138])

    Raising the threshold removes the thin ice cloud, and other
    effective radii can be assigned:

    >>> wc, ic = cloud_profiles(profile, reff_water_um=8.0, threshold_g_m3=0.05)
    >>> ic is None, wc["reff"]
    (True, [8.0, 8.0, 8.0])

    The dicts are valid ``wc_file`` / ``ic_file`` parameter values:

    >>> result = ds.pyradtran.run(  # doctest: +SKIP
    ...     config_path=cfg, params={"wc_file": wc}
    ... )

    A deeper synthetic column.  Markers are the ERA5 level values, steps
    the layers libRadtran sees:

    .. plot::

        from pyradtran.era5 import cloud_profiles

        p = np.array([300, 400, 500, 600, 700, 775, 850, 900, 950, 1000.0])
        z_km = -7.6 * np.log(p / 1013.25)
        t = 288.0 - 6.5 * z_km
        clwc = np.array([0, 0, 0, 0, 0.3, 1.6, 2.6, 1.2, 0, 0]) * 1e-4
        ciwc = np.array([0.15, 0.5, 0.35, 0.1, 0, 0, 0, 0, 0, 0]) * 1e-4
        lev = "pressure_level"
        profile = xr.Dataset(
            {
                "t": (lev, t),
                "z": (lev, z_km * 9806.65, {"units": "m2 s-2"}),
                "clwc": (lev, clwc),
                "ciwc": (lev, ciwc),
            },
            coords={lev: p},
        )
        wc, ic = cloud_profiles(profile)
        rho = 100.0 * p / (287.058 * t)

        fig, ax = plt.subplots(figsize=(5.5, 4.2))
        for prof, key, mmr, color, label in (
            (wc, "lwc", clwc, "C0", "liquid (wc_file)"),
            (ic, "iwc", ciwc, "C1", "ice (ic_file)"),
        ):
            edges = prof["z"][::-1]
            values = prof[key][:0:-1]
            ax.stairs(values, edges, orientation="horizontal", color=color,
                      fill=True, alpha=0.3)
            ax.stairs(values, edges, orientation="horizontal", color=color,
                      label=label)
            ax.plot(1000.0 * mmr * rho, z_km, "o", color=color, ms=4)
        ax.set_xlabel("water content (g m$^{-3}$)")
        ax.set_ylabel("altitude (km)")
        ax.set_ylim(0, 11)
        ax.legend()
        plt.tight_layout()
    """
    for name in ("pressure_level", "t"):
        if name not in profile.variables:
            raise InputGenerationError(
                f"ERA5 profile has no '{name}'; cloud profiles need "
                f"'pressure_level' and 't' (see normalize_era5)"
            )
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
        # ascending altitude
        sel = np.flatnonzero(ok)
        sel = sel[np.argsort(alt_all[sel])]
        p_v, t_v, mmr_v, alt_v = p[sel], t[sel], mmr[sel], alt_all[sel]

        air_density = (p_v * 100.0) / (R_DRY * t_v)  # kg m-3
        content = mmr_v * air_density * 1000.0  # g m-3

        cloudy = content > threshold_g_m3
        if not cloudy.any():
            continue

        # contiguous index range spanning all cloudy levels
        idx = np.flatnonzero(cloudy)
        levels = range(int(idx[-1]), int(idx[0]) - 1, -1)  # top down
        n = len(alt_v)

        # boundaries midway between levels; half-spacing at the ends
        def boundary_below(i):
            if i == 0:
                return max(
                    alt_v[0] - (alt_v[1] - alt_v[0]) / 2 if n > 1 else alt_v[0] - 0.05,
                    0.0,
                )
            return (alt_v[i - 1] + alt_v[i]) / 2

        def boundary_above(i):
            if i == n - 1:
                return (
                    alt_v[-1] + (alt_v[-1] - alt_v[-2]) / 2
                    if n > 1
                    else alt_v[-1] + 0.05
                )
            return (alt_v[i] + alt_v[i + 1]) / 2

        # rows from cloud top down: top boundary closes the grid (zero),
        # each lower boundary carries the content of the layer above it.
        z_rows = [boundary_above(levels[0])] + [boundary_below(i) for i in levels]
        c_rows = [0.0] + [float(max(content[i], 0.0)) for i in levels]
        r_rows = [float(reff)] * len(z_rows)

        results[var] = {"z": z_rows, key: c_rows, "reff": r_rows}

    return results["clwc"], results["ciwc"]


def recommend_atmosphere(
    latitude: float,
    time: Union[str, datetime, np.datetime64],
) -> str:
    r"""Suggest the AFGL ``atmosphere_file`` shortname for a location/date.

    The background atmosphere supplies everything a radiosonde profile
    does not: levels above the profile top and all trace gases without a
    dedicated column.  Matching it to the season and latitude band keeps
    those fill-ins consistent with the profile.

    Parameters
    ----------
    latitude : float
        Degrees north (negative in the southern hemisphere).
    time : str, datetime.datetime or numpy.datetime64
        Date, parsed with :func:`pandas.to_datetime`.  Only the month is
        used.

    Returns
    -------
    str
        One of ``"afglt"``, ``"afglms"``, ``"afglmw"``, ``"afglss"``,
        ``"afglsw"``; accepted directly by ``paths.atmosphere_profile``
        and the ``atmosphere_file`` option.

    See Also
    --------
    write_atmosphere_file : The radiosonde file the background completes.
    pyradtran.config.list_atmosphere_profiles : Profiles shipped with
        libRadtran.

    Notes
    -----
    A lookup on :math:`|\varphi|` (absolute latitude) and the half-year:

    ================================= ================== ==================
    Latitude band                     Summer half-year   Winter half-year
    ================================= ================== ==================
    :math:`|\varphi| < 30°`           ``afglt``          ``afglt``
    :math:`30° \le |\varphi| < 60°`   ``afglms``         ``afglmw``
    :math:`|\varphi| \ge 60°`         ``afglss``         ``afglsw``
    ================================= ================== ==================

    "Summer" is April–September for ``latitude >= 0`` and
    October–March for ``latitude < 0``.  The band edges belong to the
    poleward band (exactly 30° is mid-latitude, exactly 60° subarctic).
    The US standard atmosphere (``afglus``) is never returned.

    Examples
    --------
    >>> from pyradtran.era5 import recommend_atmosphere
    >>> recommend_atmosphere(78.2, "2024-07-15")  # Svalbard, summer
    'afglss'
    >>> recommend_atmosphere(78.2, "2024-01-15")  # polar night
    'afglsw'
    >>> recommend_atmosphere(48.1, "2024-10-01")  # Munich, October
    'afglmw'
    >>> recommend_atmosphere(-33.9, "2024-01-15")  # Cape Town: austral summer
    'afglms'
    >>> recommend_atmosphere(5.0, "2024-01-15")  # tropics, any month
    'afglt'

    Any date-like object works:

    >>> import numpy as np
    >>> recommend_atmosphere(-75.0, np.datetime64("2024-07-01T12:00"))
    'afglsw'

    As a parameter of a run that takes its lower atmosphere from ERA5:

    >>> result = ds.pyradtran.run(  # doctest: +SKIP
    ...     config_path=cfg,
    ...     era5_atmosphere=era5,
    ...     params={"atmosphere_file": recommend_atmosphere(78.2, "2024-07-15")},
    ... )
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
