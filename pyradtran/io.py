# pyradtran/io.py
"""
Input / output layer for pyRadtran.

Everything that crosses the boundary between Python and files lives here:
reading the plain-text tables ``uvspec`` writes, turning them into labelled
:mod:`xarray` objects, saving results as NetCDF, and preparing input
tables and atmosphere files.

Reading ``uvspec`` output:

.. autosummary::

   OutputParser
   ParsedOutput
   OutputType
   OutputToXarray
   NetCDFSaver

Preparing input:

.. autosummary::

   InputDataLoader
   ERA5AtmosphereGenerator
   RadiosondeAtmosphereGenerator

Most users never call these classes directly —
:meth:`~pyradtran.interface.PyRadtranAccessor.run` chains them for every
point of a dataset.  They are useful on their own to re-read ``.out``
files kept in the working directory, or to build custom pipelines.

Notes
-----
The data flow of one batch is

.. code-block:: text

    uvspec .out file --OutputParser.parse_output_file--> ParsedOutput
    [ParsedOutput, ...] --OutputToXarray.convert_batch--> xarray.Dataset
    xarray.Dataset --NetCDFSaver.save_results_to_netcdf--> .nc file

``uvspec`` output written with ``output_user`` is a headerless table with
one row per (wavelength, output altitude) pair.  The parser gets the
column names from the configuration (the same
:func:`~pyradtran.input_builder.effective_output_columns` that wrote the
``output_user`` line), so file and parser always agree.

Examples
--------
Parse an output file by hand and convert it (see
:meth:`OutputParser.parse_output_file` for a runnable example):

>>> from pyradtran import OutputParser, OutputToXarray, load_config
>>> cfg = load_config("config/my_simulation.yaml")  # doctest: +SKIP
>>> parsed = OutputParser(cfg).parse_output_file(out_file)  # doctest: +SKIP
>>> result = OutputToXarray.convert_batch([parsed], ds)  # doctest: +SKIP

See Also
--------
pyradtran.core.Simulation : Generates input files and calls ``uvspec``.
pyradtran.interface : High-level batch driver.
pyradtran.era5 : ERA5 atmosphere and cloud profiles.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import xarray as xr

from .config import SimulationConfig
from .exceptions import InputGenerationError, OutputParsingError
from .input_builder import effective_output_altitudes, effective_output_columns

logger = logging.getLogger(__name__)


class OutputType(Enum):
    """Layout of a ``uvspec`` output table.

    The four members encode two binary choices: spectrally integrated vs.
    spectral, and one vs. several output altitudes.  The layout determines
    how many rows the file has and which axis columns it carries:

    ==============================  ======================  =============
    Member                          Rows                    Axis columns
    ==============================  ======================  =============
    ``INTEGRATED_SINGLE_ALTITUDE``  1                       —
    ``INTEGRATED_MULTI_ALTITUDE``   ``n_alt``               ``zout``
    ``SPECTRAL_SINGLE_ALTITUDE``    ``n_wl``                ``lambda``
    ``SPECTRAL_MULTI_ALTITUDE``     ``n_wl * n_alt``        ``zout lambda``
    ==============================  ======================  =============

    See Also
    --------
    OutputParser : Decides the type from the configuration.

    Examples
    --------
    >>> from pyradtran import OutputType
    >>> OutputType.SPECTRAL_MULTI_ALTITUDE.value
    'spectral_multi_altitude'
    >>> [t.name for t in OutputType]  # doctest: +NORMALIZE_WHITESPACE
    ['INTEGRATED_SINGLE_ALTITUDE', 'INTEGRATED_MULTI_ALTITUDE',
     'SPECTRAL_SINGLE_ALTITUDE', 'SPECTRAL_MULTI_ALTITUDE']
    """

    INTEGRATED_SINGLE_ALTITUDE = "integrated_single_altitude"
    INTEGRATED_MULTI_ALTITUDE = "integrated_multi_altitude"
    SPECTRAL_SINGLE_ALTITUDE = "spectral_single_altitude"
    SPECTRAL_MULTI_ALTITUDE = "spectral_multi_altitude"


@dataclass
class ParsedOutput:
    """One parsed ``uvspec`` output file (one simulated point).

    Returned by :meth:`OutputParser.parse_output_file` and consumed by
    :class:`OutputToXarray`.  The values are still in file layout — flat
    columns — together with the axis information needed to reshape them.

    Parameters
    ----------
    output_type : OutputType
        Layout of the table (spectral or integrated, one or several
        altitudes).
    data : dict of str to numpy.ndarray
        Column name → 1-D array with one entry per file row, i.e. of
        length ``n_wl * n_alt``.  Includes the axis columns ``lambda`` /
        ``zout`` when the file has them.
    wavelengths : list of float, optional
        Sorted unique wavelengths (nm) of a spectral output; *None* for
        integrated output.
    altitudes : list of float, optional
        Sorted unique output altitudes (km) of a multi-altitude output;
        *None* when there is a single output altitude.
    source_file : pathlib.Path, optional
        The raw ``uvspec`` output file that was parsed.
    metadata : dict, optional
        Free-form metadata.  The parser stores the output level of a
        single-altitude run under ``zout``; the batch driver adds
        ``point_id``, ``time``, ``latitude`` and ``longitude`` of the
        point.  Defaults to an empty dict.
    is_brightness_temperature : bool, default ``False``
        True when the run used ``output_quantity brightness``, i.e. the
        radiance / irradiance columns hold brightness temperatures (K).

    See Also
    --------
    OutputParser.parse_output_file : Creates instances from files.
    OutputToXarray.convert_batch : Reshapes a list of instances.

    Examples
    --------
    A spectral result at a single altitude, built by hand:

    >>> import numpy as np
    >>> from pyradtran import OutputType, ParsedOutput
    >>> parsed = ParsedOutput(
    ...     output_type=OutputType.SPECTRAL_SINGLE_ALTITUDE,
    ...     data={
    ...         "lambda": np.array([500.0, 501.0]),
    ...         "eglo": np.array([950.0, 960.0]),
    ...     },
    ...     wavelengths=[500.0, 501.0],
    ... )
    >>> parsed.altitudes is None, parsed.metadata
    (True, {})
    """

    output_type: OutputType
    data: Dict[str, Any]
    wavelengths: Optional[List[float]] = None
    altitudes: Optional[List[float]] = None
    source_file: Optional[Path] = None
    metadata: Dict[str, Any] = None
    is_brightness_temperature: bool = False

    def __post_init__(self):
        """Initialize metadata if not provided."""
        if self.metadata is None:
            self.metadata = {}


class InputDataLoader:
    """Load simulation input coordinates from CSV or NetCDF files.

    The file must contain ``time``, ``latitude``, and ``longitude``
    variables (in a CSV, ``datetime`` is accepted as an alias for
    ``time``).  Additional columns are carried through as data variables
    and can be referenced per point with :class:`~pyradtran.params.Var`.

    See Also
    --------
    pyradtran.interface.run_pyradtran_simulation : File-to-file pipeline
        built on this loader.
    """

    @staticmethod
    def load_simulation_input_data(input_file: Union[str, Path]) -> xr.Dataset:
        r"""Read simulation input coordinates and ancillary data.

        Parameters
        ----------
        input_file : str or pathlib.Path
            Path to a ``.csv``, ``.nc`` or ``.netcdf`` file (the suffix
            decides the reader, case-insensitively).

        Returns
        -------
        xarray.Dataset
            Dataset guaranteed to contain ``time``, ``latitude``, and
            ``longitude`` (as dimension, coordinate or data variable).

            * CSV: the ``time`` column is parsed with
              :func:`pandas.to_datetime` and becomes the only dimension;
              every other column is a data variable along ``time``.
            * NetCDF: opened as-is with :func:`xarray.open_dataset`
              (lazily; the file stays open).

        Raises
        ------
        InputGenerationError
            If the file does not exist, has an unsupported suffix, lacks a
            ``time`` / ``datetime`` column (CSV) or one of the required
            variables, or cannot be read (the original exception is
            chained).

        Examples
        --------
        >>> import pathlib, tempfile
        >>> from pyradtran import InputDataLoader
        >>> csv = pathlib.Path(tempfile.mkdtemp()) / "points.csv"
        >>> _ = csv.write_text(
        ...     "time,latitude,longitude,albedo\n"
        ...     "2022-03-28 12:00,78.9,11.9,0.8\n"
        ...     "2022-03-28 13:00,79.0,12.0,0.7\n"
        ... )
        >>> ds = InputDataLoader.load_simulation_input_data(csv)
        >>> dict(ds.sizes), sorted(ds.data_vars)
        ({'time': 2}, ['albedo', 'latitude', 'longitude'])
        >>> ds.albedo.values
        array([0.8, 0.7])

        The extra ``albedo`` column can then drive the simulation:

        >>> from pyradtran import Var
        >>> result = ds.pyradtran.run(params={"albedo": Var("albedo")})  # doctest: +SKIP
        """
        input_file = Path(input_file)

        if not input_file.exists():
            raise InputGenerationError(f"Input file not found: {input_file}")

        try:
            suffix = input_file.suffix.lower()
            if suffix == ".csv":
                df = pd.read_csv(input_file)
                if "time" not in df.columns:
                    if "datetime" not in df.columns:
                        raise InputGenerationError(
                            f"CSV input {input_file} needs a 'time' (or "
                            f"'datetime') column; have: {list(df.columns)}"
                        )
                    df = df.rename(columns={"datetime": "time"})
                df["time"] = pd.to_datetime(df["time"])
                ds = df.set_index("time").to_xarray()
            elif suffix in (".nc", ".netcdf"):
                ds = xr.open_dataset(input_file)
            else:
                raise InputGenerationError(
                    f"Unsupported file format: {input_file.suffix}"
                )

            # required names may be dimensions, coordinates or data variables
            missing_vars = [
                var
                for var in ("time", "latitude", "longitude")
                if var not in ds.dims and var not in ds.variables
            ]
            if missing_vars:
                raise InputGenerationError(
                    f"Required variables missing: {missing_vars}"
                )

            logger.info(
                f"Loaded input data from {input_file} with {len(ds.time)} time points"
            )
            return ds

        except InputGenerationError:
            raise
        except Exception as e:
            raise InputGenerationError(
                f"Failed to load input data from {input_file}: {str(e)}"
            ) from e


class ERA5AtmosphereGenerator:
    """Create libRadtran-compatible atmosphere files from ERA5 data.

    Thin, backward-compatible wrapper around
    :func:`pyradtran.era5.era5_atmosphere_file`, which is the preferred
    entry point in new code.  The output is a radiosonde-style ASCII file
    (pressure, temperature, water vapour and — when the dataset has it —
    ozone) for the ``radiosonde`` keyword of ``uvspec``.

    See Also
    --------
    pyradtran.era5.era5_atmosphere_file : The implementation; documents
        the accepted ERA5 naming flavours and the file format.
    RadiosondeAtmosphereGenerator : Atmosphere from real soundings.
    """

    @staticmethod
    def create_era5_atmosphere_file(
        era5_ds: xr.Dataset,
        latitude: float,
        longitude: float,
        time: Union[str, datetime, np.datetime64],
        output_filepath: Union[str, Path],
    ) -> Path:
        """Create a radiosonde-style atmosphere file from an ERA5 dataset.

        Equivalent to
        ``pyradtran.era5.era5_atmosphere_file(era5_ds, latitude, longitude,
        time, output_filepath)``.

        Parameters
        ----------
        era5_ds : xarray.Dataset
            ERA5 pressure-level dataset with temperature and specific
            humidity (ozone is used when present); see
            :func:`pyradtran.era5.normalize_era5` for the accepted
            variable and coordinate names.
        latitude, longitude : float
            Target location; the nearest grid point is used.
        time : str, datetime.datetime or numpy.datetime64
            Target time; the nearest time step is used.
        output_filepath : str or pathlib.Path
            Destination file.

        Returns
        -------
        pathlib.Path
            Path of the written file.

        Raises
        ------
        InputGenerationError
            If required variables are missing or file writing fails.

        Examples
        --------
        >>> from pyradtran import ERA5AtmosphereGenerator
        >>> path = ERA5AtmosphereGenerator.create_era5_atmosphere_file(  # doctest: +SKIP
        ...     era5, 78.9, 11.9, "2022-03-28T12:00", "work/era5_atm.dat"
        ... )
        """
        from .era5 import era5_atmosphere_file

        return era5_atmosphere_file(era5_ds, latitude, longitude, time, output_filepath)


class RadiosondeAtmosphereGenerator:
    """Fetch and format radiosonde soundings for libRadtran.

    Uses the Integrated Global Radiosonde Archive (IGRA 2, accessed
    through the optional *siphon* package) to find the closest active
    station and the sounding nearest in time, then writes the profile in
    the three-column format expected by the ``radiosonde`` option of
    ``uvspec`` (pressure, temperature, relative humidity).

    All methods are static and need network access.  Import the class
    from :mod:`pyradtran.io`.

    See Also
    --------
    ERA5AtmosphereGenerator : Alternative model-based atmosphere.
    pyradtran.utils.RadiosondeFinder : Look-up of files already on disk.

    Examples
    --------
    Download the sounding closest to a measurement and store it where
    :class:`~pyradtran.utils.RadiosondeFinder` will find it (the file name
    encodes date and seconds of day):

    >>> from datetime import datetime
    >>> from pyradtran.io import RadiosondeAtmosphereGenerator
    >>> path = RadiosondeAtmosphereGenerator.create_radiosonde_atmosphere_file(  # doctest: +SKIP
    ...     datetime(2022, 3, 28, 12, 0), 78.92, 11.93,
    ...     "radiosonde/20220328_43200SOD.dat",
    ... )
    """

    @staticmethod
    def get_station_list(
        url: str = "https://www.ncei.noaa.gov/pub/data/igra/igra2-station-list.txt",
    ) -> Optional[pd.DataFrame]:
        """Download and parse the IGRA 2 station catalogue.

        Parameters
        ----------
        url : str, optional
            Location of the fixed-width station list.  Defaults to the
            NOAA NCEI file ``igra2-station-list.txt``; a local file path
            works as well.

        Returns
        -------
        pandas.DataFrame or None
            One row per station with columns ``id`` (11-character IGRA
            identifier), ``latitude`` and ``longitude`` (degrees),
            ``elevation`` (m), ``state``, ``name``, ``first_year``,
            ``last_year`` and ``num_obs``.  *None* if the list cannot be
            downloaded or parsed (the error is logged, not raised).

        Examples
        --------
        >>> from pyradtran.io import RadiosondeAtmosphereGenerator
        >>> stations = RadiosondeAtmosphereGenerator.get_station_list()  # doctest: +SKIP
        >>> stations.columns.tolist()  # doctest: +SKIP
        ['id', 'latitude', 'longitude', 'elevation', 'state', 'name', 'first_year', 'last_year', 'num_obs']
        """
        try:
            col_specs = [
                (0, 11),
                (12, 20),
                (21, 30),
                (31, 37),
                (38, 40),
                (41, 71),
                (72, 76),
                (77, 81),
                (82, 88),
            ]
            names = (
                "id latitude longitude elevation state name "
                "first_year last_year num_obs"
            ).split()
            return pd.read_fwf(url, colspecs=col_specs, names=names)
        except Exception as e:
            logger.error(f"Error downloading or parsing station list: {e}")
            return None

    @staticmethod
    def find_closest_active_stations(
        stations_df: pd.DataFrame, lat: float, lon: float, n: int = 5
    ) -> pd.DataFrame:
        r"""Return the *n* closest currently-active radiosonde stations.

        A station counts as active when its ``last_year`` is the current
        (UTC) year or the year before.

        Parameters
        ----------
        stations_df : pandas.DataFrame
            Output of :meth:`get_station_list`; needs at least the columns
            ``latitude``, ``longitude`` and ``last_year``.
        lat, lon : float
            Target location in degrees north / east.
        n : int, default ``5``
            Maximum number of stations to return.

        Returns
        -------
        pandas.DataFrame
            The matching rows of *stations_df* (a copy) with an added
            ``distance_km`` column, sorted by increasing distance.  Empty
            when no station is active.

        Notes
        -----
        Distances are great-circle distances from the haversine formula
        on a sphere of radius :math:`R = 6371` km:

        .. math::

            a = \sin^2\frac{\Delta\varphi}{2}
                + \cos\varphi_1 \cos\varphi_2\,
                  \sin^2\frac{\Delta\lambda}{2},
            \qquad
            d = 2R\,\operatorname{atan2}\!\left(\sqrt{a}, \sqrt{1-a}\right)

        with latitudes :math:`\varphi` and longitudes :math:`\lambda` in
        radians.

        Examples
        --------
        >>> from datetime import datetime, timezone
        >>> import pandas as pd
        >>> from pyradtran.io import RadiosondeAtmosphereGenerator
        >>> this_year = datetime.now(timezone.utc).year
        >>> stations = pd.DataFrame(
        ...     {
        ...         "id": ["A", "B", "C"],
        ...         "name": ["NY-ALESUND", "TROMSO", "LINDENBERG"],
        ...         "latitude": [78.92, 69.68, 52.21],
        ...         "longitude": [11.93, 18.94, 14.12],
        ...         "last_year": [this_year, this_year, 1990],
        ...     }
        ... )
        >>> near = RadiosondeAtmosphereGenerator.find_closest_active_stations(
        ...     stations, lat=78.0, lon=15.0, n=2
        ... )
        >>> near[["name", "distance_km"]].round(1)
                 name  distance_km
        0  NY-ALESUND        123.0
        1      TROMSO        932.6
        """
        current_year = datetime.now(timezone.utc).year
        active_stations = stations_df[
            stations_df["last_year"] >= current_year - 1
        ].copy()

        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)
        stations_lat_rad = np.radians(active_stations["latitude"])
        stations_lon_rad = np.radians(active_stations["longitude"])

        dlon = stations_lon_rad - lon_rad
        dlat = stations_lat_rad - lat_rad
        a = (
            np.sin(dlat / 2) ** 2
            + np.cos(lat_rad) * np.cos(stations_lat_rad) * np.sin(dlon / 2) ** 2
        )
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        active_stations["distance_km"] = 6371 * c
        return active_stations.sort_values(by="distance_km").head(n)

    @staticmethod
    def get_closest_sounding(
        target_dt: datetime, lat: float, lon: float
    ) -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[str]]:
        """Retrieve the radiosonde sounding nearest in space and time.

        Requires the **siphon** package for IGRA 2 access and a network
        connection.

        Parameters
        ----------
        target_dt : datetime.datetime
            Target time in UTC, timezone-naive.
        lat, lon : float
            Target location in degrees north / east.

        Returns
        -------
        sounding_df : pandas.DataFrame or None
            Sounding profile as returned by siphon's
            ``IGRAUpperAir.request_data`` (columns include ``pressure`` in
            hPa, ``height`` in m, ``temperature`` and ``dewpoint`` in °C).
        header : pandas.DataFrame or None
            Sounding header as returned by siphon.
        header_text : str or None
            One-line summary of the form ``"<station name> at
            <YYYY-mm-dd HH:MM> UTC with distance <d> km"``.

        All three are *None* if the station list cannot be loaded, no
        active station exists, or none of the candidates has data.

        Raises
        ------
        InputGenerationError
            If *siphon* is not installed.

        Notes
        -----
        The search is station-first: the five closest active stations
        (:meth:`find_closest_active_stations`) are tried in order of
        distance, and for each station the candidate times

        * *target_dt* itself,
        * 00 and 12 UTC of the target day,
        * 12 and 00 UTC of the previous day

        are requested in order of increasing time difference.  The first
        non-empty sounding wins, so a nearer station with a sounding up to
        a day old is preferred over a more distant one with a better time
        match.

        Examples
        --------
        >>> from datetime import datetime
        >>> from pyradtran.io import RadiosondeAtmosphereGenerator
        >>> df, header, text = RadiosondeAtmosphereGenerator.get_closest_sounding(  # doctest: +SKIP
        ...     datetime(2022, 3, 28, 12, 0), 78.92, 11.93
        ... )
        """
        stations = RadiosondeAtmosphereGenerator.get_station_list()
        if stations is None:
            return None, None, None

        closest = RadiosondeAtmosphereGenerator.find_closest_active_stations(
            stations, lat, lon, n=5
        )
        if closest.empty:
            return None, None, None

        base_date = datetime(target_dt.year, target_dt.month, target_dt.day)
        potential_times = [
            target_dt,
            base_date + timedelta(hours=12),
            base_date,
            base_date - timedelta(hours=12),
            base_date - timedelta(hours=24),
        ]
        potential_times = sorted(
            set(potential_times),
            key=lambda x: abs((x - target_dt).total_seconds()),
        )

        try:
            from siphon.simplewebservice.igra2 import IGRAUpperAir
        except Exception:
            raise InputGenerationError(
                "siphon is required for radiosonde retrieval. Install with: pip install siphon"
            ) from None

        for _, station in closest.iterrows():
            station_id = station["id"]
            station_name = station["name"].strip()
            station_distance = station["distance_km"]
            for time_to_check in potential_times:
                try:
                    logger.debug(
                        f"Trying station {station_name} ({station_id}) at {time_to_check}"
                    )
                    df, header = IGRAUpperAir.request_data(time_to_check, station_id)
                    if df is not None and not df.empty:
                        header_text = f"{station_name} at {time_to_check.strftime('%Y-%m-%d %H:%M')} UTC with distance {station_distance:.2f} km"
                        logger.info(
                            f"Found sounding for {station_name} at {time_to_check} UTC with distance {station_distance:.2f} km"
                        )
                        return df, header, header_text
                except Exception as e:
                    logger.debug(f"No data for {station_name} at {time_to_check}: {e}")

        station_names = [name.strip() for name in closest["name"]]
        logger.warning(
            f"No radiosonde data found for any of {len(potential_times)} times "
            f"across {len(station_names)} stations: {station_names}"
        )
        return None, None, None

    @staticmethod
    def create_radiosonde_atmosphere_file(
        time: datetime,
        latitude: float,
        longitude: float,
        output_filepath: Union[str, Path],
    ) -> Path:
        r"""Fetch a sounding and write it as a libRadtran radiosonde file.

        Calls :meth:`get_closest_sounding` and converts the profile to
        the format read by ``radiosonde <file> H2O RH``.

        Parameters
        ----------
        time : datetime.datetime
            Target time in UTC, timezone-naive.
        latitude, longitude : float
            Target location in degrees north / east.
        output_filepath : str or pathlib.Path
            Destination file.  Parent directories are created.

        Returns
        -------
        pathlib.Path
            Path of the written file.

        Raises
        ------
        InputGenerationError
            If *siphon* is missing, no sounding is found, or the sounding
            has no level with valid pressure, temperature, dewpoint and
            height.

        See Also
        --------
        pyradtran.utils.RadiosondeFinder : Finds such files by time when
            ``h2o_source`` is ``"radiosonde"``.

        Notes
        -----
        The file has three comment lines (the second one names station,
        sounding time and distance) followed by one line per level with
        pressure (hPa), temperature (K) and relative humidity (%), ordered
        by increasing pressure, i.e. top of the sounding first.  Levels
        with missing values and duplicate pressure levels are dropped,
        since libRadtran requires a strictly monotonic pressure grid.

        Relative humidity over liquid water is computed from temperature
        :math:`T` and dewpoint :math:`T_d` (both in °C) with the Magnus
        form of Bolton (1980) [1]_ for the saturation vapour pressure:

        .. math::

            e_s(T) = 6.112\ \mathrm{hPa}\;
                     \exp\!\left(\frac{17.67\,T}{T + 243.5}\right),
            \qquad
            \mathrm{RH} = 100\,\% \cdot \frac{e_s(T_d)}{e_s(T)}

        References
        ----------
        .. [1] Bolton, D. (1980): The computation of equivalent potential
           temperature. *Mon. Wea. Rev.*, 108, 1046–1053.

        Examples
        --------
        >>> from datetime import datetime
        >>> from pyradtran.io import RadiosondeAtmosphereGenerator
        >>> path = RadiosondeAtmosphereGenerator.create_radiosonde_atmosphere_file(  # doctest: +SKIP
        ...     datetime(2022, 3, 28, 12, 0), 78.92, 11.93, "radiosonde/20220328_43200SOD.dat"
        ... )

        The humidity conversion for a level at −10 °C with a dewpoint of
        −15 °C:

        >>> import numpy as np
        >>> e_s = lambda t: 6.112 * np.exp(17.67 * t / (t + 243.5))
        >>> round(float(100 * e_s(-15.0) / e_s(-10.0)), 1)
        66.8
        """
        sounding_df, _, header_text = (
            RadiosondeAtmosphereGenerator.get_closest_sounding(
                time, latitude, longitude
            )
        )
        if sounding_df is None or sounding_df.empty:
            raise InputGenerationError(
                "No radiosonde data found for requested parameters"
            )

        df = sounding_df.dropna(
            subset=["pressure", "temperature", "dewpoint", "height"]
        )
        if df.empty:
            raise InputGenerationError("No valid levels in radiosonde data")

        e = 6.112 * np.exp(17.67 * df["dewpoint"] / (df["dewpoint"] + 243.5))
        e_s = 6.112 * np.exp(17.67 * df["temperature"] / (df["temperature"] + 243.5))
        rh = (e / e_s) * 100  # Relative Humidity in %
        temp_k = df["temperature"] + 273.15
        pressure_hpa = df["pressure"]

        # TOA first (ascending pressure); drop duplicate levels —
        # libRadtran requires a strictly monotonic pressure grid
        sorted_idx = np.argsort(pressure_hpa.values)
        p_sorted = pressure_hpa.values[sorted_idx]
        keep = np.concatenate(([True], np.diff(p_sorted) > 0))
        sorted_idx = sorted_idx[keep]

        output_path = Path(output_filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write("# Radiosonde atmosphere profile\n")
            f.write(f"# {header_text}\n")
            f.write("# p(hPa)  T(K)  h2o(RH%)\n")
            for idx in sorted_idx:
                f.write(
                    f"{pressure_hpa.values[idx]:.2f}  "
                    f"{temp_k.values[idx]:.2f}  {rh.values[idx]:.3f}\n"
                )

        logger.info(f"Created radiosonde atmosphere file: {output_path}")
        return output_path


class OutputParser:
    r"""Parse raw ``uvspec`` output files into :class:`ParsedOutput`.

    ``uvspec`` writes a headerless, whitespace-delimited table.  The
    parser therefore needs to know how the run was set up: it derives the
    column names, the output altitudes and the table layout from the same
    configuration (and per-run overrides) that produced the input file.

    Parameters
    ----------
    config : SimulationConfig
        Configuration that was used for the simulation.  Only its
        ``simulation_defaults`` section is read (``output_columns``,
        ``output_altitudes_km``, ``wavelength_nm``,
        ``integrate_wavelength``).
    parameter_overrides : dict, optional
        The ``uvspec`` options that were overridden for this run, as
        ``{keyword: value}``.  Three keys matter here: ``output_user``
        (replaces the column list), ``zout`` (replaces the output
        altitudes) and ``output_quantity`` (``"brightness"`` marks the
        result as brightness temperatures and drops the ``albedo``
        column).
    output_altitudes : list of float, optional
        Explicit output altitudes in km.  Takes precedence over both
        *parameter_overrides* and the configuration.

    Attributes
    ----------
    output_columns : list of str
        Column names in file order, as returned by
        :func:`~pyradtran.input_builder.effective_output_columns`: the
        configured (or overridden) columns, with ``lambda`` prepended for
        spectral runs and ``zout`` prepended for multi-altitude runs.
    output_altitudes : list of float
        Output altitudes (km) of the run.
    is_integrated : bool
        ``simulation_defaults.integrate_wavelength``.
    is_brightness_output : bool
        Whether ``output_quantity`` is ``"brightness"``.
    wavelength_range : list of float
        ``simulation_defaults.wavelength_nm``.

    See Also
    --------
    OutputToXarray : Reshapes the parsed columns into labelled arrays.
    pyradtran.input_builder.effective_output_columns : Single source of
        the column list for both the input file and this parser.

    Notes
    -----
    **How the output type is decided.**  The :class:`OutputType` follows
    from the configuration alone, not from the file contents:

    ====================  ==========================  ==========================
    ..                    one output altitude         several output altitudes
    ====================  ==========================  ==========================
    integrated            INTEGRATED_SINGLE_ALTITUDE  INTEGRATED_MULTI_ALTITUDE
    spectral              SPECTRAL_SINGLE_ALTITUDE    SPECTRAL_MULTI_ALTITUDE
    ====================  ==========================  ==========================

    where "integrated" means ``integrate_wavelength`` is true and the
    number of altitudes is ``len(output_altitudes)``.

    **Array shapes.**  For :math:`N_\lambda` wavelengths and :math:`N_z`
    output altitudes the file has :math:`N_\lambda N_z` rows (with
    :math:`N_\lambda = 1` for integrated output).  Every entry of
    :attr:`ParsedOutput.data` is a 1-D array of that length, in file
    order.  ``uvspec`` writes wavelength as the outer and altitude as the
    inner loop, so the value for wavelength index :math:`i` and altitude
    index :math:`j` sits in row

    .. math::

        r = i\,N_z + j ,

    and ``column.reshape(n_wl, n_alt)`` recovers the 2-D field — this is
    what :class:`OutputToXarray` does.

    ==========================  ====================  ===========  =========
    Output type                 ``data[...]`` length  wavelengths  altitudes
    ==========================  ====================  ===========  =========
    INTEGRATED_SINGLE_ALTITUDE  1                     None         None
    INTEGRATED_MULTI_ALTITUDE   ``n_alt``             None         list
    SPECTRAL_SINGLE_ALTITUDE    ``n_wl``              list         None
    SPECTRAL_MULTI_ALTITUDE     ``n_wl * n_alt``      list         list
    ==========================  ====================  ===========  =========

    Examples
    --------
    The parser only reads ``config.simulation_defaults``, so for a
    self-contained demonstration a stand-in object is enough (normally
    you pass the :class:`~pyradtran.config.SimulationConfig` of the run):

    >>> from types import SimpleNamespace
    >>> from pyradtran import OutputParser, SimulationDefaults
    >>> config = SimpleNamespace(
    ...     simulation_defaults=SimulationDefaults(
    ...         output_columns=["edir", "edn", "eup"],
    ...         output_altitudes_km=[0.0, 5.0],
    ...     )
    ... )
    >>> parser = OutputParser(config)
    >>> parser.output_columns
    ['zout', 'lambda', 'edir', 'edn', 'eup']
    >>> parser.output_altitudes
    [0.0, 5.0]

    Per-run overrides change what the parser expects:

    >>> OutputParser(config, {"output_user": "lambda eglo", "zout": 2.0}).output_columns
    ['lambda', 'eglo']
    """

    def __init__(
        self,
        config: SimulationConfig,
        parameter_overrides: Dict[str, Any] = None,
        output_altitudes: Optional[List[float]] = None,
    ):
        self.config = config
        self.parameter_overrides = parameter_overrides or {}

        self.is_brightness_output = (
            self.parameter_overrides.get("output_quantity") == "brightness"
        )

        # Same column list the input builder wrote into output_user
        # (per-run output_user override, injected lambda/zout, and the
        # brightness albedo filter all included).
        self.output_columns = effective_output_columns(config, self.parameter_overrides)

        # explicit per-run altitudes win over overrides and config
        if output_altitudes is not None:
            self.output_altitudes = list(output_altitudes)
        else:
            self.output_altitudes = effective_output_altitudes(
                config, self.parameter_overrides
            )

        self.wavelength_range = config.simulation_defaults.wavelength_nm
        self.is_integrated = getattr(
            config.simulation_defaults, "integrate_wavelength", False
        )
        logger.debug(f"Initialized parser with columns: {self.output_columns}")

    def parse_output_file(self, output_file: Path) -> ParsedOutput:
        r"""Parse a single ``uvspec`` output file.

        The file is read with :func:`numpy.loadtxt`; its columns are named
        after :attr:`output_columns` by position.

        Parameters
        ----------
        output_file : pathlib.Path
            The ``.out`` file written by ``uvspec``.

        Returns
        -------
        ParsedOutput
            ``data`` maps each column name to a 1-D array with one entry
            per file row.  ``wavelengths`` holds the sorted unique values
            of the ``lambda`` column (spectral output only) and
            ``altitudes`` those of the ``zout`` column (multi-altitude
            output only); both are *None* otherwise.  For a single
            output altitude, ``metadata["zout"]`` holds that level (km as
            float, or a symbolic level such as ``"toa"`` as string).

        Raises
        ------
        OutputParsingError
            If the file is missing, empty, or not a rectangular numeric
            table (the original exception is chained).

        Notes
        -----
        Columns are matched purely by position.  If the file has fewer
        columns than names, the surplus names are dropped; surplus file
        columns are ignored.  No error is raised in either case, so the
        parser must be constructed with the settings of the run that
        produced the file.

        Examples
        --------
        A spectral run with three wavelengths and two output altitudes —
        six rows, wavelength varying slowest:

        >>> import pathlib, tempfile
        >>> from types import SimpleNamespace
        >>> from pyradtran import OutputParser, SimulationDefaults
        >>> out_file = pathlib.Path(tempfile.mkdtemp()) / "point_0.out"
        >>> _ = out_file.write_text(
        ...     "0.0 500.0 800.0 150.0  95.0\n"
        ...     "5.0 500.0 900.0 100.0 110.0\n"
        ...     "0.0 501.0 810.0 152.0  96.0\n"
        ...     "5.0 501.0 910.0 101.0 111.0\n"
        ...     "0.0 502.0 820.0 154.0  97.0\n"
        ...     "5.0 502.0 920.0 102.0 112.0\n"
        ... )
        >>> config = SimpleNamespace(
        ...     simulation_defaults=SimulationDefaults(
        ...         output_columns=["edir", "edn", "eup"],
        ...         output_altitudes_km=[0.0, 5.0],
        ...     )
        ... )
        >>> parsed = OutputParser(config).parse_output_file(out_file)
        >>> parsed.output_type
        <OutputType.SPECTRAL_MULTI_ALTITUDE: 'spectral_multi_altitude'>
        >>> [float(w) for w in parsed.wavelengths]
        [500.0, 501.0, 502.0]
        >>> [float(z) for z in parsed.altitudes]
        [0.0, 5.0]
        >>> parsed.data["edn"]
        array([150., 100., 152., 101., 154., 102.])

        Reshaping to ``(wavelength, altitude)`` separates the two levels:

        >>> parsed.data["edn"].reshape(3, 2)
        array([[150., 100.],
               [152., 101.],
               [154., 102.]])

        A spectrally integrated run at one altitude is a single row:

        >>> out_file = out_file.with_name("point_1.out")
        >>> _ = out_file.write_text("55.3 420.1 357.1 0.85\n")
        >>> config.simulation_defaults = SimulationDefaults(integrate_wavelength=True)
        >>> parsed = OutputParser(config).parse_output_file(out_file)
        >>> parsed.output_type.name, parsed.wavelengths, parsed.altitudes
        ('INTEGRATED_SINGLE_ALTITUDE', None, None)
        >>> {name: float(col[0]) for name, col in parsed.data.items()}
        {'sza': 55.3, 'eglo': 420.1, 'eup': 357.1, 'albedo': 0.85}

        Problems surface as :class:`~pyradtran.exceptions.OutputParsingError`:

        >>> OutputParser(config).parse_output_file(pathlib.Path("missing.out"))
        Traceback (most recent call last):
            ...
        pyradtran.exceptions.OutputParsingError: Output file not found: missing.out
        """
        try:
            if not output_file.exists():
                raise OutputParsingError(f"Output file not found: {output_file}")

            # Read the output file
            data = np.loadtxt(output_file, ndmin=2)

            if data.size == 0:
                raise OutputParsingError(f"Output file is empty: {output_file}")

            # The layout follows from the configuration, not the file
            spectral = not self.is_integrated
            multi_altitude = len(self.output_altitudes) != 1
            if multi_altitude:
                output_type = (
                    OutputType.SPECTRAL_MULTI_ALTITUDE
                    if spectral
                    else OutputType.INTEGRATED_MULTI_ALTITUDE
                )
            else:
                output_type = (
                    OutputType.SPECTRAL_SINGLE_ALTITUDE
                    if spectral
                    else OutputType.INTEGRATED_SINGLE_ALTITUDE
                )

            # Columns are named by position; names without a file column
            # are dropped (zip stops at the shorter of the two)
            parsed_data = dict(zip(self.output_columns, data.T))

            def axis(column, present):
                """Sorted unique values of an axis column, else *None*."""
                if present and column in parsed_data:
                    return sorted(set(parsed_data[column]))
                return None

            return ParsedOutput(
                output_type=output_type,
                data=parsed_data,
                wavelengths=axis("lambda", spectral),
                altitudes=axis("zout", multi_altitude),
                source_file=output_file,
                # a single output level has no zout column; record it so
                # the converters can label the altitude axis
                metadata=(
                    {"zout": self.output_altitudes[0]}
                    if len(self.output_altitudes) == 1
                    else None
                ),
                is_brightness_temperature=self.is_brightness_output,
            )

        except OutputParsingError:
            raise
        except Exception as e:
            raise OutputParsingError(
                f"Failed to parse output file {output_file}: {str(e)}"
            ) from e


class OutputToXarray:
    """Convert :class:`ParsedOutput` objects to :class:`xarray.Dataset`.

    Provides :meth:`convert` for a single output and
    :meth:`convert_batch` for a list (matching the flattened input
    order produced by
    :func:`~pyradtran.interface.execute_simulation_batch`).  Both are
    static methods.

    Both turn every ``output_user`` column into a data variable and the
    axis columns ``lambda`` / ``zout`` into ``wavelength`` (nm) /
    ``altitude`` (km) coordinates.  :meth:`convert_batch` is the one used
    by :meth:`~pyradtran.interface.PyRadtranAccessor.run`.

    See Also
    --------
    OutputParser : Produces the :class:`ParsedOutput` objects.
    """

    @staticmethod
    def convert(
        parsed_output: ParsedOutput,
        input_ds: xr.Dataset,
        time_var: str = "time",
        lat_var: str = "latitude",
        lon_var: str = "longitude",
    ) -> xr.Dataset:
        """Convert a single :class:`ParsedOutput` to an xarray Dataset.

        Parameters
        ----------
        parsed_output : ParsedOutput
            Result of one simulated point.
        input_ds : xarray.Dataset
            Dataset describing that point.  Must contain *time_var* with
            exactly one element (the parsed values are reshaped to
            ``(len(time), n_wl, n_alt)``).
        time_var, lat_var, lon_var : str
            Names of the time / latitude / longitude variables in
            *input_ds*; those present are copied into the result.

        Returns
        -------
        xarray.Dataset
            One data variable per output column (the axis columns
            ``lambda`` and ``zout`` excluded) with dimensions

            * ``(time,)`` — integrated, single altitude,
            * ``(time, altitude)`` — integrated, several altitudes,
            * ``(time, wavelength)`` — spectral, single altitude,
            * ``(time, wavelength, altitude)`` — spectral, several
              altitudes,

            and the coordinates ``wavelength`` (nm) and ``altitude`` (km)
            where the dimension exists.  A single output altitude recorded
            by the parser (``metadata["zout"]``, numeric) is attached as a
            scalar ``altitude`` coordinate.

        See Also
        --------
        convert_batch : Many points at once; always adds an ``altitude``
            dimension and restores multi-dimensional input layouts.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> import xarray as xr
        >>> from pyradtran import OutputToXarray, OutputType, ParsedOutput
        >>> parsed = ParsedOutput(
        ...     output_type=OutputType.SPECTRAL_MULTI_ALTITUDE,
        ...     data={
        ...         "zout": np.array([0.0, 5.0, 0.0, 5.0, 0.0, 5.0]),
        ...         "lambda": np.array([500.0, 500.0, 501.0, 501.0, 502.0, 502.0]),
        ...         "edn": np.array([150.0, 100.0, 152.0, 101.0, 154.0, 102.0]),
        ...     },
        ...     wavelengths=[500.0, 501.0, 502.0],
        ...     altitudes=[0.0, 5.0],
        ... )
        >>> point = xr.Dataset(coords={"time": pd.to_datetime(["2022-03-28 12:00"])})
        >>> ds = OutputToXarray.convert(parsed, point)
        >>> ds.edn.dims, ds.edn.shape
        (('time', 'wavelength', 'altitude'), (1, 3, 2))
        >>> ds.edn.sel(altitude=5.0).values
        array([[100., 101., 102.]])
        """
        # Create base dataset with coordinates from input
        ds = xr.Dataset()

        # Copy coordinates from input dataset
        for coord_name in [time_var, lat_var, lon_var]:
            if coord_name in input_ds:
                ds[coord_name] = input_ds[coord_name]

        # One axis per present dimension; shape stays aligned with dims.
        # uvspec output_user rows are wavelength-outer, zout-inner.
        dims = [time_var]
        shape = [len(input_ds[time_var])]
        for dim, axis in (
            ("wavelength", parsed_output.wavelengths),
            ("altitude", parsed_output.altitudes),
        ):
            if axis:
                ds[dim] = (dim, axis)
                dims.append(dim)
                shape.append(len(axis))

        # Add data variables
        for var_name, values in parsed_output.data.items():
            if var_name in ["zout", "lambda"]:
                continue  # Skip coordinate variables
            values = np.asarray(values)
            if len(dims) > 1:
                values = values.reshape(shape)
            ds[var_name] = (dims, values)

        zout = parsed_output.metadata.get("zout")
        if not parsed_output.altitudes and isinstance(zout, (int, float)):
            ds = ds.assign_coords(altitude=float(zout))

        return ds

    @staticmethod
    def convert_batch(
        parsed_outputs: List[Optional[ParsedOutput]],
        input_ds: xr.Dataset,
        time_var: str = "time",
        lat_var: str = "latitude",
        lon_var: str = "longitude",
    ) -> xr.Dataset:
        r"""Convert a batch of :class:`ParsedOutput` objects to a single Dataset.

        The points of a simulation batch are the input dataset with all
        its dimensions stacked into one.  This method fills one array per
        output column in that stacked order and unstacks it again, so the
        result has the layout of the input plus the spectral and vertical
        axes of the simulation.  Failed simulations (*None* entries)
        produce NaN values.

        Parameters
        ----------
        parsed_outputs : list of ParsedOutput or None
            One entry per input point, in the order of
            ``input_ds.stack(sample=list(input_ds.sizes))`` (the last
            dimension varies fastest).  *None* marks a failed or skipped
            point.
        input_ds : xarray.Dataset or xarray.DataArray
            Original input dataset; its dimensions and their coordinates
            are restored on the result.  A dataset without dimensions
            (scalar coordinates only) is a single point: the result then
            has only the ``wavelength`` / ``altitude`` dimensions and the
            scalar coordinates of the input.
        time_var, lat_var, lon_var : str
            Accepted for symmetry with :meth:`convert`; not used.

        Returns
        -------
        xarray.Dataset
            One float data variable per output column (the axis columns
            ``lambda`` and ``zout`` excluded) with dimensions

            ``(*input_ds.dims, wavelength, altitude)``

            * ``wavelength`` (nm, ``units`` attribute set) is present only
              for spectral output.
            * ``altitude`` (km, float, ``units`` attribute set) is
              **always** present.  For multi-altitude output it holds the
              ``zout`` values.  For a single output altitude it has
              length 1 and holds that altitude when all points share it
              (as recorded by the parser in ``metadata["zout"]``).
            * ``zout`` (km, on the input dimensions) — only when the
              single output altitude differs from point to point
              (``params={"zout": Var(...)}``, e.g. a flight track).  The
              length-1 ``altitude`` coordinate is then the placeholder
              ``0.0`` and ``zout`` holds the altitude of each point (NaN
              for failed points).  Not added if the input already has a
              ``zout`` coordinate.  The placeholder is also used when
              the level is unknown or symbolic (``toa``).
            * An output column named like a coordinate of *input_ds*
              (e.g. ``sza`` when sweeping over an ``sza`` dimension) is
              stored as ``<name>_out``, since the name is taken by the
              coordinate; a warning is logged.
            * The dataset attributes are the ``metadata`` of the first
              successful point, minus per-point entries (``point_id``,
              ``time``, ``latitude``, ``longitude``, …).

        Raises
        ------
        ValueError
            If *parsed_outputs* is empty or contains only *None*.

        See Also
        --------
        convert : Single point, without the forced ``altitude`` dimension.
        pyradtran.interface.PyRadtranAccessor.run : Adds the ``status``
            variable and provenance attributes on top of this result.

        Notes
        -----
        The axes are taken from the first successful point; every other
        point must have the same number of wavelengths and altitudes.  A
        point whose column cannot be reshaped to
        :math:`(N_\lambda, N_z)` is left as NaN and a warning is logged.
        A mismatch between ``len(parsed_outputs)`` and the number of input
        points is only logged as a warning as well.

        Examples
        --------
        Two time steps, three wavelengths, two altitudes; the second point
        failed:

        >>> import numpy as np
        >>> import pandas as pd
        >>> import xarray as xr
        >>> from pyradtran import OutputToXarray, OutputType, ParsedOutput
        >>> parsed = ParsedOutput(
        ...     output_type=OutputType.SPECTRAL_MULTI_ALTITUDE,
        ...     data={
        ...         "zout": np.array([0.0, 5.0, 0.0, 5.0, 0.0, 5.0]),
        ...         "lambda": np.array([500.0, 500.0, 501.0, 501.0, 502.0, 502.0]),
        ...         "edn": np.array([150.0, 100.0, 152.0, 101.0, 154.0, 102.0]),
        ...     },
        ...     wavelengths=[500.0, 501.0, 502.0],
        ...     altitudes=[0.0, 5.0],
        ... )
        >>> input_ds = xr.Dataset(
        ...     coords={"time": pd.to_datetime(["2022-03-28 12:00", "2022-03-28 13:00"])}
        ... )
        >>> ds = OutputToXarray.convert_batch([parsed, None], input_ds)
        >>> dict(ds.sizes)
        {'time': 2, 'wavelength': 3, 'altitude': 2}
        >>> ds.wavelength.attrs["units"], ds.altitude.attrs["units"]
        ('nm', 'km')
        >>> ds.edn.sel(altitude=5.0).values
        array([[100., 101., 102.],
               [ nan,  nan,  nan]])

        Integrated single-altitude output still gets a length-1
        ``altitude`` dimension, labelled with the output level the parser
        recorded:

        >>> broadband = ParsedOutput(
        ...     output_type=OutputType.INTEGRATED_SINGLE_ALTITUDE,
        ...     data={"eglo": np.array([420.1]), "eup": np.array([357.1])},
        ...     metadata={"zout": 2.0},
        ... )
        >>> ds = OutputToXarray.convert_batch([broadband, broadband], input_ds)
        >>> ds.eglo.dims, ds.eglo.values.tolist()
        (('time', 'altitude'), [[420.1], [420.1]])
        >>> ds.altitude.values
        array([2.])

        With a different level per point the coordinate is a placeholder
        and the levels come back as ``zout``:

        >>> higher = ParsedOutput(
        ...     output_type=OutputType.INTEGRATED_SINGLE_ALTITUDE,
        ...     data={"eglo": np.array([433.0]), "eup": np.array([361.2])},
        ...     metadata={"zout": 3.5},
        ... )
        >>> ds = OutputToXarray.convert_batch([broadband, higher], input_ds)
        >>> ds.altitude.values, ds.zout.values
        (array([0.]), array([2. , 3.5]))

        A gridded input keeps its layout — here ``(latitude, longitude)``
        with the points in row-major order:

        >>> grid = xr.Dataset(coords={"latitude": [78.0, 79.0], "longitude": [10.0, 12.0, 14.0]})
        >>> points = [
        ...     ParsedOutput(OutputType.INTEGRATED_SINGLE_ALTITUDE, {"eglo": np.array([v])})
        ...     for v in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
        ... ]
        >>> OutputToXarray.convert_batch(points, grid).eglo.isel(altitude=0).values
        array([[1., 2., 3.],
               [4., 5., 6.]])
        """
        if not parsed_outputs:
            raise ValueError("No parsed outputs provided")

        # Ensure input_ds is a Dataset
        if isinstance(input_ds, xr.DataArray):
            input_ds = input_ds.to_dataset()

        # Stack input dataset exactly as done in execute_simulation_batch
        dims = list(input_ds.sizes.keys())
        sample_dim = "sample_batch_dim"

        if dims:
            stacked_ds = input_ds.stack({sample_dim: dims})
        else:
            # scalar dataset: one point (the explicit index makes the
            # dimension exist even without data variables)
            stacked_ds = input_ds.expand_dims({sample_dim: [0]})

        num_points = stacked_ds.sizes[sample_dim]

        if len(parsed_outputs) != num_points:
            logger.warning(
                f"Number of outputs ({len(parsed_outputs)}) does not match input points ({num_points}). Dimensions might be wrong."
            )

        # Get structure from the first valid output
        template_output = next((x for x in parsed_outputs if x is not None), None)
        if template_output is None:
            raise ValueError(
                "All simulations failed, cannot determine output structure"
            )

        # zout/lambda are axis columns, not data (mirrors convert())
        var_names = [
            v for v in template_output.data.keys() if v not in ("zout", "lambda")
        ]
        altitudes = template_output.altitudes
        point_zout = None
        if not altitudes:
            # Single output level: no zout column; the parser recorded
            # the level of each point in metadata["zout"].
            levels = [o.metadata.get("zout") if o else None for o in parsed_outputs]
            known = [z for z in levels if z is not None]
            numeric = bool(known) and all(isinstance(z, (int, float)) for z in known)
            if numeric and len(set(known)) == 1:
                altitudes = [float(known[0])]
            else:
                altitudes = [0.0]  # placeholder: per-point / unknown / symbolic
                if numeric:
                    point_zout = [np.nan if z is None else float(z) for z in levels]

        # Extra dimensions of the output variables: [wavelength,] altitude
        coords = {}
        if template_output.wavelengths:
            coords["wavelength"] = template_output.wavelengths
        coords["altitude"] = altitudes
        extra_dims = list(coords)
        extra_shape = [len(axis) for axis in coords.values()]

        def unstack(values, value_dims):
            """Flat per-point array -> DataArray on the input dimensions."""
            da = xr.DataArray(
                values,
                coords={sample_dim: stacked_ds[sample_dim]},
                dims=[sample_dim] + value_dims,
            )
            # unstack appends the restored dims at the end; transpose them
            # back to the front so results read (time, ..., wavelength,
            # altitude).  A scalar input just drops the sample dim.
            if dims:
                return da.unstack(sample_dim).transpose(*dims, *value_dims)
            return da.isel({sample_dim: 0}, drop=True)

        # Create data dictionaries
        data_vars = {}
        renamed = []

        for var_name in var_names:
            # Initialize with NaN: [sample_dim, wavelength, altitude]
            data_arr = np.full([num_points] + extra_shape, np.nan)

            # Fill with data
            for i, output in enumerate(parsed_outputs):
                if output and var_name in output.data:
                    val = np.array(output.data[var_name])
                    try:
                        data_arr[i, ...] = val.reshape(extra_shape)
                    except ValueError:
                        logger.warning(
                            f"Shape mismatch for {var_name} at index {i}: got {val.shape}, expected {extra_shape}"
                        )

            # An output column named like an input coordinate (a sweep
            # over "sza" that also requests the sza column) cannot keep
            # its name — the coordinate owns it.
            out_name = var_name
            if var_name in input_ds.coords:
                out_name = f"{var_name}_out"
                renamed.append(var_name)
            data_vars[out_name] = unstack(data_arr, extra_dims)

        if renamed:
            logger.warning(
                f"Output column(s) {renamed} share a name with a coordinate "
                f"of the input dataset; stored as "
                f"{[f'{n}_out' for n in renamed]}"
            )
        if point_zout is not None and "zout" not in input_ds.coords:
            data_vars["zout"] = unstack(np.asarray(point_zout), [])
            data_vars["zout"].attrs.update(
                units="km", long_name="output altitude of each point"
            )

        # unstack already restored the coordinates of the input dimensions
        result_ds = xr.Dataset(data_vars, coords=coords)
        if not dims:
            # scalar input: nothing to unstack, carry its coordinates over
            result_ds = result_ds.assign_coords(input_ds.coords)
        if "wavelength" in result_ds.coords:
            result_ds["wavelength"].attrs["units"] = "nm"
        result_ds["altitude"].attrs["units"] = "km"

        # Add attributes (skip per-point fields — they only describe the
        # template point, not the whole batch)
        _POINT_KEYS = {
            "point_id",
            "time",
            "latitude",
            "longitude",
            "albedo",
            "surface_temperature",
            "surface_type",
            "altitude",
            "zout",
        }
        if template_output.metadata:
            result_ds.attrs.update(
                {
                    k: v
                    for k, v in template_output.metadata.items()
                    if k not in _POINT_KEYS
                }
            )

        return result_ds


class NetCDFSaver:
    """Persist simulation results as NetCDF files with provenance attributes.

    See Also
    --------
    pyradtran.config.OutputConfig : File naming and encoding settings.
    """

    @staticmethod
    def save_results_to_netcdf(
        data: Union[Dict[str, Any], xr.Dataset],
        output_path: Path,
        input_ds: xr.Dataset,
        config: SimulationConfig,
        simulation_params: Dict[str, Any] = None,
    ) -> Path:
        """Write results to a NetCDF file with provenance attributes.

        Parameters
        ----------
        data : xarray.Dataset or dict
            Simulation results.  A dataset is written as it is (plus the
            attributes below).  A dict is converted in a basic way: every
            entry whose key does not start with ``"_"`` becomes a variable
            along a ``time`` dimension.
        output_path : pathlib.Path
            Destination file.  Parent directories are created; an existing
            file is overwritten.
        input_ds : xarray.Dataset
            Input dataset of the run.  Currently not used.
        config : SimulationConfig
            Configuration used for the run; the source of the ``config_*``
            attributes and of the encoding
            (:attr:`~pyradtran.config.OutputConfig.netcdf_encoding`,
            applied to every data variable).
        simulation_params : dict, optional
            Per-run parameters; stored as their ``str()`` representation
            in the ``simulation_parameters`` attribute.

        Returns
        -------
        pathlib.Path
            *output_path*.

        Raises
        ------
        OutputParsingError
            If anything fails while preparing or writing the file (the
            original exception is chained).

        See Also
        --------
        pyradtran.interface.PyRadtranAccessor.run : Calls this when
            ``save_to_file=True``.

        Notes
        -----
        The following global attributes are added to the written file
        (*data* itself is not modified):

        ================================  ==================================
        Attribute                         Content
        ================================  ==================================
        ``created_by``                    ``"pyradtran"``
        ``creation_date``                 local time, ISO 8601
        ``simulation_parameters``         ``str(simulation_params)``, if any
        ``config_source``                 ``solar`` / ``thermal``
        ``config_rte_solver``             solver name
        ``config_mol_abs_param``          absorption parameterisation
        ``config_ozone_du``               ozone column (DU), if set
        ``config_h2o_mm``                 precipitable water (mm), if set
        ``config_surface_temperature_k``  surface temperature (K), if set
        ``config_viewing_geometry``       e.g. ``nadir``
        ``config_wavelength_min_nm``      lower wavelength limit (nm)
        ``config_wavelength_max_nm``      upper wavelength limit (nm)
        ``config_integrate_wavelength``   0 or 1
        ``config_output_altitudes_km``    list as a string, e.g. ``[0.0]``
        ``config_libradtran_bin``         path of ``uvspec``
        ``config_libradtran_data``        path of the data directory
        ``config_max_workers``            process count, if not *None*
        ``config_timeout_seconds``        per-call timeout (s)
        ================================  ==================================

        Attributes whose value is *None* are removed before writing,
        because NetCDF cannot store them.  The albedo is not recorded
        here.  Datasets returned by
        :meth:`~pyradtran.interface.PyRadtranAccessor.run` additionally
        carry their own, more complete provenance (``pyradtran_config``,
        ``pyradtran_params``, ``pyradtran_input_example``, …), which is
        written along unchanged.

        Examples
        --------
        >>> from pathlib import Path
        >>> import xarray as xr
        >>> from pyradtran import NetCDFSaver, load_config
        >>> cfg = load_config()  # doctest: +SKIP
        >>> result = ds.pyradtran.run(config=cfg, save_to_file=False)  # doctest: +SKIP
        >>> path = NetCDFSaver.save_results_to_netcdf(  # doctest: +SKIP
        ...     result, Path("out/res.nc"), ds, cfg, simulation_params={"note": "demo"}
        ... )
        >>> saved = xr.open_dataset(path)  # doctest: +SKIP
        >>> saved.attrs["config_rte_solver"], saved.attrs["simulation_parameters"]  # doctest: +SKIP
        ('disort', "{'note': 'demo'}")
        >>> saved.eglo.encoding["zlib"], saved.eglo.encoding["complevel"]  # doctest: +SKIP
        (True, 5)
        """
        try:
            if isinstance(data, xr.Dataset):
                ds = data.copy()  # shallow: the caller's attrs stay untouched
            else:
                # Convert dictionary data to xarray Dataset
                ds = xr.Dataset()
                # Basic conversion - would need more sophisticated logic
                for key, values in data.items():
                    if not key.startswith("_"):
                        ds[key] = ("time", values)

            # Add metadata
            ds.attrs["created_by"] = "pyradtran"
            ds.attrs["creation_date"] = datetime.now().isoformat()
            if simulation_params:
                ds.attrs["simulation_parameters"] = str(simulation_params)

            # Add configuration parameters as attributes
            NetCDFSaver._add_config_to_attrs(ds, config)

            # None values cannot be serialized to NetCDF attributes
            ds.attrs = {k: v for k, v in ds.attrs.items() if v is not None}

            # Save to file, with the configured encoding on every variable
            output_path.parent.mkdir(parents=True, exist_ok=True)
            netcdf_settings = config.output.netcdf_encoding
            encoding = (
                {name: netcdf_settings for name in ds.data_vars}
                if netcdf_settings
                else {}
            )
            ds.to_netcdf(output_path, encoding=encoding)

            logger.info(f"Results saved to {output_path}")
            return output_path

        except Exception as e:
            raise OutputParsingError(
                f"Failed to save results to {output_path}: {str(e)}"
            ) from e

    @staticmethod
    def _serialize_for_netcdf(value):
        """Convert a value to a NetCDF-compatible attribute type.

        ``bool`` → ``int``; ``int`` / ``float`` / ``str`` unchanged;
        everything else (lists, tuples, *None*, …) → ``str(value)``.

        Examples
        --------
        >>> NetCDFSaver._serialize_for_netcdf(True)
        1
        >>> NetCDFSaver._serialize_for_netcdf([0.0, 5.0])
        '[0.0, 5.0]'
        """
        if isinstance(value, bool):
            return int(value)  # 0 or 1
        if isinstance(value, (int, float, str)):
            return value  # already NetCDF compatible
        return str(value)  # lists, tuples, None, ...

    @staticmethod
    def _add_config_to_attrs(ds: xr.Dataset, config: SimulationConfig) -> None:
        """Add the ``config_*`` provenance attributes to *ds* in place.

        See :meth:`save_results_to_netcdf` for the list of attributes.
        """
        sd = config.simulation_defaults
        attrs = ds.attrs
        serialize = NetCDFSaver._serialize_for_netcdf

        # names (a missing one is recorded as the string "None")
        for name in ("source", "rte_solver", "mol_abs_param"):
            attrs[f"config_{name}"] = str(getattr(sd, name))

        # numbers, only when set
        for name in ("ozone_du", "h2o_mm", "surface_temperature_k"):
            if getattr(sd, name) is not None:
                attrs[f"config_{name}"] = float(getattr(sd, name))

        attrs["config_viewing_geometry"] = str(sd.viewing_geometry)

        if sd.wavelength_nm:
            wl_min, wl_max = sd.wavelength_nm
            attrs["config_wavelength_min_nm"] = float(wl_min)
            attrs["config_wavelength_max_nm"] = float(wl_max)
        attrs["config_integrate_wavelength"] = serialize(sd.integrate_wavelength)
        if sd.output_altitudes_km:
            attrs["config_output_altitudes_km"] = serialize(sd.output_altitudes_km)

        attrs["config_libradtran_bin"] = str(config.paths.libradtran_bin)
        attrs["config_libradtran_data"] = str(config.paths.libradtran_data)

        # max_workers None means auto pool size
        if config.execution.max_workers is not None:
            attrs["config_max_workers"] = int(config.execution.max_workers)
        attrs["config_timeout_seconds"] = int(config.execution.timeout_seconds)


# Expose main classes and functions
__all__ = [
    "OutputType",
    "ParsedOutput",
    "InputDataLoader",
    "ERA5AtmosphereGenerator",
    "RadiosondeAtmosphereGenerator",
    "OutputParser",
    "OutputToXarray",
    "NetCDFSaver",
]
