# pyradtran/io.py
"""
Input / output layer for pyRadtran.

Key components:

* :class:`OutputParser` — parse raw ``uvspec`` output files into
  :class:`ParsedOutput` containers.
* :class:`OutputToXarray` — convert parsed results to
  :class:`xarray.Dataset`.
* :class:`InputDataLoader` — load simulation coordinates from CSV or
  NetCDF.
* :class:`ERA5AtmosphereGenerator` — create libRadtran-compatible
  atmosphere files from ERA5 xarray datasets.
* :class:`RadiosondeAtmosphereGenerator` — fetch and format radiosonde
  soundings from IGRA.
* :class:`NetCDFSaver` — persist results to NetCDF with provenance
  metadata.

See Also
--------
pyradtran.core.Simulation : Generates input files and calls ``uvspec``.
pyradtran.interface : High-level batch driver.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
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
    """Enumeration of libRadtran output geometries.

    The four members encode the two binary choices
    (integrated vs. spectral) × (single vs. multi altitude).
    """

    INTEGRATED_SINGLE_ALTITUDE = "integrated_single_altitude"
    INTEGRATED_MULTI_ALTITUDE = "integrated_multi_altitude"
    SPECTRAL_SINGLE_ALTITUDE = "spectral_single_altitude"
    SPECTRAL_MULTI_ALTITUDE = "spectral_multi_altitude"


@dataclass
class ParsedOutput:
    """Container returned by :meth:`OutputParser.parse_output_file`.

    Parameters
    ----------
    output_type : OutputType
        Geometry / dimensionality of the output.
    data : dict of str to numpy.ndarray
        Column name → 1-D value array.
    wavelengths : list of float, optional
        Unique wavelengths present in spectral output.
    altitudes : list of float, optional
        Unique altitude levels present in multi-altitude output.
    source_file : pathlib.Path, optional
        The raw ``uvspec`` output file that was parsed.
    metadata : dict, optional
        Arbitrary key–value metadata (point ID, coordinates, …).
    is_brightness_temperature : bool, default ``False``
        Whether the output represents brightness temperatures.
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
    variables (or ``datetime`` as an alias for ``time``).  Additional
    columns are carried through as data variables.
    """

    @staticmethod
    def load_simulation_input_data(input_file: Union[str, Path]) -> xr.Dataset:
        """Read simulation input coordinates and ancillary data.

        Parameters
        ----------
        input_file : str or pathlib.Path
            Path to a ``.csv`` or ``.nc`` file.

        Returns
        -------
        xarray.Dataset
            Dataset guaranteed to contain ``time``, ``latitude``, and
            ``longitude``.

        Raises
        ------
        InputGenerationError
            If the file is missing, unsupported, or lacks required
            variables.
        """
        input_file = Path(input_file)

        if not input_file.exists():
            raise InputGenerationError(f"Input file not found: {input_file}")

        try:
            if input_file.suffix.lower() == ".csv":
                # Load CSV and convert to xarray
                df = pd.read_csv(input_file)

                # Convert datetime column if it exists
                if "time" in df.columns:
                    df["time"] = pd.to_datetime(df["time"])
                elif "datetime" in df.columns:
                    df["time"] = pd.to_datetime(df["datetime"])
                    df = df.drop("datetime", axis=1)

                # Create xarray Dataset
                ds = df.set_index("time").to_xarray()

            elif input_file.suffix.lower() in [".nc", ".netcdf"]:
                # Load NetCDF directly
                ds = xr.open_dataset(input_file)

            else:
                raise InputGenerationError(
                    f"Unsupported file format: {input_file.suffix}"
                )

            # Validate required coordinates
            required_vars = ["time", "latitude", "longitude"]
            missing_vars = [
                var
                for var in required_vars
                if var not in ds.dims
                and var not in ds.coords
                and var not in ds.data_vars
            ]

            if missing_vars:
                raise InputGenerationError(
                    f"Required variables missing: {missing_vars}"
                )

            logger.info(
                f"Loaded input data from {input_file} with {len(ds.time)} time points"
            )
            return ds

        except Exception as e:
            raise InputGenerationError(
                f"Failed to load input data from {input_file}: {str(e)}"
            )


class ERA5AtmosphereGenerator:
    """Create libRadtran-compatible atmosphere files from ERA5 data.

    The output is a three-column radiosonde-style ASCII file
    (pressure, temperature, water vapour) sorted from TOA to surface,
    directly consumable by the ``radiosonde`` keyword in ``uvspec``.

    See Also
    --------
    RadiosondeAtmosphereGenerator : Atmosphere from real soundings.
    """

    @staticmethod
    def create_era5_atmosphere_file(
        era5_ds: xr.Dataset,
        # h2o_unit: str,
        latitude: float,
        longitude: float,
        time: Union[str, datetime, np.datetime64],
        output_filepath: Union[str, Path],
    ) -> Path:
        """
        Create a radiosonde-style atmosphere file from an ERA5 dataset.

        Parameters
        ----------
        era5_ds : xarray.Dataset
            Must contain variables ``'z'`` (geopotential), ``'t'``
            (temperature), ``'q'`` (specific humidity) on
            ``pressure_level`` coordinates.
        latitude, longitude : float
            Target location for nearest-neighbour selection.
        time : str, datetime, or numpy.datetime64
            Target time step.
        output_filepath : str or pathlib.Path
            Destination file.

        Returns
        -------
        pathlib.Path
            *output_filepath* (echoed back for convenience).

        Raises
        ------
        ValueError
            If required variables or coordinates are missing.
        InputGenerationError
            If file writing fails.
        """
        try:
            # Physical and chemical constants for conversions
            K_B = 1.380649e-23  # Boltzmann constant (J/K)
            M_AIR = 0.0289647  # Molar mass of dry air (kg/mol)
            # M_CO2 = 0.04401      # Molar mass of Carbon Dioxide (CO2) (kg/mol)
            # M_NO2 = 0.0460055    # Molar mass of Nitrogen Dioxide (NO2) (kg/mol)
            # O2_MIXING_RATIO = 0.2095 # Volumetric mixing ratio of O2 in dry air

            # Validate required variables
            required_vars = ["z", "t", "q"]
            required_coords = ["pressure_level", "valid_time"]

            for var in required_vars:
                if var not in era5_ds.variables:
                    raise ValueError(
                        f"Required variable '{var}' not found in ERA5 dataset"
                    )
            for coord in required_coords:
                if coord not in era5_ds.coords:
                    raise ValueError(
                        f"Required coordinate '{coord}' not found in ERA5 dataset"
                    )

            # Select the data for the nearest point and specified time
            # Check if data is already spatially selected (latitude/longitude are scalars)
            if "latitude" in era5_ds.dims or "longitude" in era5_ds.dims:
                # Data still has spatial dimensions, do normal selection
                profile_data = era5_ds.sel(
                    latitude=latitude,
                    longitude=longitude,
                    valid_time=time,
                    method="nearest",
                )
            else:
                # Data is already spatially selected, just select time
                if "valid_time" in era5_ds.dims:
                    # valid_time is still a dimension
                    profile_data = era5_ds.sel(valid_time=time, method="nearest")
                elif "time" in era5_ds.dims:
                    # time is a dimension, try to select by matching time values
                    # Convert the target time to the same format as the dataset
                    target_time = pd.to_datetime(time)
                    # Find the closest time in the dataset
                    time_diffs = abs(pd.to_datetime(era5_ds.time.values) - target_time)
                    closest_idx = time_diffs.argmin()
                    profile_data = era5_ds.isel(time=closest_idx)
                else:
                    # Data is already fully selected
                    profile_data = era5_ds

            # Units: attrs may be stripped by xarray operations; default sanely
            h2o_unit = profile_data["q"].attrs.get("units", "kg kg-1")
            p_unit = profile_data["pressure_level"].attrs.get("units", "hPa")

            if p_unit == "Pa":
                pressure_pa = profile_data["pressure_level"]
            elif p_unit == "hPa":
                pressure_pa = profile_data["pressure_level"] * 100
            else:
                raise InputGenerationError(
                    f"Unsupported pressure unit '{p_unit}' in ERA5 dataset; "
                    f"expected 'Pa' or 'hPa'"
                )

            pressure_hpa = pressure_pa / 100
            temperature_k = profile_data["t"]
            h2o = profile_data["q"]

            # Air number density: calculated from ideal gas law p = NkT
            air_number_density_m3 = pressure_pa / (K_B * temperature_k)
            air_number_density_cm3 = air_number_density_m3 / 1e6

            # Function to convert mass mixing ratio (kg/kg) to number density (molecules/cm^3)
            def mmr_to_nd(mmr, m_gas):
                return mmr * (M_AIR / m_gas) * air_number_density_cm3

            # Create the atmosphere file content
            output_path = Path(output_filepath)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Sort by pressure ascending (TOA to surface: low pressure first)
            sorted_indices = np.argsort(pressure_hpa.values.ravel())
            p_sorted = pressure_hpa.values.ravel()[sorted_indices]
            t_sorted = temperature_k.values.ravel()[sorted_indices]
            h2o_sorted = h2o.values.ravel()[sorted_indices]

            # Drop consecutive duplicate pressure values (libradtran requires strictly monotonic)
            keep = np.concatenate(([True], np.diff(p_sorted) != 0))
            p_sorted = p_sorted[keep]
            t_sorted = t_sorted[keep]
            h2o_sorted = h2o_sorted[keep]

            with open(output_path, "w") as f:
                f.write("# ERA5 atmosphere profile in libradtran radiosonde style\n")
                f.write(f"# p(hPa)  T(K)  h2o({h2o_unit}) \n")

                for p_val, t_val, h2o_val in zip(p_sorted, t_sorted, h2o_sorted):
                    f.write(f"{p_val:.2f}  {t_val:.2f}  {h2o_val:.3e}\n")

            logger.info(
                f"Created ERA5 atmosphere file: {output_path} with H2O unit {h2o_unit}"
            )
            return output_path

        except InputGenerationError:
            raise
        except Exception as e:
            raise InputGenerationError(
                f"Failed to create ERA5 atmosphere file: {str(e)}"
            )


class RadiosondeAtmosphereGenerator:
    """Fetch and format radiosonde soundings for libRadtran.

    Uses the IGRA2 network (via *siphon*) to find the closest
    active station and sounding in time, then writes the profile
    in radiosonde format (pressure, temperature, relative humidity).

    See Also
    --------
    ERA5AtmosphereGenerator : Alternative model-based atmosphere.
    pyradtran.utils.RadiosondeFinder : Local radiosonde file look-up.
    """

    @staticmethod
    def get_station_list(
        url: str = "https://www.ncei.noaa.gov/pub/data/igra/igra2-station-list.txt",
    ) -> Optional[pd.DataFrame]:
        """Download and parse the IGRA station catalogue.

        Returns
        -------
        pandas.DataFrame or None
            One row per station with columns *id*, *latitude*,
            *longitude*, *elevation*, *state*, *name*, *first_year*,
            *last_year*, *num_obs*.  *None* on network error.
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
            names = [
                "id",
                "latitude",
                "longitude",
                "elevation",
                "state",
                "name",
                "first_year",
                "last_year",
                "num_obs",
            ]
            return pd.read_fwf(url, colspecs=col_specs, names=names)
        except Exception as e:
            logger.error(f"Error downloading or parsing station list: {e}")
            return None

    @staticmethod
    def find_closest_active_stations(
        stations_df: pd.DataFrame, lat: float, lon: float, n: int = 5
    ) -> pd.DataFrame:
        """Return the *n* closest currently-active radiosonde stations.

        Parameters
        ----------
        stations_df : pandas.DataFrame
            Output of :meth:`get_station_list`.
        lat, lon : float
            Target location (degrees).
        n : int, default ``5``
            Maximum number of stations to return.

        Returns
        -------
        pandas.DataFrame
            Sorted by ``distance_km`` (ascending).
        """
        current_year = datetime.utcnow().year
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
        distance = 6371 * c

        active_stations["distance_km"] = distance
        return active_stations.sort_values(by="distance_km").head(n)

    @staticmethod
    def get_closest_sounding(
        target_dt: datetime, lat: float, lon: float
    ) -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[str]]:
        """Retrieve the radiosonde sounding nearest in space and time.

        Requires the **siphon** package for IGRA2 access.

        Parameters
        ----------
        target_dt : datetime
            Target time.
        lat, lon : float
            Target location (degrees).

        Returns
        -------
        sounding_df : pandas.DataFrame or None
            Sounding profile.
        header : pandas.DataFrame or None
            Sounding header.
        header_text : str or None
            Human-readable summary.
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
            list(set(potential_times)),
            key=lambda x: abs((x - target_dt).total_seconds()),
        )

        try:
            from siphon.simplewebservice.igra2 import IGRAUpperAir
        except Exception:
            IGRAUpperAir = None

        if IGRAUpperAir is None:
            raise InputGenerationError(
                "siphon is required for radiosonde retrieval. Install with: pip install siphon"
            )

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
                    header_text = f"{station_name} at {time_to_check.strftime('%Y-%m-%d %H:%M')} UTC with distance {station_distance:.2f} km"
                    logger.info(
                        f"Found sounding for {station_name} at {time_to_check} UTC with distance {station_distance:.2f} km"
                    )
                    if df is not None and not df.empty:
                        return df, header, header_text
                except Exception as e:
                    logger.debug(f"No data for {station_name} at {time_to_check}: {e}")
                    continue

        station_names = [row["name"].strip() for _, row in closest.iterrows()]
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
        """Fetch a sounding and write it as a libRadtran atmosphere file.

        Parameters
        ----------
        time : datetime
            Target time.
        latitude, longitude : float
            Target location.
        output_filepath : str or pathlib.Path
            Destination ``.dat`` file.

        Returns
        -------
        pathlib.Path

        Raises
        ------
        InputGenerationError
            If no data can be found or the file cannot be written.
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

        sorted_idx = np.argsort(pressure_hpa.values)

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
    """Parse raw ``uvspec`` output files into :class:`ParsedOutput`.

    The parser uses the column list from
    :attr:`~pyradtran.config.SimulationDefaults.output_columns` to
    assign names to the whitespace-delimited columns produced by
    ``uvspec``.

    Parameters
    ----------
    config : SimulationConfig
        Configuration that was used for the simulation.
    parameter_overrides : dict, optional
        Per-point overrides (used to detect brightness-temperature
        mode).
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
        # (per-run output_user override, injected lambda/zout included).
        self.output_columns = []
        for col in effective_output_columns(config, self.parameter_overrides):
            if self.is_brightness_output and col == "albedo":
                logger.debug("Skipping albedo column for brightness temperature output")
                continue
            self.output_columns.append(col)

        # B7 fix: per-run altitudes win over config
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
        """Parse a single ``uvspec`` output file.

        Parameters
        ----------
        output_file : pathlib.Path

        Returns
        -------
        ParsedOutput

        Raises
        ------
        OutputParsingError
            If the file is missing, empty, or cannot be parsed.
        """
        try:
            if not output_file.exists():
                raise OutputParsingError(f"Output file not found: {output_file}")

            # Read the output file
            data = np.loadtxt(output_file, ndmin=2)

            if data.size == 0:
                raise OutputParsingError(f"Output file is empty: {output_file}")

            # Determine output type
            output_type = self._determine_output_type(data)

            # Parse the data based on type
            parsed_data = self._parse_data_by_type(data, output_type)

            return ParsedOutput(
                output_type=output_type,
                data=parsed_data,
                wavelengths=self._extract_wavelengths(data, output_type),
                altitudes=self._extract_altitudes(data, output_type),
                source_file=output_file,
                is_brightness_temperature=self.is_brightness_output,
            )

        except Exception as e:
            raise OutputParsingError(
                f"Failed to parse output file {output_file}: {str(e)}"
            )

    def _determine_output_type(self, data: np.ndarray) -> OutputType:
        """Infer the :class:`OutputType` from array shape and config."""
        n_rows, n_cols = data.shape
        n_altitudes = len(self.output_altitudes)

        if n_altitudes == 1:
            if self.is_integrated:
                return OutputType.INTEGRATED_SINGLE_ALTITUDE
            else:
                return OutputType.SPECTRAL_SINGLE_ALTITUDE
        else:
            if self.is_integrated:
                return OutputType.INTEGRATED_MULTI_ALTITUDE
            else:
                return OutputType.SPECTRAL_MULTI_ALTITUDE

    def _parse_data_by_type(
        self, data: np.ndarray, output_type: OutputType
    ) -> Dict[str, Any]:
        """Map array columns to named variables."""
        parsed = {}

        for i, col_name in enumerate(self.output_columns):
            if i < data.shape[1]:
                parsed[col_name] = data[:, i]

        return parsed

    def _extract_wavelengths(
        self, data: np.ndarray, output_type: OutputType
    ) -> Optional[List[float]]:
        """Return unique wavelength values when present, else *None*."""
        if output_type in [
            OutputType.SPECTRAL_SINGLE_ALTITUDE,
            OutputType.SPECTRAL_MULTI_ALTITUDE,
        ]:
            if "lambda" in self.output_columns:
                lambda_idx = self.output_columns.index("lambda")
                if lambda_idx < data.shape[1]:
                    return sorted(set(data[:, lambda_idx]))
        return None

    def _extract_altitudes(
        self, data: np.ndarray, output_type: OutputType
    ) -> Optional[List[float]]:
        """Return unique altitude values when present, else *None*."""
        if output_type in [
            OutputType.INTEGRATED_MULTI_ALTITUDE,
            OutputType.SPECTRAL_MULTI_ALTITUDE,
        ]:
            if "zout" in self.output_columns:
                zout_idx = self.output_columns.index("zout")
                if zout_idx < data.shape[1]:
                    return sorted(set(data[:, zout_idx]))
        return None


class OutputToXarray:
    """Convert :class:`ParsedOutput` objects to :class:`xarray.Dataset`.

    Provides :meth:`convert` for a single output and
    :meth:`convert_batch` for a list (matching the flattened input
    order produced by
    :func:`~pyradtran.interface.execute_simulation_batch`).
    """

    @staticmethod
    def convert(
        parsed_output: ParsedOutput,
        input_ds: xr.Dataset,
        time_var: str = "time",
        lat_var: str = "latitude",
        lon_var: str = "longitude",
    ) -> xr.Dataset:
        """Convert a single :class:`ParsedOutput` to an xarray Dataset."""
        # Create base dataset with coordinates from input
        ds = xr.Dataset()

        # Copy coordinates from input dataset
        for coord_name in [time_var, lat_var, lon_var]:
            if coord_name in input_ds:
                ds[coord_name] = input_ds[coord_name]

        # Add dimensions based on output type
        if parsed_output.wavelengths:
            ds["wavelength"] = ("wavelength", parsed_output.wavelengths)

        if parsed_output.altitudes:
            ds["altitude"] = ("altitude", parsed_output.altitudes)

        # Add data variables
        for var_name, values in parsed_output.data.items():
            if var_name in ["zout", "lambda"]:
                continue  # Skip coordinate variables

            # Determine dimensions based on output type
            dims = [time_var]
            if parsed_output.wavelengths:
                dims.append("wavelength")
            if parsed_output.altitudes:
                dims.append("altitude")

            # Reshape values to match dimensions
            if len(dims) == 1:
                ds[var_name] = (dims, values)
            else:
                # Need to reshape multi-dimensional data
                reshaped_values = values.reshape(
                    [len(input_ds[time_var])]
                    + [
                        (
                            len(parsed_output.wavelengths)
                            if parsed_output.wavelengths
                            else 1
                        )
                    ]
                    + [len(parsed_output.altitudes) if parsed_output.altitudes else 1]
                )
                ds[var_name] = (dims, reshaped_values)

        return ds

    @staticmethod
    def convert_batch(
        parsed_outputs: List[Optional[ParsedOutput]],
        input_ds: xr.Dataset,
        time_var: str = "time",
        lat_var: str = "latitude",
        lon_var: str = "longitude",
    ) -> xr.Dataset:
        """Convert a batch of :class:`ParsedOutput` objects to a single Dataset.

        Failed simulations (*None* entries) produce NaN values.

        Parameters
        ----------
        parsed_outputs : list of ParsedOutput or None
            One entry per flattened input point.
        input_ds : xarray.Dataset
            Original input dataset (used to reconstruct dimensions).
        time_var, lat_var, lon_var : str
            Coordinate names.

        Returns
        -------
        xarray.Dataset
            Result with original input dimensions restored.
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
            stacked_ds = input_ds.expand_dims(sample_dim)

        num_points = stacked_ds.sizes[sample_dim]

        if len(parsed_outputs) != num_points:
            # If mismatch, log warning but try to proceed if safe?
            # Actually, this means data corruption or misalignment.
            logger.warning(
                f"Number of outputs ({len(parsed_outputs)}) does not match input points ({num_points}). Dimensions might be wrong."
            )

        # Get structure from the first valid output
        first_valid_idx = next(
            (i for i, x in enumerate(parsed_outputs) if x is not None), -1
        )
        if first_valid_idx == -1:
            raise ValueError(
                "All simulations failed, cannot determine output structure"
            )

        template_output = parsed_outputs[first_valid_idx]
        var_names = list(template_output.data.keys())
        altitudes = template_output.altitudes if template_output.altitudes else [0]
        wavelengths = (
            template_output.wavelengths if template_output.wavelengths else [None]
        )

        # Define extra dimensions for the output variables
        extra_dims = []
        extra_shape = []

        if wavelengths[0] is not None:
            extra_dims.append("wavelength")
            extra_shape.append(len(wavelengths))

        extra_dims.append("altitude")
        extra_shape.append(len(altitudes))

        # Dimensions for the stacked array: [sample_dim, wavelength, altitude]
        stacked_shape = [num_points] + extra_shape

        # Create data dictionaries
        data_vars = {}

        for var_name in var_names:
            # Initialize with NaN
            data_arr = np.full(stacked_shape, np.nan)

            # Fill with data
            for i, output in enumerate(parsed_outputs):
                if output and var_name in output.data:
                    val = np.array(output.data[var_name])
                    try:
                        # Reshape val to match extra_shape (e.g. [wl, alt])
                        val_reshaped = val.reshape(extra_shape)
                        data_arr[i, ...] = val_reshaped
                    except ValueError:
                        logger.warning(
                            f"Shape mismatch for {var_name} at index {i}: got {val.shape}, expected {extra_shape}"
                        )

            # Create temporary DataArray
            da = xr.DataArray(
                data_arr,
                coords={sample_dim: stacked_ds[sample_dim]},
                dims=[sample_dim] + extra_dims,
            )

            # Unstack to restore original dimensions
            if dims:
                unstacked_da = da.unstack(sample_dim)
            else:
                # If original was scalar, just drop the sample dim
                unstacked_da = da.isel({sample_dim: 0}, drop=True)

            data_vars[var_name] = unstacked_da

        # Add coordinates for extra dimensions
        coords = {}
        # Copy original coords (relying on unstack restoration, but good to be explicit for extras)
        # Note: unstack already restores coordinates associated with stacked dims

        # Add extra new coords
        if wavelengths[0] is not None:
            coords["wavelength"] = wavelengths
        coords["altitude"] = altitudes

        # Create final dataset
        # We start with a dataset containing our variables.
        # The coordinates from input_ds should be preserved by unstacking.
        result_ds = xr.Dataset(data_vars, coords=coords)

        # Add attributes (skip per-point fields — they only describe the
        # template point, not the whole batch)
        _POINT_KEYS = {"point_id", "time", "latitude", "longitude",
                       "albedo", "surface_temperature", "surface_type",
                       "altitude"}
        if template_output.metadata:
            result_ds.attrs.update(
                {k: v for k, v in template_output.metadata.items()
                 if k not in _POINT_KEYS}
            )

        return result_ds


class NetCDFSaver:
    """Persist simulation results as CF-compliant NetCDF files."""

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
        data : dict or xarray.Dataset
            Simulation results.
        output_path : pathlib.Path
            Destination file.
        input_ds : xarray.Dataset
            Input dataset (for coordinate reference).
        config : SimulationConfig
            Configuration used for the run.
        simulation_params : dict, optional
            Extra metadata to embed.

        Returns
        -------
        pathlib.Path
            *output_path*.
        """
        try:
            if isinstance(data, xr.Dataset):
                ds = data
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

            # Filter out None values from attributes as they can't be serialized to NetCDF
            filtered_attrs = {k: v for k, v in ds.attrs.items() if v is not None}
            ds.attrs.clear()
            ds.attrs.update(filtered_attrs)

            # Save to file
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Prepare encoding for variables
            netcdf_settings = (
                config.output.netcdf_encoding if config.output.netcdf_encoding else {}
            )
            encoding = {}
            if netcdf_settings:
                for var_name in ds.data_vars:
                    encoding[var_name] = netcdf_settings

            ds.to_netcdf(output_path, encoding=encoding)

            logger.info(f"Results saved to {output_path}")
            return output_path

        except Exception as e:
            raise OutputParsingError(
                f"Failed to save results to {output_path}: {str(e)}"
            )

    @staticmethod
    def _serialize_for_netcdf(value):
        """Convert values to NetCDF-compatible types."""
        if isinstance(value, bool):
            return int(value)  # Convert boolean to integer (0 or 1)
        elif isinstance(value, (list, tuple)):
            return str(value)  # Convert lists/tuples to string representation
        elif isinstance(value, (int, float, str)):
            return value  # These are already NetCDF compatible
        elif value is None:
            return "None"  # Convert None to string
        else:
            return str(value)  # Convert everything else to string

    @staticmethod
    def _add_config_to_attrs(ds: xr.Dataset, config: SimulationConfig) -> None:
        """Add configuration parameters as NetCDF attributes."""
        # Add simulation defaults with None checks
        ds.attrs["config_source"] = (
            str(config.simulation_defaults.source)
            if config.simulation_defaults.source is not None
            else "None"
        )
        ds.attrs["config_rte_solver"] = (
            str(config.simulation_defaults.rte_solver)
            if config.simulation_defaults.rte_solver is not None
            else "None"
        )
        ds.attrs["config_mol_abs_param"] = (
            str(config.simulation_defaults.mol_abs_param)
            if config.simulation_defaults.mol_abs_param is not None
            else "None"
        )

        # Handle numeric parameters that might be None
        # if config.simulation_defaults.albedo_value is not None:
        #     ds.attrs['config_albedo_value'] = float(config.simulation_defaults.albedo_value)
        if config.simulation_defaults.ozone_du is not None:
            ds.attrs["config_ozone_du"] = float(config.simulation_defaults.ozone_du)
        if config.simulation_defaults.h2o_mm is not None:
            ds.attrs["config_h2o_mm"] = float(config.simulation_defaults.h2o_mm)
        if config.simulation_defaults.surface_temperature_k is not None:
            ds.attrs["config_surface_temperature_k"] = float(
                config.simulation_defaults.surface_temperature_k
            )

        ds.attrs["config_viewing_geometry"] = (
            str(config.simulation_defaults.viewing_geometry)
            if config.simulation_defaults.viewing_geometry is not None
            else "None"
        )

        # Add wavelength range
        if (
            hasattr(config.simulation_defaults, "wavelength_nm")
            and config.simulation_defaults.wavelength_nm
        ):
            wl_min, wl_max = config.simulation_defaults.wavelength_nm
            ds.attrs["config_wavelength_min_nm"] = float(wl_min)
            ds.attrs["config_wavelength_max_nm"] = float(wl_max)

        # Add integration setting using the helper function
        if hasattr(config.simulation_defaults, "integrate_wavelength"):
            ds.attrs["config_integrate_wavelength"] = NetCDFSaver._serialize_for_netcdf(
                config.simulation_defaults.integrate_wavelength
            )

        # Add output altitudes
        if (
            hasattr(config.simulation_defaults, "output_altitudes_km")
            and config.simulation_defaults.output_altitudes_km
        ):
            ds.attrs["config_output_altitudes_km"] = NetCDFSaver._serialize_for_netcdf(
                config.simulation_defaults.output_altitudes_km
            )

        # Add path information
        ds.attrs["config_libradtran_bin"] = str(config.paths.libradtran_bin)
        ds.attrs["config_libradtran_data"] = str(config.paths.libradtran_data)

        # Add execution settings
        ds.attrs["config_max_workers"] = int(config.execution.max_workers)
        ds.attrs["config_timeout_seconds"] = int(config.execution.timeout_seconds)


# Expose main classes and functions
__all__ = [
    "OutputType",
    "ParsedOutput",
    "InputDataLoader",
    "ERA5AtmosphereGenerator",
    "OutputParser",
    "OutputToXarray",
    "NetCDFSaver",
]
