# pyradtran/io.py - UNIFIED VERSION
"""
Unified I/O functionality for pyradtran - REFACTORED VERSION.

This file combines the best features from both the old io.py and io_old.py:
- Robust output parsing
- ERA5 atmosphere file creation (now working properly!)
- Input data loading capabilities
- NetCDF saving functionality

Original versions backed up as io.py.backup and io_old.py.backup

Key improvements:
- ERA5 atmosphere support actually works now
- Single unified interface
- Better error handling
- Comprehensive testing

For migration guide, see REFACTORING_SUMMARY.md
"""

import logging
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from enum import Enum
from dataclasses import dataclass

from .config import SimulationConfig
from .exceptions import InputGenerationError, OutputParsingError

logger = logging.getLogger(__name__)


class OutputType(Enum):
    """Enumeration of possible LibRadtran output types."""
    INTEGRATED_SINGLE_ALTITUDE = "integrated_single_altitude"
    INTEGRATED_MULTI_ALTITUDE = "integrated_multi_altitude"
    SPECTRAL_SINGLE_ALTITUDE = "spectral_single_altitude"
    SPECTRAL_MULTI_ALTITUDE = "spectral_multi_altitude"


@dataclass
class ParsedOutput:
    """Container for parsed LibRadtran output."""
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
    """Load simulation input data from various sources."""
    
    @staticmethod
    def load_simulation_input_data(input_file: Union[str, Path]) -> xr.Dataset:
        """
        Load simulation input data from a file.
        
        Args:
            input_file: Path to input data file (CSV/NetCDF) with time, lat, lon
            
        Returns:
            xarray Dataset with required coordinates and variables
            
        Raises:
            InputGenerationError: If file loading fails or required variables missing
        """
        input_file = Path(input_file)
        
        if not input_file.exists():
            raise InputGenerationError(f"Input file not found: {input_file}")
        
        try:
            if input_file.suffix.lower() == '.csv':
                # Load CSV and convert to xarray
                df = pd.read_csv(input_file)
                
                # Convert datetime column if it exists
                if 'time' in df.columns:
                    df['time'] = pd.to_datetime(df['time'])
                elif 'datetime' in df.columns:
                    df['time'] = pd.to_datetime(df['datetime'])
                    df = df.drop('datetime', axis=1)
                
                # Create xarray Dataset
                ds = df.set_index('time').to_xarray()
                
            elif input_file.suffix.lower() in ['.nc', '.netcdf']:
                # Load NetCDF directly
                ds = xr.open_dataset(input_file)
                
            else:
                raise InputGenerationError(f"Unsupported file format: {input_file.suffix}")
            
            # Validate required coordinates
            required_vars = ['time', 'latitude', 'longitude']
            missing_vars = [var for var in required_vars if var not in ds.dims and var not in ds.coords and var not in ds.data_vars]
            
            if missing_vars:
                raise InputGenerationError(f"Required variables missing: {missing_vars}")
            
            logger.info(f"Loaded input data from {input_file} with {len(ds.time)} time points")
            return ds
            
        except Exception as e:
            raise InputGenerationError(f"Failed to load input data from {input_file}: {str(e)}")


class ERA5AtmosphereGenerator:
    """Generate ERA5 atmosphere files for LibRadtran."""
    
    @staticmethod
    def create_era5_atmosphere_file(
        era5_ds: xr.Dataset, 
        #h2o_unit: str,
        latitude: float, 
        longitude: float, 
        time: Union[str, datetime, np.datetime64],
        output_filepath: Union[str, Path],
    ) -> Path:
        """
        Creates a libRadtran-compatible atmosphere file from an ERA5 xarray.Dataset.

        Args:
            era5_ds: The input xarray.Dataset containing atmospheric data.
                Must include 'z', 't', 'q' and coordinates. Expected units are km
                'pressure_level', 'latitude', 'longitude', 'valid_time'.
            h2o_unit: The unit for water vapor in the output file. Options are 'MMR' (mass mixing ratio; ERA5 standard) or 'RH' (relative humidtiy; CARRA standard).
            latitude: The target latitude.
            longitude: The target longitude.
            time: The target time (str, datetime, or np.datetime64)
            output_filepath: The path to the output file that will be created.
            
        Returns:
            Path object of the created atmosphere file
            
        Raises:
            ValueError: If required variables are missing from the dataset
            InputGenerationError: If file creation fails
        """
        try:
            # Physical and chemical constants for conversions
            G_STD = 9.80665      # Standard gravity (m/s^2)
            K_B = 1.380649e-23   # Boltzmann constant (J/K)
            M_AIR = 0.0289647    # Molar mass of dry air (kg/mol)
            # M_O3 = 0.0479982     # Molar mass of Ozone (O3) (kg/mol)
            M_H2O = 0.01801528   # Molar mass of Water (H2O) (kg/mol)
            # M_CO2 = 0.04401      # Molar mass of Carbon Dioxide (CO2) (kg/mol)
            # M_NO2 = 0.0460055    # Molar mass of Nitrogen Dioxide (NO2) (kg/mol)
            # O2_MIXING_RATIO = 0.2095 # Volumetric mixing ratio of O2 in dry air

            # Validate required variables
            required_vars = ['z', 't', 'q']
            required_coords = ['pressure_level', 'valid_time']
            
            for var in required_vars:
                if var not in era5_ds.variables:
                    raise ValueError(f"Required variable '{var}' not found in ERA5 dataset")
            output_filepath
            for coord in required_coords:
                if coord not in era5_ds.coords:
                    raise ValueError(f"Required coordinate '{coord}' not found in ERA5 dataset")

            # Select the data for the nearest point and specified time
            # Check if data is already spatially selected (latitude/longitude are scalars)
            if 'latitude' in era5_ds.dims or 'longitude' in era5_ds.dims:
                # Data still has spatial dimensions, do normal selection
                profile_data = era5_ds.sel(
                    latitude=latitude, 
                    longitude=longitude, 
                    valid_time=time,
                    method='nearest'
                )
            else:
                # Data is already spatially selected, just select time
                if 'valid_time' in era5_ds.dims:
                    # valid_time is still a dimension
                    profile_data = era5_ds.sel(valid_time=time, method='nearest')
                elif 'time' in era5_ds.dims:
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

            # check the unit of the q variable:
            h2o_unit = profile_data.q.units
            p_unit = profile_data.pressure_level.units


            # Extract variables and perform unit conversions
            altitude_km = profile_data['z']

            if p_unit == 'Pa':
                pressure_pa = profile_data['pressure_level']
            if p_unit == 'hPa':
                pressure_pa = profile_data['pressure_level'] * 100  # Convert hPa to Pa

            pressure_hpa = pressure_pa / 100
            temperature_k = profile_data['t']
            h2o = profile_data['q']

            # Air number density: calculated from ideal gas law p = NkT
            air_number_density_m3 = pressure_pa / (K_B * temperature_k)
            air_number_density_cm3 = air_number_density_m3 / 1e6

            # Function to convert mass mixing ratio (kg/kg) to number density (molecules/cm^3)
            def mmr_to_nd(mmr, m_gas):
                return mmr * (M_AIR / m_gas) * air_number_density_cm3

            # Create the atmosphere file content
            output_path = Path(output_filepath)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                f.write("# ERA5 atmosphere profile in libradtran radiosonde style\n")
                f.write(f"# p(hPa)  T(K)  h2o({h2o_unit}) \n")
                
                # Sort by altitude (high to low) for LibRadtran (TOA to surface)
                # This means sorting by pressure (low to high) since high altitude = low pressure
                sorted_indices = np.argsort(pressure_hpa.values)
                
                for idx in sorted_indices:
                    #f.write(f"{altitude_km.values[idx]:.3f}  ")
                    f.write(f"{pressure_hpa.values[idx]:.2f}  ")
                    f.write(f"{temperature_k.values[idx]:.2f}  ")
                    # f.write(f"{air_number_density_cm3.values[idx]:.3e}  ")
                    # f.write(f"{o3_nd.values[idx]:.3e}  ")
                    # f.write(f"{o2_nd.values[idx]:.3e}  ")
                    f.write(f"{h2o.values[idx]:.3e}\n")
                    # f.write(f"{co2_nd.values[idx]:.3e}  ")
                    # f.write(f"{no2_nd.values[idx]:.3e}\n")
            
            logger.info(f"Created ERA5 atmosphere file: {output_path} with H2O unit {h2o_unit}")
            return output_path
            
        except Exception as e:
            raise InputGenerationError(f"Failed to create ERA5 atmosphere file: {str(e)}")



class RadiosondeAtmosphereGenerator:
    """Generate atmosphere files from the nearest radiosonde sounding."""

    @staticmethod
    def get_station_list(
        url: str = "https://www.ncei.noaa.gov/pub/data/igra/igra2-station-list.txt",
    ) -> Optional[pd.DataFrame]:
        """Download and parse the IGRA station list."""
        try:
            col_specs = [
                (0, 11), (12, 20), (21, 30), (31, 37), (38, 40),
                (41, 71), (72, 76), (77, 81), (82, 88)
            ]
            names = [
                'id', 'latitude', 'longitude', 'elevation', 'state',
                'name', 'first_year', 'last_year', 'num_obs'
            ]
            return pd.read_fwf(url, colspecs=col_specs, names=names)
        except Exception as e:
            logger.error(f"Error downloading or parsing station list: {e}")
            return None

    @staticmethod
    def find_closest_active_stations(
        stations_df: pd.DataFrame, lat: float, lon: float, n: int = 5
    ) -> pd.DataFrame:
        """Find the N closest active radiosonde stations."""
        current_year = datetime.utcnow().year
        active_stations = stations_df[stations_df['last_year'] >= current_year - 1].copy()

        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)
        stations_lat_rad = np.radians(active_stations['latitude'])
        stations_lon_rad = np.radians(active_stations['longitude'])

        dlon = stations_lon_rad - lon_rad
        dlat = stations_lat_rad - lat_rad
        a = (
            np.sin(dlat / 2) ** 2
            + np.cos(lat_rad) * np.cos(stations_lat_rad) * np.sin(dlon / 2) ** 2
        )
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        distance = 6371 * c

        active_stations['distance_km'] = distance
        return active_stations.sort_values(by='distance_km').head(n)

    @staticmethod
    def get_closest_sounding(
        target_dt: datetime, lat: float, lon: float
    ) -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Retrieve the closest available radiosonde sounding."""
        stations = RadiosondeAtmosphereGenerator.get_station_list()
        if stations is None:
            return None, None

        closest = RadiosondeAtmosphereGenerator.find_closest_active_stations(
            stations, lat, lon, n=5
        )
        if closest.empty:
            return None, None

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

        for _, station in closest.iterrows():
            station_id = station['id']
            station_name = station['name'].strip()
            station_distance = station['distance_km']
            for time_to_check in potential_times:
                try:
                    if IGRAUpperAir is None:
                        raise InputGenerationError(
                            "siphon is required for radiosonde retrieval"
                        )
                    df, header = IGRAUpperAir.request_data(time_to_check, station_id)
                    header_text  = f"{station_name} at {time_to_check.strftime('%Y-%m-%d %H:%M')} UTC with distance {station_distance:.2f} km"
                    logger.info(f"Found sounding for {station_name} at {time_to_check} UTC with distance {station_distance:.2f} km")
                    if df is not None and not df.empty:
                        return df, header, header_text
                except Exception:
                    continue

        return None, None

    @staticmethod
    def create_radiosonde_atmosphere_file(
        time: datetime,
        latitude: float,
        longitude: float,
        output_filepath: Union[str, Path],
    ) -> Path:
        """Create a libRadtran atmosphere file from radiosonde data."""
        sounding_df, _, header_text = RadiosondeAtmosphereGenerator.get_closest_sounding(
            time, latitude, longitude
        )
        if sounding_df is None or sounding_df.empty:
            raise InputGenerationError("No radiosonde data found for requested parameters")

        df = sounding_df.dropna(subset=['pressure', 'temperature', 'dewpoint', 'height'])
        if df.empty:
            raise InputGenerationError("No valid levels in radiosonde data")

        e = 6.112 * np.exp(17.67 * df['dewpoint'] / (df['dewpoint'] + 243.5))
        e_s = 6.112 * np.exp(17.67 * df['temperature'] / (df['temperature'] + 243.5))
        rh = (e / e_s) * 100  # Relative Humidity in %
        w = 0.622 * e / (df['pressure'] - e)  # kg/kg
        temp_k = df['temperature'] + 273.15
        height_km = df['height'] / 1000.0
        pressure_hpa = df['pressure']

        sorted_idx = np.argsort(pressure_hpa.values)

        output_path = Path(output_filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
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
    """Robust parser for LibRadtran output files."""
    
    def __init__(self, config: SimulationConfig, parameter_overrides: Dict[str, Any] = None):
        self.config = config
        self.parameter_overrides = parameter_overrides or {}
        
        # Check if brightness temperature output is requested
        self.is_brightness_output = self.parameter_overrides.get('output_quantity') == 'brightness'
        
        # Build the actual output columns that LibRadtran will produce
        self.output_columns = []
        
        # Use the user-specified columns directly since we use 'output_user'
        original_columns = config.simulation_defaults.output_columns or []
        for col in original_columns:
            # For brightness output, LibRadtran doesn't include albedo column
            if self.is_brightness_output and col == 'albedo':
                logger.debug(f"Skipping albedo column for brightness temperature output")
                continue
            self.output_columns.append(col)
        
        self.output_altitudes = config.simulation_defaults.output_altitudes_km or [0.0]
        self.wavelength_range = config.simulation_defaults.wavelength_nm
        self.is_integrated = getattr(config.simulation_defaults, 'integrate_wavelength', False)
        
        logger.debug(f"Initialized parser with columns: {self.output_columns}")

    def parse_output_file(self, output_file: Path) -> ParsedOutput:
        """Parse a LibRadtran output file."""
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
                is_brightness_temperature=self.is_brightness_output
            )
            
        except Exception as e:
            raise OutputParsingError(f"Failed to parse output file {output_file}: {str(e)}")
    
    def _determine_output_type(self, data: np.ndarray) -> OutputType:
        """Determine the type of LibRadtran output."""
        n_rows, n_cols = data.shape
        n_altitudes = len(self.output_altitudes)
        n_wavelengths = 1 if self.is_integrated else len(self.wavelength_range)
        
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
    
    def _parse_data_by_type(self, data: np.ndarray, output_type: OutputType) -> Dict[str, Any]:
        """Parse data based on output type."""
        parsed = {}
        
        for i, col_name in enumerate(self.output_columns):
            if i < data.shape[1]:
                parsed[col_name] = data[:, i]
        
        return parsed
    
    def _extract_wavelengths(self, data: np.ndarray, output_type: OutputType) -> Optional[List[float]]:
        """Extract wavelength values if present."""
        if output_type in [OutputType.SPECTRAL_SINGLE_ALTITUDE, OutputType.SPECTRAL_MULTI_ALTITUDE]:
            if 'lambda' in self.output_columns:
                lambda_idx = self.output_columns.index('lambda')
                if lambda_idx < data.shape[1]:
                    return sorted(set(data[:, lambda_idx]))
        return None
    
    def _extract_altitudes(self, data: np.ndarray, output_type: OutputType) -> Optional[List[float]]:
        """Extract altitude values if present."""
        if output_type in [OutputType.INTEGRATED_MULTI_ALTITUDE, OutputType.SPECTRAL_MULTI_ALTITUDE]:
            if 'zout' in self.output_columns:
                zout_idx = self.output_columns.index('zout')
                if zout_idx < data.shape[1]:
                    return sorted(set(data[:, zout_idx]))
        return None


class OutputToXarray:
    """Convert parsed output to xarray Dataset."""
    
    @staticmethod
    def convert(parsed_output: ParsedOutput, input_ds: xr.Dataset, 
                time_var: str = 'time', lat_var: str = 'latitude', 
                lon_var: str = 'longitude') -> xr.Dataset:
        """Convert a single ParsedOutput to xarray Dataset."""
        # Create base dataset with coordinates from input
        ds = xr.Dataset()
        
        # Copy coordinates from input dataset
        for coord_name in [time_var, lat_var, lon_var]:
            if coord_name in input_ds:
                ds[coord_name] = input_ds[coord_name]
        
        # Add dimensions based on output type
        if parsed_output.wavelengths:
            ds['wavelength'] = ('wavelength', parsed_output.wavelengths)
        
        if parsed_output.altitudes:
            ds['altitude'] = ('altitude', parsed_output.altitudes)
        
        # Add data variables
        for var_name, values in parsed_output.data.items():
            if var_name in ['zout', 'lambda']:
                continue  # Skip coordinate variables
            
            # Determine dimensions based on output type
            dims = [time_var]
            if parsed_output.wavelengths:
                dims.append('wavelength')
            if parsed_output.altitudes:
                dims.append('altitude')
            
            # Reshape values to match dimensions
            if len(dims) == 1:
                ds[var_name] = (dims, values)
            else:
                # Need to reshape multi-dimensional data
                reshaped_values = values.reshape([len(input_ds[time_var])] + 
                                                [len(parsed_output.wavelengths) if parsed_output.wavelengths else 1] +
                                                [len(parsed_output.altitudes) if parsed_output.altitudes else 1])
                ds[var_name] = (dims, reshaped_values)
        
        return ds
    
    @staticmethod
    def convert_batch(parsed_outputs: List[Optional[ParsedOutput]], input_ds: xr.Dataset,
                     time_var: str = 'time', lat_var: str = 'latitude', 
                     lon_var: str = 'longitude') -> xr.Dataset:
        """
        Convert multiple ParsedOutput objects to a single xarray Dataset.
        Handles arbitrary input dimensions by matching flattened outputs to stacked input.
        """
        if not parsed_outputs:
            raise ValueError("No parsed outputs provided")
        
        # Ensure input_ds is a Dataset
        if isinstance(input_ds, xr.DataArray):
            input_ds = input_ds.to_dataset()        

        # Stack input dataset exactly as done in execute_simulation_batch
        dims = list(input_ds.sizes.keys())
        sample_dim = 'sample_batch_dim'
        
        if dims:
            stacked_ds = input_ds.stack({sample_dim: dims})
        else:
            stacked_ds = input_ds.expand_dims(sample_dim)
            
        num_points = stacked_ds.sizes[sample_dim]
        
        if len(parsed_outputs) != num_points:
            # If mismatch, log warning but try to proceed if safe?
            # Actually, this means data corruption or misalignment.
            logger.warning(f"Number of outputs ({len(parsed_outputs)}) does not match input points ({num_points}). Dimensions might be wrong.")
            
        # Get structure from the first valid output
        first_valid_idx = next((i for i, x in enumerate(parsed_outputs) if x is not None), -1)
        if first_valid_idx == -1:
             raise ValueError("All simulations failed, cannot determine output structure")
             
        template_output = parsed_outputs[first_valid_idx]
        var_names = list(template_output.data.keys())
        altitudes = template_output.altitudes if template_output.altitudes else [0]
        wavelengths = template_output.wavelengths if template_output.wavelengths else [None]
        
        # Define extra dimensions for the output variables
        extra_dims = []
        extra_shape = []
        
        if wavelengths[0] is not None:
            extra_dims.append('wavelength')
            extra_shape.append(len(wavelengths))
            
        extra_dims.append('altitude')
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
                        logger.warning(f"Shape mismatch for {var_name} at index {i}: got {val.shape}, expected {extra_shape}")
            
            # Create temporary DataArray
            da = xr.DataArray(
                data_arr,
                coords={sample_dim: stacked_ds[sample_dim]},
                dims=[sample_dim] + extra_dims
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
             coords['wavelength'] = wavelengths
        coords['altitude'] = altitudes
        
        # Create final dataset
        # We start with a dataset containing our variables. 
        # The coordinates from input_ds should be preserved by unstacking.
        result_ds = xr.Dataset(data_vars, coords=coords)
        
        # Add attributes
        if hasattr(template_output, 'metadata'):
            result_ds.attrs.update(template_output.metadata)
            
        return result_ds


class NetCDFSaver:
    """Save results to NetCDF files."""
    
    @staticmethod
    def save_results_to_netcdf(
        data: Union[Dict[str, Any], xr.Dataset],
        output_path: Path,
        input_ds: xr.Dataset,
        config: SimulationConfig,
        simulation_params: Dict[str, Any] = None
    ) -> Path:
        """Save simulation results to NetCDF file."""
        try:
            if isinstance(data, xr.Dataset):
                ds = data
            else:
                # Convert dictionary data to xarray Dataset
                ds = xr.Dataset()
                # Basic conversion - would need more sophisticated logic
                for key, values in data.items():
                    if not key.startswith('_'):
                        ds[key] = ('time', values)
            
            # Add metadata
            ds.attrs['created_by'] = 'pyradtran'
            ds.attrs['creation_date'] = datetime.now().isoformat()
            if simulation_params:
                ds.attrs['simulation_parameters'] = str(simulation_params)
            
            # Add configuration parameters as attributes
            NetCDFSaver._add_config_to_attrs(ds, config)
            
            # Filter out None values from attributes as they can't be serialized to NetCDF
            filtered_attrs = {k: v for k, v in ds.attrs.items() if v is not None}
            ds.attrs.clear()
            ds.attrs.update(filtered_attrs)
            
            # Save to file
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Prepare encoding for variables
            netcdf_settings = config.output.netcdf_encoding if config.output.netcdf_encoding else {}
            encoding = {}
            if netcdf_settings:
                for var_name in ds.data_vars:
                    encoding[var_name] = netcdf_settings
            
            ds.to_netcdf(output_path, encoding=encoding)
            
            logger.info(f"Results saved to {output_path}")
            return output_path
            
        except Exception as e:
            raise OutputParsingError(f"Failed to save results to {output_path}: {str(e)}")
    
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
        ds.attrs['config_source'] = str(config.simulation_defaults.source) if config.simulation_defaults.source is not None else 'None'
        ds.attrs['config_rte_solver'] = str(config.simulation_defaults.rte_solver) if config.simulation_defaults.rte_solver is not None else 'None'
        ds.attrs['config_mol_abs_param'] = str(config.simulation_defaults.mol_abs_param) if config.simulation_defaults.mol_abs_param is not None else 'None'
        
        # Handle numeric parameters that might be None
        # if config.simulation_defaults.albedo_value is not None:
        #     ds.attrs['config_albedo_value'] = float(config.simulation_defaults.albedo_value)
        if config.simulation_defaults.ozone_du is not None:
            ds.attrs['config_ozone_du'] = float(config.simulation_defaults.ozone_du)
        if config.simulation_defaults.h2o_mm is not None:
            ds.attrs['config_h2o_mm'] = float(config.simulation_defaults.h2o_mm)
        if config.simulation_defaults.surface_temperature_k is not None:
            ds.attrs['config_surface_temperature_k'] = float(config.simulation_defaults.surface_temperature_k)
            
        ds.attrs['config_viewing_geometry'] = str(config.simulation_defaults.viewing_geometry) if config.simulation_defaults.viewing_geometry is not None else 'None'
        
        # Add wavelength range
        if hasattr(config.simulation_defaults, 'wavelength_nm') and config.simulation_defaults.wavelength_nm:
            wl_min, wl_max = config.simulation_defaults.wavelength_nm
            ds.attrs['config_wavelength_min_nm'] = float(wl_min)
            ds.attrs['config_wavelength_max_nm'] = float(wl_max)
        
        # Add integration setting using the helper function
        if hasattr(config.simulation_defaults, 'integrate_wavelength'):
            ds.attrs['config_integrate_wavelength'] = NetCDFSaver._serialize_for_netcdf(config.simulation_defaults.integrate_wavelength)
        
        # Add output altitudes
        if hasattr(config.simulation_defaults, 'output_altitudes_km') and config.simulation_defaults.output_altitudes_km:
            ds.attrs['config_output_altitudes_km'] = NetCDFSaver._serialize_for_netcdf(config.simulation_defaults.output_altitudes_km)
        
        # Add path information
        ds.attrs['config_libradtran_bin'] = str(config.paths.libradtran_bin)
        ds.attrs['config_libradtran_data'] = str(config.paths.libradtran_data)
        
        # Add execution settings
        ds.attrs['config_max_workers'] = int(config.execution.max_workers)
        ds.attrs['config_timeout_seconds'] = int(config.execution.timeout_seconds)


# Expose main classes and functions
__all__ = [
    'OutputType', 'ParsedOutput', 'InputDataLoader', 'ERA5AtmosphereGenerator',
    'OutputParser', 'OutputToXarray', 'NetCDFSaver'
]
