# pyradtran/io_new.py
"""
Clean, robust I/O functionality for pyradtran.

This module provides a complete rewrite of the I/O functionality with:
- Clear separation of concerns
- Robust output parsing for all LibRadtran output types
- Comprehensive error handling
- Simple, testable design
"""

import logging
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime
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
    
    def __post_init__(self):
        """Initialize metadata if not provided."""
        if self.metadata is None:
            self.metadata = {}
    
    def to_xarray(self, input_ds: xr.Dataset, 
                  time_var: str = 'time',
                  lat_var: str = 'latitude', 
                  lon_var: str = 'longitude') -> xr.Dataset:
        """Convert parsed results to xarray Dataset."""
        return OutputToXarray.convert(self, input_ds, time_var, lat_var, lon_var)


class OutputParser:
    """Robust parser for LibRadtran output files."""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        # LibRadtran will always include zout and lambda in the output due to our modifications
        self.original_columns = config.simulation_defaults.output_columns or []
        
        # Build the actual output columns that LibRadtran will produce
        self.output_columns = []
        
        # Always add zout first (altitude)
        if 'zout' not in self.output_columns:
            self.output_columns.append('zout')
        
        # Always add lambda second (wavelength)  
        if 'lambda' not in self.output_columns:
            self.output_columns.append('lambda')
        
        # Add the user-specified columns, skipping zout and lambda if they're already included
        for col in self.original_columns:
            if col not in ['zout', 'lambda'] and col not in self.output_columns:
                self.output_columns.append(col)
        
        self.output_altitudes = config.simulation_defaults.output_altitudes_km or [0.0]
        self.wavelength_range = config.simulation_defaults.wavelength_nm
        self.is_integrated = getattr(config.simulation_defaults, 'integrate_wavelength', False)
        
        logger.debug(f"Initialized parser with columns: {self.output_columns}")
        logger.debug(f"Original user columns: {self.original_columns}")
    
    def parse(self, output_file: Path) -> ParsedOutput:
        """Parse a LibRadtran output file."""
        if not output_file.exists():
            raise OutputParsingError(f"Output file does not exist: {output_file}")
        
        # Read and clean the file
        data_lines = self._read_data_lines(output_file)
        if not data_lines:
            raise OutputParsingError(f"No data found in output file: {output_file}")
        
        # Determine output type
        output_type = self._determine_output_type(data_lines)
        logger.info(f"Detected output type: {output_type.value}")
        
        # Parse based on type
        try:
            if output_type == OutputType.INTEGRATED_SINGLE_ALTITUDE:
                data = self._parse_integrated_single_altitude(data_lines)
                return ParsedOutput(output_type, data, source_file=output_file)
                
            elif output_type == OutputType.INTEGRATED_MULTI_ALTITUDE:
                data = self._parse_integrated_multi_altitude(data_lines)
                return ParsedOutput(output_type, data, altitudes=self.output_altitudes, source_file=output_file)
                
            elif output_type == OutputType.SPECTRAL_SINGLE_ALTITUDE:
                data, wavelengths = self._parse_spectral_single_altitude(data_lines)
                return ParsedOutput(output_type, data, wavelengths=wavelengths, source_file=output_file)
                
            elif output_type == OutputType.SPECTRAL_MULTI_ALTITUDE:
                data, wavelengths = self._parse_spectral_multi_altitude(data_lines)
                return ParsedOutput(output_type, data, wavelengths=wavelengths, altitudes=self.output_altitudes, source_file=output_file)
                
        except Exception as e:
            raise OutputParsingError(f"Failed to parse {output_type.value} output: {e}")
    
    def _read_data_lines(self, output_file: Path) -> List[str]:
        """Read and filter data lines from output file."""
        try:
            with open(output_file, 'r') as f:
                lines = f.readlines()
            
            # Filter out comments and empty lines
            data_lines = []
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#'):
                    data_lines.append(line)
            
            return data_lines
            
        except Exception as e:
            raise OutputParsingError(f"Error reading output file {output_file}: {e}")
    
    def _determine_output_type(self, data_lines: List[str]) -> OutputType:
        """Determine the type of LibRadtran output based on data structure."""
        num_lines = len(data_lines)
        num_altitudes = len(self.output_altitudes)
        has_lambda = 'lambda' in self.output_columns
        
        # First check if we can parse any lines at all (for error detection)
        parseable_lines = 0
        for line in data_lines:
            values = line.split()
            try:
                # Try to convert to numbers
                [float(val) for val in values]
                parseable_lines += 1
            except ValueError:
                continue
        
        if parseable_lines == 0:
            raise OutputParsingError("No parseable numeric data found in output file")
        
        # Check if wavelength integration was requested
        if self.is_integrated:
            if num_altitudes == 1 and num_lines == 1:
                return OutputType.INTEGRATED_SINGLE_ALTITUDE
            elif num_altitudes > 1 and num_lines == num_altitudes:
                return OutputType.INTEGRATED_MULTI_ALTITUDE
            else:
                logger.warning(f"Expected {num_altitudes} lines for integrated multi-altitude, got {num_lines}")
                # Fall back to guessing
                if num_lines == 1:
                    return OutputType.INTEGRATED_SINGLE_ALTITUDE
                else:
                    return OutputType.INTEGRATED_MULTI_ALTITUDE
        
        # Spectral output (no integration)
        else:
            if has_lambda:
                # Parse first line to check if lambda values match expected range
                try:
                    first_line = data_lines[0].split()
                    lambda_val = float(first_line[self.output_columns.index('lambda')])
                    wl_min, wl_max = self.wavelength_range
                    
                    if wl_min <= lambda_val <= wl_max:
                        # This looks like spectral data
                        if num_altitudes == 1:
                            return OutputType.SPECTRAL_SINGLE_ALTITUDE
                        else:
                            return OutputType.SPECTRAL_MULTI_ALTITUDE
                except (ValueError, IndexError):
                    pass
            
            # Guess based on number of lines
            expected_spectral_lines = self._estimate_spectral_lines()
            if num_altitudes == 1:
                return OutputType.SPECTRAL_SINGLE_ALTITUDE
            elif num_lines >= expected_spectral_lines * num_altitudes * 0.8:  # Allow 20% tolerance
                return OutputType.SPECTRAL_MULTI_ALTITUDE
            else:
                return OutputType.SPECTRAL_SINGLE_ALTITUDE
    
    def _estimate_spectral_lines(self) -> int:
        """Estimate number of spectral lines based on wavelength range."""
        wl_min, wl_max = self.wavelength_range
        # LibRadtran typically outputs 1 nm resolution
        return int(wl_max - wl_min + 1)
    
    def _parse_integrated_single_altitude(self, data_lines: List[str]) -> Dict[str, float]:
        """Parse integrated output for single altitude."""
        if len(data_lines) != 1:
            raise OutputParsingError(f"Expected 1 line for integrated single altitude, got {len(data_lines)}")
        
        values = data_lines[0].split()
        if len(values) != len(self.output_columns):
            raise OutputParsingError(f"Expected {len(self.output_columns)} columns, got {len(values)}")
        
        result = {}
        for i, col_name in enumerate(self.output_columns):
            try:
                # Skip zout and lambda in the result dictionary since they're metadata
                if col_name in ['zout', 'lambda']:
                    continue
                result[col_name] = float(values[i])
            except (ValueError, IndexError) as e:
                raise OutputParsingError(f"Error parsing column {col_name}: {e}")
        
        return result
    
    def _parse_integrated_multi_altitude(self, data_lines: List[str]) -> Dict[str, Dict[float, float]]:
        """Parse integrated output for multiple altitudes."""
        num_altitudes = len(self.output_altitudes)
        if len(data_lines) != num_altitudes:
            logger.warning(f"Expected {num_altitudes} lines for multi-altitude, got {len(data_lines)}")
        
        # Initialize result structure for non-metadata columns only
        result = {}
        for col in self.output_columns:
            if col not in ['zout', 'lambda']:
                result[col] = {}
        
        zout_idx = self.output_columns.index('zout') if 'zout' in self.output_columns else None
        lambda_idx = self.output_columns.index('lambda') if 'lambda' in self.output_columns else None
        
        for line_idx, line in enumerate(data_lines):
            values = line.split()
            
            if len(values) != len(self.output_columns):
                raise OutputParsingError(f"Line {line_idx}: expected {len(self.output_columns)} columns, got {len(values)}")
            
            # Extract altitude from the zout column
            if zout_idx is not None:
                altitude = float(values[zout_idx])
            else:
                # Fallback to config order
                altitude = self.output_altitudes[line_idx] if line_idx < len(self.output_altitudes) else 0.0
            
            for col_idx, col_name in enumerate(self.output_columns):
                # Skip metadata columns
                if col_name in ['zout', 'lambda']:
                    continue
                    
                try:
                    result[col_name][altitude] = float(values[col_idx])
                except (ValueError, IndexError) as e:
                    raise OutputParsingError(f"Error parsing column {col_name} at altitude {altitude}: {e}")
        
        return result
    
    def _parse_spectral_single_altitude(self, data_lines: List[str]) -> tuple[Dict[str, Dict[float, float]], List[float]]:
        """Parse spectral output for single altitude."""
        # Initialize result structure for non-metadata columns only
        result = {}
        for col in self.output_columns:
            if col not in ['zout', 'lambda']:
                result[col] = {}
        
        wavelengths = []
        
        zout_idx = self.output_columns.index('zout') if 'zout' in self.output_columns else None
        lambda_idx = self.output_columns.index('lambda') if 'lambda' in self.output_columns else None
        
        valid_lines = 0
        for line in data_lines:
            values = line.split()
            if len(values) != len(self.output_columns):
                logger.warning(f"Line has {len(values)} columns, expected {len(self.output_columns)}")
                continue
            
            # Extract wavelength from the lambda column
            if lambda_idx is not None:
                try:
                    wavelength = float(values[lambda_idx])
                    wavelengths.append(wavelength)
                    valid_lines += 1
                except (ValueError, IndexError):
                    continue
            else:
                # If no lambda column, estimate wavelength
                wavelength = self.wavelength_range[0] + len(wavelengths)
                wavelengths.append(wavelength)
                valid_lines += 1
            
            # Extract other columns (skip metadata columns)
            for col_idx, col_name in enumerate(self.output_columns):
                if col_name in ['zout', 'lambda']:
                    continue
                    
                try:
                    result[col_name][wavelength] = float(values[col_idx])
                except (ValueError, IndexError) as e:
                    logger.warning(f"Error parsing {col_name} at wavelength {wavelength}: {e}")
                    result[col_name][wavelength] = np.nan
        
        if valid_lines == 0:
            raise OutputParsingError("No valid spectral data lines found")
        
        return result, wavelengths
    
    def _parse_spectral_multi_altitude(self, data_lines: List[str]) -> tuple[Dict[str, Dict[float, Dict[float, float]]], List[float]]:
        """Parse spectral output for multiple altitudes."""
        # Initialize result structure for non-metadata columns only
        result = {}
        for col in self.output_columns:
            if col not in ['zout', 'lambda']:
                result[col] = {}
                for altitude in self.output_altitudes:
                    result[col][altitude] = {}
        
        zout_idx = self.output_columns.index('zout') if 'zout' in self.output_columns else None
        lambda_idx = self.output_columns.index('lambda') if 'lambda' in self.output_columns else None
        wavelengths = []
        
        # Parse all lines and extract wavelength/altitude combinations
        for line in data_lines:
            values = line.split()
            
            if len(values) != len(self.output_columns):
                continue
            
            # Extract altitude and wavelength from the data
            try:
                if zout_idx is not None:
                    altitude = float(values[zout_idx])
                else:
                    continue  # Can't determine altitude without zout column
                    
                if lambda_idx is not None:
                    wavelength = float(values[lambda_idx])
                else:
                    continue  # Can't determine wavelength without lambda column
                
                # Collect unique wavelengths (from first altitude)
                if altitude == self.output_altitudes[0] and wavelength not in wavelengths:
                    wavelengths.append(wavelength)
                
                # Extract data columns (skip metadata columns)
                for col_idx, col_name in enumerate(self.output_columns):
                    if col_name in ['zout', 'lambda']:
                        continue
                        
                    try:
                        value = float(values[col_idx])
                        result[col_name][altitude][wavelength] = value
                    except (ValueError, IndexError):
                        result[col_name][altitude][wavelength] = np.nan
                        
            except (ValueError, IndexError) as e:
                logger.warning(f"Error parsing line: {line.strip()}: {e}")
                continue
        
        # Sort wavelengths for consistency
        wavelengths.sort()
        
        return result, wavelengths


class OutputToXarray:
    """Convert parsed LibRadtran output to xarray Datasets."""
    
    @staticmethod
    def convert(parsed_output: ParsedOutput, 
                input_ds: xr.Dataset,
                time_var: str = 'time',
                lat_var: str = 'latitude', 
                lon_var: str = 'longitude') -> xr.Dataset:
        """Convert ParsedOutput to xarray Dataset."""
        
        # Create base dataset with input coordinates
        output_ds = xr.Dataset(coords={time_var: input_ds[time_var]})
        
        # Copy lat/lon coordinates if they exist
        for coord_name in [lat_var, lon_var]:
            if coord_name in input_ds.coords:
                output_ds.coords[coord_name] = input_ds.coords[coord_name]
        
        if parsed_output.output_type == OutputType.INTEGRATED_SINGLE_ALTITUDE:
            output_ds = OutputToXarray._add_integrated_single_altitude(
                output_ds, parsed_output, time_var)
                
        elif parsed_output.output_type == OutputType.INTEGRATED_MULTI_ALTITUDE:
            output_ds = OutputToXarray._add_integrated_multi_altitude(
                output_ds, parsed_output, time_var)
                
        elif parsed_output.output_type == OutputType.SPECTRAL_SINGLE_ALTITUDE:
            output_ds = OutputToXarray._add_spectral_single_altitude(
                output_ds, parsed_output, time_var)
                
        elif parsed_output.output_type == OutputType.SPECTRAL_MULTI_ALTITUDE:
            output_ds = OutputToXarray._add_spectral_multi_altitude(
                output_ds, parsed_output, time_var)
        
        # Add metadata
        output_ds.attrs.update({
            'created': datetime.now().isoformat(),
            'source': 'PyRadtran',
            'output_type': parsed_output.output_type.value,
            'source_file': str(parsed_output.source_file) if parsed_output.source_file else None
        })
        
        return output_ds
    
    @staticmethod
    def convert_batch(parsed_outputs: List[ParsedOutput], 
                     input_ds: xr.Dataset,
                     time_var: str = 'time',
                     lat_var: str = 'latitude', 
                     lon_var: str = 'longitude') -> xr.Dataset:
        """Convert multiple ParsedOutputs to a single xarray Dataset."""
        
        if not parsed_outputs:
            raise ValueError("No parsed outputs provided")
        
        # Check that all outputs have the same type
        output_types = {po.output_type for po in parsed_outputs}
        if len(output_types) > 1:
            raise ValueError(f"Mixed output types not supported: {output_types}")
        
        output_type = next(iter(output_types))
        
        # Create base dataset with input coordinates
        output_ds = xr.Dataset()
        
        # Copy coordinates from input dataset
        for coord_name in [time_var, lat_var, lon_var]:
            if coord_name in input_ds.coords:
                output_ds.coords[coord_name] = input_ds.coords[coord_name]
            elif coord_name in input_ds.data_vars:
                output_ds.coords[coord_name] = input_ds.data_vars[coord_name]
        
        # Get dimensions and coordinates
        time_len = len(input_ds[time_var])
        
        # Handle spectral data
        if output_type in [OutputType.SPECTRAL_SINGLE_ALTITUDE, OutputType.SPECTRAL_MULTI_ALTITUDE]:
            # Get wavelengths from first parsed output
            wavelengths = parsed_outputs[0].wavelengths
            if wavelengths:
                output_ds.coords['wavelength'] = ('wavelength', wavelengths, {'units': 'nm'})
        
        # Handle multi-altitude data  
        if output_type in [OutputType.INTEGRATED_MULTI_ALTITUDE, OutputType.SPECTRAL_MULTI_ALTITUDE]:
            altitudes = parsed_outputs[0].altitudes
            if altitudes:
                output_ds.coords['altitude'] = ('altitude', altitudes, {'units': 'km'})
        
        # Now combine data from all parsed outputs based on type
        if output_type == OutputType.INTEGRATED_SINGLE_ALTITUDE:
            output_ds = OutputToXarray._combine_integrated_single_altitude(
                output_ds, parsed_outputs, time_var)
                
        elif output_type == OutputType.INTEGRATED_MULTI_ALTITUDE:
            output_ds = OutputToXarray._combine_integrated_multi_altitude(
                output_ds, parsed_outputs, time_var)
                
        elif output_type == OutputType.SPECTRAL_SINGLE_ALTITUDE:
            output_ds = OutputToXarray._combine_spectral_single_altitude(
                output_ds, parsed_outputs, time_var)
                
        elif output_type == OutputType.SPECTRAL_MULTI_ALTITUDE:
            output_ds = OutputToXarray._combine_spectral_multi_altitude(
                output_ds, parsed_outputs, time_var)
        
        # Add metadata
        output_ds.attrs.update({
            'created': datetime.now().isoformat(),
            'source': 'PyRadtran',
            'output_type': output_type.value,
            'num_simulations': len(parsed_outputs)
        })
        
        return output_ds
    
    @staticmethod
    def _add_integrated_single_altitude(ds: xr.Dataset, parsed: ParsedOutput, time_var: str) -> xr.Dataset:
        """Add integrated single altitude data to dataset."""
        time_len = len(ds[time_var])
        
        for var_name, value in parsed.data.items():
            # Replicate single value across all time steps
            data_array = np.full(time_len, value)
            ds[var_name] = xr.DataArray(
                data_array,
                dims=(time_var,),
                coords={time_var: ds[time_var]},
                attrs={'units': OutputToXarray._get_units(var_name)}
            )
        
        return ds
    
    @staticmethod
    def _add_integrated_multi_altitude(ds: xr.Dataset, parsed: ParsedOutput, time_var: str) -> xr.Dataset:
        """Add integrated multi-altitude data to dataset."""
        time_len = len(ds[time_var])
        altitudes = parsed.altitudes
        
        # Add altitude coordinate
        ds.coords['altitude'] = ('altitude', altitudes, {'units': 'km'})
        
        for var_name, alt_dict in parsed.data.items():
            # Create 2D array [time, altitude]
            data_array = np.full((time_len, len(altitudes)), np.nan)
            
            for alt_idx, altitude in enumerate(altitudes):
                if altitude in alt_dict:
                    data_array[:, alt_idx] = alt_dict[altitude]
            
            ds[var_name] = xr.DataArray(
                data_array,
                dims=(time_var, 'altitude'),
                coords={time_var: ds[time_var], 'altitude': ds['altitude']},
                attrs={'units': OutputToXarray._get_units(var_name)}
            )
        
        return ds
    
    @staticmethod
    def _add_spectral_single_altitude(ds: xr.Dataset, parsed: ParsedOutput, time_var: str) -> xr.Dataset:
        """Add spectral single altitude data to dataset."""
        time_len = len(ds[time_var])
        wavelengths = parsed.wavelengths
        
        # Add wavelength coordinate
        ds.coords['wavelength'] = ('wavelength', wavelengths, {'units': 'nm'})
        
        for var_name, wl_dict in parsed.data.items():
            # Create 2D array [time, wavelength]
            data_array = np.full((time_len, len(wavelengths)), np.nan)
            
            for wl_idx, wavelength in enumerate(wavelengths):
                if wavelength in wl_dict:
                    data_array[:, wl_idx] = wl_dict[wavelength]
            
            ds[var_name] = xr.DataArray(
                data_array,
                dims=(time_var, 'wavelength'),
                coords={time_var: ds[time_var], 'wavelength': ds['wavelength']},
                attrs={'units': OutputToXarray._get_units(var_name)}
            )
        
        return ds
    
    @staticmethod
    def _add_spectral_multi_altitude(ds: xr.Dataset, parsed: ParsedOutput, time_var: str) -> xr.Dataset:
        """Add spectral multi-altitude data to dataset."""
        time_len = len(ds[time_var])
        wavelengths = parsed.wavelengths
        altitudes = parsed.altitudes
        
        # Add coordinates
        ds.coords['wavelength'] = ('wavelength', wavelengths, {'units': 'nm'})
        ds.coords['altitude'] = ('altitude', altitudes, {'units': 'km'})
        
        for var_name, alt_dict in parsed.data.items():
            # Create 3D array [time, altitude, wavelength]
            data_array = np.full((time_len, len(altitudes), len(wavelengths)), np.nan)
            
            for alt_idx, altitude in enumerate(altitudes):
                if altitude in alt_dict:
                    wl_dict = alt_dict[altitude]
                    for wl_idx, wavelength in enumerate(wavelengths):
                        if wavelength in wl_dict:
                            data_array[:, alt_idx, wl_idx] = wl_dict[wavelength]
            
            ds[var_name] = xr.DataArray(
                data_array,
                dims=(time_var, 'altitude', 'wavelength'),
                coords={
                    time_var: ds[time_var], 
                    'altitude': ds['altitude'],
                    'wavelength': ds['wavelength']
                },
                attrs={'units': OutputToXarray._get_units(var_name)}
            )
        
        return ds
    
    @staticmethod
    def _combine_integrated_single_altitude(ds: xr.Dataset, parsed_outputs: List[ParsedOutput], time_var: str) -> xr.Dataset:
        """Combine integrated single altitude data from multiple simulations."""
        time_len = len(ds[time_var])
        
        # Get all variable names from first output
        var_names = list(parsed_outputs[0].data.keys())
        
        for var_name in var_names:
            # Create array to hold values for all time steps
            data_array = np.full(time_len, np.nan)
            
            # Fill in values from each parsed output
            for i, parsed in enumerate(parsed_outputs):
                if i < len(data_array) and var_name in parsed.data:
                    data_array[i] = parsed.data[var_name]
            
            ds[var_name] = xr.DataArray(
                data_array,
                dims=(time_var,),
                coords={time_var: ds[time_var]},
                attrs={'units': OutputToXarray._get_units(var_name)}
            )
        
        return ds
    
    @staticmethod  
    def _combine_spectral_single_altitude(ds: xr.Dataset, parsed_outputs: List[ParsedOutput], time_var: str) -> xr.Dataset:
        """Combine spectral single altitude data from multiple simulations."""
        time_len = len(ds[time_var])
        wavelengths = ds['wavelength'].values
        
        # Get all variable names from first output
        var_names = list(parsed_outputs[0].data.keys())
        
        for var_name in var_names:
            # Create 2D array [time, wavelength]
            data_array = np.full((time_len, len(wavelengths)), np.nan)
            
            # Fill in values from each parsed output
            for i, parsed in enumerate(parsed_outputs):
                if i < len(data_array) and var_name in parsed.data:
                    wl_dict = parsed.data[var_name]
                    for wl_idx, wavelength in enumerate(wavelengths):
                        if wavelength in wl_dict:
                            data_array[i, wl_idx] = wl_dict[wavelength]
            
            ds[var_name] = xr.DataArray(
                data_array,
                dims=(time_var, 'wavelength'),
                coords={time_var: ds[time_var], 'wavelength': ds['wavelength']},
                attrs={'units': OutputToXarray._get_units(var_name)}
            )
        
        return ds
    
    @staticmethod
    def _combine_integrated_multi_altitude(ds: xr.Dataset, parsed_outputs: List[ParsedOutput], time_var: str) -> xr.Dataset:
        """Combine integrated multi-altitude data from multiple simulations."""
        time_len = len(ds[time_var])
        altitudes = ds['altitude'].values
        
        # Get all variable names from first output  
        var_names = list(parsed_outputs[0].data.keys())
        
        for var_name in var_names:
            # Create 2D array [time, altitude]
            data_array = np.full((time_len, len(altitudes)), np.nan)
            
            # Fill in values from each parsed output
            for i, parsed in enumerate(parsed_outputs):
                if i < len(data_array) and var_name in parsed.data:
                    alt_dict = parsed.data[var_name]
                    for alt_idx, altitude in enumerate(altitudes):
                        if altitude in alt_dict:
                            data_array[i, alt_idx] = alt_dict[altitude]
            
            ds[var_name] = xr.DataArray(
                data_array,
                dims=(time_var, 'altitude'),
                coords={time_var: ds[time_var], 'altitude': ds['altitude']},
                attrs={'units': OutputToXarray._get_units(var_name)}
            )
        
        return ds
    
    @staticmethod
    def _combine_spectral_multi_altitude(ds: xr.Dataset, parsed_outputs: List[ParsedOutput], time_var: str) -> xr.Dataset:
        """Combine spectral multi-altitude data from multiple simulations."""
        time_len = len(ds[time_var])
        wavelengths = ds['wavelength'].values
        altitudes = ds['altitude'].values
        
        # Get all variable names from first output
        var_names = list(parsed_outputs[0].data.keys())
        
        for var_name in var_names:
            # Create 3D array [time, altitude, wavelength]
            data_array = np.full((time_len, len(altitudes), len(wavelengths)), np.nan)
            
            # Fill in values from each parsed output
            for i, parsed in enumerate(parsed_outputs):
                if i < len(data_array) and var_name in parsed.data:
                    alt_dict = parsed.data[var_name]
                    for alt_idx, altitude in enumerate(altitudes):
                        if altitude in alt_dict:
                            wl_dict = alt_dict[altitude]
                            for wl_idx, wavelength in enumerate(wavelengths):
                                if wavelength in wl_dict:
                                    data_array[i, alt_idx, wl_idx] = wl_dict[wavelength]
            
            ds[var_name] = xr.DataArray(
                data_array,
                dims=(time_var, 'altitude', 'wavelength'),
                coords={
                    time_var: ds[time_var], 
                    'altitude': ds['altitude'],
                    'wavelength': ds['wavelength']
                },
                attrs={'units': OutputToXarray._get_units(var_name)}
            )
        
        return ds
    
    @staticmethod
    def _get_units(var_name: str) -> str:
        """Get standard units for common variables."""
        units_map = {
            'lambda': 'nm',
            'eglo': 'W m⁻² nm⁻¹',
            'edir': 'W m⁻² nm⁻¹', 
            'eup': 'W m⁻² nm⁻¹',
            'edn': 'W m⁻² nm⁻¹',
            'enet': 'W m⁻² nm⁻¹',
            'sza': 'degrees',
            'albedo': 'dimensionless',
            'zout': 'km'
        }
        return units_map.get(var_name, '')


class InputGenerator:
    """Clean input file generator for LibRadtran."""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
    
    def generate(self, 
                 dt: datetime,
                 latitude: float, 
                 longitude: float,
                 radiosonde_path: Optional[Path] = None,
                 **overrides) -> str:
        """Generate LibRadtran input file content."""
        
        try:
            lines = []
            sim_defaults = self.config.simulation_defaults
            
            # Core parameters
            lines.append(f"rte_solver {sim_defaults.rte_solver}")
            lines.append(f"mol_abs_param {sim_defaults.mol_abs_param}")
            
            # Atmosphere and data paths
            lines.append(f"atmosphere_file {self.config.paths.atmosphere_profile}")
            lines.append(f"data_files_path {self.config.paths.libradtran_data}")
            
            # Add radiosonde if provided
            if radiosonde_path:
                lines.append(f"radiosonde {radiosonde_path} H2O RH")
            
            # Molecule modifications (with proper None checking)
            mol_modify = getattr(sim_defaults, 'mol_modify', None)
            if mol_modify:
                if isinstance(mol_modify, dict):
                    for molecule, params in mol_modify.items():
                        lines.append(f"mol_modify {molecule} {params['value']} {params['unit']}")
            
            # Solar spectrum
            lines.append(f"source solar {self.config.paths.solar_spectrum} per_nm")
            
            # Time and location
            lines.append(f"time {dt.year} {dt.month} {dt.day} {dt.hour} {dt.minute} {dt.second}")
            
            lat_dir = "N" if latitude >= 0 else "S"
            lines.append(f"latitude {lat_dir} {abs(latitude)}")
            
            lon_dir = "E" if longitude >= 0 else "W" 
            lines.append(f"longitude {lon_dir} {abs(longitude)}")
            
            # Surface properties
            if hasattr(sim_defaults, 'albedo_value') and sim_defaults.albedo_value is not None:
                lines.append(f"albedo {sim_defaults.albedo_value}")
            
            if hasattr(sim_defaults, 'surface_temperature_k') and sim_defaults.surface_temperature_k is not None:
                lines.append(f"sur_temperature {sim_defaults.surface_temperature_k}")
            
            # Wavelength range
            wl_min, wl_max = sim_defaults.wavelength_nm
            lines.append(f"wavelength {wl_min} {wl_max}")
            
            # Output settings
            if hasattr(sim_defaults, 'output_altitudes_km') and sim_defaults.output_altitudes_km:
                zout_str = " ".join(map(str, sim_defaults.output_altitudes_km))
                lines.append(f"zout {zout_str}")
            
            # Output processing
            lines.append("output_process per_nm")
            if getattr(sim_defaults, 'integrate_wavelength', False):
                lines.append("output_process integrate")
            
            # Output columns (must be last)
            if sim_defaults.output_columns:
                output_str = " ".join(sim_defaults.output_columns)
                lines.append(f"output_user {output_str}")
            
            # Apply any overrides
            for key, value in overrides.items():
                if value is not None:
                    lines.append(f"{key} {value}")
            
            return "\n".join(lines)
            
        except Exception as e:
            raise InputGenerationError(f"Failed to generate input content: {e}")


# Legacy function for backward compatibility
def generate_uvspec_input_content(config: SimulationConfig, dt: datetime, 
                                  latitude: float, longitude: float, 
                                  radiosonde_path: Optional[Path] = None, 
                                  **kwargs) -> str:
    """Legacy wrapper for input generation."""
    generator = InputGenerator(config)
    return generator.generate(dt, latitude, longitude, radiosonde_path, **kwargs)


def parse_uvspec_output(output_file: Path, config: SimulationConfig, 
                        input_file: Optional[Path] = None) -> Dict[str, Any]:
    """Legacy wrapper for output parsing."""
    parser = OutputParser(config)
    result = parser.parse(output_file)
    
    # Convert to legacy format for backward compatibility
    legacy_result = {
        '_simulation_type': result.output_type.value,
        '_source_file': str(result.source_file) if result.source_file else None
    }
    
    if result.wavelengths:
        legacy_result['_wavelength_values'] = result.wavelengths
        legacy_result['_num_wavelengths'] = len(result.wavelengths)
    
    if result.altitudes:
        legacy_result['_unique_altitudes'] = result.altitudes
    
    legacy_result.update(result.data)
    return legacy_result


# Data loading functions (keep existing ones)
def load_simulation_input_data(input_file: Union[str, Path]) -> xr.Dataset:
    """Load simulation input data from various file formats."""
    input_path = Path(input_file)
    if not input_path.exists():
        raise ValueError(f"Input file does not exist: {input_path}")
    
    if input_path.suffix.lower() == '.nc':
        logger.info(f"Loading NetCDF input file: {input_path}")
        ds = xr.open_dataset(input_path)
    elif input_path.suffix.lower() == '.csv':
        logger.info(f"Loading CSV input file: {input_path}")
        df = pd.read_csv(input_path)
        
        # Convert datetime columns
        for col in df.columns:
            if 'time' in col.lower() or 'date' in col.lower():
                try:
                    df[col] = pd.to_datetime(df[col])
                except:
                    logger.warning(f"Column '{col}' looks like datetime but couldn't be converted")
        
        ds = df.to_xarray()
    else:
        raise ValueError(f"Unsupported input file format: {input_path.suffix}")
    
    # Validate required coordinates
    required_vars = ['time', 'latitude', 'longitude']
    for var in required_vars:
        alt_names = [var, var.capitalize(), var.upper()]
        if not any(name in ds for name in alt_names):
            raise ValueError(f"Input data missing required variable: {var}")
    
    # Rename variables for consistency
    rename_dict = {}
    for std_name in ['time', 'latitude', 'longitude']:
        for alt in [std_name.capitalize(), std_name.upper()]:
            if alt in ds and std_name not in ds:
                rename_dict[alt] = std_name
    
    if rename_dict:
        ds = ds.rename(rename_dict)
    
    return ds


def save_results_to_netcdf(data: Dict[str, Any], output_path: Union[str, Path], 
                          input_ds: xr.Dataset, config: SimulationConfig,
                          **kwargs) -> Path:
    """Save simulation results to NetCDF file."""
    
    # For now, use the new parser to create a ParsedOutput and convert
    # In a future refactor, this could be simplified
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create a simple dataset for now - this is a placeholder
    # The real implementation would use the new system
    simple_ds = xr.Dataset(
        data_vars={k: (['time'], [v] * len(input_ds.time) if not isinstance(v, (list, dict)) else [np.nan] * len(input_ds.time)) 
                  for k, v in data.items() if not k.startswith('_')},
        coords={'time': input_ds.time}
    )
    
    simple_ds.to_netcdf(output_path)
    return output_path
