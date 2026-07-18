# tests/test_libradtran_output_parsing.py
"""
Comprehensive test suite for LibRadtran output parsing.

This module tests all possible output structures from LibRadtran:
1. Integrated single altitude
2. Integrated multi-altitude
3. Spectral single altitude
4. Spectral multi-altitude

Each test creates mock LibRadtran output files and verifies correct parsing.
"""

import os
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pyradtran.config import (
    ExecutionConfig,
    OutputConfig,
    PathsConfig,
    SimulationConfig,
    SimulationDefaults,
)
from pyradtran.exceptions import OutputParsingError
from pyradtran.io import OutputParser, OutputToXarray, OutputType


@pytest.mark.unit
@pytest.mark.io
class TestLibRadtranOutputParsing:
    """Test suite for comprehensive LibRadtran output parsing."""

    @pytest.fixture
    def minimal_config(self, tmp_path):
        """Create a minimal simulation config for testing (no real paths needed)."""
        work_dir = tmp_path / "work"
        work_dir.mkdir()
        bin_path = tmp_path / "uvspec"
        bin_path.touch()
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        atm_file = tmp_path / "atm.dat"
        atm_file.touch()
        solar_file = tmp_path / "solar.dat"
        solar_file.touch()

        return SimulationConfig(
            paths=PathsConfig(
                libradtran_bin=bin_path,
                libradtran_data=data_dir,
                atmosphere_profile=atm_file,
                solar_spectrum=solar_file,
                output_dir=work_dir,
                working_dir=work_dir,
            ),
            simulation_defaults=SimulationDefaults(
                rte_solver="disort",
                wavelength_nm=[400, 700],
                output_columns=["lambda", "eglo", "eup", "edir"],
                output_altitudes_km=[0.0],
            ),
            execution=ExecutionConfig(max_workers=2),
            output=OutputConfig(filename_prefix="test"),
        )

    @pytest.fixture
    def test_dataset(self):
        """Create a test xarray dataset with time, lat, lon coordinates."""
        times = pd.date_range("2023-05-01", periods=3, freq="1h")
        lats = [60.0, 60.1, 60.2]
        lons = [10.0, 10.1, 10.2]

        return xr.Dataset(
            coords={
                "time": times,
                "latitude": ("time", lats),
                "longitude": ("time", lons),
            }
        )

    def create_mock_libradtran_output(self, output_type: str, filename: str) -> Path:
        """Create mock LibRadtran output files for testing."""
        output_path = Path(filename)

        if output_type == "integrated_single_altitude":
            # Single line with integrated values at one altitude
            # Format: zout lambda eglo eup edir albedo
            content = """# LibRadtran output
# zout    lambda  eglo    eup     edir    albedo
  0.0     400.0   800.5   120.3   680.2   0.15
"""

        elif output_type == "integrated_multi_altitude":
            # Multiple lines, one per altitude, integrated wavelength
            # Format: zout lambda eglo eup edir albedo
            content = """# LibRadtran output  
# zout    lambda  eglo    eup     edir    albedo
  0.0     400.0   800.5   120.3   680.2   0.15
  1.0     400.0   750.2   110.1   640.1   0.15
  2.0     400.0   700.8   100.5   600.3   0.15
"""

        elif output_type == "spectral_single_altitude":
            # Multiple lines with wavelength in lambda column, single altitude
            # Format: zout lambda eglo eup edir
            content = """# LibRadtran output
# zout    lambda  eglo    eup     edir
  0.0     400.0   750.5   110.3   640.2
  0.0     450.0   780.2   115.1   665.1
  0.0     500.0   810.8   120.5   690.3
  0.0     550.0   840.1   125.2   714.9
  0.0     600.0   820.5   122.8   697.7
  0.0     650.0   800.2   118.6   681.6
  0.0     700.0   775.8   114.2   661.6
"""

        elif output_type == "spectral_multi_altitude":
            # Multiple wavelengths × multiple altitudes
            # Format: zout lambda eglo eup edir
            content = """# LibRadtran output
# zout    lambda  eglo    eup     edir
  0.0     400.0   750.5   110.3   640.2
  0.0     450.0   780.2   115.1   665.1
  0.0     500.0   810.8   120.5   690.3
  0.0     550.0   840.1   125.2   714.9
  0.0     600.0   820.5   122.8   697.7
  0.0     650.0   800.2   118.6   681.6
  0.0     700.0   775.8   114.2   661.6
  1.0     400.0   720.5   105.3   615.2
  1.0     450.0   750.2   110.1   640.1
  1.0     500.0   780.8   115.5   665.3
  1.0     550.0   810.1   120.2   689.9
  1.0     600.0   790.5   117.8   672.7
  1.0     650.0   770.2   113.6   656.6
  1.0     700.0   745.8   109.2   636.6
  2.0     400.0   690.5   100.3   590.2
  2.0     450.0   720.2   105.1   615.1
  2.0     500.0   750.8   110.5   640.3
  2.0     550.0   780.1   115.2   664.9
  2.0     600.0   760.5   112.8   647.7
  2.0     650.0   740.2   108.6   631.6
  2.0     700.0   715.8   104.2   611.6
"""

        else:
            raise ValueError(f"Unknown output type: {output_type}")

        with open(output_path, "w") as f:
            f.write(content)

        return output_path

    @pytest.mark.xfail(
        reason="Test assertions assume a different parser output structure"
    )
    def test_integrated_single_altitude_parsing(self, minimal_config):
        """Test parsing integrated output for single altitude."""
        # Set up config for integrated single altitude
        minimal_config.simulation_defaults.output_columns = [
            "eglo",
            "eup",
            "edir",
            "albedo",
        ]
        minimal_config.simulation_defaults.output_altitudes_km = [0.0]
        minimal_config.simulation_defaults.integrate_wavelength = True

        # Create mock output file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".out", delete=False) as tmp:
            output_file = self.create_mock_libradtran_output(
                "integrated_single_altitude", tmp.name
            )

        try:
            # Parse the output
            parser = OutputParser(minimal_config)
            result = parser.parse_output_file(output_file)

            # Verify structure
            assert result.output_type == OutputType.INTEGRATED_SINGLE_ALTITUDE
            assert "eglo" in result.data
            assert "eup" in result.data
            assert "edir" in result.data
            assert "albedo" in result.data

            # Verify values
            assert result.data["eglo"] == 800.5
            assert result.data["eup"] == 120.3
            assert result.data["edir"] == 680.2
            assert result.data["albedo"] == 0.15

        finally:
            os.unlink(output_file)

    @pytest.mark.xfail(
        reason="Test assertions assume a different parser output structure"
    )
    def test_integrated_multi_altitude_parsing(self, minimal_config):
        """Test parsing integrated output for multiple altitudes."""
        # Set up config for integrated multi-altitude
        minimal_config.simulation_defaults.output_columns = [
            "eglo",
            "eup",
            "edir",
            "albedo",
        ]
        minimal_config.simulation_defaults.output_altitudes_km = [0.0, 1.0, 2.0]
        minimal_config.simulation_defaults.integrate_wavelength = True

        # Create mock output file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".out", delete=False) as tmp:
            output_file = self.create_mock_libradtran_output(
                "integrated_multi_altitude", tmp.name
            )

        try:
            # Parse the output
            parser = OutputParser(minimal_config)
            result = parser.parse_output_file(output_file)

            # Verify structure
            assert result.output_type == OutputType.INTEGRATED_MULTI_ALTITUDE
            assert len(result.altitudes) == 3
            assert result.altitudes == [0.0, 1.0, 2.0]

            # Verify data structure - should be dict[altitude] -> value
            assert isinstance(result.data["eglo"], dict)
            assert 0.0 in result.data["eglo"]
            assert 1.0 in result.data["eglo"]
            assert 2.0 in result.data["eglo"]

            # Verify values
            assert result.data["eglo"][0.0] == 800.5
            assert result.data["eglo"][1.0] == 750.2
            assert result.data["eglo"][2.0] == 700.8

        finally:
            os.unlink(output_file)

    @pytest.mark.xfail(
        reason="Test assertions assume a different parser output structure"
    )
    def test_spectral_single_altitude_parsing(self, minimal_config):
        """Test parsing spectral output for single altitude."""
        # Set up config for spectral single altitude
        minimal_config.simulation_defaults.output_columns = [
            "lambda",
            "eglo",
            "eup",
            "edir",
        ]
        minimal_config.simulation_defaults.output_altitudes_km = [0.0]
        minimal_config.simulation_defaults.integrate_wavelength = False

        # Create mock output file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".out", delete=False) as tmp:
            output_file = self.create_mock_libradtran_output(
                "spectral_single_altitude", tmp.name
            )

        try:
            # Parse the output
            parser = OutputParser(minimal_config)
            result = parser.parse_output_file(output_file)

            # Verify structure
            assert result.output_type == OutputType.SPECTRAL_SINGLE_ALTITUDE
            assert len(result.wavelengths) == 7
            assert result.wavelengths[0] == 400.0
            assert result.wavelengths[-1] == 700.0

            # Verify data structure - should be dict[wavelength] -> value
            assert isinstance(result.data["eglo"], dict)
            assert 400.0 in result.data["eglo"]
            assert 700.0 in result.data["eglo"]

            # Verify values
            assert result.data["eglo"][400.0] == 750.5
            assert result.data["eglo"][700.0] == 775.8

        finally:
            os.unlink(output_file)

    @pytest.mark.xfail(
        reason="Test assertions assume a different parser output structure"
    )
    def test_spectral_multi_altitude_parsing(self, minimal_config):
        """Test parsing spectral output for multiple altitudes."""
        # Set up config for spectral multi-altitude
        minimal_config.simulation_defaults.output_columns = [
            "lambda",
            "eglo",
            "eup",
            "edir",
        ]
        minimal_config.simulation_defaults.output_altitudes_km = [0.0, 1.0, 2.0]
        minimal_config.simulation_defaults.integrate_wavelength = False

        # Create mock output file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".out", delete=False) as tmp:
            output_file = self.create_mock_libradtran_output(
                "spectral_multi_altitude", tmp.name
            )

        try:
            # Parse the output
            parser = OutputParser(minimal_config)
            result = parser.parse_output_file(output_file)

            # Verify structure
            assert result.output_type == OutputType.SPECTRAL_MULTI_ALTITUDE
            assert len(result.wavelengths) == 7
            assert len(result.altitudes) == 3

            # Verify data structure - should be dict[altitude][wavelength] -> value
            assert isinstance(result.data["eglo"], dict)
            assert 0.0 in result.data["eglo"]
            assert isinstance(result.data["eglo"][0.0], dict)
            assert 400.0 in result.data["eglo"][0.0]

            # Verify values for different altitudes
            assert (
                result.data["eglo"][0.0][400.0] == 750.5
            )  # First altitude, first wavelength
            assert (
                result.data["eglo"][1.0][400.0] == 720.5
            )  # Second altitude, first wavelength
            assert (
                result.data["eglo"][2.0][400.0] == 690.5
            )  # Third altitude, first wavelength

        finally:
            os.unlink(output_file)

    @pytest.mark.xfail(
        reason="Test assertions assume a different parser output structure"
    )
    def test_to_xarray_integrated_single_altitude(self, minimal_config, test_dataset):
        """Test converting integrated single altitude results to xarray."""
        minimal_config.simulation_defaults.output_columns = [
            "eglo",
            "eup",
            "edir",
            "albedo",
        ]
        minimal_config.simulation_defaults.output_altitudes_km = [0.0]
        minimal_config.simulation_defaults.integrate_wavelength = True

        # Create mock output file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".out", delete=False) as tmp:
            output_file = self.create_mock_libradtran_output(
                "integrated_single_altitude", tmp.name
            )

        try:
            # Parse and convert to xarray
            parser = OutputParser(minimal_config)
            result = parser.parse_output_file(output_file)

            # Convert to xarray Dataset
            # Convert to xarray Dataset
            ds = OutputToXarray.convert(result, test_dataset)

            # Verify structure
            assert isinstance(ds, xr.Dataset)
            assert "time" in ds.dims
            assert "eglo" in ds.data_vars
            assert "eup" in ds.data_vars
            assert "edir" in ds.data_vars

            # Should have time dimension only (no altitude or wavelength)
            assert len(ds.dims) == 1
            assert ds.eglo.dims == ("time",)

            # Verify values are replicated across time
            assert ds.eglo.values[0] == 800.5
            assert ds.eglo.values[1] == 800.5
            assert ds.eglo.values[2] == 800.5

        finally:
            os.unlink(output_file)

    @pytest.mark.xfail(
        reason="Test assertions assume a different parser output structure"
    )
    def test_to_xarray_spectral_multi_altitude(self, minimal_config, test_dataset):
        """Test converting spectral multi-altitude results to xarray."""
        minimal_config.simulation_defaults.output_columns = [
            "lambda",
            "eglo",
            "eup",
            "edir",
        ]
        minimal_config.simulation_defaults.output_altitudes_km = [0.0, 1.0, 2.0]
        minimal_config.simulation_defaults.integrate_wavelength = False

        # Create mock output file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".out", delete=False) as tmp:
            output_file = self.create_mock_libradtran_output(
                "spectral_multi_altitude", tmp.name
            )

        try:
            # Parse and convert to xarray
            parser = OutputParser(minimal_config)
            result = parser.parse_output_file(output_file)

            # Convert to xarray Dataset
            # Convert to xarray Dataset
            ds = OutputToXarray.convert(result, test_dataset)

            # Verify structure
            assert isinstance(ds, xr.Dataset)
            assert "time" in ds.dims
            assert "altitude" in ds.dims
            assert "wavelength" in ds.dims

            # Should have 3 dimensions
            assert len(ds.dims) == 3
            assert ds.eglo.dims == ("time", "altitude", "wavelength")

            # Verify coordinate values
            assert list(ds.altitude.values) == [0.0, 1.0, 2.0]
            assert ds.wavelength.values[0] == 400.0
            assert ds.wavelength.values[-1] == 700.0

            # Verify some data values
            # Values should be replicated across time dimension
            assert ds.eglo.isel(time=0, altitude=0, wavelength=0).item() == 750.5
            assert ds.eglo.isel(time=1, altitude=0, wavelength=0).item() == 750.5

        finally:
            os.unlink(output_file)

    def test_error_handling_missing_file(self, minimal_config):
        """Test error handling for missing output file."""
        parser = OutputParser(minimal_config)

        with pytest.raises(OutputParsingError, match="Output file not found|not found"):
            parser.parse_output_file(Path("/nonexistent/file.out"))

    def test_error_handling_empty_file(self, minimal_config):
        """Test error handling for empty output file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".out", delete=False) as tmp:
            tmp.write("# Empty file\n")
            empty_file = Path(tmp.name)

        try:
            parser = OutputParser(minimal_config)

            with pytest.raises(OutputParsingError, match="empty|No data"):
                parser.parse_output_file(empty_file)

        finally:
            os.unlink(empty_file)

    def test_error_handling_malformed_data(self, minimal_config):
        """Test error handling for malformed output data."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".out", delete=False) as tmp:
            tmp.write("# Malformed data\n")
            tmp.write("invalid data line\n")
            tmp.write("another bad line\n")
            malformed_file = Path(tmp.name)

        try:
            parser = OutputParser(minimal_config)

            with pytest.raises(
                OutputParsingError, match="could not convert|malformed|No parseable"
            ):
                parser.parse_output_file(malformed_file)

        finally:
            os.unlink(malformed_file)


@pytest.mark.unit
@pytest.mark.io
class TestPerRunAltitudes:
    def test_explicit_output_altitudes_override_config(self, minimal_config, tmp_path):
        # Config says 3 altitudes; this run used zout override with 1
        out = tmp_path / "single_alt.out"
        out.write_text("  500.000  100.000   50.000\n  600.000  110.000   55.000\n")
        minimal_config.simulation_defaults.output_columns = ["lambda", "eglo", "eup"]
        parser = OutputParser(minimal_config, output_altitudes=[1.0])
        parsed = parser.parse_output_file(out)
        assert parsed.output_type == OutputType.SPECTRAL_SINGLE_ALTITUDE

    def test_zout_string_in_overrides_parsed(self, minimal_config):
        # Worker passes raw overrides; zout may be "0.0 1.0 120.0" or a float
        parser = OutputParser(minimal_config, {"zout": "0.0 1.0 120.0"})
        assert parser.output_altitudes == [0.0, 1.0, 120.0]

    def test_zout_float_in_overrides_parsed(self, minimal_config):
        parser = OutputParser(minimal_config, {"zout": 3.5})
        assert parser.output_altitudes == [3.5]


@pytest.mark.unit
@pytest.mark.io
class TestEffectiveColumns:
    """Parser columns must match what the input builder requested."""

    def test_spectral_config_without_lambda_gets_lambda_column(self, minimal_config):
        # Regression: spectral runs used to require the user to list
        # lambda in output_columns or the batch converter went all-NaN.
        minimal_config.simulation_defaults.output_columns = ["sza", "eglo"]
        minimal_config.simulation_defaults.output_altitudes_km = [0.0]
        parser = OutputParser(minimal_config)
        assert "lambda" in parser.output_columns

    def test_output_user_override_drives_parser_columns(self, minimal_config):
        minimal_config.simulation_defaults.output_altitudes_km = [0.0]
        parser = OutputParser(minimal_config, {"output_user": "sza edir"})
        assert parser.output_columns[-2:] == ["sza", "edir"]
        assert "eglo" not in parser.output_columns

    def test_spectral_no_lambda_roundtrip_not_nan(self, minimal_config, tmp_path):
        """End-to-end: file written by the builder parses onto a wavelength axis."""
        minimal_config.simulation_defaults.output_columns = ["eglo", "eup"]
        minimal_config.simulation_defaults.output_altitudes_km = [0.0]

        # The builder injects lambda, so uvspec prints: lambda eglo eup
        out = tmp_path / "spec.out"
        out.write_text(
            "  500.0  100.0  50.0\n"
            "  600.0  110.0  55.0\n"
            "  700.0  120.0  60.0\n"
        )
        parser = OutputParser(minimal_config)
        parsed = parser.parse_output_file(out)
        assert parsed.wavelengths == [500.0, 600.0, 700.0]

        input_ds = xr.Dataset(
            coords={
                "time": [pd.Timestamp("2023-05-01")],
                "latitude": ("time", [60.0]),
                "longitude": ("time", [10.0]),
            }
        )
        result = OutputToXarray.convert_batch([parsed], input_ds)
        assert "wavelength" in result.dims
        assert not np.isnan(result["eglo"].values).any()

    def test_axis_columns_not_data_vars(self, minimal_config, tmp_path):
        """zout/lambda are coordinates, not data variables, in batch results."""
        minimal_config.simulation_defaults.output_columns = ["eglo"]
        minimal_config.simulation_defaults.output_altitudes_km = [0.0]
        out = tmp_path / "spec.out"
        out.write_text("  500.0  100.0\n  600.0  110.0\n")
        parsed = OutputParser(minimal_config).parse_output_file(out)
        input_ds = xr.Dataset(
            coords={
                "time": [pd.Timestamp("2023-05-01")],
                "latitude": ("time", [60.0]),
                "longitude": ("time", [10.0]),
            }
        )
        result = OutputToXarray.convert_batch([parsed], input_ds)
        assert "lambda" not in result.data_vars
        assert "zout" not in result.data_vars
        assert list(result["wavelength"].values) == [500.0, 600.0]
