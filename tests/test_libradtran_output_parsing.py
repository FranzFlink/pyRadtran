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

    #: axes used by the mock outputs (uvspec rows are wavelength-outer,
    #: zout-inner — verified against uvspec 2.0.6)
    WAVELENGTHS = [400.0, 450.0, 500.0, 550.0, 600.0, 650.0, 700.0]
    ALTITUDES = [0.0, 1.0, 2.0]

    @staticmethod
    def _eglo(wl, alt):
        return 750.0 + 0.1 * wl - 30.0 * alt

    @staticmethod
    def _eup(wl, alt):
        return 110.0 + 0.01 * wl - 5.0 * alt

    @staticmethod
    def _edir(wl, alt):
        return 640.0 + 0.05 * wl - 25.0 * alt

    def create_mock_libradtran_output(self, output_type: str, filename: str) -> Path:
        """Write a mock uvspec output file.

        The column layout matches effective_output_columns for the
        corresponding config: integrated runs have no lambda column,
        single-altitude runs no zout column, and multi-altitude spectral
        rows are wavelength-outer / zout-inner (real uvspec ordering).
        """
        output_path = Path(filename)
        rows = []
        if output_type == "integrated_single_altitude":
            # columns: eglo eup edir albedo
            rows.append("  800.5   120.3   680.2   0.15")
        elif output_type == "integrated_multi_altitude":
            # columns: zout eglo eup edir albedo
            for alt in self.ALTITUDES:
                rows.append(
                    f"  {alt:.1f}  {800.5 - 50 * alt:.1f}"
                    f"  {120.3 - 10 * alt:.1f}  {680.2 - 40 * alt:.1f}  0.15"
                )
        elif output_type == "spectral_single_altitude":
            # columns: lambda eglo eup edir
            for wl in self.WAVELENGTHS:
                rows.append(
                    f"  {wl:.1f}  {self._eglo(wl, 0):.2f}"
                    f"  {self._eup(wl, 0):.2f}  {self._edir(wl, 0):.2f}"
                )
        elif output_type == "spectral_multi_altitude":
            # columns: zout lambda eglo eup edir
            for wl in self.WAVELENGTHS:
                for alt in self.ALTITUDES:
                    rows.append(
                        f"  {alt:.1f}  {wl:.1f}  {self._eglo(wl, alt):.2f}"
                        f"  {self._eup(wl, alt):.2f}  {self._edir(wl, alt):.2f}"
                    )
        else:
            raise ValueError(f"Unknown output type: {output_type}")

        output_path.write_text("# mock uvspec output\n" + "\n".join(rows) + "\n")
        return output_path

    def test_integrated_single_altitude_parsing(self, minimal_config):
        """Integrated single-altitude: one row, plain columns."""
        minimal_config.simulation_defaults.output_columns = [
            "eglo",
            "eup",
            "edir",
            "albedo",
        ]
        minimal_config.simulation_defaults.output_altitudes_km = [0.0]
        minimal_config.simulation_defaults.integrate_wavelength = True

        with tempfile.NamedTemporaryFile(mode="w", suffix=".out", delete=False) as tmp:
            output_file = self.create_mock_libradtran_output(
                "integrated_single_altitude", tmp.name
            )
        try:
            parser = OutputParser(minimal_config)
            result = parser.parse_output_file(output_file)

            assert result.output_type == OutputType.INTEGRATED_SINGLE_ALTITUDE
            for col in ("eglo", "eup", "edir", "albedo"):
                assert col in result.data
            assert result.data["eglo"][0] == pytest.approx(800.5)
            assert result.data["albedo"][0] == pytest.approx(0.15)
        finally:
            os.unlink(output_file)

    def test_integrated_multi_altitude_parsing(self, minimal_config):
        """Integrated multi-altitude: zout column injected, one row per level."""
        minimal_config.simulation_defaults.output_columns = [
            "eglo",
            "eup",
            "edir",
            "albedo",
        ]
        minimal_config.simulation_defaults.output_altitudes_km = [0.0, 1.0, 2.0]
        minimal_config.simulation_defaults.integrate_wavelength = True

        with tempfile.NamedTemporaryFile(mode="w", suffix=".out", delete=False) as tmp:
            output_file = self.create_mock_libradtran_output(
                "integrated_multi_altitude", tmp.name
            )
        try:
            parser = OutputParser(minimal_config)
            result = parser.parse_output_file(output_file)

            assert result.output_type == OutputType.INTEGRATED_MULTI_ALTITUDE
            assert result.altitudes == [0.0, 1.0, 2.0]
            np.testing.assert_allclose(result.data["eglo"], [800.5, 750.5, 700.5])
        finally:
            os.unlink(output_file)

    def test_spectral_single_altitude_parsing(self, minimal_config):
        """Spectral single-altitude: lambda column, one row per wavelength."""
        minimal_config.simulation_defaults.output_columns = [
            "lambda",
            "eglo",
            "eup",
            "edir",
        ]
        minimal_config.simulation_defaults.output_altitudes_km = [0.0]
        minimal_config.simulation_defaults.integrate_wavelength = False

        with tempfile.NamedTemporaryFile(mode="w", suffix=".out", delete=False) as tmp:
            output_file = self.create_mock_libradtran_output(
                "spectral_single_altitude", tmp.name
            )
        try:
            parser = OutputParser(minimal_config)
            result = parser.parse_output_file(output_file)

            assert result.output_type == OutputType.SPECTRAL_SINGLE_ALTITUDE
            assert result.wavelengths == self.WAVELENGTHS
            np.testing.assert_allclose(
                result.data["eglo"],
                [self._eglo(wl, 0) for wl in self.WAVELENGTHS],
            )
        finally:
            os.unlink(output_file)

    def test_spectral_multi_altitude_parsing(self, minimal_config):
        """Spectral multi-altitude: flat data in wavelength-outer file order."""
        minimal_config.simulation_defaults.output_columns = [
            "lambda",
            "eglo",
            "eup",
            "edir",
        ]
        minimal_config.simulation_defaults.output_altitudes_km = [0.0, 1.0, 2.0]
        minimal_config.simulation_defaults.integrate_wavelength = False

        with tempfile.NamedTemporaryFile(mode="w", suffix=".out", delete=False) as tmp:
            output_file = self.create_mock_libradtran_output(
                "spectral_multi_altitude", tmp.name
            )
        try:
            parser = OutputParser(minimal_config)
            result = parser.parse_output_file(output_file)

            assert result.output_type == OutputType.SPECTRAL_MULTI_ALTITUDE
            assert result.wavelengths == self.WAVELENGTHS
            assert result.altitudes == self.ALTITUDES
            expected = [
                self._eglo(wl, alt) for wl in self.WAVELENGTHS for alt in self.ALTITUDES
            ]
            np.testing.assert_allclose(result.data["eglo"], expected)
        finally:
            os.unlink(output_file)

    def test_to_xarray_integrated_single_altitude(self, minimal_config, test_dataset):
        """convert() maps an integrated single-altitude output onto time."""
        minimal_config.simulation_defaults.output_columns = [
            "eglo",
            "eup",
            "edir",
            "albedo",
        ]
        minimal_config.simulation_defaults.output_altitudes_km = [0.0]
        minimal_config.simulation_defaults.integrate_wavelength = True

        with tempfile.NamedTemporaryFile(mode="w", suffix=".out", delete=False) as tmp:
            output_file = self.create_mock_libradtran_output(
                "integrated_single_altitude", tmp.name
            )
        try:
            parser = OutputParser(minimal_config)
            result = parser.parse_output_file(output_file)
            ds = OutputToXarray.convert(result, test_dataset.isel(time=[0]))

            assert isinstance(ds, xr.Dataset)
            assert ds.eglo.dims == ("time",)
            assert ds.eglo.values[0] == pytest.approx(800.5)
        finally:
            os.unlink(output_file)

    def test_to_xarray_spectral_multi_altitude(self, minimal_config, test_dataset):
        """convert() reshapes wavelength-outer rows onto (wavelength, altitude)."""
        minimal_config.simulation_defaults.output_columns = [
            "lambda",
            "eglo",
            "eup",
            "edir",
        ]
        minimal_config.simulation_defaults.output_altitudes_km = [0.0, 1.0, 2.0]
        minimal_config.simulation_defaults.integrate_wavelength = False

        with tempfile.NamedTemporaryFile(mode="w", suffix=".out", delete=False) as tmp:
            output_file = self.create_mock_libradtran_output(
                "spectral_multi_altitude", tmp.name
            )
        try:
            parser = OutputParser(minimal_config)
            result = parser.parse_output_file(output_file)
            ds = OutputToXarray.convert(result, test_dataset.isel(time=[0]))

            assert ds.eglo.dims == ("time", "wavelength", "altitude")
            assert list(ds.altitude.values) == self.ALTITUDES
            assert list(ds.wavelength.values) == self.WAVELENGTHS
            assert ds.eglo.sel(wavelength=550.0, altitude=1.0).item() == pytest.approx(
                self._eglo(550.0, 1.0)
            )
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
            "  500.0  100.0  50.0\n" "  600.0  110.0  55.0\n" "  700.0  120.0  60.0\n"
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
