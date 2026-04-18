# tests/test_clouds.py
"""
Tests for pyradtran.clouds — CloudLayer, CloudGenerator, CloudFileWriter,
and the generate_cloud_file_from_era5 convenience function.
"""

import pytest
import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path

from pyradtran.clouds import (
    CloudLayer,
    CloudGenerator,
    CloudFileWriter,
    generate_cloud_file_from_era5,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_era5_ds(n_levels: int = 10) -> xr.Dataset:
    """Return a minimal synthetic ERA5-style dataset on time + pressure levels."""
    pressure_hpa = np.linspace(1000, 200, n_levels)
    # Geopotential: rough linear increase from ~0 to ~50 000 m²/s²
    geopotential = np.linspace(0, 50_000, n_levels)
    temperature = np.linspace(288, 220, n_levels)  # K
    # Add synthetic cloud content in the mid-troposphere
    clwc = np.zeros(n_levels)
    ciwc = np.zeros(n_levels)
    cc = np.zeros(n_levels)
    clwc[3:6] = 1e-4   # kg/kg liquid
    ciwc[6:8] = 5e-5   # kg/kg ice
    cc[3:8] = 0.5

    # from_era5_dataset calls isel(time=0) then sel(latitude=..., longitude=...)
    # so we need time, latitude, longitude as dimensions
    lats = [78.0]
    lons = [15.0]
    times = [pd.Timestamp("2022-07-01T12:00")]
    shape = (1, 1, 1, n_levels)  # time, lat, lon, pressure_level
    return xr.Dataset(
        {
            "clwc": (["time", "latitude", "longitude", "pressure_level"],
                     clwc.reshape(1, 1, 1, n_levels), {"units": "kg kg-1"}),
            "ciwc": (["time", "latitude", "longitude", "pressure_level"],
                     ciwc.reshape(1, 1, 1, n_levels), {"units": "kg kg-1"}),
            "cc":   (["time", "latitude", "longitude", "pressure_level"],
                     cc.reshape(1, 1, 1, n_levels), {"units": "1"}),
            "t":    (["time", "latitude", "longitude", "pressure_level"],
                     temperature.reshape(1, 1, 1, n_levels), {"units": "K"}),
            "z":    (["time", "latitude", "longitude", "pressure_level"],
                     geopotential.reshape(1, 1, 1, n_levels), {"units": "m2 s-2"}),
        },
        coords={
            "pressure_level": (["pressure_level"], pressure_hpa, {"units": "hPa"}),
            "time": times,
            "latitude": lats,
            "longitude": lons,
        },
    )


# ---------------------------------------------------------------------------
# CloudLayer tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestCloudLayer:
    def test_basic_creation(self):
        layer = CloudLayer(z_bottom_km=1.0, z_top_km=2.0, lwc_g_m3=0.3)
        assert layer.z_bottom_km == 1.0
        assert layer.r_eff_um == 10.0  # default

    def test_default_values(self):
        layer = CloudLayer(z_bottom_km=0.5, z_top_km=1.5)
        assert layer.lwc_g_m3 == 0.0
        assert layer.iwc_g_m3 == 0.0
        assert layer.cloud_fraction == 1.0

    def test_invalid_z_order_raises(self):
        with pytest.raises(ValueError, match="less than top"):
            CloudLayer(z_bottom_km=2.0, z_top_km=1.0)

    def test_equal_z_raises(self):
        with pytest.raises(ValueError):
            CloudLayer(z_bottom_km=1.0, z_top_km=1.0)

    def test_negative_lwc_raises(self):
        with pytest.raises(ValueError, match="negative"):
            CloudLayer(z_bottom_km=1.0, z_top_km=2.0, lwc_g_m3=-0.1)

    def test_negative_iwc_raises(self):
        with pytest.raises(ValueError, match="negative"):
            CloudLayer(z_bottom_km=1.0, z_top_km=2.0, iwc_g_m3=-0.1)

    def test_zero_reff_raises(self):
        with pytest.raises(ValueError, match="positive"):
            CloudLayer(z_bottom_km=1.0, z_top_km=2.0, r_eff_um=0.0)

    def test_negative_reff_raises(self):
        with pytest.raises(ValueError):
            CloudLayer(z_bottom_km=1.0, z_top_km=2.0, r_eff_um=-5.0)

    def test_invalid_cloud_fraction_raises(self):
        with pytest.raises(ValueError, match="fraction"):
            CloudLayer(z_bottom_km=1.0, z_top_km=2.0, cloud_fraction=1.5)


# ---------------------------------------------------------------------------
# CloudGenerator.from_simple_parameters tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestCloudGeneratorSimple:
    def test_returns_list_of_cloud_layers(self):
        layers = CloudGenerator.from_simple_parameters(
            z_base_km=1.0, z_top_km=2.0, lwc_g_m3=0.3, r_eff_um=10.0, n_layers=5
        )
        assert isinstance(layers, list)
        assert all(isinstance(l, CloudLayer) for l in layers)

    def test_n_layers_respected(self):
        for n in [1, 5, 10]:
            layers = CloudGenerator.from_simple_parameters(1.0, 2.0, 0.3, 10.0, n_layers=n)
            assert len(layers) == n

    def test_altitude_coverage(self):
        layers = CloudGenerator.from_simple_parameters(1.0, 3.0, 0.2, 8.0, n_layers=4)
        assert layers[0].z_bottom_km == pytest.approx(1.0, abs=0.01)
        assert layers[-1].z_top_km == pytest.approx(3.0, abs=0.01)

    def test_no_gaps_between_layers(self):
        layers = CloudGenerator.from_simple_parameters(1.0, 3.0, 0.2, 8.0, n_layers=4)
        for i in range(len(layers) - 1):
            assert layers[i].z_top_km == pytest.approx(layers[i + 1].z_bottom_km, abs=1e-6)

    def test_lwc_and_reff_set(self):
        layers = CloudGenerator.from_simple_parameters(
            1.0, 2.0, lwc_g_m3=0.5, r_eff_um=15.0, n_layers=3
        )
        for layer in layers:
            assert layer.lwc_g_m3 == pytest.approx(0.5)
            assert layer.r_eff_um == pytest.approx(15.0)

    def test_invalid_altitudes_raise(self):
        with pytest.raises((ValueError, Exception)):
            CloudGenerator.from_simple_parameters(2.0, 1.0, 0.3, 10.0, n_layers=5)


# ---------------------------------------------------------------------------
# CloudGenerator.from_era5_dataset tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestCloudGeneratorERA5:
    def test_returns_list(self):
        ds = _make_era5_ds()
        layers = CloudGenerator.from_era5_dataset(ds, lat=78.0, lon=15.0)
        assert isinstance(layers, list)

    def test_layers_are_cloud_layer_instances(self):
        ds = _make_era5_ds()
        layers = CloudGenerator.from_era5_dataset(ds, lat=78.0, lon=15.0)
        assert all(isinstance(l, CloudLayer) for l in layers)

    def test_empty_cloud_ds_returns_empty(self):
        ds = _make_era5_ds()
        ds["clwc"].values[:] = 0.0
        ds["ciwc"].values[:] = 0.0
        layers = CloudGenerator.from_era5_dataset(ds, lat=78.0, lon=15.0)
        assert len(layers) == 0

    def test_layers_sorted_bottom_to_top(self):
        ds = _make_era5_ds()
        layers = CloudGenerator.from_era5_dataset(ds, lat=78.0, lon=15.0)
        if len(layers) > 1:
            for i in range(len(layers) - 1):
                assert layers[i].z_bottom_km <= layers[i + 1].z_bottom_km

    def test_custom_reff(self):
        ds = _make_era5_ds()
        layers = CloudGenerator.from_era5_dataset(ds, lat=78.0, lon=15.0, default_r_eff_water=8.0)
        liquid_layers = [l for l in layers if l.lwc_g_m3 > 0]
        if liquid_layers:
            assert liquid_layers[0].r_eff_um == pytest.approx(8.0)


# ---------------------------------------------------------------------------
# CloudFileWriter tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.io
class TestCloudFileWriter:
    _layers = [
        CloudLayer(z_bottom_km=1.0, z_top_km=1.5, lwc_g_m3=0.3, r_eff_um=10.0),
        CloudLayer(z_bottom_km=1.5, z_top_km=2.0, lwc_g_m3=0.3, r_eff_um=10.0),
    ]

    def test_write_water_cloud_file_creates_file(self, tmp_path):
        out = tmp_path / "wc.dat"
        CloudFileWriter.write_water_cloud_file(self._layers, out)
        assert out.exists()

    def test_write_water_cloud_file_has_data_rows(self, tmp_path):
        out = tmp_path / "wc.dat"
        CloudFileWriter.write_water_cloud_file(self._layers, out)
        data_lines = [l for l in out.read_text().splitlines() if l.strip() and not l.startswith("#")]
        assert len(data_lines) > 0

    def test_write_water_cloud_file_three_columns(self, tmp_path):
        out = tmp_path / "wc.dat"
        CloudFileWriter.write_water_cloud_file(self._layers, out)
        data_lines = [l for l in out.read_text().splitlines() if l.strip() and not l.startswith("#")]
        for line in data_lines:
            cols = line.split()
            assert len(cols) == 3, f"Expected 3 columns, got: {line}"

    def test_write_ice_cloud_file_creates_file(self, tmp_path):
        ice_layers = [
            CloudLayer(z_bottom_km=5.0, z_top_km=6.0, iwc_g_m3=0.05, r_eff_um=30.0),
        ]
        out = tmp_path / "ic.dat"
        CloudFileWriter.write_ice_cloud_file(ice_layers, out)
        assert out.exists()

    def test_write_creates_parent_directory(self, tmp_path):
        out = tmp_path / "subdir" / "wc.dat"
        CloudFileWriter.write_water_cloud_file(self._layers, out)
        assert out.exists()

    def test_altitudes_are_numeric(self, tmp_path):
        out = tmp_path / "wc.dat"
        CloudFileWriter.write_water_cloud_file(self._layers, out)
        data_lines = [l for l in out.read_text().splitlines() if l.strip() and not l.startswith("#")]
        for line in data_lines:
            cols = line.split()
            assert float(cols[0]) > 0  # altitude must be positive


# ---------------------------------------------------------------------------
# generate_cloud_file_from_era5 tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.io
class TestGenerateCloudFileFromERA5:
    def test_creates_output_file(self, tmp_path):
        ds = _make_era5_ds()
        out = tmp_path / "cloud_era5.dat"
        result = generate_cloud_file_from_era5(ds, output_path=out, lat=78.0, lon=15.0)
        assert Path(out).exists()

    def test_empty_cloud_no_crash(self, tmp_path):
        ds = _make_era5_ds()
        ds["clwc"].values[:] = 0.0
        ds["ciwc"].values[:] = 0.0
        out = tmp_path / "empty.dat"
        # Should not raise even with zero cloud content
        try:
            generate_cloud_file_from_era5(ds, output_path=out, lat=78.0, lon=15.0)
        except Exception as e:
            pytest.fail(f"generate_cloud_file_from_era5 raised unexpectedly: {e}")

    def test_returns_path_or_none(self, tmp_path):
        ds = _make_era5_ds()
        out = tmp_path / "result.dat"
        result = generate_cloud_file_from_era5(ds, output_path=out, lat=78.0, lon=15.0)
        assert result is None or isinstance(result, (str, Path))
