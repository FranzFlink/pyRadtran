"""Regression tests for the clouds.py bug fixes (exact writer, ERA5 delegation)."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from helpers import has_libradtran

from pyradtran.clouds import (
    CloudFileWriter,
    CloudGenerator,
    CloudLayer,
    generate_cloud_file_from_era5,
)
from pyradtran.era5 import cloud_profiles, normalize_era5, select_profile


def _rows(path):
    return [
        [float(v) for v in line.split()]
        for line in path.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]


def _era5(time_dim="time", lev_dim="pressure_level", names=None):
    """Column with liquid at 850/925 hPa and ice at 500 hPa (2x2 grid, 2 times)."""
    names = names or {}
    lev = np.array([500.0, 700.0, 850.0, 925.0, 1000.0])
    cols = {
        "t": [252.0, 268.0, 276.0, 280.0, 284.0],
        "z": [54000.0, 29500.0, 14500.0, 7500.0, 1000.0],
        "clwc": [0.0, 0.0, 2.0e-4, 3.0e-4, 0.0],
        "ciwc": [2.0e-5, 0.0, 0.0, 0.0, 0.0],
    }
    dims = (time_dim, lev_dim, "latitude", "longitude")
    ds = xr.Dataset(
        {
            names.get(k, k): (
                dims,
                np.broadcast_to(np.array(v)[None, :, None, None], (2, 5, 2, 2)).copy(),
            )
            for k, v in cols.items()
        },
        coords={
            time_dim: pd.to_datetime(["2024-07-15T12", "2024-07-15T18"]),
            lev_dim: lev,
            "latitude": [78.0, 77.0],
            "longitude": [15.0, 16.0],
        },
    )
    ds[names.get("z", "z")].attrs["units"] = "m2 s-2"
    return ds


def _lwp(layers):
    return sum(l.lwc_g_m3 * (l.z_top_km - l.z_bottom_km) * 1000 for l in layers)


@pytest.mark.unit
class TestWriter:
    def test_single_layer_exact_rows(self, tmp_path):
        path = CloudFileWriter.write_water_cloud_file(
            [CloudLayer(1.0, 1.5, lwc_g_m3=0.3, r_eff_um=8.0)], tmp_path / "wc.dat"
        )
        assert _rows(path) == [[1.5, 0.0, 8.0], [1.0, 0.3, 8.0]]

    def test_contiguous_and_separate_layers(self, tmp_path):
        layers = [
            CloudLayer(2.5, 3.0, lwc_g_m3=0.1, r_eff_um=6.0),
            CloudLayer(1.0, 1.5, lwc_g_m3=0.2),
            CloudLayer(1.5, 2.0, lwc_g_m3=0.4),  # unsorted input, touches the 1-1.5
        ]
        path = CloudFileWriter.write_water_cloud_file(layers, tmp_path / "wc.dat")
        assert _rows(path) == [
            [3.0, 0.0, 6.0],
            [2.5, 0.1, 6.0],
            [2.0, 0.0, 10.0],
            [1.5, 0.4, 10.0],
            [1.0, 0.2, 10.0],
        ]

    def test_sublayers_with_float_rounding_do_not_overlap(self, tmp_path):
        one = CloudGenerator.from_simple_parameters(1.0, 1.7, 0.3, n_layers=1)
        many = CloudGenerator.from_simple_parameters(1.0, 1.7, 0.3, n_layers=7)
        a = _rows(CloudFileWriter.write_water_cloud_file(one, tmp_path / "a.dat"))
        b = _rows(CloudFileWriter.write_water_cloud_file(many, tmp_path / "b.dat"))
        assert len(b) == 8 and b[0] == a[0] and b[-1] == a[-1]
        assert all(row[1] == 0.3 for row in b[1:])

    def test_thin_layer_and_small_content_survive(self, tmp_path):
        path = CloudFileWriter.write_ice_cloud_file(
            [CloudLayer(8.0, 8.02, iwc_g_m3=2e-4, r_eff_um=30.0)],
            str(tmp_path / "sub" / "ic.dat"),  # str path, missing parent
        )
        assert _rows(path) == [[8.02, 0.0, 30.0], [8.0, 2e-4, 30.0]]

    def test_overlap_raises(self, tmp_path):
        layers = [
            CloudLayer(1.0, 2.0, lwc_g_m3=0.1),
            CloudLayer(1.5, 2.5, lwc_g_m3=0.1),
        ]
        with pytest.raises(ValueError, match="overlap"):
            CloudFileWriter.write_water_cloud_file(layers, tmp_path / "wc.dat")
        # a liquid and an ice layer may share the same altitudes
        mixed = [CloudLayer(1.0, 2.0, lwc_g_m3=0.1), CloudLayer(1.5, 2.5, iwc_g_m3=0.1)]
        CloudFileWriter.write_water_cloud_file(mixed, tmp_path / "wc.dat")

    def test_legacy_grid_arguments_accepted(self, tmp_path):
        layer = [CloudLayer(1.0, 1.5, lwc_g_m3=0.3)]
        a = CloudFileWriter.write_water_cloud_file(layer, tmp_path / "a.dat")
        b = CloudFileWriter.write_water_cloud_file(
            layer, tmp_path / "b.dat", include_zero_layers=False,
            altitude_resolution_km=0.25,
        )  # fmt: skip
        assert a.read_text() == b.read_text()


@pytest.mark.unit
class TestFromERA5:
    def test_matches_era5_cloud_profiles(self):
        ds = _era5()
        layers = CloudGenerator.from_era5_dataset(ds, lat=78.0, lon=15.0)
        wc, ic = cloud_profiles(
            select_profile(normalize_era5(ds), 78.0, 15.0, "2024-07-15T12")
        )
        liquid = [l for l in layers if l.lwc_g_m3 > 0]
        ice = [l for l in layers if l.iwc_g_m3 > 0]
        # layers tile the column between level midpoints: full water path
        lwp = sum(
            c * (zt - zb) * 1000
            for c, zb, zt in zip(wc["lwc"][1:], wc["z"][1:], wc["z"][:-1])
        )
        assert _lwp(liquid) == pytest.approx(lwp)
        assert lwp > 300  # the old 50 m slabs gave ~28 g m-2
        assert [(l.z_bottom_km, l.z_top_km) for l in liquid] == [
            (wc["z"][2], wc["z"][1]),
            (wc["z"][1], wc["z"][0]),
        ]
        assert len(ice) == 1 and ice[0].r_eff_um == 20.0
        assert ice[0].iwc_g_m3 == pytest.approx(ic["iwc"][1])

    def test_other_naming_flavours(self):
        ref = CloudGenerator.from_era5_dataset(_era5(), lat=78.0, lon=15.0)
        arco = _era5(
            time_dim="valid_time",
            lev_dim="level",
            names={
                "t": "temperature",
                "z": "geopotential",
                "clwc": "specific_cloud_liquid_water_content",
                "ciwc": "specific_cloud_ice_water_content",
            },
        )
        assert CloudGenerator.from_era5_dataset(arco, lat=78.0, lon=15.0) == ref
        custom = _era5(lev_dim="p", names={"t": "T", "clwc": "LWC", "z": "PHI"})
        got = CloudGenerator.from_era5_dataset(
            custom,
            lat=78.0,
            lon=15.0,
            pressure_levels="p",
            cloud_variables={"lwc": "LWC", "temp": "T"},
            geopotential_var="PHI",
        )
        assert got == [l for l in ref if l.lwc_g_m3 > 0]  # no "iwc" key: ice off

    def test_missing_position_or_temperature_raises(self):
        with pytest.raises(ValueError, match="latitude and longitude"):
            CloudGenerator.from_era5_dataset(_era5())
        with pytest.raises(ValueError, match="temperature"):
            CloudGenerator.from_era5_dataset(_era5().drop_vars("t"), lat=78, lon=15)

    def test_thresholds_and_altitude_override(self):
        ds = _era5()
        only_liquid = CloudGenerator.from_era5_dataset(
            ds, lat=78.0, lon=15.0, iwc_threshold=0.05
        )
        assert all(l.iwc_g_m3 == 0 for l in only_liquid) and only_liquid
        alt = np.array([5.5, 3.0, 1.5, 0.75, 0.1])
        layers = CloudGenerator.from_era5_dataset(
            ds, lat=78.0, lon=15.0, altitude_levels_km=alt, iwc_threshold=0.05
        )
        assert [(l.z_bottom_km, l.z_top_km) for l in layers] == [
            (0.425, 1.125),
            (1.125, 2.25),
        ]


@pytest.mark.unit
class TestGenerateFile:
    def test_file_reproduces_layers(self, tmp_path):
        ds = _era5()
        path = generate_cloud_file_from_era5(
            ds, str(tmp_path / "wc.dat"), lat=78.0, lon=15.0, default_r_eff_water=8.0
        )
        wc, _ = cloud_profiles(
            select_profile(normalize_era5(ds), 78.0, 15.0, "2024-07-15T12"),
            reff_water_um=8.0,
        )
        assert np.allclose(
            _rows(path), np.column_stack([wc["z"], wc["lwc"], wc["reff"]]), atol=1e-6
        )

    def test_cloud_type_validated_first(self, tmp_path):
        with pytest.raises(ValueError, match="cloud type"):
            generate_cloud_file_from_era5(None, tmp_path / "x.dat", cloud_type="xx")


@pytest.mark.integration
@pytest.mark.skipif(not has_libradtran(), reason="LibRadtran not available")
def test_written_file_equals_dict_profile(tmp_path):
    """uvspec sees the same cloud from the written file as from the exact dict."""
    import pyradtran  # noqa: F401  (registers the accessor)

    w, r = 0.3, 8.0
    ds = xr.Dataset(
        coords={
            "time": pd.to_datetime(["2022-06-21T12:00"]),
            "latitude": ("time", [50.0]),
            "longitude": ("time", [10.0]),
        }
    )

    def run(wc_file):
        result = ds.pyradtran.run(
            params={
                "wc_file": wc_file,
                "wavelength": "500 500",
                "rte_solver": "disort",
                "mol_abs_param": "lowtran",
                "output_user": "lambda edir edn eup",
            },
            save_to_file=False,
            show_progress=False,
        )
        return [float(result[k].values.ravel()[0]) for k in ("edir", "edn", "eup")]

    path = CloudFileWriter.write_water_cloud_file(
        [CloudLayer(1.0, 1.5, lwc_g_m3=w, r_eff_um=r)], tmp_path / "wc.dat"
    )
    from_file = run(f"1D {path}")
    from_dict = run({"z": [1.5, 1.0], "lwc": [w, w], "reff": [r, r]})
    assert from_file == pytest.approx(from_dict, rel=1e-6)
    assert from_file[1] < 1340  # sanity: a 500 m cloud, not the old 400 m one
