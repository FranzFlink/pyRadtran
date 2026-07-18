"""Analytic tests for SRF convolution and inverse Planck."""

import numpy as np
import pytest
import xarray as xr

from pyradtran.channels import brightness_temperature, convolve_channels

H = 6.62607015e-34
C = 2.99792458e8
KB = 1.380649e-23


def planck_radiance_nm(wavelength_nm, T):
    """Planck spectral radiance in W m-2 nm-1 sr-1."""
    lam = wavelength_nm * 1e-9
    B = (2 * H * C**2 / lam**5) / (np.expm1(H * C / (lam * KB * T)))  # W m-3 sr-1
    return B * 1e-9  # per nm


@pytest.fixture
def spectral_result():
    wl = np.linspace(400.0, 700.0, 301)
    rad = np.tile(2.0 + 0.01 * (wl - 400.0), (2, 1))  # linear in wl, 2 time steps
    return xr.Dataset(
        {"uu": (("time", "wavelength"), rad), "sza": (("time",), [30.0, 40.0])},
        coords={"wavelength": wl, "time": [0, 1]},
    )


@pytest.fixture
def boxcar_srf():
    wl = np.linspace(400.0, 700.0, 301)
    phi = np.zeros((2, wl.size))
    phi[0, (wl >= 450) & (wl <= 550)] = 1.0
    phi[1, (wl >= 600) & (wl <= 650)] = 1.0
    return xr.DataArray(
        phi,
        dims=("channel", "wavelength"),
        coords={"channel": ["ch1", "ch2"], "wavelength": wl},
    )


class TestConvolve:
    def test_boxcar_average_of_linear_spectrum(self, spectral_result, boxcar_srf):
        out = convolve_channels(spectral_result, boxcar_srf)
        # Boxcar over linear ramp -> value at band centre
        expected_ch1 = 2.0 + 0.01 * (500.0 - 400.0)  # centre 500 nm
        assert out["uu"].sel(channel="ch1").values == pytest.approx(
            expected_ch1, rel=1e-3
        )

    def test_channel_dim_replaces_wavelength(self, spectral_result, boxcar_srf):
        out = convolve_channels(spectral_result, boxcar_srf)
        assert "channel" in out["uu"].dims
        assert "wavelength" not in out["uu"].dims

    def test_nonspectral_vars_pass_through(self, spectral_result, boxcar_srf):
        out = convolve_channels(spectral_result, boxcar_srf)
        assert "sza" in out
        assert list(out["sza"].values) == [30.0, 40.0]

    def test_keep_spectral(self, spectral_result, boxcar_srf):
        out = convolve_channels(spectral_result, boxcar_srf, keep_spectral=True)
        assert "uu_spectral" in out
        assert "wavelength" in out["uu_spectral"].dims


class TestBrightnessTemperature:
    def test_planck_roundtrip(self):
        T_true = 280.0
        wl_nm = 10500.0  # thermal IR
        L = planck_radiance_nm(wl_nm, T_true)  # W m-2 nm-1 sr-1
        T = brightness_temperature(L, wl_nm, radiance_units="W m-2 nm-1 sr-1")
        assert T == pytest.approx(T_true, abs=0.01)

    def test_uvspec_default_units_mw(self):
        T_true = 280.0
        wl_nm = 10500.0
        L_mw = planck_radiance_nm(wl_nm, T_true) * 1e3  # mW
        T = brightness_temperature(L_mw, wl_nm)
        assert T == pytest.approx(T_true, abs=0.01)

    def test_unknown_units_raise(self):
        with pytest.raises(ValueError, match="radiance_units"):
            brightness_temperature(1.0, 10500.0, radiance_units="furlongs")
