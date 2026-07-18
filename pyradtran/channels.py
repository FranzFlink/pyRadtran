"""
Instrument-channel convolution and brightness temperature.

Convolve spectral pyRadtran results with instrument spectral response
functions (SRFs) and convert thermal radiances to brightness temperatures.

Examples
--------
>>> channel_ds = convolve_channels(result_ds, srf)
>>> tb = brightness_temperature(channel_ds["uu"], 10500.0)

See Also
--------
pyradtran.interface.PyRadtranAccessor.run : ``channels=`` integration.
"""

import logging

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)

# CODATA 2018
_H = 6.62607015e-34   # J s
_C = 2.99792458e8     # m s-1
_KB = 1.380649e-23    # J K-1

#: radiance unit -> factor converting to W m-2 m-1 sr-1 (SI, per metre)
_UNIT_TO_SI = {
    "mW m-2 nm-1 sr-1": 1e-3 * 1e9,
    "W m-2 nm-1 sr-1": 1e9,
}


def convolve_channels(
    result: xr.Dataset,
    srf: xr.DataArray,
    keep_spectral: bool = False,
) -> xr.Dataset:
    """SRF-average all spectral variables onto a ``channel`` dimension.

    Parameters
    ----------
    result : xarray.Dataset
        Spectral simulation result with a ``wavelength`` dimension (nm).
    srf : xarray.DataArray
        Spectral response, dims ``(channel, wavelength)``. Interpolated
        onto the result's wavelength grid; values outside the SRF grid
        are treated as zero response.
    keep_spectral : bool, default ``False``
        Keep the original spectral variables under ``<name>_spectral``.

    Returns
    -------
    xarray.Dataset
        Channel-space dataset; non-spectral variables pass through.
    """
    if "wavelength" not in result.dims:
        raise ValueError("result has no 'wavelength' dimension to convolve")
    wl = result["wavelength"]
    phi = _interp_srf(srf, wl)
    norm = phi.integrate("wavelength")

    out = xr.Dataset(attrs=dict(result.attrs))
    for name, da in result.data_vars.items():
        if "wavelength" in da.dims:
            conv = (da * phi).integrate("wavelength") / norm
            out[name] = conv
            if keep_spectral:
                out[f"{name}_spectral"] = da
        else:
            out[name] = da
    out["channel"] = srf["channel"]
    return out


def _interp_srf(srf: xr.DataArray, wl: xr.DataArray) -> xr.DataArray:
    """Interpolate an SRF onto the result wavelength grid (numpy only).

    Zero response outside the SRF's own wavelength grid.
    """
    srf = srf.sortby("wavelength")
    srf_wl = srf["wavelength"].values.astype(float)
    target = wl.values.astype(float)
    phi_rows = [
        np.interp(target, srf_wl, srf.isel(channel=i).values.astype(float),
                  left=0.0, right=0.0)
        for i in range(srf.sizes["channel"])
    ]
    return xr.DataArray(
        np.stack(phi_rows),
        dims=("channel", "wavelength"),
        coords={"channel": srf["channel"].values, "wavelength": target},
    )


def brightness_temperature(
    radiance,
    wavelength_nm,
    radiance_units: str = "mW m-2 nm-1 sr-1",
):
    """Convert monochromatic radiance to brightness temperature (K).

    Inverse Planck: ``T = c2 / (lambda * ln(1 + c1 / (lambda^5 * L)))``
    with ``c1 = 2 h c^2`` and ``c2 = h c / k``.

    Parameters
    ----------
    radiance : array-like or xarray.DataArray
        Spectral radiance.
    wavelength_nm : float or array-like
        Wavelength in nanometres.
    radiance_units : str, default ``"mW m-2 nm-1 sr-1"``
        Units of *radiance* (uvspec's default radiance unit).

    Returns
    -------
    Brightness temperature in K, same shape as *radiance*.
    """
    if radiance_units not in _UNIT_TO_SI:
        raise ValueError(
            f"Unsupported radiance_units '{radiance_units}'; "
            f"expected one of {sorted(_UNIT_TO_SI)}"
        )
    L_si = np.asarray(radiance, dtype=float) * _UNIT_TO_SI[radiance_units]
    lam = np.asarray(wavelength_nm, dtype=float) * 1e-9
    c1 = 2 * _H * _C**2
    c2 = _H * _C / _KB
    return c2 / (lam * np.log1p(c1 / (lam**5 * L_si)))


__all__ = ["convolve_channels", "brightness_temperature"]
