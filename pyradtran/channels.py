"""
Instrument-channel convolution and brightness temperature.

Turn spectral ``uvspec`` output into what an instrument reports:
band averages weighted by the spectral response function (SRF) of each
channel and, for thermal channels, brightness temperatures.

.. autosummary::

   convolve_channels
   brightness_temperature

Examples
--------
Post-process a spectral result (``result_ds`` has a ``wavelength``
dimension, ``srf`` has dims ``(channel, wavelength)``):

>>> channel_ds = convolve_channels(result_ds, srf)  # doctest: +SKIP
>>> tb = brightness_temperature(  # doctest: +SKIP
...     channel_ds["uu"], 10500.0, radiance_units="W m-2 nm-1 sr-1"
... )

Or let :meth:`~pyradtran.interface.PyRadtranAccessor.run` convolve:

>>> channel_ds = ds.pyradtran.run(config_path=cfg, channels=srf)  # doctest: +SKIP

See Also
--------
pyradtran.interface.PyRadtranAccessor.run : ``channels=`` integration.
"""

import numpy as np
import xarray as xr

# CODATA 2018
_H = 6.62607015e-34  # J s
_C = 2.99792458e8  # m s-1
_KB = 1.380649e-23  # J K-1

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
    r"""SRF-average all spectral variables onto a ``channel`` dimension.

    Every variable with a ``wavelength`` dimension is replaced by its
    response-weighted band average, one value per instrument channel.

    Parameters
    ----------
    result : xarray.Dataset
        Spectral simulation result with a ``wavelength`` dimension (nm),
        e.g. the return value of
        :meth:`~pyradtran.interface.PyRadtranAccessor.run`.
    srf : xarray.DataArray
        Spectral response, dims ``(channel, wavelength)`` with
        ``wavelength`` in nm. Interpolated linearly onto the result's
        wavelength grid, with zero response outside the SRF's own grid.
        Need not be normalised.
    keep_spectral : bool, default ``False``
        Keep the original spectral variables under ``<name>_spectral``.

    Returns
    -------
    xarray.Dataset
        Channel-space dataset: the ``wavelength`` dimension is replaced by
        ``channel``. Variables without a ``wavelength`` dimension (e.g.
        ``status``) pass through unchanged, as do the dataset attributes.

    Raises
    ------
    ValueError
        If *result* has no ``wavelength`` dimension, or a channel's SRF
        does not overlap the simulated wavelength range (its integral
        would be zero).

    See Also
    --------
    brightness_temperature : Convert channel radiances to temperatures.
    pyradtran.interface.PyRadtranAccessor.run : ``channels=`` applies this
        function to the simulation output.

    Notes
    -----
    For a spectral quantity :math:`X(\lambda)` and the response
    :math:`\varphi_c(\lambda)` of channel :math:`c`, the channel value is

    .. math::

        \bar{X}_c = \frac{\int X(\lambda)\,\varphi_c(\lambda)\,
                     \mathrm{d}\lambda}
                    {\int \varphi_c(\lambda)\,\mathrm{d}\lambda}

    Both integrals use the trapezoidal rule on the *result's* wavelength
    grid. The output keeps the units of the input: a spectral radiance
    in mW m\ :sup:`-2` nm\ :sup:`-1` sr\ :sup:`-1` stays a band-mean
    spectral radiance, as :func:`brightness_temperature` expects.

    The simulated grid must resolve the SRF. With a coarse
    ``mol_abs_param`` (e.g. ``reptran coarse``) and a narrow channel,
    few grid points carry weight and the band average becomes noisy.

    Examples
    --------
    A linear spectrum at two time steps, convolved with three Gaussian
    channels:

    >>> import numpy as np
    >>> import xarray as xr
    >>> from pyradtran import convolve_channels
    >>> wl = np.arange(8000.0, 13001.0, 10.0)
    >>> result = xr.Dataset(
    ...     {"uu": (("time", "wavelength"), np.outer([1.0, 2.0], wl / 1000.0))},
    ...     coords={"time": [0, 1], "wavelength": wl},
    ... )
    >>> centres = xr.DataArray([8600.0, 10800.0, 12000.0], dims="channel")
    >>> srf = np.exp(-0.5 * ((result.wavelength - centres) / 200.0) ** 2)
    >>> srf = srf.assign_coords(channel=["8.6um", "10.8um", "12.0um"])
    >>> channel_ds = convolve_channels(result, srf)
    >>> channel_ds.uu.dims
    ('time', 'channel')
    >>> channel_ds.uu.round(2).values
    array([[ 8.6, 10.8, 12. ],
           [17.2, 21.6, 24. ]])

    Keep the spectrum next to the channel values:

    >>> sorted(convolve_channels(result, srf, keep_spectral=True).data_vars)
    ['uu', 'uu_spectral']

    A channel outside the simulated range raises:

    >>> convolve_channels(result, srf.assign_coords(wavelength=wl + 9000.0))
    Traceback (most recent call last):
        ...
    ValueError: SRF for channel(s) ['8.6um', '10.8um', '12.0um'] has no overlap with the result wavelength grid (8000.0–13000.0 nm); convolution would divide by zero

    A thermal spectrum with an ozone-like band and three imager channels:

    .. plot::

        from pyradtran import convolve_channels

        wl = np.arange(7500.0, 13501.0, 5.0)
        # smooth Planck-like continuum with an ozone-like absorption band
        spec = 7.0 - 1.5e-7 * (wl - 10000.0) ** 2
        spec -= 2.5 * np.exp(-0.5 * ((wl - 9600.0) / 180.0) ** 2)
        result = xr.Dataset({"uu": ("wavelength", spec)}, coords={"wavelength": wl})

        centres = xr.DataArray([8600.0, 9600.0, 11500.0], dims="channel")
        srf = np.exp(-0.5 * ((result.wavelength - centres) / 250.0) ** 2)
        srf = srf.assign_coords(channel=["8.6 µm", "9.6 µm", "11.5 µm"])
        channels = convolve_channels(result, srf)

        fig, ax = plt.subplots()
        ax.plot(wl / 1e3, spec, color="0.25", label="spectral radiance")
        for i, name in enumerate(srf.channel.values):
            colour = f"C{i}"
            ax.fill_between(wl / 1e3, srf.isel(channel=i) * 2.0, alpha=0.25,
                            color=colour, label=f"SRF {name} (scaled)")
            ax.plot(centres[i] / 1e3, channels.uu[i], "o", color=colour, ms=8)
        ax.set_xlabel("wavelength (µm)")
        ax.set_ylabel("radiance (mW m$^{-2}$ nm$^{-1}$ sr$^{-1}$)")
        ax.set_title("dots: channel-averaged radiance")
        ax.legend(fontsize=8, ncol=2)
    """
    if "wavelength" not in result.dims:
        raise ValueError("result has no 'wavelength' dimension to convolve")
    wl = result["wavelength"]
    phi = _interp_srf(srf, wl)
    norm = phi.integrate("wavelength")

    norm_vals = np.asarray(norm.values, dtype=float)
    bad = ~np.isfinite(norm_vals) | (norm_vals <= 0)
    if np.any(bad):
        bad_channels = [str(c) for c in np.asarray(srf["channel"].values)[bad]]
        raise ValueError(
            f"SRF for channel(s) {bad_channels} has no overlap with the "
            f"result wavelength grid "
            f"({float(wl.min())}–{float(wl.max())} nm); convolution would "
            f"divide by zero"
        )

    out = xr.Dataset(attrs=dict(result.attrs))
    for name, da in result.data_vars.items():
        if "wavelength" in da.dims:
            out[name] = (da * phi).integrate("wavelength") / norm
            if keep_spectral:
                out[f"{name}_spectral"] = da
        else:
            out[name] = da
    out["channel"] = srf["channel"]
    return out


def _interp_srf(srf: xr.DataArray, wl: xr.DataArray) -> xr.DataArray:
    """Interpolate an SRF onto the result wavelength grid, zero outside."""
    srf = srf.sortby("wavelength")
    srf_wl = srf["wavelength"].values.astype(float)
    target = wl.values.astype(float)
    phi_rows = [
        np.interp(
            target,
            srf_wl,
            srf.isel(channel=i).values.astype(float),
            left=0.0,
            right=0.0,
        )
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
    r"""Convert spectral radiance to brightness temperature (K).

    The brightness temperature is the temperature at which a blackbody
    emits the given radiance at the given wavelength (the inverse of the
    Planck function).

    Parameters
    ----------
    radiance : array-like or xarray.DataArray
        Spectral radiance :math:`L_\lambda`, e.g. the ``uu`` variable of
        a thermal simulation.
    wavelength_nm : float or array-like
        Wavelength in nanometres. Must broadcast against *radiance*; for
        channel-averaged radiances use the channel's central (or
        effective) wavelength.
    radiance_units : {"mW m-2 nm-1 sr-1", "W m-2 nm-1 sr-1"}
        Units of *radiance*, which must be *per nanometre*. The Notes
        list which uvspec settings produce which unit.

    Returns
    -------
    numpy.ndarray
        Brightness temperature in K, same shape as the broadcast inputs.
        Coordinates of an :class:`xarray.DataArray` input are dropped;
        wrap the result with :meth:`xarray.DataArray.copy` (``data=...``)
        to keep them (see Examples).

    Raises
    ------
    ValueError
        If *radiance_units* is not one of the supported unit strings.

    See Also
    --------
    convolve_channels : Band-average spectral radiances first.

    Notes
    -----
    The spectral radiance of a blackbody at temperature :math:`T` is

    .. math::

        B_\lambda(T) = \frac{2 h c^2}{\lambda^5}\,
                       \frac{1}{\exp\!\left(\dfrac{h c}{\lambda k_B T}\right) - 1}

    Solving :math:`B_\lambda(T_b) = L_\lambda` for the temperature gives

    .. math::

        T_b = \frac{c_2}{\lambda\,
              \ln\!\left(1 + \dfrac{c_1}{\lambda^5 L_\lambda}\right)},
        \qquad c_1 = 2 h c^2, \quad c_2 = \frac{h c}{k_B}

    with the CODATA 2018 values of :math:`h`, :math:`c` and :math:`k_B`.
    The logarithm is evaluated with :func:`numpy.log1p`, so the conversion
    stays accurate in the Rayleigh–Jeans limit
    (:math:`c_1 / \lambda^5 L_\lambda \ll 1`).

    The unit uvspec writes depends on the source:

    * ``source thermal``: by default uvspec reports thermal quantities
      *per spectral band* (1 cm\ :sup:`-1` wide unless changed), not per
      nanometre. Add ``params={"output_process": "per_nm"}`` to get
      W m\ :sup:`-2` nm\ :sup:`-1` sr\ :sup:`-1` and pass
      ``radiance_units="W m-2 nm-1 sr-1"``.
    * ``source solar``: the unit follows the solar spectrum file. The
      spectra shipped with libRadtran are in
      mW m\ :sup:`-2` nm\ :sup:`-1`, hence the default.

    For brightness temperatures at the simulated resolution, uvspec can
    do the inversion itself: ``params={"output_quantity": "brightness"}``
    returns every radiance and irradiance column in kelvin. Use this
    function to post-process radiances first (channel convolution,
    window corrections, noise) and convert afterwards.

    The conversion is monochromatic. Applied to a band-averaged radiance
    with the channel's central wavelength, it is a good approximation
    for narrow channels (width :math:`\lesssim 1` µm in the thermal
    window). For broad channels use an effective wavelength or a
    band-correction fit.

    Examples
    --------
    A radiance of 7.045 mW m\ :sup:`-2` nm\ :sup:`-1` sr\ :sup:`-1` at
    10.5 µm corresponds to a 280 K blackbody:

    >>> from pyradtran import brightness_temperature
    >>> float(brightness_temperature(7.045267, 10500.0).round(2))
    280.0

    Several channels in one call:

    >>> brightness_temperature([5.0, 7.5, 10.0], [8600.0, 10800.0, 12000.0]).round(1)
    array([268.6, 283.9, 308.3])

    After a thermal simulation with instrument channels
    (``output_process per_nm`` gives radiances in W per nanometre;
    ``center_nm`` holds the central wavelength of each channel):

    >>> channel_ds = ds.pyradtran.run(  # doctest: +SKIP
    ...     config_path="config/thermal_spectral.yaml",
    ...     params={"output_process": "per_nm"},
    ...     channels=srf,
    ... )
    >>> tb = channel_ds["uu"].copy(  # doctest: +SKIP
    ...     data=brightness_temperature(
    ...         channel_ds["uu"], channel_ds["center_nm"], "W m-2 nm-1 sr-1"
    ...     )
    ... )
    >>> tb.attrs["units"] = "K"  # doctest: +SKIP

    The Planck curves being inverted, in the default radiance units:

    .. plot::

        from pyradtran import brightness_temperature

        h, c, kB = 6.62607015e-34, 2.99792458e8, 1.380649e-23

        def planck(wl_nm, T):
            lam = wl_nm * 1e-9
            return 2 * h * c**2 / lam**5 / np.expm1(h * c / (lam * kB * T)) * 1e-6

        wl = np.linspace(4000.0, 20000.0, 400)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 3.6))
        for T in (220, 240, 260, 280, 300):
            ax1.plot(wl / 1e3, planck(wl, T), label=f"{T} K")
        ax1.set_xlabel("wavelength (µm)")
        ax1.set_ylabel("$B_\\lambda$ (mW m$^{-2}$ nm$^{-1}$ sr$^{-1}$)")
        ax1.set_title("Planck function")
        ax1.legend(fontsize=8)

        L = np.linspace(0.5, 12.0, 200)
        for wl_um in (8.6, 10.8, 12.0):
            ax2.plot(L, brightness_temperature(L, wl_um * 1e3), label=f"{wl_um} µm")
        ax2.set_xlabel("radiance (mW m$^{-2}$ nm$^{-1}$ sr$^{-1}$)")
        ax2.set_ylabel("$T_b$ (K)")
        ax2.set_title("brightness_temperature()")
        ax2.legend(fontsize=8)
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
