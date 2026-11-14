"""Regenerate the figures used in the user guide (needs a working libRadtran).

Run from the repository root::

    python book/img/make_guide_figures.py

The simulations take about a minute. Outputs are written next to this file.
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

import pyradtran
from pyradtran import Var, brightness_temperature, load_config

logging.getLogger("pyradtran").setLevel(logging.CRITICAL)
OUT = Path(__file__).parent
plt.rcParams.update(
    {
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "font.size": 10,
        "figure.dpi": 160,
    }
)
RUN = dict(save_to_file=False, show_progress=False)


def config(source, wavelength_nm, integrate, columns, mol_abs="reptran coarse"):
    cfg = load_config()
    sd = cfg.simulation_defaults
    sd.source = source
    sd.rte_solver = "disort"
    sd.mol_abs_param = mol_abs
    sd.wavelength_nm = wavelength_nm
    sd.integrate_wavelength = integrate
    sd.output_columns = columns
    sd.output_altitudes_km = [0.0]
    cfg.execution.max_workers = 8
    cfg.execution.cleanup_temp_files = True
    return cfg


def fig_spectrum():
    """Spectral surface irradiance for three solar zenith angles."""
    cfg = config("solar", [300, 2500], False, ["lambda", "edir", "edn"])
    ds = xr.Dataset(
        coords={
            "time": [pd.Timestamp("2025-06-21 12:00")],
            "latitude": ("time", [52.0]),
            "longitude": ("time", [13.0]),
            "sza": [30.0, 60.0, 80.0],
        }
    )
    ds["sza_deg"] = ds.sza
    res = ds.pyradtran.run(config=cfg, params={"sza": Var("sza_deg")}, **RUN)
    res = res.isel(time=0, altitude=0)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for i, sza in enumerate(res.sza.values):
        sel = res.sel(sza=sza)
        ax.plot(
            sel.wavelength,
            sel.edir + sel.edn,
            color=f"C{i}",
            lw=0.8,
            label=f"global, SZA {sza:.0f}°",
        )
        ax.plot(sel.wavelength, sel.edn, color=f"C{i}", ls=":", lw=0.8)
    ax.plot([], [], color="0.4", ls=":", lw=1, label="diffuse part")
    ax.set_xlabel("wavelength (nm)")
    ax.set_ylabel("irradiance (mW m$^{-2}$ nm$^{-1}$)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "guide_spectrum.png")
    return res


def fig_cloud_forcing():
    """Surface cloud radiative effect versus liquid water path."""
    lwp = np.array([0.0, 5, 10, 20, 40, 80, 150, 300])  # g m-2
    thickness_m = 500.0
    ds = xr.Dataset(
        coords={
            "time": [pd.Timestamp("2025-06-21 12:00")],
            "latitude": ("time", [78.9]),
            "longitude": ("time", [11.9]),
            "lwp": lwp,
        },
        data_vars={
            "lwc": ("lwp", lwp / thickness_m),  # g m-3
            "cbh": ("lwp", np.full(lwp.size, 1.0)),
            "cth": ("lwp", np.full(lwp.size, 1.0 + thickness_m / 1e3)),
        },
    )
    cloud = dict(cloud_wc_var="lwc", cloud_top_var="cth", cloud_bottom_var="cbh")
    cols = ["edir", "edn", "eup"]
    out = {}
    for albedo in (0.1, 0.8):
        sol = ds.pyradtran.run(
            config=config("solar", [300, 3500], True, cols),
            params={"albedo": albedo},
            **cloud,
            **RUN,
        ).isel(time=0, altitude=0)
        th = ds.pyradtran.run(
            config=config("thermal", [4000, 100000], True, cols),
            params={"albedo": 0.01, "sur_temperature": 273.15},
            **cloud,
            **RUN,
        ).isel(time=0, altitude=0)
        net_sw = (sol.edir + sol.edn - sol.eup) / 1e3  # mW m-2 -> W m-2
        net_lw = th.edn - th.eup
        out[albedo] = (net_sw - net_sw.sel(lwp=0), net_lw - net_lw.sel(lwp=0))

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6), sharey=True)
    cloudy = lwp > 0
    for ax, (albedo, (sw, lw)), name in zip(
        axes, out.items(), ("open ocean", "sea ice")
    ):
        ax.plot(lwp[cloudy], sw[cloudy], "o-", color="C1", label="shortwave")
        ax.plot(lwp[cloudy], lw[cloudy], "s-", color="C3", label="longwave")
        ax.plot(lwp[cloudy], (sw + lw)[cloudy], "k^-", label="net")
        ax.axhline(0, color="0.5", lw=0.8)
        ax.set_xscale("log")
        ax.set_xlabel("liquid water path (g m$^{-2}$)")
        ax.set_title(f"{name} (albedo {albedo})", fontsize=10)
    axes[0].set_ylabel("surface cloud radiative effect (W m$^{-2}$)")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "guide_cloud_forcing.png")
    return out


def fig_brightness_temperature():
    """Nadir brightness-temperature spectrum seen from 10 km."""
    cfg = config("thermal", [7500, 13500], False, ["lambda", "uu"], "reptran fine")
    cfg.simulation_defaults.output_altitudes_km = [10.0]
    ds = xr.Dataset(
        coords={
            "time": [pd.Timestamp("2025-06-21 12:00")],
            "latitude": ("time", [52.0]),
            "longitude": ("time", [13.0]),
            "surface_temperature": [260.0, 280.0, 300.0],
        }
    )
    ds["tsurf"] = ds.surface_temperature
    res = ds.pyradtran.run(
        config=cfg,
        params={
            "sur_temperature": Var("tsurf"),
            "albedo": 0.0,
            "output_process": "per_nm",  # W m-2 nm-1 sr-1 instead of per band
        },
        **RUN,
    ).isel(time=0, altitude=0)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for ts in res.surface_temperature.values:
        sel = res.sel(surface_temperature=ts)
        tb = brightness_temperature(sel.uu, sel.wavelength, "W m-2 nm-1 sr-1")
        ax.plot(sel.wavelength / 1e3, tb, label=f"surface {ts:.0f} K")
    ax.set_xlabel("wavelength (µm)")
    ax.set_ylabel("brightness temperature (K)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "guide_brightness_temperature.png")
    return res


if __name__ == "__main__":
    print(f"pyradtran {pyradtran.__version__}")
    fig_spectrum()
    fig_cloud_forcing()
    fig_brightness_temperature()
    print(f"figures written to {OUT}")
