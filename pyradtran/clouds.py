# pyradtran/clouds.py
r"""
Cloud layers and libRadtran cloud files.

Builds 1D cloud input files (``wc_file 1D`` / ``ic_file 1D``) from a
list of homogeneous slabs:

1. Describe the cloud as :class:`CloudLayer` objects, from a few numbers
   (:meth:`CloudGenerator.from_simple_parameters`) or from an ERA5 dataset
   (:meth:`CloudGenerator.from_era5_dataset`).
2. Write them with :class:`CloudFileWriter`, which puts the file rows at
   the layer boundaries so that uvspec sees exactly these slabs.
3. Pass the file to uvspec through the ``wc_file`` / ``ic_file`` parameter
   of :meth:`~pyradtran.interface.PyRadtranAccessor.run`.

.. autosummary::

   CloudLayer
   CloudGenerator
   CloudFileWriter
   generate_cloud_file_from_era5

.. note::

   This is the file-based cloud path.  A cloud profile can also be
   passed to :meth:`~pyradtran.interface.PyRadtranAccessor.run` directly
   as a dict (``params={"wc_file": {"z": ..., "lwc": ..., "reff": ...}}``);
   :func:`pyradtran.era5.cloud_profiles` builds such dicts from ERA5.
   Both routes describe the same cloud: the ERA5 functions here wrap
   :mod:`pyradtran.era5`.

Examples
--------
A 500 m thick stratus deck, written to a file:

>>> import pathlib, tempfile
>>> from pyradtran.clouds import CloudGenerator, CloudFileWriter
>>> layers = CloudGenerator.from_simple_parameters(
...     z_base_km=1.0, z_top_km=1.5, lwc_g_m3=0.3, r_eff_um=8.0
... )
>>> tmp = tempfile.TemporaryDirectory()
>>> path = CloudFileWriter.write_water_cloud_file(
...     layers, pathlib.Path(tmp.name) / "wc.dat"
... )
>>> print(path.read_text(), end="")
#        z        LWC    R_eff
#       (km)    (g/m^3)     (um)
  1.500000   0.000000    8.000
  1.000000   0.300000    8.000
>>> tmp.cleanup()

Use the file in a simulation (``1D`` is part of the option value):

>>> result = ds.pyradtran.run(  # doctest: +SKIP
...     config_path=cfg, params={"wc_file": f"1D {path}"}
... )

See Also
--------
pyradtran.era5.cloud_profiles : ERA5 cloud profiles as dict-valued
    ``wc_file`` / ``ic_file`` parameters.
pyradtran.config.CloudParameters : Declarative cloud settings in YAML configs.
"""

import logging
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import xarray as xr

from .era5 import cloud_profiles, normalize_era5, select_profile

logger = logging.getLogger(__name__)


@dataclass
class CloudLayer:
    r"""A homogeneous cloud slab between two altitudes.

    A validated data container; write lists of layers with
    :class:`CloudFileWriter`.

    Parameters
    ----------
    z_bottom_km : float
        Bottom altitude of the layer in km above sea level.
    z_top_km : float
        Top altitude of the layer in km above sea level.  Must be greater
        than *z_bottom_km*.
    lwc_g_m3 : float, default ``0.0``
        Liquid water content in g m\ :sup:`-3`.
    iwc_g_m3 : float, default ``0.0``
        Ice water content in g m\ :sup:`-3`.
    r_eff_um : float, default ``10.0``
        Effective droplet / crystal radius in µm.  A layer has a single
        radius, used for the liquid and the ice file when the layer
        contains both phases.
    cloud_fraction : float, default ``1.0``
        Cloud fraction (0–1).  Validated and stored, but nothing in
        pyRadtran reads it: :class:`CloudFileWriter` writes the water
        content as given and no cloud-overlap option is set.

    Raises
    ------
    ValueError
        If *z_bottom_km* >= *z_top_km*, a water content is negative,
        the effective radius is not positive, or the cloud fraction is
        outside [0, 1].

    See Also
    --------
    CloudGenerator.from_simple_parameters : Build layers from basic numbers.
    CloudGenerator.from_era5_dataset : Build layers from ERA5 reanalysis.
    CloudFileWriter : Write layers to libRadtran cloud files.

    Notes
    -----
    Validation runs once, at construction; attributes changed afterwards
    are not re-checked.  No upper limits are enforced; whether an
    effective radius is usable depends on the cloud parameterisation
    selected in uvspec (``wc_properties`` / ``ic_properties``).

    Examples
    --------
    >>> from pyradtran.clouds import CloudLayer
    >>> layer = CloudLayer(z_bottom_km=1.0, z_top_km=2.0, lwc_g_m3=0.3)
    >>> layer
    CloudLayer(z_bottom_km=1.0, z_top_km=2.0, lwc_g_m3=0.3, iwc_g_m3=0.0, r_eff_um=10.0, cloud_fraction=1.0)
    >>> layer.z_top_km - layer.z_bottom_km
    1.0


    Unphysical layers are rejected:

    >>> CloudLayer(z_bottom_km=2.0, z_top_km=1.0)
    Traceback (most recent call last):
        ...
    ValueError: Bottom altitude (2.0) must be less than top altitude (1.0)
    >>> CloudLayer(1.0, 2.0, lwc_g_m3=-0.1)
    Traceback (most recent call last):
        ...
    ValueError: Water content cannot be negative
    >>> CloudLayer(1.0, 2.0, r_eff_um=0.0)
    Traceback (most recent call last):
        ...
    ValueError: Effective radius must be positive
    >>> CloudLayer(1.0, 2.0, cloud_fraction=1.5)
    Traceback (most recent call last):
        ...
    ValueError: Cloud fraction must be between 0 and 1
    """

    z_bottom_km: float
    z_top_km: float
    lwc_g_m3: float = 0.0  # Liquid water content
    iwc_g_m3: float = 0.0  # Ice water content
    r_eff_um: float = 10.0  # Effective radius in micrometers
    cloud_fraction: float = 1.0  # Cloud fraction (0-1)

    def __post_init__(self):
        """Validate cloud layer parameters."""
        if self.z_bottom_km >= self.z_top_km:
            raise ValueError(
                f"Bottom altitude ({self.z_bottom_km}) must be less than top altitude ({self.z_top_km})"
            )
        if self.lwc_g_m3 < 0 or self.iwc_g_m3 < 0:
            raise ValueError("Water content cannot be negative")
        if self.r_eff_um <= 0:
            raise ValueError("Effective radius must be positive")
        if not 0 <= self.cloud_fraction <= 1:
            raise ValueError("Cloud fraction must be between 0 and 1")


class CloudGenerator:
    """Factory for :class:`CloudLayer` sequences.

    All methods are static:

    * :meth:`from_simple_parameters`: uniform slab from a few numbers;
    * :meth:`from_era5_dataset`: one layer per cloudy ERA5 level.

    See Also
    --------
    CloudFileWriter : Persist layers to libRadtran ``.dat`` files.
    pyradtran.era5.cloud_profiles : ERA5 clouds as dict-valued
        ``wc_file`` / ``ic_file`` parameters.

    Examples
    --------
    >>> from pyradtran.clouds import CloudGenerator
    >>> layers = CloudGenerator.from_simple_parameters(1.0, 2.0, n_layers=2)
    >>> [(layer.z_bottom_km, layer.z_top_km) for layer in layers]
    [(1.0, 1.5), (1.5, 2.0)]
    """

    @staticmethod
    def from_era5_dataset(
        ds: xr.Dataset,
        time: Optional[datetime] = None,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
        cloud_variables: Optional[Dict[str, str]] = None,
        altitude_levels_km: Optional[np.ndarray] = None,
        lwc_threshold: float = 1e-4,
        iwc_threshold: float = 1e-4,
        default_r_eff_water: float = 10.0,
        default_r_eff_ice: float = 20.0,
        pressure_levels: Optional[str] = None,
        geopotential_var: Optional[str] = None,
    ) -> List[CloudLayer]:
        r"""Extract cloud layers from an ERA5 :class:`xarray.Dataset`.

        Normalises the dataset (:func:`~pyradtran.era5.normalize_era5`),
        selects one column (:func:`~pyradtran.era5.select_profile`),
        converts it with :func:`~pyradtran.era5.cloud_profiles` and
        returns the profiles as :class:`CloudLayer` objects.  Layer
        boundaries lie midway between the ERA5 levels, so the layers tile
        the cloudy part of the column.

        Parameters
        ----------
        ds : xarray.Dataset
            ERA5 dataset on pressure levels in any naming flavour
            :func:`~pyradtran.era5.normalize_era5` understands (CDS
            short names, ARCO long names, ``time`` or ``valid_time``,
            pressure in hPa or Pa).  Needs temperature and at least one
            of liquid / ice water content (kg kg\ :sup:`-1`).
        time : datetime.datetime, str or numpy.datetime64, optional
            Time step to select (nearest).  *None* takes the first time
            step.
        lat, lon : float, optional
            Position of the column (nearest grid point).  Required when
            *ds* has a ``latitude`` or ``longitude`` dimension.
        cloud_variables : dict of str, optional
            Dataset variable names, for datasets with names
            ``normalize_era5`` does not know.  Keys: ``"lwc"``,
            ``"iwc"``, ``"temp"``, ``"z"``; e.g. ``{"lwc": "LWC",
            "temp": "T"}``.  A phase whose key (``"lwc"`` / ``"iwc"``)
            is missing from a dict given here is switched off;
            ``"temp"`` and ``"z"`` default to ``"t"`` and ``"z"``.  A
            ``"cc"`` entry is accepted and ignored.
        altitude_levels_km : numpy.ndarray, optional
            Altitude (km) of each pressure level, in the dataset's own
            level order.  Overrides the geopotential variable.
        lwc_threshold : float, default ``1e-4``
            A level counts as liquid cloud when its LWC is greater than
            this value (g m\ :sup:`-3`).
        iwc_threshold : float, default ``1e-4``
            Same for the IWC (g m\ :sup:`-3`).
        default_r_eff_water : float, default ``10.0``
            Effective radius assigned to liquid layers (µm).
        default_r_eff_ice : float, default ``20.0``
            Effective radius assigned to ice layers (µm).
        pressure_levels : str, optional
            Name of the pressure coordinate, if it is not one of
            ``pressure_level``, ``level``, ``plev``, ``isobaricInhPa``.
        geopotential_var : str, optional
            Name of the geopotential variable; takes precedence over
            ``cloud_variables["z"]``.

        Returns
        -------
        list of CloudLayer
            Liquid layers (``iwc_g_m3 = 0``) and ice layers
            (``lwc_g_m3 = 0``), sorted by base altitude.  Empty when no
            level exceeds a threshold.

        Raises
        ------
        ValueError
            If *ds* has spatial dimensions and *lat* or *lon* is *None*,
            if it has no pressure-level coordinate, or if it has no
            temperature variable (needed for the air density).

        See Also
        --------
        generate_cloud_file_from_era5 : One-step convenience wrapper.
        pyradtran.era5.cloud_profiles : The conversion used here; its
            Notes describe water content, altitudes and layer boundaries.

        Notes
        -----
        * **One layer per phase.**  A level with liquid and ice yields
          two layers with the same boundaries, each with its own
          effective radius.
        * Within each phase the layers are contiguous from the lowest to
          the highest level above the threshold; levels in between keep
          their own (sub-threshold) content, and levels with exactly
          zero content are left out.
        * The water contents are grid-box means, not divided by the
          cloud cover.  ``cloud_fraction`` of the returned layers is
          always 1; the ERA5 cloud cover ``cc`` is not read.
        * Without a geopotential variable (and without
          *altitude_levels_km*) the altitudes are a rough estimate from
          pressure with a constant 7 km scale height.

        Examples
        --------
        A synthetic single-column ERA5 cube: liquid water at 850 and
        925 hPa, a little ice at 500 hPa.

        >>> import numpy as np
        >>> import xarray as xr
        >>> from pyradtran.clouds import CloudGenerator
        >>> dims = ("time", "pressure_level", "latitude", "longitude")
        >>> def column(values):
        ...     return (dims, np.reshape(values, (1, 5, 1, 1)))
        >>> era5 = xr.Dataset(
        ...     {
        ...         "t": column([252.0, 268.0, 276.0, 280.0, 284.0]),
        ...         "z": column([54000.0, 29500.0, 14500.0, 7500.0, 1000.0]),
        ...         "clwc": column([0.0, 0.0, 2.0e-4, 3.0e-4, 0.0]),
        ...         "ciwc": column([2.0e-5, 0.0, 0.0, 0.0, 0.0]),
        ...     },
        ...     coords={
        ...         "time": np.array(["2024-07-15T12"], dtype="datetime64[ns]"),
        ...         "pressure_level": [500.0, 700.0, 850.0, 925.0, 1000.0],
        ...         "latitude": [78.0],
        ...         "longitude": [15.0],
        ...     },
        ... )
        >>> era5["z"].attrs["units"] = "m2 s-2"
        >>> layers = CloudGenerator.from_era5_dataset(era5, lat=78.0, lon=15.0)
        >>> for layer in layers:
        ...     print(
        ...         f"{layer.z_bottom_km:.3f}-{layer.z_top_km:.3f} km  "
        ...         f"LWC {layer.lwc_g_m3:.3f}  IWC {layer.iwc_g_m3:.4f}  "
        ...         f"r_eff {layer.r_eff_um:.0f}"
        ...     )
        0.433-1.122 km  LWC 0.345  IWC 0.0000  r_eff 10
        1.122-2.243 km  LWC 0.215  IWC 0.0000  r_eff 10
        4.257-6.756 km  LWC 0.000  IWC 0.0138  r_eff 20

        The 925 hPa level sits at 7500 / 9.80665 = 765 m; its layer
        reaches from the midpoint to the 1000 hPa level (433 m) to the
        midpoint to the 850 hPa level (1122 m).  A real file:

        >>> ds = xr.open_dataset("era5_cloud.nc")  # doctest: +SKIP
        >>> layers = CloudGenerator.from_era5_dataset(  # doctest: +SKIP
        ...     ds, lat=78.0, lon=15.0, default_r_eff_water=8.0
        ... )
        """
        canonical = {"lwc": "clwc", "iwc": "ciwc", "temp": "t", "z": "z"}
        names = dict(canonical)
        if cloud_variables is not None:
            names = {"temp": "t", "z": "z", **cloud_variables}
        if geopotential_var:
            names["z"] = geopotential_var
        rename = {
            names[key]: var
            for key, var in canonical.items()
            if key in names and names[key] != var and names[key] in ds.variables
        }
        if pressure_levels and pressure_levels != "pressure_level":
            rename[pressure_levels] = "pressure_level"
        ds = normalize_era5(
            ds.drop_vars(list(rename.values()), errors="ignore").rename(rename)
        )
        ds = ds.drop_vars(
            [var for key, var in canonical.items() if key not in names],
            errors="ignore",
        )

        if time is None and "valid_time" in ds.dims:
            ds = ds.isel(valid_time=0)
        profile = select_profile(ds, lat, lon, time)
        if "pressure_level" not in profile.coords:
            raise ValueError(
                "ERA5 dataset has no pressure-level coordinate "
                "(name it with pressure_levels=...)"
            )
        if "t" not in profile.variables:
            raise ValueError(
                "ERA5 dataset has no temperature variable; it is required to "
                "convert kg/kg to g m-3 (name it with cloud_variables={'temp': ...})"
            )
        if altitude_levels_km is not None:
            profile["z"] = (
                "pressure_level",
                np.asarray(altitude_levels_km, dtype=float),
                {"units": "km"},
            )

        reff = (default_r_eff_water, default_r_eff_ice)
        wc, _ = cloud_profiles(profile, *reff, threshold_g_m3=lwc_threshold)
        _, ic = cloud_profiles(profile, *reff, threshold_g_m3=iwc_threshold)

        # A row's value fills the layer between its altitude and the row above.
        layers = [
            CloudLayer(z[i], z[i - 1], r_eff_um=prof["reff"][i], **{f"{key}_g_m3": c})
            for prof, key in ((wc, "lwc"), (ic, "iwc"))
            if prof is not None
            for z in [prof["z"]]
            for i, c in enumerate(prof[key])
            if i > 0 and c > 0
        ]
        return sorted(layers, key=lambda layer: layer.z_bottom_km)

    @staticmethod
    def from_simple_parameters(
        z_base_km: float,
        z_top_km: float,
        lwc_g_m3: float = 0.1,
        iwc_g_m3: float = 0.0,
        r_eff_um: float = 10.0,
        cloud_fraction: float = 1.0,
        n_layers: int = 1,
    ) -> List[CloudLayer]:
        r"""Create uniform cloud layers from basic parameters.

        Splits *z_base_km* … *z_top_km* into *n_layers* sub-layers of
        equal thickness with the same microphysical values.

        Parameters
        ----------
        z_base_km : float
            Cloud base altitude (km above sea level).
        z_top_km : float
            Cloud top altitude (km above sea level).
        lwc_g_m3 : float, default ``0.1``
            Liquid water content (g m\ :sup:`-3`).  The default is
            non-zero: pass ``lwc_g_m3=0.0`` for a pure ice cloud.
        iwc_g_m3 : float, default ``0.0``
            Ice water content (g m\ :sup:`-3`).
        r_eff_um : float, default ``10.0``
            Effective droplet / crystal radius (µm).
        cloud_fraction : float, default ``1.0``
            Cloud fraction (0–1); stored on the layers but not used by
            :class:`CloudFileWriter`.
        n_layers : int, default ``1``
            Number of sub-layers to create.

        Returns
        -------
        list of CloudLayer
            *n_layers* contiguous layers ordered from bottom to top.

        Raises
        ------
        ValueError
            If *n_layers* < 1, or any value fails the
            :class:`CloudLayer` validation (*z_base_km* >= *z_top_km*,
            negative water content, non-positive radius, cloud fraction
            outside [0, 1]).

        See Also
        --------
        CloudFileWriter.write_water_cloud_file : Write layers to disk.
        CloudFileWriter.write_ice_cloud_file : Ice counterpart.

        Notes
        -----
        The sub-layers are identical, so *n_layers* does not change the
        cloud uvspec sees (the file only gains one row per internal
        boundary); it is a starting point for editing single layers.

        For orientation (pyRadtran computes none of this), the liquid
        water path of the cloud is

        .. math::

            \mathrm{LWP} = \int_{z_\mathrm{base}}^{z_\mathrm{top}}
                           \mathrm{LWC}\,\mathrm{d}z
                         = \mathrm{LWC}\,(z_\mathrm{top} - z_\mathrm{base})

        and a common rule of thumb for the visible optical thickness of
        a liquid cloud (geometric-optics limit, extinction efficiency
        of 2) is

        .. math::

            \tau \approx \frac{3\,\mathrm{LWP}}{2\,\rho_w\,r_\mathrm{eff}},

        with the density of liquid water :math:`\rho_w = 1000`
        kg m\ :sup:`-3`.  A 1 km thick cloud with 0.3 g m\ :sup:`-3`
        and :math:`r_\mathrm{eff}` = 10 µm has LWP = 300 g m\ :sup:`-2`
        and :math:`\tau \approx 45`.  The optical thickness uvspec
        actually uses follows from the selected ``wc_properties`` /
        ``ic_properties`` parameterisation.

        Examples
        --------
        >>> from pyradtran.clouds import CloudGenerator
        >>> layers = CloudGenerator.from_simple_parameters(
        ...     z_base_km=1.0, z_top_km=2.0, lwc_g_m3=0.3, n_layers=4
        ... )
        >>> len(layers)
        4
        >>> [(layer.z_bottom_km, layer.z_top_km) for layer in layers]
        [(1.0, 1.25), (1.25, 1.5), (1.5, 1.75), (1.75, 2.0)]
        >>> layers[0]
        CloudLayer(z_bottom_km=1.0, z_top_km=1.25, lwc_g_m3=0.3, iwc_g_m3=0.0, r_eff_um=10.0, cloud_fraction=1.0)

        Liquid water path in g m\ :sup:`-2` (thickness in km × 1000 m ×
        g m\ :sup:`-3`):

        >>> sum(
        ...     layer.lwc_g_m3 * (layer.z_top_km - layer.z_bottom_km) * 1000.0
        ...     for layer in layers
        ... )
        300.0

        Invalid input:

        >>> CloudGenerator.from_simple_parameters(2.0, 1.0)
        Traceback (most recent call last):
            ...
        ValueError: Bottom altitude (2.0) must be less than top altitude (1.0)
        >>> CloudGenerator.from_simple_parameters(1.0, 2.0, n_layers=0)
        Traceback (most recent call last):
            ...
        ValueError: Number of layers must be positive

        A stratus deck below a cirrus layer, drawn from the layer lists:

        .. plot::

            from pyradtran.clouds import CloudGenerator

            stratus = CloudGenerator.from_simple_parameters(
                1.0, 2.0, lwc_g_m3=0.3, r_eff_um=10.0, n_layers=4
            )
            cirrus = CloudGenerator.from_simple_parameters(
                8.0, 9.5, lwc_g_m3=0.0, iwc_g_m3=0.02, r_eff_um=30.0, n_layers=3
            )

            fig, ax = plt.subplots(figsize=(5.5, 4.2))
            for layers, attr, color, label in (
                (stratus, "lwc_g_m3", "C0", "liquid, $r_{eff}$ = 10 µm"),
                (cirrus, "iwc_g_m3", "C1", "ice, $r_{eff}$ = 30 µm"),
            ):
                for k, layer in enumerate(layers):
                    ax.barh(
                        layer.z_bottom_km,
                        getattr(layer, attr),
                        height=layer.z_top_km - layer.z_bottom_km,
                        align="edge",
                        color=color,
                        alpha=0.4,
                        edgecolor=color,
                        label=label if k == 0 else None,
                    )
            ax.set_xlabel("water content (g m$^{-3}$)")
            ax.set_ylabel("altitude (km)")
            ax.set_ylim(0, 11)
            ax.legend()
            plt.tight_layout()
        """
        if n_layers <= 0:
            raise ValueError("Number of layers must be positive")

        layer_thickness = (z_top_km - z_base_km) / n_layers
        layers = []

        for i in range(n_layers):
            z_bottom = z_base_km + i * layer_thickness
            z_top = z_bottom + layer_thickness

            layer = CloudLayer(
                z_bottom_km=z_bottom,
                z_top_km=z_top,
                lwc_g_m3=lwc_g_m3,
                iwc_g_m3=iwc_g_m3,
                r_eff_um=r_eff_um,
                cloud_fraction=cloud_fraction,
            )
            layers.append(layer)

        return layers


def _warn_ignored(include_zero_layers, altitude_resolution_km) -> None:
    """Warn about the ignored legacy cloud-writer arguments."""
    if include_zero_layers is not None or altitude_resolution_km is not None:
        warnings.warn(
            "include_zero_layers and altitude_resolution_km are ignored and "
            "will be removed; cloud files use the exact layer boundaries",
            DeprecationWarning,
            stacklevel=3,
        )


def _write_cloud_file(cloud_layers, output_path, attr, label, filler_reff) -> Path:
    """Write the layers with ``attr > 0`` as an exact libRadtran 1D file.

    See :meth:`CloudFileWriter.write_water_cloud_file` for the row layout.
    """
    layers = sorted(
        (layer for layer in cloud_layers if getattr(layer, attr) > 0),
        key=lambda layer: -layer.z_top_km,
    )
    rows = []
    base = None
    for layer in layers:
        top, bottom = round(layer.z_top_km, 6), round(layer.z_bottom_km, 6)
        if base is not None and top > base:
            raise ValueError(
                f"Cloud layers overlap: a layer with top at {top} km lies above "
                f"the base ({base} km) of the layer above it"
            )
        if top != base:
            rows.append((top, 0.0, layer.r_eff_um))
        rows.append((bottom, getattr(layer, attr), layer.r_eff_um))
        base = bottom
    if not rows:
        logger.warning(f"No layer with {label} > 0; writing a cloud-free file")
        rows = [(1.0, 0.0, filler_reff), (0.0, 0.0, filler_reff)]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(f"#        z        {label}    R_eff\n")
        f.write("#       (km)    (g/m^3)     (um)\n")
        for z, content, r_eff in rows:
            f.write(f"{z:10.6f} {content:10.6f} {r_eff:8.3f}\n")
    logger.info(f"Wrote {len(rows)} levels to {output_path}")
    return output_path


class CloudFileWriter:
    r"""Persist :class:`CloudLayer` sequences as libRadtran ``.dat`` files.

    One static method per phase:

    * :meth:`write_water_cloud_file` → ``wc_file 1D``
    * :meth:`write_ice_cloud_file`   → ``ic_file 1D``

    Both write a three-column ASCII table (altitude in km, water content
    in g m\ :sup:`-3`, effective radius in µm), top down, with rows at
    the layer boundaries so that uvspec sees exactly the given layers.
    The format is described in :meth:`write_water_cloud_file`.

    See Also
    --------
    CloudGenerator : Build :class:`CloudLayer` lists from data or parameters.
    pyradtran.core.Simulation.format_cloud_profile : Writer behind
        dict-valued ``wc_file`` / ``ic_file`` parameters.
    """

    @staticmethod
    def write_water_cloud_file(
        cloud_layers: List[CloudLayer],
        output_path: Union[str, Path],
        include_zero_layers: Optional[bool] = None,
        altitude_resolution_km: Optional[float] = None,
    ) -> Path:
        r"""Write a water-cloud ``.dat`` file for libRadtran.

        Three columns, altitude (km), liquid water content
        (g m\ :sup:`-3`) and effective radius (µm), top down, with rows
        at the layer boundaries.

        Parameters
        ----------
        cloud_layers : list of CloudLayer
            Layers to write, in any order.  Only layers with
            ``lwc_g_m3 > 0`` are used.
        output_path : str or pathlib.Path
            Destination file.  Parent directories are created; an
            existing file is overwritten.
        include_zero_layers, altitude_resolution_km : optional
            Deprecated and ignored (the file is written at the exact
            layer boundaries); passing either emits a
            :class:`DeprecationWarning`.

        Returns
        -------
        pathlib.Path
            *output_path* as a :class:`~pathlib.Path`.

        Raises
        ------
        ValueError
            If two liquid layers overlap.

        See Also
        --------
        write_ice_cloud_file : Same for ``iwc_g_m3``.
        pyradtran.era5.cloud_profiles : ERA5 cloud profiles as dicts.

        Notes
        -----
        **File format.**  Two comment lines, then rows formatted as
        ``%10.6f %10.6f %8.3f``, highest altitude first:

        ====== ======================= =================
        Column Quantity                Unit
        ====== ======================= =================
        1      altitude *z*            km
        2      liquid water content    g m\ :sup:`-3`
        3      effective radius        µm
        ====== ======================= =================

        **Rows.**  libRadtran reads the value in a ``wc_file 1D`` row as
        the property of the layer between that altitude and the next row
        above; the value in the topmost row is not used.  Going through
        the layers top down, the writer emits

        * a row with zero LWC at the layer top, unless the layer above
          ends exactly there, and
        * a row at the layer base with the layer's LWC and effective
          radius.

        uvspec then sees exactly the :class:`CloudLayer` list: no cloud
        above the highest top, in the gaps, or below the lowest base.
        Boundaries are compared after rounding to the file resolution of
        10\ :sup:`-6` km, so sub-layers whose boundaries differ only by
        floating-point rounding count as touching.

        ``cloud_fraction`` is not used.  When no layer contains liquid
        water, a warning is logged and a cloud-free placeholder (two
        zero rows at 1 and 0 km) is written.

        Examples
        --------
        >>> import pathlib, tempfile
        >>> from pyradtran.clouds import CloudFileWriter, CloudGenerator
        >>> tmp = tempfile.TemporaryDirectory()
        >>> out = pathlib.Path(tmp.name)
        >>> layers = CloudGenerator.from_simple_parameters(
        ...     z_base_km=1.0, z_top_km=1.5, lwc_g_m3=0.3, r_eff_um=8.0
        ... )
        >>> path = CloudFileWriter.write_water_cloud_file(layers, out / "wc.dat")
        >>> print(path.read_text(), end="")
        #        z        LWC    R_eff
        #       (km)    (g/m^3)     (um)
          1.500000   0.000000    8.000
          1.000000   0.300000    8.000

        libRadtran reads this as 0.3 g m\ :sup:`-3` from 1.0 to 1.5 km.
        Two separate decks, the lower one made of two touching layers:

        >>> from pyradtran.clouds import CloudLayer
        >>> decks = [
        ...     CloudLayer(1.0, 1.5, lwc_g_m3=0.2),
        ...     CloudLayer(1.5, 2.0, lwc_g_m3=0.4),
        ...     CloudLayer(2.5, 3.0, lwc_g_m3=0.1, r_eff_um=6.0),
        ... ]
        >>> path = CloudFileWriter.write_water_cloud_file(decks, out / "wc.dat")
        >>> print(path.read_text(), end="")
        #        z        LWC    R_eff
        #       (km)    (g/m^3)     (um)
          3.000000   0.000000    6.000
          2.500000   0.100000    6.000
          2.000000   0.000000   10.000
          1.500000   0.400000   10.000
          1.000000   0.200000   10.000

        The zero row at 2.0 km keeps the gap up to 2.5 km cloud-free.
        Overlapping layers are refused:

        >>> CloudFileWriter.write_water_cloud_file(
        ...     [CloudLayer(1.0, 2.0, lwc_g_m3=0.2), CloudLayer(1.5, 2.5, lwc_g_m3=0.1)],
        ...     out / "wc.dat",
        ... )
        Traceback (most recent call last):
            ...
        ValueError: Cloud layers overlap: a layer with top at 2.0 km lies above the base (1.5 km) of the layer above it
        >>> tmp.cleanup()

        Use the file in a run:

        >>> result = ds.pyradtran.run(  # doctest: +SKIP
        ...     config_path=cfg, params={"wc_file": f"1D {path}"}
        ... )
        """
        _warn_ignored(include_zero_layers, altitude_resolution_km)
        return _write_cloud_file(cloud_layers, output_path, "lwc_g_m3", "LWC", 10.0)

    @staticmethod
    def write_ice_cloud_file(
        cloud_layers: List[CloudLayer],
        output_path: Union[str, Path],
        include_zero_layers: Optional[bool] = None,
        altitude_resolution_km: Optional[float] = None,
    ) -> Path:
        r"""Write an ice-cloud ``.dat`` file for libRadtran.

        Same format and row layout as :meth:`write_water_cloud_file`,
        for the layers with ``iwc_g_m3 > 0``.

        Parameters
        ----------
        cloud_layers : list of CloudLayer
            Layers to write, in any order.  Only layers with
            ``iwc_g_m3 > 0`` are used.
        output_path : str or pathlib.Path
            Destination file.  Parent directories are created; an
            existing file is overwritten.
        include_zero_layers, altitude_resolution_km : optional
            Deprecated and ignored (the file is written at the exact
            layer boundaries); passing either emits a
            :class:`DeprecationWarning`.

        Returns
        -------
        pathlib.Path
            *output_path* as a :class:`~pathlib.Path`.

        Raises
        ------
        ValueError
            If two ice layers overlap.

        See Also
        --------
        write_water_cloud_file : File format and row layout.

        Notes
        -----
        Columns: altitude (km), ice water content (g m\ :sup:`-3`) and
        effective radius (µm), highest altitude first, formatted as
        ``%10.6f %10.6f %8.3f``.

        Whether the effective radius of each layer is valid depends on
        the ice parameterisation chosen with ``ic_properties``.  When no
        layer contains ice, a cloud-free placeholder (two zero rows at 1
        and 0 km) is written.

        Examples
        --------
        >>> import pathlib, tempfile
        >>> from pyradtran.clouds import CloudFileWriter, CloudGenerator
        >>> tmp = tempfile.TemporaryDirectory()
        >>> out = pathlib.Path(tmp.name)
        >>> cirrus = CloudGenerator.from_simple_parameters(
        ...     8.0, 9.0, lwc_g_m3=0.0, iwc_g_m3=0.002, r_eff_um=30.0
        ... )
        >>> path = CloudFileWriter.write_ice_cloud_file(cirrus, out / "ic.dat")
        >>> print(path.read_text(), end="")
        #        z        IWC    R_eff
        #       (km)    (g/m^3)     (um)
          9.000000   0.000000   30.000
          8.000000   0.002000   30.000

        Layers without ice give a cloud-free placeholder:

        >>> stratus = CloudGenerator.from_simple_parameters(1.0, 2.0, lwc_g_m3=0.3)
        >>> path = CloudFileWriter.write_ice_cloud_file(stratus, out / "ic.dat")
        >>> print(path.read_text(), end="")
        #        z        IWC    R_eff
        #       (km)    (g/m^3)     (um)
          1.000000   0.000000   30.000
          0.000000   0.000000   30.000
        >>> tmp.cleanup()
        """
        _warn_ignored(include_zero_layers, altitude_resolution_km)
        return _write_cloud_file(cloud_layers, output_path, "iwc_g_m3", "IWC", 30.0)


def generate_cloud_file_from_era5(
    era5_dataset: xr.Dataset,
    output_path: Union[str, Path],
    cloud_type: str = "wc",
    time: Optional[datetime] = None,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    **kwargs,
) -> Path:
    """Write a cloud file from an ERA5 dataset in one call.

    Combines :meth:`CloudGenerator.from_era5_dataset` and the matching
    :class:`CloudFileWriter` method.  The file holds the same cloud as
    the dict of :func:`pyradtran.era5.cloud_profiles`, which
    ``era5_clouds=True`` in
    :meth:`~pyradtran.interface.PyRadtranAccessor.run` uses without a
    file.

    Parameters
    ----------
    era5_dataset : xarray.Dataset
        ERA5 dataset on pressure levels with temperature and cloud
        water content, in any naming flavour
        :func:`pyradtran.era5.normalize_era5` understands.
    output_path : str or pathlib.Path
        Destination ``.dat`` file.
    cloud_type : {"wc", "ic"}, default ``"wc"``
        ``"wc"`` for liquid water cloud, ``"ic"`` for ice cloud
        (case-insensitive).
    time : datetime.datetime, optional
        Time step to select (nearest); the first one when *None*.
    lat, lon : float, optional
        Position of the column (nearest grid point); required when the
        dataset has spatial dimensions.
    **kwargs
        Forwarded to :meth:`CloudGenerator.from_era5_dataset`
        (thresholds, effective radii, variable names, ...).  The legacy
        writer arguments ``include_zero_layers`` and
        ``altitude_resolution_km`` are accepted and ignored.

    Returns
    -------
    pathlib.Path
        *output_path*.

    Raises
    ------
    ValueError
        If *cloud_type* is not ``"wc"`` or ``"ic"`` (checked before any
        work is done), or the dataset lacks a position, a pressure
        coordinate or temperature (see
        :meth:`CloudGenerator.from_era5_dataset`).

    See Also
    --------
    CloudGenerator.from_era5_dataset : Extract layers only (no I/O).
    CloudFileWriter : Write layers to disk separately.
    pyradtran.era5.cloud_profiles : The underlying ERA5 conversion.

    Examples
    --------
    A synthetic column with liquid water at 925 hPa (765 m) and 850 hPa
    (1479 m):

    >>> import pathlib, tempfile
    >>> import numpy as np
    >>> import xarray as xr
    >>> from pyradtran.clouds import generate_cloud_file_from_era5
    >>> dims = ("time", "pressure_level", "latitude", "longitude")
    >>> def column(values):
    ...     return (dims, np.reshape(values, (1, 4, 1, 1)))
    >>> era5 = xr.Dataset(
    ...     {
    ...         "t": column([268.0, 276.0, 280.0, 284.0]),
    ...         "z": column([29500.0, 14500.0, 7500.0, 1000.0]),
    ...         "clwc": column([0.0, 2.0e-4, 3.0e-4, 0.0]),
    ...     },
    ...     coords={
    ...         "time": np.array(["2024-07-15T12"], dtype="datetime64[ns]"),
    ...         "pressure_level": [700.0, 850.0, 925.0, 1000.0],
    ...         "latitude": [78.0],
    ...         "longitude": [15.0],
    ...     },
    ... )
    >>> era5["z"].attrs["units"] = "m2 s-2"
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     path = generate_cloud_file_from_era5(
    ...         era5, pathlib.Path(tmp) / "wc.dat", cloud_type="wc", lat=78.0, lon=15.0
    ...     )
    ...     print(path.read_text(), end="")
    #        z        LWC    R_eff
    #       (km)    (g/m^3)     (um)
      2.243376   0.000000   10.000
      1.121688   0.214571   10.000
      0.433379   0.345251   10.000

    Each ERA5 level fills the layer between the midpoints to its
    neighbours: 0.345 g m\\ :sup:`-3` from 0.433 to 1.122 km and
    0.215 g m\\ :sup:`-3` from there to 2.243 km.
    """
    write = {
        "wc": CloudFileWriter.write_water_cloud_file,
        "ic": CloudFileWriter.write_ice_cloud_file,
    }.get(str(cloud_type).lower())
    if write is None:
        raise ValueError(f"Unknown cloud type: {cloud_type}. Use 'wc' or 'ic'")
    _warn_ignored(
        kwargs.pop("include_zero_layers", None),
        kwargs.pop("altitude_resolution_km", None),
    )
    cloud_layers = CloudGenerator.from_era5_dataset(
        era5_dataset, time=time, lat=lat, lon=lon, **kwargs
    )
    return write(cloud_layers, output_path)
