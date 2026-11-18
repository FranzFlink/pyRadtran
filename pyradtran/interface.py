# pyradtran/interface.py
"""
High-level user-facing interface for pyRadtran.

Describe *where and when* to simulate as an :class:`xarray.Dataset`;
the ``uvspec`` results come back as a dataset with the same dimensions.
Importing :mod:`pyradtran` registers the ``ds.pyradtran`` accessor.

.. autosummary::

   PyRadtranAccessor
   execute_simulation_batch
   run_pyradtran_simulation
   PointOutcome
   SimPoint

* :class:`PyRadtranAccessor`: the xarray accessor (``ds.pyradtran``)
  with :meth:`~PyRadtranAccessor.run`, :meth:`~PyRadtranAccessor.explain`,
  :meth:`~PyRadtranAccessor.jacobian` and
  :meth:`~PyRadtranAccessor.inspect_cloud_file`.
* :func:`execute_simulation_batch`: the batch driver behind
  :meth:`~PyRadtranAccessor.run`; returns parsed outputs, not a dataset.
* :func:`run_pyradtran_simulation`: file in, NetCDF out.

Examples
--------
One simulation per time step, with per-point surface albedo from the
dataset and a fixed aerosol setting:

>>> import pandas as pd
>>> import xarray as xr
>>> import pyradtran
>>> from pyradtran import Var
>>> ds = xr.Dataset(
...     {"snow_albedo": ("time", [0.80, 0.85, 0.90])},
...     coords={
...         "time": pd.date_range("2025-06-21T08:00", periods=3, freq="4h"),
...         "latitude": ("time", [61.0, 61.0, 61.0]),
...         "longitude": ("time", [22.0, 22.0, 22.0]),
...     },
... )
>>> result = ds.pyradtran.run(  # doctest: +SKIP
...     config_path="config/quickstart.yaml",
...     params={"albedo": Var("snow_albedo"), "aerosol_default": ""},
... )
>>> result.eup.dims  # doctest: +SKIP
('time', 'wavelength', 'altitude')

See Also
--------
pyradtran.core.Simulation : Low-level single-run engine.
pyradtran.config.load_config : Configuration loading.
pyradtran.params.Var : Per-point dataset reference for ``params``.
"""

import json
import logging
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import xarray as xr
import yaml

try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

from .channels import convolve_channels
from .config import SimulationConfig, load_config
from .core import Simulation
from .era5 import cloud_profiles, normalize_era5, select_profile, write_atmosphere_file
from .exceptions import PyRadtranError, ValidationError
from .io import (
    InputDataLoader,
    NetCDFSaver,
    OutputParser,
    OutputToXarray,
    ParsedOutput,
)
from .params import CONFIG_FIELD_MAP, PROV_DATASET, ParamResolver, Raw, Var

logger = logging.getLogger(__name__)


def _serialize_params(params: Optional[Dict[str, Any]]) -> str:
    """JSON-encode a params mapping; non-JSON values become their repr."""

    def enc(v):
        if isinstance(v, (Var, Raw)):
            return repr(v)
        if isinstance(v, (str, int, float, bool)) or v is None:
            return v
        if isinstance(v, (list, tuple)):
            return [enc(i) for i in v]
        if isinstance(v, dict):
            return {str(k): enc(val) for k, val in v.items()}
        return repr(v)

    return json.dumps({k: enc(v) for k, v in (params or {}).items()})


def _provenance_attrs(
    config: SimulationConfig,
    params: Optional[Dict[str, Any]],
    input_example: Optional[str] = None,
) -> Dict[str, Any]:
    """Attrs recording what produced a result dataset."""
    from . import __version__

    attrs = {
        "pyradtran_version": __version__,
        "history": (
            f"{datetime.now().isoformat(timespec='seconds')}: "
            f"pyradtran {__version__} ds.pyradtran.run()"
        ),
        "pyradtran_params": _serialize_params(params),
        "pyradtran_libradtran_bin": str(config.paths.libradtran_bin),
    }
    try:
        attrs["pyradtran_config"] = yaml.safe_dump(
            config.to_dict(), default_flow_style=None, sort_keys=False
        )
    except Exception as e:  # config must never break a finished run
        logger.warning(f"Could not serialise config for provenance: {e}")
    if input_example is not None:
        attrs["pyradtran_input_example"] = input_example
    return attrs


@dataclass
class PointOutcome:
    """Result envelope for one simulation point.

    :func:`execute_simulation_batch` with ``return_outcomes=True``
    returns one per flattened input point; the ``status`` variable of a
    :meth:`PyRadtranAccessor.run` result is built from them.

    Parameters
    ----------
    parsed : ParsedOutput or None
        Parsed ``uvspec`` output; *None* unless ``status == 0``.
    status : int
        ``0``: ok; ``1``: failed (``uvspec`` error, timeout, or an
        exception while building/parsing); ``2``: skipped because time,
        latitude or longitude was missing/NaN.
    detail : str, optional
        Failure description: ``uvspec`` stderr (prefixed with the path
        of the kept input file), the exception text, or the skip reason.
        *None* for successful points.
    point_id : str
        ``"<YYYYmmdd_HHMMSS>_<lat>_<lon>_<index>"`` for points that ran,
        ``"point_<index>"`` for skipped ones.

    Examples
    --------
    >>> from pyradtran.interface import PointOutcome
    >>> skipped = PointOutcome(None, 2, "NaN coordinates", "point_7")
    >>> skipped.status, skipped.detail
    (2, 'NaN coordinates')

    Outcomes of a batch (``config`` and ``ds`` as in
    :func:`execute_simulation_batch`):

    >>> outcomes = execute_simulation_batch(  # doctest: +SKIP
    ...     config, ds, return_outcomes=True, show_progress=False
    ... )
    >>> [(o.point_id, o.status) for o in outcomes]  # doctest: +SKIP
    [('20250621_080000_61.00_22.00_0', 0),
     ('20250621_120000_61.00_22.00_1', 0),
     ('20250621_160000_61.00_22.00_2', 0)]
    """

    parsed: Optional[ParsedOutput]
    status: int  # 0 = ok, 1 = uvspec failure, 2 = skipped (NaN inputs)
    detail: Optional[str] = None
    point_id: str = ""


@dataclass
class SimPoint:
    """One flattened simulation point, fully resolved.

    The unit of work for a worker process: everything needed to write
    one ``uvspec`` input file. Built by :func:`execute_simulation_batch`.

    Parameters
    ----------
    index : int
        Position of the point in the flattened (stacked) input dataset.
    time : datetime.datetime
        Simulation time (UTC).
    latitude, longitude : float
        Location in degrees north / east.
    resolved : dict
        ``key -> (value, provenance)`` mapping from
        :meth:`pyradtran.params.ParamResolver.resolve_point`, plus any
        dict-valued ``wc_file`` / ``ic_file`` cloud profiles.
    skipped : list of str
        ``params`` keys dropped for this point because their dataset
        value was NaN (the configuration default applies).
    era5_file : pathlib.Path, optional
        Per-point atmosphere file written from ``era5_atmosphere``.
    point_id : str
        ``"<YYYYmmdd_HHMMSS>_<lat>_<lon>_<index>"``, used in log
        messages and the failure log.
    """

    index: int
    time: datetime
    latitude: float
    longitude: float
    resolved: Dict[str, Any]
    skipped: List[str] = field(default_factory=list)
    era5_file: Optional[Path] = None
    point_id: str = ""


def _scalar(ds: xr.Dataset, var: Optional[str]):
    """Python scalar of the 0-d variable *var* of *ds*; *None* if absent."""
    if var and var in ds:
        return ds[var].values.item()
    return None


def _auto_output_path(config: SimulationConfig) -> Path:
    """``<output_dir>/<filename_prefix>_<YYYYmmdd_HHMMSS><filename_suffix>``."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (
        Path(config.paths.output_dir)
        / f"{config.output.filename_prefix}_{timestamp}{config.output.filename_suffix}"
    )


def _sel_point(ds: xr.Dataset, selector: Dict[str, Any]) -> xr.Dataset:
    """``ds.sel(selector)``: nearest for numeric/time axes, exact for labels."""
    exact = {
        k: v
        for k, v in selector.items()
        if k in ds.indexes and ds.indexes[k].dtype.kind in "OUS"
    }
    nearest = {k: v for k, v in selector.items() if k not in exact}
    ds = ds.sel(exact)
    return ds.sel(nearest, method="nearest") if nearest else ds


def _era5_clouds_requested(era5_clouds) -> bool:
    """True when ERA5 clouds are requested (an empty dict means defaults)."""
    return era5_clouds is not False and era5_clouds is not None


def _point_cloud_profiles(
    get,
    cloud_wc_var,
    cloud_ic_var,
    cloud_reff_var,
    cloud_ic_reff_var,
    cloud_top_var,
    cloud_bottom_var,
):
    """Return ``(wc, ic)`` cloud-profile dicts for one point (*None* if unbuildable).

    *get* maps a dataset variable name to a scalar. Shared by the batch
    driver and :meth:`PyRadtranAccessor.inspect_cloud_file`.
    """

    def valid(v):
        return v is not None and not (isinstance(v, float) and np.isnan(v))

    cth = get(cloud_top_var) if cloud_top_var else None
    cbh = get(cloud_bottom_var) if cloud_bottom_var else None
    if not (valid(cth) and valid(cbh)):
        return None, None
    z_layer = [max(cth, cbh), min(cth, cbh)]

    def layer(content_var, content_key, reff_var, default_reff):
        content = get(content_var) if content_var else None
        if not valid(content):
            return None
        reff = get(reff_var) if reff_var else None
        r_val = float(reff if valid(reff) else default_reff)
        return {
            "z": z_layer,
            content_key: [float(content), float(content)],
            "reff": [r_val, r_val],
        }

    wc = layer(cloud_wc_var, "lwc", cloud_reff_var, 10.0)
    ic = layer(cloud_ic_var, "iwc", cloud_ic_reff_var or cloud_reff_var, 20.0)
    return wc, ic


def _validate_cloud_vars(ds, wc, ic, reff, ic_reff, top, bottom) -> None:
    """Check the ``cloud_*_var`` arguments against *ds* (run and preview)."""
    if not (wc or ic):
        return
    if not (top and bottom):
        raise ValueError(
            "Must provide cloud_top_var and cloud_bottom_var when generating clouds."
        )
    missing = [v for v in (wc, ic, reff, ic_reff, top, bottom) if v and v not in ds]
    if missing:
        raise ValueError(f"Missing cloud variables in dataset: {missing}")


def _translate_legacy_kwargs(
    params,
    albedo_var,
    surface_temperature_var,
    surface_type_var,
    altitude_var,
    parameter_overrides,
):
    """Map deprecated kwargs onto ``params`` (explicit entries win; one warning)."""
    params = dict(params or {})
    legacy_vars = {
        "albedo": albedo_var,
        "sur_temperature": surface_temperature_var,
        "brdf_rpv_type": surface_type_var,
        "zout": altitude_var,
    }
    used_legacy = any(v is not None for v in legacy_vars.values()) or bool(
        parameter_overrides
    )
    if used_legacy:
        warnings.warn(
            "albedo_var/surface_temperature_var/surface_type_var/altitude_var "
            "and parameter_overrides are deprecated; use "
            "params={'albedo': Var('...'), ...} instead",
            DeprecationWarning,
            stacklevel=3,
        )
    for key, var_name in legacy_vars.items():
        if var_name is not None:
            params.setdefault(key, Var(var_name))
    for key, value in (parameter_overrides or {}).items():
        params.setdefault(key, value)
    return params


def run_pyradtran_simulation(
    input_file: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    config_path: Optional[Union[str, Path]] = None,
    params: Optional[Dict[str, Any]] = None,
    parameter_overrides: Dict[str, Any] = None,
    max_workers: Optional[int] = None,
    show_progress: bool = True,
) -> Path:
    """Run a full simulation pipeline from a CSV/NetCDF input file.

    Loads the input points with
    :meth:`pyradtran.io.InputDataLoader.load_simulation_input_data`, runs
    ``uvspec`` for every point with :func:`execute_simulation_batch` and
    writes the results to NetCDF.

    Parameters
    ----------
    input_file : str or pathlib.Path
        ``.csv`` or ``.nc`` file with ``time``, ``latitude`` and
        ``longitude``. *params* can reference further columns with
        :class:`~pyradtran.params.Var`.
    output_path : str or pathlib.Path, optional
        Destination NetCDF. When *None*,
        ``<paths.output_dir>/<output.filename_prefix>_<YYYYmmdd_HHMMSS>``
        ``<output.filename_suffix>`` from the configuration is used.
    config_path : str or pathlib.Path, optional
        Simulation YAML, merged over the package defaults and the master
        config by :func:`~pyradtran.config.load_config`. *None* uses only
        those two layers.
    params : dict, optional
        Parameter mapping, as in :meth:`PyRadtranAccessor.run`.
    parameter_overrides : dict, optional
        Deprecated; use ``params``.
    max_workers : int, optional
        Override the ``execution.max_workers`` config value.
    show_progress : bool, default ``True``
        Show a ``tqdm`` progress bar (if ``tqdm`` is installed).

    Returns
    -------
    pathlib.Path
        Path to the written NetCDF file.

    Raises
    ------
    PyRadtranError
        If loading, running or saving fails (the original exception is
        chained), or if every simulation fails.

    See Also
    --------
    PyRadtranAccessor.run : The same pipeline from an in-memory
        dataset, with ERA5, cloud and channel options.
    execute_simulation_batch : The batch driver used internally.

    Notes
    -----
    Compared with :meth:`PyRadtranAccessor.run`, the written file has no
    per-point ``status`` variable and no ``pyradtran_*`` provenance
    attributes (only the saver's ``config_*`` attributes and
    ``simulation_parameters``), and there are no ERA5 / cloud / channel
    options. For more, open the file and call ``ds.pyradtran.run(...)``.

    Examples
    --------
    A CSV with one row per simulation point::

        time,latitude,longitude
        2025-06-21 08:00:00,61.0,22.0
        2025-06-21 12:00:00,61.0,22.0
        2025-06-21 16:00:00,61.0,22.0

    >>> from pyradtran import run_pyradtran_simulation
    >>> path = run_pyradtran_simulation(  # doctest: +SKIP
    ...     "points.csv",
    ...     output_path="out/from_csv.nc",
    ...     config_path="config/quickstart.yaml",
    ...     params={"wavelength": [400, 700]},
    ...     max_workers=1,
    ... )
    >>> path  # doctest: +SKIP
    PosixPath('out/from_csv.nc')
    >>> xr.open_dataset(path)  # doctest: +SKIP
    <xarray.Dataset> Size: 24kB
    Dimensions:     (time: 3, wavelength: 301, altitude: 1)
    Coordinates:
      * time        (time) datetime64[ns] 24B 2025-06-21T08:00:00 ... 2025-06-21T...
      * wavelength  (wavelength) float64 2kB 400.0 401.0 402.0 ... 698.0 699.0 700.0
      * altitude    (altitude) float64 8B 0.0
    Data variables:
        edir        (time, wavelength, altitude) float64 7kB ...
        edn         (time, wavelength, altitude) float64 7kB ...
        eup         (time, wavelength, altitude) float64 7kB ...
    Attributes: (12/18)
        created_by:                    pyradtran
        ...
    """
    try:
        config = load_config(config_path)
        if max_workers is not None:
            config.execution.max_workers = max_workers

        input_ds = InputDataLoader().load_simulation_input_data(input_file)

        if output_path is None:
            output_path = _auto_output_path(config)
        else:
            output_path = Path(output_path)

        # Run the simulation batch (raises if no point succeeds). The
        # resolver handles both dotted config-override keys and raw uvspec
        # keywords.
        parsed_outputs = execute_simulation_batch(
            config=config,
            input_ds=input_ds,
            params=params,
            parameter_overrides=parameter_overrides,
            show_progress=show_progress,
        )

        result_ds = OutputToXarray().convert_batch(parsed_outputs, input_ds)
        return NetCDFSaver().save_results_to_netcdf(
            data=result_ds,
            output_path=output_path,
            input_ds=input_ds,
            config=config,
            simulation_params=params or parameter_overrides,
        )

    except PyRadtranError:
        raise
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        raise PyRadtranError(f"Simulation failed: {e}") from e


def execute_simulation_batch(
    config: SimulationConfig,
    input_ds: xr.Dataset,
    params: Optional[Dict[str, Any]] = None,
    time_var: str = "time",
    lat_var: str = "latitude",
    lon_var: str = "longitude",
    albedo_var: Optional[str] = None,
    surface_temperature_var: Optional[str] = None,
    surface_type_var: Optional[str] = None,
    altitude_var: Optional[str] = None,
    era5_atmosphere: Optional[xr.Dataset] = None,
    era5_clouds: Union[bool, Dict[str, Any]] = False,
    parameter_overrides: Dict[str, Any] = None,
    progress_callback: Optional[callable] = None,
    # Cloud automation arguments
    cloud_wc_var: Optional[str] = None,
    cloud_ic_var: Optional[str] = None,
    cloud_reff_var: Optional[str] = None,  # For liquid (or shared)
    cloud_ic_reff_var: Optional[str] = None,  # For ice (optional)
    cloud_top_var: Optional[str] = None,
    cloud_bottom_var: Optional[str] = None,
    show_progress: bool = True,
    return_outcomes: bool = False,
    _example: Optional[List[str]] = None,
) -> List[Optional[ParsedOutput]]:
    r"""Run ``uvspec`` for every point in *input_ds*.

    The input dataset is stacked over **all** its dimensions; each
    combination of coordinates is one simulation. Results come back in
    the same flat order, ready for
    :meth:`~pyradtran.io.OutputToXarray.convert_batch`.
    :meth:`PyRadtranAccessor.run` builds on this and adds altitude
    handling, conversion to a dataset, the ``status`` variable, channel
    convolution, provenance and saving.

    Parameters
    ----------
    config : SimulationConfig
        Merged configuration, e.g. from
        :func:`~pyradtran.config.load_config`. Dotted config keys in
        *params* modify it in place; pass ``config.copy()`` to reuse it.
    input_ds : xarray.Dataset
        Input points. Every dimension is iterated; a dataset without
        dimensions (scalar ``time`` / ``latitude`` / ``longitude``) is
        one point. A :class:`xarray.DataArray` is converted with
        ``to_dataset()``.
    params : dict, optional
        Parameter mapping, as in :meth:`PyRadtranAccessor.run`.
        Preferred over the deprecated ``*_var`` and
        ``parameter_overrides`` arguments.
    time_var, lat_var, lon_var : str
        Names of the time / latitude / longitude variables in
        *input_ds* (coordinates or data variables).
    albedo_var : str, optional
        Deprecated; use ``params={"albedo": Var(...)}``.
    surface_temperature_var : str, optional
        Deprecated; use ``params={"sur_temperature": Var(...)}``.
    surface_type_var : str, optional
        Deprecated; use ``params={"brdf_rpv_type": Var(...)}``.
    altitude_var : str, optional
        Deprecated; use ``params={"zout": Var(...)}``.
    era5_atmosphere : xarray.Dataset, optional
        ERA5 dataset for the atmosphere files, in CDS or ARCO-ERA5
        naming (normalised with :func:`pyradtran.era5.normalize_era5`).
        An ozone profile (``o3``) is written into the radiosonde file
        and the config ``ozone_du`` scaling is then skipped.
    era5_clouds : bool or dict, default ``False``
        Also derive per-point ``wc_file`` / ``ic_file`` cloud profiles
        from the ERA5 ``clwc`` / ``ciwc`` fields.  A dict is passed as
        keyword arguments to :func:`pyradtran.era5.cloud_profiles`
        (e.g. ``{"reff_water_um": 8.0}``).  Explicit cloud parameters
        (``params`` or ``cloud_*_var``) take precedence.  Requires
        *era5_atmosphere*.
    parameter_overrides : dict, optional
        Deprecated; use ``params``.
    progress_callback : callable, optional
        ``callback(current, total)`` called after each simulation.
    show_progress : bool, default ``True``
        Show a ``tqdm`` progress bar.
    return_outcomes : bool, default ``False``
        Return :class:`PointOutcome` objects (status and failure detail)
        instead of parsed outputs.
    cloud_wc_var, cloud_ic_var : str, optional
        Dataset variables holding the liquid / ice water content
        (g m\ :sup:`-3`) of a single homogeneous cloud layer per point.
    cloud_reff_var, cloud_ic_reff_var : str, optional
        Effective-radius variables (µm) for liquid and ice. Missing or
        NaN values fall back to 10 µm (liquid) and 20 µm (ice). Without
        *cloud_ic_reff_var* the ice phase uses *cloud_reff_var*.
    cloud_top_var, cloud_bottom_var : str, optional
        Cloud-boundary variables (km).  Required with *cloud_wc_var* or
        *cloud_ic_var*.

    Returns
    -------
    list of ParsedOutput or None
        One entry per flattened input point, in stacking order
        (``input_ds.stack(sample=list(input_ds.sizes))``).  *None* for
        failed or skipped points.  With *return_outcomes*, a list of
        :class:`PointOutcome` instead.

    Raises
    ------
    ValueError
        If *time_var* / *lat_var* / *lon_var* is not in *input_ds*, if
        cloud variables lack a boundary variable or are missing from the
        dataset, or if *era5_clouds* is given without *era5_atmosphere*.
    ValidationError
        If *params* contains an unknown option, an out-of-range literal,
        a :class:`~pyradtran.params.Var` pointing to a missing variable,
        or a per-point value that fails validation.
    PyRadtranError
        If **all** simulations fail.

    See Also
    --------
    PyRadtranAccessor.run : High-level entry point; returns a dataset.
    pyradtran.io.OutputToXarray.convert_batch : Turn the returned list
        into a dataset with the input dimensions.
    pyradtran.core.Simulation : What runs for each point.

    Notes
    -----
    **Per-point handling.** Points with missing/``NaT`` time or NaN
    latitude/longitude are not run (status ``2``). A
    :class:`~pyradtran.params.Var` parameter whose dataset value is NaN
    is *dropped for that point*: the configuration default applies and
    the point still runs.

    **Parallelism.** With ``execution.max_workers`` equal to 1, points run
    serially in the calling process. ``None`` (one worker per CPU) or a
    value > 1 uses a :class:`concurrent.futures.ProcessPoolExecutor`.
    Each ``uvspec`` subprocess is limited by
    ``execution.timeout_seconds``.

    **ERA5.** One atmosphere file per unique (time, latitude, longitude)
    is written to ``<paths.working_dir>/era5_atmospheres/`` and shared by
    all points with these values (e.g. along a sweep dimension). If
    writing fails, a warning is logged and the point runs with the
    configured standard atmosphere.

    **Failures.** When a point fails or is skipped, a
    ``failures_<YYYYmmdd_HHMMSS>.log`` file in ``config.paths.working_dir``
    lists point id, status and detail/stderr per point. The ``.inp``
    file of a failed ``uvspec`` run is kept for post-mortem.

    Examples
    --------
    >>> import pandas as pd
    >>> import xarray as xr
    >>> from pyradtran import execute_simulation_batch, load_config
    >>> ds = xr.Dataset(
    ...     coords={
    ...         "time": pd.date_range("2025-06-21T08:00", periods=3, freq="4h"),
    ...         "latitude": ("time", [61.0, 61.0, 61.0]),
    ...         "longitude": ("time", [22.0, 22.0, 22.0]),
    ...     }
    ... )
    >>> config = load_config("config/quickstart.yaml")  # doctest: +SKIP
    >>> outputs = execute_simulation_batch(  # doctest: +SKIP
    ...     config, ds, params={"wavelength": [500, 502]}, show_progress=False
    ... )
    >>> len(outputs)  # doctest: +SKIP
    3
    >>> outputs[0].data["edir"]  # doctest: +SKIP
    array([1046.489, 1029.557, 1019.107])

    Status per point:

    >>> outcomes = execute_simulation_batch(  # doctest: +SKIP
    ...     config, ds, params={"wavelength": [500, 502]},
    ...     show_progress=False, return_outcomes=True,
    ... )
    >>> outcomes[0].status, outcomes[0].point_id  # doctest: +SKIP
    (0, '20250621_080000_61.00_22.00_0')
    """
    # Ensure input_ds is a Dataset
    if isinstance(input_ds, xr.DataArray):
        input_ds = input_ds.to_dataset()

    # Core coordinate variables must exist; otherwise direct callers get
    # cryptic per-point crashes deep inside the batch loop.
    for label, var in (
        ("time", time_var),
        ("latitude", lat_var),
        ("longitude", lon_var),
    ):
        if var not in input_ds:
            raise ValueError(
                f"{label} variable '{var}' not found in input dataset "
                f"(have: {sorted([*input_ds.coords, *input_ds.data_vars])})"
            )

    _validate_cloud_vars(
        input_ds,
        cloud_wc_var,
        cloud_ic_var,
        cloud_reff_var,
        cloud_ic_reff_var,
        cloud_top_var,
        cloud_bottom_var,
    )

    # Get non-empty dimensions for stacking
    dims = list(input_ds.sizes.keys())

    # Flatten the dataset to iterate linearly over all combinations
    sample_dim = "sample_batch_dim"
    if dims:
        stacked_ds = input_ds.stack({sample_dim: dims})
    else:
        # Scalar dataset (single point). The explicit index makes the
        # dimension exist even when there are only scalar coordinates.
        stacked_ds = input_ds.expand_dims({sample_dim: [0]})

    num_points = stacked_ds.sizes[sample_dim]
    logger.info(
        f"Preparing {num_points} simulations from input dataset with dims {dims}"
    )

    # Per-point time values, extracted once up front: isel on a stacked
    # MultiIndex crashes outright when the index contains NaT, so NaT
    # points must be identified from the flat array before any isel.
    time_flat = np.asarray(stacked_ds[time_var].values).reshape(-1)
    if time_flat.size == num_points:
        bad_time = pd.isna(time_flat)
    else:  # unusual layout; fall back to per-point checks
        bad_time = np.zeros(num_points, dtype=bool)

    # Unified params: translate deprecated kwargs, then resolve per point.
    # Before the ERA5 block: dotted paths.* overrides must apply to atm_dir.
    params = _translate_legacy_kwargs(
        params,
        albedo_var,
        surface_temperature_var,
        surface_type_var,
        altitude_var,
        parameter_overrides,
    )
    resolver = ParamResolver(config, params)
    resolver.validate_var_targets(input_ds)

    # Handle ERA5 atmosphere (and optionally cloud) files if provided
    era5_clouds_on = _era5_clouds_requested(era5_clouds)
    if era5_clouds_on and era5_atmosphere is None:
        raise ValueError("era5_clouds requires era5_atmosphere")

    era5_atmosphere_files = {}
    era5_cloud_cache: Dict[str, tuple] = {}
    if era5_atmosphere is not None:
        logger.info("Creating ERA5 atmosphere files for simulation points...")
        era5_atmosphere = normalize_era5(era5_atmosphere)
        cloud_kwargs = era5_clouds if isinstance(era5_clouds, dict) else {}
        # Create working directory for atmosphere files
        atm_dir = config.paths.working_dir / "era5_atmospheres"
        atm_dir.mkdir(parents=True, exist_ok=True)

    points: List[SimPoint] = []
    outcomes: List[Optional[PointOutcome]] = [None] * num_points
    for i in range(num_points):
        if bad_time[i]:
            outcomes[i] = PointOutcome(None, 2, "missing or NaT time", f"point_{i}")
            continue
        point_ds = stacked_ds.isel({sample_dim: i})
        t = _scalar(point_ds, time_var)
        lat = _scalar(point_ds, lat_var)
        lon = _scalar(point_ds, lon_var)

        # One ERA5 atmosphere file (and cloud profile) per unique
        # (time, lat, lon). On failure the point runs without it.
        if era5_atmosphere is not None:
            try:
                dt = pd.to_datetime(t).to_pydatetime()
                era5_key = f"{dt.strftime('%Y%m%d_%H%M%S')}_{lat:.2f}_{lon:.2f}"
                if era5_key not in era5_atmosphere_files:
                    profile = select_profile(era5_atmosphere, lat, lon, dt)
                    atm_file = atm_dir / f"era5_atm_{era5_key}.dat"
                    # Always regenerate to avoid stale files from previous runs
                    write_atmosphere_file(profile, atm_file)
                    era5_atmosphere_files[era5_key] = atm_file
                    if era5_clouds_on:
                        era5_cloud_cache[era5_key] = cloud_profiles(
                            profile, **cloud_kwargs
                        )
                    logger.debug(
                        f"Created ERA5 atmosphere file for {era5_key}: {atm_file}"
                    )
            except Exception as e:
                logger.warning(
                    f"Failed to create ERA5 atmosphere file for point {i}: {e}"
                )

        if t is None or pd.isna(t):
            # fallback for layouts where the flat time array was unusable
            outcomes[i] = PointOutcome(None, 2, "missing or NaT time", f"point_{i}")
            continue
        if any(v is None or (isinstance(v, float) and np.isnan(v)) for v in (lat, lon)):
            outcomes[i] = PointOutcome(None, 2, "NaN coordinates", f"point_{i}")
            continue

        resolved, skipped = resolver.resolve_point(point_ds)

        # Cloud automation: build dict-valued wc_file/ic_file entries
        try:
            if cloud_wc_var or cloud_ic_var:
                wc, ic = _point_cloud_profiles(
                    lambda v: _scalar(point_ds, v),
                    cloud_wc_var,
                    cloud_ic_var,
                    cloud_reff_var,
                    cloud_ic_reff_var,
                    cloud_top_var,
                    cloud_bottom_var,
                )
                if wc is not None:
                    resolved["wc_file"] = (wc, PROV_DATASET)
                if ic is not None:
                    resolved["ic_file"] = (ic, PROV_DATASET)
        except Exception as e:
            logger.warning(f"Failed to generate cloud parameters for point {i}: {e}")

        dt = pd.to_datetime(t).to_pydatetime()
        era5_key = f"{dt.strftime('%Y%m%d_%H%M%S')}_{lat:.2f}_{lon:.2f}"

        # ERA5-derived cloud profiles: fill in only where nothing more
        # explicit (params / cloud_*_var) provided a cloud already.
        if era5_key in era5_cloud_cache:
            era5_wc, era5_ic = era5_cloud_cache[era5_key]
            if era5_wc is not None and "wc_file" not in resolved:
                resolved["wc_file"] = (era5_wc, "era5")
            if era5_ic is not None and "ic_file" not in resolved:
                resolved["ic_file"] = (era5_ic, "era5")

        points.append(
            SimPoint(
                index=i,
                time=dt,
                latitude=lat,
                longitude=lon,
                resolved=resolved,
                skipped=skipped,
                era5_file=era5_atmosphere_files.get(era5_key),
                point_id=f"{era5_key}_{i}",
            )
        )

    if _example is not None and points:
        # Provenance: the annotated input of the first runnable point,
        # including its ERA5 atmosphere and cloud files.
        p0 = points[0]
        try:
            _example.append(
                Simulation(config).dry_run(
                    p0.time, p0.latitude, p0.longitude, p0.resolved, p0.era5_file
                )
            )
        except Exception as e:
            logger.debug(f"Could not render input example for provenance: {e}")

    # Points not in `points` were skipped above and already have an outcome
    completed = num_points - len(points)
    success_count = 0

    pbar = None
    if HAS_TQDM and show_progress:
        pbar = tqdm(total=num_points, desc="Running simulations", unit="sim")
        if completed:
            pbar.update(completed)

    def _record(idx: int, outcome: Optional[PointOutcome]) -> None:
        nonlocal completed, success_count
        completed += 1
        outcomes[idx] = outcome
        if outcome is not None and outcome.status == 0:
            success_count += 1
        else:
            logger.warning(f"Simulation {idx + 1}/{num_points} produced no output")
        if pbar:
            pbar.update(1)
            pbar.set_postfix({"Success": success_count, "Total": num_points})
        if progress_callback:
            # report the completed count, not the success count
            progress_callback(completed, num_points)

    max_workers = config.execution.max_workers
    if max_workers is None or max_workers > 1:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(
                    _run_single_simulation_unified, config, point
                ): point.index
                for point in points
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    outcome = future.result()
                except Exception as e:
                    logger.error(
                        f"Simulation {idx + 1}/{num_points} failed with error: {e}"
                    )
                    outcome = PointOutcome(None, 1, str(e), f"point_{idx}")
                _record(idx, outcome)
    else:
        # Single-worker runs skip the process pool entirely: no pickling
        # round-trip, and results are available synchronously for callers
        # driving simulations from within an already-parallel context.
        for point in points:
            _record(point.index, _run_single_simulation_unified(config, point))

    if pbar:
        pbar.close()

    # Write the failure log before the all-failed raise so it exists even
    # when every simulation in the batch failed.
    failures = [o for o in outcomes if o is not None and o.status != 0]
    if failures:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = Path(config.paths.working_dir) / f"failures_{run_id}.log"
        with open(log_path, "w") as f:
            for o in failures:
                f.write(f"=== point {o.point_id} (status {o.status}) ===\n")
                f.write((o.detail or "no detail") + "\n\n")
        logger.warning(
            f"{len(failures)}/{num_points} simulations failed; "
            f"details in {log_path}"
        )

    if success_count == 0:
        raise PyRadtranError("All simulations failed - no valid results produced")

    logger.info(
        f"Batch execution completed: {success_count}/{num_points} simulations successful"
    )
    if return_outcomes:
        return outcomes
    return [o.parsed if o is not None else None for o in outcomes]


def _output_has_nan(data) -> bool:
    """NaN check over ParsedOutput.data (array or dict of arrays)."""
    try:
        if data is None:
            return False
        if isinstance(data, dict):
            return any(
                np.isnan(np.asarray(v, dtype=float)).any() for v in data.values()
            )
        return bool(np.isnan(np.asarray(data, dtype=float)).any())
    except (TypeError, ValueError):
        return False


def _run_single_simulation_unified(
    config: SimulationConfig,
    point: SimPoint,
) -> PointOutcome:
    """Run one point and parse its output (in a worker or in-process)."""
    try:
        sim = Simulation(config)
        output_file = sim.run_simulation(
            dt=point.time,
            latitude=point.latitude,
            longitude=point.longitude,
            resolved_params=point.resolved,
            era5_atmosphere_file=point.era5_file,
        )

        if output_file and output_file.exists():
            # Mirror the input-file layering: config-level parameter_overrides
            # (layer 1) shape the file too, so the parser must see them,
            # with per-point resolved values winning.
            raw_overrides = {
                **(config.simulation_defaults.parameter_overrides or {}),
                **{k: v for k, (v, _p) in point.resolved.items()},
            }
            parser = OutputParser(config, raw_overrides)
            parsed_output = parser.parse_output_file(output_file)
            if (
                _output_has_nan(parsed_output.data)
                and config.simulation_defaults.rte_solver == "twostr"
                and any(k in point.resolved for k in ("wc_file", "ic_file"))
            ):
                logger.warning(
                    f"Point {point.point_id}: NaN in uvspec output — the "
                    f"twostr solver is numerically unstable for some "
                    f"optically thick cloud layers at individual "
                    f"wavelengths, which poisons integrated output. Use "
                    f"rte_solver disort for cloudy runs."
                )
            parsed_output.metadata.update(
                {
                    "point_id": point.point_id,
                    "time": point.time.isoformat(),
                    "latitude": point.latitude,
                    "longitude": point.longitude,
                }
            )
            if config.execution.cleanup_temp_files:
                # the parsed values are in memory; the raw .out file would
                # otherwise accumulate forever in the working directory
                try:
                    output_file.unlink()
                except OSError:
                    pass
            return PointOutcome(parsed_output, 0, None, point.point_id)
        detail = sim.last_stderr or "no output produced"
        if sim.last_failed_input is not None:
            detail = f"input: {sim.last_failed_input}\n{detail}"
        return PointOutcome(None, 1, detail, point.point_id)
    except Exception as e:
        logger.error(f"Single simulation failed for point {point.point_id}: {e}")
        return PointOutcome(None, 1, str(e), point.point_id)


@xr.register_dataset_accessor("pyradtran")
class PyRadtranAccessor:
    """xarray accessor for running libRadtran simulations (``ds.pyradtran``).

    Registered on every :class:`xarray.Dataset` when :mod:`pyradtran` is
    imported. The dataset describes *where and when* to simulate, one
    ``uvspec`` run per point:

    .. autosummary::

       run
       explain
       jacobian
       inspect_cloud_file

    * :meth:`run`: run ``uvspec`` for every point; results have the
      same dimensions.
    * :meth:`explain`: annotated ``uvspec`` input file for one point,
      without running.
    * :meth:`jacobian`: finite-difference sensitivity to one scalar
      parameter.
    * :meth:`inspect_cloud_file`: preview the cloud-profile file built
      from the ``cloud_*_var`` arguments.

    ``run_uvspec`` is a backwards-compatible alias of :meth:`run`.

    Parameters
    ----------
    xarray_obj : xarray.Dataset
        The dataset the accessor is attached to. Use ``ds.pyradtran``;
        do not construct directly.

    See Also
    --------
    execute_simulation_batch : The underlying batch driver.
    pyradtran.params.Var : Reference a dataset variable from ``params``.
    pyradtran.config.load_config : How the configuration is assembled.

    Examples
    --------
    Three time steps at one location:

    >>> import pandas as pd
    >>> import xarray as xr
    >>> import pyradtran  # registers ds.pyradtran
    >>> ds = xr.Dataset(
    ...     coords={
    ...         "time": pd.date_range("2025-06-21T08:00", periods=3, freq="4h"),
    ...         "latitude": ("time", [61.0, 61.0, 61.0]),
    ...         "longitude": ("time", [22.0, 22.0, 22.0]),
    ...     }
    ... )
    >>> type(ds.pyradtran).__name__
    'PyRadtranAccessor'

    Preview the input file, then run:

    >>> print(ds.pyradtran.explain(config_path="config/quickstart.yaml"))  # doctest: +SKIP
    rte_solver twostr                                                   # config
    mol_abs_param lowtran per_nm                                        # config
    ...
    >>> result = ds.pyradtran.run(config_path="config/quickstart.yaml")  # doctest: +SKIP
    >>> result.edir.dims  # doctest: +SKIP
    ('time', 'wavelength', 'altitude')
    """

    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        self._config = None

    def run(
        self,
        config_path: Optional[Union[str, Path]] = None,
        config: Optional[SimulationConfig] = None,
        params: Optional[Dict[str, Any]] = None,
        parameter_overrides: Dict[str, Any] = None,
        time_var: str = "time",
        lat_var: str = "latitude",
        lon_var: str = "longitude",
        albedo_var: Optional[str] = None,
        surface_temperature_var: Optional[str] = None,
        surface_type_var: Optional[str] = None,
        era5_atmosphere: Optional[xr.Dataset] = None,
        era5_clouds: Union[bool, Dict[str, Any]] = False,
        return_dataset: bool = True,
        save_to_file: bool = True,
        output_path: Optional[Union[str, Path]] = None,
        progress_callback: Optional[callable] = None,
        # Cloud automation arguments
        cloud_wc_var: Optional[str] = None,
        cloud_ic_var: Optional[str] = None,
        cloud_reff_var: Optional[str] = None,
        cloud_ic_reff_var: Optional[str] = None,
        cloud_top_var: Optional[str] = None,
        cloud_bottom_var: Optional[str] = None,
        show_progress: bool = True,
        channels: Optional[xr.DataArray] = None,
        keep_spectral: bool = False,
    ) -> Union[xr.Dataset, Path]:
        r"""Run ``uvspec`` for every point in the dataset.

        Every combination of the dataset's dimensions is one ``uvspec``
        simulation. The result has the same dimensions plus
        ``wavelength`` and ``altitude``.

        **Input dataset.**

        * ``time`` (UTC), ``latitude`` (degrees north) and ``longitude``
          (degrees east) are required; map other names with
          *time_var* / *lat_var* / *lon_var*. ``time`` must be a
          coordinate; latitude and longitude may be data variables.
          Solar runs compute the solar zenith angle and day of year
          from them unless ``sza`` is set.
        * **Every dimension is a simulation axis.** A ``time`` dimension
          with latitude/longitude along it is a track or time series; a
          ``(time, latitude, longitude)`` grid runs every combination;
          any extra dimension (``albedo``, ``sza``, ``cloud_top`` ...) is
          a sweep axis. A dataset without dimensions (scalar ``time``,
          ``latitude`` and ``longitude``) is one point; its result has
          only ``wavelength`` / ``altitude``.
        * Data variables and coordinates are per-point inputs for
          :class:`~pyradtran.params.Var` in *params* and for the
          ``cloud_*_var`` arguments.
        * ``altitude`` (km, fixed name) is special:

          - as a **coordinate**, its unique sorted values are the
            ``zout`` levels of *every* simulation, replacing
            ``simulation_defaults.output_altitudes_km``. As a dimension
            it is not iterated: one run returns all levels.
          - as a **data variable**, it is a per-point output altitude
            (e.g. an aircraft track), short for
            ``params={"zout": Var("altitude")}``. An explicit ``"zout"``
            in *params* wins. The result holds each point's altitude in
            the variable ``zout``.

        Parameters
        ----------
        config_path : str or pathlib.Path, optional
            Simulation YAML, merged over the package defaults and the
            master config (``~/.pyradtran/config.yaml``) by
            :func:`~pyradtran.config.load_config`. With neither
            *config_path* nor *config*, those two layers are used alone.
        config : SimulationConfig, optional
            Pre-built configuration; takes precedence over *config_path*.
            It is copied, so the caller's object is not modified.
        params : dict, optional
            Per-run parameters on top of the configuration. Keys are
            uvspec option names (``"albedo"``, ``"sza"``,
            ``"mol_modify O3"``, ``"aerosol_default"`` ...) or dotted
            config paths (``"simulation_defaults.wavelength_nm"``,
            ``"execution.max_workers"``). Values are

            * a literal, applied to every point; lists/tuples are
              space-joined (``{"wavelength": [400, 700]}``), ``""`` or
              ``True`` gives a bare keyword;
            * :class:`~pyradtran.params.Var` ``("name")``: the value of
              dataset variable or coordinate *name* at each point;
            * :class:`~pyradtran.params.Raw` ``(value)``: a literal
              passed through without validation;
            * for ``"wc_file"`` / ``"ic_file"``, a dict
              ``{"z": [...], "lwc" | "iwc": [...], "reff": [...]}``,
              written to a cloud file (see
              :meth:`~pyradtran.core.Simulation.format_cloud_profile`).

            Literals are validated before anything runs (see
            :func:`pyradtran.describe`). Preferred over the deprecated
            ``*_var`` and ``parameter_overrides`` arguments.
        parameter_overrides : dict, optional
            Deprecated; use ``params``.
        time_var, lat_var, lon_var : str
            Names of the time / latitude / longitude variables. Defaults
            ``"time"``, ``"latitude"``, ``"longitude"``.
        albedo_var : str, optional
            Deprecated; use ``params={"albedo": Var(...)}``.
        surface_temperature_var : str, optional
            Deprecated; use ``params={"sur_temperature": Var(...)}``.
        surface_type_var : str, optional
            Deprecated; use ``params={"brdf_rpv_type": Var(...)}``.
        era5_atmosphere : xarray.Dataset, optional
            ERA5 pressure-level data (temperature and specific humidity,
            optionally ozone and cloud water) used as the atmosphere of
            each point instead of a scaled standard profile. CDS or
            ARCO-ERA5 naming is normalised with
            :func:`pyradtran.era5.normalize_era5`. The nearest column in
            time and space is written to a radiosonde-format file per
            point. Gases in that file are not rescaled: the config
            ``h2o_mm`` is ignored, and so is ``ozone_du`` when the
            dataset has an ``o3`` field.
        era5_clouds : bool or dict, default ``False``
            Also derive per-point cloud profiles (``wc_file`` /
            ``ic_file``) from the ERA5 ``clwc`` / ``ciwc`` fields.  A
            dict (possibly empty) is passed to
            :func:`pyradtran.era5.cloud_profiles` (e.g.
            ``{"reff_water_um": 8.0}``).  Cloud settings in ``params`` or
            ``cloud_*_var`` win.  Requires *era5_atmosphere*.
        return_dataset : bool, default ``True``
            Return the results dataset; with ``False``, the path of the
            saved file.
        save_to_file : bool, default ``True``
            Write the results to NetCDF.
        output_path : str or pathlib.Path, optional
            Destination file; parent directories are created. Default:
            ``<paths.output_dir>/<output.filename_prefix>_``
            ``<YYYYmmdd_HHMMSS><output.filename_suffix>`` is used.
        progress_callback : callable, optional
            ``callback(completed, total)``, called in the main process
            after every finished point.
        cloud_wc_var, cloud_ic_var : str, optional
            Dataset variables with the liquid / ice water content
            (g m\ :sup:`-3`) of one homogeneous cloud layer per point.
        cloud_reff_var, cloud_ic_reff_var : str, optional
            Effective-radius variables (µm) for liquid and ice. Missing
            or NaN values fall back to 10 µm (liquid) / 20 µm (ice).
            Without *cloud_ic_reff_var* the ice phase uses
            *cloud_reff_var*.
        cloud_top_var, cloud_bottom_var : str, optional
            Cloud top / base altitude variables (km). Required with
            *cloud_wc_var* / *cloud_ic_var*. Points where either is NaN
            run cloud-free.
        show_progress : bool, default ``True``
            Show a ``tqdm`` progress bar (if ``tqdm`` is installed).
        channels : xarray.DataArray, optional
            Instrument spectral response functions, dims
            ``(channel, wavelength)``.  For spectral results, every
            spectral variable is SRF-averaged onto a ``channel``
            dimension with :func:`~pyradtran.channels.convolve_channels`;
            the returned (and saved) dataset is in channel space.
        keep_spectral : bool, default ``False``
            With *channels*, also keep the original spectral variables
            under ``<name>_spectral``.

        Returns
        -------
        xarray.Dataset or pathlib.Path
            With ``return_dataset=True`` (default) the results dataset:

            * **Dimensions**: the simulated input dimensions, in input
              order, then ``wavelength`` (nm; absent for
              wavelength-integrated output) and ``altitude`` (km; always
              present). With *channels*, ``channel`` replaces
              ``wavelength``.
            * ``altitude`` **coordinate**: output levels in km (float),
              i.e. the ``zout`` levels of a multi-level run or the one
              level all points share (``0.0`` for the surface default).
              With a per-point output level (``altitude`` data variable,
              ``"zout": Var(...)``) the coordinate is the placeholder
              ``0.0`` (select with ``isel(altitude=0)`` or
              ``squeeze("altitude")``) and the data variable ``zout``
              (km, on the input dimensions) holds each point's level.
              The placeholder also stands for a symbolic level
              (``"zout": "toa"``).
            * **Data variables**: one per ``uvspec`` output column
              (``simulation_defaults.output_columns``, e.g. ``edir``,
              ``edn``, ``eup``), in ``uvspec``'s units. Failed or
              skipped points are NaN. A column named like an input
              coordinate, typically a sweep dimension such as ``sza`` or
              ``albedo``, is returned as ``<name>_out`` (``sza_out``)
              with a logged warning.
            * ``status``: integer flag on the input dimensions:
              ``0`` ok, ``1`` failed (``uvspec`` error or timeout),
              ``2`` skipped (missing time or NaN latitude/longitude).
            * **Attributes**: provenance, i.e. ``pyradtran_version``,
              ``history``, ``pyradtran_params`` (JSON), ``pyradtran_config``
              (the merged configuration as YAML),
              ``pyradtran_libradtran_bin``, ``pyradtran_input_example``
              (annotated input file of the first point) and, with
              *channels*, ``pyradtran_channels`` /
              ``pyradtran_keep_spectral``.

            With ``return_dataset=False``, the path of the NetCDF file
            holding the same dataset.

        Raises
        ------
        PyRadtranError
            If a required variable is missing from the dataset, the ERA5
            dataset lacks temperature / humidity / ``pressure_level``
            (or cloud water with *era5_clouds*), **all** simulations
            fail, or both *return_dataset* and *save_to_file* are false.
        ValidationError
            If *params* has an unknown option, an invalid literal, a
            ``Var`` naming a variable not in the dataset, or a bare
            string equal to a dataset variable name (use ``Var``).
        ValueError
            If cloud variables lack a boundary variable or are missing
            from the dataset.

        Warns
        -----
        DeprecationWarning
            If ``parameter_overrides`` or one of the ``*_var`` surface
            arguments is used.

        See Also
        --------
        explain : Preview the input file for one point, without running.
        jacobian : Sensitivity of the result to one parameter.
        inspect_cloud_file : Preview the ``cloud_*_var`` cloud profile.
        pyradtran.params.Var : Per-point dataset reference.
        pyradtran.describe : Documentation of a single uvspec option.
        pyradtran.channels.convolve_channels : What ``channels=`` applies.
        pyradtran.era5.normalize_era5 : Accepted ERA5 naming.
        execute_simulation_batch : The batch driver behind this method.

        Notes
        -----
        **Layers.** Each point's input file is built from three layers,
        later ones winning:

        1. the *configuration*, merged from package defaults, the master
           config and *config_path* (see
           :func:`~pyradtran.config.load_config`);
        2. *params* literals, the same for every point;
        3. *params* :class:`~pyradtran.params.Var` references, resolved
           per point from the dataset.

        :meth:`explain` shows the result with each line tagged by its
        layer. Dotted keys in *params* edit layer 1 for this call only.

        **Missing values.** A point with missing time or NaN
        latitude/longitude is skipped (``status == 2``). A ``Var``
        parameter that is NaN at a point is dropped *for that point*: the
        configuration value applies, the point runs and reports
        ``status == 0``. Mask such points yourself if you do not want
        the fallback.

        **Failures.** A failing point does not abort the batch: its
        values are NaN, ``status`` is 1, and the ``uvspec`` error and the
        path of the kept ``.inp`` file go to
        ``<paths.working_dir>/failures_<timestamp>.log``.
        :class:`~pyradtran.exceptions.PyRadtranError` is raised only when
        *every* point fails.

        **Parallel execution.** ``execution.max_workers`` simulations run
        concurrently in worker processes (``None``: one per CPU; ``1``:
        serially in the calling process, best for debugging). Change it
        per call with ``params={"execution.max_workers": 8}``. A
        ``uvspec`` run is killed after ``execution.timeout_seconds``.

        **Clouds.** The ``cloud_*_var`` arguments describe one vertically
        homogeneous layer per point, written as a two-level libRadtran
        cloud file (see :meth:`inspect_cloud_file`). For vertically
        resolved clouds pass a profile dict as ``params["wc_file"]`` /
        ``params["ic_file"]`` or use *era5_clouds*. The ``twostr`` solver
        can return NaN for optically thick clouds; use
        ``"rte_solver": "disort"`` for cloudy runs.

        **Files.** Input, output and cloud files are written to
        ``paths.working_dir`` (relative to the current directory if
        relative) and removed after a successful run when
        ``execution.cleanup_temp_files`` is true. ERA5 atmosphere files
        are kept in its ``era5_atmospheres/`` subdirectory.

        Examples
        --------
        These use ``config/quickstart.yaml`` from the Quickstart (solar
        spectrum 350–2500 nm, ``edir`` / ``edn`` / ``eup`` at the
        surface) and need libRadtran.

        >>> import numpy as np
        >>> import pandas as pd
        >>> import xarray as xr
        >>> import pyradtran
        >>> from pyradtran import Var
        >>> ds = xr.Dataset(
        ...     coords={
        ...         "time": pd.date_range("2025-06-21T08:00", periods=3, freq="4h"),
        ...         "latitude": ("time", [61.0, 61.0, 61.0]),
        ...         "longitude": ("time", [22.0, 22.0, 22.0]),
        ...     }
        ... )

        **Minimal run**, one spectrum per time step:

        >>> result = ds.pyradtran.run(config_path="config/quickstart.yaml")  # doctest: +SKIP
        >>> result  # doctest: +SKIP
        <xarray.Dataset> Size: 125kB
        Dimensions:     (time: 3, wavelength: 1565, altitude: 1)
        Coordinates:
          * time        (time) datetime64[ns] 24B 2025-06-21T08:00:00 ... 2025-06-21T...
          * wavelength  (wavelength) float64 13kB 350.0 350.5 ... 2.495e+03 2.5e+03
          * altitude    (altitude) float64 8B 0.0
        Data variables:
            edir        (time, wavelength, altitude) float64 38kB 294.5 311.7 ... 8.855
            edn         (time, wavelength, altitude) float64 38kB 237.8 ... 0.00284
            eup         (time, wavelength, altitude) float64 38kB 92.19 97.21 ... 1.534
            status      (time) int64 24B 0 0 0
        Attributes:
            generated_by:              pyradtran
            generation_date:           2026-11-14T02:32:06.154631
            pyradtran_version:         0.3.0
            history:                   2026-11-14T02:32:06: pyradtran 0.3.0 ds.pyradt...
            pyradtran_params:          {}
            pyradtran_libradtran_bin:  /opt/libRadtran-2.0.6/bin/uvspec
            pyradtran_config:          paths: {libradtran_bin: /opt/libRadtran-2.0.6/...
            pyradtran_input_example:   rte_solver twostr                             ...
        >>> result.edir.sel(wavelength=550).squeeze("altitude").round(1).values  # doctest: +SKIP
        array([1076.5, 1194.7,  565.9])

        **Literal and per-point parameters:**

        >>> ds_alb = ds.assign(snow_albedo=("time", [0.80, 0.85, 0.90]))
        >>> result = ds_alb.pyradtran.run(  # doctest: +SKIP
        ...     config_path="config/quickstart.yaml",
        ...     params={
        ...         "albedo": Var("snow_albedo"),  # per point, from the dataset
        ...         "rte_solver": "disort",        # literal
        ...         "wavelength": [400, 700],      # -> "wavelength 400 700"
        ...         "aerosol_default": "",         # bare keyword
        ...     },
        ...     save_to_file=False,
        ...     show_progress=False,
        ... )
        >>> dict(result.sizes)  # doctest: +SKIP
        {'time': 3, 'wavelength': 301, 'altitude': 1}
        >>> ratio = result.eup / (result.edir + result.edn)  # doctest: +SKIP
        >>> ratio.sel(wavelength=550).squeeze("altitude").round(3).values  # doctest: +SKIP
        array([0.8 , 0.85, 0.9 ])
        >>> result.attrs["pyradtran_params"]  # doctest: +SKIP
        '{"albedo": "Var(name=\'snow_albedo\')", "rte_solver": "disort", "wavelength": [400, 700], "aerosol_default": ""}'

        **Parameter sweep**, 3 albedos × 2 solar zenith angles = 6 runs:

        >>> noon = ds.isel(time=[1])
        >>> sweep = noon.expand_dims(albedo=[0.05, 0.5, 0.9], sza=[30.0, 60.0])
        >>> dict(sweep.sizes)
        {'albedo': 3, 'sza': 2, 'time': 1}
        >>> result = sweep.pyradtran.run(  # doctest: +SKIP
        ...     config_path="config/quickstart.yaml",
        ...     params={
        ...         "albedo": Var("albedo"),
        ...         "sza": Var("sza"),
        ...         "rte_solver": "disort",
        ...         "wavelength": [550, 550],
        ...     },
        ...     save_to_file=False,
        ...     show_progress=False,
        ... )
        >>> result.eup.dims  # doctest: +SKIP
        ('albedo', 'sza', 'time', 'wavelength', 'altitude')
        >>> result.eup.squeeze(["time", "wavelength", "altitude"]).round(1)  # doctest: +SKIP
        <xarray.DataArray 'eup' (albedo: 3, sza: 2)> Size: 48B
        array([[  73.6,   40. ],
               [ 764.1,  415.4],
               [1423.3,  773.8]])
        Coordinates:
          * albedo      (albedo) float64 24B 0.05 0.5 0.9
          * sza         (sza) float64 16B 30.0 60.0
            time        datetime64[ns] 8B 2025-06-21T12:00:00
            wavelength  float64 8B 550.0
            altitude    float64 8B 0.0

        **Output altitudes.** An ``altitude`` *coordinate* sets the levels
        of every run:

        >>> profile = noon.assign_coords(altitude=[0.0, 2.0, 10.0])
        >>> result = profile.pyradtran.run(  # doctest: +SKIP
        ...     config_path="config/quickstart.yaml", save_to_file=False
        ... )
        >>> dict(result.sizes)  # doctest: +SKIP
        {'time': 1, 'wavelength': 1565, 'altitude': 3}
        >>> result.edn.sel(wavelength=550).squeeze("time").round(1).values  # doctest: +SKIP
        array([98.2, 79.1, 28.1])

        An ``altitude`` *data variable* sets one level per point (flight
        track):

        >>> flight = ds.assign(altitude=("time", [0.0, 1.5, 3.0]))
        >>> result = flight.pyradtran.run(  # doctest: +SKIP
        ...     config_path="config/quickstart.yaml", save_to_file=False
        ... )
        >>> dict(result.sizes)  # doctest: +SKIP
        {'time': 3, 'wavelength': 1565, 'altitude': 1}
        >>> result.zout.values  # doctest: +SKIP
        array([0. , 1.5, 3. ])
        >>> result.edn.sel(wavelength=550).squeeze("altitude").round(1).values  # doctest: +SKIP
        array([95.9, 83.7, 61.1])

        **ERA5 atmosphere** (sample file shipped with the book):

        >>> era5 = xr.open_dataset("data/era5_arctic_subset.nc")  # doctest: +SKIP
        >>> era5 = era5.sel(time="2019-06-21")  # doctest: +SKIP
        >>> list(era5.data_vars)  # doctest: +SKIP
        ['temperature', 'specific_humidity', 'sea_ice_cover']
        >>> arctic = xr.Dataset(
        ...     coords={
        ...         "time": pd.date_range("2019-06-21T06:00", periods=3, freq="6h"),
        ...         "latitude": ("time", [78.9, 78.9, 78.9]),
        ...         "longitude": ("time", [11.9, 11.9, 11.9]),
        ...     }
        ... )
        >>> result = arctic.pyradtran.run(  # doctest: +SKIP
        ...     config_path="config/quickstart.yaml",
        ...     era5_atmosphere=era5,
        ...     save_to_file=False,
        ... )
        >>> # direct irradiance in the 940 nm water-vapour band
        >>> result.edir.sel(wavelength=940).squeeze("altitude").round(1).values  # doctest: +SKIP
        array([185.8, 270.1, 142.9])

        Without ``era5_atmosphere`` the same call gives
        ``[272.5, 373.8, 221.2]`` (standard profile scaled to the config
        ``h2o_mm``). With cloud liquid/ice water (``clwc`` / ``ciwc``) in
        the ERA5 dataset, add the clouds:

        >>> result = arctic.pyradtran.run(  # doctest: +SKIP
        ...     config_path="config/quickstart.yaml",
        ...     era5_atmosphere=era5_with_clouds,
        ...     era5_clouds={"reff_water_um": 8.0},  # or simply True
        ...     params={"rte_solver": "disort"},
        ... )

        **Instrument channels** (``wavelength`` becomes ``channel``):

        >>> wl = np.arange(350.0, 2501.0)
        >>> centres = np.array([550.0, 865.0, 1610.0])
        >>> srf = xr.DataArray(
        ...     np.exp(-0.5 * ((wl - centres[:, None]) / 10.0) ** 2),
        ...     dims=("channel", "wavelength"),
        ...     coords={"channel": ["VIS", "NIR", "SWIR"], "wavelength": wl},
        ... )
        >>> result = ds.pyradtran.run(  # doctest: +SKIP
        ...     config_path="config/quickstart.yaml",
        ...     channels=srf,
        ...     save_to_file=False,
        ... )
        >>> dict(result.sizes)  # doctest: +SKIP
        {'time': 3, 'altitude': 1, 'channel': 3}
        >>> result.edir.squeeze("altitude").round(1).values  # doctest: +SKIP
        array([[1055.5,  633.8,  155.7],
               [1171.6,  694.3,  170.7],
               [ 553.9,  370. ,   90.8]])
        >>> result.attrs["pyradtran_channels"]  # doctest: +SKIP
        'VIS NIR SWIR'

        **Clouds from dataset variables**, one liquid layer per point:

        >>> cloudy = ds.assign(
        ...     lwc=("time", [0.05, 0.1, 0.3]),        # g m-3
        ...     reff=("time", [10.0, 10.0, 8.0]),      # µm
        ...     cloud_top=("time", [1.5, 1.5, 1.5]),   # km
        ...     cloud_base=("time", [1.0, 1.0, 1.0]),  # km
        ... )
        >>> result = cloudy.pyradtran.run(  # doctest: +SKIP
        ...     config_path="config/quickstart.yaml",
        ...     params={"rte_solver": "disort", "sza": 50.0, "wavelength": [550, 550]},
        ...     cloud_wc_var="lwc",
        ...     cloud_reff_var="reff",
        ...     cloud_top_var="cloud_top",
        ...     cloud_bottom_var="cloud_base",
        ...     save_to_file=False,
        ... )
        >>> eglo = (result.edir + result.edn).squeeze(["wavelength", "altitude"])  # doctest: +SKIP
        >>> eglo.round(1).values  # clear sky: 1070.6  # doctest: +SKIP
        array([783.5, 626.8, 300. ])

        **Saving.** By default the result is also written to an
        auto-named file in ``paths.output_dir``. Choose the file and
        return only the path:

        >>> path = ds.pyradtran.run(  # doctest: +SKIP
        ...     config_path="config/quickstart.yaml",
        ...     output_path="out/quickstart.nc",
        ...     return_dataset=False,
        ... )
        >>> path  # doctest: +SKIP
        PosixPath('out/quickstart.nc')
        >>> xr.open_dataset(path).status.values  # doctest: +SKIP
        array([0, 0, 0])

        **Failed and skipped points** are flagged. The second point fails
        in ``uvspec`` (sun below the horizon); the third has no latitude:

        >>> bad = ds.assign(sun=("time", [30.0, 95.0, 60.0]))
        >>> bad["latitude"] = ("time", [61.0, 61.0, np.nan])
        >>> result = bad.pyradtran.run(  # doctest: +SKIP
        ...     config_path="config/quickstart.yaml",
        ...     params={"sza": Var("sun")},
        ...     save_to_file=False,
        ... )
        >>> result.status.values  # doctest: +SKIP
        array([0, 1, 2])
        >>> result.edir.sel(wavelength=550).squeeze("altitude").round(1).values  # doctest: +SKIP
        array([1385.1,    nan,    nan])
        >>> ok = result.where(result.status == 0)  # doctest: +SKIP
        """
        # Load configuration.  A caller-supplied config is copied: this
        # method (and dotted config overrides in params) adjusts the
        # config per run and must not mutate the caller's object.
        if config:
            self._config = config.copy()
        else:
            self._config = load_config(config_path)

        # Normalise ERA5 naming once, then validate everything
        if era5_atmosphere is not None:
            era5_atmosphere = normalize_era5(era5_atmosphere)

        # Validate input dataset
        self._validate_input_dataset(
            time_var,
            lat_var,
            lon_var,
            albedo_var,
            surface_temperature_var,
            surface_type_var,
            era5_atmosphere,
            era5_clouds,
        )

        # Altitude coordinate -> zout levels; data variable -> per-point zout
        params, ds_to_execute = self._apply_altitude(self._config, params)

        # Generate output path if saving and not provided
        if save_to_file and output_path is None:
            output_path = _auto_output_path(self._config)
            output_path.parent.mkdir(exist_ok=True, parents=True)
            logger.info(f"Auto-generating output path: {output_path}")
        elif output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(exist_ok=True, parents=True)

        # Run the simulation batch (raises if no point succeeds)
        example: List[str] = []
        outcomes = execute_simulation_batch(
            config=self._config,
            input_ds=ds_to_execute,
            params=params,
            time_var=time_var,
            lat_var=lat_var,
            lon_var=lon_var,
            albedo_var=albedo_var,
            surface_temperature_var=surface_temperature_var,
            surface_type_var=surface_type_var,
            era5_atmosphere=era5_atmosphere,
            era5_clouds=era5_clouds,
            parameter_overrides=parameter_overrides,
            progress_callback=progress_callback,
            # Forward cloud args
            cloud_wc_var=cloud_wc_var,
            cloud_ic_var=cloud_ic_var,
            cloud_reff_var=cloud_reff_var,
            cloud_ic_reff_var=cloud_ic_reff_var,
            cloud_top_var=cloud_top_var,
            cloud_bottom_var=cloud_bottom_var,
            show_progress=show_progress,
            return_outcomes=True,
            _example=example,
        )
        parsed_outputs = [o.parsed if o is not None else None for o in outcomes]

        if not return_dataset and not (save_to_file and output_path):
            raise PyRadtranError(
                "Nothing to do: return_dataset=False and save_to_file=False"
            )

        # Convert to xarray Dataset. The saved file and the returned
        # dataset are built identically: status codes, channel
        # convolution, and provenance attrs in both.
        result_ds = OutputToXarray().convert_batch(
            parsed_outputs, ds_to_execute, time_var, lat_var, lon_var
        )

        # Attach per-point status codes (0 ok / 1 failed / 2 skipped)
        status_flat = np.array([o.status if o is not None else 1 for o in outcomes])
        status_dims = list(ds_to_execute.sizes.keys())
        if status_dims:
            status_stacked = ds_to_execute.stack({"sample_batch_dim": status_dims})
            status_da = xr.DataArray(
                status_flat,
                coords={"sample_batch_dim": status_stacked["sample_batch_dim"]},
                dims=["sample_batch_dim"],
            ).unstack("sample_batch_dim")
        else:
            status_da = xr.DataArray(int(status_flat[0]))
        result_ds["status"] = status_da
        result_ds["status"].attrs["flag_values"] = "0: ok, 1: failed, 2: skipped"

        # Instrument-channel convolution
        if channels is not None:
            result_ds = self._apply_channels(result_ds, channels, keep_spectral)

        # Provenance: record what produced this dataset
        result_ds.attrs["generated_by"] = "pyradtran"
        result_ds.attrs["generation_date"] = datetime.now().isoformat()
        result_ds.attrs.update(
            _provenance_attrs(
                self._config,
                params,
                input_example=example[0] if example else None,
            )
        )
        if channels is not None:
            result_ds.attrs["pyradtran_channels"] = " ".join(
                str(c) for c in channels["channel"].values
            )
            result_ds.attrs["pyradtran_keep_spectral"] = int(keep_spectral)

        saved_path = None
        if save_to_file and output_path:
            saved_path = NetCDFSaver().save_results_to_netcdf(
                data=result_ds,
                output_path=output_path,
                input_ds=self._obj,
                config=self._config,
                simulation_params=params or parameter_overrides,
            )
            logger.info(f"Results saved to {output_path}")

        if return_dataset:
            return result_ds
        return saved_path

    def _apply_altitude(self, config: SimulationConfig, params):
        """Apply the ``altitude`` convention of :meth:`run`; return ``(params, ds)``.

        A coordinate sets ``output_altitudes_km`` (*config* is modified) and
        its dimension is dropped; a data variable becomes ``zout = Var``.
        """
        ds, alt_var = self._obj, "altitude"
        if alt_var in ds.data_vars:
            # Injected as a zout Var in params (not via the deprecated
            # altitude_var kwarg, which would warn for a documented layout).
            if "zout" not in (params or {}):
                params = {**(params or {}), "zout": Var(alt_var)}
            return params, ds
        if alt_var in ds.dims or alt_var in ds.coords:
            levels = np.atleast_1d(ds[alt_var].values)
            if levels.size > 0:
                # sorted+deduped: direct assignment bypasses the dataclass
                # __post_init__ sort, and uvspec rejects unsorted zout
                config.simulation_defaults.output_altitudes_km = sorted(
                    {float(alt) for alt in levels}
                )
            if alt_var in ds.dims:
                ds = ds.drop_dims(alt_var)
        return params, ds

    @staticmethod
    def _render_point(
        config: SimulationConfig,
        ds: xr.Dataset,
        params: Optional[Dict[str, Any]],
        point: Optional[Dict[str, Any]],
        time_var: str,
        lat_var: str,
        lon_var: str,
    ) -> str:
        """Annotated input file of one point of *ds* (first one by default)."""
        resolver = ParamResolver(config, params)
        resolver.validate_var_targets(ds)
        if point is None:
            point_ds = ds.isel({d: 0 for d in ds.dims})
        else:
            point_ds = _sel_point(ds, point)

        resolved, _skipped = resolver.resolve_point(point_ds)
        dt = pd.to_datetime(_scalar(point_ds, time_var)).to_pydatetime()
        return Simulation(config).dry_run(
            dt,
            _scalar(point_ds, lat_var),
            _scalar(point_ds, lon_var),
            resolved_params=resolved,
        )

    @staticmethod
    def _apply_channels(
        result_ds: xr.Dataset,
        channels: xr.DataArray,
        keep_spectral: bool,
    ) -> xr.Dataset:
        """SRF-convolve a spectral result; ``status`` passes through."""
        if "wavelength" not in result_ds.dims:
            logger.warning(
                "channels= given but result has no wavelength dimension; "
                "skipping convolution"
            )
            return result_ds
        status_var = result_ds.get("status")
        result_ds = convolve_channels(
            result_ds.drop_vars("status", errors="ignore"),
            channels,
            keep_spectral=keep_spectral,
        )
        if status_var is not None:
            result_ds["status"] = status_var
        return result_ds

    #: Alias for :meth:`run` — kept for backwards compatibility with older
    #: notebooks that call ``ds.pyradtran.run_uvspec(...)``.
    run_uvspec = run

    def inspect_cloud_file(
        self,
        selector: Dict[str, Any] = None,
        params: Optional[Dict[str, Any]] = None,
        parameter_overrides: Dict[str, Any] = None,
        cloud_wc_var: Optional[str] = None,
        cloud_ic_var: Optional[str] = None,
        cloud_reff_var: Optional[str] = None,
        cloud_ic_reff_var: Optional[str] = None,
        cloud_top_var: Optional[str] = None,
        cloud_bottom_var: Optional[str] = None,
    ) -> str:
        r"""Preview the cloud-profile file :meth:`run` would write for a point.

        The profile is built as in :meth:`run`, from the ``cloud_*_var``
        variables or a dict-valued ``wc_file`` / ``ic_file`` in *params*.
        Nothing is written and ``uvspec`` is not run.

        Parameters
        ----------
        selector : dict, optional
            Passed to ``Dataset.sel(..., method="nearest")`` to pick one
            point, e.g. ``{"time": "2025-06-21T16:00"}``. Default: the
            first element along every dimension.
        params : dict, optional
            Parameter mapping, as in :meth:`run`. Only dict-valued
            ``"wc_file"`` / ``"ic_file"`` entries are used; they replace
            the ``cloud_*_var`` profile of that phase.
            :class:`~pyradtran.params.Var` entries resolve at the
            selected point.
        parameter_overrides : dict, optional
            Deprecated; use ``params``.
        cloud_wc_var, cloud_ic_var : str, optional
            Liquid / ice water content variables (g m\ :sup:`-3`).
        cloud_reff_var, cloud_ic_reff_var : str, optional
            Effective-radius variables (µm), as in :meth:`run`.
        cloud_top_var, cloud_bottom_var : str, optional
            Cloud top / base altitude variables (km).

        Returns
        -------
        str
            One block per phase, headed ``# wc_file profile`` /
            ``# ic_file profile``, with the columns altitude (km), water
            content (g m\ :sup:`-3`) and effective radius (µm), top
            level first. Without a cloud (no variables given, or NaN
            boundaries / water content), the message
            ``"No valid cloud profile generated for this point."``.

        Raises
        ------
        ValueError
            If cloud variables lack a boundary variable or are missing
            from the dataset.

        See Also
        --------
        run : Uses the same profile construction for every point.
        explain : Preview the complete ``uvspec`` input file.
        pyradtran.core.Simulation.format_cloud_profile : The formatter.
        pyradtran.era5.cloud_profiles : Profiles derived from ERA5.

        Notes
        -----
        The ``cloud_*_var`` arguments describe a vertically homogeneous
        layer between cloud base :math:`z_b` and cloud top :math:`z_t`.
        It is written as two rows, :math:`(z_t, w, r_\mathrm{eff})` and
        :math:`(z_b, w, r_\mathrm{eff})`, with the same water content
        :math:`w` on both. Its liquid water path is

        .. math::

            \mathrm{LWP} = w\,(z_t - z_b)

        e.g. 0.3 g m\ :sup:`-3` over 0.5 km gives 150 g m\ :sup:`-2`.
        libRadtran reads the value at a level as the content of the layer
        *above* it, so only the base row carries the cloud; the top row
        closes the layer (its water content is ignored). ERA5 clouds
        (``era5_clouds``) are not previewed.

        Examples
        --------
        >>> import pandas as pd
        >>> import xarray as xr
        >>> import pyradtran
        >>> cloudy = xr.Dataset(
        ...     {
        ...         "lwc": ("time", [0.05, 0.1, 0.3]),        # g m-3
        ...         "reff": ("time", [10.0, 10.0, 8.0]),      # µm
        ...         "cloud_top": ("time", [1.5, 1.5, 1.5]),   # km
        ...         "cloud_base": ("time", [1.0, 1.0, 1.0]),  # km
        ...     },
        ...     coords={
        ...         "time": pd.date_range("2025-06-21T08:00", periods=3, freq="4h"),
        ...         "latitude": ("time", [61.0, 61.0, 61.0]),
        ...         "longitude": ("time", [22.0, 22.0, 22.0]),
        ...     },
        ... )
        >>> cloud_vars = dict(
        ...     cloud_wc_var="lwc",
        ...     cloud_reff_var="reff",
        ...     cloud_top_var="cloud_top",
        ...     cloud_bottom_var="cloud_base",
        ... )

        First point (default), then a selected one:

        >>> print(cloudy.pyradtran.inspect_cloud_file(**cloud_vars))
        # wc_file profile
        1.500000 0.050000 10.000000
        1.000000 0.050000 10.000000
        <BLANKLINE>
        >>> print(
        ...     cloudy.pyradtran.inspect_cloud_file(
        ...         {"time": "2025-06-21T16:00"}, **cloud_vars
        ...     )
        ... )
        # wc_file profile
        1.500000 0.300000 8.000000
        1.000000 0.300000 8.000000
        <BLANKLINE>

        A vertically resolved profile from ``params``:

        >>> profile = {
        ...     "z": [2.0, 1.5, 1.0],
        ...     "lwc": [0.0, 0.3, 0.1],
        ...     "reff": [0.0, 12.0, 8.0],
        ... }
        >>> print(cloudy.pyradtran.inspect_cloud_file(params={"wc_file": profile}))
        # wc_file profile
        2.000000 0.000000 0.000000
        1.500000 0.300000 12.000000
        1.000000 0.100000 8.000000
        <BLANKLINE>

        No cloud information:

        >>> cloudy.pyradtran.inspect_cloud_file()
        'No valid cloud profile generated for this point.'
        """
        _validate_cloud_vars(
            self._obj,
            cloud_wc_var,
            cloud_ic_var,
            cloud_reff_var,
            cloud_ic_reff_var,
            cloud_top_var,
            cloud_bottom_var,
        )
        if selector is None:
            # Default to first point
            point_ds = self._obj.isel({d: 0 for d in self._obj.dims})
        else:
            point_ds = _sel_point(self._obj, selector)

        # Deprecated parameter_overrides: a string naming a dataset
        # variable resolves to its value at the point
        point_overrides = dict(parameter_overrides or {})
        for key, val in point_overrides.items():
            if isinstance(val, str) and val in point_ds:
                point_overrides[key] = _scalar(point_ds, val)

        # Unified params: literals pass through, Var resolves per point
        for key, val in (params or {}).items():
            if not isinstance(val, Var):
                point_overrides[key] = val
            elif val.name in point_ds:
                v = point_ds[val.name].values
                point_overrides[key] = v.item() if v.size == 1 else v

        # Same construction as the batch driver, so the preview matches
        wc, ic = _point_cloud_profiles(
            lambda var: _scalar(point_ds, var),
            cloud_wc_var,
            cloud_ic_var,
            cloud_reff_var,
            cloud_ic_reff_var,
            cloud_top_var,
            cloud_bottom_var,
        )

        # Explicit dict-valued overrides win, per phase
        if isinstance(point_overrides.get("wc_file"), dict):
            wc = point_overrides["wc_file"]
        if isinstance(point_overrides.get("ic_file"), dict):
            ic = point_overrides["ic_file"]

        blocks = []
        if wc:
            blocks.append("# wc_file profile\n" + Simulation.format_cloud_profile(wc))
        if ic:
            blocks.append("# ic_file profile\n" + Simulation.format_cloud_profile(ic))
        if blocks:
            return "\n".join(blocks)
        return "No valid cloud profile generated for this point."

    def explain(
        self,
        point: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        config_path: Optional[Union[str, Path]] = None,
        config: Optional[SimulationConfig] = None,
        time_var: str = "time",
        lat_var: str = "latitude",
        lon_var: str = "longitude",
    ) -> str:
        """Preview the annotated ``uvspec`` input file for one point.

        Returns the input file :meth:`run` would pass to ``uvspec`` for one
        point, each line tagged with the origin of its value. *params* are
        validated as in :meth:`run`; nothing is simulated.

        Parameters
        ----------
        point : dict, optional
            ``Dataset.sel(..., method="nearest")`` selector, e.g.
            ``{"time": "2025-06-21T16:00"}``. Default: the first element
            along every dimension. An ``altitude`` dimension is not a
            simulation axis (see :meth:`run`) and cannot be selected.
        params : dict, optional
            Parameter mapping, as in :meth:`run`.
        config_path : str or pathlib.Path, optional
            Simulation YAML, as in :meth:`run`.
        config : SimulationConfig, optional
            Pre-built configuration, as in :meth:`run` (copied).
        time_var, lat_var, lon_var : str
            Names of the time / latitude / longitude variables.

        Returns
        -------
        str
            The input file, one option per line, each followed by a
            ``# <provenance>`` comment:

            ``config``
                From the merged configuration (package defaults, master
                config, *config_path*), including values derived from it
                and the point, e.g. ``sza`` and ``day_of_year`` computed
                from time and position.
            ``params-literal``
                A literal from *params* that passed validation.
            ``dataset-var``
                A :class:`~pyradtran.params.Var` reference, resolved
                from the dataset at this point.
            ``unvalidated``
                Not checked: a :class:`~pyradtran.params.Raw` value, or
                an option with no libRadtran option schema available.

        Raises
        ------
        ValidationError
            If *params* contains an unknown option or invalid value, or
            a ``Var`` naming a variable not in the dataset.

        See Also
        --------
        run : Run the simulations.
        inspect_cloud_file : Preview a generated cloud file.
        pyradtran.describe : Documentation of a single uvspec option.
        pyradtran.core.Simulation.dry_run : The renderer.

        Notes
        -----
        The preview covers the configuration, *params* and the
        ``altitude`` handling of :meth:`run`. It does not reflect
        ``era5_atmosphere`` / ``era5_clouds`` or the ``cloud_*_var``
        arguments. A dict-valued ``wc_file`` / ``ic_file`` (from *params*
        or a ``cloud_source: parametric`` configuration) appears as
        ``wc_file 1D <generated: 2 levels, 1-2 km>`` instead of the path
        of the temporary file written at run time.

        A run stores the annotated input file of its first point in
        ``result.attrs["pyradtran_input_example"]``.

        Examples
        --------
        >>> import pandas as pd
        >>> import xarray as xr
        >>> import pyradtran
        >>> from pyradtran import Raw, Var
        >>> ds = xr.Dataset(
        ...     {"snow_albedo": ("time", [0.80, 0.85, 0.90])},
        ...     coords={
        ...         "time": pd.date_range("2025-06-21T08:00", periods=3, freq="4h"),
        ...         "latitude": ("time", [61.0, 61.0, 61.0]),
        ...         "longitude": ("time", [22.0, 22.0, 22.0]),
        ...     },
        ... )

        All four provenance tags:

        >>> print(  # doctest: +SKIP
        ...     ds.pyradtran.explain(
        ...         config_path="config/quickstart.yaml",
        ...         params={
        ...             "albedo": Var("snow_albedo"),
        ...             "sza": 60.0,
        ...             "aerosol_default": "",
        ...             "aerosol_visibility": Raw(23.0),
        ...         },
        ...     )
        ... )
        rte_solver twostr                                                   # config
        mol_abs_param lowtran per_nm                                        # config
        data_files_path /opt/libRadtran-2.0.6/data                          # config
        atmosphere_file /opt/libRadtran-2.0.6/data/atmmod/afglus.dat        # config
        mol_modify O3 300.0 DU                                              # config
        mol_modify H2O 2.0 MM                                               # config
        source solar /opt/libRadtran-2.0.6/data/solar_flux/NewGuey2003.dat  # config
        day_of_year 172                                                     # config
        wavelength 350 2500                                                 # config
        output_user lambda edir edn eup                                     # config
        zout 0.0000                                                         # config
        umu 1.0                                                             # config
        sza 60.0                                                            # params-literal
        aerosol_default                                                     # params-literal
        aerosol_visibility 23.0                                             # unvalidated
        albedo 0.8                                                          # dataset-var
        quiet                                                               # config

        Without *params*, ``sza`` is computed for the selected point and
        the albedo comes from the config:

        >>> text = ds.pyradtran.explain(  # doctest: +SKIP
        ...     point={"time": "2025-06-21T16:00"},
        ...     config_path="config/quickstart.yaml",
        ... )
        >>> print(*[ln for ln in text.splitlines() if "sza" in ln or "albedo" in ln], sep="\\n")  # doctest: +SKIP
        sza 65.59                                                           # config
        albedo 0.15                                                         # config

        Invalid parameters raise before anything runs:

        >>> ds.pyradtran.explain(  # doctest: +SKIP
        ...     config_path="config/quickstart.yaml", params={"albedo": 1.5}
        ... )
        Traceback (most recent call last):
            ...
        pyradtran.exceptions.ValidationError: Invalid parameter value(s): 'albedo' must be in [0.0, 1.0], got 1.5
        >>> ds.pyradtran.explain(  # doctest: +SKIP
        ...     config_path="config/quickstart.yaml", params={"albedoo": 0.5}
        ... )
        Traceback (most recent call last):
            ...
        pyradtran.exceptions.ValidationError: Invalid parameter value(s): 'albedoo' is not a known uvspec option of the local libRadtran install; did you mean albedo / albedo_map / albedo_file? (wrap the value in Raw(...) to bypass validation)
        """
        # Copied like in run(): dotted config keys in params and the
        # altitude handling adjust the config for this call only.
        cfg = config.copy() if config is not None else load_config(config_path)
        params, ds = self._apply_altitude(cfg, params)
        if isinstance(point, dict) and "altitude" not in ds.variables:
            # the altitude levels went into zout; they are not a point axis
            point = {k: v for k, v in point.items() if k != "altitude"}
        return self._render_point(cfg, ds, params, point, time_var, lat_var, lon_var)

    def jacobian(
        self,
        param: str,
        delta: float,
        params: Optional[Dict[str, Any]] = None,
        config_path: Optional[Union[str, Path]] = None,
        config: Optional[SimulationConfig] = None,
        **run_kwargs,
    ) -> xr.Dataset:
        r"""Finite-difference sensitivity kernel for one scalar parameter.

        Runs the batch twice, with the base value of *param* and with
        ``param + delta``, and returns the difference quotient of every
        output variable on the dimensions of :meth:`run`.

        Parameters
        ----------
        param : str
            Parameter to perturb, as written in *params* (``"albedo"``,
            ``"sza"``, ``"mol_modify O3"`` ...). Its base value must be a
            single number: a literal in *params*, or one of the
            configuration defaults ``albedo``, ``sur_temperature``,
            ``sza``, ``mol_modify O3`` and ``mol_modify H2O``. A
            :class:`~pyradtran.params.Var` is not allowed.
        delta : float
            Perturbation :math:`\delta` in the units of the parameter.
            Negative values give a backward difference.
        params : dict, optional
            Base parameter mapping, as in :meth:`run`. Other entries may
            be ``Var`` references.
        config_path : str or pathlib.Path, optional
            Simulation YAML, as in :meth:`run`.
        config : SimulationConfig, optional
            Pre-built configuration; takes precedence over *config_path*.
        **run_kwargs
            Forwarded to both :meth:`run` calls (e.g.
            ``show_progress=False``, ``era5_atmosphere=...``).
            ``save_to_file`` defaults to ``False`` here;
            ``return_dataset=False`` is not allowed.

        Returns
        -------
        xarray.Dataset
            Variables and dimensions of a :meth:`run` result, holding
            the derivative with respect to *param* (output units per unit
            of *param*). ``status`` is the larger status of the two runs
            per point; a point that failed in either run is NaN.
            Attributes: ``jacobian_param``,
            ``jacobian_delta``, ``jacobian_base_value`` and the usual
            provenance of the base run.

        Raises
        ------
        ValidationError
            If *param* is a ``Var`` reference, has no base value in
            *params* or the configuration (e.g. ``"sza"`` when it is
            computed from time and position), or has a non-numeric one
            (``"rte_solver"``); also for ``return_dataset=False``.

        See Also
        --------
        run : The simulation that is differentiated.

        Notes
        -----
        For an output :math:`F` and parameter :math:`x` with base value
        :math:`x_0`, the kernel is the forward difference

        .. math::

            K = \frac{F(x_0 + \delta) - F(x_0)}{\delta}
              = \left.\frac{\partial F}{\partial x}\right|_{x_0}
                + \frac{\delta}{2}\,
                  \left.\frac{\partial^2 F}{\partial x^2}\right|_{x_0}
                + \mathcal{O}(\delta^2)

        so the truncation error grows linearly with :math:`\delta`.
        A very small :math:`\delta` does not help: ``uvspec`` prints
        about seven significant digits, so the difference of two outputs
        has a round-off error of order :math:`\varepsilon |F|` with
        :math:`\varepsilon \approx 10^{-7}`, which the division amplifies
        to

        .. math::

            \Delta K_\mathrm{round} \approx \frac{\varepsilon\,|F|}{\delta}

        Choose :math:`\delta` between the two regimes: large enough that
        the outputs change in their first four or five digits, small
        enough that the response is linear. About 1 % of the parameter's
        natural range usually works (0.01 for an albedo, 1 K for a
        temperature, a few DU for ozone). To check, repeat with
        :math:`\delta/2` or :math:`-\delta`: if the kernel changes
        noticeably, the truncation term matters. Monte Carlo solvers
        (``mystic``) add statistical noise to
        :math:`\Delta K_\mathrm{round}` and need a larger :math:`\delta`.

        The cost is two full batches. All other inputs (``Var``
        parameters, clouds, ERA5 profiles) are identical in both runs.

        Examples
        --------
        Surface irradiances against surface albedo at local noon (base
        value 0.15 from the configuration):

        >>> import pandas as pd
        >>> import xarray as xr
        >>> import pyradtran
        >>> noon = xr.Dataset(
        ...     coords={
        ...         "time": [pd.Timestamp("2025-06-21T12:00")],
        ...         "latitude": ("time", [61.0]),
        ...         "longitude": ("time", [22.0]),
        ...     }
        ... )
        >>> kernel = noon.pyradtran.jacobian(  # doctest: +SKIP
        ...     "albedo",
        ...     0.01,
        ...     config_path="config/quickstart.yaml",
        ...     params={"rte_solver": "disort", "wavelength": [400, 700]},
        ...     show_progress=False,
        ... )
        >>> dict(kernel.sizes)  # doctest: +SKIP
        {'time': 1, 'wavelength': 301, 'altitude': 1}
        >>> {k: v for k, v in kernel.attrs.items() if k.startswith("jacobian")}  # doctest: +SKIP
        {'jacobian_param': 'albedo', 'jacobian_delta': 0.01, 'jacobian_base_value': 0.15}

        ``eup`` responds strongly, ``edn`` through multiple reflection
        between surface and atmosphere (more in the blue), ``edir`` not
        at all:

        >>> sel = kernel.sel(wavelength=[450, 550, 650]).squeeze()  # doctest: +SKIP
        >>> sel.eup.round(1).values  # doctest: +SKIP
        array([1439.8, 1307.6, 1071.7])
        >>> sel.edn.round(1).values  # doctest: +SKIP
        array([234.9, 105.7,  46.1])
        >>> sel.edir.round(1).values  # doctest: +SKIP
        array([0., 0., 0.])

        Step size and ``d eup / d albedo`` at 550 nm: round-off noise,
        a plateau, then curvature:

        >>> for delta in (1e-6, 1e-4, 1e-2, 1e-1, 0.5):  # doctest: +SKIP
        ...     k = noon.pyradtran.jacobian(
        ...         "albedo", delta, config_path="config/quickstart.yaml",
        ...         params={"rte_solver": "disort", "wavelength": [550, 550]},
        ...         show_progress=False,
        ...     )
        ...     print(f"{delta:8.0e}  {float(k.eup.squeeze()):8.1f}")
           1e-06    1300.0
           1e-04    1306.0
           1e-02    1307.6
           1e-01    1317.4
           5e-01    1362.3

        Without a configuration default, give the base value in *params*:

        >>> kernel = noon.pyradtran.jacobian(  # doctest: +SKIP
        ...     "sza",
        ...     1.0,
        ...     config_path="config/quickstart.yaml",
        ...     params={"sza": 40.0, "wavelength": [550, 550]},
        ...     show_progress=False,
        ... )
        >>> round(float(kernel.edir.squeeze()), 2)  # per degree  # doctest: +SKIP
        -20.65
        """
        if not run_kwargs.get("return_dataset", True):
            raise ValidationError(
                "jacobian() differences the result datasets of two runs; "
                "return_dataset=False is not supported"
            )
        params = dict(params or {})
        base_value = params.get(param)
        if isinstance(base_value, Var):
            raise ValidationError(
                f"jacobian() cannot perturb '{param}': it is a per-point "
                f"Var reference; supply a scalar literal instead"
            )
        cfg = config if config is not None else load_config(config_path)
        if base_value is None:
            field_name = CONFIG_FIELD_MAP.get(param)
            if field_name is not None:
                base_value = getattr(cfg.simulation_defaults, field_name, None)
        if base_value is None:
            raise ValidationError(
                f"jacobian() needs a base value for '{param}': set it in "
                f"params or in the configuration"
            )

        try:
            base_value = float(base_value)
        except (TypeError, ValueError):
            raise ValidationError(
                f"jacobian() can only perturb a numeric parameter; "
                f"'{param}' has the base value {base_value!r}"
            ) from None

        run_kwargs.setdefault("save_to_file", False)
        base_params = {**params, param: base_value}
        pert_params = {**params, param: base_value + float(delta)}

        base = self.run(config=cfg, params=base_params, **run_kwargs)
        perturbed = self.run(config=cfg, params=pert_params, **run_kwargs)

        # status is a flag, not a physical quantity — keep the worst of the
        # two runs per point instead of differentiating it
        base_status = base.get("status")
        pert_status = perturbed.get("status")
        jac = (
            perturbed.drop_vars("status", errors="ignore")
            - base.drop_vars("status", errors="ignore")
        ) / float(delta)
        if base_status is not None and pert_status is not None:
            jac["status"] = np.maximum(base_status, pert_status)
            jac["status"].attrs["flag_values"] = "0: ok, 1: failed, 2: skipped"
        jac.attrs["jacobian_param"] = param
        jac.attrs["jacobian_delta"] = float(delta)
        jac.attrs["jacobian_base_value"] = base_value
        jac.attrs.update(_provenance_attrs(cfg, base_params))
        return jac

    def _validate_input_dataset(
        self,
        time_var: str,
        lat_var: str,
        lon_var: str,
        albedo_var: Optional[str],
        surface_temperature_var: Optional[str],
        surface_type_var: Optional[str],
        era5_atmosphere: Optional[xr.Dataset],
        era5_clouds: Union[bool, Dict[str, Any]] = False,
    ):
        """Validate that expected variables exist in the dataset."""
        ds = self._obj
        checks = (
            ("Time", time_var, time_var in ds.dims or time_var in ds.coords),
            ("Latitude", lat_var, lat_var in ds.dims or lat_var in ds),
            ("Longitude", lon_var, lon_var in ds.dims or lon_var in ds),
            ("Albedo", albedo_var, not albedo_var or albedo_var in ds),
            (
                "Surface temperature",
                surface_temperature_var,
                not surface_temperature_var or surface_temperature_var in ds,
            ),
            (
                "Surface type",
                surface_type_var,
                not surface_type_var or surface_type_var in ds,
            ),
        )
        for label, var, found in checks:
            if not found:
                raise PyRadtranError(f"{label} variable '{var}' not found in dataset")

        # Validate ERA5 atmosphere dataset if provided (already normalised)
        if era5_atmosphere is not None:
            for var in ("t", "q"):
                if var not in era5_atmosphere.variables:
                    raise PyRadtranError(
                        f"Required variable '{var}' not found in ERA5 atmosphere "
                        f"dataset (accepted aliases are normalised automatically; "
                        f"have: {sorted(era5_atmosphere.data_vars)})"
                    )
            if "pressure_level" not in era5_atmosphere.coords:
                raise PyRadtranError(
                    "Required coordinate 'pressure_level' not found in ERA5 "
                    "atmosphere dataset"
                )
            if _era5_clouds_requested(era5_clouds) and not (
                "clwc" in era5_atmosphere.variables
                or "ciwc" in era5_atmosphere.variables
            ):
                raise PyRadtranError(
                    "era5_clouds requires 'clwc' and/or 'ciwc' in the ERA5 dataset"
                )

            n_pressure_levels = era5_atmosphere.sizes.get(
                "pressure_level", era5_atmosphere.pressure_level.size
            )
            logger.info(
                f"ERA5 atmosphere dataset validated with {n_pressure_levels} pressure levels"
            )
        elif _era5_clouds_requested(era5_clouds):
            raise PyRadtranError("era5_clouds requires era5_atmosphere")


# Expose main functions
__all__ = ["run_pyradtran_simulation", "execute_simulation_batch", "PyRadtranAccessor"]
