# pyradtran/interface.py
"""
High-level user-facing interface for pyRadtran.

This module provides the three main entry points:

* :class:`PyRadtranAccessor` — xarray accessor registered as
  ``ds.pyradtran``.
* :func:`execute_simulation_batch` — parallel batch driver.
* :func:`run_pyradtran_simulation` — standalone simulation from a file.

Examples
--------
Run all time steps in an xarray dataset:

>>> result = ds.pyradtran.run(
...     config_path="config/my_config.yaml",
...     params={"albedo": 0.85},
... )

See Also
--------
pyradtran.core.Simulation : Low-level single-run engine.
pyradtran.config.load_config : Configuration loading.
"""

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

try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

from .config import SimulationConfig, load_config
from .core import Simulation
from .exceptions import PyRadtranError
from .io import (
    ERA5AtmosphereGenerator,
    InputDataLoader,
    NetCDFSaver,
    OutputParser,
    OutputToXarray,
    ParsedOutput,
)
from .params import ParamResolver, Var

logger = logging.getLogger(__name__)


@dataclass
class SimPoint:
    """One flattened simulation point, fully resolved."""

    index: int
    time: datetime
    latitude: float
    longitude: float
    resolved: Dict[str, Any]
    skipped: List[str] = field(default_factory=list)
    era5_file: Optional[Path] = None
    point_id: str = ""


def _translate_legacy_kwargs(
    params,
    albedo_var,
    surface_temperature_var,
    surface_type_var,
    altitude_var,
    parameter_overrides,
):
    """Map deprecated kwargs onto the unified ``params`` mapping.

    Explicit ``params`` entries always win. Emits a single
    DeprecationWarning if any legacy kwarg is used.
    """
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
        if var_name is not None and key not in params:
            params[key] = Var(var_name)
    for key, value in (parameter_overrides or {}).items():
        if key not in params:
            params[key] = value
    return params


def run_pyradtran_simulation(
    input_file: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    config_path: Optional[Union[str, Path]] = None,
    parameter_overrides: Dict[str, Any] = None,
    max_workers: Optional[int] = None,
) -> Path:
    """Run a full simulation pipeline from a CSV/NetCDF input file.

    Loads the input data, runs ``uvspec`` in parallel for every
    (time, latitude, longitude) point, and saves the results to
    NetCDF.

    Parameters
    ----------
    input_file : str or pathlib.Path
        Path to a ``.csv`` or ``.nc`` file with ``time``, ``latitude``,
        ``longitude`` columns.
    output_path : str or pathlib.Path, optional
        Destination NetCDF.  Auto-generated from the output config when
        *None*.
    config_path : str or pathlib.Path, optional
        YAML configuration file.  Uses package defaults when *None*.
    parameter_overrides : dict, optional
        Extra ``key: value`` pairs for ``uvspec``.
    max_workers : int, optional
        Override the ``execution.max_workers`` config value.

    Returns
    -------
    pathlib.Path
        Path to the written NetCDF file.

    Raises
    ------
    PyRadtranError
        If the simulation pipeline fails.
    """
    try:
        # Load configuration
        config = load_config(config_path)

        # Override max_workers if specified
        if max_workers is not None:
            config.execution.max_workers = max_workers

        # Load input data
        loader = InputDataLoader()
        input_ds = loader.load_simulation_input_data(input_file)

        # Generate output path if not provided
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = (
                Path(config.paths.output_dir)
                / f"{config.output.filename_prefix}_{timestamp}{config.output.filename_suffix}"
            )
        else:
            output_path = Path(output_path)

        # Run the simulation batch. The resolver handles both dotted
        # config-override keys and raw uvspec keywords.
        parsed_outputs = execute_simulation_batch(
            config=config, input_ds=input_ds, params=parameter_overrides
        )

        # Convert to xarray and save results
        if parsed_outputs:
            converter = OutputToXarray()
            result_ds = converter.convert_batch(parsed_outputs, input_ds)

            saver = NetCDFSaver()
            return saver.save_results_to_netcdf(
                data=result_ds,
                output_path=output_path,
                input_ds=input_ds,
                config=config,
                simulation_params=parameter_overrides,
            )
        else:
            raise PyRadtranError("No valid simulation results produced")

    except Exception as e:
        logger.error(f"Simulation failed: {str(e)}")
        raise PyRadtranError(f"Simulation failed: {str(e)}")


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
) -> List[Optional[ParsedOutput]]:
    """Run ``uvspec`` in parallel for every point in *input_ds*.

    The input dataset is flattened (stacked) over all its dimensions so
    that each combination of coordinates becomes one simulation.  Results
    are returned in the same flat order, ready for
    :meth:`~pyradtran.io.OutputToXarray.convert_batch`.

    Parameters
    ----------
    config : SimulationConfig
        Merged configuration.
    input_ds : xarray.Dataset
        Input coordinates (arbitrary number of dimensions).
    params : dict, optional
        Unified parameter mapping: registry keys / raw uvspec keywords /
        dotted config paths to literal values or :class:`~pyradtran.params.Var`
        per-point dataset references. Preferred over the deprecated
        ``*_var`` and ``parameter_overrides`` kwargs below.
    time_var, lat_var, lon_var : str
        Names of core coordinate variables.
    albedo_var : str, optional
        Deprecated — use ``params={"albedo": Var(...)}``.
    surface_temperature_var : str, optional
        Deprecated — use ``params={"sur_temperature": Var(...)}``.
    surface_type_var : str, optional
        Deprecated — use ``params={"brdf_rpv_type": Var(...)}``.
    altitude_var : str, optional
        Deprecated — use ``params={"zout": Var(...)}``.
    era5_atmosphere : xarray.Dataset, optional
        ERA5 dataset for atmosphere file generation.
    parameter_overrides : dict, optional
        Deprecated — use ``params`` instead.
    progress_callback : callable, optional
        ``callback(current, total)`` invoked after each simulation.
    show_progress : bool, default ``True``
        Show a ``tqdm`` progress bar.  Set to ``False`` to suppress it
        (e.g. when running inside a rendered Jupyter notebook).
    cloud_wc_var, cloud_ic_var : str, optional
        Dataset variables for liquid / ice water content.
    cloud_reff_var, cloud_ic_reff_var : str, optional
        Effective-radius variables.
    cloud_top_var, cloud_bottom_var : str, optional
        Cloud-boundary variables (km).  Required when
        *cloud_wc_var* or *cloud_ic_var* is set.

    Notes
    -----
    When ``execution.max_workers`` is 1, points run serially in-process (no
    process pool); ``None`` or >1 uses a process pool.

    Returns
    -------
    list of ParsedOutput or None
        One entry per flattened input point.  *None* for failed runs.

    Raises
    ------
    PyRadtranError
        If **all** simulations fail.
    """
    # Ensure input_ds is a Dataset
    if isinstance(input_ds, xr.DataArray):
        input_ds = input_ds.to_dataset()

    # Validate cloud variables if enabled
    if cloud_wc_var or cloud_ic_var:
        if not (cloud_top_var and cloud_bottom_var):
            logger.error(
                "Cloud generation enabled but cloud_top_var or cloud_bottom_var missing."
            )
            raise ValueError(
                "Must provide cloud_top_var and cloud_bottom_var when generating clouds."
            )

        required_vars = [
            v
            for v in [
                cloud_wc_var,
                cloud_ic_var,
                cloud_reff_var,
                cloud_ic_reff_var,
                cloud_top_var,
                cloud_bottom_var,
            ]
            if v
        ]
        missing = [v for v in required_vars if v not in input_ds]
        if missing:
            logger.error(f"Missing cloud variables in dataset: {missing}")
            raise ValueError(f"Missing cloud variables in dataset: {missing}")

    # Get non-empty dimensions for stacking
    dims = list(input_ds.sizes.keys())

    # Flatten the dataset to iterate linearly over all combinations
    sample_dim = "sample_batch_dim"
    if dims:
        stacked_ds = input_ds.stack({sample_dim: dims})
    else:
        # Handle scalar dataset (single point)
        stacked_ds = input_ds.expand_dims(sample_dim)

    num_points = stacked_ds.sizes[sample_dim]
    logger.info(
        f"Preparing {num_points} simulations from input dataset with dims {dims}"
    )

    # Helper to safely extract scalar values from 0-d xarray objects
    def get_val(ds, var):
        if var and var in ds:
            val = ds[var].values
            # Unwrap numpy scalars
            if hasattr(val, "item"):
                val = val.item()
            return val
        return None

    # Handle ERA5 atmosphere files if provided
    era5_atmosphere_files = {}
    if era5_atmosphere is not None:
        logger.info("Creating ERA5 atmosphere files for simulation points...")
        # Create working directory for atmosphere files
        atm_dir = config.paths.working_dir / "era5_atmospheres"
        atm_dir.mkdir(parents=True, exist_ok=True)

        era5_generator = ERA5AtmosphereGenerator()

        # Cache: one atmosphere file per unique (time, lat, lon)
        for i in range(num_points):
            point_ds = stacked_ds.isel({sample_dim: i})
            t = get_val(point_ds, time_var)
            lat = get_val(point_ds, lat_var)
            lon = get_val(point_ds, lon_var)

            try:
                dt = pd.to_datetime(t).to_pydatetime()
                point_id = f"{dt.strftime('%Y%m%d_%H%M%S')}_{lat:.2f}_{lon:.2f}"

                # Check if we already generated it for this time point
                if point_id not in era5_atmosphere_files:
                    atm_file = atm_dir / f"era5_atm_{point_id}.dat"
                    # Always regenerate to avoid stale files from previous runs
                    era5_generator.create_era5_atmosphere_file(
                        era5_atmosphere, lat, lon, dt, atm_file
                    )
                    era5_atmosphere_files[point_id] = atm_file
                    logger.debug(
                        f"Created ERA5 atmosphere file for {point_id}: {atm_file}"
                    )

            except Exception as e:
                logger.warning(
                    f"Failed to create ERA5 atmosphere file for point {i}: {e}"
                )
                # We'll continue, and the simulation might fail later or use default

    # Unified params: translate deprecated kwargs, then resolve per point.
    params = _translate_legacy_kwargs(
        params, albedo_var, surface_temperature_var, surface_type_var,
        altitude_var, parameter_overrides,
    )
    resolver = ParamResolver(config, params)
    resolver.validate_var_targets(input_ds)

    points: List[SimPoint] = []
    for i in range(num_points):
        point_ds = stacked_ds.isel({sample_dim: i})
        t = get_val(point_ds, time_var)
        lat = get_val(point_ds, lat_var)
        lon = get_val(point_ds, lon_var)

        resolved, skipped = resolver.resolve_point(point_ds)

        # Cloud automation: build dict-valued wc_file/ic_file entries
        try:
            if cloud_wc_var or cloud_ic_var:
                cth = get_val(point_ds, cloud_top_var)
                cbh = get_val(point_ds, cloud_bottom_var)
                if (
                    cth is not None and cbh is not None
                    and not np.isnan(cth) and not np.isnan(cbh)
                ):
                    z_layer = [max(cth, cbh), min(cth, cbh)]
                    if cloud_wc_var:
                        lwc = get_val(point_ds, cloud_wc_var)
                        reff = (
                            get_val(point_ds, cloud_reff_var)
                            if cloud_reff_var else 10.0
                        )
                        if lwc is not None and not np.isnan(lwc):
                            r_val = (
                                reff
                                if (reff is not None and not np.isnan(reff))
                                else 10.0
                            )
                            resolved["wc_file"] = (
                                {
                                    "z": z_layer,
                                    "lwc": [float(lwc), float(lwc)],
                                    "reff": [float(r_val), float(r_val)],
                                },
                                "dataset-var",
                            )
                    if cloud_ic_var:
                        iwc = get_val(point_ds, cloud_ic_var)
                        r_key = (
                            cloud_ic_reff_var if cloud_ic_reff_var
                            else cloud_reff_var
                        )
                        reff_ice = get_val(point_ds, r_key) if r_key else 20.0
                        if iwc is not None and not np.isnan(iwc):
                            r_val = (
                                reff_ice
                                if (reff_ice is not None and not np.isnan(reff_ice))
                                else 20.0
                            )
                            resolved["ic_file"] = (
                                {
                                    "z": z_layer,
                                    "iwc": [float(iwc), float(iwc)],
                                    "reff": [float(r_val), float(r_val)],
                                },
                                "dataset-var",
                            )
        except Exception as e:
            logger.warning(f"Failed to generate cloud parameters for point {i}: {e}")

        dt = pd.to_datetime(t).to_pydatetime()
        era5_key = f"{dt.strftime('%Y%m%d_%H%M%S')}_{lat:.2f}_{lon:.2f}"
        points.append(
            SimPoint(
                index=i,
                time=dt,
                latitude=lat,
                longitude=lon,
                resolved=resolved,
                skipped=skipped,
                era5_file=(
                    era5_atmosphere_files.get(era5_key)
                    if era5_atmosphere_files else None
                ),
                point_id=f"{era5_key}_{i}",
            )
        )

    # Run simulations in parallel
    results = [None] * num_points  # Pre-allocate results list to preserve order

    # Initialize progress bar
    if HAS_TQDM and show_progress:
        pbar = tqdm(total=num_points, desc="Running simulations", unit="sim")
    else:
        pbar = None

    completed = 0
    success_count = 0

    def _record(idx: int, result: Optional[ParsedOutput]) -> None:
        nonlocal completed, success_count
        completed += 1
        results[idx] = result
        if result:
            success_count += 1
        else:
            logger.warning(f"Simulation {idx + 1}/{num_points} produced no output")
        if pbar:
            pbar.update(1)
            pbar.set_postfix({"Success": success_count, "Total": num_points})
        if progress_callback:
            # B14 fix: report completed count, not success count
            progress_callback(completed, num_points)

    max_workers = config.execution.max_workers
    use_pool = max_workers is None or max_workers > 1
    if use_pool:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(_run_single_simulation_unified, config, point): point.index
                for point in points
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result = future.result()
                except Exception as e:
                    logger.error(
                        f"Simulation {idx + 1}/{num_points} failed with error: {str(e)}"
                    )
                    result = None
                _record(idx, result)
    else:
        # Single-worker runs skip the process pool entirely: no pickling
        # round-trip, and results are available synchronously for callers
        # driving simulations from within an already-parallel context.
        for point in points:
            result = _run_single_simulation_unified(config, point)
            _record(point.index, result)

    # Close progress bar
    if pbar:
        pbar.close()

    if success_count == 0:
        raise PyRadtranError("All simulations failed - no valid results produced")

    logger.info(
        f"Batch execution completed: {success_count}/{num_points} simulations successful"
    )
    return results


def _coerce_datetime(time) -> datetime:
    """Convert any supported time representation to datetime."""
    if isinstance(time, datetime):
        return time
    if isinstance(time, np.datetime64) or isinstance(time, (int, np.integer)):
        return pd.to_datetime(time).to_pydatetime()
    if isinstance(time, str):
        return datetime.fromisoformat(time)
    return time


def _run_single_simulation_unified(
    config: SimulationConfig,
    point: SimPoint,
) -> Optional[ParsedOutput]:
    """Execute a single ``uvspec`` run (called by the process pool, or in-process when max_workers == 1)."""
    try:
        sim = Simulation(config)
        dt = _coerce_datetime(point.time)

        output_file = sim.run_simulation(
            dt=dt,
            latitude=point.latitude,
            longitude=point.longitude,
            resolved_params=point.resolved,
            era5_atmosphere_file=point.era5_file,
        )

        if output_file and output_file.exists():
            raw_overrides = {k: v for k, (v, _p) in point.resolved.items()}
            parser = OutputParser(config, raw_overrides)
            parsed_output = parser.parse_output_file(output_file)
            parsed_output.metadata.update(
                {
                    "point_id": point.point_id,
                    "time": dt.isoformat(),
                    "latitude": point.latitude,
                    "longitude": point.longitude,
                }
            )
            return parsed_output
        logger.error(f"No output file produced for point {point.point_id}")
        return None
    except Exception as e:
        logger.error(f"Single simulation failed for point {point.point_id}: {str(e)}")
        return None


@xr.register_dataset_accessor("pyradtran")
class PyRadtranAccessor:
    """xarray accessor for running libRadtran simulations.

    Registered as ``ds.pyradtran``.  The primary method is :meth:`run`,
    which parallelises ``uvspec`` over every point in the dataset.

    Examples
    --------
    >>> result = ds.pyradtran.run(
    ...     config_path="config/my_config.yaml",
    ...     era5_atmosphere=era5_ds,
    ...     params={"albedo": 0.85},
    ... )

    See Also
    --------
    execute_simulation_batch : The underlying parallel driver.
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
    ) -> Union[xr.Dataset, Path]:
        """
        Run ``uvspec`` for every point in the dataset.

        Parameters
        ----------
        config_path : str or pathlib.Path, optional
            YAML configuration file.
        config : SimulationConfig, optional
            Pre-built config (overrides *config_path*).
        params : dict, optional
            Unified parameter mapping: registry keys / raw uvspec keywords /
            dotted config paths to literal values or
            :class:`~pyradtran.params.Var` per-point dataset references.
            Preferred over the deprecated ``*_var`` and
            ``parameter_overrides`` kwargs below.
        parameter_overrides : dict, optional
            Deprecated — use ``params`` instead.
        time_var, lat_var, lon_var : str
            Coordinate variable names.
        albedo_var : str, optional
            Deprecated — use ``params={"albedo": Var(...)}``.
        surface_temperature_var : str, optional
            Deprecated — use ``params={"sur_temperature": Var(...)}``.
        surface_type_var : str, optional
            Deprecated — use ``params={"brdf_rpv_type": Var(...)}``.
        era5_atmosphere : xarray.Dataset, optional
            ERA5 dataset for custom atmosphere profiles.
        return_dataset : bool, default ``True``
            Return results as an xarray Dataset.
        save_to_file : bool, default ``True``
            Write results to NetCDF.
        output_path : str or pathlib.Path, optional
            Destination file (auto-generated when *None*).
        progress_callback : callable, optional
            ``callback(current, total)``.
        show_progress : bool, default ``True``
            Show a ``tqdm`` progress bar.  Pass ``False`` to suppress it
            (useful when the output will be rendered as HTML).
        cloud_wc_var, cloud_ic_var : str, optional
            LWC / IWC dataset variables.
        cloud_reff_var, cloud_ic_reff_var : str, optional
            Effective-radius variables.
        cloud_top_var, cloud_bottom_var : str, optional
            Cloud geometry variables (km).

        Returns
        -------
        xarray.Dataset or pathlib.Path
            Results dataset when *return_dataset* is True, otherwise
            the output file path.

        Raises
        ------
        PyRadtranError
            If no valid results are produced.
        """
        # Load configuration
        if config:
            self._config = config
        else:
            self._config = load_config(config_path)

        # Validate input dataset
        self._validate_input_dataset(
            time_var,
            lat_var,
            lon_var,
            albedo_var,
            surface_temperature_var,
            surface_type_var,
            era5_atmosphere,
        )

        # Handle altitude information
        alt_var = "altitude"
        altitude_as_data_var = False

        if alt_var in self._obj.dims or alt_var in self._obj.coords:
            dataset_altitudes = np.atleast_1d(self._obj[alt_var].values)
            if dataset_altitudes.size > 0:
                logger.info(
                    f"Altitude found as coordinate - using "
                    f"{dataset_altitudes.size} levels for zout: {dataset_altitudes}"
                )
                self._config.simulation_defaults.output_altitudes_km = [
                    float(alt) for alt in dataset_altitudes
                ]
        if alt_var in self._obj.data_vars:
            # Altitude is a data variable - treat as scalar per time step
            altitude_as_data_var = True
            logger.info(
                "Altitude found as data variable - will be treated as scalar altitude for each time step"
            )

        # Generate output path if saving and not provided
        if save_to_file and output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = (
                Path(self._config.paths.output_dir)
                / f"{self._config.output.filename_prefix}_{timestamp}{self._config.output.filename_suffix}"
            )
            output_path.parent.mkdir(exist_ok=True, parents=True)
            logger.info(f"Auto-generating output path: {output_path}")
        elif output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(exist_ok=True, parents=True)

        # Determine dataset to pass to execution batch
        # If altitude was used as config coordinate, we should NOT iterate over it in the batch execution
        if alt_var in self._obj.dims and not altitude_as_data_var:
            ds_to_execute = self._obj.drop_dims(alt_var)
        else:
            ds_to_execute = self._obj

        # Run the simulation batch

        parsed_outputs = execute_simulation_batch(
            config=self._config,
            input_ds=ds_to_execute,
            params=params,
            time_var=time_var,
            lat_var=lat_var,
            lon_var=lon_var,
            albedo_var=albedo_var,
            surface_temperature_var=surface_temperature_var,
            surface_type_var=surface_type_var,
            altitude_var=alt_var if altitude_as_data_var else None,
            era5_atmosphere=era5_atmosphere,
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
        )

        # Convert to xarray Dataset
        if return_dataset and parsed_outputs:
            converter = OutputToXarray()
            result_ds = converter.convert_batch(
                parsed_outputs, ds_to_execute, time_var, lat_var, lon_var
            )

            # Add metadata
            result_ds.attrs["generated_by"] = "pyradtran"
            result_ds.attrs["pyradtran_version"] = "unified_system"
            result_ds.attrs["generation_date"] = datetime.now().isoformat()

            # Save to file if requested
            if save_to_file and output_path:
                saver = NetCDFSaver()
                saver.save_results_to_netcdf(
                    data=result_ds,
                    output_path=output_path,
                    input_ds=self._obj,
                    config=self._config,
                    simulation_params=parameter_overrides,
                )
                logger.info(f"Results saved to {output_path}")

            return result_ds

        elif save_to_file and parsed_outputs and output_path:
            # Just save to file without returning dataset
            converter = OutputToXarray()
            result_ds = converter.convert_batch(
                parsed_outputs, ds_to_execute, time_var, lat_var, lon_var
            )

            saver = NetCDFSaver()
            return saver.save_results_to_netcdf(
                data=result_ds,
                output_path=output_path,
                input_ds=self._obj,
                config=self._config,
                simulation_params=parameter_overrides,
            )
        else:
            raise PyRadtranError("No valid simulation results to return or save")

    #: Alias for :meth:`run` — kept for backwards compatibility with older
    #: notebooks that call ``ds.pyradtran.run_uvspec(...)``.
    run_uvspec = run

    def inspect_cloud_file(
        self,
        selector: Dict[str, Any] = None,
        parameter_overrides: Dict[str, Any] = None,
        cloud_wc_var: Optional[str] = None,
        cloud_ic_var: Optional[str] = None,
        cloud_reff_var: Optional[str] = None,
        cloud_ic_reff_var: Optional[str] = None,
        cloud_top_var: Optional[str] = None,
        cloud_bottom_var: Optional[str] = None,
    ) -> str:
        """Preview the cloud-profile file that would be generated.

        Parameters
        ----------
        selector : dict, optional
            Passed to ``Dataset.sel()`` to pick a single point.
            Defaults to the first element along every dimension.
        parameter_overrides : dict, optional
        cloud_wc_var, cloud_ic_var, cloud_reff_var : str, optional
        cloud_ic_reff_var, cloud_top_var, cloud_bottom_var : str, optional

        Returns
        -------
        str
            Column-formatted cloud profile, or an explanatory message
            when no cloud can be constructed.
        """
        if selector is None:
            # Default to first point
            point_ds = self._obj.isel({d: 0 for d in self._obj.dims})
        else:
            point_ds = self._obj.sel(selector, method="nearest")

        # Resolve overrides
        point_overrides = parameter_overrides.copy() if parameter_overrides else {}
        if parameter_overrides:
            for key, val in parameter_overrides.items():
                if isinstance(val, str) and val in point_ds:
                    val_scalar = point_ds[val].values
                    if hasattr(val_scalar, "item"):
                        val_scalar = val_scalar.item()
                    # If the variable is still an array (e.g. from sel nearest but dim remains?), squeeze it
                    if hasattr(val_scalar, "ndim") and val_scalar.ndim > 0:
                        val_scalar = (
                            val_scalar.item() if val_scalar.size == 1 else val_scalar
                        )
                    point_overrides[key] = val_scalar

        # Extract variables helper
        def get_val(var):
            if var and var in point_ds:
                val = point_ds[var].values
                if hasattr(val, "item"):
                    val = val.item()
                if hasattr(val, "ndim") and val.ndim > 0:
                    val = val.item() if val.size == 1 else val
                return val
            return None

        # Construct content dict
        content_dict = None

        cth = get_val(cloud_top_var)
        cbh = get_val(cloud_bottom_var)

        if cth is not None and cbh is not None:
            z_layer = [max(cth, cbh), min(cth, cbh)]

            if cloud_wc_var:
                lwc = get_val(cloud_wc_var)
                reff = get_val(cloud_reff_var) if cloud_reff_var else 10.0
                r_val = reff if (reff is not None and not np.isnan(reff)) else 10.0

                content_dict = {
                    "z": z_layer,
                    "lwc": [float(lwc), float(lwc)],
                    "reff": [float(r_val), float(r_val)],
                }

            elif cloud_ic_var:
                iwc = get_val(cloud_ic_var)
                r_key = cloud_ic_reff_var if cloud_ic_reff_var else cloud_reff_var
                reff = get_val(r_key) if r_key else 20.0
                r_val = reff if (reff is not None and not np.isnan(reff)) else 20.0

                content_dict = {
                    "z": z_layer,
                    "iwc": [float(iwc), float(iwc)],
                    "reff": [float(r_val), float(r_val)],
                }

        # Check explicit overrides for dict-based clouds
        if hasattr(point_overrides, "get"):
            if "wc_file" in point_overrides and isinstance(
                point_overrides["wc_file"], dict
            ):
                content_dict = point_overrides["wc_file"]
            if "ic_file" in point_overrides and isinstance(
                point_overrides["ic_file"], dict
            ):
                content_dict = point_overrides["ic_file"]

        if content_dict:
            return Simulation.format_cloud_profile(content_dict)
        else:
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
        """Preview the annotated uvspec input file for one point.

        No simulation is run. Each line is tagged with the layer that
        produced it (``config`` / ``params-literal`` / ``dataset-var`` /
        ``unvalidated``).

        Parameters
        ----------
        point : dict, optional
            ``Dataset.sel()``-style selector (nearest match). Defaults to
            the first element along every dimension.
        params : dict, optional
            Same mapping accepted by :meth:`run`.
        config_path, config
            Configuration source, same as :meth:`run`.

        Returns
        -------
        str
        """
        cfg = config if config is not None else load_config(config_path)
        resolver = ParamResolver(cfg, params)
        resolver.validate_var_targets(self._obj)

        if point is None:
            point_ds = self._obj.isel({d: 0 for d in self._obj.dims})
        else:
            point_ds = self._obj.sel(point, method="nearest")

        def scalar(var):
            if var in point_ds:
                v = point_ds[var].values
                return v.item() if hasattr(v, "item") else v
            return None

        resolved, _skipped = resolver.resolve_point(point_ds)
        dt = pd.to_datetime(scalar(time_var)).to_pydatetime()
        sim = Simulation(cfg)
        return sim.dry_run(
            dt, scalar(lat_var), scalar(lon_var), resolved_params=resolved
        )

    def _validate_input_dataset(
        self,
        time_var: str,
        lat_var: str,
        lon_var: str,
        albedo_var: Optional[str],
        surface_temperature_var: Optional[str],
        surface_type_var: Optional[str],
        era5_atmosphere: Optional[xr.Dataset],
    ):
        """Validate that expected variables exist in the dataset."""
        # Check required variables
        if time_var not in self._obj.dims and time_var not in self._obj.coords:
            raise PyRadtranError(f"Time variable '{time_var}' not found in dataset")

        if (
            lat_var not in self._obj.dims
            and lat_var not in self._obj.coords
            and lat_var not in self._obj.data_vars
        ):
            raise PyRadtranError(f"Latitude variable '{lat_var}' not found in dataset")

        if (
            lon_var not in self._obj.dims
            and lon_var not in self._obj.coords
            and lon_var not in self._obj.data_vars
        ):
            raise PyRadtranError(f"Longitude variable '{lon_var}' not found in dataset")

        # Check optional variables
        if albedo_var and albedo_var not in self._obj:
            raise PyRadtranError(f"Albedo variable '{albedo_var}' not found in dataset")

        if surface_temperature_var and surface_temperature_var not in self._obj:
            raise PyRadtranError(
                f"Surface temperature variable '{surface_temperature_var}' not found in dataset"
            )

        if surface_type_var and surface_type_var not in self._obj:
            raise PyRadtranError(
                f"Surface type variable '{surface_type_var}' not found in dataset"
            )

        # Validate ERA5 atmosphere dataset if provided
        if era5_atmosphere is not None:
            required_era5_vars = ["z", "t", "q"]
            required_era5_coords = ["pressure_level", "valid_time"]

            for var in required_era5_vars:
                if var not in era5_atmosphere.variables:
                    raise PyRadtranError(
                        f"Required variable '{var}' not found in ERA5 atmosphere dataset"
                    )

            for coord in required_era5_coords:
                if coord not in era5_atmosphere.coords:
                    raise PyRadtranError(
                        f"Required coordinate '{coord}' not found in ERA5 atmosphere dataset"
                    )

            n_pressure_levels = era5_atmosphere.sizes.get(
                "pressure_level", era5_atmosphere.pressure_level.size
            )
            logger.info(
                f"ERA5 atmosphere dataset validated with {n_pressure_levels} pressure levels"
            )


# Expose main functions
__all__ = ["run_pyradtran_simulation", "execute_simulation_batch", "PyRadtranAccessor"]
