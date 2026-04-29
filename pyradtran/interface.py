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
...     parameter_overrides={"albedo": 0.85},
... )

See Also
--------
pyradtran.core.Simulation : Low-level single-run engine.
pyradtran.config.load_config : Configuration loading.
"""

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

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

logger = logging.getLogger(__name__)


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

        # Apply parameter overrides if provided
        if parameter_overrides:
            _apply_parameter_overrides(config, parameter_overrides)

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

        # Run the simulation batch
        parsed_outputs = execute_simulation_batch(
            config=config, input_ds=input_ds, parameter_overrides=parameter_overrides
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
    time_var, lat_var, lon_var : str
        Names of core coordinate variables.
    albedo_var : str, optional
        Dataset variable to use as per-point albedo.
    surface_temperature_var : str, optional
        Per-point surface temperature variable.
    surface_type_var : str, optional
        Per-point IGBP surface-type variable (1–20).
    altitude_var : str, optional
        Per-point scalar altitude variable.
    era5_atmosphere : xarray.Dataset, optional
        ERA5 dataset for atmosphere file generation.
    parameter_overrides : dict, optional
        Extra ``key: value`` pairs forwarded to ``uvspec``.
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

        # We need to iterate over all points to generate ERA5 files
        # Note: Optimization possible by finding unique (time, lat, lon) tuples
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

    # Prepare simulation points
    points = []
    for i in range(num_points):
        point_ds = stacked_ds.isel({sample_dim: i})

        t = get_val(point_ds, time_var)
        lat = get_val(point_ds, lat_var)
        lon = get_val(point_ds, lon_var)

        alb = get_val(point_ds, albedo_var)
        surf_temp = get_val(point_ds, surface_temperature_var)
        surf_type = get_val(point_ds, surface_type_var)
        alt = get_val(point_ds, altitude_var)

        # Cloud automation extraction
        point_overrides = parameter_overrides.copy() if parameter_overrides else {}

        # Improve Variational Logic:
        # Check if any parameter override is a reference to a dataset variable
        # This allows varying parameters like 'sza', 'albedo', 'mol_abs_param' etc.
        # by mapping them to dataset dimensions/variables.
        if parameter_overrides:
            for key, val in parameter_overrides.items():
                if isinstance(val, str) and val in point_ds:
                    # It's a reference to a variable!
                    variable_val = get_val(point_ds, val)
                    if variable_val is not None:
                        point_overrides[key] = variable_val
                        # logger.debug(f"Resolved override for '{key}': '{val}' -> {variable_val}")

        try:
            if cloud_wc_var or cloud_ic_var:
                cth = get_val(point_ds, cloud_top_var)
                cbh = get_val(point_ds, cloud_bottom_var)

                # Check validity
                if (
                    cth is not None
                    and cbh is not None
                    and not np.isnan(cth)
                    and not np.isnan(cbh)
                ):
                    # Sort Z descending (uvspec requirement)
                    z_layer = [max(cth, cbh), min(cth, cbh)]

                    # Liquid Cloud
                    if cloud_wc_var:
                        lwc = get_val(point_ds, cloud_wc_var)
                        reff = (
                            get_val(point_ds, cloud_reff_var)
                            if cloud_reff_var
                            else 10.0
                        )  # Default 10um
                        if lwc is not None and not np.isnan(lwc):
                            # Use reff if valid, else default
                            r_val = (
                                reff
                                if (reff is not None and not np.isnan(reff))
                                else 10.0
                            )
                            point_overrides["wc_file"] = {
                                "z": z_layer,
                                "lwc": [float(lwc), float(lwc)],
                                "reff": [float(r_val), float(r_val)],
                            }

                    # Ice Cloud
                    if cloud_ic_var:
                        iwc = get_val(point_ds, cloud_ic_var)
                        # Use ic_reff if provided, else shared reff, else default
                        r_key = (
                            cloud_ic_reff_var if cloud_ic_reff_var else cloud_reff_var
                        )
                        reff_ice = (
                            get_val(point_ds, r_key) if r_key else 20.0
                        )  # Default 20um for ice
                        if iwc is not None and not np.isnan(iwc):
                            r_val = (
                                reff_ice
                                if (reff_ice is not None and not np.isnan(reff_ice))
                                else 20.0
                            )
                            point_overrides["ic_file"] = {
                                "z": z_layer,
                                "iwc": [float(iwc), float(iwc)],
                                "reff": [float(r_val), float(r_val)],
                            }
                else:
                    pass  # Skip cloud if geometry invalid (clear sky fallback?)
        except Exception as e:
            logger.warning(f"Failed to generate cloud parameters for point {i}: {e}")

        # Convert time to proper datetime object
        dt = pd.to_datetime(t).to_pydatetime()

        # Determine Point ID - use index to ensure uniqueness for iteration, but also keep physical ID
        point_id = f"{dt.strftime('%Y%m%d_%H%M%S')}_{lat:.2f}_{lon:.2f}_{i}"

        # ERA5 file lookup uses physical ID components
        era5_key = f"{dt.strftime('%Y%m%d_%H%M%S')}_{lat:.2f}_{lon:.2f}"
        era5_atm_file = (
            era5_atmosphere_files.get(era5_key) if era5_atmosphere_files else None
        )

        # Note: We append point_overrides now
        points.append(
            (
                dt,
                lat,
                lon,
                alb,
                surf_temp,
                surf_type,
                alt,
                era5_atm_file,
                point_id,
                point_overrides,
            )
        )

    # Run simulations in parallel
    results = [None] * num_points  # Pre-allocate results list to preserve order

    # Initialize progress bar
    if HAS_TQDM and show_progress:
        pbar = tqdm(total=num_points, desc="Running simulations", unit="sim")
    else:
        pbar = None

    with ProcessPoolExecutor(max_workers=config.execution.max_workers) as executor:
        # Submit all simulations
        future_to_idx = {}
        for i, point_data in enumerate(points):
            # Unpack the new 10-element tuple (added surf_type)
            (
                dt,
                lat,
                lon,
                alb,
                surf_temp,
                surf_type,
                alt,
                era5_atm_file,
                point_id,
                p_overrides,
            ) = point_data

            # Helper wrapper needs to handle the tuple, but _run_single_simulation_unified signature expects args
            # Actually, executor.submit calls the fn with *args.
            # We need to match _run_single_simulation_unified signature or wrap it.
            # _run_single_simulation_unified(config, point, parameter_overrides) <-- this was designed for point tuple
            # Wait, let's check _run_single_simulation_unified implementation first!
            # It likely unpacks 'point'. If I change 'point' structure errors will occur.

            # Let's adjust how we submit.
            # We should probably pass arguments explicitly to submit,
            # Or assume _run_... handles a point object.

            # Let's Modify the submission to pass p_overrides explicitly INSTEAD of the global one.
            # But point_data needs to be compatible if it's passed as a single arg.
            # Let's see how _run_single_simulation_unified is implemented.
            pass

            # TEMPORARY PLACEHOLDER: I need to verify _run_single_simulation_unified before concluding this edit.
            # But I am inside ReplaceFileContent...
            # I will trust that I can modify _run_single_simulation_unified later if needed,
            # OR I pass p_overrides as the 3rd argument to _run_single_... which is `parameter_overrides`.

            # Current call:
            # _run_single_simulation_unified(config, point, parameter_overrides)
            # If point is (dt, lat, lon, alb, surf, alt, era5, id),
            # and I changed points.append(...) to include override.
            # I should strip override from point before passing to function, and pass it as 3rd arg.

            point_tuple = (
                dt,
                lat,
                lon,
                alb,
                surf_temp,
                surf_type,
                alt,
                era5_atm_file,
                point_id,
            )
            future = executor.submit(
                _run_single_simulation_unified, config, point_tuple, p_overrides
            )
            future_to_idx[future] = i

        # Collect results as they complete
        success_count = 0
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]

            try:
                result = future.result()
                results[idx] = result  # Store in correct position

                if result:
                    success_count += 1
                    logger.debug(
                        f"Simulation {idx + 1}/{num_points} completed successfully"
                    )
                else:
                    logger.warning(
                        f"Simulation {idx + 1}/{num_points} produced no output"
                    )

            except Exception as e:
                logger.error(
                    f"Simulation {idx + 1}/{num_points} failed with error: {str(e)}"
                )

            # Update progress bar
            if pbar:
                pbar.update(1)
                pbar.set_postfix({"Success": success_count, "Total": num_points})

            # Progress callback
            if progress_callback:
                progress_callback(success_count, num_points)

    # Close progress bar
    if pbar:
        pbar.close()

    if success_count == 0:
        raise PyRadtranError("All simulations failed - no valid results produced")

    logger.info(
        f"Batch execution completed: {success_count}/{num_points} simulations successful"
    )
    return results


def _run_single_simulation_unified(
    config: SimulationConfig,
    point_data: Tuple,
    parameter_overrides: Dict[str, Any] = None,
) -> Optional[ParsedOutput]:
    """Execute a single ``uvspec`` run (called by the process pool)."""
    try:
        time, lat, lon, albedo, surf_temp, surf_type, altitude, era5_file, point_id = (
            point_data
        )

        # Initialize simulation
        sim = Simulation(config)

        # Convert datetime to datetime object if needed
        if isinstance(time, datetime):
            dt = time
        elif isinstance(time, (np.datetime64, str)):
            if isinstance(time, np.datetime64):
                dt = pd.to_datetime(time).to_pydatetime()
            else:
                dt = datetime.fromisoformat(time)
        elif isinstance(time, (int, np.integer)):
            # Handle timestamp integers (e.g., from pd.date_range)
            dt = pd.to_datetime(time).to_pydatetime()
        else:
            dt = time

        # Run simulation with parameters
        output_file = sim.run_simulation(
            dt=dt,
            latitude=lat,
            longitude=lon,
            override_albedo=albedo,
            override_surface_temperature=surf_temp,
            override_surface_type=surf_type,
            override_altitude_km=altitude,
            era5_atmosphere_file=era5_file,
            parameter_overrides=parameter_overrides,
        )

        if output_file and output_file.exists():
            # Parse the output
            parser = OutputParser(config, parameter_overrides)
            parsed_output = parser.parse_output_file(output_file)

            # Add point metadata
            parsed_output.metadata.update(
                {
                    "point_id": point_id,
                    "time": dt.isoformat(),
                    "latitude": lat,
                    "longitude": lon,
                    "albedo": albedo,
                    "surface_temperature": surf_temp,
                    "surface_type": surf_type,
                    "altitude": altitude,
                }
            )

            return parsed_output
        else:
            logger.error(f"No output file produced for point {point_id}")
            return None

    except Exception as e:
        logger.error(
            f"Single simulation failed for point {point_data[-1] if len(point_data) > 7 else 'unknown'}: {str(e)}"
        )
        return None


def _apply_parameter_overrides(
    config: SimulationConfig, parameter_overrides: Dict[str, Any]
):
    """Apply dotted-path overrides (e.g. ``simulation_defaults.albedo_value``) to *config*."""
    for key, value in parameter_overrides.items():
        parts = key.split(".")
        if len(parts) == 2:
            # Config parameter override (e.g., "simulation_defaults.albedo_value")
            section, param = parts
            if hasattr(config, section) and hasattr(getattr(config, section), param):
                setattr(getattr(config, section), param, value)
                logger.info(f"Overriding config: {section}.{param} = {value}")
            else:
                logger.warning(f"Unknown config parameter: {section}.{param}")
        else:
            # LibRadtran command override (e.g., "wc_file 1D")
            # These are passed directly to the core simulation
            logger.debug(f"LibRadtran parameter override: {key} {value}")


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
    ...     parameter_overrides={"albedo": 0.85},
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
        parameter_overrides : dict, optional
            Extra ``key: value`` pairs for ``uvspec``.
        time_var, lat_var, lon_var : str
            Coordinate variable names.
        albedo_var : str, optional
            Per-point albedo variable.
        surface_temperature_var : str, optional
            Per-point surface-temperature variable.
        surface_type_var : str, optional
            Per-point IGBP surface-type variable.
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

        # Apply parameter overrides if provided
        if parameter_overrides:
            _apply_parameter_overrides(self._config, parameter_overrides)

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
            # Altitude is a coordinate - use as list of zout levels
            dataset_altitudes = self._obj[alt_var].values
            if len(dataset_altitudes) > 0:
                logger.info(
                    f"Altitude found as coordinate - using {len(dataset_altitudes)} levels for zout: {dataset_altitudes}"
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
