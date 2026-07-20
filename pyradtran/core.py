# pyradtran/core.py
"""
Simulation engine for pyRadtran.

This module contains the :class:`Simulation` class, which is the
low-level workhorse behind every ``uvspec`` invocation.  It is
responsible for:

* Generating a complete ``uvspec`` input file from a
  :class:`~pyradtran.config.SimulationConfig` and per-point overrides.
* Spawning the ``uvspec`` subprocess and capturing its stdout.
* Cleaning up temporary files.

Most users should interact with the higher-level
:class:`~pyradtran.interface.PyRadtranAccessor` (``ds.pyradtran.run()``)
rather than calling :class:`Simulation` directly.

See Also
--------
pyradtran.interface : High-level batch interface and xarray accessor.
pyradtran.io.OutputParser : Parse ``uvspec`` output files.
"""

import logging
import os
import subprocess
import tempfile
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from .config import SimulationConfig
from .exceptions import UvspecExecutionError
from .input_builder import InputFileBuilder, calculate_solar_zenith_angle  # noqa: F401
from .params import PROV_CONFIG, PROV_LITERAL, PROV_UNVALIDATED
from .utils import RadiosondeFinder

logger = logging.getLogger(__name__)


class Simulation:
    """Low-level wrapper around a single ``uvspec`` execution.

    Parameters
    ----------
    config : SimulationConfig
        Fully merged configuration object.

    See Also
    --------
    pyradtran.interface.execute_simulation_batch : Parallel driver.
    """

    def __init__(self, config: SimulationConfig):
        """Initialise with a merged :class:`SimulationConfig`."""
        self.config = config
        self.builder = InputFileBuilder(config)
        self.last_stderr: Optional[str] = None
        self.last_failed_input: Optional[Path] = None
        self.radiosonde_finder = (
            RadiosondeFinder(config.paths.radiosonde_base)
            if config.paths.radiosonde_base
            else None
        )

    def _legacy_kwargs_to_resolved(
        self,
        resolved_params,
        override_albedo=None,
        override_surface_temperature=None,
        override_altitude_km=None,
        override_surface_type=None,
        parameter_overrides=None,
    ):
        """Merge deprecated kwargs (and config-level overrides) into a resolved-params dict.

        Layering, later wins:

        1. ``config.simulation_defaults.parameter_overrides`` (PROV_CONFIG)
        2. legacy ``override_*`` kwargs (PROV_LITERAL)
        3. runtime ``parameter_overrides`` dict (PROV_UNVALIDATED)
        4. explicit *resolved_params* (caller-supplied provenance) — wins over all
        """
        legacy = {
            "albedo": override_albedo,
            "sur_temperature": override_surface_temperature,
            "zout": override_altitude_km,
            "brdf_rpv_type": override_surface_type,
        }
        used = [k for k, v in legacy.items() if v is not None]
        if used or parameter_overrides:
            warnings.warn(
                "override_* kwargs and parameter_overrides are deprecated; "
                "use resolved_params / the params mapping instead",
                DeprecationWarning,
                stacklevel=3,
            )

        resolved: Dict[str, Any] = {}

        # Layer 1: config-level escape hatch (no deprecation warning).
        config_overrides = getattr(
            self.config.simulation_defaults, "parameter_overrides", None
        )
        for key, value in (config_overrides or {}).items():
            resolved[key] = (value, PROV_CONFIG)

        # Layer 2: legacy override_* kwargs.
        for key, value in legacy.items():
            if value is None:
                continue
            if isinstance(value, float) and np.isnan(value):
                continue  # B4: NaN never reaches the input file
            resolved[key] = (value, PROV_LITERAL)

        # Layer 3: runtime parameter_overrides dict.
        for key, value in (parameter_overrides or {}).items():
            resolved[key] = (value, PROV_UNVALIDATED)

        # Layer 4: explicit resolved_params — new API always wins.
        resolved.update(resolved_params or {})

        return resolved

    def build_input_lines(
        self,
        dt: datetime,
        latitude: float,
        longitude: float,
        resolved_params=None,
        era5_atmosphere_file: Optional[Path] = None,
        override_albedo: Optional[float] = None,
        override_surface_temperature: Optional[float] = None,
        override_altitude_km: Optional[float] = None,
        override_surface_type: Optional[int] = None,
        parameter_overrides: Dict[str, Any] = None,
    ):
        """Build the provenance-tagged input lines for one point."""
        resolved = self._legacy_kwargs_to_resolved(
            resolved_params,
            override_albedo,
            override_surface_temperature,
            override_altitude_km,
            override_surface_type,
            parameter_overrides,
        )
        radiosonde_path = None
        if (
            self.config.simulation_defaults.h2o_source == "radiosonde"
            and self.radiosonde_finder
        ):
            radiosonde_path = self.radiosonde_finder.find_radiosonde_file(
                dt, latitude, longitude
            )
        return self.builder.build(
            dt,
            latitude,
            longitude,
            resolved=resolved,
            radiosonde_path=radiosonde_path,
            era5_atmosphere_file=era5_atmosphere_file,
        )

    def dry_run(
        self,
        dt: datetime,
        latitude: float,
        longitude: float,
        resolved_params=None,
        era5_atmosphere_file: Optional[Path] = None,
    ) -> str:
        """Render the annotated input file for one point without running uvspec.

        Returns
        -------
        str
            Input-file content, one ``# <provenance>`` comment per line.
        """
        lines = self.build_input_lines(
            dt,
            latitude,
            longitude,
            resolved_params=resolved_params,
            era5_atmosphere_file=era5_atmosphere_file,
        )
        return self.builder.render_annotated(lines)

    def run_simulation(
        self,
        dt: datetime,
        latitude: float,
        longitude: float,
        resolved_params=None,
        era5_atmosphere_file: Optional[Path] = None,
        override_albedo: Optional[float] = None,
        override_surface_temperature: Optional[float] = None,
        override_altitude_km: Optional[float] = None,
        override_surface_type: Optional[int] = None,
        parameter_overrides: Dict[str, Any] = None,
    ) -> Optional[Path]:
        """Run a single ``uvspec`` simulation.

        Parameters
        ----------
        dt : datetime
            Simulation date/time (UTC).
        latitude, longitude : float
            Location in degrees.
        resolved_params : dict, optional
            ``key -> (value, provenance)`` mapping from
            :meth:`pyradtran.params.ParamResolver.resolve_point`.
        era5_atmosphere_file : pathlib.Path, optional
            Custom ERA5 atmosphere file (radiosonde format).
        override_albedo, override_surface_temperature : float, optional
            Deprecated — use *resolved_params*.
        override_altitude_km : float, optional
            Deprecated — use *resolved_params*.
        override_surface_type : int, optional
            Deprecated — use *resolved_params*.
        parameter_overrides : dict, optional
            Deprecated — use *resolved_params*.

        Returns
        -------
        pathlib.Path or None
            Path to the ``uvspec`` output file, or *None* on failure
            (:attr:`last_stderr` then holds the captured stderr).
        """
        self.last_stderr = None
        self.last_failed_input = None
        temp_cloud_files = []
        try:
            resolved = self._legacy_kwargs_to_resolved(
                resolved_params,
                override_albedo,
                override_surface_temperature,
                override_altitude_km,
                override_surface_type,
                parameter_overrides,
            )

            # Dynamic clouds: dict/list values for wc_file / ic_file
            cloud_overrides = {
                k: v
                for k, (v, _p) in resolved.items()
                if k in ("wc_file", "ic_file") and isinstance(v, (dict, list))
            }
            if cloud_overrides:
                new_overrides, files_to_clean = self._handle_dynamic_clouds(
                    cloud_overrides
                )
                temp_cloud_files.extend(files_to_clean)
                for k, v in new_overrides.items():
                    resolved[k] = (v, resolved[k][1])

            lines = self.build_input_lines(
                dt,
                latitude,
                longitude,
                resolved_params=resolved,
                era5_atmosphere_file=era5_atmosphere_file,
            )
            input_content = self.builder.render(lines)

            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".inp",
                delete=False,
                dir=self.config.paths.working_dir,
            ) as inp_file:
                inp_file.write(input_content)
                input_path = Path(inp_file.name)

            output_path = input_path.with_suffix(".out")
            success = self._run_uvspec(input_path, output_path)

            if success and output_path.exists():
                logger.debug(f"Simulation completed successfully: {output_path}")
                if self.config.execution.cleanup_temp_files:
                    input_path.unlink(missing_ok=True)
                return output_path
            # Failure: keep input and output for post-mortem (B12)
            self.last_failed_input = input_path
            logger.error(f"Simulation failed; input kept at {input_path}")
            return None

        except Exception as e:
            logger.error(f"Simulation failed: {str(e)}")
            raise UvspecExecutionError(f"Simulation failed: {str(e)}") from e

        finally:
            # On failure the kept input file references these cloud files —
            # deleting them would break the promised post-mortem.
            keep_for_postmortem = self.last_failed_input is not None
            if self.config.execution.cleanup_temp_files and not keep_for_postmortem:
                for p in temp_cloud_files:
                    try:
                        if os.path.exists(p):
                            os.unlink(p)
                    except Exception as e:
                        logger.warning(f"Failed to cleanup temp file {p}: {e}")

    @staticmethod
    def format_cloud_profile(data: Dict[str, Any]) -> str:
        """Format a cloud-profile dict as a libRadtran column file.

        Parameters
        ----------
        data : dict
            Must contain ``'z'`` (altitude in km) and either ``'lwc'``
            or ``'iwc'`` (water content in g m⁻³), plus ``'reff'``
            (effective radius in µm).

        Returns
        -------
        str
            Multi-line string ready to write to a ``.dat`` file.

        Raises
        ------
        ValueError
            If required keys are missing.
        """
        # Determine columns
        cols = []
        if "z" in data:
            cols.append(data["z"])
        else:
            raise ValueError("Profile data must contain 'z'")

        if "lwc" in data:
            cols.append(data["lwc"])
        elif "iwc" in data:
            cols.append(data["iwc"])
        else:
            raise ValueError("Profile data must contain 'lwc' or 'iwc'")

        if "reff" in data:
            cols.append(data["reff"])
        else:
            raise ValueError("Profile data must contain 'reff'")

        # Format data
        try:
            import numpy as np

            # Ensure all are arrays/lists of same length
            data_matrix = np.column_stack(cols)

            # Format as string
            lines = []
            for row in data_matrix:
                lines.append(f"{row[0]:.6f} {row[1]:.6f} {row[2]:.6f}")
            return "\n".join(lines) + "\n"

        except Exception as e:
            raise RuntimeError(f"Failed to format cloud profile: {e}") from e

    def _handle_dynamic_clouds(
        self, overrides: Dict[str, Any]
    ) -> tuple[Dict[str, Any], list[str]]:
        """Convert dict-valued ``wc_file`` / ``ic_file`` overrides to temp files."""
        new_updates = {}
        cleanup_list = []

        # Helper to write profile
        def write_profile_file(key_prefix, data):
            content = self.format_cloud_profile(data)

            # Create temp file
            output_dir = self.config.paths.working_dir
            if not output_dir.exists():
                output_dir.mkdir(parents=True, exist_ok=True)

            fd, path = tempfile.mkstemp(
                prefix=f"pyradtran_{key_prefix}_", suffix=".dat", dir=output_dir
            )
            with os.fdopen(fd, "w") as f:
                f.write(content)

            return path

        if "wc_file" in overrides and isinstance(overrides["wc_file"], (dict, list)):
            # It's not a path, it's data
            # Check if it's not already a string
            if not isinstance(overrides["wc_file"], str):
                path = write_profile_file("wc", overrides["wc_file"])
                new_updates["wc_file"] = f"1D {path}"
                cleanup_list.append(path)

        if "ic_file" in overrides and isinstance(overrides["ic_file"], (dict, list)):
            if not isinstance(overrides["ic_file"], str):
                path = write_profile_file("ic", overrides["ic_file"])
                new_updates["ic_file"] = f"1D {path}"
                cleanup_list.append(path)

        return new_updates, cleanup_list

    def _generate_input_content(
        self,
        dt: datetime,
        latitude: float,
        longitude: float,
        radiosonde_path: Optional[Path] = None,
        override_albedo: Optional[float] = None,
        override_surface_temperature: Optional[float] = None,
        override_altitude_km: Optional[float] = None,
        override_surface_type: Optional[int] = None,
        era5_atmosphere_file: Optional[Path] = None,
        parameter_overrides: Dict[str, Any] = None,
    ) -> str:
        """Deprecated shim: render the input file via :class:`InputFileBuilder`.

        Kept for backwards compatibility with callers of the old private
        API; use :meth:`build_input_lines` instead.
        """
        warnings.warn(
            "_generate_input_content is deprecated; use build_input_lines",
            DeprecationWarning,
            stacklevel=2,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            resolved = self._legacy_kwargs_to_resolved(
                None,
                override_albedo,
                override_surface_temperature,
                override_altitude_km,
                override_surface_type,
                parameter_overrides,
            )
        lines = self.builder.build(
            dt,
            latitude,
            longitude,
            resolved=resolved,
            radiosonde_path=radiosonde_path,
            era5_atmosphere_file=era5_atmosphere_file,
        )
        return self.builder.render(lines)

    def _run_uvspec(self, input_path: Path, output_path: Path) -> bool:
        """Spawn ``uvspec``, piping *input_path* to stdin and writing stdout to *output_path*."""
        try:
            cmd = [str(self.config.paths.libradtran_bin)]

            logger.debug(f"Running LibRadtran: {' '.join(cmd)}")
            logger.debug(f"Input file: {input_path}")
            logger.debug(f"Output file: {output_path}")

            with open(input_path, "r") as inp_file, open(output_path, "w") as out_file:
                result = subprocess.run(
                    cmd,
                    stdin=inp_file,
                    stdout=out_file,
                    stderr=subprocess.PIPE,
                    timeout=self.config.execution.timeout_seconds,
                    text=True,
                )

            if result.returncode != 0:
                error_msg = result.stderr.strip() if result.stderr else "Unknown error"
                self.last_stderr = error_msg
                logger.error(
                    f"LibRadtran failed with return code {result.returncode}: {error_msg}"
                )
                return False

            return True

        except subprocess.TimeoutExpired:
            self.last_stderr = f"timeout after {self.config.execution.timeout_seconds}s"
            logger.error(
                f"LibRadtran execution timed out after {self.config.execution.timeout_seconds} seconds"
            )
            return False
        except Exception as e:
            self.last_stderr = str(e)
            logger.error(f"Failed to execute LibRadtran: {str(e)}")
            return False


# near bottom of core.py, before __all__
# Backwards-compatible alias (was a Simulation method)
Simulation._calculate_solar_zenith_angle = staticmethod(
    lambda dt, latitude, longitude: calculate_solar_zenith_angle(
        dt, latitude, longitude
    )
)


# Expose main class
__all__ = ["Simulation"]
