# pyradtran/core.py
"""
Simulation engine for pyRadtran.

This module contains the :class:`Simulation` class, the low-level
workhorse behind every ``uvspec`` invocation: one instance, one
configuration, one simulation point per call.

.. autosummary::

   Simulation

It is responsible for:

* Generating a complete ``uvspec`` input file from a
  :class:`~pyradtran.config.SimulationConfig` and the per-point resolved
  parameters (:meth:`Simulation.build_input_lines`,
  :meth:`Simulation.dry_run`).
* Writing dict-valued cloud profiles to temporary cloud files
  (:meth:`Simulation.format_cloud_profile`).
* Spawning the ``uvspec`` subprocess and capturing its stdout / stderr
  (:meth:`Simulation.run_simulation`).
* Cleaning up temporary files.

Most users should interact with the higher-level
:class:`~pyradtran.interface.PyRadtranAccessor` (``ds.pyradtran.run()``)
rather than calling :class:`Simulation` directly; it adds the loop over
points, parameter resolution, output parsing and the conversion to
:class:`xarray.Dataset`.

Examples
--------
Render the input file of a single point without running ``uvspec``:

>>> from datetime import datetime
>>> from pyradtran import Simulation, load_config
>>> sim = Simulation(load_config("config/quickstart.yaml"))  # doctest: +SKIP
>>> print(sim.dry_run(datetime(2025, 6, 21, 12), 61.0, 22.0))  # doctest: +SKIP
rte_solver twostr                                                   # config
mol_abs_param lowtran per_nm                                        # config
...
quiet                                                               # config

See Also
--------
pyradtran.interface : High-level batch interface and xarray accessor.
pyradtran.input_builder.InputFileBuilder : Assembles the input lines.
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
from .input_builder import (  # noqa: F401
    InputFileBuilder,
    calculate_solar_zenith_angle,
    parametric_cloud_profiles,
)
from .params import PROV_CONFIG, PROV_LITERAL, PROV_UNVALIDATED
from .utils import RadiosondeFinder

logger = logging.getLogger(__name__)


class Simulation:
    """Low-level wrapper around a single ``uvspec`` execution.

    A :class:`Simulation` binds a configuration; each call to
    :meth:`run_simulation` then writes one input file, runs ``uvspec`` on
    it and returns the path of the raw output file. :meth:`dry_run` and
    :meth:`build_input_lines` build the same input without running
    anything. Instances hold no per-point state apart from the failure
    diagnostics below, so one instance can be reused for many points.

    Parameters
    ----------
    config : SimulationConfig
        Fully merged configuration, e.g. from
        :func:`~pyradtran.config.load_config`. Not copied.

    Attributes
    ----------
    config : SimulationConfig
        The configuration passed in.
    builder : InputFileBuilder
        Turns configuration + resolved parameters into input lines.
    last_stderr : str or None
        ``uvspec`` stderr (or a timeout / launch error message) of the
        most recent :meth:`run_simulation` call if it failed, else
        *None*.
    last_failed_input : pathlib.Path or None
        Input file of the most recent failed run, kept on disk for
        post-mortem; *None* after a successful run.
    radiosonde_finder : RadiosondeFinder or None
        Radiosonde lookup, present when ``paths.radiosonde_base`` is
        configured.

    See Also
    --------
    pyradtran.interface.PyRadtranAccessor.run : High-level entry point.
    pyradtran.interface.execute_simulation_batch : Parallel driver that
        creates one :class:`Simulation` per point.
    pyradtran.io.OutputParser : Parses the output file returned by
        :meth:`run_simulation`.

    Notes
    -----
    Per-point parameters are passed as *resolved params*: a mapping
    ``key -> (value, provenance)`` where *key* is a uvspec option name
    and *provenance* one of ``"config"``, ``"params-literal"``,
    ``"dataset-var"``, ``"unvalidated"`` (the ``PROV_*`` constants in
    :mod:`pyradtran.params`). :class:`~pyradtran.params.ParamResolver`
    produces this mapping from a user ``params`` dict; values given here
    are **not** validated again. A resolved parameter replaces the line
    the configuration would have produced for the same option.

    Examples
    --------
    >>> from datetime import datetime
    >>> from pyradtran import Simulation, load_config
    >>> sim = Simulation(load_config("config/quickstart.yaml"))  # doctest: +SKIP

    Inspect the input file, then run it and read the raw output:

    >>> noon = datetime(2025, 6, 21, 12)
    >>> params = {"wavelength": ("500 502", "params-literal")}
    >>> text = sim.dry_run(noon, 61.0, 22.0, resolved_params=params)  # doctest: +SKIP
    >>> out = sim.run_simulation(noon, 61.0, 22.0, resolved_params=params)  # doctest: +SKIP
    >>> out.name  # doctest: +SKIP
    'tmpvouk2nni.out'
    >>> print(out.read_text())  # lambda edir edn eup  # doctest: +SKIP
      500.000  1.165645e+03  1.452510e+02  2.270538e+02
      501.000  1.146696e+03  1.416279e+02  2.231443e+02
      502.000  1.134968e+03  1.389458e+02  2.206483e+02

    A failing run returns *None* and leaves the diagnostics on the
    instance:

    >>> bad = {"rte_solver": ("nonsense", "unvalidated")}
    >>> sim.run_simulation(noon, 61.0, 22.0, resolved_params=bad) is None  # doctest: +SKIP
    True
    >>> print(sim.last_stderr)  # doctest: +SKIP
    Option rte_solver nonsense  has an invalid argument 1 on line 14: nonsense
    Error in input file! Please check your input file and retry!
    >>> sim.last_failed_input.name  # doctest: +SKIP
    'tmpwzgv161d.inp'
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

        # Layer 1: config-level escape hatch (no deprecation warning).
        config_overrides = getattr(
            self.config.simulation_defaults, "parameter_overrides", None
        )
        resolved: Dict[str, Any] = {
            key: (value, PROV_CONFIG) for key, value in (config_overrides or {}).items()
        }

        # Layer 2: legacy override_* kwargs (NaN never reaches the input file).
        for key in used:
            if not (isinstance(legacy[key], float) and np.isnan(legacy[key])):
                resolved[key] = (legacy[key], PROV_LITERAL)

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
        """Build the provenance-tagged input lines for one point.

        The structured form of the input file: one
        :class:`~pyradtran.input_builder.InputLine` (``keyword``,
        ``text``, ``provenance``) per uvspec option, in file order.
        :meth:`dry_run` and :meth:`run_simulation` are built on it.

        Parameters
        ----------
        dt : datetime.datetime
            Simulation date/time (UTC). Sets ``day_of_year`` and, for
            solar runs without an explicit ``sza``, the solar zenith
            angle.
        latitude, longitude : float
            Location in degrees north / east.
        resolved_params : dict, optional
            ``key -> (value, provenance)`` mapping from
            :meth:`pyradtran.params.ParamResolver.resolve_point`. Wins
            over the configuration and over the deprecated arguments.
        era5_atmosphere_file : pathlib.Path, optional
            Radiosonde-format atmosphere file (see
            :func:`pyradtran.era5.write_atmosphere_file`); takes
            precedence over a radiosonde lookup.
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
        list of InputLine
            The ordered input lines.

        Warns
        -----
        DeprecationWarning
            If any ``override_*`` argument or ``parameter_overrides`` is
            used.

        See Also
        --------
        dry_run : The same lines rendered as annotated text.
        pyradtran.input_builder.InputFileBuilder.build : Does the work.

        Notes
        -----
        Entries of ``simulation_defaults.parameter_overrides`` in the
        configuration are applied first (provenance ``config``), then
        the deprecated arguments, then *resolved_params*. With
        ``h2o_source: radiosonde`` and a configured
        ``paths.radiosonde_base``, the radiosonde file closest in time
        to *dt* is looked up and included.

        Examples
        --------
        >>> from datetime import datetime
        >>> from pyradtran import Simulation, load_config
        >>> sim = Simulation(load_config("config/quickstart.yaml"))  # doctest: +SKIP
        >>> lines = sim.build_input_lines(  # doctest: +SKIP
        ...     datetime(2025, 6, 21, 12), 61.0, 22.0,
        ...     resolved_params={"albedo": (0.8, "params-literal")},
        ... )
        >>> lines[0]  # doctest: +SKIP
        InputLine(keyword='rte_solver', text='rte_solver twostr', provenance='config')
        >>> [ln.text for ln in lines if ln.provenance != "config"]  # doctest: +SKIP
        ['albedo 0.8']
        """
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

        Parameters
        ----------
        dt : datetime.datetime
            Simulation date/time (UTC).
        latitude, longitude : float
            Location in degrees north / east.
        resolved_params : dict, optional
            ``key -> (value, provenance)`` mapping, as in
            :meth:`run_simulation`.
        era5_atmosphere_file : pathlib.Path, optional
            Radiosonde-format atmosphere file to reference.

        Returns
        -------
        str
            Input-file content, one option per line, each followed by an
            aligned ``# <provenance>`` comment (``config``,
            ``params-literal``, ``dataset-var`` or ``unvalidated``).

        See Also
        --------
        pyradtran.interface.PyRadtranAccessor.explain : The same preview
            for a point of a dataset, starting from a user ``params``
            mapping.
        build_input_lines : The underlying structured lines.

        Notes
        -----
        Nothing is written to disk. Consequently a dict-valued
        ``wc_file`` / ``ic_file`` — from *resolved_params* or from a
        ``cloud_source: parametric`` configuration — appears as a
        placeholder such as ``wc_file 1D <generated: 2 levels, 1-2 km>``,
        whereas :meth:`run_simulation` first writes it to a temporary
        file and puts ``1D <path>`` on the line.

        Examples
        --------
        >>> from datetime import datetime
        >>> from pyradtran import Simulation, load_config
        >>> sim = Simulation(load_config("config/quickstart.yaml"))  # doctest: +SKIP
        >>> print(  # doctest: +SKIP
        ...     sim.dry_run(
        ...         datetime(2025, 6, 21, 12), 61.0, 22.0,
        ...         resolved_params={"albedo": (0.8, "params-literal")},
        ...     )
        ... )
        rte_solver twostr                                                   # config
        mol_abs_param lowtran per_nm                                        # config
        data_files_path /opt/libRadtran-2.0.6/data                          # config
        atmosphere_file /opt/libRadtran-2.0.6/data/atmmod/afglus.dat        # config
        mol_modify O3 300.0 DU                                              # config
        mol_modify H2O 2.0 MM                                               # config
        source solar /opt/libRadtran-2.0.6/data/solar_flux/NewGuey2003.dat  # config
        sza 40.39                                                           # config
        day_of_year 172                                                     # config
        wavelength 350 2500                                                 # config
        output_user lambda edir edn eup                                     # config
        zout 0.0000                                                         # config
        umu 1.0                                                             # config
        albedo 0.8                                                          # params-literal
        quiet                                                               # config
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

        Builds the input file for one point, writes it to
        ``paths.working_dir``, pipes it into the ``uvspec`` binary
        (``paths.libradtran_bin``) and stores stdout next to it as
        ``<name>.out``.

        Parameters
        ----------
        dt : datetime.datetime
            Simulation date/time (UTC).
        latitude, longitude : float
            Location in degrees north / east.
        resolved_params : dict, optional
            ``key -> (value, provenance)`` mapping from
            :meth:`pyradtran.params.ParamResolver.resolve_point`. A
            dict value for ``"wc_file"`` / ``"ic_file"`` is written to a
            temporary cloud file with :meth:`format_cloud_profile` and
            replaced by ``1D <path>``. The layer of a
            ``cloud_source: parametric`` configuration is handled the
            same way (see
            :func:`~pyradtran.input_builder.parametric_cloud_profiles`)
            unless *resolved_params* has that key.
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
            Path to the raw ``uvspec`` output file, or *None* if
            ``uvspec`` exited with an error, timed out
            (``execution.timeout_seconds``) or could not be started.
            :attr:`last_stderr` then holds the error message and
            :attr:`last_failed_input` the kept input file.

        Raises
        ------
        UvspecExecutionError
            If the input file cannot be built or written (e.g. an
            invalid cloud-profile dict); the original exception is
            chained. A failing ``uvspec`` does *not* raise.

        Warns
        -----
        DeprecationWarning
            If any ``override_*`` argument or ``parameter_overrides`` is
            used.

        See Also
        --------
        dry_run : Show the input file without running.
        pyradtran.io.OutputParser.parse_output_file : Parse the returned
            file.
        pyradtran.interface.execute_simulation_batch : Runs this for
            every point of a dataset.

        Notes
        -----
        With ``execution.cleanup_temp_files`` enabled, the input file and
        temporary cloud files are deleted after a successful run; the
        output file is left for the caller to parse and remove. After a
        failure, the input file and the cloud files it references are
        always kept.

        Examples
        --------
        >>> from datetime import datetime
        >>> from pyradtran import Simulation, load_config
        >>> sim = Simulation(load_config("config/quickstart.yaml"))  # doctest: +SKIP
        >>> out = sim.run_simulation(  # doctest: +SKIP
        ...     datetime(2025, 6, 21, 12), 61.0, 22.0,
        ...     resolved_params={"wavelength": ("500 502", "params-literal")},
        ... )
        >>> print(out.read_text())  # lambda edir edn eup  # doctest: +SKIP
          500.000  1.165645e+03  1.452510e+02  2.270538e+02
          501.000  1.146696e+03  1.416279e+02  2.231443e+02
          502.000  1.134968e+03  1.389458e+02  2.206483e+02
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

            # Config-driven parametric cloud layer: same dict form as a
            # params cloud profile; an explicit wc_file / ic_file wins.
            for key, profile in parametric_cloud_profiles(self.config).items():
                resolved.setdefault(key, (profile, PROV_CONFIG))

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
            # Failure: keep input and output for post-mortem
            self.last_failed_input = input_path
            logger.error(f"Simulation failed; input kept at {input_path}")
            return None

        except Exception as e:
            logger.error(f"Simulation failed: {e}")
            raise UvspecExecutionError(f"Simulation failed: {e}") from e

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
        r"""Format a cloud-profile dict as a libRadtran column file.

        Produces the three-column text expected by the ``wc_file 1D`` /
        ``ic_file 1D`` options: altitude, water content, effective
        radius. This is what a dict-valued ``wc_file`` / ``ic_file``
        parameter is turned into before ``uvspec`` runs.

        Parameters
        ----------
        data : dict
            Profile with equally long sequences under the keys

            ``"z"``
                Level altitudes in km, **top of atmosphere first**
                (descending), as libRadtran requires.
            ``"lwc"`` or ``"iwc"``
                Liquid or ice water content in g m\ :sup:`-3`. If both
                are present, ``"lwc"`` is used.
            ``"reff"``
                Effective radius in µm.

        Returns
        -------
        str
            One line per level, ``z content reff`` with six decimals,
            newline-terminated. Rows are written in the order given; no
            sorting or validation of the values takes place.

        Raises
        ------
        ValueError
            If ``"z"``, ``"reff"`` or both of ``"lwc"`` / ``"iwc"`` are
            missing.
        RuntimeError
            If the columns cannot be combined (e.g. unequal lengths).

        See Also
        --------
        pyradtran.interface.PyRadtranAccessor.inspect_cloud_file :
            Preview the profile built for a point of a dataset.
        pyradtran.era5.cloud_profiles : Build such dicts from ERA5.

        Notes
        -----
        libRadtran interprets the value at level :math:`z_i` as the
        (constant) content of the layer between :math:`z_i` and the next
        level above, :math:`z_{i-1}`. The topmost row therefore only
        closes the grid — give it zero content. The water path of a
        profile with levels :math:`z_0 > z_1 > \dots > z_n` is

        .. math::

            \mathrm{LWP} = \sum_{i=1}^{n} w_i\,(z_{i-1} - z_i)

        (in g m\ :sup:`-2` with :math:`w` in g m\ :sup:`-3` and
        :math:`z` in m).

        Examples
        --------
        A 1 km thick liquid cloud with 0.3 g m\ :sup:`-3` and 10 µm
        droplets (LWP = 300 g m\ :sup:`-2`):

        >>> from pyradtran import Simulation
        >>> print(
        ...     Simulation.format_cloud_profile(
        ...         {"z": [2.0, 1.0], "lwc": [0.0, 0.3], "reff": [0.0, 10.0]}
        ...     )
        ... )
        2.000000 0.000000 0.000000
        1.000000 0.300000 10.000000
        <BLANKLINE>

        A two-layer ice cloud; NumPy arrays work as well as lists:

        >>> import numpy as np
        >>> text = Simulation.format_cloud_profile(
        ...     {
        ...         "z": np.array([2.0, 1.5, 1.0]),
        ...         "iwc": np.array([0.0, 0.02, 0.01]),
        ...         "reff": np.array([0.0, 30.0, 25.0]),
        ...     }
        ... )
        >>> text.splitlines()
        ['2.000000 0.000000 0.000000', '1.500000 0.020000 30.000000', '1.000000 0.010000 25.000000']

        Incomplete profiles are rejected:

        >>> Simulation.format_cloud_profile({"z": [2.0, 1.0], "reff": [0.0, 10.0]})
        Traceback (most recent call last):
            ...
        ValueError: Profile data must contain 'lwc' or 'iwc'
        """
        if "z" not in data:
            raise ValueError("Profile data must contain 'z'")
        content_key = "lwc" if "lwc" in data else "iwc"
        if content_key not in data:
            raise ValueError("Profile data must contain 'lwc' or 'iwc'")
        if "reff" not in data:
            raise ValueError("Profile data must contain 'reff'")

        try:
            # column_stack rejects columns of unequal length
            rows = np.column_stack([data["z"], data[content_key], data["reff"]])
            return "\n".join(f"{r[0]:.6f} {r[1]:.6f} {r[2]:.6f}" for r in rows) + "\n"
        except Exception as e:
            raise RuntimeError(f"Failed to format cloud profile: {e}") from e

    def _handle_dynamic_clouds(
        self, overrides: Dict[str, Any]
    ) -> tuple[Dict[str, Any], list[str]]:
        """Convert dict-valued ``wc_file`` / ``ic_file`` overrides to temp files."""
        new_updates = {}
        cleanup_list = []
        output_dir = self.config.paths.working_dir
        for key in ("wc_file", "ic_file"):
            if not isinstance(overrides.get(key), (dict, list)):
                continue
            content = self.format_cloud_profile(overrides[key])
            if not output_dir.exists():
                output_dir.mkdir(parents=True, exist_ok=True)
            fd, path = tempfile.mkstemp(
                prefix=f"pyradtran_{key[:2]}_", suffix=".dat", dir=output_dir
            )
            with os.fdopen(fd, "w") as f:
                f.write(content)
            new_updates[key] = f"1D {path}"
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
            logger.debug(f"Running LibRadtran: {cmd[0]} < {input_path} > {output_path}")

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
            logger.error(f"Failed to execute LibRadtran: {e}")
            return False


# Expose main class
__all__ = ["Simulation"]
