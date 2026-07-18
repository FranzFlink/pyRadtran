"""
uvspec input-file generation with per-line provenance.

:class:`InputFileBuilder` turns a :class:`~pyradtran.config.SimulationConfig`
plus a resolved parameter mapping (from
:class:`~pyradtran.params.ParamResolver`) into an ordered list of
:class:`InputLine` objects. Each line knows which layer produced it, which
powers ``ds.pyradtran.explain()``.

See Also
--------
pyradtran.params : Parameter registry and resolution.
pyradtran.core.Simulation : Consumes the rendered input file.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .params import PROV_CONFIG, REGISTRY

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InputLine:
    """One uvspec input line with provenance.

    Parameters
    ----------
    keyword : str
        The (possibly multi-word) uvspec keyword, e.g. ``"mol_modify O3"``.
    text : str
        The complete input line.
    provenance : str
        One of the ``PROV_*`` constants in :mod:`pyradtran.params`.
    """

    keyword: str
    text: str
    provenance: str


def calculate_solar_zenith_angle(
    dt: datetime, latitude: float, longitude: float
) -> float:
    """Approximate solar zenith angle (degrees) from time and location."""
    import numpy as np

    day_of_year = dt.timetuple().tm_yday
    declination = 23.45 * np.sin(np.radians((360 * (284 + day_of_year)) / 365))
    time_decimal = dt.hour + dt.minute / 60.0 + dt.second / 3600.0
    hour_angle = 15 * (time_decimal - 12) + longitude

    lat_rad = np.radians(latitude)
    decl_rad = np.radians(declination)
    hour_rad = np.radians(hour_angle)

    cos_sza = np.sin(lat_rad) * np.sin(decl_rad) + np.cos(lat_rad) * np.cos(
        decl_rad
    ) * np.cos(hour_rad)
    return round(float(np.degrees(np.arccos(np.clip(cos_sza, -1, 1)))), 2)


def effective_output_altitudes(config, overrides: Optional[Dict[str, Any]] = None) -> List[float]:
    """Altitudes this run reports at: per-run ``zout`` override, else config."""
    overrides = overrides or {}
    zout = overrides.get("zout")
    if zout is not None:
        if isinstance(zout, str):
            return [float(tok) for tok in zout.split()]
        if isinstance(zout, (list, tuple)):
            return [float(v) for v in zout]
        return [float(zout)]
    return list(config.simulation_defaults.output_altitudes_km or [0.0])


def effective_output_columns(config, overrides: Optional[Dict[str, Any]] = None) -> List[str]:
    """Columns the ``output_user`` line requests, in file order.

    A per-run ``output_user`` override replaces the config columns.
    Spectral runs get ``lambda`` injected and multi-altitude runs get
    ``zout``, so the output file can always be mapped back onto its
    wavelength/altitude axes — without them the batch converter cannot
    reshape the rows and every value ends up NaN. The
    :class:`~pyradtran.io.OutputParser` uses this same function, so the
    file and the parser cannot disagree.
    """
    overrides = overrides or {}
    sd = config.simulation_defaults
    output_user = overrides.get("output_user")
    if output_user is not None:
        cols = (
            list(output_user.split())
            if isinstance(output_user, str)
            else list(output_user)
        )
    else:
        cols = list(sd.output_columns or [])
    if not cols:
        return cols  # empty means uvspec's default output — leave untouched
    if not getattr(sd, "integrate_wavelength", False) and "lambda" not in cols:
        cols.insert(0, "lambda")
    if len(effective_output_altitudes(config, overrides)) > 1 and "zout" not in cols:
        cols.insert(0, "zout")
    return cols


class InputFileBuilder:
    """Build provenance-tagged uvspec input lines.

    Parameters
    ----------
    config : SimulationConfig
        Merged configuration.
    """

    def __init__(self, config):
        self.config = config

    def build(
        self,
        dt: datetime,
        latitude: float,
        longitude: float,
        resolved: Optional[Dict[str, Tuple[Any, str]]] = None,
        radiosonde_path: Optional[Path] = None,
        era5_atmosphere_file: Optional[Path] = None,
    ) -> List[InputLine]:
        """Build the full ordered line list for one simulation point."""
        resolved = dict(resolved or {})
        sd = self.config.simulation_defaults
        lines: List[InputLine] = []

        def add(keyword: str, text: str, provenance: str = PROV_CONFIG):
            lines.append(InputLine(keyword, text, provenance))

        add("rte_solver", f"rte_solver {sd.rte_solver}")
        if sd.mol_abs_param:
            add("mol_abs_param", f"mol_abs_param {sd.mol_abs_param}")

        add("data_files_path", f"data_files_path {self.config.paths.libradtran_data}")
        add("atmosphere_file", f"atmosphere_file {self.config.paths.atmosphere_profile}")

        # Atmosphere: ERA5 file wins over radiosonde
        if era5_atmosphere_file is not None:
            era5_abs_path = Path(era5_atmosphere_file).resolve()
            h2o_unit = _sniff_h2o_unit(era5_abs_path)
            add("radiosonde", f"radiosonde {era5_abs_path} H2O {h2o_unit}")
        elif radiosonde_path and sd.h2o_source == "radiosonde":
            add("radiosonde", f"radiosonde {radiosonde_path} H2O RH")

        # Molecular columns (B2 fix: emit when configured)
        if sd.ozone_du is not None:
            add("mol_modify O3", REGISTRY["mol_modify O3"].format_line(sd.ozone_du))
        if sd.h2o_mm is not None:
            add("mol_modify H2O", REGISTRY["mol_modify H2O"].format_line(sd.h2o_mm))

        # Source
        if sd.source == "solar":
            if sd.integrate_wavelength:
                add("source", f"source solar {self.config.paths.solar_spectrum} per_nm")
            else:
                add("source", f"source solar {self.config.paths.solar_spectrum}")
            if "sza" not in resolved:
                sza = sd.sza if sd.sza is not None else calculate_solar_zenith_angle(
                    dt, latitude, longitude
                )
                add("sza", f"sza {sza}")
            add(
                "day_of_year",
                REGISTRY["day_of_year"].format_line(dt.timetuple().tm_yday),
            )
        elif sd.source == "thermal":
            add("source", "source thermal")
            # B3 fix: explicit None check, no `or`
            if "sur_temperature" not in resolved and sd.surface_temperature_k is not None:
                add(
                    "sur_temperature",
                    f"sur_temperature {sd.surface_temperature_k}",
                )

        # Spectral range
        if sd.wavelength_nm and len(sd.wavelength_nm) == 2:
            wl_min, wl_max = sd.wavelength_nm
            add("wavelength", f"wavelength {wl_min} {wl_max}")
        if sd.integrate_wavelength:
            add("output_process", "output_process integrate")

        # Surface: BRDF type disables plain albedo
        if "brdf_rpv_type" not in resolved:
            if "albedo" not in resolved and sd.albedo_value is not None:
                add("albedo", f"albedo {sd.albedo_value}")

        # Output columns: per-run output_user override wins; lambda/zout
        # are injected so the output stays parseable (see
        # effective_output_columns). Consumed here, not in the generic
        # override loop below.
        output_user_override = resolved.pop("output_user", None)
        raw_overrides = {k: v for k, (v, _p) in resolved.items()}
        if output_user_override is not None:
            raw_overrides["output_user"] = output_user_override[0]
        columns = effective_output_columns(self.config, raw_overrides)
        if columns:
            add(
                "output_user",
                "output_user " + " ".join(columns),
                output_user_override[1] if output_user_override else PROV_CONFIG,
            )

        # Output altitudes
        if "zout" not in resolved and sd.output_altitudes_km:
            alt_str = " ".join(f"{alt:.4f}" for alt in sd.output_altitudes_km)
            add("zout", f"zout {alt_str}")

        # Viewing geometry
        if sd.viewing_geometry == "nadir":
            add("umu", "umu 1.0")

        # Static cloud settings from config
        if sd.clouds.enabled:
            self._add_cloud_lines(lines)

        # BRDF surface type unconditionally replaces plain albedo,
        # regardless of dict ordering.
        if "brdf_rpv_type" in resolved:
            resolved.pop("albedo", None)
            lines = [ln for ln in lines if ln.keyword != "albedo"]

        # Apply resolved parameters: replace same-keyword lines, then append
        for key, (value, provenance) in resolved.items():
            spec = REGISTRY.get(key)
            keyword = spec.keyword if spec is not None else key
            lines = [ln for ln in lines if ln.keyword != keyword]
            if isinstance(value, list):
                # Repeatable option (non_unique): one line per entry
                for item in value:
                    lines.append(
                        InputLine(keyword, f"{keyword} {item}", provenance)
                    )
                continue
            if spec is not None:
                text = spec.format_line(value)
            elif value is True or value is None or value == "":
                text = keyword  # flag option, e.g. aerosol_default
            else:
                text = f"{key} {value}"
            lines.append(InputLine(keyword, text, provenance))
            # brdf implies the IGBP library line
            if keyword == "brdf_rpv_type":
                lines = [ln for ln in lines if ln.keyword not in ("albedo", "brdf_rpv_library")]
                lines.append(
                    InputLine("brdf_rpv_library", "brdf_rpv_library IGBP", provenance)
                )

        lines.append(InputLine("quiet", "quiet", PROV_CONFIG))
        return lines

    def _add_cloud_lines(self, lines: List[InputLine]) -> None:
        """Append config-driven cloud lines (file or parametric)."""
        clouds = self.config.simulation_defaults.clouds
        if clouds.cloud_source == "file":
            if clouds.cloud_type in ["wc", "mixed"] and clouds.wc_file:
                lines.append(
                    InputLine("wc_file", f"wc_file {clouds.wc_file}", PROV_CONFIG)
                )
            if clouds.cloud_type in ["ic", "mixed"] and clouds.ic_file:
                lines.append(
                    InputLine("ic_file", f"ic_file {clouds.ic_file}", PROV_CONFIG)
                )
        elif clouds.cloud_source == "parametric":
            if clouds.cloud_type in ["wc", "mixed"]:
                lines.append(
                    InputLine(
                        "wc_layer",
                        f"wc_layer {clouds.layer_bottom_km} {clouds.layer_top_km} "
                        f"{clouds.water_content_g_m3} {clouds.effective_radius_um}",
                        PROV_CONFIG,
                    )
                )
            if clouds.cloud_type in ["ic", "mixed"]:
                lines.append(
                    InputLine(
                        "ic_layer",
                        f"ic_layer {clouds.layer_bottom_km} {clouds.layer_top_km} "
                        f"{clouds.ice_content_g_m3} {clouds.effective_radius_um}",
                        PROV_CONFIG,
                    )
                )

    @staticmethod
    def render(lines: List[InputLine]) -> str:
        """Render lines to the plain uvspec input file content."""
        return "\n".join(ln.text for ln in lines) + "\n"

    @staticmethod
    def render_annotated(lines: List[InputLine]) -> str:
        """Render lines with aligned ``# <provenance>`` annotations."""
        if not lines:
            return ""
        width = max(len(ln.text) for ln in lines) + 2
        return "\n".join(
            f"{ln.text:<{width}}# {ln.provenance}" for ln in lines
        ) + "\n"


def _sniff_h2o_unit(era5_abs_path: Path) -> str:
    """Read the header of an ERA5 atmosphere file to determine the H2O unit."""
    h2o_unit = "MMR"
    try:
        with open(era5_abs_path, "r") as f:
            second_line = f.readlines()[1].strip()
        if "%" in second_line:
            h2o_unit = "RH"
    except (OSError, IndexError):
        logger.warning(
            f"Could not sniff H2O unit from {era5_abs_path}; assuming MMR"
        )
    return h2o_unit


__all__ = [
    "InputLine",
    "InputFileBuilder",
    "calculate_solar_zenith_angle",
    "effective_output_columns",
    "effective_output_altitudes",
]
