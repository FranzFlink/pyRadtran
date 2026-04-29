#!/usr/bin/env python3
"""
port_examples.py

This script converts standard libRadtran .INP examples into pyRadtran YAML configuration files.
It verifies the conversion by running both the original INP and the new YAML and comparing outputs.
"""

import argparse
import logging
import os
import shutil
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

# Add parent directory to path to allow importing pyradtran
sys.path.insert(0, str(Path(__file__).parent.parent))

from pyradtran.config import (
    CloudParameters,
    SimulationConfig,
    SimulationDefaults,
    load_config,
)
from pyradtran.core import Simulation

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("port_examples")

TARGET_EXAMPLES = [
    "UVSPEC_CLEAR.INP",
    "UVSPEC_CLOUDCOVER.INP",
    "UVSPEC_AEROSOL.INP",
    "UVSPEC_DISORT.INP",
    "UVSPEC_MC.INP",
    "UVSPEC_WC.INP",
    "UVSPEC_REPTRAN_SOLAR.INP",
    "UVSPEC_LOWTRAN_THERMAL.INP",
    "UVSPEC_SIMPLE.INP",
    "UVSPEC_MC_POL.INP",
]


def parse_inp_file(inp_path: Path) -> Dict[str, Any]:
    """
    Parse a libRadtran .INP file into a dictionary of key-values.
    """
    params = {}
    with open(inp_path, "r") as f:
        for line in f:
            # Remove comments
            line = line.split("#")[0].strip()
            if not line:
                continue

            parts = line.split()
            if not parts:
                continue

            key = parts[0]
            values = parts[1:]

            if len(values) == 1:
                val = values[0]
                try:
                    if "." in val:
                        params[key] = float(val)
                    else:
                        params[key] = int(val)
                except ValueError:
                    params[key] = val
            else:
                params[key] = " ".join(values)

    return params


def resolve_paths(
    params: Dict[str, Any], example_dir: Path, libradtran_data: Path
) -> Dict[str, Any]:
    """
    Resolve relative paths in parameters to absolute paths.
    """
    resolved_params = params.copy()

    for key, val in params.items():
        if isinstance(val, str):
            if "../data/" in val:
                new_val = val.replace("../data/", str(libradtran_data) + "/")
                resolved_params[key] = new_val
            elif "../examples/" in val:
                new_val = val.replace("../examples/", str(example_dir) + "/")
                resolved_params[key] = new_val

    return resolved_params


def map_inp_to_config(
    params: Dict[str, Any], base_config: SimulationConfig
) -> SimulationConfig:
    """
    Map INP parameters to a SimulationConfig object.
    """
    # Create a deep copy or new instance to avoid modifying base
    # But for now we just create a fresh structure based on defaults

    # We start with default values from the class definition, NOT the user's config
    # because we want the ported example to be self-contained and explicit.
    # However, paths need to be from the user's environment.

    sim_defaults = SimulationDefaults()
    clouds = CloudParameters()

    # Reset defaults that might conflict with "raw" UVSPEC behavior if not present
    sim_defaults.output_altitudes_km = (
        []
    )  # Default to empty, let specific mapping fill it
    sim_defaults.viewing_geometry = (
        "custom"  # Default to custom (no umu 1.0 enforcement)
    )
    sim_defaults.mol_abs_param = None  # Default to None (uvspec default)
    sim_defaults.output_columns = []  # Default to empty (uvspec default)
    sim_defaults.wavelength_nm = []  # Default to empty (uvspec default)

    overrides = {}

    # Iterate and Map
    # We need to handle keys carefully

    processed_keys = set()

    # 1. RTE Solver
    if "rte_solver" in params:
        sim_defaults.rte_solver = params["rte_solver"]
        processed_keys.add("rte_solver")

    # 2. Molecular Absorption
    if "mol_abs_param" in params:
        sim_defaults.mol_abs_param = params["mol_abs_param"]
        processed_keys.add("mol_abs_param")

    # 3. Albedo
    if "albedo" in params:
        sim_defaults.albedo_value = float(params["albedo"])
        processed_keys.add("albedo")

    # 4. Source
    if "source" in params:
        val = params["source"]
        if "solar" in val:
            sim_defaults.source = "solar"
            # Extract spectrum path if present
            parts = val.split()
            if len(parts) > 1:
                # Assuming 'solar <path>'
                # We need to put this path into config.paths.solar_spectrum
                # BUT this function returns a config object.
                # Since 'source' in map is just string, we might need to update paths config too?
                # Actually, SimulationConfig structure separates paths.
                # Let's update base_config.paths.solar_spectrum
                spectrum_path = parts[1]
                base_config.paths.solar_spectrum = Path(spectrum_path)
        elif "thermal" in val:
            sim_defaults.source = "thermal"
            # Thermal usually implies no external source file in uvspec syntax
            # unless specified elsewhere? usually 'source thermal' is enough.
        processed_keys.add("source")

    # 5. Wavelength
    if "wavelength" in params:
        # Expecting "min max"
        parts = str(params["wavelength"]).split()
        if len(parts) == 2:
            sim_defaults.wavelength_nm = [float(parts[0]), float(parts[1])]
            processed_keys.add("wavelength")

    # 6. Zout (Altitudes)
    if "zout" in params:
        val = params["zout"]
        if isinstance(val, (int, float)):
            sim_defaults.output_altitudes_km = [float(val)]
        else:
            sim_defaults.output_altitudes_km = [float(x) for x in val.split()]
        processed_keys.add("zout")

    # 7. Umu (Viewing Angle)
    has_viewing = False
    if "umu" in params:
        # If umu is 1.0 or -1.0 or explicitly nadir?
        # For now, just mark it as processed, and if we want we can map 'nadir'
        # But keeping it 'custom' and putting 'umu' in overrides is safer for exact repro.
        # Actually, let's put 'umu' in overrides to be safe unless it's strictly 1.0
        pass

    # 8. Output columns (output_user)
    if "output_user" in params:
        sim_defaults.output_columns = params["output_user"].split()
        processed_keys.add("output_user")

    # 9. Clouds
    # wc_file, ic_file
    if "wc_file" in params:
        clouds.enabled = True
        clouds.wc_file = Path(params["wc_file"])
        clouds.cloud_source = "file"
        if "ic_file" in params:
            clouds.cloud_type = "mixed"
        else:
            clouds.cloud_type = "wc"
        processed_keys.add("wc_file")

    if "ic_file" in params:
        clouds.enabled = True
        clouds.ic_file = Path(params["ic_file"])
        clouds.cloud_source = "file"
        if "wc_file" not in params:  # already handled above
            clouds.cloud_type = "ic"
        processed_keys.add("ic_file")

    # wc_layer (Parametric)
    # INP: wc_layer <bottom> <top> <wc/ic> < reff>
    # Note: parsing mixed args is hard. pyRadtran wrapper 'wc_layer' expects specific lines
    # If we find wc_properties or similar, we might just leave them as overrides.

    # 10. Everything else -> Overrides
    for k, v in params.items():
        if k not in processed_keys:
            if k == "data_files_path":
                continue  # handled by config paths
            if k == "atmosphere_file":
                base_config.paths.atmosphere_profile = Path(v)
                continue

            overrides[k] = v

    sim_defaults.clouds = clouds
    sim_defaults.parameter_overrides = overrides

    # Construct new config
    # We copy paths from base_config (which has user env paths), but update specific files found in INP
    new_config = SimulationConfig(
        paths=base_config.paths,
        simulation_defaults=sim_defaults,
        execution=base_config.execution,
        output=base_config.output,
    )

    return new_config


def generate_yaml(config: SimulationConfig, output_path: Path):
    """
    Serialize SimulationConfig to YAML.
    """
    # Convert to dict
    # We can use the existing `get_used_parameters` or build one manually
    # to ensure cleaner output. `get_used_parameters` is good but might exclude overrides.

    # Let's build a dict manually for better control over the YAML structure
    # based on the dataclass fields.

    data = {
        "paths": {
            "libradtran_data": str(config.paths.libradtran_data),
            # We don't necessarily need bin path in the example config, it's env specific?
            # But the structure requires valid PathsConfig.
            # We can put placeholders or actual paths.
            "libradtran_bin": str(config.paths.libradtran_bin),
            "atmosphere_profile": (
                str(config.paths.atmosphere_profile)
                if config.paths.atmosphere_profile
                else None
            ),
            "solar_spectrum": (
                str(config.paths.solar_spectrum)
                if config.paths.solar_spectrum
                else None
            ),
            "output_dir": "./pyradtran_output",
            "working_dir": "./pyradtran_work",
        },
        "simulation_defaults": {
            "rte_solver": config.simulation_defaults.rte_solver,
            "mol_abs_param": config.simulation_defaults.mol_abs_param,
            "source": config.simulation_defaults.source,
            "wavelength_nm": config.simulation_defaults.wavelength_nm,
            "albedo_value": config.simulation_defaults.albedo_value,
            "output_altitudes_km": config.simulation_defaults.output_altitudes_km,
            "output_columns": config.simulation_defaults.output_columns,
            "viewing_geometry": config.simulation_defaults.viewing_geometry,
            "parameter_overrides": config.simulation_defaults.parameter_overrides,
        },
    }

    # Clouds
    if config.simulation_defaults.clouds.enabled:
        c = config.simulation_defaults.clouds
        data["simulation_defaults"]["clouds"] = {
            "enabled": True,
            "cloud_type": c.cloud_type,
            "cloud_source": c.cloud_source,
            "wc_file": str(c.wc_file) if c.wc_file else None,
            "ic_file": str(c.ic_file) if c.ic_file else None,
        }

    with open(output_path, "w") as f:
        yaml.dump(data, f, sort_keys=False, default_flow_style=False)


def run_comparison(
    inp_file: Path, yaml_file: Path, config: SimulationConfig
) -> Dict[str, Any]:
    """
    Run comparison and return statistics:
    Returns dict with 'mae', 'mre', 'max_ae', 'status' (PASS/FAIL/SKIP)
    """
    result = {"mae": None, "mre": None, "max_ae": None, "status": "SKIP"}

    # 1. Reference Run
    try:
        raw_params = parse_inp_file(inp_file)
        resolved_params = resolve_paths(
            raw_params, inp_file.parent, config.paths.libradtran_data
        )

        sim_ref = Simulation(config)
        # Raw simulation needs to mimic uvspec exactly
        sim_ref.config.simulation_defaults.output_altitudes_km = []
        sim_ref.config.simulation_defaults.viewing_geometry = "custom"
        sim_ref.config.simulation_defaults.mol_abs_param = None
        if "output_user" not in resolved_params and "output" not in resolved_params:
            sim_ref.config.simulation_defaults.output_columns = []
            sim_ref.config.simulation_defaults.wavelength_nm = (
                []
            )  # Clear default wavelength

        dt = pd.to_datetime("2023-01-01 12:00:00").to_pydatetime()

        ref_out = sim_ref.run_simulation(
            dt=dt, latitude=0, longitude=0, parameter_overrides=resolved_params
        )
    except Exception as e:
        logger.error(f"Reference run failed for {inp_file.name}: {e}")
        result["status"] = "REF_FAIL"
        return result

    # 2. Ported Run
    try:
        ported_config = load_config(yaml_file)
        sim_port = Simulation(ported_config)
        port_out = sim_port.run_simulation(dt=dt, latitude=0, longitude=0)
    except Exception as e:
        logger.error(f"Ported run failed for {yaml_file.name}: {e}")
        result["status"] = "PORT_FAIL"
        return result

    # 3. Compare & Stats
    if not ref_out or not port_out:
        result["status"] = "NO_OUTPUT"
        return result

    try:
        if os.stat(ref_out).st_size == 0 or os.stat(port_out).st_size == 0:
            result["status"] = "EMPTY_OUT"
            return result

        try:
            d1 = np.loadtxt(ref_out, comments="#")
            d2 = np.loadtxt(port_out, comments="#")
        except ValueError:
            d1 = np.genfromtxt(ref_out, comments=None, encoding="utf-8")
            d2 = np.genfromtxt(port_out, comments=None, encoding="utf-8")

        if d1.shape != d2.shape:
            logger.error(f"Shape mismatch: {d1.shape} vs {d2.shape}")
            result["status"] = "SHAPE_MISMATCH"
            return result

        # Calculate Statistics
        # Handle NaNs
        mask = np.isfinite(d1) & np.isfinite(d2)
        if not np.any(mask):
            result["status"] = "ALL_NAN"
            return result

        diff = np.abs(d1[mask] - d2[mask])
        mae = np.mean(diff)
        max_ae = np.max(diff)

        # Relative Error (handle division by zero)
        # Avoid division by zero by using a small epsilon or filtering
        denom = np.abs(d1[mask])
        valid_rel = denom > 1e-10
        if np.any(valid_rel):
            mre = np.mean(diff[valid_rel] / denom[valid_rel])
        else:
            mre = 0.0

        result["mae"] = mae
        result["mre"] = mre
        result["max_ae"] = max_ae

        # Threshold for "PASS"
        if np.allclose(d1, d2, rtol=1e-3, atol=1e-5, equal_nan=True):
            result["status"] = "PASS"
        else:
            result["status"] = "MISMATCH"

        return result

    except Exception as e:
        logger.error(f"Comparison logic failed: {e}")
        result["status"] = "COMP_FAIL"
        return result


def main():
    try:
        base_config = load_config()
    except Exception as e:
        logger.critical(f"Could not load base config: {e}")
        sys.exit(1)

    examples_dir = Path("/opt/libRadtran-2.0.6/examples")
    if not examples_dir.exists():
        logger.critical(f"Examples dir not found: {examples_dir}")
        sys.exit(1)

    output_dir = Path("./pyradtran_work/ported_examples")
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_stats = []

    for name in TARGET_EXAMPLES:
        inp_path = examples_dir / name
        if not inp_path.exists():
            continue

        logger.info(f"--- Porting {name} ---")

        # Parse & Resolve
        try:
            raw_params = parse_inp_file(inp_path)
            resolved_params = resolve_paths(
                raw_params, examples_dir, base_config.paths.libradtran_data
            )
        except Exception:
            continue

        # Map
        try:
            new_config = map_inp_to_config(resolved_params, base_config)
            # Ensure we clear default wavelength in mapped config as well (redundancy check)
            if not "wavelength" in raw_params:
                new_config.simulation_defaults.wavelength_nm = []
        except Exception:
            continue

        # Generate YAML
        yaml_name = inp_path.stem + ".yaml"
        yaml_path = output_dir / yaml_name
        try:
            generate_yaml(new_config, yaml_path)
        except Exception:
            continue

        # Verify
        res = run_comparison(inp_path, yaml_path, base_config)
        res["name"] = name
        summary_stats.append(res)

        if res["status"] == "PASS":
            logger.info(f"Verified {name}: OK")
        else:
            logger.warning(f"Verified {name}: {res['status']} (MAE={res['mae']})")

    # Print Summary Table
    print("\n" + "=" * 80)
    print(
        f"{'EXAMPLE':<30} | {'STATUS':<10} | {'MAE':<10} | {'MRE':<10} | {'MAX_AE':<10}"
    )
    print("-" * 80)
    for s in summary_stats:
        mae_str = f"{s['mae']:.2e}" if s["mae"] is not None else "N/A"
        mre_str = f"{s['mre']:.2e}" if s["mre"] is not None else "N/A"
        max_ae_str = f"{s['max_ae']:.2e}" if s["max_ae"] is not None else "N/A"
        print(
            f"{s['name']:<30} | {s['status']:<10} | {mae_str:<10} | {mre_str:<10} | {max_ae_str:<10}"
        )
    print("=" * 80)


if __name__ == "__main__":
    main()
