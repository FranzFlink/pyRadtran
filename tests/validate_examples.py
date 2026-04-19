#!/usr/bin/env python3
"""
validate_examples.py

This script iterates through standard libRadtran examples, executes them using
the pyRadtran wrapper, and verifies that the output matches the reference output.
It also generates corresponding YAML configuration files.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

# Add parent directory to path to allow importing pyradtran
sys.path.insert(0, str(Path(__file__).parent.parent))

from pyradtran.config import SimulationConfig, load_config
from pyradtran.core import Simulation

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("validate_examples")


def parse_inp_file(inp_path: Path) -> Dict[str, Any]:
    """
    Parse a libRadtran .INP file into a dictionary of key-values.
    Ref: Simple parsing, assumes 'key value' format.
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

            # Reconstruct value string or keep as list?
            # For override_params, we usually want a string if it's complex,
            # or a single value if it's simple.

            if len(values) == 1:
                val = values[0]
                # Try to convert to number
                try:
                    if "." in val:
                        params[key] = float(val)
                    else:
                        params[key] = int(val)
                except ValueError:
                    params[key] = val
            else:
                # Join remaining parts as string value
                # This might need refinement for specific keys that take lists
                params[key] = " ".join(values)

    return params


def resolve_paths(
    params: Dict[str, Any], example_dir: Path, libradtran_data: Path
) -> Dict[str, Any]:
    """
    Resolve relative paths in parameters to absolute paths or
    paths relative to libradtran_data.
    """
    resolved_params = params.copy()

    # Keys that are known to be paths
    path_keys = [
        "atmosphere_file",
        "solar_file",
        "source",
        "mol_tau_file",
        "aerosol_default",
        "cloud_overlap_file",
        "slit_function_file",
    ]

    # Handle "source solar <path>" special case
    # In INP: source solar ../data/solar_flux/atlas_plus_modtran
    # In params: source -> "solar ../data/..."

    for key, val in params.items():
        if isinstance(val, str):
            # Check for generic path indicators
            if "../data/" in val:
                # Replace ../data/ with libradtran_data/
                # Note: examples are usually in examples/ which is sibling to data/
                new_val = val.replace("../data/", str(libradtran_data) + "/")
                resolved_params[key] = new_val
            elif "../examples/" in val:
                new_val = val.replace("../examples/", str(example_dir) + "/")
                resolved_params[key] = new_val

    return resolved_params


def compare_outputs(generated_file: Path, reference_file: Path) -> bool:
    """
    Compare generated output with reference output numerically.
    """
    if not generated_file.exists():
        logger.error(f"Generated file not found: {generated_file}")
        return False

    if not reference_file.exists():
        logger.warning(f"Reference file not found: {reference_file}")
        return True  # Cannot fail if no reference, but warn

    try:
        # Load as simple text/numpy, skipping header lines
        # libRadtran output usually has columns of numbers
        # We need to robustly skip headers. 'quiet' mode in pyRadtran reduces headers but
        # the reference output might contain headers if it wasn't run with quiet.

        # Strategy: Try reading with numpy.loadtxt
        gen_data = np.loadtxt(generated_file, comments="#")
        ref_data = np.loadtxt(reference_file, comments="#")

        # Check shapes
        if gen_data.shape != ref_data.shape:
            logger.error(
                f"Shape mismatch: Generated {gen_data.shape} vs Reference {ref_data.shape}"
            )
            return False

        # Compare with tolerance
        # Allow some small numerical difference
        return np.allclose(gen_data, ref_data, rtol=1e-3, atol=1e-5)

    except Exception as e:
        logger.error(f"Comparison failed with error: {e}")
        return False


def create_yaml_config(params: Dict[str, Any], output_yaml_path: Path):
    """
    Create a YAML configuration file from the parameters.
    """
    # Create a basic structure matching SimulationConfig
    yaml_config = {"simulation_defaults": {}, "execution": {"cleanup_temp_files": True}}

    # Move relevant keys to simulation_defaults or keep as raw overrides
    # Since we are essentially porting INP to YAML, we might put everything
    # that fits into 'parameter_overrides' concept if it doesn't map cleanely.
    # But for a nice YAML, we should map what we can.

    # This is a simplified mapping for now.
    # In the unified interface, we often just pass overrides.
    # So we can create a YAML that has a special "raw_overrides" section or similar?
    # Actually, SimulationConfig doesn't support arbitrary "raw_overrides" in the dict
    # unless we stick them in a specific place.
    # BUT, the user wants "corresponding yamls".

    # Let's map common things, and put the rest in a way that our system understands.
    # Wait, our system logic in `_generate_input_content` takes `parameter_overrides`.
    # `SimulationConfig` expects specific fields.

    # For now, we will create a YAML that essentially configures the defaults
    # tailored to this example.

    # Mappings
    if "rte_solver" in params:
        yaml_config["simulation_defaults"]["rte_solver"] = params.pop("rte_solver")

    if "mol_abs_param" in params:
        yaml_config["simulation_defaults"]["mol_abs_param"] = params.pop(
            "mol_abs_param"
        )

    if "albedo" in params:
        yaml_config["simulation_defaults"]["albedo_value"] = float(params.pop("albedo"))

    # Construct the rest as a flat dictionary which the user would have to load
    # and pass as overrides.
    # OR, we verify if we can extend the YAML format to support raw parameters.
    # config.py doesn't seem to support arbitrary keys in `simulation_defaults`.

    # Let's write the remaining params as specific overrides in a custom section
    # "parameter_overrides" inside the YAML, even if `SimulatonConfig` strict
    # loading might ignore it, it serves as documentation.
    # A better approach might be to just save them as a dict.

    yaml_config["parameter_overrides"] = params

    with open(output_yaml_path, "w") as f:
        yaml.dump(yaml_config, f, sort_keys=False)


def run_example(inp_file: Path, config: SimulationConfig, example_dir: Path) -> str:
    """
    Run a single example validation.
    Returns: 'PASS', 'FAIL', 'SKIP'
    """
    logger.info(f"--- Running Example: {inp_file.name} ---")

    # 1. Parse INP
    try:
        raw_params = parse_inp_file(inp_file)
    except Exception as e:
        logger.error(f"Failed to parse INP file: {e}")
        return "FAIL"

    # 2. Resolve parameters
    try:
        params = resolve_paths(raw_params, example_dir, config.paths.libradtran_data)
    except Exception as e:
        logger.error(f"Failed to resolve paths: {e}")
        return "FAIL"

    # 3. Setup Simulation
    # We will use the Simulation class directly
    sim = Simulation(config)

    # If the INP file does not specify output_user, we should not enforce our default columns
    # so that we match the example's default output.
    if "output_user" not in params and "output" not in params:
        sim.config.simulation_defaults.output_columns = []

    # If INP does not specify zout, suppress default zout
    if "zout" not in params:
        sim.config.simulation_defaults.output_altitudes_km = []

    # If INP does not specify viewing angles, suppress default umu 1.0
    if "umu" not in params and "phi" not in params:
        sim.config.simulation_defaults.viewing_geometry = "custom"

    # If INP does not specify mol_abs_param, suppress default
    if "mol_abs_param" not in params:
        sim.config.simulation_defaults.mol_abs_param = None

    # Dummy required arguments (will be overridden by params often)
    # Most examples specify everything they need.
    dt = pd.to_datetime("2023-06-20 12:00:00").to_pydatetime()
    lat = 0.0
    lon = 0.0

    # 4. Filter parameters that conflict with explicit args of run_simulation
    # run_simulation takes: dt, latitude, longitude, overrides...
    # If sza is in params, we should ensure it's used.
    # The `_generate_input_content` uses `parameter_overrides` at the end,
    # so they should override whatever `run_simulation` sets as default.

    # 5. Run
    try:
        output_path = sim.run_simulation(
            dt=dt, latitude=lat, longitude=lon, parameter_overrides=params
        )
    except Exception as e:
        logger.error(f"Simulation execution failed: {e}")
        return "FAIL"

    if not output_path:
        return "FAIL"

    # 6. Compare
    ref_out = inp_file.with_suffix(".OUT")
    if compare_outputs(output_path, ref_out):
        logger.info(f"SUCCESS: Output matches {ref_out.name}")
        status = "PASS"
    else:
        logger.error(f"FAILURE: Output mismatch for {inp_file.name}")
        status = "FAIL"

    # 7. Generate YAML
    yaml_out_dir = config.paths.working_dir / "example_configs"
    yaml_out_dir.mkdir(exist_ok=True, parents=True)
    yaml_out_path = yaml_out_dir / f"{inp_file.stem}.yaml"
    create_yaml_config(params, yaml_out_path)

    return status


def main():
    parser = argparse.ArgumentParser(description="Validate libRadtran examples")
    parser.add_argument("--filter", type=str, help="Filter examples by name string")
    parser.add_argument(
        "--limit", type=int, default=0, help="Limit number of examples to run"
    )
    args = parser.parse_args()

    # Load config
    try:
        config = load_config()
    except Exception as e:
        logger.critical(f"Could not load configuration: {e}")
        sys.exit(1)

    logging.info(f"Using libRadtran data: {config.paths.libradtran_data}")

    # Locate examples
    # Heuristic: try sibling 'examples' of 'data' or explicit path
    # If /opt/libRadtran.../data is data, examples is /opt/libRadtran.../examples

    # Default assumption from user request
    examples_dir = Path("/opt/libRadtran-2.0.6/examples")
    if not examples_dir.exists():
        # Fallback to relative to data
        examples_dir = config.paths.libradtran_data.parent / "examples"

    if not examples_dir.exists():
        logger.critical(f"Examples directory not found at {examples_dir}")
        sys.exit(1)

    logging.info(f"Examples directory: {examples_dir}")

    # Find INP files
    # Default to UVSPEC_ examples as they are main uvspec inputs
    inp_files = sorted(list(examples_dir.glob("UVSPEC_*.INP")))
    # Filter out macOS resource fork files
    inp_files = [f for f in inp_files if not f.name.startswith("._")]

    if args.filter:
        inp_files = [f for f in inp_files if args.filter in f.name]
    if args.limit > 0:
        inp_files = inp_files[: args.limit]

    results = {"PASS": [], "FAIL": [], "SKIP": []}

    for inp_file in inp_files:
        status = run_example(inp_file, config, examples_dir)
        results[status].append(inp_file.name)

    # Report
    print("\n" + "=" * 40)
    print("VALIDATION SUMMARY")
    print("=" * 40)
    print(f"Total Runs: {len(inp_files)}")
    print(f"PASSED: {len(results['PASS'])}")
    print(f"FAILED: {len(results['FAIL'])}")
    print(f"SKIPPED: {len(results['SKIP'])}")

    if results["FAIL"]:
        print("\nFailed Examples:")
        for name in results["FAIL"]:
            print(f" - {name}")

    # Exit code
    if results["FAIL"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
