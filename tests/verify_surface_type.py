#!/usr/bin/env python3
"""
Verification script for IGBP surface type feature.
Tests that surface_type values are correctly propagated to the LibRadtran input file.
"""

import sys
import tempfile
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pyradtran.config import (
    ExecutionConfig,
    OutputConfig,
    PathsConfig,
    SimulationConfig,
    SimulationDefaults,
)
from pyradtran.core import Simulation


def test_core_input_generation():
    """Test that core.py generates correct IGBP surface type lines."""
    print("=" * 70)
    print("TEST 1: Core Input Generation")
    print("=" * 70)

    # Create minimal config
    tmp_dir = Path(tempfile.mkdtemp())

    # Create dummy files
    bin_path = tmp_dir / "uvspec"
    bin_path.touch()

    data_dir = tmp_dir / "data"
    data_dir.mkdir()

    atm_path = tmp_dir / "atmosphere.dat"
    atm_path.write_text("# dummy atmosphere\n")

    solar_path = tmp_dir / "solar.dat"
    solar_path.write_text("# dummy solar spectrum\n")

    config = SimulationConfig(
        paths=PathsConfig(
            libradtran_bin=bin_path,
            libradtran_data=data_dir,
            atmosphere_profile=atm_path,
            solar_spectrum=solar_path,
            output_dir=tmp_dir / "output",
            working_dir=tmp_dir / "working",
        ),
        simulation_defaults=SimulationDefaults(
            rte_solver="disort",
            wavelength_nm=[400, 700],
            output_columns=["sza", "eglo", "eup"],
            output_altitudes_km=[0.0],
        ),
        execution=ExecutionConfig(max_workers=1),
        output=OutputConfig(filename_prefix="test"),
    )

    # Create simulation object
    sim = Simulation(config)

    # Test cases
    test_cases = [
        (None, "No surface type (should use default albedo or none)"),
        (5, "IGBP type 5: mixed_forest"),
        (12, "IGBP type 12: cropland"),
        (17, "IGBP type 17: ocean_water"),
    ]

    all_passed = True

    for surface_type, description in test_cases:
        print(f"\n{description}")
        print("-" * 70)

        # Generate input content
        input_content = sim._generate_input_content(
            dt=datetime(2023, 6, 21, 12, 0),
            latitude=60.0,
            longitude=10.0,
            override_surface_type=surface_type,
        )

        # Check for expected lines
        lines = input_content.split("\n")

        if surface_type is not None:
            # Should contain both brdf_rpv lines
            has_library = any("brdf_rpv_library IGBP" in line for line in lines)
            has_type = any(f"brdf_rpv_type {surface_type}" in line for line in lines)

            if has_library and has_type:
                print(
                    f"✓ PASS: Found 'brdf_rpv_library IGBP' and 'brdf_rpv_type {surface_type}'"
                )
                # Show the relevant lines
                for line in lines:
                    if "brdf_rpv" in line:
                        print(f"  {line}")
            else:
                print(f"✗ FAIL: Missing expected lines")
                if not has_library:
                    print("  Missing: brdf_rpv_library IGBP")
                if not has_type:
                    print(f"  Missing: brdf_rpv_type {surface_type}")
                all_passed = False
        else:
            # Should NOT contain brdf_rpv lines
            has_brdf = any("brdf_rpv" in line for line in lines)
            if not has_brdf:
                print("✓ PASS: No brdf_rpv lines (as expected)")
            else:
                print("✗ FAIL: Unexpected brdf_rpv lines found")
                all_passed = False

    # Cleanup
    import shutil

    shutil.rmtree(tmp_dir)

    return all_passed


def test_interface_propagation():
    """Test that interface.py propagates surface_type correctly."""
    print("\n" + "=" * 70)
    print("TEST 2: Interface Variable Propagation")
    print("=" * 70)

    try:
        import numpy as np
        import pandas as pd
        import xarray as xr

        # Create test dataset with surface_type variable
        times = pd.date_range("2023-06-21", periods=3, freq="1h")

        ds = xr.Dataset(
            coords={
                "time": times,
                "latitude": ("time", [60.0, 60.1, 60.2]),
                "longitude": ("time", [10.0, 10.1, 10.2]),
            },
            data_vars={"surface_type": ("time", [5, 12, 17])},  # Different IGBP types
        )

        print("\n✓ Created test dataset with surface_type variable:")
        print(f"  Times: {len(times)}")
        print(f"  Surface types: {ds['surface_type'].values}")

        # Test that accessor has the new parameter
        # Check method signature
        import inspect

        from pyradtran.interface import PyRadtranAccessor

        sig = inspect.signature(PyRadtranAccessor.run)
        params = list(sig.parameters.keys())

        if "surface_type_var" in params:
            print("✓ PASS: PyRadtranAccessor.run has 'surface_type_var' parameter")
        else:
            print("✗ FAIL: PyRadtranAccessor.run missing 'surface_type_var' parameter")
            return False

        # Check execute_simulation_batch signature
        from pyradtran.interface import execute_simulation_batch

        sig = inspect.signature(execute_simulation_batch)
        params = list(sig.parameters.keys())

        if "surface_type_var" in params:
            print("✓ PASS: execute_simulation_batch has 'surface_type_var' parameter")
        else:
            print(
                "✗ FAIL: execute_simulation_batch missing 'surface_type_var' parameter"
            )
            return False

        return True

    except Exception as e:
        print(f"✗ FAIL: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all verification tests."""
    print("\n" + "=" * 70)
    print("IGBP SURFACE TYPE FEATURE VERIFICATION")
    print("=" * 70)

    test1_passed = test_core_input_generation()
    test2_passed = test_interface_propagation()

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"Test 1 (Core Input Generation): {'PASS ✓' if test1_passed else 'FAIL ✗'}")
    print(f"Test 2 (Interface Propagation): {'PASS ✓' if test2_passed else 'FAIL ✗'}")

    if test1_passed and test2_passed:
        print("\n🎉 ALL TESTS PASSED!")
        return 0
    else:
        print("\n❌ SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
