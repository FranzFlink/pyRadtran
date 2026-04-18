import xarray as xr
import pandas as pd
import numpy as np
from pyradtran.interface import PyRadtranAccessor
from pyradtran.config import SimulationConfig, PathsConfig, SimulationDefaults
from pathlib import Path
import shutil
import logging

logging.basicConfig(level=logging.DEBUG)

def reproduce_no_clouds():
    work_dir = Path("pyradtran_work_reproduction")
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir()

    # Create user dataset-like structure
    times = pd.date_range("2022-01-01 12:00", periods=2, freq="h")
    ds = xr.Dataset(
        coords={
            "time": times, 
            "latitude": (("time",), [0, 0]), 
            "longitude": (("time",), [0, 0]),
            # altitude dim is not strictly needed for this test unless we use it
            "altitude": (("altitude",), [0, 1]) 
        },
        data_vars={
            "lwc": (("time",), [0.1, 0.5]),
            "reff": (("time",), [10.0, 15.0]),
            "cth": (("time",), [2.0, 3.0]),
            "cbh": (("time",), [1.0, 2.0]),
            "sza": (("time",), [10.0, 20.0])
        }
    )

    # Config with defaults
    config_defaults = SimulationConfig(
        paths=PathsConfig(
            libradtran_bin=Path("/opt/libRadtran-2.0.6/bin/uvspec"),
            libradtran_data=Path("/opt/libRadtran-2.0.6/data"),
            working_dir=work_dir,
            output_dir=work_dir
        ),
        simulation_defaults=SimulationDefaults(
            wavelength_nm=[500, 500],
            mol_abs_param="lowtran",
        )
    )

    # Load system config to get real paths if possible, but fallback to above
    from pyradtran.config import load_config
    try:
        config = load_config()
        # Override to ensure output goes to our clean dir
        config.paths.working_dir = work_dir
        config.paths.output_dir = work_dir
        config.simulation_defaults.wavelength_nm = [500, 500] 
        config.execution.debug_mode = True # Ensure we see logs
    except:
        config = config_defaults

    print("Running simulation WITHOUT explicit cloud args (expecting NO wc_file)...")
    try:
        # This mirrors user's call: ds.pyradtran.run(config_path='...')
        # We pass config object instead of path for simplicity.
        ds.pyradtran.run(
            config=config,
            # MISSING: cloud_wc_var='lwc', etc.
            save_to_file=False,
            return_dataset=False
        )
    except Exception as e:
        print(f"Run failed (unexpected?): {e}")

    # Check the generated input files
    inp_files = list(work_dir.glob("*.inp"))
    found_wc_file = False
    if not inp_files:
        print("FAILURE: No input files generated.")
        return

    for inp in inp_files:
        content = inp.read_text()
        if "wc_file" in content:
            print(f"FOUND wc_file in {inp.name}")
            found_wc_file = True
        else:
            print(f"NO wc_file in {inp.name}")

    if not found_wc_file:
        print("CONFIRMED: Issue reproduced automatically. No wc_file generated when args are missing.")
    else:
        print("SURPRISE: wc_file WAS generated. Logic might be different than expected.")

if __name__ == "__main__":
    reproduce_no_clouds()
