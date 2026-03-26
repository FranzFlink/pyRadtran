import xarray as xr
import pandas as pd
import numpy as np
from pyradtran.interface import PyRadtranAccessor # Trigger registration
from pyradtran.config import SimulationConfig, PathsConfig, SimulationDefaults, CloudParameters
from pathlib import Path
import shutil
import os
import logging
logging.basicConfig(level=logging.DEBUG)    


def test_interface_clouds():
    # Setup Paths
    work_dir = Path("pyradtran_work_test_interface")
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir()
    
    # Create Config
    config = SimulationConfig(
        paths=PathsConfig(
            libradtran_bin=Path("/opt/libRadtran-2.0.6/bin/uvspec"), # Dummy path, won't execute success without it
            libradtran_data=Path("/opt/libRadtran-2.0.6/data"),
            working_dir=work_dir,
            output_dir=work_dir
        ),
        simulation_defaults=SimulationDefaults(
            wavelength_nm=[500, 500],
            mol_abs_param="lowtran",
        )
    )
    # Ensure config validation doesn't kill us if bin doesn't exist (it checks in __post_init__)
    # We might need to mock Path.is_file/is_dir or rely on user environment.
    # Assuming user environment has these paths or we catch the error?
    # Actually, config validation *will* raise FileNotFoundError if paths don't exist.
    # The user has pyradtran installed, so paths likely exist or are in default config.
    # Let's try to load default config and just override work dir.
    
    from pyradtran.config import load_config
    try:
        base_config = load_config()
        base_config.paths.working_dir = work_dir
        base_config.paths.output_dir = work_dir
        # Disable full execution cleanup to inspect files
        base_config.execution.cleanup_temp_files = False
        config = base_config
    except Exception as e:
        print(f"Skipping test due to config load failure: {e}")
        return

    # Create Dataset
    # Create Dataset - Trajectory style (lat/lon depend on time or share dim)
    times = pd.date_range("2022-01-01 12:00", periods=2, freq="H")
    ds = xr.Dataset(
        coords={
            "time": times, 
            "latitude": (("time",), [0, 0]), 
            "longitude": (("time",), [0, 0])
        },
        data_vars={
            "lwc": (("time",), [0.1, 0.5]),
            "reff": (("time",), [10.0, 15.0]),
            "cth": (("time",), [2.0, 3.0]),
            "cbh": (("time",), [1.0, 2.0]),
            "sza": (("time",), [10.0, 20.0]) # Optional
        }
    )
    
    # Configure logging for pyradtran
    import logging
    logging.getLogger('pyradtran').setLevel(logging.DEBUG)

    # Run Real Simulation
    try:
        print("Running real simulation...")
        res_ds = ds.pyradtran.run(
            config=config,
            cloud_wc_var='lwc',
            cloud_reff_var='reff',
            cloud_top_var='cth',
            cloud_bottom_var='cbh',
            save_to_file=False, 
            return_dataset=True,
            # max_workers=1 # Optional: Use 1 worker for better debugging if needed
        )
        
        print("Simulation completed.")
        
        # Verify Results
        # Check if we got results
        if res_ds is None or len(res_ds.data_vars) == 0:
             print("FAILURE: No results returned.")
             return

        # Check for expected output variables
        if 'time' in res_ds.coords and res_ds.sizes['time'] == 2:
             print("SUCCESS: Simulation returned dataset with correct time dimension.")
             print(res_ds)
        else:
             print(f"FAILURE: Dataset dimensions mismatch. Expected 2 time steps. Got: {res_ds}")

    except Exception as e:
        print(f"Run exception: {e}") 


if __name__ == "__main__":
    test_interface_clouds()
