import xarray as xr
import pandas as pd
import numpy as np
from pyradtran.interface import PyRadtranAccessor
from pyradtran.config import SimulationConfig, PathsConfig, SimulationDefaults
import logging

logging.basicConfig(level=logging.DEBUG)

def test_inspect_feature():
    times = pd.date_range("2022-01-01 12:00", periods=2, freq="h")
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
            "cbh": (("time",), [1.0, 2.0])
        }
    )
    
    # 1. Inspect first point
    print("\n--- Inspecting Point 0 (LWC=0.1) ---")
    content_0 = ds.pyradtran.inspect_cloud_file(
        selector={'time': times[0]},
        cloud_wc_var='lwc',
        cloud_top_var='cth',
        cloud_bottom_var='cbh',
        cloud_reff_var='reff'
    )
    print(content_0)
    
    # Verify content
    if "2.000000 0.100000 10.000000" in content_0 and "1.000000 0.100000 10.000000" in content_0:
        print("SUCCESS: Point 0 content matches expected values.")
    else:
        print(f"FAILURE: Point 0 Content mismatch.\nGot:\n{content_0}")

    # 2. Inspect second point
    print("\n--- Inspecting Point 1 (LWC=0.5) ---")
    content_1 = ds.pyradtran.inspect_cloud_file(
        selector={'time': times[1]},
        cloud_wc_var='lwc',
        cloud_top_var='cth',
        cloud_bottom_var='cbh',
        cloud_reff_var='reff'
    )
    print(content_1)
    
     # Verify content
    if "3.000000 0.500000 15.000000" in content_1 and "2.000000 0.500000 15.000000" in content_1:
         print("SUCCESS: Point 1 content matches expected values.")
    else:
         print(f"FAILURE: Point 1 Content mismatch.\nGot:\n{content_1}")

    # 3. Test parameter override dictionary
    print("\n--- Inspecting Custom Override ---")
    custom_profile = {
        'z': [5.0, 4.0],
        'lwc': [1.0, 1.0],
        'reff': [20.0, 20.0]
    }
    content_custom = ds.pyradtran.inspect_cloud_file(
        selector={'time': times[0]},
        parameter_overrides={'wc_file': custom_profile}
    )
    print(content_custom)
    if "5.000000 1.000000 20.000000" in content_custom:
         print("SUCCESS: Custom override content correct.")
    else:
         print(f"FAILURE: Custom override mismatch\n{content_custom}")


if __name__ == "__main__":
    test_inspect_feature()
