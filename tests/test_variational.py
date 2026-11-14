import logging
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pyradtran.config import PathsConfig, SimulationConfig, SimulationDefaults
from pyradtran.interface import PyRadtranAccessor
from pyradtran.params import Var

logging.basicConfig(level=logging.DEBUG)


from helpers import has_libradtran


@pytest.mark.integration
@pytest.mark.skipif(not has_libradtran(), reason="LibRadtran not available")
def test_variational_logic():
    work_dir = Path("pyradtran_work_variational")
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir()

    # User's example logic:
    # "return a ds_sim object with dims (time, altitude, lwc, sza)"
    # Note: 'altitude' is an output dimension from uvspec.
    # 'lwc' and 'sza' are input dimensions to vary.

    times = pd.date_range("2022-01-01 12:00", periods=2, freq="h")

    # Constructing the Dataset exactly as user described
    ds = xr.Dataset(
        coords={
            "time": times,
            "latitude": (("time",), [0, 0]),
            "longitude": (("time",), [0, 0]),
            # altitude coordinate in input is often used for zout config,
            # but user wants it as a dimension in output.
            # In interface.py, if 'altitude' is in coords, it sets zout levels.
            "altitude": (("altitude",), [0, 1]),  # Reduced to 2 levels for speed
            "lwc_dim": [0.1, 0.5],
            "sza_dim": [50, 60],
        },
        data_vars={
            # LWC varies along own dim
            "lwc": (("lwc_dim",), [0.1, 0.5]),
            # Reff varies along lwc dim (linked)
            "reff": (("lwc_dim",), [10.0, 15.0]),
            # CTH/CBH vary along lwc dim
            "cth": (("lwc_dim",), [2.0, 3.0]),
            "cbh": (("lwc_dim",), [1.0, 2.0]),
            # SZA varies independently
            "sza": (("sza_dim",), [50, 60]),
        },
    )

    # Config
    config = SimulationConfig(
        paths=PathsConfig(
            libradtran_bin=Path("/opt/libRadtran-2.0.6/bin/uvspec"),
            libradtran_data=Path("/opt/libRadtran-2.0.6/data"),
            working_dir=work_dir,
            output_dir=work_dir,
        ),
        simulation_defaults=SimulationDefaults(
            wavelength_nm=[500, 500],
            mol_abs_param="lowtran",
        ),
    )

    print("Running variational simulation...")
    ds_sim = ds.pyradtran.run(
        config=config,
        cloud_wc_var="lwc",
        cloud_reff_var="reff",
        cloud_top_var="cth",
        cloud_bottom_var="cbh",
        # 'sza' is a dataset variable (varies along sza_dim); Var(...) tells
        # the resolver to pull a fresh value per simulation point instead of
        # treating "sza" as a literal.
        params={"sza": Var("sza")},
        save_to_file=False,
        return_dataset=True,
    )

    print("Simulation completed.")
    print(ds_sim)

    # Expected dimensions: time (2) * lwc (2) * sza (2) = 8 simulations
    # Result dimensions: time, lwc_dim, sza_dim, altitude, ...

    expected_dims = {"time", "lwc_dim", "sza_dim", "altitude"}
    if expected_dims.issubset(set(ds_sim.dims)):
        print("SUCCESS: Output has expected variational dimensions.")
    else:
        print(
            f"FAILURE: Missing dimensions. Got {ds_sim.dims}, expected subset {expected_dims}"
        )


if __name__ == "__main__":
    test_variational_logic()
