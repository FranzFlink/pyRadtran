.. pyradtran documentation master file, created by
   sphinx-quickstart on Tue May  6 02:02:10 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

===================
PyRadtran
===================

A Python interface for the libRadtran Radiative Transfer Model.

PyRadtran provides a Pythonic interface to the libRadtran radiative transfer model, which is used 
for modeling radiation transport in Earth's atmosphere. This package aims to make it easier to 
set up, run, and analyze radiative transfer simulations with support for both solar and thermal infrared radiation.

.. image:: https://img.shields.io/github/license/FranzFlink/pyRadtran
   :target: https://github.com/FranzFlink/pyRadtran/blob/main/LICENSE
   :alt: License

.. image:: https://img.shields.io/github/stars/FranzFlink/pyRadtran
   :target: https://github.com/FranzFlink/pyRadtran
   :alt: GitHub stars

.. image:: https://readthedocs.org/projects/pyradtran/badge/?version=latest
   :target: https://franzflink.github.io/pyRadtran/
   :alt: Documentation Status

Key Features
============

🌞 **Solar & Thermal Simulations**
   Support for both solar radiation (UV, visible, near-IR) and thermal infrared simulations
🗂️ **Flexible Configuration**
   YAML-based configuration system with sensible defaults and easy customization
📊 **xarray Integration**
   Native support for xarray datasets with automatic coordinate handling and metadata preservation
🌍 **Atmospheric Profiles**
   Support for standard atmospheric profiles, radiosondes, and ERA5 reanalysis data
⚡ **multi-core Processing**
   Batch processing capabilities to run multiple simulations at once

.. toctree::
   :maxdepth: 2
   :caption: Getting Started:

   installation
   usage

.. toctree::
   :maxdepth: 2
   :caption: User Guide:

   examples
   notebooks

.. toctree::
   :maxdepth: 2
   :caption: Reference:

   api

.. toctree::
   :maxdepth: 1
   :caption: Development:

   contributing
   changelog

Installation
===========

See the :doc:`installation` page for installation instructions.

Quick Start
===========

Here's a simple example to get you started:

.. code-block:: python

from pyradtran.interface import PyRadtranAccessor  # This should register the accessor automatically
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from pathlib import Path
import pandas as pd

# Load the configuration from the YAML file
config_path = Path('../config/spectral_config.yaml')
import logging
# change to "DEBUG" if you want to see more output
logging.getLogger('pyradtran').setLevel(logging.CRITICAL)

N_timesteps = 24

# stationary simulation at constant altitude, latitude, and longitude but with varying time
ds = xr.Dataset(
    coords={
        'time' : pd.date_range('2025-04-04', periods=N_timesteps, freq='h'),
        'latitude' : ('time', [61.0] * N_timesteps),
        'longitude' : ('time', [22.0] * N_timesteps),
        'altitude' : ('altitude', [10]),
    }
)

# Run a spectral simulation for a single point
ds_sim = ds.pyradtran.run_uvspec(
    config_path=config_path,
    return_dataset=True,
    save_to_file=True,
)

See :doc:`examples` for more detailed examples.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

