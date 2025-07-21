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

🐍 **Pythonic Interface**
   Clean, intuitive Python API with excellent integration into the scientific Python ecosystem

📊 **xarray Integration**
   Native support for xarray datasets with automatic coordinate handling and metadata preservation

🌍 **Atmospheric Profiles**
   Support for standard atmospheric profiles, radiosondes, and ERA5 reanalysis data

⚡ **Batch Processing**
   Efficient batch processing capabilities for time series and spatial analyses

🔬 **Scientific Accuracy**
   Built on the robust and well-validated libRadtran radiative transfer model

📈 **Analysis Tools**
   Built-in utilities for result visualization and analysis

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

    from datetime import datetime
    from pyradtran.config import load_config
    from pyradtran.core import Simulation
    from pyradtran.io import read_output
    import matplotlib.pyplot as plt
    
    # Load configuration
    config = load_config()
    
    # Create a simulation
    sim = Simulation(config)
    
    # Set up and run a simulation
    dt = datetime.now()
    latitude = 78.9  # North latitude in degrees
    longitude = 11.9  # East longitude in degrees
    
    # Run the simulation
    output_file = sim.run(dt, latitude, longitude)
    
    # Process results
    if output_file:
        result = read_output(output_file)
        # Plot results
        plt.figure(figsize=(10, 6))
        plt.plot(result['wavelength'], result['edir'])
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Direct Irradiance')
        plt.title('Spectral Direct Irradiance')
        plt.grid(True)
        plt.show()

See :doc:`examples` for more detailed examples.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

