.. pyradtran documentation master file, created by
   sphinx-quickstart on Tue May  6 02:02:10 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

===================
pyradtran
===================

A Python interface for the libRadtran Radiative Transfer Model.

pyradtran provides a Pythonic interface to the libRadtran radiative transfer model, which is used 
for modeling radiation transport in Earth's atmosphere. This package aims to make it easier to 
set up, run, and analyze radiative transfer simulations.

Features
========

* Pythonic interface to libRadtran
* Robust configuration management
* Support for radiative transfer simulations with various parameters
* Integration with scientific Python tools
* Input/output utilities

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   usage
   examples
   api
   notebooks

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

