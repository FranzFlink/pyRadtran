=========
Notebooks
=========

This section provides interactive Jupyter notebooks demonstrating various features and use cases of PyRadtran.
Each notebook is designed to be self-contained and can be run independently to explore different aspects
of radiative transfer modeling.

.. note::
   All notebooks are available in the ``notebooks/`` directory of the PyRadtran repository.
   You can download and run them locally to experiment with the code and modify parameters.

Quick Start Tutorial
====================

Start here if you're new to PyRadtran. This notebook covers the basic workflow for setting up and running
radiative transfer simulations.

.. toctree::
   :maxdepth: 1

   notebooks/quickstart

Atmospheric Radiation Examples
==============================

Radiosonde Data Analysis
------------------------

This notebook demonstrates how to use PyRadtran with real atmospheric measurement data from the HALO-AC3 campaign.
It shows how to compare simulated and measured albedo values at different altitudes.

.. toctree::
   :maxdepth: 1

   notebooks/radiosonde

Thermal Infrared Simulations
-----------------------------

Learn how to perform thermal infrared radiative transfer simulations, including temperature-dependent calculations
and visualization of thermal radiation fields.

.. toctree::
   :maxdepth: 1

   notebooks/thermal

Atmospheric Profiles and Data
=============================

ERA5 Atmospheric Data
---------------------

This notebook shows how to integrate PyRadtran with ERA5 reanalysis data for realistic atmospheric profile modeling.

.. toctree::
   :maxdepth: 1

   notebooks/era5_atmosphere

Surface and Albedo Studies
===========================

Albedo Testing and Validation
-----------------------------

Explore different surface albedo models and their impact on radiative transfer calculations.

.. toctree::
   :maxdepth: 1

   notebooks/albedo_test

Downloading and Running Notebooks
==================================

To run these notebooks locally:

1. **Clone the repository**:

   .. code-block:: bash

      git clone https://github.com/FranzFlink/pyRadtran.git
      cd pyRadtran

2. **Install PyRadtran** (see :doc:`installation` for details):

   .. code-block:: bash

      pip install -e .

3. **Install Jupyter** (if not already installed):

   .. code-block:: bash

      pip install jupyter matplotlib

4. **Navigate to the notebooks directory**:

   .. code-block:: bash

      cd notebooks

5. **Start Jupyter**:

   .. code-block:: bash

      jupyter notebook

6. **Open any notebook** and run the cells to explore PyRadtran's capabilities.

Requirements for Notebooks
===========================

Most notebooks require the following additional packages:

- ``matplotlib`` - For plotting and visualization
- ``numpy`` - For numerical computations  
- ``xarray`` - For handling multi-dimensional data
- ``pandas`` - For data manipulation
- ``netcdf4`` - For reading/writing NetCDF files

Install these with:

.. code-block:: bash

   pip install matplotlib numpy xarray pandas netcdf4

Some notebooks may require additional packages like ``cartopy`` for geographic plotting or ``scipy`` for advanced calculations.
These dependencies are mentioned at the beginning of each notebook.