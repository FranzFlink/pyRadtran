=====
Usage
=====

Basic Concepts
=============

pyradtran provides a Pythonic interface to the libRadtran radiative transfer model. The main components are:

* **Simulation**: The core class for setting up and running simulations
* **SimulationConfig**: Configuration management for libRadtran parameters
* **I/O Utilities**: Functions for reading and processing simulation results

Creating a Simulation
====================

To create a new simulation, first load the configuration and then initialize a Simulation instance:

.. code-block:: python

    from pyradtran.config import load_config
    from pyradtran.core import Simulation
    
    # Load default configuration
    config = load_config()
    
    # Or load a custom configuration
    # config = load_config("path/to/my_config.yaml")
    
    # Create a new simulation
    sim = Simulation(config)

Configuration Structure
======================

The configuration system uses a hierarchical structure with several components:

* **PathsConfig**: Defines paths to libRadtran executables, data files, and working directories
* **SimulationDefaults**: Defines default parameters for libRadtran simulations
* **ExecutionConfig**: Controls execution settings like parallelization and debugging
* **OutputConfig**: Controls output file formats and encoding

You can customize these settings in your YAML configuration file:

.. code-block:: yaml

    paths:
      libradtran_bin: /path/to/uvspec
      libradtran_data: /path/to/libRadtran/data
      atmosphere_profile: /path/to/atmfile.dat
      solar_spectrum: /path/to/solar_spectrum.dat
      working_dir: ./work
      output_dir: ./output
      
    simulation_defaults:
      rte_solver: disort
      mol_abs_param: reptran
      wavelength_nm: [300, 2500]
      albedo_type: const
      albedo_value: 0.3
      
    execution:
      max_workers: 4
      debug_mode: false
      
    output:
      filename_prefix: my_simulation
      netcdf_encoding:
        zlib: true
        complevel: 5

Running Simulations
==================

To run a simulation, you need to specify the date/time and location:

.. code-block:: python

    from datetime import datetime
    
    # Define simulation parameters
    dt = datetime(2025, 5, 6, 12, 0, 0)  # May 6, 2025, 12:00:00 UTC
    latitude = 78.9   # North latitude in degrees
    longitude = 11.9  # East longitude in degrees
    
    # Run the simulation
    output_file = sim.run(dt, latitude, longitude)

Working with Results
===================

You can process the simulation results using the I/O utilities:

.. code-block:: python

    from pyradtran.io import read_output
    import matplotlib.pyplot as plt
    
    # Read the output file
    if output_file:
        result = read_output(output_file)
        
        # Access the data (dictionary format)
        wavelengths = result['wavelength']  # in nm
        direct_irradiance = result['edir']  # direct irradiance
        diffuse_down = result['edn']        # diffuse downward
        diffuse_up = result['eup']          # diffuse upward
        
        # Plot the results
        plt.figure(figsize=(10, 6))
        plt.plot(wavelengths, direct_irradiance, label='Direct')
        plt.plot(wavelengths, diffuse_down, label='Diffuse Down')
        plt.plot(wavelengths, diffuse_up, label='Diffuse Up')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Irradiance')
        plt.legend()
        plt.grid(True)
        plt.show()

Advanced Configuration
=====================

For more advanced configurations, you can modify the config object directly:

.. code-block:: python

    # Update configuration parameters
    config.simulation_defaults.rte_solver = "disort"
    config.simulation_defaults.output_columns = ["lambda", "edir", "edn", "eup"]
    
    # Enable aerosols
    config.simulation_defaults.aerosols.enabled = True
    config.simulation_defaults.aerosols.aerosol_type = "maritime"
    config.simulation_defaults.aerosols.aerosol_visibility_km = 50.0
    
    # Enable clouds
    config.simulation_defaults.clouds.enabled = True
    config.simulation_defaults.clouds.layer_heights_km = [(2.0, 3.0)]
    config.simulation_defaults.clouds.layer_water_content = [0.1]  # g/m³
    config.simulation_defaults.clouds.layer_effective_radius_um = [10.0]  # μm