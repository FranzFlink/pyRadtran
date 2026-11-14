# API Reference

Everything public in pyRadtran, grouped by what you want to do. Most
names are importable straight from the top-level package
(`from pyradtran import Var, load_config, brightness_temperature`).

```{tip}
New here? Start with {meth}`ds.pyradtran.run() <pyradtran.interface.PyRadtranAccessor.run>`,
{class}`~pyradtran.params.Var` and {func}`~pyradtran.config.load_config` —
those three cover most notebooks.
```

## Running simulations

The `ds.pyradtran` accessor is registered on every {class}`xarray.Dataset`
as soon as `pyradtran` is imported.

```{eval-rst}
.. currentmodule:: pyradtran

.. autosummary::
   :toctree: generated
   :template: autosummary/class_nomembers.rst

   interface.PyRadtranAccessor

.. autosummary::
   :toctree: generated

   interface.PyRadtranAccessor.run
   interface.PyRadtranAccessor.explain
   interface.PyRadtranAccessor.jacobian
   interface.PyRadtranAccessor.inspect_cloud_file
   interface.run_pyradtran_simulation
   interface.execute_simulation_batch
   core.Simulation
```

## Parameters

How values reach the `uvspec` input file — see {doc}`/guide/parameters`.

```{eval-rst}
.. autosummary::
   :toctree: generated

   params.Var
   params.Raw
   params.describe
   params.search_options
   params.ParamSpec
   params.ParamResolver
   params.get_schema
   params.validate_against_schema
```

## Configuration

```{eval-rst}
.. autosummary::
   :toctree: generated

   config.load_config
   config.save_master_config
   config.create_example_config
   config.list_atmosphere_profiles
   config.list_solar_spectra
   config.SimulationConfig
   config.SimulationDefaults
   config.PathsConfig
   config.ExecutionConfig
   config.OutputConfig
   config.CloudParameters
```

## Instrument channels & brightness temperature

```{eval-rst}
.. autosummary::
   :toctree: generated

   channels.convolve_channels
   channels.brightness_temperature
```

## Atmospheres: ERA5 and radiosondes

```{eval-rst}
.. autosummary::
   :toctree: generated

   era5.normalize_era5
   era5.recommend_atmosphere
   era5.era5_atmosphere_file
   era5.write_atmosphere_file
   era5.select_profile
   era5.cloud_profiles
   io.ERA5AtmosphereGenerator
   io.RadiosondeAtmosphereGenerator
   utils.RadiosondeFinder
```

## Clouds

```{eval-rst}
.. autosummary::
   :toctree: generated

   clouds.CloudLayer
   clouds.CloudGenerator
   clouds.CloudFileWriter
   clouds.generate_cloud_file_from_era5
```

## Input files

```{eval-rst}
.. autosummary::
   :toctree: generated

   input_builder.InputFileBuilder
   input_builder.InputLine
   input_builder.calculate_solar_zenith_angle
   input_builder.effective_output_columns
   input_builder.effective_output_altitudes
```

## Output parsing & storage

```{eval-rst}
.. autosummary::
   :toctree: generated

   io.OutputParser
   io.ParsedOutput
   io.OutputType
   io.OutputToXarray
   io.NetCDFSaver
   io.InputDataLoader
   interface.PointOutcome
   interface.SimPoint
```

## libRadtran option schema

```{eval-rst}
.. autosummary::
   :toctree: generated

   schema.load_schema
   schema.extract_schema
   schema.find_libradtran_root
```

## Exceptions

```{eval-rst}
.. autosummary::
   :toctree: generated

   exceptions.PyRadtranError
   exceptions.ConfigurationError
   exceptions.ValidationError
   exceptions.InputGenerationError
   exceptions.UvspecExecutionError
   exceptions.OutputParsingError
   exceptions.RadiosondeError
```
