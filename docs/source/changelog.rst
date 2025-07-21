Changelog
=========

Version 0.1.0 (2025-07-21)
---------------------------

Initial release of PyRadtran.

**New Features:**

- Pythonic interface to libRadtran radiative transfer model
- YAML-based configuration system with flexible parameter management
- Support for solar radiation simulations (UV, visible, near-IR)
- Support for thermal infrared simulations (4.5-42 μm)
- xarray integration with automatic coordinate and metadata handling
- Scalar and coordinate-based altitude handling for time series data
- Batch processing capabilities for efficient multi-timestep simulations
- Support for various atmospheric profiles:
  - Standard atmospheric profiles (AFGL, etc.)
  - Radiosonde data integration
  - ERA5 reanalysis data support
- Comprehensive configuration options:
  - Surface properties (temperature, albedo, BRDF models)
  - Cloud properties (water and ice clouds)
  - Molecular absorption parameters
  - Spectral resolution and wavelength ranges
- Input/output utilities with automatic file management
- Integration with scientific Python ecosystem (NumPy, pandas, xarray)
- Jupyter notebook examples and tutorials
- Comprehensive documentation with Sphinx

**Core Components:**

- ``pyradtran.config``: Configuration management and validation
- ``pyradtran.core``: Core simulation engine and libRadtran interface
- ``pyradtran.interface``: xarray accessor for seamless integration
- ``pyradtran.io``: Input/output parsing and file management
- ``pyradtran.utils``: Utility functions and helpers

**Examples and Documentation:**

- Getting started tutorials
- Solar and thermal simulation examples
- xarray integration examples
- Batch processing examples
- API reference documentation

**Dependencies:**

- Python ≥ 3.8
- NumPy ≥ 1.20.0
- pandas ≥ 1.3.0
- xarray ≥ 0.20.0
- PyYAML ≥ 6.0
- libRadtran (external dependency)

**Testing:**

- Comprehensive test suite with pytest
- Integration tests for core functionality
- Configuration validation tests
- IO parsing tests
