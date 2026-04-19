# Changelog

## Version 0.2.0 (Upcoming)

### Documentation & Notebooks
- Overhauled all 21 Jupyter Book notebooks: standardized titles, structure, and API usage
- Fixed broken TOC references and added 11 previously orphaned notebooks to the book
- Filled the "Check your input file!" section with detailed debugging guidance
- Expanded installation guide with `~/.pyradtran/config.yaml` master config documentation
- Expanded usage guide with sections on clouds, ERA5, batch processing, and `parameter_overrides`
- Overhauled README with configuration section, CI badge, and corrected examples

### API Standardization
- Standardized all examples to use `ds.pyradtran.run(config_path=...)` as the canonical API
- Made libRadtran paths version-agnostic in all YAML configs

### CI/CD
- Added GitHub Actions workflow for unit tests (Python 3.9, 3.10, 3.11)
- Added Jupyter Book build checks

### Bug Fixes
- Fixed various typos in documentation and notebooks
- Unified Python version requirement to `>= 3.9` across all documentation

## Version 0.1.0 (Development)

- Initial release
- Basic pyRadtran functionality with `uvspec` wrapper
- xarray integration via `.pyradtran` accessor
- YAML configuration system with layered defaults
- Parallel simulation execution
- Jupyter notebook examples
