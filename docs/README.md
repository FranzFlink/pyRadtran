# PyRadtran Documentation

This directory contains the Sphinx documentation for PyRadtran.

## Building Documentation Locally

### Prerequisites

Make sure you have PyRadtran installed and the documentation dependencies:

```bash
# From the repository root
pip install -e .
pip install -r docs/requirements.txt
```

### Quick Build

Use the provided script:

```bash
# From the repository root
./build_docs.sh
```

### Manual Build

```bash
# Navigate to docs directory
cd docs

# Clean previous builds
make clean

# Build HTML documentation
make html

# Open the documentation
open build/html/index.html  # macOS
xdg-open build/html/index.html  # Linux
start build/html/index.html  # Windows
```

## Documentation Structure

- `source/index.rst` - Main landing page
- `source/installation.rst` - Installation instructions
- `source/usage.rst` - Basic usage guide
- `source/examples.rst` - Detailed examples
- `source/api.rst` - API reference
- `source/notebooks.rst` - Jupyter notebook tutorials
- `source/contributing.rst` - Contribution guidelines
- `source/changelog.rst` - Version history

## GitHub Pages Deployment

Documentation is automatically built and deployed to GitHub Pages when changes are pushed to the `main` branch. The deployment uses GitHub Actions (see `.github/workflows/docs.yml`).

The documentation will be available at: https://franzflink.github.io/pyRadtran/

## Configuration

The documentation is configured in `source/conf.py`. Key settings:

- **Theme**: `sphinx_rtd_theme` (Read the Docs theme)
- **Extensions**: Autodoc, Napoleon, nbsphinx, MyST parser
- **Notebook execution**: Disabled by default (`nbsphinx_execute = 'never'`)

## Troubleshooting

### Common Issues

1. **Import errors**: Make sure PyRadtran is installed (`pip install -e .`)
2. **Missing dependencies**: Install docs requirements (`pip install -r docs/requirements.txt`)
3. **Notebook errors**: Check if notebooks can run independently
4. **Build warnings**: Review and fix Sphinx warnings for better documentation quality

### GitHub Pages Issues

1. **404 errors**: Ensure `.nojekyll` file is present in the built documentation
2. **Missing CSS/JS**: Check that static files are included in the artifact upload
3. **Permission errors**: Verify repository settings allow GitHub Pages deployment

## Adding New Documentation

### New Pages

1. Create a new `.rst` file in `source/`
2. Add it to the appropriate `toctree` in `index.rst`
3. Follow the existing structure and style

### New Notebooks

1. Add notebooks to the `notebooks/` directory
2. Reference them in `source/notebooks.rst`
3. Ensure notebooks can run without external dependencies

### API Documentation

The API documentation is automatically generated from docstrings. To improve it:

1. Write comprehensive docstrings following NumPy style
2. Include examples in docstrings
3. Use type hints where appropriate

## Maintenance

- Regularly update dependency versions in `requirements.txt`
- Check for broken links and outdated examples
- Update the changelog when releasing new versions
- Review and improve documentation based on user feedback
