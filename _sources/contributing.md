# Contributing

We welcome contributions to pyRadtran! Whether it's bug reports, documentation improvements, or new features — every contribution helps.

## Development Setup

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR-USERNAME/pyRadtran.git
   cd pyRadtran
   ```
3. Install in development mode with dev dependencies:
   ```bash
   pip install -e '.[dev]'
   ```
4. Set up your `~/.pyradtran/config.yaml` (see {doc}`installation`)

## Running Tests

pyRadtran uses pytest with markers to separate tests by type:

```bash
# Run all unit tests (no libRadtran installation needed)
python run_tests.py --unit

# Run integration tests (requires libRadtran)
python run_tests.py --integration

# Run all tests with coverage report
python run_tests.py --coverage

# Run fast tests only (skip slow integration tests)
python run_tests.py --fast

# Or use pytest directly
pytest -m unit
pytest -m "not slow"
```

## Code Style

We use the following tools for code formatting and linting:

- **black** for code formatting (line length: 88)
- **isort** for import sorting (black-compatible profile)
- **flake8** for linting

Before submitting a PR, run:

```bash
black pyradtran/
isort pyradtran/
flake8 pyradtran/
```

## Notebook Guidelines

The Jupyter notebooks in `book/notebooks/` are the core of the documentation. When contributing or editing notebooks:

1. **Always include a title** as the first markdown cell: `# Quickstart: Topic Name` or `# Advanced: Topic Name`
2. **Add an introduction** — 1–3 sentences explaining what the notebook demonstrates
3. **Use section headings** (`## Setup`, `## Configuration`, `## Running the Simulation`, `## Results`) to structure the notebook
4. **Never hardcode absolute paths** — use `config_path=Path('config/...')` with configs in `book/notebooks/config/`
5. **Use the canonical API**: `ds.pyradtran.run(config_path=...)` — not `run_uvspec()`
6. **Add result interpretation cells** — explain what the reader should observe in the output/plots
7. **Clear all cell outputs** before committing — the Jupyter Book build will re-execute them
8. **Note required datasets** — if a notebook needs non-public data, add an admonition at the top

## Documentation

Documentation is built using Jupyter Book:

```bash
# Build the documentation
jupyter-book build book/

# View locally
open book/_build/html/index.html
```

## Pull Requests

1. Create a feature branch from `main`
2. Make your changes with clear commit messages
3. Ensure tests pass: `python run_tests.py --unit`
4. Submit a pull request against the `main` branch
5. Describe what your PR does and link any related issues
