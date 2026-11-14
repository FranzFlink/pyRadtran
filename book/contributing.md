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
# Everything except the slow network tests (IGRA2, ERA5 Zarr)
pytest -m "not slow"

# Only the fast unit tests (no libRadtran installation needed)
pytest -m unit

# Integration tests (require libRadtran)
pytest -m integration

# With a coverage report
pytest -m "not slow" --cov=pyradtran
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

The documentation is a Jupyter Book: narrative pages and notebooks in
`book/`, and an API reference generated from the docstrings.

```bash
pip install -e '.[docs]'
jupyter-book build book/
open book/_build/html/index.html
```

### Docstrings

Docstrings follow the [numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html)
layout and are the single source of the API reference —
`pyradtran/channels.py` is the model to copy. A public function should
have:

- a one-line summary and a short description of what it is *for*;
- `Parameters` / `Returns` / `Raises` with types and **units**;
- a `Notes` section with the formula it implements, in LaTeX
  (`.. math::` or `` :math:`...` ``);
- an `Examples` section a user could paste.

Examples are doctests and are checked:

```bash
pytest --doctest-modules pyradtran -p no:cacheprovider
```

Anything that needs libRadtran, the network or user data gets a
`# doctest: +SKIP` on each statement (the flag is hidden in the rendered
page) — but paste real output, never invented numbers.

Figures in docstrings use the matplotlib `.. plot::` directive. They are
executed on every documentation build, **without** libRadtran, so they
must be pure numpy/matplotlib (`np`, `xr` and `plt` are pre-imported):

```rst
.. plot::

    from pyradtran import brightness_temperature

    L = np.linspace(0.5, 12.0, 200)
    plt.plot(L, brightness_temperature(L, 10800.0))
    plt.xlabel("radiance (mW m$^{-2}$ nm$^{-1}$ sr$^{-1}$)")
    plt.ylabel("$T_b$ (K)")
```

New public objects must also be listed in `book/api/index.md`.

### Guide figures

Figures in the user guide that come from real simulations are produced
by `book/img/make_guide_figures.py` and committed as PNG — rerun the
script (needs libRadtran) when the underlying behaviour changes.

## Pull Requests

1. Create a feature branch from `main`
2. Make your changes with clear commit messages
3. Ensure tests pass: `pytest -m "not slow"`
4. Submit a pull request against the `main` branch
5. Describe what your PR does and link any related issues
