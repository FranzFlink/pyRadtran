# Contributing

Bug reports, documentation fixes and new features are welcome.

## Development Setup

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR-USERNAME/pyRadtran.git
   cd pyRadtran
   ```
3. Install in development mode:
   ```bash
   pip install -e '.[dev]'
   ```
4. Set up your `~/.pyradtran/config.yaml` (see {doc}`installation`)

## Running Tests

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

black (line length 88), isort (black profile) and flake8. Before
submitting a PR, run:

```bash
black pyradtran/
isort pyradtran/
flake8 pyradtran/
```

## Notebook Guidelines

1. First markdown cell: `# Title`, then 1–3 sentences on what the
   notebook shows and what it needs (data, optional packages).
2. Short section headings (`## Setup`, `## Run`, `## Results`).
3. No absolute paths: `config_path=Path('config/...')`, configs in
   `book/notebooks/config/`.
4. Use `ds.pyradtran.run(config_path=..., params={...})`.
5. Comment on a figure only when it shows something the reader would
   not see unaided.
6. Commit the executed outputs: the book build does not run notebooks.
7. Add the notebook to `book/_toc.yml` and to its landing page under
   `book/tutorials/` or `book/case_studies/`.

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
layout and are the source of the API reference;
`pyradtran/channels.py` is the model to copy. A public function has:

- a one-line summary and a short description of what it is *for*;
- `Parameters` / `Returns` / `Raises` with types and **units**;
- a `Notes` section with the formula it implements, in LaTeX
  (`.. math::` or `` :math:`...` ``);
- an `Examples` section a user could paste.

Examples are doctests:

```bash
pytest --doctest-modules pyradtran -p no:cacheprovider
```

Anything that needs libRadtran, the network or user data gets a
`# doctest: +SKIP` on each statement (the flag is hidden in the rendered
page). Paste real output, never invented numbers.

Figures in docstrings use the matplotlib `.. plot::` directive. They run
on every documentation build, **without** libRadtran, so they must be
pure numpy/matplotlib (`np`, `xr` and `plt` are pre-imported):

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
by `book/img/make_guide_figures.py` and committed as PNG. Rerun the
script (needs libRadtran) when the behaviour changes.

## Pull Requests

1. Create a feature branch from `main`
2. Make sure the tests pass: `pytest -m "not slow"`
3. Open a pull request against `main`, say what it does and link
   related issues
