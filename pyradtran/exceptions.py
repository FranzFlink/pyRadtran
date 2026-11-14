# pyradtran/exceptions.py
"""
Exception hierarchy for pyRadtran.

All package-specific exceptions inherit from :class:`PyRadtranError`, so
one ``except`` clause handles any pyRadtran failure::

    Exception
    └── PyRadtranError            base class; also raised directly by run()
        ├── ValidationError       bad ``params`` value / unknown option
        ├── InputGenerationError  input data or atmosphere file unusable
        ├── UvspecExecutionError  a single uvspec run could not complete
        ├── OutputParsingError    uvspec output unreadable / NetCDF save failed
        ├── ConfigurationError    bad configuration file, path or value
        └── RadiosondeError       reserved (see class docstring)

.. autosummary::

   PyRadtranError
   ValidationError
   InputGenerationError
   UvspecExecutionError
   OutputParsingError
   ConfigurationError
   RadiosondeError

All of them are importable from the package top level.

:class:`ConfigurationError` also derives from :class:`FileNotFoundError`
and :class:`ValueError`, so ``except FileNotFoundError`` /
``except ValueError`` code keeps working.

Examples
--------
Catch everything pyRadtran-specific in one place:

>>> import pyradtran
>>> try:  # doctest: +SKIP
...     result = ds.pyradtran.run(config_path="my_config.yaml")
... except pyradtran.PyRadtranError as exc:
...     print(f"pyRadtran failed: {exc}")

See Also
--------
pyradtran.params : Parameter validation (raises :class:`ValidationError`).
pyradtran.core : Simulation engine (raises :class:`UvspecExecutionError`).
pyradtran.io : I/O layer (raises :class:`OutputParsingError`,
    :class:`InputGenerationError`).
"""


class PyRadtranError(Exception):
    """Base exception for all pyRadtran errors.

    Also raised directly by
    :meth:`~pyradtran.interface.PyRadtranAccessor.run` when the run as a
    whole cannot produce a result: the time / latitude / longitude
    variable is missing from the dataset, an option combination is
    inconsistent, or every point failed. A partially failed batch does
    not raise; failed points are flagged in the ``status`` variable of
    the result.

    Examples
    --------
    It catches every other exception of the package:

    >>> from pyradtran.exceptions import PyRadtranError
    >>> from pyradtran.params import REGISTRY
    >>> try:
    ...     REGISTRY["albedo"].validate(1.5)
    ... except PyRadtranError as exc:
    ...     print(type(exc).__name__, "-", exc)
    ValidationError - 'albedo' must be in [0.0, 1.0], got 1.5

    A dataset without the expected coordinate variables:

    >>> try:  # doctest: +SKIP
    ...     ds.drop_vars("latitude").pyradtran.run()
    ... except PyRadtranError as exc:
    ...     print(exc)
    Latitude variable 'latitude' not found in dataset
    """


class ConfigurationError(PyRadtranError, FileNotFoundError, ValueError):
    """Raised when the configuration is unusable.

    Raised by :func:`~pyradtran.config.load_config`,
    :meth:`SimulationConfig.from_yaml
    <pyradtran.config.SimulationConfig.from_yaml>` and the config
    dataclasses when

    * the simulation YAML file is missing, unreadable or not valid YAML;
    * ``paths.libradtran_bin``, ``paths.libradtran_data`` or the solar
      spectrum file do not exist (e.g. no master config has been written
      yet; see :func:`~pyradtran.config.save_master_config`);
    * a value fails the
      :class:`~pyradtran.config.SimulationDefaults` validation.

    Also a :class:`FileNotFoundError` and a :class:`ValueError`.

    Examples
    --------
    >>> import pyradtran
    >>> try:
    ...     cfg = pyradtran.load_config("no_such_config.yaml")
    ... except pyradtran.ConfigurationError as exc:
    ...     print(f"bad configuration: {exc}")
    bad configuration: Cannot read configuration file no_such_config.yaml: [Errno 2] No such file or directory: 'no_such_config.yaml'
    >>> issubclass(pyradtran.ConfigurationError, (FileNotFoundError, ValueError))
    True
    """


class InputGenerationError(PyRadtranError):
    """Raised when the inputs of a simulation cannot be prepared.

    Raised by :mod:`pyradtran.io` and :mod:`pyradtran.era5` when

    * an input data file is missing, has an unsupported format, lacks a
      ``time`` column or other required variables, or cannot be read;
    * an ERA5 dataset has an unsupported pressure unit, lacks the
      required ``t`` / ``q`` variables or the ``pressure_level``
      coordinate, or contains no valid (non-NaN) levels for a point;
    * radiosonde retrieval is requested but ``siphon`` is not installed,
      or no sounding is found.

    Examples
    --------
    >>> from pyradtran import InputGenerationError, era5_atmosphere_file
    >>> try:  # doctest: +SKIP
    ...     path = era5_atmosphere_file(era5, lat, lon, time, "atm.dat")
    ... except InputGenerationError as exc:
    ...     print(f"no atmosphere for this point: {exc}")
    """


class UvspecExecutionError(PyRadtranError):
    """Raised when a single ``uvspec`` simulation cannot be carried out.

    Raised by :class:`~pyradtran.core.Simulation` when building the
    input file, preparing cloud files or launching ``uvspec`` fails with
    an unexpected error; the original exception is chained as
    ``__cause__``. A ``uvspec`` run that only exits with a non-zero code
    does not raise: the simulation returns ``None`` and keeps the input
    file for inspection.

    In a batch run
    (:meth:`~pyradtran.interface.PyRadtranAccessor.run`) the error is
    caught per point: the point gets ``status == 1`` and the remaining
    points still run.

    Examples
    --------
    Driving the engine directly (``sim`` is a
    :class:`~pyradtran.core.Simulation`):

    >>> from pyradtran import UvspecExecutionError
    >>> try:  # doctest: +SKIP
    ...     out_path = sim.run_simulation(dt, 78.9, 11.9)
    ... except UvspecExecutionError as exc:
    ...     print(exc, "caused by", repr(exc.__cause__))
    """


class OutputParsingError(PyRadtranError):
    """Raised when simulation output cannot be read or written.

    Raised by :mod:`pyradtran.io` when a ``uvspec`` output file is
    missing, empty or cannot be parsed into the expected columns, and
    when saving the result dataset to NetCDF fails. The message contains
    the file path.

    Examples
    --------
    >>> from pyradtran import OutputParser, OutputParsingError
    >>> try:  # doctest: +SKIP
    ...     parsed = OutputParser(cfg).parse_output_file("uvspec.out")
    ... except OutputParsingError as exc:
    ...     print(exc)
    Output file not found: uvspec.out
    """


class RadiosondeError(PyRadtranError):
    """Reserved for radiosonde file discovery and parsing errors.

    Not raised by pyRadtran itself: radiosonde retrieval
    problems surface as :class:`InputGenerationError`, and
    :class:`~pyradtran.utils.RadiosondeFinder` returns *None* when it
    finds no (close enough) sounding.

    Examples
    --------
    >>> from pyradtran import InputGenerationError, RadiosondeError
    >>> try:  # doctest: +SKIP
    ...     ds.pyradtran.run(config_path="sonde.yaml")
    ... except (RadiosondeError, InputGenerationError) as exc:
    ...     print(f"radiosonde problem: {exc}")
    """


class ValidationError(PyRadtranError):
    """Raised when a user-supplied parameter value fails validation.

    Raised by :class:`~pyradtran.params.ParamResolver`, and so by
    :meth:`~pyradtran.interface.PyRadtranAccessor.run` and
    :meth:`~pyradtran.interface.PyRadtranAccessor.explain` before any
    ``uvspec`` process starts, when a ``params`` entry

    * is outside the valid range or not one of the allowed choices,
    * names an option the local libRadtran install does not know,
    * has too few or too many values for the option,
    * is a :class:`~pyradtran.params.Var` whose variable is missing from
      the dataset, or a bare string that matches a dataset variable name
      (almost certainly a forgotten ``Var``).

    All offending entries are reported in one exception.
    :meth:`~pyradtran.interface.PyRadtranAccessor.jacobian` raises it
    when the perturbed parameter has no scalar base value.

    Examples
    --------
    >>> from types import SimpleNamespace
    >>> from pyradtran import ValidationError
    >>> from pyradtran.params import ParamResolver
    >>> cfg = SimpleNamespace()  # stand-in; normally pyradtran.load_config()
    >>> try:
    ...     ParamResolver(cfg, {"albedo": 1.5, "sza": 999.0}, schema=None)
    ... except ValidationError as exc:
    ...     print(exc)
    Invalid parameter value(s): 'albedo' must be in [0.0, 1.0], got 1.5; 'sza' must be in [0.0, 180.0] deg, got 999.0

    The same error from the accessor:

    >>> try:  # doctest: +SKIP
    ...     ds.pyradtran.run(params={"albedoo": 0.5})
    ... except ValidationError as exc:
    ...     print(exc)
    Invalid parameter value(s): 'albedoo' is not a known uvspec option of the local libRadtran install; did you mean albedo / albedo_map / albedo_file? (wrap the value in Raw(...) to bypass validation)
    """
