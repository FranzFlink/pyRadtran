# pyradtran/exceptions.py
"""
Exception hierarchy for pyRadtran.

All package-specific exceptions inherit from :class:`PyRadtranError`,
so callers can catch the base class to handle any pyRadtran failure:

.. code-block:: python

    try:
        result = ds.pyradtran.run(config_path="my_config.yaml")
    except pyradtran.PyRadtranError as exc:
        print(f"pyRadtran failed: {exc}")

See Also
--------
pyradtran.config : Configuration loading (may raise :class:`ConfigurationError`).
pyradtran.core : Simulation engine (may raise :class:`UvspecExecutionError`).
pyradtran.io : I/O layer (may raise :class:`OutputParsingError`,
    :class:`InputGenerationError`).
"""


class PyRadtranError(Exception):
    """Base exception for all pyRadtran errors."""


class ConfigurationError(PyRadtranError):
    """Raised when a configuration file is missing, malformed, or invalid."""


class InputGenerationError(PyRadtranError):
    """Raised when libRadtran input-file generation fails.

    Common causes include missing atmosphere files, invalid ERA5
    datasets, or unsupported input-data formats.
    """


class UvspecExecutionError(PyRadtranError):
    """Raised when the libRadtran ``uvspec`` binary returns a non-zero exit code."""


class OutputParsingError(PyRadtranError):
    """Raised when a libRadtran output file cannot be parsed.

    The raw file path is usually included in the message so the caller
    can inspect it manually.
    """


class RadiosondeError(PyRadtranError):
    """Raised for errors related to radiosonde file discovery or parsing."""


class ValidationError(PyRadtranError):
    """Raised when user-supplied values fail validation checks."""