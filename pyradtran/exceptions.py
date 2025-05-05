# pyradtran/exceptions.py
"""
Custom exceptions for the pyradtran package.
"""

class PyRadtranError(Exception):
    """Base exception for the pyradtran package."""
    pass


class ConfigurationError(PyRadtranError):
    """Exception raised for configuration errors."""
    pass


class InputGenerationError(PyRadtranError):
    """Exception raised when input file generation fails."""
    pass


class UvspecExecutionError(PyRadtranError):
    """Exception raised when uvspec execution fails."""
    pass


class OutputParsingError(PyRadtranError):
    """Exception raised when output parsing fails."""
    pass


class RadiosondeError(PyRadtranError):
    """Exception raised for errors related to radiosonde handling."""
    pass


class ValidationError(PyRadtranError):
    """Exception raised for validation errors."""
    pass