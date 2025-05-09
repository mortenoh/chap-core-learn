# Improvement Suggestions:
# 1. **Base Exception Class**: Introduce a base `ChapCoreException` from which all other exceptions in this module inherit. This allows for a common catch-all for application-specific errors.
# 2. **Contextual Information**: Enhance exceptions to carry more context. For example, `InvalidModelException` could store the model name or path that was invalid. This can be achieved by customizing their `__init__` methods.
# 3. **Consistency in Body Definition**: Use `pass` consistently for classes that do not add new methods or attributes, instead of `...` (ellipsis), for conventional Python style.
# 4. **Granularity of Exceptions**: For complex areas like model operations, consider if more specific exceptions (e.g., `ModelTrainingError`, `ModelInferenceError` inheriting from `ModelFailedException`) would improve error handling downstream.
# 5. **Error Codes/Documentation Links**: If applicable, include error codes or references/links to external documentation in docstrings for exceptions that might require more detailed troubleshooting steps for users or developers.

"""
This module defines custom exceptions used throughout the CHAP-core application.

These exceptions provide more specific error information than built-in Python
exceptions, allowing for more targeted error handling and clearer diagnostics.
It is recommended to have a base `ChapCoreException` for all custom exceptions.
"""


class ChapCoreException(Exception):
    """Base class for all custom exceptions in the chap-core application."""

    pass


class ModelFailedException(ChapCoreException):
    """Raised when a model execution (e.g., training, prediction) fails unexpectedly."""

    pass


class InvalidModelException(ChapCoreException):
    """Raised when a provided model configuration, reference, or structure is invalid."""

    pass


class CommandLineException(ChapCoreException):
    """Raised when a command-line interface invocation fails, receives invalid arguments, or is misused."""

    pass


class NoPredictionsError(ChapCoreException):
    """Raised when a model, expected to produce predictions, yields no output."""

    pass


class GEEError(ChapCoreException):
    """Raised when an operation involving Google Earth Engine (GEE) fails.
    This could be due to authentication issues, query errors, or GEE service problems.
    """

    pass


class ModelConfigurationException(ChapCoreException):
    """
    Raised when a model's configuration (e.g., from a YAML file or parameters)
    is found to be invalid, incomplete, or improperly formatted.
    """

    pass


class InvalidDateError(ChapCoreException):
    """
    Raised when a provided date string or object is not parseable,
    does not conform to an expected format, or falls outside an acceptable range.
    """

    pass
