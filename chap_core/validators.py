# Improvement Suggestions:
# 1. **Detailed Docstrings**: Expand the docstring for `validate_training_data` to clearly describe all validations performed, the purpose of each parameter (especially `estimator`), and the specific exceptions raised. Add a descriptive module-level docstring.
# 2. **Usage of `estimator` Parameter**: The `estimator` parameter is currently unused. Either implement validation logic that utilizes it (e.g., checking if the dataset's features match the estimator's requirements) or remove it if it's not planned for use.
# 3. **Specific Exception Types**: Replace generic `ValueError` and `assert` statements with more specific custom exceptions (e.g., `InvalidTrainingDataError` from `chap_core.exceptions`, `TypeError` for incorrect `dataset` type) for better error handling and clarity.
# 4. **Comprehensive Validation Logic**: Extend the validation logic beyond just dataset duration. Consider adding checks for data integrity, such as presence of NaNs in critical columns, minimum number of observations, consistency of time periods, or feature compatibility with the (potentially used) `estimator`.
# 5. **Explicit Return Type Hint**: Add an explicit `-> None` return type hint to `validate_training_data` for clarity, as it does not return any value.

"""
This module provides data validation functions for the CHAP-core application.

These validators are used to ensure that datasets, particularly training data,
meet certain criteria قبل being used in models or other processing pipelines.
"""

from typing import Optional

# Assuming Estimator is a type or class defined elsewhere, representing a model estimator.
# from chap_core.assessment.prediction_evaluator import Estimator # Original import
# If Estimator is an abstract type or not strictly needed for current validation,
# one might use a forward reference string 'Estimator' or a more generic type.
# For now, keeping the original import path, assuming it's valid.
from chap_core.assessment.prediction_evaluator import (
    Estimator,  # type: ignore # If Estimator is complex/causes import issues here
)
from chap_core.exceptions import ChapCoreException  # For custom exceptions, e.g., InvalidTrainingDataError
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet
from chap_core.time_period.date_util_wrapper import delta_year


class InvalidTrainingDataError(ChapCoreException):
    """Custom exception for invalid training data."""

    pass


def validate_training_data(dataset: DataSet, estimator: Optional[Estimator] = None) -> None:
    """
    Validates a training dataset against a set of predefined criteria.

    Currently, it checks:
    1. If the `dataset` is an instance of `DataSet`.
    2. If the `dataset` covers a time period of at least two full years.

    The `estimator` parameter is currently not used but is reserved for future
    validations that might be specific to a particular model estimator (e.g.,
    checking for required features).

    Args:
        dataset (DataSet): The training dataset to validate.
        estimator (Optional[Estimator]): The model estimator for which the data is
                                         intended. Currently unused. Defaults to None.

    Raises:
        TypeError: If `dataset` is not an instance of `DataSet`.
        InvalidTrainingDataError: If the training data does not meet validation criteria
                                  (e.g., insufficient time coverage).
    """
    if not isinstance(dataset, DataSet):
        raise TypeError(f"Expected 'dataset' to be an instance of DataSet, but got {type(dataset)}.")

    if not dataset.time_period:  # Check if time_period itself is empty or invalid
        raise InvalidTrainingDataError("Training data 'time_period' attribute is missing or empty.")

    # Validate that the dataset covers at least two whole years
    # This check assumes dataset.start_timestamp and dataset.end_timestamp are valid.
    try:
        min_duration = dataset.start_timestamp + (2 * delta_year)
    except TypeError as e:  # Handles cases where start_timestamp might be None or not a compatible type
        raise InvalidTrainingDataError(f"Could not calculate minimum duration due to invalid start_timestamp: {e}")

    if dataset.end_timestamp < min_duration:
        raise InvalidTrainingDataError(
            f"Training data must cover at least two whole years. "
            f"Current coverage: {dataset.start_timestamp.date()} to {dataset.end_timestamp.date()}."
        )

    # Placeholder for future validations using the estimator
    if estimator is not None:
        # Example: Check if dataset features are compatible with estimator's requirements
        # logger.debug(f"Estimator '{estimator_name}' provided, but estimator-specific validation is not yet implemented.")
        pass

    # Add other validations here as needed, e.g.:
    # - Check for excessive NaNs in key columns
    # - Minimum number of data points per location/series
    # - Consistency of data types for features

    # If all checks pass:
    # logger.info("Training data validation successful.") # Consider adding logging
    return None  # Explicitly return None
