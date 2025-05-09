# Improvement Suggestions:
# 1. **Module Docstring**: Add a comprehensive module docstring that describes the purpose of this file: to define common error functions, aggregation functions, pre-configured `ComponentBasedEvaluator` instances, and suites of these evaluators for standardized model assessment. (Primary task).
# 2. **Correct `ComponentBasedEvaluator` Parameter Names**: Ensure the parameter names used when instantiating `ComponentBasedEvaluator` (e.g., `error_function`, `temporal_aggregation`) match the actual parameter names in its `__init__` method (which are `errorFunc`, `timeAggregationFunc`, `regionAggregationFunc`). This is a functional correction.
# 3. **Clarify `predictions` Argument in Error Functions**: The docstrings for `mae_error` and `mse_error` should explicitly state what the `predictions` list represents (e.g., multiple samples from a probabilistic forecast, where the mean of samples is compared to the truth).
# 4. **Comprehensive Type Hinting**: Add specific type hints for all function parameters and return values (e.g., `truth: float`, `predictions: List[float]`), and for complex structures like `evaluator_suite_options` (e.g., `Dict[str, List[ComponentBasedEvaluator]]`).
# 5. **Extensibility and Configuration of Suites**: The `evaluator_suite_options` dictionary is hardcoded. For greater flexibility, consider if these suites could be defined in a configuration file (e.g., YAML) and loaded dynamically, or if a registration mechanism for custom evaluators and suites would be beneficial.

"""
This module defines standard error functions, aggregation functions, and pre-configured
evaluation suites for assessing model performance in CHAP-core.

It provides common metrics like Mean Absolute Error (MAE) and Mean Squared Error (MSE)
at the point forecast level (comparing truth to the mean of prediction samples).
It also includes functions for aggregating these errors over time and/or across regions.

These components are used to instantiate `ComponentBasedEvaluator` objects from
`chap_core.assessment.evaluator`, which are then grouped into named suites
in `evaluator_suite_options` for convenient use in standardized evaluation pipelines.
"""

import math
from typing import Dict, List  # Added List, Dict

from chap_core.assessment.evaluator import ComponentBasedEvaluator

# --- Error Functions ---
# These functions typically compare a single ground truth value to a forecast.
# If the forecast is probabilistic (multiple samples), the mean of samples is often used.


def mae_error(truth: float, predictions: List[float]) -> float:
    """
    Calculates Mean Absolute Error (MAE) for a single time point.

    Compares the truth value against the mean of the prediction samples.

    Args:
        truth (float): The ground truth value.
        predictions (List[float]): A list of prediction samples for the time point.
                                   If a single point forecast, this list contains one element.

    Returns:
        float: The Mean Absolute Error.
    """
    if not predictions:
        return abs(truth)  # Or handle as NaN, or raise error
    return abs(truth - (sum(predictions) / len(predictions)))


def mse_error(truth: float, predictions: List[float]) -> float:
    """
    Calculates Mean Squared Error (MSE) for a single time point.

    Compares the truth value against the mean of the prediction samples.

    Args:
        truth (float): The ground truth value.
        predictions (List[float]): A list of prediction samples for the time point.

    Returns:
        float: The Mean Squared Error.
    """
    if not predictions:
        return truth**2  # Or handle as NaN, or raise error
    return (truth - (sum(predictions) / len(predictions))) ** 2


# --- Aggregation Functions ---


def mean_across_time(errors: List[float]) -> float:
    """
    Aggregates a list of error values (typically over time for a single region)
    by calculating their mean.

    Args:
        errors (List[float]): A list of error values.

    Returns:
        float: The mean of the errors. Returns float('nan') if errors list is empty.
    """
    if not errors:
        return float("nan")
    return sum(errors) / len(errors)


def sqrt_mean_across_time(errors: List[float]) -> float:
    """
    Aggregates a list of squared error values by taking the square root of their mean.
    This is used for calculating Root Mean Squared Error (RMSE) when `mse_error`
    is used as the point error function.

    Args:
        errors (List[float]): A list of squared error values.

    Returns:
        float: The root mean of the squared errors. Returns float('nan') if errors list is empty.
    """
    if not errors:
        return float("nan")
    return math.sqrt(sum(errors) / len(errors))


def mean_across_regions(errors: List[float]) -> float:
    """
    Aggregates a list of error values (typically from different regions, possibly already
    time-aggregated) by calculating their mean.

    Args:
        errors (List[float]): A list of error values.

    Returns:
        float: The mean of the errors. Returns float('nan') if errors list is empty.
    """
    if not errors:
        return float("nan")
    return sum(errors) / len(errors)


# --- Evaluator Instances ---
# These are pre-configured instances of ComponentBasedEvaluator for common metrics.
# Corrected parameter names: errorFunc, timeAggregationFunc, regionAggregationFunc

# Mean Absolute Error per location, averaged over the evaluation period.
mae_component_evaluator = ComponentBasedEvaluator(
    name="MAE_per_location",  # More descriptive name
    errorFunc=mae_error,
    timeAggregationFunc=mean_across_time,
    regionAggregationFunc=None,
)

# Mean Absolute Error, first averaged over time per location, then these MAEs are averaged across all regions.
mae_country_evaluator = ComponentBasedEvaluator(
    name="MAE_global_avg",  # More descriptive name
    errorFunc=mae_error,
    timeAggregationFunc=mean_across_time,
    regionAggregationFunc=mean_across_regions,
)

# Absolute Error computed at each time point for each location, without any time or spatial aggregation.
# Useful for getting a time series of errors.
absError_timepoint_evaluator = ComponentBasedEvaluator(
    name="AbsoluteError_per_timepoint",  # More descriptive name
    errorFunc=mae_error,  # Uses mae_error as it calculates absolute error for a point
    timeAggregationFunc=None,
    regionAggregationFunc=None,
)

# Root Mean Squared Error per location, calculated over the evaluation period.
rmse_evaluator = ComponentBasedEvaluator(
    name="RMSE_per_location",  # More descriptive name
    errorFunc=mse_error,  # Input errors are squared errors
    timeAggregationFunc=sqrt_mean_across_time,  # Aggregation takes sqrt of mean
    regionAggregationFunc=None,
)

# --- Evaluator Suite Registry ---
# A dictionary grouping predefined evaluators into named suites for convenience.
evaluator_suite_options: Dict[str, List[ComponentBasedEvaluator]] = {
    "onlyLocalMAE": [mae_component_evaluator],
    "localAndGlobalMAE": [mae_component_evaluator, mae_country_evaluator],
    "localMAEandRMSE": [mae_component_evaluator, rmse_evaluator],
    "detailed": [  # Renamed 'mix' to 'detailed'
        mae_component_evaluator,
        rmse_evaluator,
        absError_timepoint_evaluator,
        mae_country_evaluator,
    ],
}
