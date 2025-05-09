# Improvement Suggestions:
# 1. **Module Docstring**: Add a comprehensive module docstring that describes the purpose of this module: defining an evaluation framework for comparing model forecasts against ground truth data, including base classes and a flexible component-based evaluator. (Primary task).
# 2. **Specific Callable Type Hints**: In `ComponentBasedEvaluator.__init__`, refine the type hints for `errorFunc`, `timeAggregationFunc`, and `regionAggregationFunc`. Instead of just `Callable`, use more specific signatures like `Callable[[float, List[float]], float]` for `errorFunc` (truth, samples -> error) and `Callable[[List[float]], float]` for aggregation functions, to clarify their expected inputs and outputs.
# 3. **Robust Error Handling in `evaluate`**: Replace `assert` statements in `ComponentBasedEvaluator.evaluate` (e.g., for length and time period mismatches) with proper error handling that raises informative exceptions (e.g., `ValueError` or a custom `EvaluationAlignmentError`) to clearly indicate issues with input data alignment.
# 4. **Documentation of Aggregation Output Structure**: Clearly document the structure of the output `MultiLocationErrorTimeSeries` when time and/or region aggregations are applied. Specifically, explain how the `time_period` and location keys (e.g., "Full_period", "Full_region") are set for aggregated results.
# 5. **Extensibility of `Error` Object for Multiple Metrics**: The `Error` object currently stores a single `value`. If there's a need to compute and store multiple error metrics simultaneously (e.g., MAE, RMSE, bias) for each evaluation point or aggregation, consider whether the `Error` object should support a dictionary of metric values, or if the framework expects separate `ErrorTimeSeries` for each metric.

"""
This module defines the framework for evaluating model forecasts in CHAP-core.

It provides an abstract base class `Evaluator` that defines the interface for
all evaluators. It also includes `ComponentBasedEvaluator`, a flexible concrete
implementation that allows composing an evaluator from custom error functions
and aggregation functions for time and spatial dimensions.

The evaluators operate on `MultiLocationDiseaseTimeSeries` (for ground truth)
and `MultiLocationForecast` (for model predictions), producing a
`MultiLocationErrorTimeSeries` containing the calculated error metrics.
"""

import logging

logger = logging.getLogger(__name__)

from abc import ABC, abstractmethod
from typing import Any, Callable, List, Optional  # Added List, Any, Optional

from chap_core.assessment.representations import (
    Error,
    ErrorTimeSeries,
    MultiLocationDiseaseTimeSeries,
    MultiLocationErrorTimeSeries,
    MultiLocationForecast,
)
from chap_core.exceptions import ChapCoreException  # For custom exceptions


class EvaluationError(ChapCoreException):
    """Base exception for errors during evaluation."""

    pass


class DataAlignmentError(EvaluationError):
    """Exception raised for misaligned truth and forecast data during evaluation."""

    pass


class Evaluator(ABC):
    """
    Abstract base class for all evaluators.

    Evaluators are responsible for comparing ground truth data with forecasted data
    and calculating one or more error metrics. Subclasses must implement the
    `evaluate` method.
    """

    @abstractmethod
    def evaluate(
        self,
        all_truths: MultiLocationDiseaseTimeSeries,
        all_forecasts: MultiLocationForecast,
    ) -> MultiLocationErrorTimeSeries:
        """
        Compare ground truth data and forecasted data to compute evaluation metrics.

        Args:
            all_truths (MultiLocationDiseaseTimeSeries): A collection of time series
                representing the ground truth values for multiple locations.
            all_forecasts (MultiLocationForecast): A collection of forecasts
                (often probabilistic, e.g., samples) for the same locations and time periods.

        Returns:
            MultiLocationErrorTimeSeries: A collection of time series, where each series
                contains the calculated error metric(s) for a location over time, or
                aggregated error values.
        """
        pass

    def get_name(self) -> str:
        """
        Return a human-readable name for the evaluator.
        Defaults to the class name. Subclasses can override for a custom name.

        Returns:
            str: The name of the evaluator.
        """
        return self.__class__.__name__


class ComponentBasedEvaluator(Evaluator):
    """
    A flexible evaluator that is composed of several functional components:
    - An error function: Calculates the error between a single truth point and a forecast (e.g., MAE, MSE).
    - An optional time aggregation function: Aggregates errors over time for each location (e.g., mean error).
    - An optional region aggregation function: Aggregates errors across locations (e.g., global mean error).

    This design allows for constructing various evaluation schemes by plugging in different
    functions for these components.
    """

    def __init__(
        self,
        name: str,
        errorFunc: Callable[[Any, List[Any]], float],  # e.g., truth_value, forecast_samples -> error_float
        timeAggregationFunc: Optional[
            Callable[[List[float]], float]
        ] = None,  # e.g., list_of_errors -> aggregated_error
        regionAggregationFunc: Optional[
            Callable[[List[float]], float]
        ] = None,  # e.g., list_of_regional_errors -> aggregated_error
    ):
        """
        Initializes the ComponentBasedEvaluator.

        Args:
            name (str): A human-readable name for this specific evaluator configuration.
            errorFunc (Callable[[Any, List[Any]], float]): A function that takes a single ground truth value
                and a list of forecast samples (or a single forecast value) and returns a float error score.
                Example signature: `(truth: float, forecast_samples: List[float]) -> float`.
            timeAggregationFunc (Optional[Callable[[List[float]], float]]): An optional function to aggregate
                a list of error scores (over time for a single location) into a single score.
                Example signature: `(errors_over_time: List[float]) -> float`. If None, per-timepoint errors are kept.
            regionAggregationFunc (Optional[Callable[[List[float]], float]]): An optional function to aggregate
                a list of error scores (from different regions, possibly already time-aggregated)
                into a single global score. Example signature: `(regional_errors: List[float]) -> float`.
                If None, per-region (possibly time-aggregated) errors are kept.
        """
        self._name = name
        self._errorFunc = errorFunc
        self._timeAggregationFunc = timeAggregationFunc
        self._regionAggregationFunc = regionAggregationFunc

    def get_name(self) -> str:
        """
        Returns the human-readable name provided during initialization.

        Returns:
            str: The name of this evaluator instance.
        """
        return self._name

    def evaluate(
        self,
        all_truths: MultiLocationDiseaseTimeSeries,
        all_forecasts: MultiLocationForecast,
    ) -> MultiLocationErrorTimeSeries:
        """
        Computes errors between forecasts and ground truths for each location.

        The process involves:
        1. For each location and timepoint, calculate an error using `self._errorFunc`.
        2. If `self._timeAggregationFunc` is provided, aggregate these per-timepoint errors
           for each location into a single value per location (time_period="Full_period").
           Otherwise, keep per-timepoint errors.
        3. If `self._regionAggregationFunc` is provided, aggregate the (possibly time-aggregated)
           errors across all locations into a single global error value (location="Full_region",
           time_period="Full_period").

        Args:
            all_truths (MultiLocationDiseaseTimeSeries): Ground truth data.
            all_forecasts (MultiLocationForecast): Forecasted data.

        Returns:
            MultiLocationErrorTimeSeries: Calculated errors, possibly aggregated.

        Raises:
            DataAlignmentError: If truth and forecast series lengths or time periods do not match for a location.
            EvaluationError: For other issues during evaluation.
        """
        evaluation_result = MultiLocationErrorTimeSeries(timeseries_dict={})

        for location in all_truths.locations():
            if location not in all_forecasts.timeseries:
                # Log or handle missing forecast for a location present in truth
                logging.warning(f"No forecast found for location '{location}' present in truth data. Skipping.")
                continue

            current_error_series = ErrorTimeSeries(observations=[])

            truth_series_data = all_truths[location].observations
            forecast_series_data = all_forecasts.timeseries[location].predictions

            if len(truth_series_data) != len(forecast_series_data):
                raise DataAlignmentError(
                    f"Length mismatch for location '{location}': "
                    f"truth has {len(truth_series_data)} points, forecast has {len(forecast_series_data)} points."
                )

            errors_for_location: List[float] = []

            for truth_obs, prediction_obs in zip(truth_series_data, forecast_series_data):
                if truth_obs.time_period != prediction_obs.time_period:
                    raise DataAlignmentError(
                        f"Time period mismatch for location '{location}': "
                        f"truth at {truth_obs.time_period}, forecast at {prediction_obs.time_period}."
                    )

                try:
                    # disease_cases from truth, disease_case_samples from forecast
                    error_value = self._errorFunc(truth_obs.disease_cases, prediction_obs.disease_case_samples)
                    errors_for_location.append(error_value)
                except Exception as e:
                    logging.error(
                        f"Error function failed for location '{location}', period '{truth_obs.time_period}': {e}",
                        exc_info=True,
                    )
                    # Decide: skip this point, use NaN, or re-raise wrapped error
                    error_value = float("nan")  # Or some other placeholder / skip logic
                    errors_for_location.append(error_value)  # Keep length consistent if aggregating later

                if self._timeAggregationFunc is None:
                    # Store per-timepoint error
                    current_error_series.observations.append(
                        Error(time_period=truth_obs.time_period, value=error_value)
                    )

            if not errors_for_location and self._timeAggregationFunc is not None:
                logging.warning(f"No errors calculated for location '{location}' to perform time aggregation.")
                # Decide how to handle: skip, or add a NaN/default error record
                # evaluation_result[location] = current_error_series # which would be empty
                # continue

            if self._timeAggregationFunc is not None and errors_for_location:
                try:
                    aggregated_time_error = self._timeAggregationFunc(errors_for_location)
                    # Store aggregated error over time, using a special "Full_period" key for time_period
                    current_error_series.observations.append(
                        Error(time_period="Full_period", value=aggregated_time_error)
                    )
                except Exception as e:
                    logging.error(f"Time aggregation function failed for location '{location}': {e}", exc_info=True)
                    current_error_series.observations.append(Error(time_period="Full_period", value=float("nan")))

            if current_error_series.observations:  # Only add if there are errors (either per-timepoint or aggregated)
                evaluation_result[location] = current_error_series

        # Aggregate across regions if specified
        if self._regionAggregationFunc is not None:
            # This aggregation logic assumes that if timeAggregationFunc was None,
            # we want to aggregate across regions for each timepoint.
            # If timeAggregationFunc was provided, then each location has one "Full_period" error.

            # If errors are per-timepoint (timeAggregationFunc is None):
            if self._timeAggregationFunc is None:
                aggregated_by_region_per_timepoint = ErrorTimeSeries(observations=[])
                # locationvalues_per_timepoint() yields dicts like {location: Error_at_t} for each timepoint t
                for (
                    timepoint_str,
                    errors_at_timepoint_dict,
                ) in evaluation_result.items_grouped_by_timeperiod_str().items():
                    if not errors_at_timepoint_dict:
                        continue

                    region_values_at_t = [
                        err.value for err in errors_at_timepoint_dict.values() if not np.isnan(err.value)
                    ]
                    if not region_values_at_t:
                        # All values were NaN or empty for this timepoint
                        aggregated_value_for_timepoint = float("nan")
                    else:
                        try:
                            aggregated_value_for_timepoint = self._regionAggregationFunc(region_values_at_t)
                        except Exception as e:
                            logging.error(
                                f"Region aggregation function failed for timepoint '{timepoint_str}': {e}",
                                exc_info=True,
                            )
                            aggregated_value_for_timepoint = float("nan")

                    # Find the original TimePeriod object for this timepoint_str
                    # This is a bit inefficient; might be better to iterate differently if this is common.
                    original_tp = next(
                        (
                            obs.time_period
                            for loc_errors in evaluation_result.values()
                            for obs in loc_errors.observations
                            if str(obs.time_period) == timepoint_str
                        ),
                        timepoint_str,
                    )

                    aggregated_by_region_per_timepoint.observations.append(
                        Error(time_period=original_tp, value=aggregated_value_for_timepoint)
                    )
                if aggregated_by_region_per_timepoint.observations:
                    return MultiLocationErrorTimeSeries(
                        timeseries_dict={"Full_region": aggregated_by_region_per_timepoint}
                    )
                else:  # No data to aggregate
                    return MultiLocationErrorTimeSeries(timeseries_dict={})

            else:  # Errors are already time-aggregated (one "Full_period" value per location)
                all_location_time_aggregated_errors: List[float] = []
                for loc_error_series in evaluation_result.values():
                    if loc_error_series.observations:  # Should be one observation with time_period="Full_period"
                        val = loc_error_series.observations[0].value
                        if not np.isnan(val):  # Only include non-NaNs in aggregation
                            all_location_time_aggregated_errors.append(val)

                if not all_location_time_aggregated_errors:
                    global_aggregated_value = float("nan")
                else:
                    try:
                        global_aggregated_value = self._regionAggregationFunc(all_location_time_aggregated_errors)
                    except Exception as e:
                        logging.error(f"Global region aggregation function failed: {e}", exc_info=True)
                        global_aggregated_value = float("nan")

                final_aggregated_result = MultiLocationErrorTimeSeries(
                    timeseries_dict={
                        "Full_region": ErrorTimeSeries(
                            observations=[Error(time_period="Full_period", value=global_aggregated_value)]
                        )
                    }
                )
                return final_aggregated_result

        return evaluation_result


# Required for np.isnan
import numpy as np
