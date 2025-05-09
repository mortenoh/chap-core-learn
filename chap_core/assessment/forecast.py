# Improvement Suggestions:
# 1. **Module Docstring**: Add a comprehensive module docstring that describes the purpose of this module: providing functions to generate forecasts under various scenarios, primarily for model assessment, backtesting, and future prediction with synthesized weather. (Primary task).
# 2. **Typo Correction (`prediction_lenght`)**: Correct the typo in the `prediction_lenght` parameter in `multi_forecast` to `prediction_length`. This is a minor functional correction for consistency.
# 3. **Model Parameter Configuration in `forecast`**: The direct setting of `model._num_warmup` and `model._num_samples` in the `forecast` function is specific to certain model types (e.g., Bayesian). This should be documented clearly, or ideally, such parameters should be configured via the model's own interface or initialization, making the `forecast` function more generic. The dead code `if False and hasattr(model, "diagnose"):` should be removed.
# 4. **Specific Type Hints for Models**: Refine type hints for `model`, `Estimator`, and `Predictor` parameters. If there's a common base class or protocol (e.g., defining `train`, `forecast`, `set_graph` methods), use it for more precise typing instead of generic `Any` or relying on duck typing implicitly.
# 5. **Error Handling and Input Validation**: Enhance error handling for model operations (train, predict) and data processing steps. For example, `train_test_split_with_weather` could return empty datasets if splits are invalid; ensure subsequent operations handle this. Document expected exceptions from model methods.

"""
This module provides functions for generating forecasts in various contexts,
such as single look-ahead predictions, rolling forecasts for backtesting,
and forecasting with predicted future weather.

These functions are typically used within model assessment and evaluation pipelines
to understand model performance under different conditions and to generate
predictions for periods where actual future covariates are not yet available.
"""

import logging
from typing import Any, Dict, Iterable, List, Optional  # Added Dict, List, Tuple

from chap_core.assessment.dataset_splitting import train_test_split_with_weather
from chap_core.assessment.evaluator import EvaluationError  # Added import
from chap_core.assessment.prediction_evaluator import Estimator, Predictor  # These are likely protocols or base classes
from chap_core.climate_predictor import get_climate_predictor
from chap_core.datatypes import ClimateData, Samples, TimeSeriesData  # Added ClimateData
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet
from chap_core.time_period.date_util_wrapper import Month, PeriodRange, TimeDelta
from chap_core.validators import validate_training_data

logger = logging.getLogger(__name__)


def forecast(
    model: Any,  # Should be a specific model protocol/base class
    dataset: DataSet[TimeSeriesData],
    prediction_length: TimeDelta,
    graph: Optional[Any] = None,  # Type for graph if known (e.g., libpysal.weights.W)
) -> Dict[str, Samples]:  # Return type is Dict[location_id, Samples]
    """
    Generates a forecast for a specified duration using a given model and dataset.

    The process involves:
    1. Splitting the dataset into training data and a test period (for future weather).
    2. Optionally setting a graph for spatial models.
    3. Setting hardcoded warmup and sample numbers (NOTE: specific to Bayesian models).
    4. Training the model on the training data.
    5. Generating predictions using future weather covariates.

    Args:
        model (Any): A forecasting model instance. Expected to have `train()` and `forecast()`
                     methods, and optionally `set_graph()`.
        dataset (DataSet[TimeSeriesData]): The full input dataset, including features and labels.
        prediction_length (TimeDelta): The duration into the future to forecast.
        graph (Optional[Any]): An optional dependency graph (e.g., spatial weights matrix)
                               if required by the model. Defaults to None.

    Returns:
        Dict[str, Samples]: A dictionary mapping location identifiers to `Samples` objects
                            containing the forecast distributions.
    """
    logger.info(f"Generating forecast for {prediction_length} duration into the future.")

    # Determine forecast starting period based on the end of the dataset and prediction_length
    # This assumes prediction_length is a duration to subtract from the dataset's end.
    if dataset.end_timestamp is None or prediction_length > (
        dataset.end_timestamp - dataset.start_timestamp
    ):  # Basic check
        raise ValueError("Dataset duration is too short for the given prediction_length or timestamps are invalid.")

    # split_point is the timestamp marking the end of training data / start of prediction period context
    split_point_timestamp = dataset.end_timestamp - prediction_length
    # Convert timestamp to a TimePeriod object (Month in this case, should be dynamic or configurable)
    # This assumes monthly data if not otherwise specified by dataset.period_range.delta
    # TODO: Make split_period derivation more robust based on dataset.period_range.delta
    split_period = Month(split_point_timestamp.year, split_point_timestamp.month)
    logger.debug(f"Split point for train/test: {split_period} (derived from {split_point_timestamp})")

    train_data, _test_set, future_weather = train_test_split_with_weather(
        dataset, split_period, extension=prediction_length
    )
    # _test_set contains actuals for the forecast period, future_weather contains covariates.

    if not train_data:
        logger.warning("Training data is empty after split. Model training might fail or be trivial.")
    if not future_weather:
        logger.warning("Future weather data is empty after split. Prediction might fail or be trivial.")

    if graph is not None and hasattr(model, "set_graph"):
        logger.debug("Setting graph for the model.")
        model.set_graph(graph)

    # WARNING: Direct attribute setting is specific and might break encapsulation.
    # Prefer configuring models via their __init__ or dedicated methods.
    if hasattr(model, "_num_warmup"):
        model._num_warmup = 1000
        logger.debug(f"Set model._num_warmup to {model._num_warmup}")
    if hasattr(model, "_num_samples"):
        model._num_samples = 400
        logger.debug(f"Set model._num_samples to {model._num_samples}")

    logger.info("Training the model...")
    model.train(train_data)
    logger.info("Model training complete.")

    # The `if False` block is dead code and has been removed.
    # if hasattr(model, "diagnose"):
    #     logger.info("Diagnosing model posterior (if supported)...")
    #     model.diagnose()

    logger.info("Generating forecasts using future weather...")
    # n_samples for forecast might also need to be configurable
    predictions = model.forecast(future_weather, n_samples=10, forecast_delta=prediction_length)
    logger.info("Forecast generation complete.")
    return predictions


def multi_forecast(
    model: Any,  # Should be a specific model protocol/base class
    dataset: DataSet[TimeSeriesData],
    prediction_length: TimeDelta,  # Corrected typo from prediction_lenght
    pre_train_delta: TimeDelta,
) -> Iterable[Dict[str, Samples]]:
    """
    Performs multiple rolling forecasts over a dataset.

    It iteratively shortens the dataset from the end, creating multiple windows.
    For each window, it calls the `forecast` function. This is useful for
    backtesting a model's performance over different periods using a rolling origin.

    Args:
        model (Any): The model to use for forecasting. Must have `train` and `forecast` methods.
        dataset (DataSet[TimeSeriesData]): The full dataset to perform rolling forecasts on.
        prediction_length (TimeDelta): The length of each individual forecast window.
        pre_train_delta (TimeDelta): The minimum length of training data required before
                                     the first forecast window can start.

    Returns:
        Iterable[Dict[str, Samples]]: A generator yielding prediction dictionaries (location to Samples)
                                      for each rolling forecast window, in reverse chronological order.
    """
    logger.info(f"Setting up multi-forecast: prediction_length={prediction_length}, pre_train_delta={pre_train_delta}")
    current_processing_dataset = dataset
    datasets_for_forecasting: List[DataSet[TimeSeriesData]] = []

    # Determine the timestamp for the start of the first forecast, ensuring enough pre-train data
    # This is the earliest point in time for which the *end* of a forecast window can occur.
    min_end_of_forecast_timestamp = dataset.start_timestamp + pre_train_delta + prediction_length

    while current_processing_dataset.end_timestamp >= min_end_of_forecast_timestamp:
        datasets_for_forecasting.append(current_processing_dataset)

        # Determine the split point to shorten the dataset for the next iteration's training data
        # This split_point becomes the end of the training data for the *next* iteration's forecast call.
        # The current `current_processing_dataset` is used as is for the current forecast call in the generator.
        split_point_timestamp = current_processing_dataset.end_timestamp - prediction_length

        # TODO: Make split_period derivation more robust based on dataset.period_range.delta
        split_period = Month(split_point_timestamp.year, split_point_timestamp.month)

        # Reduce the dataset for the next iteration by taking the training part of the current one
        # The actual forecast for `current_processing_dataset` will use its full history up to split_period for training.
        train_part, _, _ = train_test_split_with_weather(
            current_processing_dataset, split_period, extension=prediction_length
        )
        if not train_part or not train_part.period_range or len(train_part.period_range) == 0:
            logger.warning("Training part became empty or invalid during multi_forecast setup. Stopping iteration.")
            break
        current_processing_dataset = train_part

    num_windows = len(datasets_for_forecasting)
    if num_windows == 0:
        logger.warning("No forecast windows could be generated with the given parameters and dataset size.")
        return iter([])  # Return empty iterator

    logger.info(f"Will generate forecasts for {num_windows} rolling windows.")

    # Forecasts are generated for datasets in reversed order (from earliest to latest end time)
    return (forecast(model, ds, prediction_length) for ds in reversed(datasets_for_forecasting))


def forecast_ahead(
    estimator: Estimator,  # Protocol/base class for estimators
    dataset: DataSet[TimeSeriesData],
    prediction_length: int,  # Number of periods (e.g., months, weeks)
) -> Dict[str, Samples]:
    """
    Trains an estimator on the provided dataset and then forecasts into the future
    using predicted weather for the forecast horizon.

    Args:
        estimator (Estimator): A model estimator instance with a `train()` method
                               that returns a `Predictor`.
        dataset (DataSet[TimeSeriesData]): The training dataset.
        prediction_length (int): The number of periods (e.g., months, weeks) to forecast ahead.

    Returns:
        Dict[str, Samples]: A dictionary mapping location IDs to `Samples` objects
                            containing the forecast distributions.

    Raises:
        ValueError: If `prediction_length` is not positive.
        Exceptions from `validate_training_data`, `estimator.train`, or `forecast_with_predicted_weather`.
    """
    logger.info(f"Forecasting {prediction_length} periods ahead using estimator {estimator.__class__.__name__}.")
    if prediction_length <= 0:
        raise ValueError("prediction_length must be a positive integer.")

    validate_training_data(dataset, estimator)  # Validate before training

    logger.info("Training the estimator...")
    predictor = estimator.train(dataset)  # Train model on the full dataset provided
    logger.info("Estimator training complete.")

    return forecast_with_predicted_weather(
        predictor=predictor,
        historic_data=dataset,  # Use the same dataset as basis for predicting weather
        prediction_length=prediction_length,
    )


def forecast_with_predicted_weather(
    predictor: Predictor,  # Protocol/base class for predictors
    historic_data: DataSet[TimeSeriesData],
    prediction_length: int,  # Number of periods
) -> Dict[str, Samples]:
    """
    Generates future forecasts using a trained predictor and internally predicted future weather.

    Future weather is predicted using a simple climate predictor trained on the `historic_data`.

    Args:
        predictor (Predictor): A trained predictor object with a `predict` method.
        historic_data (DataSet[TimeSeriesData]): The historical dataset, used both as context for
                                                 the main forecast and as training data for the
                                                 internal climate predictor.
        prediction_length (int): The number of periods (e.g., months, weeks) to forecast ahead.

    Returns:
        Dict[str, Samples]: A dictionary mapping location IDs to `Samples` objects
                            containing the forecast distributions.

    Raises:
        ValueError: If `prediction_length` is not positive or `historic_data` is empty/invalid.
        Exceptions from `get_climate_predictor` or `predictor.predict`.
    """
    if prediction_length <= 0:
        raise ValueError("prediction_length must be a positive integer.")
    if not historic_data or not historic_data.period_range:
        raise ValueError("historic_data must be a non-empty DataSet with a valid period_range.")

    # Determine time delta from historic data and create prediction range for future weather
    # This assumes all locations/series in historic_data share the same time_delta.
    time_delta_of_data = historic_data.period_range.time_delta
    if time_delta_of_data is None:  # Should ideally be set in PeriodRange
        # Attempt to infer from first period if PeriodRange.time_delta is None
        first_period_in_range = historic_data.period_range[0]
        if hasattr(first_period_in_range, "time_delta"):
            time_delta_of_data = first_period_in_range.time_delta
        else:
            raise ValueError("Cannot determine time_delta from historic_data.period_range.")

    future_start_period = historic_data.end_timestamp  # Prediction starts right after historic data ends
    # Create a PeriodRange for the future weather prediction
    # The end timestamp for the range needs to be calculated carefully.
    # If prediction_length is N periods, and current end is T, new end is T + N*delta.
    # PeriodRange constructor might handle this if end is exclusive.
    # Example: if end_timestamp is Dec 2023, delta is 1M, prediction_length is 3,
    # we need Jan, Feb, Mar 2024. Range end would be start of Apr 2024.
    future_end_timestamp = future_start_period
    for _ in range(prediction_length):
        future_end_timestamp = future_end_timestamp + time_delta_of_data

    prediction_time_range = PeriodRange(
        start_timestamp=future_start_period,  # Start from the end of historic data
        end_timestamp=future_end_timestamp,  # This should be exclusive end for N periods
        time_delta=time_delta_of_data,
    )
    # Correction: PeriodRange end is typically inclusive. If we need N periods *after* historic_data.end_timestamp:
    # The first future period starts *after* historic_data.end_timestamp.
    # So, if historic_data.end_timestamp is Dec-2023, first future period is Jan-2024.
    # If prediction_length = 1, future_range is Jan-2024 to Jan-2024.
    # If prediction_length = 3, future_range is Jan-2024 to Mar-2024.

    # Let's use PeriodRange.from_start_and_steps
    actual_future_start_period = historic_data.period_range.end_time_period.next()
    prediction_time_range = PeriodRange.from_start_and_steps(actual_future_start_period, prediction_length - 1)

    logger.info(
        f"Predicting future weather for range: {prediction_time_range.start_time_period} to {prediction_time_range.end_time_period}"
    )
    # Predict future weather using a simple climate predictor trained on historic_data
    # Note: historic_data might contain more than just climate variables.
    # get_climate_predictor needs to handle this (e.g., by selecting relevant fields).
    try:
        # Ensure historic_data passed to get_climate_predictor is suitable (e.g. contains climate features)
        # This might require selecting only climate-related fields from historic_data if it's mixed.
        # For now, assume get_climate_predictor can handle it or historic_data is purely climate.
        climate_predictor = get_climate_predictor(
            historic_data.to_type(ClimateData, remove_missing_fields=True)
        )  # Ensure it's ClimateData
    except Exception as e:
        logger.error(f"Failed to initialize climate predictor: {e}", exc_info=True)
        raise EvaluationError(f"Could not get climate predictor: {e}")

    future_weather = climate_predictor.predict(prediction_time_range)
    logger.info("Future weather prediction complete.")

    logger.info("Running main forecast with predicted future weather...")
    # The main predictor uses historic_data as context and future_weather for covariates.
    predictions = predictor.predict(historic_data, future_weather)  # Pass num_samples if Predictor supports it
    logger.info("Main forecast with predicted weather complete.")
    return predictions
