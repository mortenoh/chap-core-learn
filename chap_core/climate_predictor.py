# Improvement Suggestions:
# 1. **Type Hinting**: Enhance type hints for attributes (e.g., `self._models` in `MonthlyClimatePredictor`) and ensure all methods have explicit return type hints for better code clarity and static analysis.
# 2. **Error Handling**: Implement more robust error handling, particularly in `train` methods (e.g., for empty `train_data`, insufficient data for training) and in `FutureWeatherFetcher` implementations if data loading or prediction fails.
# 3. **`SeasonalForecastFetcher` Implementation**: The `SeasonalForecastFetcher` is currently a stub. If it's a planned feature, its implementation for loading and providing actual seasonal forecasts needs to be completed and documented.
# 4. **Predictor Model Configurability**: `MonthlyClimatePredictor` and `WeeklyClimatePredictor` hardcode `sklearn.linear_model.LinearRegression`. Consider allowing the regression model type or its parameters to be configurable for greater flexibility.
# 5. **`FetcherNd` Logic**: The slicing logic in `FetcherNd.get_future_weather` (`getattr(data, field.name)[-len(period_range) :]`) could fail if `len(period_range)` exceeds available historical data. Add checks or alternative logic (e.g., repeat the very last known value, raise a specific error).

"""
This module provides tools for predicting climate data and fetching future weather forecasts.

It includes:
- A factory function `get_climate_predictor` to obtain a suitable climate predictor.
- `MonthlyClimatePredictor` and `WeeklyClimatePredictor` using linear regression
  based on one-hot encoded time periods (month or week).
- An interface `FutureWeatherFetcher` and several implementations:
    - `SeasonalForecastFetcher`: Placeholder for using pre-generated seasonal forecasts.
    - `QuickForecastFetcher`: Trains a simple predictor on historical data on-the-fly.
    - `FetcherNd`: A naive predictor that repeats the last observed values.
"""

import dataclasses
from collections import defaultdict
from typing import Dict, Optional, Type, Union  # Added for type hints

import numpy as np
from sklearn import linear_model
from sklearn.base import RegressorMixin  # Added for type hints

from chap_core.spatio_temporal_data.temporal_dataclass import DataSet
from chap_core.time_period import Month, PeriodRange, Week

from .datatypes import ClimateData, SimpleClimateData


def get_climate_predictor(train_data: DataSet[ClimateData]) -> "MonthlyClimatePredictor | WeeklyClimatePredictor":
    """
    Factory function that selects, trains, and returns a climate predictor.

    The choice of predictor (Monthly or Weekly) is based on the time period
    type of the input `train_data`.

    Args:
        train_data (DataSet[ClimateData]): Historical climate data to train the predictor on.

    Returns:
        MonthlyClimatePredictor | WeeklyClimatePredictor: A trained climate predictor instance.

    Raises:
        AssertionError: If the time period type in `train_data` is not Month or Week.
    """
    if not train_data:
        raise ValueError("Training data cannot be empty.")

    first_period = train_data.period_range[0]
    if isinstance(first_period, Month):
        estimator = MonthlyClimatePredictor()
    elif isinstance(first_period, Week):
        estimator = WeeklyClimatePredictor()
    else:
        raise AssertionError(f"Unsupported time period type: {type(first_period)}. Expected Month or Week.")

    estimator.train(train_data)
    return estimator


# --- Monthly Predictor using one-hot encoding of month as features ---
class MonthlyClimatePredictor:
    """
    Predicts monthly climate variables using linear regression.

    This predictor trains a separate linear regression model for each climate variable
    (e.g., temperature, precipitation) and for each location. The features for the
    regression are one-hot encoded months of the year.
    """

    def __init__(self):
        """Initializes the MonthlyClimatePredictor."""
        self._models: Dict[str, Dict[str, RegressorMixin]] = defaultdict(dict)  # Stores [location][field] -> model
        self._cls: Optional[Type[Union[SimpleClimateData, ClimateData]]] = (  # Use Union with Optional
            None  # Stores the data class type (e.g., ClimateData)
        )

    def _feature_matrix(self, time_period: PeriodRange) -> np.ndarray:
        """
        Create a one-hot encoded feature matrix based on the month of each period.

        Args:
            time_period (PeriodRange): The range of time periods to encode.

        Returns:
            np.ndarray: A binary matrix where columns represent months (1-12)
                        and rows correspond to periods in `time_period`.
        """
        # Ensure time_period contains Month instances
        if not all(isinstance(p, Month) for p in time_period):  # More robust check
            raise ValueError("MonthlyClimatePredictor expects Month periods.")
        return time_period.month[:, None] == np.arange(1, 13)

    def train(self, train_data: DataSet[ClimateData | SimpleClimateData]):
        """
        Train one linear regression model per climate variable, per location.

        Args:
            train_data (DataSet[ClimateData | SimpleClimateData]): Labeled climate data for training.
                                                                  Can be `ClimateData` or `SimpleClimateData`.

        Raises:
            ValueError: If training data is empty or contains NaN values in target variables.
        """
        if not train_data:
            raise ValueError("Training data for MonthlyClimatePredictor cannot be empty.")

        # Remove 'disease_cases' if present, as it's not a climate variable
        # This assumes ClimateData might have it, SimpleClimateData should not.
        if any(hasattr(data_item, "disease_cases") for data_item in train_data.values()):
            train_data = train_data.remove_field("disease_cases")

        for location, data_item in train_data.items():
            if not data_item.time_period:  # Check if data_item itself has time periods
                logger.warning(f"Skipping location {location} due to empty time_period in data_item.")
                continue

            self._cls = data_item.__class__  # Store the type of the data item (e.g. SimpleClimateData)
            x = self._feature_matrix(data_item.time_period)

            for field in dataclasses.fields(data_item):
                if field.name == "time_period":
                    continue

                y = getattr(data_item, field.name)
                if np.isnan(y).any():
                    raise ValueError(
                        f"Missing (NaN) values found in training data for field '{field.name}' at location '{location}'."
                    )

                model = linear_model.LinearRegression()
                model.fit(x, y[:, None])  # Fit with shape (n_samples, 1)
                self._models[location][field.name] = model

        if not self._models:
            raise ValueError("No models were trained. Check training data content and structure.")

    def predict(self, time_period: PeriodRange) -> DataSet[SimpleClimateData]:
        """
        Predict climate variables for a given future time period range.

        Args:
            time_period (PeriodRange): The range of time periods for which to predict.

        Returns:
            DataSet[SimpleClimateData]: A DataSet containing the forecasted climate data,
                                        keyed by location. Each entry will be of the type
                                        stored in `self._cls` during training.

        Raises:
            RuntimeError: If the predictor has not been trained yet.
            ValueError: If `time_period` is empty.
        """
        if not self._models or self._cls is None:
            raise RuntimeError("Predictor has not been trained. Call train() before predict().")
        if not time_period:
            raise ValueError("Prediction time_period cannot be empty.")

        x = self._feature_matrix(time_period)
        prediction_dict = {}

        for location, models_for_location in self._models.items():
            field_predictions = {}
            for field_name, model in models_for_location.items():
                field_predictions[field_name] = model.predict(x).ravel()

            # Ensure self._cls is callable (it should be a type)
            if not callable(self._cls):
                raise TypeError(f"Stored class type self._cls is not callable: {self._cls}")

            prediction_dict[location] = self._cls(time_period=time_period, **field_predictions)

        return DataSet(prediction_dict)


# --- Weekly predictor variant using one-hot encoded week number ---
class WeeklyClimatePredictor(MonthlyClimatePredictor):
    """
    Predicts weekly climate variables using linear regression.

    This predictor is a variant of `MonthlyClimatePredictor` adapted for weekly data.
    It uses one-hot encoded week numbers (1-52) as features. Week 53, if present,
    is typically mapped to week 52's features.
    """

    def _feature_matrix(self, time_period: PeriodRange) -> np.ndarray:
        """
        Create a one-hot encoded feature matrix based on the week number of each period.

        Week 53 is mapped to the features of week 52.

        Args:
            time_period (PeriodRange): The range of time periods to encode.

        Returns:
            np.ndarray: A binary matrix where columns represent weeks (1-52)
                        and rows correspond to periods in `time_period`.

        Raises:
            ValueError: If `time_period` does not consist of `Week` instances.
        """
        if not all(isinstance(p, Week) for p in time_period):
            raise ValueError("WeeklyClimatePredictor expects Week periods.")

        week_numbers = time_period.week
        # Map week 53 to 52 for feature matrix alignment
        week_numbers_adjusted = np.where(week_numbers == 53, 52, week_numbers)

        # Create one-hot encoding for weeks 1-52
        t = week_numbers_adjusted[:, None] == np.arange(1, 53)
        return t


# --- Forecast Fetchers ---
class FutureWeatherFetcher:
    """
    Abstract base class (interface) for classes that fetch future weather data.

    Subclasses must implement the `get_future_weather` method.
    """

    def get_future_weather(self, period_range: PeriodRange) -> DataSet[SimpleClimateData]:
        """
        Fetch or generate future weather data for the given period range.

        Args:
            period_range (PeriodRange): The time period range for which to get weather data.

        Returns:
            DataSet[SimpleClimateData]: A DataSet containing the future weather data.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement get_future_weather.")


class SeasonalForecastFetcher(FutureWeatherFetcher):
    """
    Placeholder for a forecast fetcher that uses pre-generated seasonal forecasts.

    This class is intended to load seasonal forecast data from a specified storage location.
    The actual data loading and parsing logic is currently a stub.
    """

    def __init__(self, folder_path: str):
        """
        Initializes the SeasonalForecastFetcher.

        Args:
            folder_path (str): Path to the directory or file(s) containing
                               the pre-generated seasonal forecast data.
        """
        self.folder_path = folder_path
        logger.info(
            f"SeasonalForecastFetcher initialized with folder_path: {folder_path}. Note: Implementation is a stub."
        )

    def get_future_weather(self, period_range: PeriodRange) -> DataSet[SimpleClimateData]:
        """
        Load and return pre-generated seasonal forecasts for the specified period range.

        NOTE: This is currently a stub implementation and will raise NotImplementedError.
        The actual logic to read from `self.folder_path`, parse the data,
        and align it with `period_range` needs to be implemented.

        Args:
            period_range (PeriodRange): The time period range for which to retrieve forecasts.

        Returns:
            DataSet[SimpleClimateData]: Forecasted data.

        Raises:
            NotImplementedError: As this is a stub implementation.
        """
        # Stub implementation — to be filled with real forecast loading logic
        logger.warning(f"SeasonalForecastFetcher.get_future_weather called for {period_range}, but it's a STUB.")
        raise NotImplementedError("Seasonal forecast loading logic is not yet implemented.")


class QuickForecastFetcher(FutureWeatherFetcher):
    """
    Forecast fetcher that uses historical data to train a climate predictor on-the-fly.

    This fetcher is useful for quick simulations, as a fallback when other forecast
    sources are unavailable, or when a simple extrapolation of historical climate
    patterns is sufficient. It uses `get_climate_predictor` to create and train
    either a `MonthlyClimatePredictor` or `WeeklyClimatePredictor`.
    """

    def __init__(self, historical_data: DataSet[ClimateData | SimpleClimateData]):
        """
        Initializes the QuickForecastFetcher and trains a climate predictor.

        Args:
            historical_data (DataSet[ClimateData | SimpleClimateData]): A DataSet containing
                                                                       historical climate data
                                                                       to train the internal predictor.

        Raises:
            ValueError: If `historical_data` is empty.
        """
        if not historical_data:
            raise ValueError("Historical data for QuickForecastFetcher cannot be empty.")
        self._climate_predictor = get_climate_predictor(historical_data)

    def get_future_weather(self, period_range: PeriodRange) -> DataSet[SimpleClimateData]:
        """
        Generate future weather forecasts using the internally trained climate predictor.

        Args:
            period_range (PeriodRange): The time period range for which to generate forecasts.

        Returns:
            DataSet[SimpleClimateData]: A DataSet containing the forecasted weather data.

        Raises:
            ValueError: If `period_range` is empty.
        """
        if not period_range:
            raise ValueError("Prediction period_range for QuickForecastFetcher cannot be empty.")
        return self._climate_predictor.predict(period_range)


class FetcherNd(FutureWeatherFetcher):
    """
    Naive "No-Delta" (Nd) forecast fetcher that repeats the last observed values.

    This fetcher provides a baseline forecast by taking the most recent
    `N` observations from historical data, where `N` is the length of the
    requested `period_range`, and repeating them.
    """

    def __init__(self, historical_data: DataSet[SimpleClimateData | ClimateData]):
        """
        Initializes the FetcherNd with historical data.

        Args:
            historical_data (DataSet[SimpleClimateData | ClimateData]): Historical data to draw from.

        Raises:
            ValueError: If `historical_data` is empty or contains no valid data items.
        """
        if not historical_data:
            raise ValueError("Historical data for FetcherNd cannot be empty.")

        # Get the class type from the first valid data item
        first_valid_item = next((item for item in historical_data.values() if item is not None), None)
        if first_valid_item is None:
            raise ValueError("Historical data for FetcherNd contains no valid data items.")

        self.historical_data: DataSet[SimpleClimateData | ClimateData] = historical_data
        self._cls: Type[Union[SimpleClimateData, ClimateData]] = first_valid_item.__class__  # Use Union

    def get_future_weather(
        self, period_range: PeriodRange
    ) -> DataSet[Union[SimpleClimateData, ClimateData]]:  # Use Union
        """
        Generate naive forecasts by repeating the last N observed values.

        For each location and climate variable, this method takes the last `len(period_range)`
        values from the historical data. If historical data is shorter than `len(period_range)`,
        it will use all available historical data for that variable, effectively repeating
        the most recent block of observations.

        Args:
            period_range (PeriodRange): The desired period range for the forecast.
                                        The length of this range determines how many
                                        recent historical values are repeated.

        Returns:
            DataSet[SimpleClimateData | ClimateData]: A DataSet containing the naive forecasts.
                                                      The type of data items will match `self._cls`.

        Raises:
            ValueError: If `period_range` is empty.
        """
        if not period_range:
            raise ValueError("Prediction period_range for FetcherNd cannot be empty.")

        prediction_dict = {}
        num_periods_to_forecast = len(period_range)

        for location, hist_data_item in self.historical_data.items():
            forecast_fields = {}
            for field in dataclasses.fields(hist_data_item):
                if field.name == "time_period":
                    continue

                historical_values = getattr(hist_data_item, field.name)
                if len(historical_values) == 0:
                    logger.warning(f"No historical values for field '{field.name}' at location '{location}'. Skipping.")
                    # Or fill with NaNs, depending on desired behavior
                    # forecast_fields[field.name] = np.full(num_periods_to_forecast, np.nan)
                    continue

                # Take the last N values, or all if N > available length
                # If num_periods_to_forecast is larger than available, this will take all available.
                # We need to ensure the output array has length num_periods_to_forecast.

                if len(historical_values) >= num_periods_to_forecast:
                    forecast_values = historical_values[-num_periods_to_forecast:]
                else:
                    # Repeat the last value if historical data is shorter than forecast length
                    # Or tile the available data. Here, we'll tile and then take the last N.
                    # This ensures the output is always of length num_periods_to_forecast.
                    num_repeats = (num_periods_to_forecast + len(historical_values) - 1) // len(historical_values)
                    tiled_values = np.tile(historical_values, num_repeats)
                    forecast_values = tiled_values[-num_periods_to_forecast:]

                forecast_fields[field.name] = forecast_values

            if forecast_fields:  # Only add if we have some fields
                prediction_dict[location] = self._cls(time_period=period_range, **forecast_fields)
            else:
                logger.warning(f"No forecastable fields found for location {location} in FetcherNd.")

        return DataSet(prediction_dict)


# Need to add logger definition if not already present at module level
import logging

logger = logging.getLogger(__name__)
