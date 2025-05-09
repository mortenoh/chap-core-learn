# Improvement Suggestions:
# 1. **Module Docstring**: Add a comprehensive module docstring explaining the purpose of this file: to define a set of dataclasses used for representing disease observations, evaluation errors, and forecast samples, particularly in multi-location time series contexts for model assessment. (Primary task).
# 2. **`time_period` Type Hint**: The `time_period` attribute in `DiseaseObservation`, `Error`, and `Samples` is currently typed as `str`. If these are intended to represent structured time periods (e.g., specific months, weeks), consider using a more specific type like `chap_core.time_period.TimePeriod` or `pandas.Period`. If "Full_period" is a special string literal, document this usage.
# 3. **Complete Method Docstrings**: Add clear and informative docstrings for all methods within `MultiLocationDiseaseTimeSeries` and `MultiLocationErrorTimeSeries` that currently lack them (e.g., `__setitem__`, `__getitem__`, `locations`, `timeseries`, and various utility methods in `MultiLocationErrorTimeSeries`).
# 4. **Robust Error Handling in Utility Methods**: Replace `assert` statements in methods like `get_the_only_location`, `get_all_timeperiods`, and `timeseries_length` within `MultiLocationErrorTimeSeries` with more specific exceptions (e.g., `ValueError`, custom `InconsistentDataError`) and informative error messages.
# 5. **Consistency and Naming Conventions**:
#    - Review the initialization of `timeseries_dict` in `MultiLocationDiseaseTimeSeries` (uses `default_factory=dict`) versus `MultiLocationErrorTimeSeries` (no default factory, implying it's required at init). Ensure this difference is intentional and documented.
#    - Clarify the naming: `Forecast` class holds `List[Samples]` for one location. `MultiLocationForecast` holds `Dict[str, Forecast]`. This seems logical, but ensure docstrings make the single-location vs. multi-location distinction clear for each class.

"""
This module defines a collection of dataclasses used to represent various
data structures pertinent to model assessment and evaluation workflows.

These structures include representations for:
- Disease observations and time series, both for single and multiple locations.
- Evaluation error metrics, also structured for single/multiple locations and time series.
- Forecasted samples and collections of forecasts.

These dataclasses facilitate standardized handling of truth data, predictions,
and evaluation results throughout the CHAP-core assessment pipeline.
"""

from collections import defaultdict  # Added import
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple  # Added Optional, Tuple

# Consider importing a specific TimePeriod type if 'time_period: str' is too generic.
# from chap_core.time_period import TimePeriodBase # Example

# -------------------------------
# Disease Observations
# -------------------------------


@dataclass
class DiseaseObservation:
    """
    Represents a single observation of disease cases for a specific time period.

    Attributes:
        time_period (str): Identifier for the time period of the observation.
                           This could be a date string, a period code (e.g., "202301"),
                           or a special string like "Full_period" for aggregated values.
        disease_cases (int): The number of observed disease cases.
    """

    time_period: str  # Consider using a more specific TimePeriod type
    disease_cases: int


@dataclass
class DiseaseTimeSeries:
    """
    Represents a time series of disease case observations for a single location.

    Attributes:
        observations (List[DiseaseObservation]): A list of disease observations,
                                                 typically ordered chronologically.
    """

    observations: List[DiseaseObservation]


@dataclass
class MultiLocationDiseaseTimeSeries:
    """
    A collection of disease time series, keyed by location name or ID.

    This class acts like a dictionary mapping location identifiers (strings)
    to `DiseaseTimeSeries` objects.

    Attributes:
        timeseries_dict (Dict[str, DiseaseTimeSeries]): The dictionary holding
            location-specific disease time series. Initialized as an empty
            dictionary by default.
    """

    timeseries_dict: Dict[str, DiseaseTimeSeries] = field(default_factory=dict)

    def __setitem__(self, location: str, timeseries: DiseaseTimeSeries) -> None:
        """Sets the disease time series for a given location."""
        self.timeseries_dict[location] = timeseries

    def __getitem__(self, location: str) -> DiseaseTimeSeries:
        """Gets the disease time series for a given location."""
        return self.timeseries_dict[location]

    def locations(self) -> Iterator[str]:
        """Returns an iterator over the location keys."""
        return iter(self.timeseries_dict.keys())

    def timeseries(self) -> Iterator[DiseaseTimeSeries]:
        """Returns an iterator over the DiseaseTimeSeries objects."""
        return iter(self.timeseries_dict.values())

    def items(self) -> Iterator[Tuple[str, DiseaseTimeSeries]]:  # Added type hint and method
        """Returns an iterator over (location, DiseaseTimeSeries) pairs."""
        return iter(self.timeseries_dict.items())

    def values(self) -> Iterator[DiseaseTimeSeries]:  # Alias for timeseries() for dict-like behavior
        """Returns an iterator over the DiseaseTimeSeries objects."""
        return iter(self.timeseries_dict.values())

    def __len__(self) -> int:  # Added for dict-like behavior
        """Returns the number of locations."""
        return len(self.timeseries_dict)

    def __iter__(self) -> Iterator[str]:  # Iterate over locations by default
        """Returns an iterator over the location keys."""
        return iter(self.timeseries_dict.keys())


# -------------------------------
# Error Representations (MAE, RMSE, etc.)
# -------------------------------


@dataclass
class Error:
    """
    Represents a single calculated error value for a specific time period.

    Attributes:
        time_period (str): Identifier for the time period of the error value.
                           Can be a specific period string or "Full_period" for aggregated errors.
        value (float): The calculated error metric value.
    """

    time_period: str  # Consider using a more specific TimePeriod type or Union[TimePeriodType, str]
    value: float


@dataclass
class ErrorTimeSeries:
    """
    Represents a time series of error values for a single location or aggregation.

    Attributes:
        observations (List[Error]): A list of `Error` objects, typically ordered
                                    chronologically if representing per-timepoint errors.
    """

    observations: List[Error]


@dataclass
class MultiLocationErrorTimeSeries:
    """
    A collection of error time series, keyed by location name or ID.

    If errors are aggregated across regions, a special key like "Full_region" might be used.

    Attributes:
        timeseries_dict (Dict[str, ErrorTimeSeries]): The dictionary holding
            location-specific (or aggregated) error time series.
            Note: Unlike MultiLocationDiseaseTimeSeries, this does not use default_factory,
            implying timeseries_dict must be provided at initialization.
    """

    timeseries_dict: Dict[str, ErrorTimeSeries]  # No default_factory, must be initialized

    def __getitem__(self, location: str) -> ErrorTimeSeries:
        """Gets the error time series for a given location."""
        return self.timeseries_dict[location]

    def __setitem__(self, location: str, timeseries: ErrorTimeSeries) -> None:
        """Sets the error time series for a given location."""
        self.timeseries_dict[location] = timeseries

    def locations(self) -> Iterator[str]:
        """Returns an iterator over the location keys."""
        return iter(self.timeseries_dict.keys())

    def timeseries(self) -> Iterator[ErrorTimeSeries]:
        """Returns an iterator over the ErrorTimeSeries objects."""
        return iter(self.timeseries_dict.values())

    def items(self) -> Iterator[Tuple[str, ErrorTimeSeries]]:  # Added type hint and method
        """Returns an iterator over (location, ErrorTimeSeries) pairs."""
        return iter(self.timeseries_dict.items())

    def values(self) -> Iterator[ErrorTimeSeries]:  # Alias for timeseries()
        """Returns an iterator over the ErrorTimeSeries objects."""
        return iter(self.timeseries_dict.values())

    def __len__(self) -> int:  # Added
        """Returns the number of locations/aggregated series."""
        return len(self.timeseries_dict)

    def __iter__(self) -> Iterator[str]:  # Iterate over locations by default
        """Returns an iterator over the location keys."""
        return iter(self.timeseries_dict.keys())

    def num_locations(self) -> int:
        """
        Returns the number of distinct locations (or aggregated series) in the collection.
        """
        return len(self.timeseries_dict)

    def num_timeperiods(self) -> int:
        """
        Returns the number of unique time periods present in the error series.
        This assumes all locations have observations for the same set of time periods
        if not aggregated (verified by `get_all_timeperiods`).
        If time-aggregated, this might return 1 (for "Full_period").
        """
        all_periods = self.get_all_timeperiods()
        return len(all_periods) if all_periods is not None else 0

    def get_the_only_location(self) -> str:
        """
        Retrieves the location key when the collection is expected to contain data
        for only a single location or a single aggregated result (e.g., "Full_region").

        Returns:
            str: The single location key.

        Raises:
            ValueError: If the collection does not contain exactly one location/series.
        """
        if len(self.timeseries_dict) != 1:
            raise ValueError(
                f"Expected exactly one location/series, but found {len(self.timeseries_dict)}. Keys: {list(self.timeseries_dict.keys())}"
            )
        return list(self.timeseries_dict.keys())[0]

    def get_the_only_timeseries(self) -> ErrorTimeSeries:
        """
        Retrieves the `ErrorTimeSeries` when the collection is expected to contain
        data for only a single location or a single aggregated result.

        Returns:
            ErrorTimeSeries: The single error time series.

        Raises:
            ValueError: If the collection does not contain exactly one location/series.
        """
        if len(self.timeseries_dict) != 1:
            raise ValueError(
                f"Expected exactly one location/series, but found {len(self.timeseries_dict)}. Keys: {list(self.timeseries_dict.keys())}"
            )
        return list(self.timeseries_dict.values())[0]

    def get_all_timeperiods(self) -> Optional[List[str]]:
        """
        Retrieves a list of unique time period identifiers from the error series.

        It asserts that all contained `ErrorTimeSeries` (if multiple locations exist
        and are not time-aggregated) share the exact same sequence of time periods.
        If time periods are inconsistent across locations, it raises an AssertionError.
        If the collection is empty, returns None.

        Returns:
            Optional[List[str]]: A list of time period strings, or None if no series exist.

        Raises:
            AssertionError: If time periods are inconsistent across different locations' series.
        """
        timeperiods_set: Optional[List[str]] = None
        if not self.timeseries_dict:
            return None

        for ts in self.timeseries():
            current_periods = [o.time_period for o in ts.observations]
            if timeperiods_set is None:
                timeperiods_set = current_periods
            elif timeperiods_set != current_periods:
                # This indicates a data consistency issue if non-aggregated series are expected to align.
                raise AssertionError("Inconsistent time periods found across locations.")
        return timeperiods_set

    def timeseries_length(self) -> int:
        """
        Returns the number of timepoints (observations) in the error series.

        This method assumes that all `ErrorTimeSeries` objects in the collection
        have the same length (i.e., observations for the same number of timepoints).
        If lengths are inconsistent, it raises an AssertionError.
        If the collection is empty, returns 0.

        Returns:
            int: The common length of the error time series.

        Raises:
            AssertionError: If `ErrorTimeSeries` objects have inconsistent lengths.
        """
        if not self.timeseries_dict:
            return 0
        lengths = [len(ts.observations) for ts in self.timeseries()]
        if not lengths:  # Should not happen if timeseries_dict is not empty, but defensive.
            return 0
        if len(set(lengths)) != 1:
            raise AssertionError(f"ErrorTimeSeries objects have inconsistent lengths: {lengths}")
        return lengths[0]

    def items_grouped_by_timeperiod_str(self) -> Dict[str, Dict[str, Error]]:  # Added method
        """
        Restructures the data to group errors by time period string, then by location.
        Useful for aggregating errors across locations for each specific time point.

        Example:
        If original is {"locA": [Error(t1, vA1), Error(t2, vA2)], "locB": [Error(t1, vB1), Error(t2, vB2)]}
        Returns: {"t1": {"locA": Error(t1, vA1), "locB": Error(t1, vB1)},
                  "t2": {"locA": Error(t2, vA2), "locB": Error(t2, vB2)}}

        Returns:
            Dict[str, Dict[str, Error]]: A dictionary where keys are time period strings,
                                         and values are dictionaries mapping location ID to
                                         the `Error` object for that time period and location.
        """
        grouped_data: Dict[str, Dict[str, Error]] = defaultdict(dict)
        for location_id, error_ts in self.items():
            for error_observation in error_ts.observations:
                grouped_data[str(error_observation.time_period)][location_id] = error_observation
        return dict(grouped_data)  # Convert back to regular dict


# -------------------------------
# Forecasted Samples
# -------------------------------


@dataclass
class Samples:
    """
    Represents forecasted samples for a single time period at a single location.

    Attributes:
        time_period (str): Identifier for the time period of the forecast.
        disease_case_samples (List[float]): A list of forecast samples (e.g., from a
                                            probabilistic model's posterior distribution).
    """

    time_period: str  # Consider using a more specific TimePeriod type
    disease_case_samples: List[float]


@dataclass
class Forecast:  # This class represents a forecast for a single location over multiple time periods
    """
    Represents a time series of forecasted samples for a single location.

    Attributes:
        predictions (List[Samples]): A list of `Samples` objects, each corresponding
                                     to a specific time period in the forecast horizon.
                                     Typically ordered chronologically.
    """

    predictions: List[Samples]


@dataclass
class MultiLocationForecast:
    """
    A collection of forecasts, keyed by location name or ID.

    Each forecast pertains to a single location and consists of a time series
    of predicted samples.

    Attributes:
        timeseries (Dict[str, Forecast]): A dictionary mapping location identifiers (strings)
                                          to `Forecast` objects.
    """

    timeseries: Dict[str, Forecast]  # Should not use default_factory if it's always expected

    def __getitem__(self, location: str) -> Forecast:  # Added method
        """Gets the forecast for a given location."""
        return self.timeseries[location]

    def __setitem__(self, location: str, forecast_obj: Forecast) -> None:  # Added method
        """Sets the forecast for a given location."""
        self.timeseries[location] = forecast_obj

    def locations(self) -> Iterator[str]:  # Added method
        """Returns an iterator over the location keys."""
        return iter(self.timeseries.keys())

    def items(self) -> Iterator[Tuple[str, Forecast]]:  # Added method
        """Returns an iterator over (location, Forecast) pairs."""
        return iter(self.timeseries.items())

    def values(self) -> Iterator[Forecast]:  # Added method
        """Returns an iterator over the Forecast objects."""
        return iter(self.timeseries.values())

    def __len__(self) -> int:  # Added
        """Returns the number of locations with forecasts."""
        return len(self.timeseries)

    def __iter__(self) -> Iterator[str]:  # Iterate over locations by default
        """Returns an iterator over the location keys."""
        return iter(self.timeseries.keys())


# For type hint Tuple
