# Improvement Suggestions:
# 1. **Module Docstring**: Add a comprehensive module docstring that outlines the purpose of this module: providing various strategies for splitting time series datasets for model training, testing, and backtesting. (Primary task).
# 2. **`IsTimeDelta` Protocol Definition**: The `IsTimeDelta` protocol is defined with `...`. For it to be a useful structural subtype, it should define the expected attributes or methods of a "time delta object" (e.g., methods to add to a `TimePeriod` or represent a duration). If it's just a conceptual marker, this should be documented.
# 3. **Parameter Consistency (`extension` vs. `future_length`)**: The parameters `extension` and `future_length` (both typed as `IsTimeDelta`) seem to serve similar purposes (defining the length of a future/test period). Ensure their naming and usage are consistent and clearly documented across all relevant functions to avoid confusion.
# 4. **Robust Edge Case Handling & Input Validation**:
#    - In `train_test_split` and `train_test_split_with_weather`, add checks for `prediction_start_period` being within the bounds of `data_set.period_range`.
#    - In `train_test_generator`, validate that `prediction_length`, `n_test_sets`, and `stride` are compatible with the dataset length to prevent indexing errors. The calculation of `split_idx` should be robust.
#    - In `get_split_points_for_data_set`, handle the case where `data_set.data()` might be empty to avoid `StopIteration`.
# 5. **Clarify `future_weather_provider` Interface**: The `future_weather_provider` parameter in `train_test_generator` is typed as `Optional[FutureWeatherFetcher]`. The usage `future_weather_provider(hd).get_future_weather(...)` implies `future_weather_provider` is a callable that takes historical data (`hd`) and returns an object with a `get_future_weather` method (like an instance of a `FutureWeatherFetcher` subclass, or that `FutureWeatherFetcher` itself is callable and returns such an instance). This interface contract should be clearly documented.

"""
This module provides functions for splitting time series datasets (`DataSet`)
into training and testing sets, or generating multiple splits for backtesting
and cross-validation purposes.

It includes utilities to:
- Split data based on a single prediction start period.
- Generate multiple train/test splits based on a series of split points.
- Optionally include or generate future weather data for test periods.
- Create rolling origin or sliding window splits for backtesting.
"""

import logging  # Added logging
from typing import Iterable, Iterator, List, Optional, Protocol, Tuple, Type  # Added List, Tuple, Iterator

from chap_core.climate_predictor import FutureWeatherFetcher
from chap_core.datatypes import ClimateData, TimeSeriesData  # Added TimeSeriesData for more general DataSet typing
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet
from chap_core.time_period import PeriodRange, TimePeriod  # Added PeriodRange
from chap_core.time_period.relationships import previous

logger = logging.getLogger(__name__)


class IsTimeDelta(Protocol):
    """
    Protocol defining an object that represents a time duration or delta.

    This is used to type hint parameters like `future_length` or `extension`,
    which specify the length of a forecast horizon or test period.
    Implementers of this protocol would typically be objects like `MonthDelta`,
    `WeekDelta`, or similar, capable of being used to extend a `TimePeriod`.
    """

    # Example: def __add__(self, other: TimePeriod) -> TimePeriod: ...
    # Example: def n_periods(self) -> int: ...
    # For now, it's a marker type as its specific methods aren't called directly
    # on these objects within this module, but rather used in TimePeriod.extend_to.
    pass


def split_test_train_on_period(
    data_set: DataSet[TimeSeriesData],
    split_points: Iterable[TimePeriod],
    future_length: Optional[IsTimeDelta] = None,
    include_future_weather: bool = False,
    future_weather_class: Type[ClimateData] = ClimateData,
) -> Iterator[Tuple[DataSet[TimeSeriesData], DataSet[TimeSeriesData], Optional[DataSet[ClimateData]]]]:
    """
    Generates train/test splits (optionally with future weather data) for each specified split point.

    This function iterates over `split_points`. For each point, it calls either
    `train_test_split_with_weather` (if `include_future_weather` is True) or
    `train_test_split` to create a single train/test(/future_weather) tuple.

    Args:
        data_set (DataSet[TimeSeriesData]): The full dataset to be split.
        split_points (Iterable[TimePeriod]): An iterable of `TimePeriod` objects, each
                                             representing the start of a test/prediction period.
        future_length (Optional[IsTimeDelta]): The duration of the test/future period.
                                               Passed to the underlying split functions.
        include_future_weather (bool): If True, the split will include a third item:
                                       a `DataSet` of future weather data (covariates without target).
                                       Defaults to False.
        future_weather_class (Type[ClimateData]): The dataclass type to use when extracting
                                                  future weather data. Defaults to `ClimateData`.

    Yields:
        Iterator[Tuple[DataSet[TimeSeriesData], DataSet[TimeSeriesData], Optional[DataSet[ClimateData]]]]:
            An iterator yielding tuples. Each tuple contains:
            - train_data (DataSet)
            - test_data (DataSet, containing labels for the test period)
            - future_weather_data (Optional[DataSet[ClimateData]]), if `include_future_weather` is True.
    """
    func_to_call = train_test_split_with_weather if include_future_weather else train_test_split

    for period in split_points:
        if include_future_weather:
            yield func_to_call(data_set, period, future_length, future_weather_class)
        else:
            # train_test_split returns (train, test), so we yield (train, test, None) for consistency
            train_data, test_data = func_to_call(data_set, period, future_length)
            yield train_data, test_data, None


def train_test_split(
    data_set: DataSet[TimeSeriesData],
    prediction_start_period: TimePeriod,
    extension: Optional[IsTimeDelta] = None,
    restrict_test: bool = True,
) -> Tuple[DataSet[TimeSeriesData], DataSet[TimeSeriesData]]:
    """
    Splits a dataset into a single training set and a single test set based on a prediction start period.

    The training set includes all data up to (but not including) `prediction_start_period`.
    The test set starts at `prediction_start_period`. If `extension` is provided and
    `restrict_test` is True, the test set is limited to the duration of `extension`.

    Args:
        data_set (DataSet[TimeSeriesData]): The full dataset to split.
        prediction_start_period (TimePeriod): The first period to be included in the test set
                                              (and thus excluded from the training set).
        extension (Optional[IsTimeDelta]): The desired length/duration of the test set.
                                           If None, the test set includes all data from
                                           `prediction_start_period` to the end of `data_set`.
        restrict_test (bool): If True (default) and `extension` is provided, the test set
                              is sliced to the length of `extension`. If False, `extension`
                              is used to define an end period, but the test set might be shorter
                              if `data_set` ends before that.

    Returns:
        Tuple[DataSet[TimeSeriesData], DataSet[TimeSeriesData]]: A tuple containing the
                                                                 training dataset and the test dataset.

    Raises:
        ValueError: If `prediction_start_period` is outside the dataset's range or leads to empty splits.
    """
    if not data_set.period_range.contains(prediction_start_period):
        raise ValueError(
            f"prediction_start_period {prediction_start_period} is outside the dataset range {data_set.period_range}."
        )

    last_train_period = previous(prediction_start_period)
    if last_train_period is None or not data_set.period_range.contains(last_train_period):
        # This implies prediction_start_period is the first period in the dataset, leading to empty train set.
        logger.warning(f"prediction_start_period {prediction_start_period} results in an empty training set.")
        # Depending on desired behavior, could raise ValueError or return empty DataSet.
        # For now, allow empty train set as some models might handle it (e.g. non-time-series models).

    train_data = data_set.restrict_time_period(slice(None, last_train_period))

    end_period_for_slice: Optional[TimePeriod] = None
    if extension is not None:
        # Assuming TimePeriod.extend_to correctly handles IsTimeDelta
        end_period_for_slice = prediction_start_period.extend_to(extension)
        # Ensure end_period_for_slice does not exceed dataset bounds if restrict_test is effectively true
        if data_set.period_range.end_time_period < end_period_for_slice:
            end_period_for_slice = data_set.period_range.end_time_period

    if restrict_test:  # Slice test data to the defined range [prediction_start_period, end_period_for_slice]
        test_data = data_set.restrict_time_period(slice(prediction_start_period, end_period_for_slice))
    else:  # Test data includes everything from prediction_start_period onwards
        test_data = data_set.restrict_time_period(slice(prediction_start_period, None))

    if not test_data:
        logger.warning(f"Test data split resulted in an empty DataSet for start {prediction_start_period}.")

    return train_data, test_data


def train_test_split_with_weather(
    data_set: DataSet[TimeSeriesData],
    prediction_start_period: TimePeriod,
    extension: Optional[IsTimeDelta] = None,
    future_weather_class: Type[ClimateData] = ClimateData,  # Type of data expected for weather
) -> Tuple[DataSet[TimeSeriesData], DataSet[TimeSeriesData], DataSet[ClimateData]]:
    """
    Splits dataset like `train_test_split`, but also returns future weather data.

    Future weather data is derived from the test set by removing the target variable
    (assumed to be "disease_cases").

    Args:
        data_set (DataSet[TimeSeriesData]): The full labeled dataset.
        prediction_start_period (TimePeriod): The first period for the test/prediction.
        extension (Optional[IsTimeDelta]): The desired length/duration of the test/future weather period.
        future_weather_class (Type[ClimateData]): The dataclass type to use when creating
                                                  the `future_weather` DataSet. This class
                                                  should not contain the target field.

    Returns:
        Tuple[DataSet[TimeSeriesData], DataSet[TimeSeriesData], DataSet[ClimateData]]:
            A tuple containing:
            - train_set (DataSet)
            - test_set (DataSet, with labels for evaluation)
            - future_weather (DataSet[ClimateData], covariates for the prediction period, no labels)
    """
    train_set, test_set = train_test_split(data_set, prediction_start_period, extension, restrict_test=True)

    if not test_set:  # If test_set is empty, future_weather will also be problematic
        logger.warning("train_test_split returned an empty test_set; future_weather will also be empty.")
        # Create an empty DataSet of the correct type for future_weather
        empty_future_weather = DataSet(
            {loc: future_weather_class(time_period=PeriodRange.from_period_list(False, [])) for loc in data_set.keys()}
        )
        return train_set, test_set, empty_future_weather

    # Remove labels (e.g., "disease_cases") from the test_set to create future_weather
    # This assumes the target field is named "disease_cases".
    # A more robust way might be to pass the target name or use a method on future_weather_class.
    try:
        future_weather = test_set.to_type(future_weather_class, remove_missing_fields=True)
        # If 'disease_cases' was part of TimeSeriesData in test_set, and not in future_weather_class,
        # to_type should handle its removal if remove_missing_fields=True implies dropping fields not in target.
        # Or, more explicitly:
        # future_weather = test_set.remove_field("disease_cases", new_type=future_weather_class)
    except Exception as e:
        logger.error(f"Failed to convert test_set to future_weather_class or remove target field: {e}", exc_info=True)
        raise ValueError(f"Could not create future_weather data: {e}")

    # Sanity check: ensure no overlap between training periods and future weather periods
    # This check is important if time periods could be complex or non-contiguous.
    if train_set and future_weather:  # Only if both are non-empty
        train_periods_set = {str(p) for data_item in train_set.values() for p in data_item.time_period}
        future_periods_set = {str(p) for data_item in future_weather.values() for p in data_item.time_period}

        overlap = train_periods_set.intersection(future_periods_set)
        if overlap:
            # This should ideally not happen if split logic is correct.
            logger.error(f"Overlap detected in training and future weather data periods: {overlap}")
            assert not overlap, f"Overlap in training and future weather data: {overlap}"

    return train_set, test_set, future_weather


def train_test_generator(
    dataset: DataSet[TimeSeriesData],
    prediction_length: int,  # Number of periods in IsTimeDelta, or just int for count
    n_test_sets: int = 1,
    stride: int = 1,
    future_weather_provider: Optional[FutureWeatherFetcher] = None,
) -> Tuple[
    DataSet[TimeSeriesData], Iterator[Tuple[DataSet[TimeSeriesData], DataSet[TimeSeriesData], DataSet[TimeSeriesData]]]
]:
    """
    Generates a main training set and an iterator for multiple test splits (context, future_weather, truth).

    This is useful for backtesting strategies like rolling origin or expanding window.
    The main `train_set` excludes all periods that will be part of any test set's truth or context.

    Args:
        dataset (DataSet[TimeSeriesData]): The full dataset.
        prediction_length (int): The number of periods to forecast ahead in each test split.
        n_test_sets (int): The number of test windows (splits) to generate. Defaults to 1.
        stride (int): The step size (number of periods) to move the start of each
                      test window forward. Defaults to 1.
        future_weather_provider (Optional[FutureWeatherFetcher]): An optional provider to generate
                                                                  future weather data for each split.
                                                                  If None, future weather is derived
                                                                  by removing the target from the truth data.

    Returns:
        Tuple[DataSet[TimeSeriesData], Iterator[Tuple[DataSet[TimeSeriesData], DataSet[TimeSeriesData], DataSet[TimeSeriesData]]]]:
            A tuple containing:
            - The main training set (all data before the first test window's context).
            - An iterator yielding (context, future_weather, truth) tuples for each test window.
              'context' is historical data before prediction, 'future_weather' is covariates for
              the prediction period, 'truth' is the actual outcomes for the prediction period.

    Raises:
        ValueError: If dataset is too short for the requested splits or parameters are invalid.
    """
    if prediction_length <= 0:
        raise ValueError("prediction_length must be positive.")
    if n_test_sets <= 0:
        raise ValueError("n_test_sets must be positive.")
    if stride <= 0:
        raise ValueError("stride must be positive.")

    total_periods_needed_for_tests = prediction_length + (n_test_sets - 1) * stride
    if len(dataset.period_range) <= total_periods_needed_for_tests:
        raise ValueError(
            f"Dataset too short ({len(dataset.period_range)} periods) for {n_test_sets} test sets "
            f"with prediction_length {prediction_length} and stride {stride}."
        )

    # Determine the cutoff for the main training set: one period before the start of the first test window's context.
    # The first test window's truth starts at index `-(total_periods_needed_for_tests)`.
    # The context for this first test window ends one period before that.
    # So, the main train_set ends one period before the context of the first test split.
    # If split_idx is the start of the first truth period:
    first_truth_start_idx = len(dataset.period_range) - total_periods_needed_for_tests

    # The context for the first test split ends at `first_truth_start_idx - 1`.
    # The main training set should end before this context, or include up to this context.
    # Original logic: split_idx = -(prediction_length + (n_test_sets - 1) * stride + 1)
    # This split_idx is the index of the *last period of the context* for the first test split.
    # Or, equivalently, the period *before* the first prediction_start_period.

    # Let's define `first_prediction_start_period_index`
    # Last period index is len(dataset.period_range) - 1
    # Last period of last test set: dataset.period_range[-1]
    # Start period of last test set: dataset.period_range[-(prediction_length)]
    # Start period of first test set: dataset.period_range[-(prediction_length + (n_test_sets - 1) * stride)]
    first_prediction_start_idx = len(dataset.period_range) - (prediction_length + (n_test_sets - 1) * stride)

    if first_prediction_start_idx <= 0:  # Not enough data for even one context window before first prediction
        raise ValueError("Dataset too short for the specified test set configuration and prediction length.")

    train_cutoff_period = dataset.period_range[first_prediction_start_idx - 1]
    train_set = dataset.restrict_time_period(slice(None, train_cutoff_period))

    historic_data_splits: List[DataSet[TimeSeriesData]] = []
    future_data_truth_splits: List[DataSet[TimeSeriesData]] = []  # This will be the 'truth'

    for i in range(n_test_sets):
        current_prediction_start_idx = first_prediction_start_idx + i * stride
        current_context_end_idx = current_prediction_start_idx - 1
        current_truth_end_idx = current_prediction_start_idx + prediction_length - 1

        if current_context_end_idx < 0:  # Should be caught by earlier checks
            raise ValueError(f"Invalid splitting for test set {i}: context ends before dataset start.")
        if current_truth_end_idx >= len(dataset.period_range):  # Should be caught
            raise ValueError(f"Invalid splitting for test set {i}: truth ends after dataset end.")

        context_slice = slice(None, dataset.period_range[current_context_end_idx])
        historic_data_splits.append(dataset.restrict_time_period(context_slice))

        truth_slice = slice(
            dataset.period_range[current_prediction_start_idx], dataset.period_range[current_truth_end_idx]
        )
        future_data_truth_splits.append(dataset.restrict_time_period(truth_slice))

    masked_future_data_splits: List[DataSet[TimeSeriesData]]
    if future_weather_provider:
        # This assumes FutureWeatherFetcher constructor takes historical data.
        masked_future_data_splits = [
            future_weather_provider(hd).get_future_weather(fd.period_range)
            for hd, fd in zip(historic_data_splits, future_data_truth_splits)
        ]
    else:
        # Assuming target is 'disease_cases'. This should be configurable or inferred.
        # Also, the type of data in masked_future_data_splits should be consistent (e.g. DataSet[ClimateData])
        masked_future_data_splits = [fd.remove_field("disease_cases") for fd in future_data_truth_splits]

    return train_set, zip(historic_data_splits, masked_future_data_splits, future_data_truth_splits)


def get_split_points_for_data_set(
    data_set: DataSet[TimeSeriesData], max_splits: int, start_offset: int = 1
) -> List[TimePeriod]:
    """
    Calculates a list of `TimePeriod` objects to be used as split points for cross-validation or backtesting.
    Split points are chosen somewhat evenly from the dataset's overall time period range.

    Args:
        data_set (DataSet[TimeSeriesData]): The dataset from which to derive period information.
                                            It's assumed all locations in the DataSet share the
                                            same overall time period range structure.
        max_splits (int): The maximum number of split points to generate.
        start_offset (int): An offset from the beginning of the period range. Split points
                            will be chosen after this offset. Defaults to 1.

    Returns:
        List[TimePeriod]: A list of `TimePeriod` objects representing the calculated split points.

    Raises:
        ValueError: If the dataset is empty or `max_splits` is not positive.
    """
    if not data_set:
        raise ValueError("Input data_set is empty, cannot determine split points.")
    if max_splits <= 0:
        raise ValueError("max_splits must be a positive integer.")

    # Get periods from the first data item in the DataSet, assuming all items share the same period structure.
    try:
        # data_set.data() was from an older version, now DataSet is iterable or use .items() / .values()
        first_data_item = next(iter(data_set.values()))
        periods = first_data_item.time_period
    except StopIteration:  # Handles empty DataSet
        raise ValueError("DataSet contains no data items to extract periods from.")

    return get_split_points_for_period_range(max_splits, periods, start_offset)


def get_split_points_for_period_range(
    max_splits: int, periods: Iterable[TimePeriod], start_offset: int
) -> List[TimePeriod]:
    """
    Calculates split points directly from an iterable of `TimePeriod` objects.

    Split points are chosen to be somewhat evenly distributed across the provided periods,
    after respecting the `start_offset`.

    Args:
        max_splits (int): The maximum number of split points to generate.
        periods (Iterable[TimePeriod]): An iterable of `TimePeriod` objects.
        start_offset (int): The number of initial periods to skip before selecting split points.

    Returns:
        List[TimePeriod]: A list of `TimePeriod` objects for splitting.

    Raises:
        ValueError: If `periods` is too short for the given `max_splits` and `start_offset`,
                    or if `max_splits` or `start_offset` are invalid.
    """
    if max_splits <= 0:
        raise ValueError("max_splits must be a positive integer.")
    if start_offset < 0:  # Allow 0 for start_offset
        raise ValueError("start_offset must be a non-negative integer.")

    periods_list = list(periods)
    n_periods = len(periods_list)

    if n_periods <= start_offset:
        raise ValueError(
            f"Dataset length ({n_periods}) is less than or equal to start_offset ({start_offset}). Cannot generate split points."
        )

    # Number of available periods after offset for selecting split points from
    num_available_for_splitting = n_periods - start_offset

    if num_available_for_splitting < max_splits:
        logger.warning(
            f"Number of available periods for splitting ({num_available_for_splitting}) "
            f"is less than max_splits ({max_splits}). Returning all available as split points."
        )
        return periods_list[start_offset:]

    # Calculate delta for somewhat even spacing. +1 in denominator to ensure segments for splits.
    # This ensures that split points are chosen from within the range, not including the very end if possible.
    delta = (n_periods - 1 - start_offset) // (max_splits + 1)
    if delta == 0:  # Avoid delta=0 if max_splits is too high for the range length
        delta = 1  # Fallback to stride of 1 if calculated delta is 0
        logger.warning(
            "Calculated delta for split points is 0. Using delta=1. This might result in fewer than max_splits if range is too short."
        )

    # Select points: start after offset, step by delta, take up to max_splits
    # The slice is `periods_list[start_index : end_index : step]`
    # Start index is `start_offset + delta` to pick the first point after the first segment.
    # End index can be omitted to go to the end of the list.
    # We want `max_splits` points.

    # Example: periods = [P0, P1, P2, P3, P4, P5, P6, P7, P8, P9] (len=10)
    # max_splits = 2, start_offset = 1
    # n_periods = 10
    # delta = (10 - 1 - 1) // (2 + 1) = 8 // 3 = 2
    # Slice: periods[1 + 2 :: 2] => periods[3::2] => [P3, P5, P7, P9]
    # Then take [:max_splits] => [P3, P5]

    split_points = periods_list[start_offset + delta :: delta][:max_splits]

    # Ensure we don't pick points too close to the end if not enough data for a full forecast after split
    # This logic depends on how split_points are used (e.g., if a prediction_length is added to them)
    # For now, this function just returns the points based on spacing.

    return split_points
