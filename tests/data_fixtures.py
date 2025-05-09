# Improvement Suggestions:
# 1. **Detailed Fixture Docstrings**: Provide clear and concise docstrings for each fixture, explaining the characteristics of the dataset it generates (e.g., time range, locations, data types, specific scenarios like "good" vs. "bad" predictions). (Primary task of this refactoring).
# 2. **Data Realism/Variability**: While simple repetitive data (e.g., `[1] * T`) is easy to set up, consider if introducing minor, controlled variations or more realistic patterns in the fixture data would make tests more robust or cover more edge cases, without overcomplicating the fixtures.
# 3. **Purpose of `train_data_new_period_range`**: The docstring for `train_data_new_period_range` should clearly explain its purpose. It appears to re-wrap existing data with a potentially identical time period; clarify the scenario this fixture is designed to test. The `data.data()` call might be redundant if `data` is already the `TimeSeriesData` object.
# 4. **Define "Bad" vs. "Good" Predictions**: For `bad_predictions` and `good_predictions`, their docstrings should explicitly state what criteria make these predictions "bad" or "good" in the context of the tests that use them (e.g., deviation from expected values, specific patterns).
# 5. **Consider Fixture Parameterization**: If multiple fixtures generate datasets with only slight variations (e.g., different time ranges, number of locations, or data values), explore using `pytest.mark.parametrize` in tests or creating parameterized fixture factories to reduce code duplication and make test variations clearer.

"""
This module provides pytest fixtures that generate sample `DataSet` objects
for use in various tests within the CHAP-core test suite.

These fixtures create `DataSet` instances containing different types of time series
data (e.g., `ClimateHealthData`, `ClimateData`, `HealthData`, `FullData`)
for predefined locations ("oslo", "bergen") and time periods, facilitating
consistent and reproducible test setups.
"""

import bionumpy as bnp  # bnp.replace is used
import numpy as np  # Added numpy for potential future use if data becomes more complex
import pytest

from chap_core.datatypes import ClimateData, ClimateHealthData, FullData, HealthData
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet
from chap_core.time_period import Month, PeriodRange


@pytest.fixture()
def full_data() -> DataSet[ClimateHealthData]:
    """
    Provides a DataSet with `ClimateHealthData` for two locations ("oslo", "bergen")
    spanning a full year (January to December 2012).
    Contains simple, repetitive data for rainfall, temperature, and disease cases.
    """
    time_period = PeriodRange.from_time_periods(Month(2012, 1), Month(2012, 12))
    T = len(time_period)
    d = {
        "oslo": ClimateHealthData(
            time_period,
            rainfall=np.array([1.0] * T, dtype=float),
            mean_temperature=np.array([1.0] * T, dtype=float),
            disease_cases=np.array([20] * T, dtype=int),
        ),
        "bergen": ClimateHealthData(
            time_period,
            rainfall=np.array([100.0] * T, dtype=float),
            mean_temperature=np.array([1.0] * T, dtype=float),
            disease_cases=np.array([1] * T, dtype=int),
        ),
    }
    return DataSet(d)


@pytest.fixture()
def train_data(full_data: DataSet[ClimateHealthData]) -> DataSet[ClimateHealthData]:
    """
    Provides a subset of `full_data` intended for training, covering January to July 2012.
    Contains `ClimateHealthData` for "oslo" and "bergen".
    This fixture depends on `full_data` but redefines the data for a shorter period.
    Consider deriving from `full_data` by slicing if data consistency is critical.
    """
    time_period = PeriodRange.from_time_periods(Month(2012, 1), Month(2012, 7))
    T = len(time_period)
    # Data is redefined here; could also be sliced from full_data for consistency
    d = {
        "oslo": ClimateHealthData(
            time_period,
            rainfall=np.array([1.0] * T, dtype=float),
            mean_temperature=np.array([1.0] * T, dtype=float),
            disease_cases=np.array([20] * T, dtype=int),
        ),
        "bergen": ClimateHealthData(
            time_period,
            rainfall=np.array([100.0] * T, dtype=float),
            mean_temperature=np.array([1.0] * T, dtype=float),
            disease_cases=np.array([1] * T, dtype=int),
        ),
    }
    return DataSet(d)


@pytest.fixture()
def train_data_pop() -> DataSet[FullData]:  # Changed return type to FullData
    """
    Provides training data similar to `train_data` but using `FullData`,
    which includes a 'population' field. Covers January to July 2012.
    """
    time_period = PeriodRange.from_time_periods(Month(2012, 1), Month(2012, 7))
    T = len(time_period)
    d = {
        "oslo": FullData(
            time_period,
            rainfall=np.array([1.0] * T, dtype=float),
            mean_temperature=np.array([1.0] * T, dtype=float),
            disease_cases=np.array([20] * T, dtype=int),
            population=np.array([400000] * T, dtype=int),
        ),
        "bergen": FullData(
            time_period,
            rainfall=np.array([100.0] * T, dtype=float),
            mean_temperature=np.array([1.0] * T, dtype=float),
            disease_cases=np.array([1] * T, dtype=int),
            population=np.array([100000] * T, dtype=int),
        ),
    }
    return DataSet(d)


@pytest.fixture()
def train_data_new_period_range(train_data: DataSet[ClimateHealthData]) -> DataSet[ClimateHealthData]:
    """
    Takes the `train_data` fixture and re-wraps its contents with a new `PeriodRange`
    object that covers the same time span (January to July 2012).

    This might be used to test scenarios where data objects are reconstructed or
    their `time_period` attribute is explicitly replaced. The `data.data()` call
    is a pattern from older versions of TimeSeriesData; direct field access is now common.
    """
    # The new time_period is identical to the one in train_data.
    # This fixture's utility might be to test bnp.replace or ensure
    # that operations work correctly with newly constructed PeriodRange objects.
    new_time_period = PeriodRange.from_time_periods(Month(2012, 1), Month(2012, 7))
    return DataSet(
        {
            loc: bnp.replace(data_item, time_period=new_time_period)  # data_item is ClimateHealthData
            for loc, data_item in train_data.items()
        }
    )


@pytest.fixture()
def future_climate_data() -> DataSet[ClimateData]:
    """
    Provides a DataSet with `ClimateData` (rainfall, mean_temperature, max_temperature)
    for "oslo" and "bergen", covering August to December 2012.
    Intended to represent future climate scenarios for prediction.
    """
    time_period = PeriodRange.from_time_periods(Month(2012, 8), Month(2012, 12))
    T = len(time_period)
    d = {
        "oslo": ClimateData(
            time_period,
            rainfall=np.array([20.0] * T, dtype=float),
            mean_temperature=np.array([1.0] * T, dtype=float),
            max_temperature=np.array([1.0] * T, dtype=float),
        ),  # Added max_temperature
        "bergen": ClimateData(
            time_period,
            rainfall=np.array([1.0] * T, dtype=float),
            mean_temperature=np.array([100.0] * T, dtype=float),
            max_temperature=np.array([1.0] * T, dtype=float),
        ),  # Added max_temperature
    }
    return DataSet(d)


@pytest.fixture()
def bad_predictions() -> DataSet[HealthData]:
    """
    Provides a DataSet with `HealthData` representing "bad" predictions for a single month (August 2012).
    "Bad" typically means these values are far from an expected or true value in a test scenario.
    - Oslo: 2 cases
    - Bergen: 19 cases
    """
    time_period = PeriodRange.from_time_periods(Month(2012, 8), Month(2012, 8))
    T = len(time_period)  # T will be 1
    d = {
        "oslo": HealthData(time_period, disease_cases=np.array([2] * T, dtype=int)),
        "bergen": HealthData(time_period, disease_cases=np.array([19] * T, dtype=int)),
    }
    return DataSet(d)


@pytest.fixture()
def good_predictions() -> DataSet[HealthData]:
    """
    Provides a DataSet with `HealthData` representing "good" predictions for a single month (August 2012).
    "Good" typically means these values are close to an expected or true value in a test scenario.
    - Oslo: 19 cases
    - Bergen: 2 cases
    """
    time_period = PeriodRange.from_time_periods(Month(2012, 8), Month(2012, 8))
    T = len(time_period)  # T will be 1
    d = {
        "oslo": HealthData(time_period, disease_cases=np.array([19] * T, dtype=int)),
        "bergen": HealthData(time_period, disease_cases=np.array([2] * T, dtype=int)),
    }
    return DataSet(d)
