# Improvement Suggestions:
# 1. **Docstring Coverage**: Ensure all classes, methods, and functions have comprehensive docstrings explaining their purpose, arguments, and return values. (Primary goal of this refactoring).
# 2. **Consistency in `to_pandas` vs `topandas`**: Standardize method naming. For example, `TimeSeriesData` has both `topandas` and an alias `to_pandas`. Prefer `to_pandas` for consistency with the pandas library.
# 3. **Error Handling in `from_pandas` / `from_csv`**: Refine error handling in `from_pandas`. The current broad `except Exception:` could be made more specific (e.g., catch `KeyError`, `ValueError`) to provide better diagnostics.
# 4. **`BNPDataClass` Interaction**: Clarify the implications of using `bionumpy.bnpdataclass` (e.g., data storage as numpy arrays, performance aspects) in the relevant class docstrings.
# 5. **`ClimateHealthTimeSeriesModel` (Pydantic)**: Document the specific role of this Pydantic model in relation to the `tsdataclass` versions. Is it for API validation, serialization, or another distinct purpose?

"""
This module defines core data types for time series data used within the CHAP-core application.

It leverages `dataclasses` and `bionumpy` (via a custom `tsdataclass` decorator)
to create efficient, type-annotated time series data structures. Key features include:
- A base `TimeSeriesData` class with common time series operations (pandas conversion, CSV I/O, interpolation).
- Specialized dataclasses for health data, climate data, and combined types.
- Utility classes for statistical summaries (`SummaryStatistics`, `Samples`).
- Helper functions for manipulating these dataclass fields.
"""

import dataclasses
from typing import Any, Dict, List, Optional, Type  # Added Any, Type, Dict

import bionumpy as bnp
import numpy as np
import pandas as pd
from bionumpy.bnpdataclass import BNPDataClass
from pydantic import BaseModel, field_validator
from typing_extensions import deprecated

from .api_types import PeriodObservation
from .time_period import PeriodRange
from .time_period.date_util_wrapper import TimeStamp
from .util import interpolate_nans


# --- Decorator to wrap dataclasses with BioNumPy and enforce PeriodRange typing ---
def tsdataclass(cls: Type[Any]) -> Type[BNPDataClass]:
    """
    A decorator that enhances a dataclass for time series data.

    It applies `bionumpy.bnpdataclass.bnpdataclass` for efficient array-based storage
    and operations, and enforces that the `time_period` attribute is of type `PeriodRange`.

    Args:
        cls (Type[Any]): The dataclass to be wrapped.

    Returns:
        Type[BNPDataClass]: The enhanced dataclass.
    """
    tmp_cls = bnp.bnpdataclass.bnpdataclass(cls)
    # Ensure time_period annotation is set to PeriodRange
    if hasattr(tmp_cls, "__annotations__"):
        tmp_cls.__annotations__["time_period"] = PeriodRange
    else:
        # This case should ideally not happen for a dataclass being decorated
        setattr(tmp_cls, "__annotations__", {"time_period": PeriodRange})
    return tmp_cls


# --- Base TimeSeries Data Class ---
@tsdataclass
class TimeSeriesData:
    """
    Base class for time series data, built upon `bionumpy.bnpdataclass`.

    Attributes:
        time_period (PeriodRange): The time periods corresponding to the data.
    """

    time_period: PeriodRange

    def model_dump(self) -> Dict[str, Any]:
        """
        Dump fields as lists, primarily for Pydantic model interoperability.
        Converts numpy arrays to lists.

        Returns:
            Dict[str, Any]: A dictionary representation of the instance data.
        """
        return {
            field.name: getattr(self, field.name).tolist()
            if isinstance(getattr(self, field.name), np.ndarray)
            else getattr(self, field.name)
            for field in dataclasses.fields(self)
        }

    def __getstate__(self) -> Dict[str, Any]:
        """
        Custom state for pickling, converting to a dictionary.
        Uses `todict` which handles `PeriodRange` specifically.
        """
        return self.todict()

    def __setstate__(self, state: Dict[str, Any]):
        """
        Custom state restoration for unpickling.
        Relies on `from_dict` or direct attribute setting if state is simple.
        Note: This simple __dict__.update might not correctly reconstruct PeriodRange
        if it was serialized by a simple todict(). `from_dict` is safer if used by pickler.
        For bnpdataclass, default pickling might be sufficient or require specific handling.
        The current `__getstate__` uses `self.todict()`, so `from_dict` logic is more aligned.
        However, standard pickling calls __setstate__ with the result of __getstate__.
        Let's assume `todict` and `from_dict` are for custom serialization paths,
        and bnp's default pickling handles its internal state.
        If custom pickling is strictly needed via these, `__setstate__` should mirror `from_dict`.
        For now, keeping it simple as bnp might handle it.
        """
        # If state comes from `self.todict()`, then:
        # new_instance = self.__class__.from_dict(state)
        # self.__dict__.update(new_instance.__dict__)
        # However, bnpdataclass might have its own optimized pickling.
        # This simple update is often used if __getstate__ returns __dict__.
        self.__dict__.update(state)

    def join(self, other: "TimeSeriesData") -> "TimeSeriesData":
        """
        Concatenates this TimeSeriesData instance with another.
        Assumes underlying bnpdataclass concatenation behavior.

        Args:
            other (TimeSeriesData): The other TimeSeriesData instance to join.

        Returns:
            TimeSeriesData: A new instance containing concatenated data.

        Raises:
            TypeError: If `other` is not of a compatible type for concatenation.
        """
        # bnpdataclass supports concatenation using np.concatenate
        return np.concatenate([self, other])

    def resample(self, freq: str) -> "TimeSeriesData":
        """
        Resamples the TimeSeriesData to a new frequency, interpolating values.

        Args:
            freq (str): The target frequency string (e.g., 'M' for monthly, 'D' for daily).

        Returns:
            TimeSeriesData: A new instance with resampled and interpolated data.
        """
        df = self.topandas()
        df["time_period"] = self.time_period.to_period_index()
        df = df.set_index("time_period").resample(freq).interpolate()  # Default linear interpolation
        return self.from_pandas(df.reset_index())

    def topandas(self) -> pd.DataFrame:
        """
        Converts the TimeSeriesData instance to a pandas DataFrame.

        Returns:
            pd.DataFrame: A DataFrame representation of the data.
        """
        data_dict = {field.name: getattr(self, field.name) for field in dataclasses.fields(self)}
        # Handle multi-dimensional arrays by converting them to lists of lists if necessary
        for key, value in data_dict.items():
            if isinstance(value, np.ndarray) and value.ndim > 1:
                data_dict[key] = value.tolist()  # Or handle more gracefully
        data_dict["time_period"] = self.time_period.topandas()  # Assumes PeriodRange has topandas
        return pd.DataFrame(data_dict)

    to_pandas = topandas  # Alias for consistency

    def to_csv(self, csv_file: str, **kwargs):
        """
        Saves the TimeSeriesData instance to a CSV file.

        Args:
            csv_file (str): The path to the output CSV file.
            **kwargs: Additional keyword arguments passed to `pd.DataFrame.to_csv()`.
        """
        self.to_pandas().to_csv(csv_file, index=False, **kwargs)

    def to_pickle_dict(self) -> Dict[str, Any]:
        """
        Serializes the TimeSeriesData to a dictionary suitable for pickling or JSON.
        `time_period` is converted to a list of strings.

        Returns:
            Dict[str, Any]: A serializable dictionary representation.
        """
        data_dict = {field.name: getattr(self, field.name) for field in dataclasses.fields(self)}
        data_dict["time_period"] = self.time_period.tolist()  # Assumes PeriodRange.tolist() gives strings
        return data_dict

    @classmethod
    def from_pickle_dict(cls: Type["TST"], data: Dict[str, Any]) -> "TST":  # TST as TypeVar for cls
        """
        Deserializes a TimeSeriesData instance from a dictionary (e.g., from `to_pickle_dict`).

        Args:
            cls (Type["TST"]): The TimeSeriesData subclass to instantiate.
            data (Dict[str, Any]): The dictionary containing the data.

        Returns:
            TST: An instance of the TimeSeriesData subclass.
        """
        # TypeVar TST = TypeVar('TST', bound=TimeSeriesData)
        constructor_args = {}
        for key, value in data.items():
            if key == "time_period":
                constructor_args[key] = PeriodRange.from_strings(value)
            else:
                # Ensure other fields are converted to numpy arrays if they were lists from JSON
                # This depends on how bnpdataclass handles initialization
                constructor_args[key] = np.asanyarray(value) if isinstance(value, list) else value
        return cls(**constructor_args)

    @classmethod
    def create_class_from_basemodel(cls: Type["TST"], pydantic_model_class: Type[PeriodObservation]) -> Type["TST"]:
        """
        Dynamically creates a TimeSeriesData subclass from a Pydantic BaseModel.

        The created class will have fields corresponding to the Pydantic model,
        with `time_period` typed as `PeriodRange`.

        Args:
            cls (Type["TST"]): The base class (typically TimeSeriesData or a derivative).
            pydantic_model_class (Type[PeriodObservation]): The Pydantic model to base the new class on.

        Returns:
            Type["TST"]: A new dataclass derived from `cls`.
        """
        fields = pydantic_model_class.model_fields
        # Ensure field annotations are actual types, not ForwardRefs, if any
        field_definitions = []
        for name, field_info in fields.items():
            annotation = field_info.annotation
            if name == "time_period":
                annotation = PeriodRange
            field_definitions.append((name, annotation))

        return dataclasses.make_dataclass(
            pydantic_model_class.__name__ + "TSData",  # Ensure unique name
            field_definitions,
            bases=(cls,),  # Use the calling class (e.g. TimeSeriesData) as base
        )

    @staticmethod
    def _fill_missing(data: np.ndarray, missing_indices: List[int]) -> np.ndarray:
        """
        Fills missing entries in a data array with NaNs based on indices.

        Args:
            data (np.ndarray): The original data array (without missing entries).
            missing_indices (List[int]): Indices where NaNs should be inserted in the final array.

        Returns:
            np.ndarray: A new array with NaNs inserted at `missing_indices`.
        """
        if not missing_indices:  # Optimized for common case
            return data

        n_entries = len(data) + len(missing_indices)
        # Determine dtype, prefer float if NaNs are to be inserted, or original dtype if no NaNs.
        # If data is int and missing_indices is not empty, it must become float.
        dtype = np.result_type(data.dtype, float) if missing_indices else data.dtype
        filled_data = np.full(n_entries, np.nan, dtype=dtype)  # Use np.nan for float, handle others if needed

        mask = np.full(n_entries, True, dtype=bool)
        mask[missing_indices] = False
        filled_data[mask] = data
        return filled_data

    @classmethod
    def from_pandas(cls: Type["TST"], data: pd.DataFrame, fill_missing: bool = False) -> "TST":
        """
        Creates a TimeSeriesData instance from a pandas DataFrame.

        Args:
            cls (Type["TST"]): The TimeSeriesData subclass to instantiate.
            data (pd.DataFrame): The input DataFrame. Must contain a 'time_period' column
                                 and columns matching the fields of `cls`.
            fill_missing (bool): If True, attempt to fill missing time periods with NaNs.

        Returns:
            TST: An instance of the TimeSeriesData subclass.

        Raises:
            AssertionError: If duplicate time_periods are detected.
            ValueError: If 'time_period' column is missing or other data issues occur.
        """
        if "time_period" not in data.columns:
            raise ValueError("DataFrame must contain a 'time_period' column.")

        try:
            time_strings = data.time_period.astype(str)
            if not time_strings.is_unique:  # More pandas idiomatic check
                # Find duplicates for better error message
                duplicates = time_strings[time_strings.duplicated()].unique()
                raise AssertionError(f"Duplicate time_periods detected: {duplicates.tolist()}")

            # PeriodRange.from_strings handles fill_missing internally now
            time_period_result = PeriodRange.from_strings(time_strings, fill_missing=fill_missing)
        except Exception as e:
            # logger.error(f"Error parsing time period from DataFrame: {data.time_period}", exc_info=True)
            print(f"Error parsing time period from DataFrame: {data.time_period}")  # Basic print for now
            raise ValueError(f"Failed to parse 'time_period' column: {e}")

        actual_time_periods: PeriodRange
        missing_indices: List[int] = []
        if fill_missing:
            actual_time_periods, missing_indices = time_period_result  # Unpack if fill_missing was True
        else:
            actual_time_periods = time_period_result

        constructor_args = {"time_period": actual_time_periods}
        variable_names = [field.name for field in dataclasses.fields(cls) if field.name != "time_period"]

        for name in variable_names:
            if name not in data.columns:
                raise ValueError(f"DataFrame is missing required column for field: {name}")
            col_data = data[name].values
            constructor_args[name] = cls._fill_missing(col_data, missing_indices)

        return cls(**constructor_args)

    @classmethod
    def from_csv(cls: Type["TST"], csv_file: str, **kwargs) -> "TST":
        """
        Creates a TimeSeriesData instance from a CSV file.

        Args:
            cls (Type["TST"]): The TimeSeriesData subclass to instantiate.
            csv_file (str): Path to the CSV file.
            **kwargs: Additional keyword arguments passed to `pd.read_csv()`.

        Returns:
            TST: An instance of the TimeSeriesData subclass.
        """
        return cls.from_pandas(pd.read_csv(csv_file, **kwargs))

    def interpolate(self, field_names: Optional[List[str]] = None) -> "TimeSeriesData":
        """
        Interpolates NaN values in specified fields (or all fields if None).

        Args:
            field_names (Optional[List[str]]): A list of field names to interpolate.
                                               If None, all non-'time_period' fields are interpolated.

        Returns:
            TimeSeriesData: A new instance with specified fields interpolated.
        """
        data_dict = {field.name: getattr(self, field.name) for field in dataclasses.fields(self)}
        interpolated_fields = {}
        for key, value in data_dict.items():
            if key == "time_period":
                continue
            if field_names is None or key in field_names:
                if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.number):
                    interpolated_fields[key] = interpolate_nans(value)
                else:
                    interpolated_fields[key] = value  # Cannot interpolate non-numeric or non-array
            else:
                interpolated_fields[key] = value

        return self.__class__(self.time_period, **interpolated_fields)

    @deprecated("Compatibility with old code. Access fields directly or use specific conversion methods.")
    def data(self) -> "TimeSeriesData":
        """
        Returns self. Deprecated method for compatibility.
        """
        return self

    @property
    def start_timestamp(self) -> pd.Timestamp:
        """
        Returns the start timestamp of the first period in the time_period range.
        """
        if not self.time_period:
            raise ValueError("time_period is empty, cannot get start_timestamp.")
        return self.time_period[0].start_timestamp

    @property
    def end_timestamp(self) -> pd.Timestamp:
        """
        Returns the end timestamp of the last period in the time_period range.
        """
        if not self.time_period:
            raise ValueError("time_period is empty, cannot get end_timestamp.")
        return self.time_period[-1].end_timestamp

    def fill_to_endpoint(self, end_time_stamp: TimeStamp) -> "TimeSeriesData":
        """
        Extends or truncates the time series to a new end timestamp, filling with NaNs if extending.

        Args:
            end_time_stamp (TimeStamp): The target end timestamp.

        Returns:
            TimeSeriesData: A new instance adjusted to the specified end timestamp.

        Raises:
            AssertionError: If `n_missing` is negative (i.e. `end_time_stamp` is before current end).
                            This behavior might need review for truncation.
        """
        if self.end_timestamp == end_time_stamp:
            return self

        # This calculates periods from current end to new end.
        # If end_time_stamp is earlier, n_missing will be negative.
        n_missing = self.time_period.delta.n_periods_between(self.end_timestamp, end_time_stamp)

        if n_missing == 0 and self.end_timestamp < end_time_stamp:  # Same period, but timestamp is later
            n_missing = 1  # Add one full period if target is within next period after current end

        if n_missing < 0:
            # Truncation logic: find index for new end and slice
            # This part is more complex than just padding.
            # For now, let's assume fill_to_endpoint is for extension or same end.
            # Or, if truncation is desired, it needs a different approach.
            # The original `assert n_missing >= 0` implies only extension.
            # Let's stick to extension or no-op.
            if end_time_stamp < self.end_timestamp:
                raise ValueError(
                    "end_time_stamp for fill_to_endpoint cannot be before current end_timestamp for extension logic."
                )
            return self  # No change if new end is within the last period or earlier

        new_time = PeriodRange(self.time_period.start_timestamp, end_time_stamp, self.time_period.delta)
        padded_data = {}
        for name in self.__dataclass_fields__:  # __dataclass_fields__ is standard for dataclasses
            if name == "time_period":
                continue
            original_field_data = getattr(self, name)
            # Ensure data is float for NaN padding, or handle dtype appropriately
            padded_data[name] = np.pad(
                original_field_data.astype(float), (0, n_missing), mode="constant", constant_values=np.nan
            )
        return self.__class__(new_time, **padded_data)

    def fill_to_range(self, start_timestamp: TimeStamp, end_timestamp: TimeStamp) -> "TimeSeriesData":
        """
        Adjusts the time series to a new start and end timestamp, padding with NaNs.

        Args:
            start_timestamp (TimeStamp): The target start timestamp.
            end_timestamp (TimeStamp): The target end timestamp.

        Returns:
            TimeSeriesData: A new instance adjusted to the specified range.

        Raises:
            AssertionError: If calculated padding amounts (`n_start`, `n_end`) are negative.
        """
        if self.start_timestamp == start_timestamp and self.end_timestamp == end_timestamp:
            return self

        # n_periods calculates count of delta periods.
        # If new start is after current start, n_start will be negative (truncation from start).
        # If new end is before current end, n_end will be negative (truncation from end).
        n_start = self.time_period.delta.n_periods_between(start_timestamp, self.start_timestamp)
        n_end = self.time_period.delta.n_periods_between(self.end_timestamp, end_timestamp)

        # This method, like fill_to_endpoint, seems designed for padding, not truncation.
        # The asserts `n_start >= 0 and n_end >= 0` enforce this.
        if n_start < 0 or n_end < 0:
            raise ValueError(
                "fill_to_range with current logic only supports expanding the range or matching it, not truncation."
            )

        new_time = PeriodRange(start_timestamp, end_timestamp, self.time_period.delta)
        padded_data = {}
        for name in self.__dataclass_fields__:
            if name == "time_period":
                continue
            original_field_data = getattr(self, name)
            padded_data[name] = np.pad(
                original_field_data.astype(float), (n_start, n_end), mode="constant", constant_values=np.nan
            )
        return self.__class__(new_time, **padded_data)

    def to_array(self) -> np.ndarray:
        """
        Converts data fields (excluding time_period) into a 2D numpy array.
        Each field becomes a column.

        Returns:
            np.ndarray: A 2D array where rows are time points and columns are variables.
        """
        return np.array(
            [getattr(self, field.name) for field in dataclasses.fields(self) if field.name != "time_period"]
        ).T  # Transpose to make fields as columns

    def todict(self) -> Dict[str, Any]:
        """
        Converts the TimeSeriesData instance to a dictionary.
        `time_period` is converted using its `topandas()` method (likely to pd.Series of periods).

        Returns:
            Dict[str, Any]: A dictionary representation.
        """
        d = {field.name: getattr(self, field.name) for field in dataclasses.fields(self)}
        if hasattr(self.time_period, "topandas"):
            d["time_period"] = self.time_period.topandas()
        else:  # Fallback if PeriodRange doesn't have topandas (it should)
            d["time_period"] = self.time_period.tolist()
        return d

    @classmethod
    def from_dict(cls: Type["TST"], data: Dict[str, Any]) -> "TST":
        """
        Creates a TimeSeriesData instance from a dictionary.
        Assumes 'time_period' value can be processed by `PeriodRange.from_strings`.

        Args:
            cls (Type["TST"]): The TimeSeriesData subclass to instantiate.
            data (Dict[str, Any]): The dictionary containing the data.

        Returns:
            TST: An instance of the TimeSeriesData subclass.
        """
        constructor_args = {}
        for key, value in data.items():
            if key == "time_period":
                # If value from todict() is already a PeriodIndex or similar, from_strings might not be right.
                # This depends on the exact format from todict()'s time_period serialization.
                # Assuming it's a list of strings or compatible with from_strings.
                if isinstance(value, (pd.Series, list, np.ndarray)):  # Common outputs of topandas/tolist
                    constructor_args[key] = PeriodRange.from_strings(value)
                elif isinstance(value, PeriodRange):  # If it's already a PeriodRange
                    constructor_args[key] = value
                else:
                    raise TypeError(f"Unsupported type for 'time_period' in from_dict: {type(value)}")
            else:
                constructor_args[key] = np.asanyarray(value) if isinstance(value, list) else value
        return cls(**constructor_args)

    def merge(self, other: "TimeSeriesData", result_class: Type["TimeSeriesData"]) -> "TimeSeriesData":
        """
        Merges data from this instance with another, producing an instance of `result_class`.
        Requires time periods to be identical. Fields are taken from `self` or `other`.

        Args:
            other (TimeSeriesData): The other TimeSeriesData instance to merge with.
            result_class (Type[TimeSeriesData]): The dataclass type for the merged result.

        Returns:
            TimeSeriesData: An instance of `result_class` containing merged data.

        Raises:
            ValueError: If time periods do not match, or if there's a field conflict
                        (field exists in both and `result_class` expects one) or a field
                        required by `result_class` is missing from both.
        """
        if not np.array_equal(self.time_period, other.time_period):  # Relies on PeriodRange __eq__
            raise ValueError("Time periods do not match for merge operation.")

        merged_data_dict = {}
        for field in dataclasses.fields(result_class):
            name = field.name
            if name == "time_period":
                continue

            has_in_self = hasattr(self, name)
            has_in_other = hasattr(other, name)

            if has_in_self and hasattr(result_class, name):
                merged_data_dict[name] = getattr(self, name)
            elif has_in_other and hasattr(result_class, name):
                merged_data_dict[name] = getattr(other, name)
            elif hasattr(result_class, name):  # Field required by result_class but not in self or other
                raise ValueError(f"Field '{name}' required by result_class but not found in source objects.")
            # If field is not in result_class, it's ignored.
            # This logic assumes fields in result_class are a subset of union of fields in self and other.
            # A more robust merge might check for conflicts if field exists in both self and other.
            # Current logic prioritizes `self`. If `result_class` has fields not in `self` but in `other`, they are used.

        return result_class(self.time_period, **merged_data_dict)


# --- Core Typed TimeSeries Classes ---
@tsdataclass
class TimeSeriesArray(TimeSeriesData):
    """A simple time series with a single float 'value' field."""

    value: float


@tsdataclass
class SimpleClimateData(TimeSeriesData):
    """Represents basic climate data with rainfall and mean temperature."""

    rainfall: float
    mean_temperature: float


@tsdataclass
class ClimateData(TimeSeriesData):
    """Extends SimpleClimateData with maximum temperature."""

    rainfall: float
    mean_temperature: float
    max_temperature: float  # Often different from mean_temperature


@tsdataclass
class HealthData(TimeSeriesData):
    """Represents health data, typically disease case counts."""

    disease_cases: int


@tsdataclass
class ClimateHealthTimeSeries(TimeSeriesData):
    """Combines climate (rainfall, mean temperature) and health (disease cases) data."""

    rainfall: float
    mean_temperature: float
    disease_cases: int

    @classmethod
    def combine(
        cls,
        health_data: HealthData,
        climate_data: ClimateData,  # fill_missing was unused
    ) -> "ClimateHealthTimeSeries":
        """
        Combines HealthData and ClimateData into a ClimateHealthTimeSeries instance.
        Assumes `health_data` and `climate_data` have aligned `time_period` attributes.

        Args:
            health_data (HealthData): The health data component.
            climate_data (ClimateData): The climate data component.

        Returns:
            ClimateHealthTimeSeries: The combined time series data.

        Raises:
            ValueError: If time_periods of input data do not match.
        """
        if not np.array_equal(health_data.time_period, climate_data.time_period):
            raise ValueError("Cannot combine: HealthData and ClimateData time_periods do not match.")

        return cls(
            time_period=health_data.time_period,  # or climate_data.time_period
            rainfall=climate_data.rainfall,
            mean_temperature=climate_data.mean_temperature,
            disease_cases=health_data.disease_cases,
        )


ClimateHealthData = ClimateHealthTimeSeries  # Alias


@tsdataclass
class FullData(ClimateHealthTimeSeries):
    """Extends ClimateHealthTimeSeries with population data."""

    population: int

    @classmethod
    def combine(cls, health_data: HealthData, climate_data: ClimateData, population: float) -> "FullData":
        """
        Combines health, climate, and a scalar population value into FullData.
        The population value is broadcast across all time periods.
        Assumes `health_data` and `climate_data` have aligned `time_period`.

        Args:
            health_data (HealthData): Health data component.
            climate_data (ClimateData): Climate data component.
            population (float): A scalar population value to be applied to all time periods.

        Returns:
            FullData: The combined time series data including population.

        Raises:
            ValueError: If time_periods of input data do not match.
        """
        if not np.array_equal(health_data.time_period, climate_data.time_period):
            raise ValueError("Cannot combine: HealthData and ClimateData time_periods do not match.")

        return cls(
            time_period=health_data.time_period,
            rainfall=climate_data.rainfall,
            mean_temperature=climate_data.mean_temperature,
            disease_cases=health_data.disease_cases,
            population=np.full(len(health_data.time_period), population, dtype=int),  # Ensure population is array
        )


@tsdataclass
class LocatedClimateHealthTimeSeries(ClimateHealthTimeSeries):
    """ClimateHealthTimeSeries data associated with a specific location identifier."""

    location: str  # Typically a string ID, could be other types if bnp handles


class ClimateHealthTimeSeriesModel(BaseModel):
    """
    Pydantic model for validating or serializing single entries of climate-health time series data.
    Likely used for API interactions or configuration where Pydantic's validation is beneficial.
    """

    time_period: str | pd.Period
    rainfall: float
    mean_temperature: float
    disease_cases: int

    class Config:
        """Pydantic configuration settings."""

        arbitrary_types_allowed = True  # Allows pd.Period

    @field_validator("time_period")
    def parse_time_period(cls, data: str | pd.Period) -> pd.Period:
        """
        Validates and parses the time_period field to a pandas Period object.
        """
        return data if isinstance(data, pd.Period) else pd.Period(data)


@tsdataclass
class HealthPopulationData(HealthData):
    """Combines health data (disease cases) with population data."""

    population: int


@dataclasses.dataclass
class Location:
    """Represents a geographical location with latitude and longitude."""

    latitude: float
    longitude: float


# --- Statistical Time Series ---
@tsdataclass
class SummaryStatistics(TimeSeriesData):
    """Stores common summary statistics for a time series."""

    mean: float
    median: float
    std: float  # Standard deviation
    min: float
    max: float
    quantile_low: float  # Lower quantile value (e.g., 25th percentile)
    quantile_high: float  # Upper quantile value (e.g., 75th percentile)


@tsdataclass
class Samples(TimeSeriesData):
    """
    Represents multiple forecast samples (realizations) over a time period.
    The 'samples' attribute is typically a 2D array (time, n_samples).
    """

    samples: float  # BNPDataClass will handle this as np.ndarray based on input

    def topandas(self) -> pd.DataFrame:
        """
        Converts Samples data to a pandas DataFrame.
        Each sample trace becomes a column named 'sample_i'.

        Returns:
            pd.DataFrame: DataFrame with 'time_period' and 'sample_0', 'sample_1', ... columns.
        """
        if self.samples.ndim != 2:
            raise ValueError(f"Samples.samples attribute expected to be 2D (time, n_samples), got {self.samples.ndim}D")
        n_samples = self.samples.shape[-1]
        df_dict = {"time_period": self.time_period.topandas()}
        for i in range(n_samples):
            df_dict[f"sample_{i}"] = self.samples[:, i]
        return pd.DataFrame(df_dict)

    @classmethod
    def from_pandas(cls: Type["Samples"], data: pd.DataFrame, fill_missing: bool = False) -> "Samples":
        """
        Creates a Samples instance from a pandas DataFrame.
        Expects 'time_period' column and 'sample_i' columns.

        Args:
            cls (Type["Samples"]): The Samples class.
            data (pd.DataFrame): Input DataFrame.
            fill_missing (bool): If True, fill missing time periods.

        Returns:
            Samples: An instance of Samples.

        Raises:
            ValueError: If 'sample_i' columns are missing or samples contain non-finite values.
        """
        time_period_result = PeriodRange.from_strings(data.time_period.astype(str), fill_missing=fill_missing)

        actual_time_periods: PeriodRange
        missing_indices: List[int] = []
        if fill_missing and isinstance(time_period_result, tuple):  # from_strings returns tuple if fill_missing
            actual_time_periods, missing_indices = time_period_result
        else:
            actual_time_periods = time_period_result

        sample_cols = [col for col in data.columns if col.startswith("sample_")]
        if not sample_cols:
            raise ValueError("No 'sample_i' columns found in DataFrame for Samples.from_pandas.")

        # Sort sample_cols to ensure consistent order if numbers are not zero-padded (e.g. sample_1, sample_10)
        sample_cols.sort(key=lambda x: int(x.split("_")[1]))

        # Stack samples into a 2D array (n_time_periods, n_samples)
        raw_samples_list = [cls._fill_missing(data[col].values, missing_indices) for col in sample_cols]
        samples_array = np.array(raw_samples_list, dtype=float).T  # Transpose to (time, samples)

        if not np.isfinite(samples_array).all():
            # Consider logging which samples/periods have issues
            raise ValueError("Samples data from DataFrame contains non-finite values after processing.")
        return cls(actual_time_periods, samples_array)

    to_pandas = topandas  # Alias

    def summaries(self, q_low: float = 0.25, q_high: float = 0.75) -> SummaryStatistics:
        """
        Calculates summary statistics (mean, median, std, min, max, quantiles)
        across the samples for each time period.

        Args:
            q_low (float): Lower quantile for `quantile_low` (default: 0.25).
            q_high (float): Upper quantile for `quantile_high` (default: 0.75).

        Returns:
            SummaryStatistics: A TimeSeriesData object containing the calculated summaries.
        """
        if self.samples.ndim != 2 or self.samples.shape[0] != len(self.time_period):
            raise ValueError("Samples array shape is inconsistent with time_period or not 2D.")

        return SummaryStatistics(
            self.time_period,
            mean=np.mean(self.samples, axis=-1),
            median=np.median(self.samples, axis=-1),
            std=np.std(self.samples, axis=-1),
            min=np.min(self.samples, axis=-1),
            max=np.max(self.samples, axis=-1),
            quantile_low=np.quantile(self.samples, q_low, axis=-1),
            quantile_high=np.quantile(self.samples, q_high, axis=-1),
        )


@tsdataclass
class SamplesWithTruth(Samples):
    """Extends Samples to include actual observed 'disease_cases' alongside forecast samples."""

    disease_cases: float  # Actual observed values corresponding to the samples' time periods


@dataclasses.dataclass
class Quantile:
    """Represents a quantile range with low, high, and size (high - low)."""

    low: float
    high: float
    size: float  # Typically high - low, representing the width of the quantile interval


# --- Field Manipulation Helpers ---
def add_field(data: BNPDataClass, new_class: Type[BNPDataClass], **field_data: Any) -> BNPDataClass:
    """
    Adds new fields to an existing BNPDataClass instance, returning a new instance of `new_class`.

    Args:
        data (BNPDataClass): The original data instance.
        new_class (Type[BNPDataClass]): The target class type for the new instance.
                                        It should include original fields plus new ones.
        **field_data (Any): Keyword arguments for the new fields and their values.

    Returns:
        BNPDataClass: An instance of `new_class` with added fields.
    """
    existing_fields = {f.name: getattr(data, f.name) for f in dataclasses.fields(data)}
    return new_class(**existing_fields, **field_data)


def remove_field(data: BNPDataClass, field_name: str, new_class: Optional[Type[BNPDataClass]] = None) -> BNPDataClass:
    """
    Removes a field from a BNPDataClass instance, returning a new instance.

    If `new_class` is not provided, a new class type is dynamically created
    without the specified field.

    Args:
        data (BNPDataClass): The original data instance.
        field_name (str): The name of the field to remove.
        new_class (Optional[Type[BNPDataClass]]): The target class type for the new instance.
                                                  If None, a new class is created dynamically.

    Returns:
        BNPDataClass: An instance of `new_class` (or the dynamically created class)
                      without the specified field.
    """
    if new_class is None:
        # Dynamically create a new dataclass type without the specified field
        original_cls_name = data.__class__.__name__
        # Ensure new class name is unique if this function is called multiple times for same original class
        # This simple naming might not be robust for all scenarios of dynamic class creation.
        new_cls_name = f"{original_cls_name}_Without_{field_name.capitalize()}"

        new_fields = [(f.name, f.type) for f in dataclasses.fields(data) if f.name != field_name]
        # Ensure TimeSeriesData is a base if the original class inherited from it.
        bases = (TimeSeriesData,) if isinstance(data, TimeSeriesData) else (BNPDataClass,)  # Or more specific base

        new_class = tsdataclass(  # Apply tsdataclass decorator if it was a TimeSeriesData derivative
            dataclasses.make_dataclass(
                new_cls_name,
                new_fields,
                bases=bases,
            )
        )

    # Populate the new class instance
    constructor_args = {f.name: getattr(data, f.name) for f in dataclasses.fields(data) if f.name != field_name}
    return new_class(**constructor_args)


# --- Earth Observation Derived Time Series ---
@tsdataclass
class GEEData(TimeSeriesData):
    """Represents data typically derived from Google Earth Engine (GEE), like temperature and precipitation."""

    temperature_2m: float  # Temperature at 2 meters
    total_precipitation_sum: float  # Sum of total precipitation


@tsdataclass
class FullGEEData(HealthPopulationData):
    """Combines HealthPopulationData with GEE-derived climate variables."""

    temperature_2m: float
    total_precipitation_sum: float


# --- Factory ---
def create_tsdataclass(field_names: List[str]) -> Type[TimeSeriesData]:
    """
    Dynamically creates a new TimeSeriesData subclass with specified float fields.

    Args:
        field_names (List[str]): A list of names for the fields to be created.
                                 All fields will be typed as `float`.

    Returns:
        Type[TimeSeriesData]: A new dataclass derived from TimeSeriesData.
    """
    # Generate a class name based on fields, or use a generic one if too complex
    class_name_suffix = "_".join(field_names)
    dynamic_class_name = f"DynamicTimeSeriesData_{class_name_suffix}"
    if len(dynamic_class_name) > 50:  # Keep class name reasonable
        dynamic_class_name = "CustomTimeSeriesData"

    return tsdataclass(
        dataclasses.make_dataclass(dynamic_class_name, [(name, float) for name in field_names], bases=(TimeSeriesData,))
    )


# For TypeVar TST in from_pickle_dict, from_pandas, from_csv
from typing import TypeVar

TST = TypeVar("TST", bound=TimeSeriesData)

# Add logger if it's used (e.g. in from_pandas error logging)
import logging

logger = logging.getLogger(__name__)
