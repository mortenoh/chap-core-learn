# Improvement Suggestions:
# 1. **Comprehensive Docstrings**: Add a module-level docstring, and detailed docstrings for the `DataElement` Pydantic model, the `SeasonalForecast` class, and all its methods (`__init__`, `add_json`, `get_forecasts`). Clearly explain the structure of `data_dict` within `SeasonalForecast`. (Primary task).
# 2. **Refined Type Hinting**:
#    - Use `typing.Dict` for `data_dict` in `SeasonalForecast.__init__` for consistency and pre-Python 3.9 compatibility (i.e., `Dict[str, Dict[str, Dict[str, float]]]`).
#    - The `json_data` parameter in `add_json` should be typed as `List[Dict[str, Any]]` if it receives raw dictionaries from JSON parsing, before Pydantic validation.
#    - Type hint `period_range` in `get_forecasts` (e.g., as `chap_core.time_period.PeriodRange`).
# 3. **Robust Error Handling**:
#    - In `add_json`, wrap `DataElement(**data)` in a `try-except pydantic.ValidationError` to handle cases where input data doesn't conform to the `DataElement` model, logging or raising appropriately.
#    - In `get_forecasts`, replace `assert` statements with `KeyError` or `ValueError` for missing fields, orgUnits, or periods, providing more context in error messages.
# 4. **Unused `start_date` Parameter**: The `start_date` parameter in `SeasonalForecast.get_forecasts` is currently unused. It should either be implemented for a specific purpose (e.g., filtering or aligning forecasts) and documented, or removed if it's redundant.
# 5. **Clarity of `data_dict` Structure**: While the type hint for `data_dict` defines its nesting, the docstring for `SeasonalForecast` should explicitly describe this structure (e.g., `Dict[field_name, Dict[org_unit_id, Dict[period_str, value_float]]]`) to make it immediately understandable how data is organized and accessed.

"""
This module defines classes for handling and accessing seasonal forecast data.

It includes:
- `DataElement`: A Pydantic model representing a single data point from a seasonal forecast,
                 typically containing an organization unit, period, and value.
- `SeasonalForecast`: A class to store and manage seasonal forecast data for multiple
                      fields (e.g., temperature, precipitation), locations (orgUnits),
                      and time periods. It provides methods to load data from JSON-like
                      structures and retrieve specific forecast time series.
"""

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional  # Added Dict, List, Any, Optional

from pydantic import BaseModel, ValidationError  # Added ValidationError

from chap_core.datatypes import TimeSeriesArray
from chap_core.time_period import PeriodRange  # Added PeriodRange

logger = logging.getLogger(__name__)


class DataElement(BaseModel):
    """
    Represents a single data element from a seasonal forecast.

    Attributes:
        orgUnit (str): The identifier for the organization unit (location).
        period (str): The time period identifier (e.g., "202301" for Jan 2023).
        value (float): The forecasted value for the given orgUnit and period.
    """

    orgUnit: str
    period: str
    value: float


class SeasonalForecast:
    """
    Manages seasonal forecast data for multiple climate fields, locations, and periods.

    The data is stored internally in a nested dictionary structure:
    `data_dict[field_name][org_unit_id][period_string] = forecast_value`

    Provides methods to add data (typically from JSON) and retrieve forecasts
    as `TimeSeriesArray` objects.
    """

    def __init__(self, data_dict: Optional[Dict[str, Dict[str, Dict[str, float]]]] = None):
        """
        Initializes the SeasonalForecast store.

        Args:
            data_dict (Optional[Dict[str, Dict[str, Dict[str, float]]]]):
                An optional pre-existing nested dictionary of forecast data.
                Structure: {field_name: {orgUnit: {period: value}}}
                Defaults to None, which initializes an empty store.
        """
        if data_dict is None:
            self.data_dict: Dict[str, Dict[str, Dict[str, float]]] = {}
        else:
            # Perform a basic check or deep copy if necessary
            self.data_dict = data_dict
        logger.debug("SeasonalForecast initialized.")

    def add_json(self, field_name: str, json_data: List[Dict[str, Any]]) -> None:
        """
        Adds or updates forecast data for a specific climate field from a list of
        JSON-like data element dictionaries.

        Each dictionary in `json_data` is validated against the `DataElement` model.

        Args:
            field_name (str): The name of the climate field these data elements belong to
                              (e.g., "temperature", "precipitation").
            json_data (List[Dict[str, Any]]): A list of dictionaries, where each dictionary
                                              represents a data element with 'orgUnit',
                                              'period', and 'value' keys.
        """
        # Ensure the top-level field_name entry exists.
        # Use defaultdict for the inner dicts to simplify adding new orgUnits/periods.
        if field_name not in self.data_dict:
            self.data_dict[field_name] = defaultdict(dict)

        # Cast to the specific nested type for type checker satisfaction
        # This assumes that if field_name exists, its value is already Dict[str, Dict[str, float]]
        # or defaultdict(dict) which behaves similarly for assignment.
        location_period_data: Dict[str, Dict[str, float]] = self.data_dict[field_name]

        processed_org_units = set()
        for raw_element_data in json_data:
            try:
                data_element = DataElement(**raw_element_data)
                # Ensure the orgUnit sub-dictionary exists if using regular dicts
                if data_element.orgUnit not in location_period_data:
                    location_period_data[data_element.orgUnit] = {}

                location_period_data[data_element.orgUnit][data_element.period] = data_element.value
                processed_org_units.add(data_element.orgUnit)
            except ValidationError as e:
                logger.error(
                    f"Invalid data element encountered for field '{field_name}': {raw_element_data}. Error: {e}"
                )
                # Decide: skip this element, or raise an error to stop processing.
                # For now, skipping invalid elements.
                continue

        if processed_org_units:
            logger.info(
                f"Added/updated data for field '{field_name}', affecting orgUnits: {list(processed_org_units)}."
            )
        else:
            logger.info(f"No valid data elements processed for field '{field_name}'.")

        # If defaultdict was used internally, convert back to regular dict if preferred for storage,
        # though defaultdict is fine.
        # self.data_dict[field_name] = dict(location_period_data)

    def get_forecasts(
        self,
        org_unit: str,
        period_range: PeriodRange,
        field_name: str,
        start_date: Any = None,  # This parameter is unused
    ) -> TimeSeriesArray:
        """
        Retrieves seasonal forecasts for a specific organization unit, time period range,
        and climate field as a `TimeSeriesArray`.

        Args:
            org_unit (str): The identifier of the organization unit (location).
            period_range (PeriodRange): The range of time periods for which to retrieve forecasts.
                                        The `id` attribute of each period in this range is used
                                        as the key for lookup.
            field_name (str): The name of the climate field (e.g., "temperature").
            start_date (Any, optional): This parameter is currently unused. Defaults to None.

        Returns:
            TimeSeriesArray: A TimeSeriesArray containing the forecast values for the
                             specified org_unit, period_range, and field_name.

        Raises:
            KeyError: If `field_name` is not found in the stored data, or if `org_unit`
                      is not found for that field, or if any period in `period_range`
                      is missing for that org_unit and field.
        """
        if start_date is not None:
            logger.warning("The 'start_date' parameter in get_forecasts is currently unused.")

        if field_name not in self.data_dict:
            raise KeyError(
                f"Field '{field_name}' not found in available forecast data. Available fields: {list(self.data_dict.keys())}"
            )

        if org_unit not in self.data_dict[field_name]:
            raise KeyError(
                f"Organization unit '{org_unit}' not found for field '{field_name}'. Available orgUnits: {list(self.data_dict[field_name].keys())}"
            )

        location_specific_data = self.data_dict[field_name][org_unit]

        forecast_values: List[float] = []
        missing_periods: List[str] = []

        for period in period_range:  # Iterate through TimePeriod objects in PeriodRange
            period_id_str = str(period.id)  # Assuming period.id is the string key like "202301"
            if period_id_str not in location_specific_data:
                missing_periods.append(period_id_str)
            else:
                forecast_values.append(location_specific_data[period_id_str])

        if missing_periods:
            raise KeyError(
                f"Not all periods found in data for field '{field_name}', orgUnit '{org_unit}'. "
                f"Missing periods: {missing_periods}. Available periods: {list(location_specific_data.keys())}"
            )

        return TimeSeriesArray(
            time_period=period_range, value=np.array(forecast_values, dtype=float)
        )  # Ensure value is numpy array


# Required for np.array
import numpy as np
