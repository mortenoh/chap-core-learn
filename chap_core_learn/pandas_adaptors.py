# Improvement Suggestions:
# 1. **Implement Full Functionality**: Complete the `get_time_period` function to handle daily (`day_name`) and weekly (`week_name`) period constructions as suggested by its parameters, instead of raising an error for non-monthly data.
# 2. **Refined Error Handling**: Replace `AssertionError` with more specific exceptions like `ValueError` for invalid parameter combinations (e.g., providing both `month_name` and `week_name`) or `NotImplementedError` for currently unsupported period types.
# 3. **Comprehensive Type Hinting**: Add complete type hints for all function parameters (e.g., `df: pd.DataFrame`, `year_name: str`) and the function's return type (e.g., `List[pd.Period]`).
# 4. **Input Column Validation**: Before accessing columns in the DataFrame (e.g., `df[year_name]`), validate that these columns exist to prevent `KeyError` exceptions. Also, consider validating the data types within these columns.
# 5. **Flexibility in Period Construction**: For monthly data, the period string is hardcoded as "YYYY-MM". If input data might have different year/month formats, consider adding flexibility or clearer documentation on expected input formats.

"""
This module provides utility functions for adapting pandas DataFrames,
particularly for constructing time period information from DataFrame columns.

Currently, it includes a function to generate pandas Period objects from
year and month columns, with placeholders for future support of daily or weekly data.
"""

from typing import List, Optional  # Added List, Optional

import pandas as pd


def get_time_period(
    df: pd.DataFrame,
    year_name: str,
    month_name: Optional[str] = None,
    day_name: Optional[str] = None,
    week_name: Optional[str] = None,
) -> List[pd.Period]:
    """
    Constructs a list of pandas Period objects from DataFrame columns.

    Currently, this function primarily supports monthly period construction.
    Support for daily and weekly periods is not yet fully implemented.

    Args:
        df (pd.DataFrame): The input DataFrame.
        year_name (str): The name of the column containing year information.
        month_name (Optional[str]): The name of the column containing month information.
                                    Required for monthly periods. Defaults to None.
        day_name (Optional[str]): The name of the column containing day information.
                                  (Currently not implemented). Defaults to None.
        week_name (Optional[str]): The name of the column containing week information.
                                   (Currently not implemented). Defaults to None.

    Returns:
        List[pd.Period]: A list of pandas Period objects.

    Raises:
        ValueError: If incompatible combinations of period arguments are provided
                    (e.g., both month and week).
        NotImplementedError: If attempting to use day-based or week-based period
                             construction, which is not yet supported.
        KeyError: If specified column names (`year_name`, `month_name`, etc.)
                  are not found in the DataFrame.
    """
    # Validate input columns exist
    required_cols = [year_name]
    if month_name:
        required_cols.append(month_name)
    if day_name:
        required_cols.append(day_name)
    if week_name:
        required_cols.append(week_name)

    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"Required column '{col}' not found in DataFrame.")

    if month_name is not None:
        if day_name is not None:
            raise ValueError("Cannot specify both 'day_name' and 'month_name' for monthly period construction.")
        if week_name is not None:
            raise ValueError("Cannot specify both 'week_name' and 'month_name' for monthly period construction.")

        try:
            # Ensure year and month columns can be iterated together and are suitable for period construction
            return [pd.Period(f"{year}-{month}", freq="M") for year, month in zip(df[year_name], df[month_name])]
        except Exception as e:
            # Catch potential errors during Period creation (e.g., invalid date parts)
            raise ValueError(f"Error creating monthly periods from columns '{year_name}' and '{month_name}': {e}")

    elif day_name is not None:
        # TODO: Implement daily period construction
        # Example: return [pd.Period(f"{year}-{month}-{day}", freq="D") for year, month, day in zip(df[year_name], df[month_name], df[day_name])]
        # This would also require month_name to be present.
        raise NotImplementedError("Daily period construction is not yet implemented.")

    elif week_name is not None:
        # TODO: Implement weekly period construction
        # Example: return [pd.Period(f"{year}-W{week:02d}", freq="W-SUN") for year, week in zip(df[year_name], df[week_name])]
        # This might need more complex logic for ISO weeks or specific week definitions.
        raise NotImplementedError("Weekly period construction is not yet implemented.")

    else:
        raise ValueError("At least one of 'month_name', 'day_name', or 'week_name' must be provided.")
