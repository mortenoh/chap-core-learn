import numpy as np
from npstructures import RaggedArray

from chap_core.time_period import Month


def pack_to_period(time_period, data, goal_period):
    """
    Group raw time series data into a coarser time unit (e.g. weeks → months).

    Parameters
    ----------
    time_period : array-like
        The original time period object (e.g. list of Week or Month instances).
    data : array-like
        The corresponding data values (e.g. temperatures, cases).
    goal_period : type
        The class of the desired target period (currently only Month is supported).

    Returns
    -------
    new_index : Month
        Array of Month objects representing the grouped periods.
    grouped_data : RaggedArray
        Values grouped by period (e.g. one array per month).

    Notes
    -----
    - This assumes `time_period` has `month` and `year` attributes.
    - Returns a RaggedArray: a structure where each row can have a different length.
    """
    if goal_period is Month:
        # Find where the month changes (i.e., start of a new group)
        changes = np.flatnonzero(np.diff(time_period.month)) + 1

        # Insert 0 as the start of the first group
        period_starts = np.insert(changes, 0, 0)

        # Extract period identifiers at group boundaries
        new_index = time_period[period_starts]
        new_index = Month(month=new_index.month, year=new_index.year)

        # Compute lengths of each group
        period_lengths = np.diff(np.append(period_starts, len(time_period)))

        # Return grouped periods and corresponding values
        return new_index, RaggedArray(data, period_lengths)
