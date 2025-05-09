from . import Month, TimePeriod, Year, delta_week


def previous(period: TimePeriod) -> TimePeriod:
    """
    Return the time period that comes immediately before the given one.

    Supports:
    - Month: handles wrapping from January to December of the previous year.
    - Year: subtracts one year.
    - Week: subtracts one delta_week (usually 1 week).

    Parameters
    ----------
    period : TimePeriod
        The time period object to shift backwards.

    Returns
    -------
    TimePeriod
        The previous time period.

    Raises
    ------
    NotImplementedError
        If the type of `period` is unsupported.
    """
    if period.__class__.__name__ == "Month":
        # If it's January, wrap to December of the previous year
        prev_year = period.year - (period.month == 1)
        prev_month = (period.month - 2) % 12 + 1
        return Month(prev_year, prev_month)

    elif period.__class__.__name__ == "Year":
        # Simply subtract one year
        return Year(period.year - 1)

    elif period.__class__.__name__ == "Week":
        # Subtract one week using delta_week
        return period - delta_week

    raise NotImplementedError(f"previous not implemented for {type(period)}")
