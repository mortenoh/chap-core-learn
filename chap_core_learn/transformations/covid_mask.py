import numpy as np
from bionumpy import replace

from chap_core.datatypes import TimeSeriesData
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet
from chap_core.time_period.date_util_wrapper import TimeStamp


def mask_covid_data(
    data: DataSet,
    start_date: TimeStamp = TimeStamp.parse("2020-03"),
    end_date: TimeStamp = TimeStamp.parse("2021-12-31"),
) -> DataSet:
    """
    Mask disease case values during the COVID-19 period (or any user-defined period)
    by replacing them with NaNs.

    Parameters
    ----------
    data : DataSet
        Dataset with time series data to mask.
    start_date : TimeStamp
        Start date of the mask period (inclusive).
    end_date : TimeStamp
        End date of the mask period (inclusive).

    Returns
    -------
    DataSet
        New dataset where all disease cases between `start_date` and `end_date`
        are replaced with NaN values.
    """

    def insert_nans(ts: TimeSeriesData) -> TimeSeriesData:
        """
        Replace disease_cases with NaN for periods between start_date and end_date.
        """
        mask_start = ts.time_period >= start_date
        mask_end = ts.time_period <= end_date
        mask = mask_start & mask_end

        # Apply mask: keep values outside range, set NaN inside range
        disease_cases = np.where(~mask, ts.disease_cases, np.nan)

        # Return a new TimeSeriesData instance with masked values
        return replace(ts, disease_cases=disease_cases)

    # Apply masking to each location in the dataset
    masked_dict = {location: insert_nans(ts) for location, ts in data.items()}

    return DataSet(masked_dict)
