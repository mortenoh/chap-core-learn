import dataclasses
import logging
from typing import Generic, Iterable, TypeVar

import numpy as np
import pandas as pd

from ..time_period import PeriodRange
from ..time_period.date_util_wrapper import TimeStamp

logger = logging.getLogger(__name__)

# Type variables
FeaturesT = TypeVar("FeaturesT")
TemporalIndexType = slice


class TemporalDataclass(Generic[FeaturesT]):
    """
    Wraps a single time series object with temporal slicing capabilities.
    """

    def __init__(self, data: FeaturesT):
        self._data = data

    def __repr__(self):
        return f"{self.__class__.__name__}({self._data})"

    def _restrict_by_slice(self, period_range: slice):
        """
        Restrict data using start/stop values from a time period slice.
        """
        assert period_range.step is None
        start, stop = None, None
        if period_range.start is not None:
            start = self._data.time_period.searchsorted(period_range.start)
        if period_range.stop is not None:
            stop = self._data.time_period.searchsorted(period_range.stop, side="right")
        return self._data[start:stop]

    def fill_to_endpoint(self, end_time_stamp: TimeStamp) -> "TemporalDataclass[FeaturesT]":
        """
        Extend the time series with NaNs until a given endpoint.
        """
        if self.end_timestamp == end_time_stamp:
            return self

        n_missing = self._data.time_period.delta.n_periods(self.end_timestamp, end_time_stamp)
        assert n_missing >= 0, (f"{n_missing} < 0", end_time_stamp, self.end_timestamp)

        # Build a new time period range
        old_time_period = self._data.time_period
        new_time_period = PeriodRange(old_time_period.start_timestamp, end_time_stamp, old_time_period.delta)

        # Pad all fields with np.nan
        d = {
            field.name: getattr(self._data, field.name)
            for field in dataclasses.fields(self._data)
            if field.name != "time_period"
        }

        for name, data in d.items():
            d[name] = np.pad(data.astype(float), (0, n_missing), constant_values=np.nan)

        return self._data.__class__(new_time_period, **d)

    def fill_to_range(self, start_timestamp, end_timestamp):
        """
        Extend time series to cover the full [start, end] range.
        """
        if self.end_timestamp == end_timestamp and self.start_timestamp == start_timestamp:
            return self

        n_missing_start = self._data.time_period.delta.n_periods(start_timestamp, self.start_timestamp)
        n_missing_end = self._data.time_period.delta.n_periods(self.end_timestamp, end_timestamp)
        assert n_missing_end >= 0 and n_missing_start >= 0, (
            "Start or end padding invalid",
            start_timestamp,
            self.start_timestamp,
            end_timestamp,
            self.end_timestamp,
        )

        old_time_period = self._data.time_period
        new_time_period = PeriodRange(start_timestamp, end_timestamp, old_time_period.delta)

        # Pad fields at both ends with np.nan
        d = {
            field.name: getattr(self._data, field.name)
            for field in dataclasses.fields(self._data)
            if field.name != "time_period"
        }

        for name, data in d.items():
            d[name] = np.pad(data.astype(float), (n_missing_start, n_missing_end), constant_values=np.nan)

        return self._data.__class__(new_time_period, **d)

    def restrict_time_period(self, period_range: TemporalIndexType) -> "TemporalDataclass[FeaturesT]":
        """
        Slice the temporal range using a slice object (start:stop).
        """
        assert isinstance(period_range, slice)
        assert period_range.step is None

        # Use searchsorted if available
        if hasattr(self._data.time_period, "searchsorted"):
            return self._restrict_by_slice(period_range)

        # Fallback: boolean mask filtering
        mask = np.full(len(self._data.time_period), True)
        if period_range.start is not None:
            mask = mask & (self._data.time_period >= period_range.start)
        if period_range.stop is not None:
            mask = mask & (self._data.time_period <= period_range.stop)
        return self._data[mask]

    def data(self) -> Iterable[FeaturesT]:
        """Return raw dataclass object (e.g. for reuse or testing)."""
        return self._data

    def to_pandas(self) -> pd.DataFrame:
        """Convert wrapped time series to pandas DataFrame."""
        return self._data.to_pandas()

    def join(self, other):
        """Concatenate underlying arrays."""
        return np.concatenate([self._data, other._data])

    @property
    def start_timestamp(self) -> pd.Timestamp:
        return self._data.time_period[0].start_timestamp

    @property
    def end_timestamp(self) -> pd.Timestamp:
        return self._data.time_period[-1].end_timestamp


class Polygon:
    """
    Placeholder for typed polygon integration.
    Actual usage handled by FeatureCollectionModel or geometry.Polygons.
    """

    pass
