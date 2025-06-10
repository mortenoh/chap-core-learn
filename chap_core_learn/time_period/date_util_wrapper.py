import logging
from datetime import datetime

from dateutil.parser import parse as _parse
from dateutil.relativedelta import relativedelta
from pytz import utc

logger = logging.getLogger(__name__)


# ------------------------------
# Simple wrapper around dateutil.parser.parse
# ------------------------------
def parse(date_string: str, default: datetime = None):
    """
    Parses a string into a datetime object, allowing short 'YYYYMM' strings.
    """
    if len(date_string) == 6 and date_string.isdigit():
        date_string = date_string[:4] + "-" + date_string[4:6]  # Expand to YYYY-MM
    return _parse(date_string, default=default)


# ------------------------------
# Base class that exposes selected datetime attributes
# ------------------------------
class DateUtilWrapper:
    _used_attributes: tuple = ()

    def __init__(self, date: datetime):
        self._date = date

    def __getattr__(self, item: str):
        # Only allow access to specified attributes from self._date
        if item in self._used_attributes:
            return getattr(self._date, item)
        return super().__getattribute__(item)


# ------------------------------
# Thin wrapper for working with exact timestamps
# ------------------------------
class TimeStamp(DateUtilWrapper):
    _used_attributes = ("year", "month", "day", "__str__", "__repr__")

    @property
    def week(self):
        # Returns ISO week number
        return self._date.isocalendar()[1]

    def __init__(self, date: datetime):
        self._date = date

    @property
    def date(self) -> datetime:
        # Exposes the inner datetime
        return self._date

    @classmethod
    def parse(cls, text_repr: str):
        # Parses a string into a TimeStamp
        return cls(parse(text_repr))

    # --- Comparisons ---
    def __le__(self, other: "TimeStamp"):
        return self._comparison(other, "__le__")

    def __ge__(self, other: "TimeStamp"):
        return self._comparison(other, "__ge__")

    def __gt__(self, other: "TimeStamp"):
        return self._comparison(other, "__gt__")

    def __lt__(self, other: "TimeStamp"):
        return self._comparison(other, "__lt__")

    def __eq__(self, other):
        # Equality based on raw datetime
        return self._date == other._date

    def __sub__(self, other: "TimeStamp"):
        # Difference gives TimeDelta
        if not isinstance(other, TimeStamp):
            return NotImplemented
        return TimeDelta(relativedelta(self._date, other._date))

    def _comparison(self, other: "TimeStamp", func_name: str):
        # Perform UTC-aware comparisons
        return getattr(self._date.replace(tzinfo=utc), func_name)(other._date.replace(tzinfo=utc))

    def __repr__(self):
        return f"TimeStamp({self.year}-{self.month}-{self.day})"
