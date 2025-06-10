# Improvement Suggestions:
# 1. Module-Level Docstring: Add a comprehensive module docstring explaining the purpose of this file (defining a table for debug entries).
# 2. Detailed Class and Field Docstrings: Provide docstrings for the `DebugEntry` class and its `id` and `timestamp` fields.
# 3. Use `datetime` for Timestamp: Change `timestamp` field from `float` to `datetime.datetime`, using `default_factory=datetime.utcnow`.
# 4. Clarify Timestamp Meaning (if float retained): If `float` is kept for `timestamp`, explicitly document it as a Unix timestamp.
# 5. Example Usage Context: Briefly mention in docstrings a typical use case, e.g., with `SessionWrapper.add_debug()`.

from datetime import datetime
from typing import Optional

from sqlmodel import Field

from chap_core.database.base_tables import DBModel


class DebugEntry(DBModel, table=True):
    """
    Represents a simple debug entry in the database.

    This table is typically used for basic testing of database connectivity
    or for logging simple, timestamped events for debugging purposes.
    An example of its creation can be found in `SessionWrapper.add_debug()`.
    """

    id: Optional[int] = Field(default=None, primary_key=True, description="Primary key for the debug entry.")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Timestamp indicating when the debug entry was created (UTC)."
    )
