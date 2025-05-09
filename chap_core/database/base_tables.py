# Improvement Suggestions:
# 1. **Module Docstring**: Add a comprehensive module docstring explaining that this file provides foundational elements for database table models, including a base SQLModel class with Pydantic configurations for API interaction and common type aliases. (Primary task).
# 2. **`PeriodID` Documentation**: Add a comment to the `PeriodID = str` type alias to specify the expected format or conventions for period identifiers (e.g., "YYYYMM" for monthly, "YYYY-WW" for weekly, or if it's a generic string).
# 3. **Elaborate `DBModel` Docstring**: Expand the docstring for `DBModel` to clearly state its purpose as a common base for all SQLModel table definitions in the project. Emphasize that its Pydantic `ConfigDict` (using `to_camel` for aliases and `populate_by_name`) standardizes JSON serialization/deserialization, particularly for REST API interactions.
# 4. **Clarify `table=True` Expectation**: The `DBModel` itself is not a table. Its docstring or a comment could note that subclasses intended to be database tables must include `table=True` in their class definition (e.g., `class MyTable(DBModel, table=True): ...`).
# 5. **Illustrative Example (Optional)**: Consider adding a brief, commented-out example of a simple model inheriting from `DBModel` to illustrate how the camelCase aliasing works with field definitions and Pydantic, e.g., `my_field: str = Field(default=None, alias="myField")`.

"""
This module defines foundational elements for database table models within CHAP-core,
primarily using SQLModel and Pydantic.

It provides:
- A base class `DBModel` that all SQLModel table models should inherit from.
  This class includes Pydantic configurations to automatically generate camelCase
  aliases for field names, facilitating interoperability with JSON-based REST APIs
  that typically use camelCase.
- Common type aliases like `PeriodID`.
"""

from pydantic import ConfigDict
from pydantic.alias_generators import to_camel
from sqlmodel import SQLModel

# Type alias for Period Identifiers.
# Expected format is typically a string like "YYYYMM" (e.g., "202301" for January 2023)
# or "YYYYWW" (e.g., "2023W01" for the first week of 2023), depending on the context.
PeriodID = str


class DBModel(SQLModel):
    """
    Base class for all SQLModel table models in the CHAP-core project.

    It provides a Pydantic `model_config` to:
    - Automatically generate camelCase aliases for field names (e.g., `my_field` becomes `myField`).
    - Allow Pydantic models to be populated using these alias names when parsing data
      (e.g., from JSON request bodies).

    This setup is primarily intended to standardize data exchange with REST APIs
    that commonly use camelCase for JSON keys. Subclasses that represent database
    tables should also include `table=True` in their class definition, e.g.:
    `class MyActualTable(DBModel, table=True): ...`
    """

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    # Example of how a field would be defined in a subclass:
    # id: Optional[int] = Field(default=None, primary_key=True)
    # my_database_column: str = Field(alias="myDatabaseColumnApi")
