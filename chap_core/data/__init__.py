# Improvement Suggestions:
# 1. **Module Docstring**: Add a comprehensive module docstring that clearly defines the scope and purpose of the `chap_core.data` package. This should explain that it handles core data structures, adaptors, and specific dataset utilities. (Primary task).
# 2. **Rationale for Re-exports**: In the module docstring or via comments, explain why `DataSet` and `PeriodObservation` are specifically chosen for re-export. This helps users understand the intended public API of this package.
# 3. **Completeness of `__all__`**: Review the contents of the `chap_core.data` package (including submodules like `adaptors.py`, `datasets.py`, `open_dengue.py`, and the `gluonts_adaptor` sub-package). If other classes or functions are considered part of its primary public interface, they should also be imported here and added to `__all__`.
# 4. **API Stability**: Re-exporting types can create a more stable API for users of the `chap_core.data` package, as the internal location of these types can change without breaking external code (as long as the re-export in this `__init__.py` is updated). This benefit could be mentioned if it's an intentional design choice.
# 5. **Overview of Sub-components**: The module docstring could briefly outline the roles of the key modules and sub-packages within `chap_core.data` (e.g., "Includes `datasets.py` for loading example/standard datasets, `gluonts_adaptor/` for interfacing with GluonTS, etc.") to provide a better map for developers.

"""
Initialization file for the `chap_core.data` package.

This package provides core data handling capabilities for the CHAP-core application.
It includes:
- The main `DataSet` class (re-exported from `chap_core.spatio_temporal_data.temporal_dataclass`)
  which is a fundamental structure for holding multi-location time series data.
- `PeriodObservation` (re-exported from `chap_core.api_types`), a Pydantic model often
  used as a helper for constructing time series.
- Submodules for specific dataset loading (e.g., `open_dengue.py`), example datasets
  (`datasets.py`), and adaptors for external libraries like GluonTS (`gluonts_adaptor/`
  and `adaptors.py`).

The re-exports of `DataSet` and `PeriodObservation` aim to provide convenient access
to these commonly used types directly from the `chap_core.data` namespace.
"""

from ..api_types import PeriodObservation
from ..spatio_temporal_data.temporal_dataclass import DataSet

# Defines the public API of this package when `from chap_core.data import *` is used.
# It also signals to users and tools which names are intended for public use from this package.
__all__ = ["DataSet", "PeriodObservation"]
