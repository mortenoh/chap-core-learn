# Improvement Suggestions:
# 1. **Module Docstring**: Add a clear module docstring explaining the purpose of this `__init__.py` file. For example, specify if it's primarily to mark the directory as a Python package, or if it's intended to expose common utilities or mocks for tests within the `climate_data` sub-package.
# 2. **Clarify Import Usage**: Document the reason for importing `ClimateDataBaseMock`. If it's meant to be re-exported for easier access by other test modules within this sub-package, consider adding it to an `__all__` list.
# 3. **Test Package Overview**: If this `__init__.py` serves as an entry point or organizational hub for `tests/climate_data`, a brief comment outlining the types of tests contained within this sub-package (e.g., GEE integration, climate data processing) could be helpful.
# 4. **Review Necessity of Import**: If `ClimateDataBaseMock` is imported here but not re-exported or directly used by other modules importing from `tests.climate_data` itself, evaluate if this import is necessary in this specific `__init__.py`. It might be imported directly where needed.
# 5. **Verify Relative Import Path**: The relative import `from ..mocks import ClimateDataBaseMock` assumes a `mocks` module or package exists in the parent directory (`tests/`). Confirm this structure and ensure it's robust.

"""
Initialization file for the `tests.climate_data` test sub-package.

This file makes the `climate_data` directory a Python package.
It may also be used to expose common fixtures, mocks, or utilities
for tests related to climate data processing, fetching, and integration
(e.g., with Google Earth Engine).

Currently, it imports `ClimateDataBaseMock` from the parent `tests.mocks` module,
potentially making it available for use within the `tests.climate_data` namespace.
"""

from ..mocks import ClimateDataBaseMock

# To explicitly re-export for easier import by modules in this package,
# you could add:
# __all__ = ['ClimateDataBaseMock']
