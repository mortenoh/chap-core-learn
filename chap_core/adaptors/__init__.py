# Improvement Suggestions:
# 1. **Module Docstring**: Ensure a clear module docstring explains the role of the `adaptors` package (e.g., "Contains modules for adapting CHAP-core data and functionalities to various external interfaces, libraries, or use cases."). (Primary task).
# 2. **Selective Namespace Exports**: If specific classes or functions from modules within `adaptors` (like `CommandLineInterfaceAdapter` from `command_line_interface.py` or `RestApiAdapter` from `rest_api.py`) are intended as the primary public interface of this package, import them here and list them in `__all__` for cleaner access by users of the `adaptors` package.
# 3. **Package-Level Initialization**: If there's any setup or configuration common to all adaptors in this package (e.g., initializing a common service, setting up a specific logger for adaptors), this `__init__.py` could be a place for it.
# 4. **Consider Sub-Package Structure**: If the number of adaptors grows significantly, evaluate if further sub-packaging within `adaptors` (e.g., `adaptors.data_format_adaptors`, `adaptors.library_wrappers`) would improve organization.
# 5. **Documentation Entry Point**: This `__init__.py`'s docstring can serve as an entry point for developers looking to understand the `adaptors` package. It could briefly list the key adaptors available and their main purpose.

"""
This `__init__.py` file marks the `chap_core/adaptors` directory as a Python package.

The `adaptors` package is intended to house modules that provide interfaces
or transformations between CHAP-core's internal data structures and functionalities,
and external systems, libraries, or specific use-case formats. For example,
it might contain adaptors for different command-line interfaces, REST APIs,
or specific data science libraries like GluonTS.
"""

# Example of how commonly used adaptors could be re-exported for convenience:
# from .command_line_interface import SomeCLIAdapter
# from .gluonts import GluonTSDataAdapter
# from .rest_api import RestAPIAdapter
#
# __all__ = [
#     'SomeCLIAdapter',
#     'GluonTSDataAdapter',
#     'RestAPIAdapter',
# ]

# By default, an empty __init__.py simply makes the directory a package.
