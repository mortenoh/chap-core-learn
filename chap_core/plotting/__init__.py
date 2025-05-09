# Improvement Suggestions:
# 1. **Module Docstring**: Add a comprehensive module docstring that clearly defines the scope and purpose of the `chap_core.plotting` package, highlighting its role in providing visualization utilities for time series data, forecasts, and model evaluations. (Primary task).
# 2. **Rationale for Re-exports**: In the module docstring or via comments, explain that `plot_timeseries_data` and `plot_multiperiod` are re-exported to provide a convenient and stable public API for common plotting tasks from this package.
# 3. **Completeness of `__all__`**: Review other functions in `plotting.py` and `prediction_plot.py`. If any of those are also considered core, frequently used plotting utilities, consider importing and adding them to the `__all__` list here.
# 4. **External Dependencies Note**: The module docstring could briefly mention key external dependencies for this package, such as `matplotlib`, which is essential for the plotting functions.
# 5. **Usage Examples (Optional)**: If these plotting functions have common usage patterns, a brief example could be included in the module docstring or linked to more detailed documentation/examples.

"""
Initialization file for the `chap_core.plotting` package.

This package provides utilities for creating various visualizations related to
time series data, model forecasts, and evaluation results within the CHAP-core
framework. It aims to offer convenient functions for common plotting tasks.

Currently, it re-exports the following key plotting functions from its
`plotting` submodule for easier access:
- `plot_timeseries_data`: For general time series plotting.
- `plot_multiperiod`: For plotting data across multiple periods, potentially
  highlighting specific ranges or events.
"""

from .plotting import plot_multiperiod, plot_timeseries_data

# Defines the public API of this package when `from chap_core.plotting import *` is used.
# It also signals to users and tools which names are intended for public use from this package.
__all__ = ["plot_timeseries_data", "plot_multiperiod"]
