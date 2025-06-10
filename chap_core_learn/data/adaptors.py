# Improvement Suggestions:
# 1. **Module Docstring**: Add a clear module docstring explaining that this module provides pre-instantiated data adaptors, currently focusing on an adaptor for GluonTS datasets. (Primary task).
# 2. **Rationale for Singleton Instance**: Document the reason for providing `gluonts` as a pre-instantiated singleton of `DataSetAdaptor`. Is it intended to be used globally with a default configuration, or is it mainly for convenience?
# 3. **Naming Clarity**: While `gluonts` is concise, consider if a more descriptive name like `default_gluonts_dataset_adaptor` or `gluonts_adaptor_instance` would improve clarity for new developers, especially if other adaptors are introduced.
# 4. **Future Extensibility**: If more data adaptors (e.g., for other time series libraries) are anticipated, briefly mention in the docstring how this module might evolve or how users might contribute/add new adaptors.
# 5. **Configuration of `DataSetAdaptor`**: If the `DataSetAdaptor` itself has configurable parameters or behaviors, document how users can obtain a differently configured instance if the default `gluonts` instance provided here is not suitable for their specific needs. Alternatively, this module could provide a factory function.

"""
This module serves as a central access point for pre-instantiated data adaptors
used within the CHAP-core data processing pipeline.

Currently, it provides a default instance of `DataSetAdaptor` for interfacing
with the GluonTS library. This allows for convenient conversion between
CHAP-core's native `DataSet` format and the formats expected by GluonTS
for training and prediction.

The intention is to simplify the use of common adaptors by providing a
ready-to-use instance.
"""

from .gluonts_adaptor.dataset import DataSetAdaptor

# A pre-instantiated default DataSetAdaptor for GluonTS.
# This can be imported and used directly by other modules in CHAP-core
# that need to convert data to/from GluonTS formats.
# Example usage:
# from chap_core.data.adaptors import gluonts
# gluonts_formatted_data = gluonts.to_gluonts(my_chap_dataset)
gluonts = DataSetAdaptor()
