# Improvement Suggestions:
# 1. Module Docstring: Add a module docstring explaining the purpose of the `chap_core.external` package.
# 2. Selective Re-exports (`__all__`): Consider re-exporting key classes/functions from submodules for easier access.
# 3. Package-Level Utilities: If common utilities for external interactions arise, define or import them here.
# 4. Documentation Link: Link to high-level documentation on external model integration if available.
# 5. Version/Compatibility Notes: If relevant, add notes on versioning or compatibility for external components.

"""
The `chap_core.external` package.

This package contains modules and classes for interacting with external models,
tools, and services. This includes wrappers for MLflow projects, R models,
and general external model execution logic.
"""

# Example of potential re-exports (to be reviewed and uncommented if appropriate):
# from .external_model import ExternalModel, ExternalModelPredictor, ExternalModelTemplate
# from .mlflow_wrappers import MLflowProject
# from .python_model import PythonModel
# from .r_model import RModel
#
# __all__ = [
#     "ExternalModel",
#     "ExternalModelPredictor",
#     "ExternalModelTemplate",
#     "MLflowProject",
#     "PythonModel",
#     "RModel",
# ]
