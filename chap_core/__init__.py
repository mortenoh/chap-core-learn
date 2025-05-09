# Improvement Suggestions:
# 1. Add a more descriptive module-level docstring explaining the purpose of the `chap-core` package.
# 2. Consider using a more structured way to manage package metadata (e.g., `importlib.metadata` if Python 3.8+).
# 3. Review the `__all__` list to ensure it accurately reflects the public API of the package.
# 4. Add a link to the project's documentation or repository in the module docstring.
# 5. Consider adding type hints for `__author__`, `__email__`, and `__version__` for better static analysis.

"""
Top-level package for chap-core.

This package initializes the chap-core library, providing access to its core
functionalities including data fetching, data handling, and model interfaces.
It also defines package-level metadata such as author and version.
"""

__author__ = """Sandvelab"""
__email__ = "knutdrand@gmail.com"
__version__ = "1.0.7"

from . import data, fetch
from .models.model_template_interface import ModelTemplateInterface

__all__ = ["fetch", "data", "ModelTemplateInterface"]
