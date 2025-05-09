# Improvement Suggestions:
# 1. **Clarity on Path Usage**: Add comments above each path constant (`EXAMPLE_DATA_PATH`, `TMP_DATA_PATH`, `TEST_PATH`) briefly explaining its purpose or how it's used within the test suite.
# 2. **`TMP_DATA_PATH` Management**: If tests write files to `TMP_DATA_PATH`, ensure this directory is created before tests run (e.g., in a global test setup fixture or `setUpModule`) and potentially cleaned up afterwards to avoid test artifacts.
# 3. **Explicit Type Hinting for Constants**: Add explicit type hints for the path constants (e.g., `EXAMPLE_DATA_PATH: Path`). While type is inferred, explicit hints improve readability and static analysis.
# 4. **Consider `importlib.resources` for Package Data**: For accessing data files within the package (like `example_data`), `importlib.resources` (Python 3.7+) provides a more robust way than `__file__`-based paths, especially for packaged distributions. (This is a more advanced consideration).
# 5. **Test Suite Configuration Point**: If there are global configurations or setup required for the entire test suite (e.g., setting environment variables, initializing services), this `__init__.py` (using `setUpModule` and `tearDownModule`) could be an appropriate place to manage them.

"""Unit test package for chap_core."""

from pathlib import Path

# Path to the directory containing example data files used in tests.
EXAMPLE_DATA_PATH: Path = Path(__file__).parent.parent / "example_data"

# Path to a temporary directory for tests to write data to.
# Note: Ensure this directory is created and cleaned up appropriately by test setup/teardown.
TMP_DATA_PATH: Path = Path(__file__).parent / "tmp_data"

# Path to the root directory of the tests package itself.
TEST_PATH: Path = Path(__file__).parent
