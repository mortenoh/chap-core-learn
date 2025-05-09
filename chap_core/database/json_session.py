# Improvement Suggestions:
# 1. Module-Level Docstring: Add a comprehensive module docstring explaining `JsonSession`'s role (e.g., mock/in-memory session).
# 2. Detailed Class and Method Docstrings: Provide docstrings for `JsonSession` and its methods, clarifying behavior.
# 3. Type Hinting: Add type hints for method parameters (e.g., `elem: Any`) and internal attributes (e.g., `self._data: List[Any]`).
# 4. Clarify "No-Op" Commit: Emphasize in the `commit` method's docstring that it's a no-operation.
# 5. Expand on Use Cases: Briefly mention intended use cases (e.g., testing, temporary data holding) in docstrings.

from typing import Any, List


class JsonSession:
    """
    A mock or in-memory session-like object that collects added elements.

    This class mimics a database session interface by providing `add` and `commit`
    methods, but it only stores elements in an internal list and does not
    perform any actual database persistence. It can be useful for testing
    components that expect a session object or for temporarily holding data.
    """

    def __init__(self) -> None:
        """Initializes a new JsonSession with an empty data list."""
        self._data: List[Any] = []

    def add(self, elem: Any) -> None:
        """
        Adds an element to the internal data list.

        Args:
            elem: The element to add to the session's data list.
        """
        self._data.append(elem)

    def commit(self) -> None:
        """
        Placeholder for a commit operation. Currently does nothing.

        This method is provided to mimic a database session interface but
        does not persist any data.
        """
        pass
