# Improvement Suggestions:
# 1. Module and Class Docstrings: Add module docstring and class docstring for `SimpleFileContextManager`.
# 2. Evaluate Necessity: Critically evaluate if `SimpleFileContextManager` is needed given Python's `with open()`.
# 3. Method Docstrings and Full Type Hinting: Add docstrings and complete type hints for all methods and parameters.
# 4. Error Handling in `write`/`read`: Current `write`/`read` methods fail silently if file not open; consider raising errors.
# 5. File Closing in `__exit__`: Ensure robust file closing in `__exit__`.

import logging
import os  # For PathLike
from types import TracebackType
from typing import IO, AnyStr, Optional, Type

logger = logging.getLogger(__name__)


class SimpleFileContextManager:
    """
    A basic context manager for file operations.

    Note: Python's built-in `with open(...)` statement is generally preferred
    for file context management as it is more idiomatic and robust.
    This class provides a custom implementation which might have been created
    for specific reasons not immediately apparent or could be a candidate for
    refactoring to use standard library features.
    """

    def __init__(self, filename: os.PathLike | str, mode: str = "r"):
        """
        Initializes the SimpleFileContextManager.

        Args:
            filename: The path to the file.
            mode: The mode in which to open the file (e.g., 'r', 'w', 'a', 'rb', 'wb').
        """
        self.filename: os.PathLike | str = filename
        self.mode: str = mode
        self.file: Optional[IO[AnyStr]] = None

    def __enter__(self) -> Optional[IO[AnyStr]]:
        """
        Opens the file and returns the file object.

        Returns:
            The opened file object, or None if opening fails (though open() raises exceptions).
        """
        try:
            self.file = open(self.filename, self.mode)
            return self.file
        except Exception as e:
            logger.error(f"Failed to open file {self.filename} in mode {self.mode}: {e}")
            # Optionally re-raise or handle more gracefully
            # For now, __exit__ will handle self.file being None
            return None

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> Optional[bool]:
        """
        Closes the file if it was opened.

        Args:
            exc_type: The type of the exception that occurred, if any.
            exc_value: The exception instance that occurred, if any.
            traceback: A traceback object, if an exception occurred.

        Returns:
            Optionally, a boolean indicating if the exception was handled.
            Returning True suppresses the exception.
        """
        if self.file:
            try:
                self.file.close()
            except Exception as e:
                logger.error(f"Error closing file {self.filename}: {e}")
        # Do not suppress exceptions by default
        return None

    def write(self, data: AnyStr) -> Optional[int]:
        """
        Writes data to the file if it is open.

        Args:
            data: The data (str or bytes) to write.

        Returns:
            The number of bytes/characters written, or None if the file is not open.
            Note: Standard file.write() raises ValueError if file is closed.
        """
        if self.file and not self.file.closed:
            try:
                return self.file.write(data)
            except Exception as e:
                logger.error(f"Error writing to file {self.filename}: {e}")
                return None
        else:
            logger.warning(f"Attempted to write to closed or unopened file: {self.filename}")
            return None

    def read(self) -> Optional[AnyStr]:
        """
        Reads data from the file if it is open.

        Returns:
            The data read (str or bytes), or None if the file is not open or on read error.
            Note: Standard file.read() raises ValueError if file is closed.
        """
        if self.file and not self.file.closed:
            try:
                return self.file.read()
            except Exception as e:
                logger.error(f"Error reading from file {self.filename}: {e}")
                return None
        else:
            logger.warning(f"Attempted to read from closed or unopened file: {self.filename}")
            return None
