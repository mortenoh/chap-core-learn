# Improvement Suggestions:
# 1. **Comprehensive Docstrings**: Add detailed docstrings for all classes and methods to explain their purpose, arguments, return values, and any side effects or raised exceptions. (Primary task of this refactoring).
# 2. **Safe Division in `get_progress`**: In `TrainingControl.get_progress`, add a check for `self._total_samples == 0` (in addition to `None`) to prevent `ZeroDivisionError` and return a sensible default (e.g., 0.0 or 1.0 if `_n_finished` is also 0).
# 3. **Complete Type Hinting**: Ensure all method parameters and return values have appropriate type hints for improved code clarity and static analysis (e.g., `total_samples: int`, `n_sampled: int`, return types for all methods).
# 4. **Logging in `PrintingTrainingControl`**: Replace `print()` statements in `PrintingTrainingControl` with calls to the `logging` module. This allows for better control over message verbosity and output streams by the application using this class.
# 5. **Clarity on `CancelledError` Usage**: Document the rationale for using `asyncio.CancelledError`. If these classes are not exclusively for `asyncio` contexts, consider defining and using a custom `OperationCancelledError` for broader applicability and to avoid confusion with asyncio-specific cancellation.

"""
This module provides classes for controlling and monitoring the progress
of training processes or other potentially long-running operations.

It includes a base `TrainingControl` class for managing status, progress,
and cancellation, and a `PrintingTrainingControl` subclass that adds
console output for status and progress updates.
"""

import logging  # For PrintingTrainingControl
from asyncio import CancelledError
from typing import Optional  # Added Optional

logger = logging.getLogger(__name__)


class TrainingControl:
    """
    A base class to manage and monitor the state of a training process or similar operation.

    Provides mechanisms to set total work units (samples), register progress,
    update status messages, and handle cancellation requests.
    Uses `asyncio.CancelledError` to signal cancellation, implying it might be
    intended for or compatible with asynchronous programming contexts.
    """

    def __init__(self) -> None:
        """Initializes the TrainingControl state."""
        self._total_samples: Optional[int] = None
        self._cancelled: bool = False
        self._status: str = "Initialized"  # Provide a more descriptive initial status
        self._n_finished: int = 0

    def set_total_samples(self, total_samples: int) -> None:
        """
        Sets the total number of samples or work units for the operation.

        Args:
            total_samples (int): The total number of samples. Must be positive.

        Raises:
            ValueError: If `total_samples` is not positive.
        """
        if total_samples <= 0:
            raise ValueError("total_samples must be a positive integer.")
        self._total_samples = total_samples
        self._n_finished = 0  # Reset finished count when total samples change

    def get_progress(self) -> float:
        """
        Calculates the progress of the operation as a fraction (0.0 to 1.0).

        Returns:
            float: The fraction of completed work. Returns 0.0 if total samples
                   is not set or is zero.
        """
        if self._total_samples is None or self._total_samples == 0:
            return 0.0
        return self._n_finished / self._total_samples

    def get_status(self) -> str:
        """
        Gets the current status message of the operation.

        Returns:
            str: The current status string.
        """
        return self._status

    def register_progress(self, n_sampled: int) -> None:
        """
        Registers that a certain number of samples/work units have been processed.

        Args:
            n_sampled (int): The number of newly processed samples.

        Raises:
            CancelledError: If the operation has been cancelled.
            ValueError: If `n_sampled` is not a positive integer.
        """
        if self._cancelled:
            raise CancelledError("Operation has been cancelled.")
        if not isinstance(n_sampled, int) or n_sampled < 0:
            # Allow n_sampled = 0 for idempotent calls, but typically positive.
            raise ValueError("n_sampled must be a non-negative integer.")
        self._n_finished += n_sampled
        # Ensure _n_finished does not exceed _total_samples if set
        if self._total_samples is not None and self._n_finished > self._total_samples:
            self._n_finished = self._total_samples

    def set_status(self, status: str) -> None:
        """
        Sets a new status message for the operation.

        Args:
            status (str): The new status message.

        Raises:
            CancelledError: If the operation has been cancelled.
        """
        if self._cancelled:
            raise CancelledError("Operation has been cancelled.")
        self._status = status

    def cancel(self) -> None:
        """
        Flags the operation as cancelled.
        Subsequent calls to `register_progress` or `set_status` will raise `CancelledError`.
        """
        logger.info("Cancellation requested for the current operation.")
        self._cancelled = True

    def is_cancelled(self) -> bool:
        """
        Checks if the operation has been flagged as cancelled.

        Returns:
            bool: True if cancelled, False otherwise.
        """
        return self._cancelled


class PrintingTrainingControl(TrainingControl):
    """
    A subclass of `TrainingControl` that prints progress and status updates to the console
    (or standard logger if configured).
    """

    def __init__(self, print_to_logger: bool = True):
        """
        Initializes PrintingTrainingControl.

        Args:
            print_to_logger (bool): If True (default), messages are sent to the logger.
                                    If False, messages are sent to `print()`.
        """
        super().__init__()
        self._print_to_logger = print_to_logger
        if print_to_logger and not logger.hasHandlers():
            # Basic config if no handlers for the logger used by this class
            logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    def register_progress(self, n_sampled: int) -> None:
        """
        Registers progress and prints the current progress percentage.
        Overrides `TrainingControl.register_progress`.

        Args:
            n_sampled (int): The number of newly processed samples.
        """
        super().register_progress(n_sampled)
        message = f"Progress: {self.get_progress() * 100:.2f}% ({self._n_finished}/{self._total_samples or '?'})"
        if self._print_to_logger:
            logger.info(message)
        else:
            print(message)

    def set_status(self, status: str) -> None:
        """
        Sets a new status and prints it.
        Overrides `TrainingControl.set_status`.

        Args:
            status (str): The new status message.
        """
        super().set_status(status)
        message = f"Status: {self.get_status()}"
        if self._print_to_logger:
            logger.info(message)
        else:
            print(message)
