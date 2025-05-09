# Improvement Suggestions:
# 1. **Clarity on `Control._controls`**: The `Control.__init__` docstring should clearly explain the expected structure and purpose of the `controls` dictionary argument (e.g., what types of keys and values it expects, how these relate to different states).
# 2. **`CancelledError` Context**: The use of `asyncio.CancelledError` in `Control.set_status` implies an asynchronous context. The class docstring for `Control` should clarify if it's primarily designed for asyncio-based applications or if this error is used more generally for cancellation signaling.
# 3. **Type Hint for `Control._controls`**: Provide a more specific type hint for the `controls` parameter in `Control.__init__` and the `_controls` attribute, such as `Dict[str, AnyControlType]` where `AnyControlType` represents the expected type of control objects.
# 4. **`InternalState.current_data` Specificity**: The `current_data` field in `InternalState` is typed as `dict`. If this data has a known, more specific structure (e.g., a Pydantic model or a TypedDict), using that type hint would improve clarity and type safety.
# 5. **Concurrency Considerations**: If instances of `Control` or `InternalState` are intended to be accessed or modified concurrently (e.g., by multiple threads or asyncio tasks), document any necessary synchronization mechanisms (like locks) or state whether they are designed for single-threaded/task access.

"""
This module defines classes for managing internal application state and control flow.

It includes:
- `Control`: A class for managing operational states, cancellation, and progress reporting,
             potentially for long-running tasks or jobs.
- `InternalState`: A dataclass that aggregates various pieces of application state,
                   including a `Control` instance, current data, model paths, and job status.
"""

import dataclasses
from asyncio import CancelledError
from typing import Any, Dict, Optional  # Added Dict, Any

from chap_core.worker.interface import (
    Job,  # Assuming Job has a defined interface (e.g., get_status, get_progress, cancel)
)


class Control:
    """
    Manages the control flow, status, and cancellation of operations.

    This class can be used to track the state of a process, associate different
    sub-control objects with various statuses, and signal cancellation.
    It seems designed to work in contexts where operations can be cancelled,
    potentially including asynchronous environments given the use of `CancelledError`.
    """

    def __init__(self, controls: Dict[str, Any]):
        """
        Initializes the Control object.

        Args:
            controls (Dict[str, Any]): A dictionary mapping status strings to
                                       corresponding control objects. These control
                                       objects are expected to have methods like
                                       `cancel()`, `get_status()`, and `get_progress()`.
                                       Example: {"processing": data_processor_control, "idle": None}
        """
        self._controls: Dict[str, Any] = controls
        self._status: str = "idle"
        self._current_control: Optional[Any] = None
        self._is_cancelled: bool = False

    @property
    def current_control(self) -> Optional[Any]:
        """
        Gets the control object associated with the current status, if any.

        Returns:
            Optional[Any]: The current control object, or None.
        """
        return self._current_control

    def cancel(self):
        """
        Signals cancellation for the current operation.

        If a `_current_control` object is active, its `cancel()` method is called.
        Sets an internal flag to indicate cancellation has been requested.
        """
        if self._current_control is not None and hasattr(self._current_control, "cancel"):
            try:
                self._current_control.cancel()
            except Exception as e:  # pylint: disable=broad-except
                # Log or handle error during sub-control cancellation
                print(f"Error cancelling sub-control: {e}")  # Replace with logger
        self._is_cancelled = True

    def set_status(self, status: str):
        """
        Sets the current operational status and updates the current control object.

        Args:
            status (str): The new status string.

        Raises:
            CancelledError: If the operation has already been cancelled.
        """
        if self._is_cancelled:
            raise CancelledError("Operation was cancelled.")

        self._current_control = self._controls.get(status, None)
        self._status = status

    def get_status(self) -> str:
        """
        Gets the current status, potentially including status from a sub-control.

        Returns:
            str: A string describing the current status. If a sub-control is active
                 and has a `get_status` method, its status is appended.
        """
        if self._current_control is not None and hasattr(self._current_control, "get_status"):
            return f"{self._status}:  {self._current_control.get_status()}"
        return self._status

    def get_progress(self) -> float:  # Assuming progress is a float (e.g., 0.0 to 1.0 or percentage)
        """
        Gets the progress of the current operation, if available from a sub-control.

        Returns:
            float: The progress value (e.g., percentage complete from 0 to 100, or
                   a fraction from 0.0 to 1.0). Returns 0 if no sub-control
                   is active or if the sub-control does not provide progress.
        """
        if self._current_control is not None and hasattr(self._current_control, "get_progress"):
            return self._current_control.get_progress()
        return 0.0  # Default progress if none available


@dataclasses.dataclass
class InternalState:
    """
    A dataclass representing the internal state of an application or a component.

    It aggregates various pieces of state information, such as control objects,
    current data being processed, model paths, and job status.

    Attributes:
        control (Optional[Control]): An instance of the `Control` class for managing
                                     the state's lifecycle (status, cancellation).
        current_data (dict): A dictionary holding the primary data relevant to the
                             current state. The structure of this dictionary is
                             application-specific.
        model_path (Optional[str]): An optional path to a model file or directory.
        current_job (Optional[Job]): An optional reference to a `Job` object,
                                     representing an ongoing task or process.
    """

    control: Optional[Control]
    current_data: Dict[str, Any]  # Changed from dict to Dict[str, Any] for clarity
    model_path: Optional[str] = None
    current_job: Optional[Job] = None  # Type hint was Job | None, Optional[Job] is equivalent and common

    def is_ready(self) -> bool:
        """
        Checks if the component associated with this state is ready for new work.

        It is considered ready if there is no current job, or if the current job
        has finished its execution.

        Returns:
            bool: True if ready, False otherwise.
        """
        return self.current_job is None or (
            hasattr(self.current_job, "is_finished") and self.current_job.is_finished()
        )  # Assuming Job has is_finished method
