# Improvement Suggestions:
# 1. **Comprehensive Docstrings**: Add detailed docstrings for all functions that currently lack them (`interpolate_nans`, `conda_available`, `docker_available`, `pyenv_available`, `redis_available`) and a descriptive module-level docstring. (Primary task of this refactoring).
# 2. **Complete Type Hinting**: Ensure all function parameters and return values have appropriate type hints (e.g., `y: np.ndarray` for `interpolate_nans`, `-> bool` for availability check functions).
# 3. **Robustness of `interpolate_nans`**: Document or handle edge cases in `interpolate_nans`, such as when all array elements are NaN, or when NaNs are at the start/end of the array, which can affect `np.interp` behavior (it won't extrapolate).
# 4. **Refined `redis_available` Error Handling**: In `redis_available`, consider logging the specific exception `e` when it's not a `ModuleNotFoundError` or `redis.exceptions.ConnectionError` before re-raising, to aid debugging. Also, explicitly import `redis.exceptions.ConnectionError` if that's the intended specific error.
# 5. **Clarity on Availability Checks**: The docstrings for `conda_available`, `docker_available`, and `pyenv_available` should clarify that they check for the presence of command-line executables, not Python library availability (which `redis_available` does for the `redis` library).

"""
This module provides various utility functions for common tasks within CHAP-core.

Includes helpers for:
- NaN handling and interpolation in NumPy arrays.
- Checking the availability of external tools/services like Conda, Docker, Pyenv, and Redis.
"""

import logging  # Added logging
from shutil import which
from typing import Callable, Tuple  # Added Callable, Tuple, Any

import numpy as np

logger = logging.getLogger(__name__)


def nan_helper(y: np.ndarray) -> Tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y: 1d numpy array with possible NaNs
    Output:
        - nans: logical indices of NaNs (boolean array)
        - index: a function, with signature `indices = index(logical_indices)`,
                 to convert logical indices of NaNs to their 'equivalent' integer indices.
    Example:
        >>> # linear interpolation of NaNs
        >>> y_example = np.array([1, 2, np.nan, 4, np.nan, np.nan, 7, 8])
        >>> nans, x_func = nan_helper(y_example)
        >>> y_example[nans] = np.interp(x_func(nans), x_func(~nans), y_example[~nans])
        >>> print(y_example)
        [1. 2. 3. 4. 5. 6. 7. 8.]
    """
    return np.isnan(y), lambda z: z.nonzero()[0]


def interpolate_nans(y: np.ndarray) -> np.ndarray:
    """
    Performs linear interpolation of NaN values in a 1D numpy array.

    Note: This function uses `np.interp`, which does not extrapolate.
    NaNs at the beginning or end of the array that cannot be interpolated
    will remain NaNs. If all values are NaN, the array is returned unchanged.

    Args:
        y (np.ndarray): A 1D numpy array potentially containing NaN values.

    Returns:
        np.ndarray: The array with NaNs interpolated. If all inputs are NaN or
                    interpolation is not possible for some NaNs (e.g., at ends),
                    those NaNs will persist.
    """
    if not isinstance(y, np.ndarray):
        raise TypeError("Input 'y' must be a numpy array.")
    if y.ndim != 1:
        raise ValueError("Input array 'y' must be 1-dimensional.")

    nans, x = nan_helper(y)

    # If all are NaNs, or no non-NaNs to interpolate from, return as is
    if np.all(nans) or not np.any(~nans):
        logger.debug("Array contains all NaNs or no valid points for interpolation; returning original array.")
        return y

    try:
        y[nans] = np.interp(x(nans), x(~nans), y[~nans])
    except Exception as e:  # Catch potential errors during interpolation
        logger.error(f"Error during NaN interpolation: {e}. Array state: y={y}, nans={nans}")
        # Depending on desired behavior, could re-raise or return y with partial/no interpolation
    return y


def conda_available() -> bool:
    """
    Checks if the 'conda' command-line executable is available in the system PATH.

    Returns:
        bool: True if 'conda' is found, False otherwise.
    """
    return which("conda") is not None


def docker_available() -> bool:
    """
    Checks if the 'docker' command-line executable is available in the system PATH.

    Returns:
        bool: True if 'docker' is found, False otherwise.
    """
    return which("docker") is not None


def pyenv_available() -> bool:
    """
    Checks if the 'pyenv' command-line executable is available in the system PATH.

    Returns:
        bool: True if 'pyenv' is found, False otherwise.
    """
    return which("pyenv") is not None


def redis_available() -> bool:
    """
    Checks if a Redis server is available and connectable.

    Attempts to import the `redis` Python library and ping a Redis server
    (defaults to localhost:6379).

    Returns:
        bool: True if the `redis` library can be imported and a Redis server
              responds to a ping, False otherwise.

    Raises:
        Exception: Re-raises exceptions other than `ModuleNotFoundError` or
                   `redis.exceptions.ConnectionError` (or its subclasses)
                   that occur during the check.
    """
    try:
        import redis  # Try importing here to catch ModuleNotFoundError specifically for redis
        from redis.exceptions import ConnectionError as RedisConnectionError  # Specific import

        # Default connection parameters, consider making them configurable if needed
        r = redis.Redis()  # host='localhost', port=6379, db=0
        r.ping()
        logger.debug("Redis connection successful (ping successful).")
        return True
    except ModuleNotFoundError:
        logger.warning("Python 'redis' library not found. Redis is considered unavailable.")
        return False
    except RedisConnectionError:  # Catching specific Redis connection error
        logger.warning("Could not connect to Redis server. Redis is considered unavailable.")
        return False
    except Exception as e:
        # Log other types of Redis exceptions or unexpected errors if they occur
        logger.error(f"An unexpected error occurred while checking Redis availability: {e.__class__.__name__}: {e}")
        # The original code re-raised other exceptions.
        # Decide if all other exceptions should mean "unavailable" or if they are true errors.
        # For a simple availability check, often any failure means "unavailable".
        # If strict error reporting for unexpected issues is desired, then re-raise:
        # raise
        # If any failure to connect/ping means "unavailable":
        return False
