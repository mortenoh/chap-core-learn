# Improvement Suggestions:
# 1. **Logger Specificity**: Configure a logger specific to 'chap_core' (e.g., `logging.getLogger('chap_core')`) instead of the root logger to avoid conflicts with applications using this library. The root logger configuration should ideally be left to the final application.
# 2. **Handler Idempotency**: Ensure `initialize_logging` is idempotent regarding file handlers. If called multiple times with a `log_file`, it should not add duplicate handlers to the logger, which can cause repeated log messages.
# 3. **Robust File Operation Error Handling**: Wrap file system operations (directory creation, file creation/touching, chmod, file reading in `get_logs`) in `try-except` blocks to handle potential `IOError` or `OSError` exceptions gracefully (e.g., permission issues).
# 4. **Consistent Use of Logging**: Replace `print()` statements used for status messages within `initialize_logging` with appropriate `logger` calls (e.g., `logger.info`, `logger.warning`) for consistent log output.
# 5. **Managing Global State (`_global_log_file`)**: The use of a global variable `_global_log_file` can be problematic. Consider encapsulating logging configuration (like the log file path) in a class or returning it from `initialize_logging` to be managed by the caller, reducing reliance on global state.

"""
This module provides logging configuration utilities for the CHAP-core application.

It allows for initializing the logging system with a specified debug level and
log file path, retrieving the configured log file path, and fetching log content.
Logging can be directed to a file specified via argument or an environment variable.
"""

import logging
import os
from pathlib import Path
from typing import Optional  # Added Optional

# Get a logger instance for this module, or for 'chap_core' if preferred for library use.
# Using root logger for now as per original, but see suggestion #1.
logger = logging.getLogger()  # Root logger
_global_log_file: Optional[str] = None


def initialize_logging(debug: bool = False, log_file: Optional[str] = None) -> None:
    """
    Initializes or reconfigures the logging system for the application.

    Sets the logging level (DEBUG if `debug` is True, INFO otherwise).
    Configures a FileHandler if `log_file` is provided or if the
    `CHAP_LOG_FILE` environment variable is set.

    Args:
        debug (bool): If True, sets logging level to DEBUG. Otherwise, INFO.
                      Defaults to False.
        log_file (Optional[str]): Path to the desired log file. If None,
                                  the `CHAP_LOG_FILE` environment variable is checked.
                                  Defaults to None (console logging only, unless env var is set).
    """
    global _global_log_file

    # Set logging level
    if debug:
        effective_level = logging.DEBUG
        level_name = "DEBUG"
    else:
        effective_level = logging.INFO
        level_name = "INFO"

    # Check if logger already has handlers to avoid adding duplicate stream handlers
    # or to decide on replacing/clearing them. For simplicity, this example
    # assumes basic configuration or that it's called once.
    # A more robust solution would manage handlers more carefully.

    # Ensure basic configuration for console output if no handlers exist
    if not logger.hasHandlers():
        logging.basicConfig(level=effective_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        logger.info(f"Basic logging configured. Level set to {level_name}.")
    else:  # If handlers exist, just set level
        logger.setLevel(effective_level)
        logger.info(f"Logging level set to {level_name}.")

    if debug:  # This message will now use the configured logger
        logger.debug("Debug mode enabled via initialize_logging.")

    # Determine log file path
    final_log_file_path = log_file
    env_log_file = os.getenv("CHAP_LOG_FILE")
    if env_log_file and final_log_file_path is None:
        final_log_file_path = env_log_file
        logger.info(f"Using log file from CHAP_LOG_FILE environment variable: {final_log_file_path}")

    if final_log_file_path is not None:
        try:
            log_file_p = Path(final_log_file_path)
            # Create parent directories if they don't exist
            log_file_p.parent.mkdir(parents=True, exist_ok=True)

            # Create file if it doesn't exist and set permissions
            if not log_file_p.exists():
                logger.info(f"Log file does not exist. Creating log file at {final_log_file_path}")
                log_file_p.touch()
                # Setting permissions might fail if user doesn't have rights; handle this.
                try:
                    os.chmod(final_log_file_path, 0o664)  # Read/write for owner/group, read for others
                except OSError as e:
                    logger.warning(f"Could not set permissions for log file {final_log_file_path}: {e}")

            # Add FileHandler - check if one for this file already exists to avoid duplication
            # This is a simple check; a more robust way might involve naming handlers
            # or storing a reference to the file handler.
            if not any(
                isinstance(h, logging.FileHandler) and h.baseFilename == str(log_file_p.resolve())
                for h in logger.handlers
            ):
                file_handler = logging.FileHandler(final_log_file_path)
                # Consider adding a formatter to the file_handler if different from basicConfig
                # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                # file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
                logger.info(f"Successfully configured logging to file: {final_log_file_path}")
            else:
                logger.info(f"File handler for {final_log_file_path} already exists. Not adding duplicate.")

            _global_log_file = str(final_log_file_path)  # Store the absolute path

        except (OSError, IOError) as e:
            logger.error(f"Error setting up log file at {final_log_file_path}: {e}. Logging to console only.")
            _global_log_file = None  # Ensure it's None if file setup failed
    else:
        logger.info("No log file specified. Logging to console only.")
        _global_log_file = None


def get_log_file_path() -> Optional[str]:
    """
    Retrieves the path to the currently configured global log file, if any.

    Returns:
        Optional[str]: The absolute path to the log file, or None if file logging
                       is not configured or failed to initialize.
    """
    return _global_log_file


def get_logs() -> Optional[str]:
    """
    Reads and returns the content of the configured global log file.

    Returns:
        Optional[str]: The content of the log file as a string, or None if
                       no log file is configured or if the file cannot be read.
    """
    if _global_log_file is not None:
        try:
            with open(_global_log_file, "r", encoding="utf-8") as f:  # Specify encoding
                return f.read()
        except (IOError, OSError) as e:
            logger.error(f"Could not read log file {_global_log_file}: {e}")
            return None
    logger.info("No global log file configured to read from.")
    return None
