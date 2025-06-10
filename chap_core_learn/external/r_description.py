# Improvement Suggestions:
# 1. Comprehensive Docstrings: Add module and function docstrings.
# 2. Type Hinting: Add type hints for parameters and return values.
# 3. Robust Error Handling: Add try-except blocks for file operations and parsing.
# 4. Document Dummy Section Trick: Explain the workaround in `parse_description_file`.
# 5. Improve Readability of `get_imports`: Add comments or refactor the list comprehension.

import configparser
import logging
import os
from typing import Dict, List

logger = logging.getLogger(__name__)


def parse_description_file(file_path: os.PathLike | str) -> Dict[str, str]:
    """
    Parses an R-style DESCRIPTION file into a dictionary.

    R DESCRIPTION files are typically in a DCF (Debian Control File) format,
    which is similar to INI files but without section headers for the main content.
    This function uses a workaround by prepending a dummy section header
    to allow `configparser` to parse it.

    Args:
        file_path: The path to the DESCRIPTION file.

    Returns:
        A dictionary where keys are field names (e.g., 'Package', 'Version')
        and values are their corresponding string values.

    Raises:
        FileNotFoundError: If the specified file_path does not exist.
        configparser.Error: If there's an issue parsing the file content.
        Exception: For other potential I/O errors.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
    except FileNotFoundError:
        logger.error(f"DESCRIPTION file not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error reading DESCRIPTION file {file_path}: {e}")
        raise

    # Add a dummy section header to make it parsable by configparser
    content_with_header = "[dummy_section]\n" + content

    config = configparser.ConfigParser()
    try:
        config.read_string(content_with_header)
    except configparser.Error as e:
        logger.error(f"Error parsing DESCRIPTION file content from {file_path}: {e}")
        raise

    if "dummy_section" not in config:
        # This might happen if the file is empty or has unexpected structure
        logger.warning(f"Could not find dummy_section in parsed DESCRIPTION file: {file_path}")
        return {}

    description_data = dict(config["dummy_section"])

    # Remove any leading or trailing whitespace from keys and values
    description_data = {k.strip(): v.strip() for k, v in description_data.items()}

    return description_data


def get_imports(file_path: os.PathLike | str) -> List[str]:
    """
    Extracts package dependencies from an R DESCRIPTION file.

    It parses the DESCRIPTION file and looks for 'Imports' and 'Depends' fields,
    which typically list required R packages. Package names are often
    comma-separated and may include version specifications (which are currently
    not parsed out by this function, only the names are extracted).

    Args:
        file_path: The path to the DESCRIPTION file.

    Returns:
        A list of unique package names extracted from 'Imports' and 'Depends' fields.
        Returns an empty list if parsing fails or fields are not found.
    """
    try:
        description_data = parse_description_file(file_path)
    except Exception:  # Catching broad exception as parse_description_file already logs
        return []

    # R DESCRIPTION files list dependencies in 'Imports' and 'Depends' fields.
    # Packages can be comma-separated and might include version info in parentheses,
    # e.g., "R (>= 3.5.0), Rcpp (>= 0.12.0)"
    # This comprehension extracts the package names, stripping whitespace.
    # It does not currently parse out version specifiers.
    dependencies = set()
    for key in ["imports", "depends"]:  # Case-insensitive keys are handled by configparser
        field_value = description_data.get(key.lower(), "")  # configparser keys are lowercased
        if field_value:
            # Split by comma, then strip each part. Also handle cases where a package might have version info.
            # A more robust parser would handle versions and comments correctly.
            pkgs = [p.split("(")[0].strip() for p in field_value.split(",") if p.strip()]
            dependencies.update(p for p in pkgs if p)  # Add non-empty, stripped package names

    return sorted(list(dependencies))
