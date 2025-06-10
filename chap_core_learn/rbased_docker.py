# Improvement Suggestions:
# 1. **Dockerfile Path Bug**: Ensure the correct Dockerfile path is used for cleanup. The Dockerfile is created in a temporary directory, but `os.remove` targets "./Dockerfile". This should be fixed to remove the temporary Dockerfile.
# 2. **Specific Docker Exception Handling**: Catch more specific exceptions from `client.images.build` (e.g., `docker.errors.BuildError`, `docker.errors.APIError`) instead of a generic `Exception` for more informative error messages and handling.
# 3. **Explicit Temporary Directory Cleanup**: While `tempfile.TemporaryDirectory` cleans up on garbage collection, explicitly calling `folder.cleanup()` in a `finally` block ensures timely removal of the temporary directory, especially if errors occur.
# 4. **Base Image Documentation**: Document the base Docker image (`rocker/r-base:latest`) used in the generated Dockerfile, including potential implications like the R version and underlying OS.
# 5. **Security of R Package Names**: If the `r_packages` list can be influenced by untrusted external input, there's a minor risk of command injection into the `RUN R -e "..."` line. While `install.packages` is generally robust, this should be noted if inputs are not controlled.

"""
This module provides functionality to dynamically create Docker images
pre-configured with R and a specified list of R packages.

It generates a Dockerfile on the fly using 'rocker/r-base:latest' as the base,
installs system dependencies, and then installs the requested R packages via CRAN.
The resulting image can be used for R-based computations in a containerized environment.
"""

import logging  # Added logging
import tempfile
from pathlib import Path  # Added Path
from typing import List  # Added List

import docker

logger = logging.getLogger(__name__)


def create_image(r_packages: List[str], image_name: str = "r-custom-image") -> None:
    """
    Dynamically creates a Docker image with R installed and a specified list of R packages.

    The function generates a Dockerfile using 'rocker/r-base:latest', adds commands
    to install necessary system dependencies and the provided R packages from CRAN.
    It then builds the Docker image using the Docker SDK.

    Args:
        r_packages (List[str]): A list of R package names to install (e.g., ['dplyr', 'fable']).
        image_name (str): The name (and optionally tag) for the Docker image to be created.
                          Defaults to "r-custom-image".

    Raises:
        docker.errors.BuildError: If the Docker image build process fails.
        docker.errors.APIError: For other errors from the Docker API during image build.
        FileNotFoundError: If the temporary Dockerfile cannot be created or accessed.
        Exception: For other unexpected errors during the process.
    """
    if not r_packages:
        logger.warning("No R packages specified. Building a base R image without additional packages.")
        # Proceed, but package_string will be empty, install.packages(c()) is valid.

    package_string = ", ".join([f"'{pkg}'" for pkg in r_packages])

    # Using rocker/r-base:latest provides a recent version of R on a Debian-based OS.
    # System dependencies are included for common R package compilation needs.
    dockerfile_content = f"""
    FROM rocker/r-base:latest

    # Install required system dependencies for R package compilation
    # APT::Install-Suggests and APT::Install-Recommends "0" to minimize image size.
    RUN echo 'APT::Install-Suggests "0";' >> /etc/apt/apt.conf.d/00-docker && \\
        echo 'APT::Install-Recommends "0";' >> /etc/apt/apt.conf.d/00-docker && \\
        DEBIAN_FRONTEND=noninteractive apt-get update && \\
        apt-get install -y --no-install-recommends \\
            libudunits2-dev \\
            libgdal-dev \\
            libssl-dev \\
            libfontconfig1-dev \\
            libgsl-dev && \\
        apt-get clean && \\
        rm -rf /var/lib/apt/lists/*

    # Install specified R packages from CRAN
    RUN R -e "install.packages(c({package_string}), repos='http://cran.r-project.org')"
    """

    # Use TemporaryDirectory for automatic cleanup of the directory and its contents
    with tempfile.TemporaryDirectory() as temp_dir_name:
        temp_dockerfile_path = Path(temp_dir_name) / "Dockerfile"

        try:
            with open(temp_dockerfile_path, "w", encoding="utf-8") as dockerfile:
                dockerfile.write(dockerfile_content)
            logger.info(f"Temporary Dockerfile created at: {temp_dockerfile_path}")
        except IOError as e:
            logger.error(f"Failed to write temporary Dockerfile: {e}")
            raise FileNotFoundError(f"Could not write temporary Dockerfile: {e}")

        client = docker.from_env()

        try:
            logger.info(f"Building Docker image: '{image_name}' from path: '{temp_dir_name}'...")
            # The 'path' argument for build should be the directory containing the Dockerfile.
            image, logs = client.images.build(path=temp_dir_name, tag=image_name, rm=True)

            for log_entry in logs:
                if "stream" in log_entry:
                    log_line = log_entry["stream"].strip()
                    if log_line:  # Avoid printing empty log lines
                        logger.debug(f"Docker build log: {log_line}")  # Use logger.debug for verbose logs
                elif "errorDetail" in log_entry:
                    error_msg = log_entry["errorDetail"]["message"]
                    logger.error(f"Docker build error: {error_msg}")
                    # BuildError might be raised by SDK, but if not, raise it from errorDetail
                    raise docker.errors.BuildError(error_msg, build_log=logs)
                else:  # Other types of log entries
                    logger.debug(f"Docker build event: {log_entry}")

            logger.info(f"Docker image '{image_name}' (ID: {image.id}) created successfully.")

        except docker.errors.BuildError as e:
            logger.error(f"Error building Docker image '{image_name}': {e}")
            # Log detailed build failure logs if possible
            if hasattr(e, "build_log"):
                for log_item in e.build_log:
                    if "stream" in log_item:
                        logger.error(f"Build log detail: {log_item['stream'].strip()}")
            raise
        except docker.errors.APIError as e:
            logger.error(f"Docker API error while building image '{image_name}': {e}")
            raise
        except Exception as e:  # Catch-all for other unexpected issues
            logger.error(f"An unexpected error occurred while building Docker image '{image_name}': {e}")
            raise
        # No need for explicit os.remove or folder.cleanup() due to TemporaryDirectory context manager


if __name__ == "__main__":
    # Configure basic logging for the example script execution
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # Example usage:
    target_image_name = "r-dplyr-fable-test"
    logger.info(f"Attempting to create Docker image: {target_image_name}")
    try:
        create_image(["dplyr", "fable"], image_name=target_image_name)
        logger.info(f"Successfully completed create_image call for {target_image_name}.")
        # Here you could add code to verify the image, e.g., client.images.get(target_image_name)
    except Exception as e:
        logger.error(f"Failed to create image {target_image_name} in __main__: {e}")
