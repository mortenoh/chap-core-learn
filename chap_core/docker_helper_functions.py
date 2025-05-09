# Improvement Suggestions:
# 1. **Error Handling in `docker_image_from_fo`**: The function should parse the Docker build response to explicitly check for build success or failure and raise an appropriate exception (e.g., `docker.errors.BuildError`) on failure, rather than just printing stream output.
# 2. **Security of `command` in `run_command_through_docker_container`**: If the `command` string can originate from untrusted input, it poses a command injection risk. Document this potential vulnerability or, if applicable, implement sanitization or validation for the command string.
# 3. **Resource Management for Docker Client**: While `docker.from_env()` is convenient, for applications making many Docker calls or in long-running services, explicitly managing the Docker client lifecycle (e.g., `client.close()`) could be beneficial, though often handled by the SDK.
# 4. **Clarity on `working_directory` Mounting**: The `run_command_through_docker_container` mounts the `working_directory` to a fixed `/home/run` inside the container. Clarify if this mount point is configurable or why this specific path is chosen (e.g., convention, user permissions within the image).
# 5. **Structured Output for `run_command_through_docker_container`**: The function currently returns `log_output` (which seems to be combined stdout/stderr from `container.logs()`). Consider returning a more structured object (e.g., a dataclass or tuple) containing `stdout`, `stderr`, and `exit_code` separately for easier programmatic access and error handling.

"""
This module provides helper functions for interacting with Docker,
facilitating the creation of Docker images and the execution of commands
within Docker containers. It uses the `docker` Python SDK.
"""

import logging
from pathlib import Path
from typing import IO  # For fileobject type hint

import docker
from docker.errors import APIError, BuildError  # For more specific error handling

logger = logging.getLogger(__name__)


def create_docker_image(dockerfile_directory: Path | str) -> str:
    """
    Build a Docker image from a Dockerfile located in the specified directory.

    The name of the Docker image is inferred from the name of the directory
    containing the Dockerfile (e.g., a directory named 'myimage' will result
    in an image tagged 'myimage').

    Args:
        dockerfile_directory (Path | str): The path to the directory that
                                           contains the Dockerfile.

    Returns:
        str: The name of the built Docker image.

    Raises:
        FileNotFoundError: If the Dockerfile is not found in the specified directory.
        docker.errors.BuildError: If the Docker image build fails.
        docker.errors.APIError: For other errors from the Docker API.
    """
    dockerfile_directory = Path(dockerfile_directory)
    image_name = dockerfile_directory.stem  # Uses directory name as image name
    dockerfile_path = dockerfile_directory / "Dockerfile"

    logger.info(f"Attempting to create Docker image '{image_name}' from Dockerfile at {dockerfile_path}")

    if not dockerfile_path.exists():
        logger.error(f"Dockerfile not found at: {dockerfile_path}")
        raise FileNotFoundError(f"Dockerfile not found at: {dockerfile_path}")

    try:
        with open(dockerfile_path, "rb") as fileobj:  # Open in binary mode for Docker API
            return docker_image_from_fo(fileobj, image_name)
    except (BuildError, APIError) as e:
        logger.error(f"Failed to build Docker image '{image_name}': {e}")
        raise
    except Exception as e:  # Catch other potential errors
        logger.error(f"An unexpected error occurred during Docker image creation for '{image_name}': {e}")
        raise


def docker_image_from_fo(fileobject: IO[bytes], name: str) -> str:
    """
    Build a Docker image from a Dockerfile provided as a file-like object.

    Args:
        fileobject (IO[bytes]): A file-like object (opened in binary mode)
                                containing the Dockerfile content.
        name (str): The name (and optionally tag) for the Docker image
                    (e.g., 'myimage:latest').

    Returns:
        str: The name of the built Docker image if successful.

    Raises:
        docker.errors.BuildError: If the Docker image build process fails.
        docker.errors.APIError: For other errors encountered while communicating
                                with the Docker API.
    """
    client = docker.from_env()
    logger.info(f"Building Docker image '{name}' from file object.")

    last_event = None
    try:
        # The `build` method streams logs; iterate to process them and check for errors.
        response_stream = client.api.build(
            fileobj=fileobject, tag=name, decode=True, rm=True
        )  # rm=True removes intermediate containers

        for event in response_stream:
            if "stream" in event:
                log_line = event["stream"].strip()
                if log_line:  # Avoid printing empty lines
                    print(log_line)  # Or use logger.debug(log_line)
            elif "errorDetail" in event:
                error_message = event["errorDetail"]["message"]
                logger.error(f"Docker build error for image '{name}': {error_message}")
                raise BuildError(error_message, build_log=response_stream)  # Pass full log if available
            last_event = event  # Keep track of the last event, might contain image ID

        # After stream, check if the image was built successfully.
        # A more robust check might involve inspecting client.images.get(name)
        # but BuildError should be raised by the stream if there's an issue.
        logger.info(f"Successfully built Docker image '{name}'.")
        return name

    except BuildError:  # Re-raise if already a BuildError
        raise
    except APIError as e:
        logger.error(f"Docker API error while building image '{name}': {e}")
        raise
    except Exception as e:  # Catch other unexpected errors
        logger.error(f"An unexpected error occurred while building image '{name}' from file object: {e}")
        raise


def run_command_through_docker_container(
    docker_image_name: str, working_directory: str, command: str, remove_after_run: bool = False
) -> str:  # Consider returning a more structured result (stdout, stderr, exit_code)
    """
    Run a shell command inside a new Docker container based on a specified image.
    The specified local `working_directory` is mounted to `/home/run` inside the container.

    Args:
        docker_image_name (str): The name of the Docker image to use for the container.
        working_directory (str): The local directory path to be mounted as the
                                 container's working directory (`/home/run`).
        command (str): The shell command to execute inside the container.
                       WARNING: Ensure this command is from a trusted source or properly
                       sanitized to prevent command injection vulnerabilities.
        remove_after_run (bool): If True, the container will be automatically removed
                                 after execution. Defaults to False.

    Returns:
        str: The combined stdout and stderr logs captured from the container run.

    Raises:
        FileNotFoundError: If the specified `working_directory` does not exist.
        docker.errors.ContainerError: If the command in the container exits with a non-zero status.
        docker.errors.ImageNotFound: If the specified Docker image does not exist.
        docker.errors.APIError: For other errors from the Docker API.
        AssertionError: If the command fails (non-zero exit code) - this is the original behavior.
                        Consider replacing with specific Docker exceptions.
    """
    client = docker.from_env()

    try:
        working_dir_full_path = Path(working_directory).resolve(strict=True)
    except FileNotFoundError:
        logger.error(f"Working directory not found: {working_directory}")
        logger.error(f"Current process working directory is: {Path.cwd()}")
        raise

    logger.info(f"Preparing to run command in Docker container based on image: '{docker_image_name}'")
    logger.info(f"Mounting local directory '{working_dir_full_path}' to '/home/run' in container.")
    logger.info(f"Executing command in container: {command}")

    container = None
    try:
        container = client.containers.run(
            image=docker_image_name,
            command=command,  # Could be list of strings or a single string
            volumes={str(working_dir_full_path): {"bind": "/home/run", "mode": "rw"}},
            working_dir="/home/run",  # Sets the default directory for the command
            detach=True,  # Detach to manage logs and exit code manually
            auto_remove=False,  # Manage removal manually to ensure logs can be fetched
        )

        # Stream logs (optional, good for long commands)
        # for log_line in container.logs(stream=True, stdout=True, stderr=True, follow=True, decode=True):
        #     print(log_line, end="")

        result = container.wait(timeout=None)  # Wait indefinitely for completion
        exit_code = result.get("StatusCode", -1)  # Default to -1 if StatusCode not found

        # Fetch logs after completion
        stdout_logs = container.logs(stdout=True, stderr=False).decode("utf-8")
        stderr_logs = container.logs(stdout=False, stderr=True).decode("utf-8")

        if stdout_logs:
            logger.info(f"Container STDOUT:\n{stdout_logs.strip()}")
        if stderr_logs:
            logger.warning(f"Container STDERR:\n{stderr_logs.strip()}")

        combined_logs = stdout_logs + stderr_logs  # Or keep separate

        if exit_code != 0:
            error_message = (
                f"Command '{command}' in Docker image '{docker_image_name}' failed with exit code {exit_code}."
            )
            logger.error(error_message)
            # Consider raising ContainerError here for more specific Docker error handling
            # raise docker.errors.ContainerError(container, exit_code, command, docker_image_name, stderr_logs)
            assert exit_code == 0, f"{error_message}\nLogs:\n{combined_logs.strip()}"  # Original assertion

        logger.info(f"Command executed successfully in container. Exit code: {exit_code}")
        return combined_logs.strip()  # Return combined logs

    except docker.errors.ImageNotFound:
        logger.error(f"Docker image '{docker_image_name}' not found.")
        raise
    except docker.errors.APIError as e:
        logger.error(f"Docker API error during container run: {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred running command in Docker: {e}")
        raise
    finally:
        if container:
            if remove_after_run or (exit_code == 0 and remove_after_run):  # Ensure removal if requested
                try:
                    container.remove()
                    logger.info(f"Container '{container.id[:12]}' removed.")
                except docker.errors.APIError as e:
                    logger.warning(f"Failed to remove container '{container.id[:12]}': {e}")
            elif exit_code != 0 and not remove_after_run:
                logger.warning(
                    f"Container '{container.id[:12]}' was not removed due to failure (remove_after_run=False)."
                )
