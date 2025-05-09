import logging
from pathlib import Path

import docker  # Docker Python SDK

from chap_core.runners.command_line_runner import CommandLineTrainPredictRunner

from ..docker_helper_functions import create_docker_image, run_command_through_docker_container
from .runner import Runner

logger = logging.getLogger(__name__)


class DockerImageRunner(Runner):
    """
    Runner that builds and runs commands inside a Docker container using a Dockerfile.

    This is useful for models or workflows that need to be containerized from source.
    """

    def __init__(self, docker_file_path: str, working_dir: str | Path):
        self._docker_file_path = Path(working_dir) / docker_file_path  # Path to Dockerfile
        self._docker_name = None  # Will hold the built image name
        self._working_dir = Path(working_dir)
        self._is_setup = False  # Flag to ensure image is built only once

    def setup(self):
        """
        Build the Docker image using the Dockerfile.
        """
        if self._is_setup:
            return
        self._docker_name = create_docker_image(self._docker_file_path)
        self._is_setup = True

    def run_command(self, command: str):
        """
        Run a command inside the Docker container.

        The image is built (if not already) using the Dockerfile.
        """
        self.setup()
        return run_command_through_docker_container(self._docker_name, self._working_dir, command)


class DockerRunner(Runner):
    """
    Runner that uses an existing Docker image (e.g., from DockerHub or pre-built).

    It assumes the image is already built or pulled, and runs commands inside it.
    """

    def __init__(self, docker_name: str, working_dir: str | Path):
        self._docker_name = docker_name  # Image name/tag
        self._working_dir = Path(working_dir)

    def run_command(self, command: str):
        """
        Run a shell command inside the specified Docker container.
        """
        logger.info(f"Running command: {command} in docker container: {self._docker_name} at {self._working_dir}")
        return run_command_through_docker_container(self._docker_name, self._working_dir, command)

    def teardown(self):
        """
        Remove the Docker image after use (forcefully).
        """
        client = docker.from_env()
        client.images.remove(self._docker_name, force=True)


class DockerTrainPredictRunner(CommandLineTrainPredictRunner):
    """
    Drop-in replacement for CommandLineTrainPredictRunner, but uses DockerRunner internally.

    Used to encapsulate `train` and `predict` commands inside a containerized environment.
    """

    def __init__(self, runner: DockerRunner, train_command: str, predict_command: str):
        # Leverages base class logic but replaces the command-line backend with DockerRunner
        super().__init__(runner, train_command, predict_command)

    def teardown(self):
        """
        Call the underlying DockerRunner's teardown method to clean up the image.
        """
        self._runner.teardown()
