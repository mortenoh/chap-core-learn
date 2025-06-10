import logging
import subprocess
from pathlib import Path

from chap_core.exceptions import CommandLineException, ModelConfigurationException
from chap_core.runners.runner import Runner, TrainPredictRunner

logger = logging.getLogger(__name__)


class CommandLineRunner(Runner):
    """
    Generic runner for executing shell commands inside a given working directory.
    """

    def __init__(self, working_dir: str | Path):
        self._working_dir = Path(working_dir)

    def run_command(self, command: str):
        """
        Run a shell command using subprocess and return its output.
        """
        return run_command(command, self._working_dir)

    def store_file(self):
        """
        Stub method for storing outputs, if needed by derived implementations.
        """
        pass


def run_command(command: str, working_directory: Path = Path(".")) -> str:
    """
    Runs a UNIX shell command and handles errors.

    Args:
        command: Shell command to run (as a string).
        working_directory: Path to run the command from.

    Returns:
        Combined stdout and stderr as a string.

    Raises:
        CommandLineException: If the command exits with non-zero code.
    """
    logging.info(f"Running command: {command}")

    try:
        # Run the command in the shell
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=working_directory,
            shell=True,
        )

        # Capture both stdout and stderr
        stdout, stderr = process.communicate()
        output = stdout.decode() + "\n" + stderr.decode()

        return_code = process.returncode

        if return_code != 0:
            logger.error(f"Command '{command}' failed with return code {return_code}, " f"Output: {output.strip()}")
            raise CommandLineException(
                f"Command '{command}' failed with return code {return_code}.\n"
                f"Full output:\n-----\n{output.strip()}\n-----"
            )

    except subprocess.CalledProcessError as e:
        # Handle unexpected subprocess failures
        error = e.output.decode()
        logger.error(error)
        raise e

    return output


class CommandLineTrainPredictRunner(TrainPredictRunner):
    """
    High-level wrapper for executing `train` and `predict` operations via command line.
    """

    def __init__(self, runner: CommandLineRunner, train_command: str, predict_command: str):
        self._runner = runner
        self._train_command = train_command
        self._predict_command = predict_command

    def _format_command(self, command: str, keys: dict) -> str:
        """
        Apply `.format()` substitution on command templates using a key dictionary.
        Raises informative errors if placeholders are missing.
        """
        try:
            return command.format(**keys)
        except KeyError as e:
            raise ModelConfigurationException(
                f"Unable to format command '{command}'. " f"Missing or incorrect placeholder key: {e}"
            ) from e

    def _handle_polygons(self, command: str, keys: dict, polygons_file_name: str | None = None) -> dict:
        """
        Inserts polygons path into keys if expected by the command.
        Warns if polygons are passed but not referenced.
        """
        if polygons_file_name is not None:
            if "{polygons}" not in command:
                logger.warning(
                    f"Polygons file provided, but command '{command}' does not include a {{polygons}} placeholder."
                )
            else:
                keys["polygons"] = polygons_file_name
        return keys

    def train(self, train_file_name: str, model_file_name: str, polygons_file_name: str | None = None) -> str:
        """
        Run the training command, optionally with polygons.
        """
        keys = {
            "train_data": train_file_name,
            "model": model_file_name,
        }
        keys = self._handle_polygons(self._train_command, keys, polygons_file_name)
        command = self._format_command(self._train_command, keys)
        return self._runner.run_command(command)

    def predict(
        self,
        model_file_name: str,
        historic_data: str,
        future_data: str,
        output_file: str,
        polygons_file_name: str | None = None,
    ) -> str:
        """
        Run the prediction command, optionally with polygons.
        """
        keys = {
            "historic_data": historic_data,
            "future_data": future_data,
            "model": model_file_name,
            "out_file": output_file,
        }
        keys = self._handle_polygons(self._predict_command, keys, polygons_file_name)
        command = self._format_command(self._predict_command, keys)
        return self._runner.run_command(command)
