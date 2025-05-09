import logging

import mlflow.exceptions
import mlflow.projects
from mlflow.utils.process import ShellCommandException

from chap_core.exceptions import ModelFailedException
from chap_core.runners.runner import TrainPredictRunner

logger = logging.getLogger(__name__)


class MlFlowTrainPredictRunner(TrainPredictRunner):
    """
    A TrainPredictRunner that executes MLflow projects using the 'train' and 'predict' entry points.

    This runner wraps around the `mlflow.projects.run()` command and provides error handling and logging.
    """

    def __init__(self, model_path):
        """
        Args:
            model_path (Path | str): Path to the directory containing the MLproject file.
        """
        self.model_path = model_path

    def train(self, train_file_name, model_file_name, polygons_file_name=None):
        """
        Executes the 'train' entry point of an MLflow project.

        Args:
            train_file_name: Path to training data file.
            model_file_name: Path to save the trained model.
            polygons_file_name: Ignored in this runner.

        Returns:
            Completed MLflow run result object.

        Raises:
            ModelFailedException: If the MLflow subprocess or execution fails.
        """
        logger.info("Training model using MLflow")

        try:
            return mlflow.projects.run(
                uri=str(self.model_path),
                entry_point="train",
                parameters={
                    "train_data": str(train_file_name),
                    "model": str(model_file_name),
                },
                build_image=True,  # Ensures Docker is used if specified in MLproject
            )
        except ShellCommandException as e:
            logger.error(
                "Error running MLflow project. This may be caused by a missing pyenv or environment misconfiguration.\n"
                "See: https://github.com/pyenv/pyenv#installation"
            )
            raise ModelFailedException(str(e)) from e

        except mlflow.exceptions.ExecutionException as e:
            logger.error("Execution of the MLflow project failed. Check the logs above for more information.")
            raise ModelFailedException(str(e)) from e

    def predict(self, model_file_name, historic_data, future_data, output_file, polygons_file_name=None):
        """
        Executes the 'predict' entry point of an MLflow project.

        Args:
            model_file_name: Path to trained model.
            historic_data: Input file path for historical data.
            future_data: Input file path for future data.
            output_file: Path where prediction results should be saved.
            polygons_file_name: Ignored in this runner.

        Returns:
            Completed MLflow run result object.
        """
        return mlflow.projects.run(
            uri=str(self.model_path),
            entry_point="predict",
            parameters={
                "historic_data": str(historic_data),
                "future_data": str(future_data),
                "model": str(model_file_name),
                "out_file": str(output_file),
            },
        )
