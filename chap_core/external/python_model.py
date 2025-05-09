# Improvement Suggestions:
# 1. Comprehensive Docstrings: Add module, class, and method docstrings.
# 2. Clarify/Implement Unused Features: Document/implement `train`, `_lead_time`, `_adaptors`.
# 3. Update Legacy Type Hints: Replace `IsSpatioTemporalDataSet` with `DataSet`.
# 4. Robust External Script Interaction: Document script contract and ensure `run_command` handles errors.
# 5. Temporary File Management: Document assumptions about script's filename handling.

import logging
import os  # Added import for os.remove
import tempfile
from typing import Any, Dict, Optional, Type

from chap_core.time_period import Month

from ..datatypes import ClimateData, ClimateHealthTimeSeries, HealthData
from ..spatio_temporal_data.temporal_dataclass import DataSet

# Assuming run_command is defined in .external_model and handles execution/errors
from .external_model import run_command

logger = logging.getLogger(__name__)


class ExternalPythonModel:
    """
    A wrapper for executing an external Python script as a predictive model.

    This class facilitates running a standalone Python script that performs
    predictions. It handles data exchange with the script via temporary CSV files.
    The external script is expected to read input data filenames from command-line
    arguments and write its output predictions to a CSV file specified as a
    command-line argument.

    Attributes:
        script (str): Path to the external Python script to be executed.
        lead_time (Optional[Type[Month]]): Intended lead time for predictions.
                                            Currently not fully utilized in the provided methods.
        adaptors (Optional[Dict[str, Any]]): Configuration for data adaptors.
                                              Currently not utilized.
    """

    def __init__(
        self,
        script: str,
        lead_time: Optional[Type[Month]] = Month,  # Or consider PeriodType
        adaptors: Optional[Dict[str, Any]] = None,
    ):
        """
        Initializes the ExternalPythonModel.

        Args:
            script: The file path of the Python script to execute.
            lead_time: The lead time for predictions (e.g., chap_core.time_period.Month).
                       Its specific usage needs further definition.
            adaptors: Optional dictionary for data adaptors.
                      Its specific usage needs further definition.
        """
        self._script = script
        self._lead_time = lead_time  # Purpose and usage to be clarified
        self._adaptors = adaptors  # Purpose and usage to be clarified

    def train(self, training_data: DataSet[Any], model_path: str, **kwargs: Any) -> None:
        """
        Trains the external Python model.

        Note: This method is currently a placeholder and not implemented.
        The external script would need a corresponding training entry point.

        Args:
            training_data: The dataset to train the model on.
            model_path: Path where the trained model should be saved by the script.
            **kwargs: Additional keyword arguments for training.

        Raises:
            NotImplementedError: This method is not yet implemented.
        """
        logger.warning(
            f"Training is not implemented for ExternalPythonModel with script {self._script}. "
            "The external script would need to handle its own training and model persistence."
        )
        raise NotImplementedError("Training for ExternalPythonModel is not implemented.")

    def get_predictions(
        self,
        train_data: DataSet[ClimateHealthTimeSeries],
        future_climate_data: DataSet[ClimateData],
    ) -> DataSet[HealthData]:
        """
        Executes the external Python script to get predictions.

        The method creates temporary CSV files for training data and future climate
        data. It then calls the external Python script, passing these filenames
        and an output filename as command-line arguments. The script is expected
        to write its predictions to the specified output CSV file.

        Expected external script signature (approximate):
        `python your_script.py <train_data_csv_path> <future_climate_csv_path> <output_predictions_csv_path>`

        Args:
            train_data: Historical training data (climate and health).
            future_climate_data: Future climate data for the prediction horizon.

        Returns:
            A DataSet containing the health predictions from the external script.

        Raises:
            FileNotFoundError: If the output file from the script is not found.
            Exception: If `run_command` raises an exception due to script errors.
        """
        # Using delete=False for NamedTemporaryFile to ensure files exist until explicitly closed,
        # especially important if the external script opens them by name.
        # Suffix can help identify these files if they are not cleaned up.
        train_data_file = tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix="_train.csv", encoding="utf-8")
        future_climate_data_file = tempfile.NamedTemporaryFile(
            mode="w+", delete=False, suffix="_future_climate.csv", encoding="utf-8"
        )
        output_file = tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix="_output.csv", encoding="utf-8")

        try:
            train_data.to_csv(train_data_file.name)
            train_data_file.flush()  # Ensure data is written before script access
            future_climate_data.to_csv(future_climate_data_file.name)
            future_climate_data_file.flush()

            command = (
                f"python {self._script} {train_data_file.name} " f"{future_climate_data_file.name} {output_file.name}"
            )
            logger.info(f"Executing external Python model command: {command}")
            # run_command should handle stdout/stderr logging and raise on error
            run_command(command)

            # Ensure output file has content before trying to read
            output_file.seek(0)  # Rewind to read from the beginning
            if not output_file.read():
                logger.error(f"Output file {output_file.name} from script {self._script} is empty.")
                # Consider raising a custom exception here
                raise ValueError(f"External script {self._script} produced an empty output file.")
            output_file.seek(0)  # Rewind again for DataSet.from_csv

            results = DataSet.from_csv(output_file.name, HealthData)
            logger.info(f"Successfully loaded predictions from {output_file.name}")

        except Exception as e:
            logger.error(f"Error during external Python model execution or data handling: {e}")
            raise  # Re-raise the exception after logging
        finally:
            # Ensure temporary files are closed and deleted
            train_data_file.close()
            os.remove(train_data_file.name)
            future_climate_data_file.close()
            os.remove(future_climate_data_file.name)
            output_file.close()
            os.remove(output_file.name)

        return results
