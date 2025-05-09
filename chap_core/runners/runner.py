from typing import Optional


class Runner:
    """
    Interface for runners that can execute generic commands (e.g., through Docker or CLI).

    Methods:
        - run_command(command): Run a given command (string).
        - store_file(file_path): Optional method to persist output or intermediate data.
        - teardown(): Cleanup logic after execution (e.g., remove Docker images).
    """

    def run_command(self, command: str):
        """
        Execute a command. Must be implemented by subclasses.
        """
        raise NotImplementedError

    def store_file(self, file_path: str):
        """
        Store or persist a file (optional in many runners).
        """
        raise NotImplementedError

    def teardown(self):
        """
        Cleanup any resources after execution (e.g., delete containers or images).
        """
        raise NotImplementedError


class TrainPredictRunner:
    """
    Interface for runners that execute ML workflows with 'train' and 'predict' steps.

    Methods:
        - train(): Execute a training command.
        - predict(): Execute a prediction command.
        - teardown(): Cleanup resources after both steps.
    """

    def train(self, train_data: str, model_file_name: str, polygons_file_name: Optional[str]):
        """
        Run the training process using the provided training data and output model file path.

        Args:
            train_data: Path to the input training dataset.
            model_file_name: Output path to save the trained model.
            polygons_file_name: Optional path to a GeoJSON or shapefile with polygon data.
        """
        raise NotImplementedError

    def predict(
        self,
        model_file_name: str,
        historic_data: str,
        future_data: str,
        output_file: str,
        polygons_file_name: Optional[str],
    ):
        """
        Run the prediction process using a trained model and input data.

        Args:
            model_file_name: Path to the trained model.
            historic_data: Path to historical observations.
            future_data: Path to future predictors.
            output_file: Path to store prediction results.
            polygons_file_name: Optional path to polygon file for spatial context.
        """
        raise NotImplementedError

    def teardown(self):
        """
        Optional cleanup after train/predict (e.g., removing Docker containers/images).
        """
        raise NotImplementedError
