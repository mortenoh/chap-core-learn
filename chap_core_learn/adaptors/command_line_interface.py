# Improvement Suggestions:
# 1. **Comprehensive Docstrings**: Add a module-level docstring and detailed docstrings for the main generator functions (`generate_app`, `generate_template_app`). Review and correct the docstrings of the inner `train` and `predict` CLI command functions, ensuring parameter descriptions are accurate and complete. (Primary task).
# 2. **Robust Error Handling**: Implement error handling for file I/O operations (e.g., `DataSet.from_csv`, model loading/saving, config parsing) using `try-except` blocks to catch `FileNotFoundError`, `pd.errors.ParserError`, `yaml.YAMLError`, and other potential exceptions, providing informative error messages to the CLI user.
# 3. **Configuration Handling in `generate_template_app`**:
#    - Clarify the role and necessity of `model_config_path` in both `train` and `predict` commands of `generate_template_app`. If a trained model (predictor) is self-contained, the config might not be needed for prediction.
#    - Handle cases where `model_config_path` might be `None` or invalid when `parse_file` is called.
# 4. **Dynamic Dataclass and Feature Determination**: The dynamic creation of dataclasses and determination of `covariate_names` (related to the `TODO` comment) should be made more robust and clearly documented. Ensure that the assumptions about how `estimator.covariate_names` are obtained are reliable.
# 5. **Review Commented-Out Code**: Evaluate the large block of commented-out code at the end of `generate_template_app`. If it represents an obsolete approach or an unfinished feature not currently planned, it should be removed to improve code clarity. If it's relevant for future work, it should be documented as such or moved to an issue tracker.

"""
This module provides functions to dynamically generate command-line interfaces (CLIs)
for CHAP-core models using the `cyclopts` library.

It allows creating `train` and `predict` commands for a given model estimator
or a `ModelTemplate`, enabling interaction with models via the command line.
This is useful for running models in environments without a full API deployment
or for scripting and automation purposes.
"""

import dataclasses  # Added import
import logging
from pathlib import Path  # Added for type hinting file paths
from typing import Any, Optional, Tuple, Type  # Added Optional

import pandas as pd  # For potential pandas errors
from cyclopts import App

from chap_core.datatypes import create_tsdataclass, remove_field
from chap_core.model_spec import get_dataclass
from chap_core.models import ModelTemplate  # Assuming this is chap_core.models.model_template.ModelTemplate
from chap_core.models.model_template_interface import ModelPredictor  # For type hint
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet

logger = logging.getLogger(__name__)


# TODO: Consider if 'estimator' should be an instance or a class type.
# If it's a class, train would instantiate it. If instance, it's pre-configured.
# Current usage `estimator.train(dataset)` implies it's an instance or has static/classmethod train.
# However, `get_dataclass(estimator)` implies estimator is a class. This needs clarification.
# Assuming 'estimator' here refers to an uninitialized model class for now.
def generate_app(estimator_class: Type[Any]) -> App:  # Renamed estimator to estimator_class
    """
    Generates a `cyclopts` CLI application with 'train' and 'predict' commands
    for a given model estimator class.

    The estimator class is expected to have `train` and `load_predictor` methods,
    and its `train` method's first argument's type hint is used to determine
    the input data structure.

    Args:
        estimator_class (Type[Any]): The model estimator class for which to generate the CLI.
                                     This class should conform to a defined interface
                                     (e.g., having `train` and `load_predictor` methods).

    Returns:
        App: A `cyclopts.App` instance with configured 'train' and 'predict' commands.
    """
    app = App()
    # Determine the expected input data type (dataclass) for the estimator's train method
    # This relies on type hints being correctly set on the estimator's train method.
    dc = get_dataclass(estimator_class)
    if dc is None:
        # Fallback or raise error if dataclass cannot be determined.
        # This could happen if type hints are missing or estimator_class is not as expected.
        logger.warning(
            f"Could not determine input dataclass for {estimator_class.__name__}. CLI might not work correctly."
        )
        # As a fallback, one might try to use a generic TimeSeriesData or raise an error.
        # For now, proceeding with dc as None might lead to errors later.
        # A better approach would be to raise a configuration error here.
        raise ValueError(
            f"Input data structure (dataclass) could not be inferred for estimator {estimator_class.__name__}."
        )

    @app.command()
    def train(training_data_filename: Path, model_path: Path) -> None:
        """
        Train the model using historic data provided in a CSV file.

        The trained model (predictor) is saved to the specified `model_path`.

        Parameters
        ----------
        training_data_filename : Path
            The path to the CSV file containing training data.
            The CSV format should be compatible with `DataSet.from_csv` and the model's expected dataclass.
        model_path : Path
            The path where the trained model predictor will be saved.
        """
        logger.info(f"CLI: Training model {estimator_class.__name__}...")
        logger.info(
            f"Loading training data from '{training_data_filename}' as type '{dc.__name__ if dc else 'UnknownDC'}'."
        )
        try:
            dataset = DataSet.from_csv(str(training_data_filename), dc)  # DataSet.from_csv expects str
            # Instantiate the estimator if estimator_class is a type
            estimator_instance = estimator_class()
            predictor = estimator_instance.train(dataset)
            predictor.save(str(model_path))  # save expects str
            logger.info(f"Training complete. Model saved to '{model_path}'.")
        except FileNotFoundError:
            logger.error(f"Training data file not found: {training_data_filename}")
        except (
            pd.errors.ParserError,
            ValueError,
        ) as e:  # Catch pandas parsing errors or other value errors from from_csv
            logger.error(f"Error loading or parsing training data from {training_data_filename}: {e}")
        except Exception as e:  # Catch-all for other training or saving errors
            logger.error(f"An error occurred during training or saving model: {e}", exc_info=True)

    @app.command()
    def predict(
        model_filename: Path, historic_data_filename: Path, future_data_filename: Path, output_filename: Path
    ) -> None:
        """
        Predict using a previously trained model.

        Loads a trained model predictor, historic data, and future predictor data from CSV files,
        generates forecasts, and saves them to an output CSV file.

        Parameters
        ----------
        model_filename : Path
            The path to the saved model predictor file (trained using the 'train' command).
        historic_data_filename : Path
            The path to the CSV file containing historic data (real data up to prediction start).
        future_data_filename : Path
            The path to the CSV file containing future data for predictors (e.g., forecasted weather).
        output_filename : Path
            The path where the generated forecasts (predictions) will be saved as a CSV file.
        """
        logger.info(f"CLI: Predicting with model from '{model_filename}'...")
        try:
            # Instantiate the estimator class to call load_predictor
            estimator_instance = estimator_class()
            predictor: ModelPredictor = estimator_instance.load_predictor(str(model_filename))

            logger.info(f"Loading historic data from '{historic_data_filename}'.")
            dataset = DataSet.from_csv(str(historic_data_filename), dc)

            # For future data, remove the target variable (e.g., "disease_cases")
            # This assumes 'dc' is the dataclass for training data including the target.
            if "disease_cases" in [f.name for f in dataclasses.fields(dc)]:
                future_dc = remove_field(dc, "disease_cases")
            else:  # If target is not in dc, future_dc is same as dc (or needs specific logic)
                future_dc = dc
                logger.warning(
                    "Target 'disease_cases' not found in inferred dataclass for future data. Using full dataclass."
                )

            logger.info(f"Loading future data from '{future_data_filename}' as type '{future_dc.__name__}'.")
            future_data = DataSet.from_csv(str(future_data_filename), future_dc)

            forecasts = predictor.predict(dataset, future_data)
            forecasts.to_csv(str(output_filename))
            logger.info(f"Predictions saved to '{output_filename}'.")
        except FileNotFoundError as e:
            logger.error(f"File not found during prediction: {e.filename}")
        except (pd.errors.ParserError, ValueError) as e:
            logger.error(f"Error loading or parsing data during prediction: {e}")
        except Exception as e:
            logger.error(f"An error occurred during prediction: {e}", exc_info=True)

    return app


# The return type (App, Callable, Callable) is unusual. Consider if only App is needed.
def generate_template_app(model_template: ModelTemplate) -> Tuple[App, Any, Any]:
    """
    Generates a `cyclopts` CLI application with 'train' and 'predict' commands
    for a given `ModelTemplate`.

    This allows for CLI interaction with models defined via templates, including
    handling of model configurations.

    Args:
        model_template (ModelTemplate): The model template instance.

    Returns:
        Tuple[App, Any, Any]: A tuple containing the `cyclopts.App` instance and
                              the generated `train` and `predict` functions.
                              (Returning inner functions is unconventional for CLI app generation).
    """
    app = App()

    @app.command()
    def train(training_data_filename: Path, model_path: Path, model_config_path: Optional[Path] = None) -> None:
        """
        Train a model defined by a ModelTemplate using historic data.

        Requires a training data CSV and a path to save the model.
        Optionally accepts a model configuration YAML/JSON file.

        Parameters
        ----------
        training_data_filename : Path
            Path to the CSV file containing training data.
        model_path : Path
            Path where the trained model predictor will be saved.
        model_config_path : Optional[Path], optional
            Path to a model configuration file (e.g., YAML or JSON).
            If None, default model configuration is used. Defaults to None.
        """
        logger.info(f"CLI Template: Training model '{model_template.name}'...")
        try:
            model_config = None
            if model_config_path:
                logger.info(f"Loading model configuration from: {model_config_path}")
                # Assuming get_config_class returns a Pydantic model or similar with parse_file
                config_class = model_template.get_config_class()
                if config_class:
                    model_config = config_class.parse_file(model_config_path)
                else:
                    logger.warning(
                        f"Model template {model_template.name} does not define a config class. Ignoring config path."
                    )

            # TODO: create method in ModelTemplate to get the actual fields for dc creation
            # Or give the model the responsibility to define its input data structure.
            # Current: estimator.covariate_names might not include target or all necessary fields.
            estimator = model_template.get_model(model_config)  # Instantiates the model

            # Infer dataclass based on estimator's expected features (covariates + target)
            # This part needs robust logic to determine all fields for the training dataclass.
            # For now, assuming covariate_names + a default target "disease_cases".
            data_fields = estimator.covariate_names + [
                estimator.target_name or "disease_cases"
            ]  # Assuming target_name attr
            dc = create_tsdataclass(list(set(data_fields)))  # Use set to avoid duplicates if target is in covariates

            logger.info(f"Loading training data from '{training_data_filename}' as type '{dc.__name__}'.")
            dataset = DataSet.from_csv(str(training_data_filename), dc)

            predictor = estimator.train(dataset)
            predictor.save(str(model_path))
            logger.info(f"Training complete. Model saved to '{model_path}'.")
        except FileNotFoundError as e:
            logger.error(f"File not found: {e.filename}")
        except Exception as e:
            logger.error(f"An error occurred during template app training: {e}", exc_info=True)

    @app.command()
    def predict(
        model_filename: Path,
        historic_data_filename: Path,
        future_data_filename: Path,
        output_filename: Path,
        model_config_path: Optional[Path] = None,
    ) -> None:
        """
        Predict using a trained model defined by a ModelTemplate.

        Loads a trained model, historic data, future predictor data, and optionally
        a model configuration. Generates forecasts and saves them to CSV.

        Parameters
        ----------
        model_filename : Path
            Path to the saved model predictor file.
        historic_data_filename : Path
            Path to CSV file with historic data (up to prediction start).
        future_data_filename : Path
            Path to CSV file with future data for predictors.
        output_filename : Path
            Path where the forecast CSV will be saved.
        model_config_path : Optional[Path], optional
            Path to a model configuration file. This might be needed if the
            model loading or prediction process depends on configuration not
            serialized with the model. Defaults to None.
        """
        logger.info(f"CLI Template: Predicting with model from '{model_filename}'...")
        try:
            model_config = None
            if model_config_path:
                logger.info(f"Loading model configuration from: {model_config_path}")
                config_class = model_template.get_config_class()
                if config_class:
                    model_config = config_class.parse_file(model_config_path)
                else:
                    logger.warning(
                        f"Model template {model_template.name} does not define a config class. Ignoring config path."
                    )

            estimator = model_template.get_model(
                model_config
            )  # Get estimator instance (might be needed for load_predictor context)

            # Infer dataclass based on estimator's expected features for historic data
            # This logic should mirror the one in train for consistency
            historic_data_fields = estimator.covariate_names + [estimator.target_name or "disease_cases"]
            dc = create_tsdataclass(list(set(historic_data_fields)))

            # For future data, remove the target variable
            if (estimator.target_name or "disease_cases") in [f.name for f in dataclasses.fields(dc)]:
                future_dc = remove_field(dc, estimator.target_name or "disease_cases")
            else:
                future_dc = dc  # Should not happen if dc was created correctly with target
                logger.warning(
                    "Target field not found in inferred historic dataclass for future data. Using full dataclass."
                )

            predictor = estimator.load_predictor(
                str(model_filename)
            )  # load_predictor might be a method of the estimator instance

            logger.info(f"Loading historic data from '{historic_data_filename}'.")
            dataset = DataSet.from_csv(str(historic_data_filename), dc)

            logger.info(f"Loading future data from '{future_data_filename}' as type '{future_dc.__name__}'.")
            future_data = DataSet.from_csv(str(future_data_filename), future_dc)

            forecasts = predictor.predict(dataset, future_data)
            forecasts.to_csv(str(output_filename))
            logger.info(f"Predictions saved to '{output_filename}'.")
        except FileNotFoundError as e:
            logger.error(f"File not found during prediction: {e.filename}")
        except Exception as e:
            logger.error(f"An error occurred during template app prediction: {e}", exc_info=True)

    # The commented-out code block below seems to be related to defining
    # a more structured configuration for model entry points, possibly for
    # integration with systems like MLflow or other workflow managers.
    # If this is an active area of development or a planned feature,
    # it should be documented or moved to a separate module/issue.
    # If obsolete, it should be removed.
    #
    # from pydantic import BaseModel
    # class CommandConfig(BaseModel):
    #     command: str
    #     parameters: dict
    # class EntryPointConfig(BaseModel):
    #     train: CommandConfig
    #     predict: CommandConfig
    # class ModelTemplateConfig(BaseModel):
    #     name: str
    #     entry_points: EntryPointConfig
    #
    # model_template_config = ModelTemplateConfig(
    #     name=model_template.name,
    #     entry_points=EntryPointConfig(
    #         train=CommandConfig(command='python main.py train {train_data} {model} {model_config}',
    #                             parameters={'train_data': 'str', 'model': 'str', 'model_config': 'str'}),
    #         predict=CommandConfig("python main.py predict {model} {historic_data} {future_data} {out_file} {model_config}",
    #                               parameters={n: 'str' for n in ['historic_data', 'future_data', 'out_file', 'model_config']})),
    # )

    return app, train, predict  # Returning train and predict functions is unusual for CLI app generation.
    # Typically, only the `app` object would be returned.
