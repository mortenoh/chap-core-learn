# Improvement Suggestions:
# 1. Error Handling: Add comprehensive try-except blocks in `train` and `predict` for file I/O and model operations.
# 2. Logging: Replace all `print()` statements with the `logging` module for standardized and more controllable output.
# 3. Model Configuration: Consider if `NaiveEstimator` or `NaivePredictor` have configurable parameters that could be exposed via CLI options.
# 4. Docstring Detail: Expand placeholder docstrings for `train` and `predict` to clearly explain their functionality, arguments, and expected behavior.
# 5. Code Structure: Remove the redundant `main()` function, as `if __name__ == "__main__": app()` is sufficient for Cyclopts.

"""
Command-line interface for training a NaiveEstimator and making predictions
with a NaivePredictor, typically for use within an MLflow Project.

This script provides two main commands:
- `train`: Trains a naive model using provided training data and saves the predictor.
- `predict`: Loads a trained naive predictor and generates forecasts.
"""

import logging
import os
import sys  # For sys.exit

from cyclopts import App

from chap_core.data import DataSet
from chap_core.datatypes import FullData, remove_field  # ClimateData not used
from chap_core.predictor.naive_estimator import NaiveEstimator, NaivePredictor

# Configure basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

app = App(help="CLI for Naive Model Training and Prediction with MLflow Project context.")


@app.command()
def train(
    training_data_filename: str,  # Removed typer.Argument, Cyclopts infers from signature
    model_path: str,  # Removed typer.Argument
):
    """
    Trains a NaiveEstimator on the provided training data and saves the
    resulting NaivePredictor model to the specified path.

    Parameters
    ----------
    training_data_filename : str
        Path to the CSV file containing training data (FullData format).
    model_path : str
        Path where the trained NaivePredictor model will be saved.
    """
    logging.info("Starting training process...")
    logging.info(f"Current directory: {os.getcwd()}")
    logging.info(f"Training data filename: {training_data_filename}")
    logging.info(f"Model output path: {model_path}")

    try:
        logging.info("Loading training data...")
        dataset = DataSet.from_csv(training_data_filename, item_type=FullData)
        logging.info(f"Loaded {len(dataset.items)} records for training.")

        chap_estimator = NaiveEstimator  # Consider if this can be configured

        logging.info("Training NaiveEstimator...")
        predictor = chap_estimator().train(dataset)
        logging.info("Model training completed.")

        logging.info(f"Saving model to {model_path}...")
        predictor.save(model_path)  # NaivePredictor.save uses pickle
        logging.info("Model saved successfully.")

    except FileNotFoundError:
        logging.error(f"Error: File not found. Check path: {training_data_filename}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"An error occurred during training: {e}", exc_info=True)
        sys.exit(1)


@app.command()
def predict(
    model_filename: str,  # Removed typer.Argument
    historic_data_filename: str,  # Removed typer.Argument
    future_data_filename: str,  # Removed typer.Argument
    output_filename: str,  # Removed typer.Argument
):
    """
    Loads a trained NaivePredictor model and generates predictions.

    The historic data is used by some predictors to establish a baseline or context,
    while future data provides the covariates for the prediction period.

    Parameters
    ----------
    model_filename : str
        Path to the trained NaivePredictor model file.
    historic_data_filename : str
        Path to the CSV file containing historic data (FullData format).
    future_data_filename : str
        Path to the CSV file containing future data (FullData format, target field removed).
    output_filename : str
        Path where the predictions (CSV) will be saved.
    """
    logging.info("Starting prediction process...")
    logging.info(f"Model filename: {model_filename}")
    logging.info(f"Historic data filename: {historic_data_filename}")
    logging.info(f"Future data filename: {future_data_filename}")
    logging.info(f"Output filename: {output_filename}")

    try:
        logging.info("Loading historic data...")
        dataset = DataSet.from_csv(historic_data_filename, item_type=FullData)
        logging.info(f"Loaded {len(dataset.items)} historic records.")

        logging.info("Loading future data (covariates)...")
        # Assuming 'disease_cases' is the target field to be removed for future covariate data
        future_data_item_type = remove_field(FullData, "disease_cases")
        future_data = DataSet.from_csv(future_data_filename, item_type=future_data_item_type)
        logging.info(f"Loaded {len(future_data.items)} future records for prediction.")

        chap_predictor = NaivePredictor
        logging.info(f"Loading model from {model_filename}...")
        predictor = chap_predictor.load(model_filename)  # NaivePredictor.load uses pickle
        logging.info("Model loaded successfully.")

        logging.info("Generating forecasts...")
        forecasts = predictor.predict(dataset, future_data)  # NaivePredictor.predict API
        logging.info("Forecasts generated.")

        logging.info(f"Saving forecasts to {output_filename}...")
        forecasts.to_csv(output_filename)
        logging.info("Forecasts saved successfully.")

    except FileNotFoundError:
        logging.error(
            f"Error: File not found. Check paths: {model_filename}, {historic_data_filename}, or {future_data_filename}"
        )
        sys.exit(1)
    except Exception as e:
        logging.error(f"An error occurred during prediction: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    app()  # Removed duplicated app() call
