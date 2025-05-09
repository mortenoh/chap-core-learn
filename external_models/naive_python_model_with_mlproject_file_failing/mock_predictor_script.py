# Improvement Suggestions:
# 1. Explicit Failure Documentation: Clearly document that the `train` command is designed to fail.
# 2. Conditional Failure (Optional): For more flexible testing, the `assert False` could be made conditional (e.g., via an env var or argument).
# 3. Logging: Use the `logging` module for output, including information leading up to the intentional failure in `train`.
# 4. Error Handling in `predict`: The `predict` command should still have robust error handling for file I/O and prediction steps.
# 5. Code Structure: Remove the redundant `main()` function and ensure consistent import organization.

"""
Console script for `ch_modelling` - FAIILNG VERSION.

This script provides CLI commands for a naive predictor model.
The `train` command is INTENTIONALLY DESIGNED TO FAIL due to an `assert False`
for testing error handling and failure conditions in MLflow projects or CI/CD.
The `predict` command provides standard prediction logic.
"""

import logging
import sys  # For sys.exit

from cyclopts import App

# Assuming chap_core modules are accessible.
from chap_core.data import DataSet
from chap_core.datatypes import FullData, remove_field
from chap_core.predictor.naive_estimator import NaiveEstimator, NaivePredictor

# Configure basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

app = App(help="CLI for a Naive Model (with intentional training failure).")


@app.command()
def train(training_data_filename: str, model_path: str):
    """
    Attempts to train a NaiveEstimator model.

    THIS COMMAND IS DESIGNED TO FAIL due to an `assert False` statement.
    It simulates a failure scenario in a model training process.

    Parameters
    ----------
    training_data_filename : str
        Path to the CSV file containing training data.
    model_path : str
        Path where the trained model would have been saved.
    """
    logging.info("Starting 'train' command (designed to fail)...")
    logging.info(f"Training data filename: {training_data_filename}")
    logging.info(f"Model output path: {model_path}")

    # Intentionally failing assertion for testing purposes
    assert False, "Intentional failure in train command for testing."

    # The following code is unreachable due to the assert False above.
    # It's kept here to mirror the structure of a non-failing train script.
    try:
        logging.info("Loading training data (this part will not be reached)...")
        dataset = DataSet.from_csv(training_data_filename, item_type=FullData)
        logging.info(f"Loaded {len(dataset.items)} records for training (unreachable).")

        chap_estimator = NaiveEstimator
        logging.info("Training NaiveEstimator (unreachable)...")
        predictor = chap_estimator().train(dataset)
        logging.info("Model training completed (unreachable).")

        logging.info(f"Saving model to {model_path} (unreachable)...")
        predictor.save(model_path)
        logging.info("Model saved successfully (unreachable).")
    except Exception as e:  # Should not be reached due to assert False
        logging.error(f"An unexpected error occurred during the unreachable part of training: {e}", exc_info=True)
        sys.exit(1)


@app.command()
def predict(model_filename: str, historic_data_filename: str, future_data_filename: str, output_filename: str):
    """
    Loads a trained NaivePredictor model and generates predictions.

    This command assumes a model file can be loaded, even though the `train`
    command in this script is designed to fail and not produce one.

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
        future_data_item_type = remove_field(FullData, "disease_cases")
        future_data = DataSet.from_csv(future_data_filename, item_type=future_data_item_type)
        logging.info(f"Loaded {len(future_data.items)} future records for prediction.")

        chap_predictor = NaivePredictor
        logging.info(f"Loading model from {model_filename}...")
        predictor = chap_predictor.load(model_filename)
        logging.info("Model loaded successfully.")

        logging.info("Generating forecasts...")
        forecasts = predictor.predict(dataset, future_data)
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
    app()
