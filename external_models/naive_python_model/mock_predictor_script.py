# Improvement Suggestions:
# 1. Error Handling: Add try-except blocks in each command for robust file I/O and model operations.
# 2. Logging: Replace print() statements with the `logging` module for better traceability and standardized output.
# 3. Model Configuration: Allow `MultiRegionNaivePredictor` parameters to be configurable, perhaps via Typer options or a config file.
# 4. Efficiency in `predict_values`: This command re-trains the model on each call. Clarify if this is intended or if it should load a pre-trained model. If re-training is always needed, make this explicit.
# 5. Clearer Output Management: For predictions, ensure output (e.g., to CSV) is handled cleanly, and consider options for output verbosity.

"""
CLI script for training and running predictions with a MultiRegionNaivePredictor.

This script provides commands to:
- Train a naive predictor model and save it.
- Load a trained model and make predictions on new climate data.
- Perform a train-and-predict operation in one go.
"""

import logging
import pickle
import sys  # For sys.exit in error handling

import typer

# Assuming chap_core modules are accessible in the execution environment.
# Proper packaging or MLproject setup would manage this for external models.
from chap_core.datatypes import ClimateData, ClimateHealthTimeSeries
from chap_core.predictor.naive_predictor import MultiRegionNaivePredictor
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet

# Configure basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

app = typer.Typer(help="CLI for Naive Predictor Model Training and Prediction.")


@app.command()
def train(
    train_data_set: str = typer.Argument(..., help="Path to the CSV file containing training data."),
    model_output_file: str = typer.Argument(..., help="Path where the trained model will be saved (pickle format)."),
):
    """
    Trains a MultiRegionNaivePredictor on the provided dataset and saves the model.
    """
    logging.info("Starting training process...")
    logging.info(f"Training data: {train_data_set}")
    logging.info(f"Model output file: {model_output_file}")

    try:
        predictor = MultiRegionNaivePredictor()  # Consider making parameters configurable
        logging.info("Loading training data...")
        train_data = DataSet.from_csv(train_data_set, item_type=ClimateHealthTimeSeries)
        logging.info(f"Loaded {len(train_data.items)} records for training.")

        logging.info("Training model...")
        predictor.train(train_data)
        logging.info("Model training completed.")

        logging.info(f"Saving model to {model_output_file}...")
        with open(model_output_file, "wb") as f:
            pickle.dump(predictor, f)
        logging.info("Model saved successfully.")

    except FileNotFoundError:
        logging.error(f"Error: File not found. Check paths: {train_data_set} or {model_output_file}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"An error occurred during training: {e}", exc_info=True)
        sys.exit(1)


@app.command()
def predict(
    future_climate_data_set: str = typer.Argument(..., help="Path to the CSV file containing future climate data."),
    model_file: str = typer.Argument(..., help="Path to the trained model file (pickle format)."),
    output_file: str = typer.Argument(..., help="Path where the predictions (CSV) will be saved."),
):
    """
    Loads a trained model and makes predictions on new future climate data.
    """
    logging.info("Starting prediction process...")
    logging.info(f"Future climate data: {future_climate_data_set}")
    logging.info(f"Model file: {model_file}")
    logging.info(f"Output file: {output_file}")

    try:
        logging.info(f"Loading model from {model_file}...")
        with open(model_file, "rb") as f:
            predictor: MultiRegionNaivePredictor = pickle.load(f)
        logging.info("Model loaded successfully.")

        logging.info("Loading future climate data...")
        future_climate_data = DataSet.from_csv(future_climate_data_set, item_type=ClimateData)
        logging.info(f"Loaded {len(future_climate_data.items)} records for prediction.")

        logging.info("Making predictions...")
        predictions = predictor.predict(future_climate_data)
        logging.info("Predictions generated.")
        # logging.debug(f"Predictions content:\n{predictions}") # Potentially verbose

        logging.info(f"Saving predictions to {output_file}...")
        predictions.to_csv(output_file)
        logging.info("Predictions saved successfully.")

    except FileNotFoundError:
        logging.error(f"Error: File not found. Check paths: {future_climate_data_set}, {model_file}, or {output_file}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"An error occurred during prediction: {e}", exc_info=True)
        sys.exit(1)


@app.command()
def predict_values(
    train_data_set: str = typer.Argument(..., help="Path to the CSV file containing training data."),
    future_climate_data_set: str = typer.Argument(..., help="Path to the CSV file containing future climate data."),
    output_file: str = typer.Argument(..., help="Path where the predictions (CSV) will be saved."),
):
    """
    Trains a new model and then makes predictions (train-then-predict).
    Note: This command re-trains the model on each call.
    """
    logging.info("Starting train-then-predict process...")
    logging.info(f"Training data: {train_data_set}")
    logging.info(f"Future climate data: {future_climate_data_set}")
    logging.info(f"Output file: {output_file}")

    try:
        predictor = MultiRegionNaivePredictor()
        logging.info("Loading training data for predict_values...")
        train_data = DataSet.from_csv(train_data_set, item_type=ClimateHealthTimeSeries)
        logging.info(f"Loaded {len(train_data.items)} records for training.")

        logging.info("Training model for predict_values...")
        predictor.train(train_data)
        logging.info("Model training completed for predict_values.")

        logging.info("Loading future climate data for predict_values...")
        future_climate_data = DataSet.from_csv(future_climate_data_set, item_type=ClimateData)
        logging.info(f"Loaded {len(future_climate_data.items)} records for prediction.")

        logging.info("Making predictions for predict_values...")
        predictions = predictor.predict(future_climate_data)
        logging.info("Predictions generated for predict_values.")
        # logging.debug(f"Predictions content:\n{predictions}")

        logging.info(f"Saving predictions to {output_file}...")
        predictions.to_csv(output_file)
        logging.info("Predictions saved successfully for predict_values.")

    except FileNotFoundError:
        logging.error(
            f"Error: File not found. Check paths: {train_data_set}, {future_climate_data_set}, or {output_file}"
        )
        sys.exit(1)
    except Exception as e:
        logging.error(f"An error occurred during train-then-predict: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    app()
