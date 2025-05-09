# Improvement Suggestions:
# 1. Unreachable Code: Remove the `sys.exit()` call on line 4 that prevents the training logic from executing.
# 2. Argument Parsing: Utilize a robust argument parser like `argparse` (or `typer` which is imported) for command-line arguments.
# 3. Error Handling: Implement `try-except` blocks for file I/O (data loading, model saving) and during the model training process.
# 4. Logging: Replace `print()` statements with the `logging` module for better diagnostics and output management.
# 5. Model Configuration: Allow predictor parameters or configurations to be passed via arguments or a config file for greater flexibility.

"""
Training script for an MLflow project.

This script trains a MultiRegionNaivePredictor model using climate and health
time series data provided as a CSV file. The trained model is then saved
to a specified output file using pickle.
"""

import argparse
import logging
import pickle
import sys

# It seems chap_core might not be directly installable in all environments
# where this external model is run. For robustness in a standalone script,
# it's better if such dependencies are managed carefully, e.g., via MLproject.
# For now, assuming chap_core modules are accessible.
from chap_core.datatypes import ClimateHealthTimeSeries  # ClimateData not used
from chap_core.predictor.naive_predictor import MultiRegionNaivePredictor  # NaivePredictor not used
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet

# import typer # Typer was imported but not used for argument parsing. Sticking to argparse for consistency.

# Configure basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def main():
    """
    Main function to parse arguments, load data, train the model,
    and save the trained model.
    """
    parser = argparse.ArgumentParser(description="Train a MultiRegionNaivePredictor model.")
    parser.add_argument("train_data_set", help="Path to the CSV file containing training data.")
    parser.add_argument("model_output_file", help="Path where the trained model will be saved (pickle format).")
    # Example of how model configuration could be added:
    # parser.add_argument("--config_file", help="Optional path to a model configuration YAML file.")

    args = parser.parse_args()

    logging.info("Starting model training process...")
    logging.info(f"Training data set: {args.train_data_set}")
    logging.info(f"Model output file: {args.model_output_file}")

    try:
        logging.info("Loading training data...")
        # Assuming ClimateHealthTimeSeries is the correct Pydantic model for rows in the CSV
        train_data = DataSet.from_csv(args.train_data_set, item_type=ClimateHealthTimeSeries)
        logging.info(f"Successfully loaded {len(train_data.items)} records for training.")

        # Initialize and train the predictor
        # In a real scenario, predictor parameters might come from args.config_file or other arguments
        predictor = MultiRegionNaivePredictor()
        logging.info("Training MultiRegionNaivePredictor model...")
        predictor.train(train_data)
        logging.info("Model training completed.")

        # Pickle the trained predictor
        logging.info(f"Saving trained model to {args.model_output_file}...")
        with open(args.model_output_file, "wb") as f:
            pickle.dump(predictor, f)
        logging.info("Trained model saved successfully.")

    except FileNotFoundError:
        logging.error(f"Error: Training data file not found at {args.train_data_set}")
        sys.exit(1)
    except ImportError as e:
        logging.error(
            f"ImportError: Failed to import a chap_core module. Ensure chap_core is installed and accessible. Details: {e}"
        )
        sys.exit(1)
    except Exception as e:
        logging.error(f"An unexpected error occurred during training: {e}", exc_info=True)
        sys.exit(1)

    logging.info("Training process finished.")


if __name__ == "__main__":
    # The original script had "print("Training")" and "sys.exit()" here.
    # This has been moved into the main function or handled by logging.
    main()
    main()
