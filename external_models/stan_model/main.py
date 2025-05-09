# Improvement Suggestions:
# 1. Argument Parsing: Use `argparse` to specify input data file, Stan model file, output fit file, and sampling parameters.
# 2. Error Handling: Implement try-except blocks for file I/O, Stan model compilation, and MCMC sampling.
# 3. Logging: Use the `logging` module for informative messages about script progress and potential issues.
# 4. Stan Model Caching: Investigate and implement PyStan's model caching to avoid recompilation on every run if the model code hasn't changed.
# 5. Configurable Sampling Parameters: Allow `iter`, `chains`, `seed`, etc., for `sm.sampling()` to be set via arguments.

"""
Trains a Bayesian model using Stan (via PyStan).

This script performs the following steps:
1. Loads training data from a CSV file.
2. Prepares the data into the format required by the Stan model.
3. Reads Stan model code from a .stan file.
4. Compiles the Stan model.
5. Fits the model to the data using MCMC sampling.
6. Saves the fitted Stan model object (fit object) to a pickle file.
"""

import argparse
import logging
import pickle
import sys

import pandas as pd
import stan as pystan  # PyStan library, often imported as `stan` or `pystan`

# Configure basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_and_prepare_data(data_csv_path: str) -> dict:
    """Loads data from CSV and prepares it for Stan."""
    logging.info(f"Loading training data from: {data_csv_path}")
    try:
        training_data = pd.read_csv(data_csv_path)
        logging.info(f"Successfully loaded {len(training_data)} records.")
    except FileNotFoundError:
        logging.error(f"Data file not found: {data_csv_path}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error loading data: {e}", exc_info=True)
        sys.exit(1)

    logging.info("Preparing data for Stan...")
    # Example data preparation, this will be highly model-specific
    # Ensure these columns exist in your CSV or adjust as needed
    try:
        stan_data = {
            "N": len(training_data),
            "K": training_data["location"].nunique() if "location" in training_data else 1,
            "M": training_data["month"].nunique() if "month" in training_data else 1,
            "cases": training_data["cases"].values.astype(int)
            if "cases" in training_data
            else [],  # Stan typically expects int for counts
            "location": training_data["location"].astype("category").cat.codes.values + 1
            if "location" in training_data
            else [],  # Stan 1-indexed
            "month": training_data["month"].astype("category").cat.codes.values + 1
            if "month" in training_data
            else [],  # Stan 1-indexed
            "temperature": training_data["temperature"].values if "temperature" in training_data else [],
        }
        logging.info("Data preparation for Stan complete.")
        return stan_data
    except KeyError as e:
        logging.error(f"Missing expected column in CSV for Stan data preparation: {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error preparing Stan data: {e}", exc_info=True)
        sys.exit(1)


def compile_stan_model(model_code_path: str, stan_data: dict):
    """Compiles the Stan model from a .stan file."""
    logging.info(f"Loading Stan model code from: {model_code_path}")
    try:
        with open(model_code_path, "r") as f:
            model_code = f.read()
        logging.info("Stan model code loaded.")
    except FileNotFoundError:
        logging.error(f"Stan model file not found: {model_code_path}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error reading Stan model file: {e}", exc_info=True)
        sys.exit(1)

    logging.info("Compiling Stan model (this may take a while)...")
    try:
        # Consider using pystan.StanModel for caching if this version of PyStan supports it,
        # or check PyStan3's `build` function's caching behavior.
        # For PyStan 3:
        posterior = pystan.build(program_code=model_code, data=stan_data)
        logging.info("Stan model compiled successfully.")
        return posterior
    except Exception as e:
        logging.error(f"Error compiling Stan model: {e}", exc_info=True)
        # Provide more specific error details if possible, Stan compilation errors can be verbose
        if hasattr(e, "message"):
            logging.error(f"Stan C++ compiler error: {e.message}")
        sys.exit(1)


def fit_stan_model(posterior, stan_data: dict, iterations: int, chains: int, seed: int):
    """Fits the compiled Stan model to the data."""
    logging.info(f"Starting MCMC sampling with {iterations} iterations and {chains} chains...")
    try:
        fit = posterior.sample(
            num_chains=chains, num_samples=iterations // 2, num_warmup=iterations // 2, data=stan_data, seed=seed
        )
        # Note: PyStan 3 uses num_samples and num_warmup per chain.
        # If 'iter' was total iterations including warmup, then num_samples = iter/2, num_warmup = iter/2 for typical half warmup.
        logging.info("MCMC sampling completed.")
        return fit
    except Exception as e:
        logging.error(f"Error during Stan model sampling: {e}", exc_info=True)
        sys.exit(1)


def save_fit_object(fit, output_path: str):
    """Saves the Stan fit object to a pickle file."""
    logging.info(f"Saving fit object to: {output_path}")
    try:
        with open(output_path, "wb") as f:
            pickle.dump(fit, f)
        logging.info("Fit object saved successfully.")
    except Exception as e:
        logging.error(f"Error saving fit object: {e}", exc_info=True)
        sys.exit(1)


def main():
    """
    Main function to orchestrate Stan model training.
    """
    parser = argparse.ArgumentParser(description="Train a Stan model and save the fit object.")
    parser.add_argument("data_csv_path", help="Path to the training data CSV file.")
    parser.add_argument("model_stan_path", help="Path to the .stan model code file.")
    parser.add_argument("output_fit_path", help="Path to save the pickled Stan fit object.")
    parser.add_argument("--iter", type=int, default=1000, help="Total MCMC iterations (including warmup).")
    parser.add_argument("--chains", type=int, default=4, help="Number of MCMC chains.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")

    args = parser.parse_args()

    logging.info("Stan model training script started.")

    stan_data = load_and_prepare_data(args.data_csv_path)
    compiled_model = compile_stan_model(args.model_stan_path, stan_data)  # Pass stan_data for PyStan3 build
    fit = fit_stan_model(compiled_model, stan_data, args.iter, args.chains, args.seed)
    save_fit_object(fit, args.output_fit_path)

    logging.info("Stan model training script finished successfully.")


if __name__ == "__main__":
    main()
