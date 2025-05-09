# Improvement Suggestions:
# 1. **Consistent Logging**: Ensure `initialize_logging` is called consistently across all relevant CLI commands for uniform log handling, or establish a clear global logging setup at the app's entry point.
# 2. **Error Handling and User Feedback**: Enhance error handling for file I/O, invalid model names, dataset loading issues, and external API/tool interactions. Provide clear, informative error messages to the user.
# 3. **Parameter Validation**: Implement more robust validation for command-line arguments beyond basic type checking (e.g., file existence, value ranges, consistency between related parameters).
# 4. **Code Duplication/Complexity in `evaluate`**: Review the model loading and configuration loop in the `evaluate` command for potential refactoring to reduce complexity and improve clarity, especially regarding the handling of multiple models and their configurations.
# 5. **`AreaPolygons` and `main_function`**: Clarify the purpose and status of the `AreaPolygons` dataclass and the `main_function`. If they are placeholders, document their intended use. If obsolete, consider removal.

"""
CHAP-CORE Command-Line Interface.

This script provides a suite of command-line tools for interacting with the
CHAP-core functionalities, including model evaluation, forecasting, data processing,
and API management. It uses the `cyclopts` library for command parsing.
"""

import dataclasses
import json
import logging
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
from cyclopts import App

from chap_core import api
from chap_core.assessment.dataset_splitting import train_test_generator
from chap_core.assessment.forecast import multi_forecast as do_multi_forecast
from chap_core.assessment.prediction_evaluator import backtest as _backtest
from chap_core.assessment.prediction_evaluator import evaluate_model
from chap_core.climate_predictor import QuickForecastFetcher
from chap_core.datatypes import FullData
from chap_core.exceptions import NoPredictionsError
from chap_core.file_io.example_data_set import DataSetType, datasets
from chap_core.geometry import Polygons
from chap_core.log_config import initialize_logging
from chap_core.models.model_template import ModelTemplate
from chap_core.models.utils import get_model_from_directory_or_github_url
from chap_core.plotting.prediction_plot import plot_forecast_from_summaries
from chap_core.predictor import ModelType
from chap_core.predictor.model_registry import registry
from chap_core.rest_api_src.worker_functions import dataset_to_datalist, samples_to_evaluation_response
from chap_core.spatio_temporal_data.multi_country_dataset import MultiCountryDataSet
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet
from chap_core.time_period.date_util_wrapper import delta_month

# Initialize logger early. Specific command configurations can override.
# initialize_logging() # Consider a default global initialization
logger = logging.getLogger(__name__)  # Use __name__ for module-specific logger
# logger.setLevel(logging.INFO) # Default level, can be changed by initialize_logging

app = App(help_format="markdown")


def append_to_csv(file_object, data_frame: pd.DataFrame):
    """
    Append a Pandas DataFrame to an existing CSV file without writing the header.

    Args:
        file_object: A file object opened in append mode (e.g., `open(filename, 'a')`).
        data_frame (pd.DataFrame): The DataFrame to append.
    """
    data_frame.to_csv(file_object, mode="a", header=False, index=False)


@app.command()
def evaluate(
    model_name: ModelType | str,
    dataset_name: Optional[DataSetType] = None,
    dataset_country: Optional[str] = None,
    dataset_csv: Optional[Path] = None,
    polygons_json: Optional[Path] = None,
    polygons_id_field: Optional[str] = "id",
    prediction_length: int = 6,
    n_splits: int = 7,
    report_filename: Optional[str] = "report.pdf",
    ignore_environment: bool = False,
    debug: bool = False,
    log_file: Optional[str] = None,
    run_directory_type: Optional[Literal["latest", "timestamp", "use_existing"]] = "timestamp",
    model_configuration_yaml: Optional[str] = None,
):
    """
    Evaluate one or more models on a dataset using forecast cross-validation.

    This command performs a rigorous evaluation by splitting the dataset into
    multiple train/test sets. It outputs performance metrics to both a PDF
    report and a CSV file. Multiple models can be specified as a comma-separated string.

    Args:
        model_name: Name(s) of the model(s) to evaluate (e.g., "ewars_plus_model" or URL, comma-separated for multiple).
        dataset_name: Key for a predefined dataset (e.g., "laos_minimal").
        dataset_country: If `dataset_name` refers to a MultiCountryDataSet, specify the country key.
        dataset_csv: Path to a custom dataset CSV file. Required if `dataset_name` is not used.
        polygons_json: Path to a GeoJSON file containing polygons for the custom dataset.
        polygons_id_field: The property field in `polygons_json` used as the location identifier.
        prediction_length: Number of time periods to forecast ahead in each split.
        n_splits: Number of train/test splits to generate for cross-validation.
        report_filename: Base name for the output PDF report and CSV metrics file.
        ignore_environment: If True, ignore the model's Conda environment specification.
        debug: If True, enable debug logging.
        log_file: Optional path to a file for saving logs.
        run_directory_type: Strategy for creating/using run directories for models.
        model_configuration_yaml: Path(s) to model configuration YAML file(s), comma-separated if multiple models.
    """
    initialize_logging(debug, log_file)
    logger.info(f"Starting evaluation for model(s): {model_name}")

    if dataset_name is None:
        if not dataset_csv:
            logger.error("Must provide dataset_csv if dataset_name is not set.")
            raise ValueError("Either dataset_name or dataset_csv must be provided.")
        logger.info(f"Loading custom dataset from: {dataset_csv}")
        dataset = DataSet.from_csv(dataset_csv, FullData)
        if polygons_json:
            logger.info(f"Loading polygons from: {polygons_json} using ID field: {polygons_id_field}")
            polygons = Polygons.from_file(polygons_json, id_property=polygons_id_field)
            dataset.set_polygons(polygons.data)
    else:
        logger.info(f"Loading predefined dataset: {dataset_name}")
        dataset = datasets[dataset_name].load()
        if isinstance(dataset, MultiCountryDataSet):
            if not dataset_country or dataset_country not in dataset.countries:
                logger.error(f"Country '{dataset_country}' not in dataset. Available: {dataset.countries}")
                raise ValueError(f"Invalid country for MultiCountryDataSet. Available: {dataset.countries}")
            logger.info(f"Selecting country: {dataset_country} from MultiCountryDataSet.")
            dataset = dataset[dataset_country]

    model_list = [m.strip() for m in model_name.split(",")]
    model_config_list_paths = (
        [Path(c.strip()) for c in model_configuration_yaml.split(",")]
        if model_configuration_yaml
        else [None] * len(model_list)
    )

    if len(model_list) != len(model_config_list_paths):
        logger.error("Mismatch in the number of models and model configurations provided.")
        raise ValueError("Number of models must match number of model configurations.")

    results_dict = {}
    base_run_dir = Path("./runs/")  # Define base run directory

    for i, current_model_name in enumerate(model_list):
        config_path = model_config_list_paths[i]
        logger.info(f"Processing model: {current_model_name} with config: {config_path}")
        try:
            template = ModelTemplate.from_directory_or_github_url(
                current_model_name,
                base_working_dir=base_run_dir,
                ignore_env=ignore_environment,
                run_dir_type=run_directory_type,
            )
            configuration = template.get_model_configuration_from_yaml(config_path) if config_path else None
            model_instance = template.get_model(configuration)()  # Instantiate the model

            eval_results = evaluate_model(
                model_instance,
                dataset,
                prediction_length=prediction_length,
                n_test_sets=n_splits,
                report_filename=report_filename,  # Note: This might overwrite PDF for each model if not handled
            )
            results_dict[current_model_name] = eval_results
            logger.info(f"Evaluation successful for model: {current_model_name}")
        except NoPredictionsError as e:
            logger.error(f"No predictions generated for model {current_model_name}: {e}")
            results_dict[current_model_name] = (pd.Series(dtype=float), None)  # Store empty results
        except Exception as e:
            logger.error(f"Error evaluating model {current_model_name}: {e}", exc_info=debug)
            results_dict[current_model_name] = (pd.Series(dtype=float), None)  # Store empty results on error

    # Aggregate results to CSV
    rows_for_df = []
    header_row = ["Model"]
    first_model_processed = False

    for model_name_key, (metrics, _) in results_dict.items():
        if metrics is not None and not metrics.empty:
            if not first_model_processed:
                header_row.extend(list(metrics.keys()))
                first_model_processed = True
            current_row_values = [model_name_key] + list(metrics.values())
            rows_for_df.append(current_row_values)
        else:
            # Ensure a row is added even if metrics are empty/None, with NAs for metric values
            if not first_model_processed and header_row == ["Model"]:  # If no model had metrics yet
                # Cannot determine header from empty metrics, could use a predefined list or skip
                logger.warning(f"No metrics available to determine CSV header from model {model_name_key}.")
            # Create a row with NaNs for this model
            num_metrics = len(header_row) - 1 if len(header_row) > 1 else 0  # Number of expected metric columns
            current_row_values = [model_name_key] + [np.nan] * num_metrics
            rows_for_df.append(current_row_values)

    if not rows_for_df:
        logger.warning("No evaluation results to save to CSV.")
        return

    # Prepend header to rows_for_df for DataFrame creation
    if first_model_processed:  # Header was successfully created
        df_data = [header_row] + rows_for_df
        df = pd.DataFrame(df_data[1:], columns=df_data[0])
    else:  # No model produced metrics, create DataFrame with just model names
        df = pd.DataFrame(rows_for_df, columns=["Model"])

    csv_path = Path(report_filename).with_suffix(".csv")
    try:
        df.to_csv(csv_path, index=False)  # header=True is default for df.to_csv
        logger.info(f"Aggregated evaluation metrics saved to {csv_path}")
    except Exception as e:
        logger.error(f"Failed to save aggregated CSV results: {e}")


@app.command()
def sanity_check_model(model_url: str, use_local_environement: bool = False, dataset_path: Optional[str] = None):
    """
    Quick test to ensure a model can be loaded, trained, and run on a small dataset.

    This command attempts to load a model from the given URL (or local path),
    train it on a small sample dataset (either specified or a default one),
    and generate predictions. It checks for NaN values in predictions.

    Args:
        model_url (str): URL or local path to the model directory.
        use_local_environement (bool): If True, ignores the model's Conda environment
                                       and uses the current local environment.
        dataset_path (Optional[str]): Path to a CSV file for a custom dataset.
                                      If None, a default small dataset is used.
    Returns:
        bool: True if the sanity check passes, False otherwise.
    """
    initialize_logging()  # Basic logging for this command
    logger.info(f"Starting sanity check for model: {model_url}")
    dataset = DataSet.from_csv(dataset_path, FullData) if dataset_path else datasets["hydromet_5_filtered"].load()
    logger.info(f"Using dataset: {'custom from ' + dataset_path if dataset_path else 'hydromet_5_filtered'}")

    # Generate a single train/test split
    train_data, test_data_generator = train_test_generator(dataset, prediction_delta=3 * delta_month, n_test_sets=1)

    try:
        context_data, future_weather_data, true_values = next(test_data_generator)
    except StopIteration:
        logger.error("Failed to generate a train/test split from the dataset. Dataset might be too small.")
        return False

    try:
        logger.info("Loading model...")
        model_template = get_model_from_directory_or_github_url(model_url, ignore_env=use_local_environement)
        estimator = model_template()  # Instantiate the model

        logger.info("Training model...")
        predictor = estimator.train(train_data)

        logger.info("Generating predictions...")
        predictions = predictor.predict(context_data, future_weather_data)

        if not predictions:
            logger.error("Prediction dictionary is empty.")
            return False

        for loc, pred_data in predictions.items():
            if hasattr(pred_data, "samples") and np.isnan(pred_data.samples).any():
                logger.error(f"NaN values found in predictions for location {loc}.")
                return False
            elif hasattr(pred_data, "values") and np.isnan(pred_data.values).any():  # For summary objects
                logger.error(f"NaN values found in prediction summary for location {loc}.")
                return False

        logger.info("Sanity check passed successfully.")
        return True
    except Exception as e:
        logger.error(f"Sanity check failed: {e}", exc_info=True)
        return False


@app.command()
def forecast(
    model_name: str,  # Can be a registered name or a URL/path
    dataset_name: DataSetType,
    n_months: int,
    model_path: Optional[str] = None,  # Redundant if model_name can be a path/URL
    out_path: Optional[str] = "./",
):
    """
    Forecast `n_months` ahead using a given model and dataset.

    This command loads a specified model (either by registered name or from a path/URL
    if `model_name` is a path/URL or if `model_path` is provided) and a dataset.
    It then generates forecasts for the specified number of months and saves
    the resulting plots as an HTML file.

    Args:
        model_name (str): Name of the registered model, or path/URL to a model directory.
        dataset_name (DataSetType): Key for the predefined dataset to use.
        n_months (int): Number of months to forecast into the future.
        model_path (Optional[str]): Deprecated/alternative way to specify model path if `model_name` is just a name.
                                   Prefer using `model_name` for path/URL directly.
        out_path (Optional[str]): Directory where the output HTML file will be saved. Defaults to current directory.
    """
    initialize_logging()
    logger.info(f"Generating forecast: model='{model_name}', dataset='{dataset_name}', months={n_months}")

    # Consolidate model_name and model_path logic
    actual_model_identifier = model_path if model_path else model_name

    output_dir = Path(out_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Sanitize dataset_name if it's an enum for filename
    dataset_name_str = dataset_name.value if isinstance(dataset_name, DataSetType) else str(dataset_name)
    # Sanitize model_name for filename (e.g. replace slashes from URLs)
    safe_model_name = (
        actual_model_identifier.split("/")[-1].replace(".git", "")
        if "/" in actual_model_identifier
        else actual_model_identifier
    )

    output_file = output_dir / f"{safe_model_name}_{dataset_name_str}_forecast_results_{n_months}months.html"

    try:
        with open(output_file, "w") as f:
            # api.forecast expects model_name and optionally model_path
            # If actual_model_identifier is a path/URL, model_name for api.forecast should be 'external'
            # and model_path should be actual_model_identifier.
            # If actual_model_identifier is a registered name, model_name is actual_model_identifier and model_path is None.

            api_model_name = actual_model_identifier
            api_model_path = None
            if "/" in actual_model_identifier or Path(actual_model_identifier).exists():  # Heuristic for path/URL
                api_model_name = "external"  # Special key for api.forecast
                api_model_path = actual_model_identifier

            figs = api.forecast(api_model_name, dataset_name, n_months, api_model_path)
            if not figs:
                logger.warning("No figures were generated by the forecast.")
                f.write("<p>No forecast figures generated.</p>")
            else:
                for fig_index, fig in enumerate(figs):
                    f.write(f"<h2>Forecast Plot {fig_index + 1}</h2>")
                    f.write(fig.to_html(full_html=False, include_plotlyjs="cdn"))
            logger.info(f"Forecast plots saved to {output_file}")
    except Exception as e:
        logger.error(f"Error during forecast generation or saving: {e}", exc_info=True)


@app.command()
def multi_forecast(
    model_name: str,  # Path or URL to the model
    dataset_name: DataSetType,
    n_months: int,
    pre_train_months: int,
    out_path: Path = Path("."),  # Use Path type hint directly
):
    """
    Run multi-forecasting using a sliding training window and save visual output.

    This command repeatedly trains a model on a sliding window of past data and
    forecasts `n_months` ahead. It's useful for evaluating model performance
    over different time periods. Plots are saved to an HTML file.

    Args:
        model_name (str): Path or URL to the model directory.
        dataset_name (DataSetType): Key for the predefined dataset to use.
        n_months (int): Number of months to forecast ahead in each window.
        pre_train_months (int): Number of months to use for the initial training period.
        out_path (Path): Directory where the output HTML file will be saved.
    """
    initialize_logging()
    logger.info(
        f"Starting multi-forecast: model='{model_name}', dataset='{dataset_name}', months={n_months}, pre-train={pre_train_months}"
    )

    out_path.mkdir(parents=True, exist_ok=True)

    try:
        model_instance = get_model_from_directory_or_github_url(model_name)()  # Instantiate
        model_display_name = model_instance.name if hasattr(model_instance, "name") else Path(model_name).stem
        dataset_name_str = dataset_name.value if isinstance(dataset_name, DataSetType) else str(dataset_name)

        filename = out_path / f"{model_display_name}_{dataset_name_str}_multi_forecast_results_{n_months}months.html"
        dataset = datasets[dataset_name].load()

        with open(filename, "w") as f:
            f.write(f"<h1>Multi-Forecast Results for {model_display_name} on {dataset_name_str}</h1>")
            f.write(f"<p>Forecasting {n_months} months ahead, with {pre_train_months} pre-train months.</p>")

            predictions_iterable = do_multi_forecast(
                model_instance, dataset, n_months * delta_month, pre_train_months * delta_month
            )

            # Collect all predictions first to handle potential errors early
            all_predictions = list(predictions_iterable)
            if not all_predictions:
                logger.warning("No predictions generated by multi_forecast.")
                f.write("<p>No predictions were generated.</p>")
                return

            for location_id, truth_series in dataset.items():
                # Extract predictions for the current location from each forecast step
                location_predictions = []
                for forecast_step_prediction_set in all_predictions:
                    if location_id in forecast_step_prediction_set:
                        location_predictions.append(forecast_step_prediction_set.get_location(location_id).data())

                if not location_predictions:
                    logger.warning(f"No predictions found for location {location_id} in multi_forecast results.")
                    continue

                fig = plot_forecast_from_summaries(location_predictions, truth_series.data())
                f.write(f"<h2>Forecast for Location: {location_id}</h2>")
                f.write(fig.to_html(full_html=False, include_plotlyjs="cdn"))
            logger.info(f"Multi-forecast plots saved to {filename}")

    except Exception as e:
        logger.error(f"Error during multi-forecast: {e}", exc_info=True)


@app.command()
def serve(
    seedfile: Optional[str] = None,
    debug: bool = False,
    auto_reload: bool = False,
    host: str = "127.0.0.1",
    port: int = 8000,
):
    """
    Start the CHAP REST API server using Uvicorn.

    Args:
        seedfile (Optional[str]): Path to a JSON file to seed initial data for the API.
        debug (bool): If True, enable debug mode for the server (may affect auto_reload).
        auto_reload (bool): If True, enable auto-reloading of the server on code changes.
                            Typically used during development.
        host (str): Host address to bind the server to.
        port (int): Port number for the server.
    """
    initialize_logging(debug_on=debug)  # debug_on is the param name for initialize_logging
    logger.info(f"Preparing to start CHAP API server on {host}:{port}")

    try:
        import uvicorn

        from chap_core.rest_api_src.v1.rest_api import main_backend_app  # Assuming this is the FastAPI app instance
    except ImportError:
        logger.error("Failed to import necessary modules for serving (FastAPI/Uvicorn). Ensure they are installed.")
        return

    initial_data = None
    if seedfile:
        try:
            with open(seedfile, "r") as f:
                initial_data = json.load(f)
            logger.info(f"Loaded seed data from {seedfile}")
        except Exception as e:
            logger.error(f"Failed to load seed file {seedfile}: {e}")
            # Decide if to proceed without seed data or exit
            # return

    # How main_backend_app uses initial_data needs to be handled.
    # If it's a global or needs to be passed at startup, this might need adjustment.
    # For now, assuming uvicorn.run can launch the app directly.
    # The original call was main_backend(data, auto_reload=auto_reload)
    # This implies main_backend was a function that started uvicorn.
    # If main_backend_app is the app, we call uvicorn.run directly.

    # If main_backend was indeed the Uvicorn runner:
    # from chap_core.rest_api_src.v1.rest_api import main_backend
    # logger.info("Starting CHAP API server...")
    # main_backend(initial_data, auto_reload=auto_reload, host=host, port=port) # Assuming main_backend accepts host/port

    # If main_backend_app is the FastAPI instance:
    uvicorn.run(main_backend_app, host=host, port=port, reload=auto_reload, log_level="debug" if debug else "info")
    logger.info("CHAP API server has shut down.")


@app.command()
def write_open_api_spec(out_path: str = "openapi.json"):
    """
    Export the OpenAPI schema of the CHAP REST API to a JSON file.

    Args:
        out_path (str): The file path where the OpenAPI JSON specification will be saved.
                        Defaults to "openapi.json" in the current directory.
    """
    initialize_logging()
    logger.info(f"Exporting OpenAPI schema to {out_path}")
    try:
        from chap_core.rest_api_src.v1.rest_api import get_openapi_schema

        schema = get_openapi_schema()
        with open(out_path, "w") as f:
            json.dump(schema, f, indent=4)
        logger.info(f"OpenAPI schema successfully saved to {out_path}")
    except ImportError:
        logger.error("Failed to import get_openapi_schema. Ensure API components are available.")
    except Exception as e:
        logger.error(f"Failed to write OpenAPI spec: {e}", exc_info=True)


@app.command()
def test(**kwargs):
    """
    Run a simple smoke test for the CLI system.

    This command initializes logging and prints info/debug messages
    to confirm basic CLI and logging functionality.
    It accepts arbitrary keyword arguments which are currently ignored.
    """
    # debug_on might be passed via kwargs if cyclopts supports it or if we parse it manually
    debug_mode = kwargs.get("debug", False)
    initialize_logging(debug_on=debug_mode)  # Pass the debug flag
    logger.debug("Debug mode is active for test command.")
    logger.info("Test command executed successfully. Logging is configured.")
    if kwargs:
        logger.debug(f"Test command received additional arguments: {kwargs}")


@app.command()
def backtest(
    data_filename: Path,
    model_name: registry.model_type | str,  # Accepts registered name or path/URL
    out_folder: Path,
    prediction_length: int = 12,
    n_test_sets: int = 20,
    stride: int = 2,
):
    """
    Run a backtest over a full dataset and write both CSV and JSON outputs.

    This command performs backtesting by training a model on historical data and
    predicting future periods, simulating how the model would have performed in the past.
    It saves detailed prediction entries and evaluation responses.

    Args:
        data_filename (Path): Path to the CSV dataset file.
        model_name (str): Registered model identifier or path/URL to a model directory.
        out_folder (Path): Directory where the output CSV and JSON files will be saved.
        prediction_length (int): Number of time periods to predict ahead in each backtest window.
        n_test_sets (int): Number of test sets (backtesting windows) to evaluate.
        stride (int): Step size to move the training window forward for each test set.
    """
    initialize_logging()
    logger.info(f"Running backtest on {data_filename} with model: {model_name}")
    out_folder.mkdir(parents=True, exist_ok=True)

    try:
        dataset = DataSet.from_csv(data_filename, FullData)
    except FileNotFoundError:
        logger.error(f"Data file not found: {data_filename}")
        return
    except Exception as e:
        logger.error(f"Error loading dataset from {data_filename}: {e}", exc_info=True)
        return

    try:
        # Determine if model_name is a path/URL or a registered name
        if "/" in model_name or Path(model_name).exists():  # Heuristic for path/URL
            logger.info(f"Loading model from path/URL: {model_name}")
            estimator_class = get_model_from_directory_or_github_url(model_name)
        else:
            logger.info(f"Loading registered model: {model_name}")
            estimator_class = registry.get_model(model_name)  # This should return the class

        # Instantiate the estimator
        estimator_instance = estimator_class()

    except KeyError:  # For registry.get_model
        logger.error(f"Model '{model_name}' not found in registry.")
        return
    except Exception as e:
        logger.error(f"Error loading or instantiating model '{model_name}': {e}", exc_info=True)
        return

    try:
        logger.info(
            f"Starting backtesting with prediction length {prediction_length}, {n_test_sets} sets, stride {stride}."
        )
        predictions = _backtest(
            estimator_instance, dataset, prediction_length, n_test_sets, stride, weather_provider=QuickForecastFetcher
        )

        # Assuming 'dengue' is a placeholder or needs to be configurable for dataset_to_datalist
        # This might need to come from dataset metadata or a parameter
        disease_identifier = "dengue"  # Placeholder
        logger.info(f"Converting predictions to evaluation response for disease: {disease_identifier}")
        response = samples_to_evaluation_response(
            predictions,
            quantiles=[0.05, 0.25, 0.5, 0.75, 0.95],
            real_data=dataset_to_datalist(dataset, disease_identifier),
        )

        safe_model_name_part = Path(model_name).stem if "/" in model_name else model_name
        csv_output_path = out_folder / f"{data_filename.stem}_evaluation_{safe_model_name_part}.csv"
        json_output_path = out_folder / f"{data_filename.stem}_evaluation_response_{safe_model_name_part}.json"

        logger.info(f"Saving CSV results to: {csv_output_path}")
        df = pd.DataFrame([entry.model_dump() for entry in response.predictions])
        df.to_csv(csv_output_path, index=False)

        logger.info(f"Saving JSON response to: {json_output_path}")
        with open(json_output_path, "w") as f:
            f.write(response.model_dump_json(indent=4))  # Use model_dump_json for Pydantic models

        logger.info("Backtest completed successfully.")

    except Exception as e:
        logger.error(f"Error during backtesting process: {e}", exc_info=True)


@dataclasses.dataclass
class AreaPolygons:
    """
    Represents geographic area polygons.

    This class is currently a placeholder and intended for future integration
    with geographic data handling and schema definitions. Its attributes and methods
    are yet to be defined.
    """

    # Attributes for polygon data would be defined here, e.g.:
    # features: list[Any] # Or a more specific GeoJSON feature type
    # crs: Optional[str] = None
    pass  # Ellipsis (...) removed as it's not standard for empty class body if docstring exists.


def main_function():
    """
    Placeholder for a potential callable main entry point of the application.

    This function is reserved for future use, possibly as a direct programmatic
    entry point if the CLI structure evolves or if chap-core is used as a library
    in a way that requires such an entry point. Currently, it does not perform any action.
    """
    logger.info("main_function called (placeholder).")
    # Potential future logic here
    return


def main():
    """
    Main entry point for the CHAP-core CLI application.

    This function initializes and runs the `cyclopts` application, making all
    registered commands available from the command line.
    """
    # Consider global logging initialization here if not done per-command
    # initialize_logging()
    try:
        app()
    except Exception as e:
        # Catch-all for errors not handled by cyclopts or commands themselves
        logger.critical(f"Unhandled error in CLI execution: {e}", exc_info=True)
        # Optionally, exit with a non-zero status code
        # import sys
        # sys.exit(1)


if __name__ == "__main__":
    main()
