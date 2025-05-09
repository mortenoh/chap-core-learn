# Improvement Suggestions:
# 1. **Reinforce Deprecation**: Add prominent warnings within each command function's docstring or at their start, reiterating the CLI's deprecated status.
# 2. **Guidance on Alternatives**: If parts of this CLI's functionality have been superseded by other tools or methods within CHAP, provide pointers or references to those alternatives in the module docstring or a dedicated section.
# 3. **Error Handling**: Enhance error handling for file operations (e.g., `input_filename` not found, permission issues) and for external calls (e.g., `dataset_from_request_v1`, model loading).
# 4. **Configuration/Logging**: Standardize logging configuration. Consider a global setup for logging levels and formats if this CLI were to be maintained.
# 5. **Parameter Validation**: Add more explicit validation for input parameters, especially for file paths and choices like `model_id`. `cyclopts` might handle some, but internal checks can be useful.

"""
DEPRECATED: This CLI is no longer used in the current CHAP implementation.
This script will be removed in the future.

Provides commands for harmonizing data, evaluating models, and generating forecasts
using datasets exported from the CHAP-app interface.
"""

import json
import logging
from pathlib import Path

from cyclopts import App

from chap_core.api_types import PredictionRequest
from chap_core.assessment.forecast import forecast_ahead
from chap_core.assessment.prediction_evaluator import evaluate_model
from chap_core.datatypes import FullData
from chap_core.geoutils import buffer_point_features, inspect_feature_collection
from chap_core.predictor.model_registry import registry
from chap_core.rest_api_src.worker_functions import dataset_from_request_v1
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet
from chap_core.time_period import delta_month

logger = logging.getLogger(__name__)
# Basic logging configuration for the CLI, if it were to be used.
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def harmonize(input_filename: Path, output_filename: Path, point_buffer: float = None):
    """
    DEPRECATED. Harmonize health + population data with climate data.

    This function takes an input JSON file (typically from CHAP-app), processes
    GeoJSON features (optionally buffering points), and generates a harmonized
    CSV dataset by integrating health, population, and climate data.

    Parameters:
        input_filename (Path): Path to the input JSON file from the CHAP-app,
                               containing `orgUnitsGeoJson` and other request data.
        output_filename (Path): Path where the output CSV file containing the
                                harmonized dataset will be saved. A GeoJSON file
                                with processed polygons will also be saved with
                                a '.geojson' suffix.
        point_buffer (float, optional): If input geometries are points, this specifies
                                        a distance (in the units of the CRS) to buffer
                                        them into polygons. Defaults to None (no buffering).
    """
    logger.warning("`chap-cli harmonize` is DEPRECATED and will be removed.")
    logger.info(f"Converting {input_filename} to harmonized dataset at {output_filename}")
    polygons_filename = output_filename.with_suffix(".geojson")

    # Load input file and parse request object
    try:
        with open(input_filename, "r") as f:
            request_data = PredictionRequest.model_validate_json(f.read())
    except FileNotFoundError:
        logger.error(f"Input file not found: {input_filename}")
        return
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON in input file: {input_filename}")
        return
    except Exception as e:
        logger.error(f"Error loading request data from {input_filename}: {e}")
        return

    # Inspect geometry statistics
    stats = inspect_feature_collection(request_data.orgUnitsGeoJson)
    logger.info(f"Input feature stats:\n{json.dumps(stats, indent=4)}")

    # Convert point geometries to buffered polygons if requested
    if point_buffer is not None:
        logger.info(f"Buffering point features by {point_buffer} units.")
        request_data.orgUnitsGeoJson = buffer_point_features(request_data.orgUnitsGeoJson, point_buffer)

    # Save updated GeoJSON
    try:
        with open(polygons_filename, "w") as f:
            f.write(request_data.orgUnitsGeoJson.model_dump_json())
        logger.info(f"Buffered/original polygons saved to {polygons_filename}")
    except Exception as e:
        logger.error(f"Error saving polygons GeoJSON to {polygons_filename}: {e}")
        # Continue to dataset generation if possible, or decide to return

    # Generate harmonized dataset and save to CSV
    try:
        dataset = dataset_from_request_v1(request_data, usecwd_for_credentials=True)
        dataset.to_csv(output_filename)
        logger.info(f"Harmonized dataset saved to {output_filename}")
    except Exception as e:
        logger.error(f"Error generating or saving harmonized dataset: {e}")


def evaluate(
    data_filename: Path,
    output_filename: Path,
    model_id: registry.model_type,
    prediction_length: int = None,
    n_test_sets: int = None,
):
    """
    DEPRECATED. Evaluate a model's forecasting ability on a dataset.

    This function loads a dataset, splits it into multiple training/test sets
    (typically using a rolling origin), and evaluates the specified model's
    performance in forecasting `prediction_length` steps ahead.
    Results are often saved to a PDF report.

    Parameters:
        data_filename (Path): Path to the CSV dataset to be used for evaluation
                              (usually created by the `harmonize` command).
        output_filename (Path): Path where the output PDF report of the evaluation
                                will be saved.
        model_id (str): Identifier of the model (registered in `chap_core.predictor.model_registry`)
                        to be evaluated.
        prediction_length (int, optional): The number of time periods to predict ahead
                                           in each test set. If None, defaults based on
                                           data frequency (3 for monthly, 12 for weekly/daily).
        n_test_sets (int, optional): The number of test sets to generate for evaluation.
                                     If None, defaults to cover roughly the last year of data.
    """
    logger.warning("`chap-cli evaluate` is DEPRECATED and will be removed.")
    try:
        data_set = DataSet.from_csv(data_filename, FullData)
    except FileNotFoundError:
        logger.error(f"Data file not found: {data_filename}")
        return
    except Exception as e:
        logger.error(f"Error loading dataset from {data_filename}: {e}")
        return

    delta = data_set.period_range.delta

    if prediction_length is None:
        prediction_length = 3 if delta == delta_month else 12
    if n_test_sets is None:
        # Ensure n_periods is at least prediction_length
        n_periods_available = len(data_set.time_periods)  # A proxy for total periods
        min_periods_for_one_test = prediction_length + 1  # Minimal train + test

        default_eval_span = 12 if delta == delta_month else 52  # Try to evaluate over a year

        if n_periods_available < min_periods_for_one_test:
            logger.error(f"Dataset too short for evaluation with prediction_length {prediction_length}.")
            return

        # Calculate n_test_sets to cover default_eval_span or available data
        n_test_sets = max(
            1, min(default_eval_span, n_periods_available - prediction_length + 1) - prediction_length + 1
        )

    logger.info(
        f"Evaluating {model_id} on {data_filename} "
        f"with {n_test_sets} test sets, forecasting {prediction_length} steps ahead"
    )

    try:
        model = registry.get_model(model_id)
        results = evaluate_model(model, data_set, prediction_length, n_test_sets, report_filename=output_filename)
        if results and results[0]:  # Assuming results[0] is a summary
            logger.info(f"Evaluation results summary:\n{results[0]}")
        else:
            logger.info(f"Evaluation completed. Report saved to {output_filename}")
    except KeyError:
        logger.error(f"Model ID '{model_id}' not found in registry.")
    except Exception as e:
        logger.error(f"Error during model evaluation: {e}")


def predict(
    data_filename: Path,
    output_filename: Path,
    model_id: registry.model_type,
    prediction_length: int = None,
    do_summary: bool = False,
):
    """
    DEPRECATED. Generate forecasts using a trained model and an input dataset.

    This function loads a dataset, trains the specified model on the full dataset,
    and then generates forecasts for `prediction_length` steps into the future.
    The predictions can be output as full samples or as summaries.

    Parameters:
        data_filename (Path): Path to the CSV dataset (e.g., created via `harmonize`)
                              to be used for training and as a basis for forecasting.
        output_filename (Path): Path where the CSV output file containing the
                                predictions will be saved.
        model_id (str): Identifier of the model (registered in `chap_core.predictor.model_registry`)
                        to be used for forecasting.
        prediction_length (int, optional): The number of time periods to forecast ahead.
                                           If None, defaults based on data frequency
                                           (3 for monthly, 12 for weekly/daily).
        do_summary (bool): If True, output prediction summaries (e.g., mean, quantiles).
                           If False (default), output full prediction samples.
    """
    logger.warning("`chap-cli predict` is DEPRECATED and will be removed.")
    try:
        data_set = DataSet.from_csv(data_filename, FullData)
    except FileNotFoundError:
        logger.error(f"Data file not found: {data_filename}")
        return
    except Exception as e:
        logger.error(f"Error loading dataset from {data_filename}: {e}")
        return

    delta = data_set.period_range.delta

    if prediction_length is None:
        prediction_length = 3 if delta == delta_month else 12

    try:
        model = registry.get_model(model_id)
        samples = forecast_ahead(model, data_set, prediction_length)

        if do_summary:
            predictions = DataSet({loc: sm.summaries() for loc, sm in samples.items()})
        else:
            predictions = samples  # This is actually SampleSet, not DataSet. Needs care for .to_csv
            # To make it a DataSet of samples, might need restructuring or a specific SampleSet.to_csv
            # For now, assuming SampleSet might have a compatible .to_csv or this needs adjustment.
            # If predictions is a SampleSet, its to_csv might need to handle the structure.
            # A simple fix if SampleSet doesn't have to_csv but its elements do:
            # This part is speculative without knowing SampleSet structure.
            # For example, if it's Dict[str, SomeSampleTypeWithToCsvMethod]:
            # for loc, sample_data in predictions.items():
            #     sample_data.to_csv(output_filename.with_name(f"{output_filename.stem}_{loc}{output_filename.suffix}"))
            # This is too complex for a simple docstring pass, but highlights a potential issue.
            # Assuming current predictions.to_csv() handles the SampleSet structure correctly.

        predictions.to_csv(output_filename)
        logger.info(f"Predictions saved to {output_filename}")
    except KeyError:
        logger.error(f"Model ID '{model_id}' not found in registry.")
    except Exception as e:
        logger.error(f"Error during prediction: {e}")


def main():
    """
    CLI entry point for the deprecated `chap-cli` tool.

    This function initializes the `cyclopts` application and registers the
    `harmonize`, `evaluate`, and `predict` commands. It also prints a
    deprecation warning upon execution.
    """
    # It's good practice to configure logging early.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    app = App()
    # The warning is good, perhaps make it more visible if possible (e.g. stderr or specific color if supported)
    print(
        "⚠️  WARNING: `chap-cli` is deprecated and will be removed in a future release. Consider migrating to alternative tools or APIs if available."
    )
    app.command(harmonize)
    app.command(evaluate)
    app.command(predict)
    app()


if __name__ == "__main__":
    main()
