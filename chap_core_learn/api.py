# Improvement Suggestions:
# 1. The `DummyControl` class seems like a placeholder. If it's still needed, its purpose and potential future implementation should be clarified. Otherwise, consider removing it if it's dead code.
# 2. `AreaPolygons` is also a placeholder. Define its structure or integrate it with a proper GIS library if actual polygon data is to be handled.
# 3. The `extract_disease_name` function makes a strong assumption about data structure ("first row's first column"). This is brittle. Consider a more robust way to identify or pass the disease name, perhaps via metadata.
# 4. The `train_with_validation` and `forecast` functions return `list[Figure]`. Consider returning structured data instead of (or in addition to) plots for easier programmatic use of results.
# 5. Error handling could be improved. For instance, what happens if `dataset_name` is invalid or a model fails to load/train?

"""
This module provides high-level functions for training models, generating forecasts,
and handling related data structures within the CHAP (Climate and Health Analytics Platform) core.
It serves as an interface for common forecasting workflows.
"""

import dataclasses
import logging
from typing import Any, Dict, List, Optional  # Added Any, Dict for extract_disease_name

from .assessment.dataset_splitting import train_test_split_with_weather
from .assessment.forecast import forecast as do_forecast
from .datatypes import ClimateData, HealthData, HealthPopulationData
from .file_io.example_data_set import DataSetType, datasets
from .models.utils import get_model_from_directory_or_github_url
from .plotting.prediction_plot import plot_forecast_from_summaries
from .predictor import get_model
from .spatio_temporal_data.temporal_dataclass import DataSet
from .time_period.date_util_wrapper import Month, delta_month
from .transformations.covid_mask import mask_covid_data

logger = logging.getLogger(__name__)


# --- Utility classes ---


class DummyControl:
    """
    A dummy placeholder for a control interface.

    This class is intended for situations where a control object is expected
    but no actual control functionality is needed or available.
    """

    def set_status(self, status: Any):
        """
        Sets a status message. In this dummy implementation, it does nothing.

        Args:
            status: The status to set.
        """
        pass

    @property
    def current_control(self) -> None:
        """
        Gets the current control object. In this dummy implementation, it returns None.

        Returns:
            None.
        """
        return None


@dataclasses.dataclass
class AreaPolygons:
    """
    Placeholder for geographic shape reference.

    Attributes:
        shape_file (str): Path or identifier for the shape file.
    """

    shape_file: str


@dataclasses.dataclass
class PredictionData:
    """
    Container for prediction-related datasets and metadata.

    This dataclass aggregates various data components required for or produced by
    prediction processes.
    """

    area_polygons: Optional[AreaPolygons] = None
    health_data: Optional[DataSet[HealthData]] = None
    climate_data: Optional[DataSet[ClimateData]] = None
    population_data: Optional[DataSet[HealthPopulationData]] = None
    disease_id: Optional[str] = None
    features: Optional[List[object]] = None  # Consider defining a more specific type for features


# --- Utility functions ---


def extract_disease_name(health_data: Dict[str, List[List[Any]]]) -> str:
    """
    Extract disease name from uploaded health data.

    Assumes the input `health_data` is a dictionary with a "rows" key,
    where `health_data["rows"]` is a list of lists, and the disease name
    is located in the first cell of the first row (e.g., `health_data["rows"][0][0]`).

    Args:
        health_data (Dict[str, List[List[Any]]]): A dictionary representing health data,
            expected to have a "rows" key containing a list of data rows.

    Returns:
        str: The extracted disease name.

    Raises:
        IndexError: If the "rows" list is empty or the first row is empty.
        KeyError: If the "rows" key is not found in `health_data`.
    """
    return health_data["rows"][0][0]


# --- Forecasting ---


def train_with_validation(model_name: str, dataset_name: DataSetType, n_months: int = 12):
    """
    Train a model with validation using a specified dataset and forecast horizon.

    This function loads a dataset, splits it into training and test sets,
    initializes and trains the specified model, generates a forecast,
    and returns plots of the forecast against actual data.

    Parameters:
        model_name (str): Name of the model to use (e.g., "ewars_plus_model").
        dataset_name (DataSetType): An enum or string key identifying the dataset to load.
        n_months (int): The number of months into the future to forecast. Defaults to 12.

    Returns:
        list[Figure]: A list of matplotlib Figure objects, each plotting the forecast
                      for a specific location within the dataset.
    """
    dataset = datasets[dataset_name].load()
    dataset = mask_covid_data(dataset)

    # Set up forecast window
    prediction_length = n_months * delta_month
    split_point = dataset.end_timestamp - prediction_length
    split_period = Month(split_point.year, split_point.month)

    # Split data
    train_data, test_set, future_weather = train_test_split_with_weather(dataset, split_period)

    # Initialize and train model
    model = get_model(model_name)(n_iter=32000)  # Consider making n_iter configurable
    model.set_validation_data(test_set)
    model.train(train_data)

    # Generate forecast
    predictions = model.forecast(
        future_weather, forecast_delta=prediction_length, n_samples=100
    )  # n_samples configurable?

    # Plot forecasts
    figs = []
    for location, prediction in predictions.items():
        fig = plot_forecast_from_summaries(prediction.data(), dataset.get_location(location).data())
        figs.append(fig)
    return figs


def forecast(
    model_name: str,
    dataset_name: DataSetType,
    n_months: int,
    model_path: Optional[str] = None,
):
    """
    Run a forecast using a specified model, dataset, and forecast horizon.

    This function loads a dataset and a model (either built-in or external via `model_path`),
    generates a forecast for the specified number of months, and returns plots of the results.

    Parameters:
        model_name (str): Name of the model. If "external", `model_path` must be provided.
        dataset_name (DataSetType): An enum or string key identifying the dataset to load.
        n_months (int): Number of months to forecast ahead.
        model_path (Optional[str]): Path or GitHub URL to an external model.
                                    Required if `model_name` is "external". Defaults to None.

    Returns:
        list[Figure]: A list of matplotlib Figure objects, each plotting the forecast
                      for a specific location within the dataset.
    """
    logging.basicConfig(level=logging.INFO)  # Consider if this should be configured globally

    dataset = datasets[dataset_name].load()
    forecast_horizon = n_months * delta_month

    # Load model
    if model_name == "external":
        if not model_path:
            raise ValueError("model_path must be provided when model_name is 'external'.")
        model = get_model_from_directory_or_github_url(model_path)
    else:
        model = get_model(model_name)()

    # Generate forecast
    predictions = do_forecast(model, dataset, forecast_horizon)

    # Plot and return figures
    figs = []
    for location, prediction in predictions.items():
        fig = plot_forecast_from_summaries(prediction.data(), dataset.get_location(location).data())
        figs.append(fig)
    return figs
