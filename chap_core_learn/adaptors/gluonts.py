# Improvement Suggestions:
# 1. **Comprehensive Docstrings**: Add a module-level docstring and detailed docstrings for the `GluonTSPredictor` and `GluonTSEstimator` classes and all their methods. Explain their roles as adaptors, the purpose of their parameters, what they return, and how they interact with the GluonTS library. (Primary task).
# 2. **Specific Type Hinting**: Enhance type hints for `DataSet` arguments (e.g., `DataSet[SomeSpecificDataType]`) and clarify the types returned by `DataSetAdaptor` methods if possible. Ensure all methods have explicit return type hints.
# 3. **Robust Error Handling**: Implement `try-except` blocks for operations involving the GluonTS library (training, prediction, serialization, deserialization) and file system interactions (`save`, `load`) to catch and handle potential errors gracefully (e.g., GluonTS-specific exceptions, `FileNotFoundError`, `IOError`, pickling/serialization errors).
# 4. **Dynamic `freq` in `ListDataset`**: The frequency (`freq="m"`) for `gluonts.dataset.common.ListDataset` in `GluonTSEstimator.train` is hardcoded. This should be made dynamic, ideally inferred from the input `dataset.time_period.freq_str` (if available) or passed as a parameter, as GluonTS models are sensitive to this.
# 5. **Context for `DataSetAdaptor`**: Add a brief comment explaining the role of `DataSetAdaptor` (imported from `..data.gluonts_adaptor.dataset`) in converting between CHAP-core's `DataSet` format and the format expected by GluonTS.

"""
This module provides adaptor classes that bridge CHAP-core's data structures
and model interfaces with the GluonTS time series forecasting library.

It defines:
- `GluonTSPredictor`: A wrapper around a trained GluonTS `Predictor` object,
  adapting its `predict` method to work with CHAP-core `DataSet` objects and
  providing `save`/`load` functionality.
- `GluonTSEstimator`: A wrapper around a GluonTS `Estimator` object,
  adapting its `train` method to accept a CHAP-core `DataSet` and return
  a `GluonTSPredictor`.

These adaptors rely on `DataSetAdaptor` (from `chap_core.data.gluonts_adaptor`)
to handle the conversion between CHAP-core's `DataSet` and GluonTS's dataset formats.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Type  # Added for type hints

from gluonts.dataset.common import ListDataset
from gluonts.model.estimator import Estimator as GluonTSEstimatorNative  # Alias to avoid name clash
from gluonts.model.predictor import Predictor as GluonTSPredictorNative  # Alias

from ..data import DataSet  # Assuming this is chap_core.spatio_temporal_data.temporal_dataclass.DataSet
from ..data.gluonts_adaptor.dataset import DataSetAdaptor  # Handles conversion to GluonTS datasets
from ..datatypes import Samples, TimeSeriesData  # Samples for output, TimeSeriesData for type hints
from ..time_period import PeriodRange

logger = logging.getLogger(__name__)


@dataclass
class GluonTSPredictor:
    """
    A wrapper for a pre-trained GluonTS `Predictor` object.

    This class adapts the GluonTS predictor to integrate with CHAP-core's
    `DataSet` objects for making predictions. It also provides methods for
    serializing and deserializing the underlying GluonTS predictor.

    Attributes:
        gluonts_predictor (GluonTSPredictorNative): The underlying GluonTS predictor instance.
    """

    gluonts_predictor: GluonTSPredictorNative

    def predict(
        self, history: DataSet[TimeSeriesData], future_data: DataSet[TimeSeriesData], num_samples: int = 100
    ) -> DataSet[Samples]:
        """
        Generates forecasts using the wrapped GluonTS predictor.

        Converts CHAP-core `DataSet` objects (history and future covariate data)
        into a format suitable for GluonTS, then calls the predictor's `predict` method.
        The GluonTS forecast results are then converted back into a CHAP-core `DataSet`
        of `Samples`.

        Args:
            history (DataSet[TimeSeriesData]): Historical data leading up to the forecast point.
                                               The specific TimeSeriesData type should match what the
                                               model was trained on.
            future_data (DataSet[TimeSeriesData]): Future values of any dynamic features/covariates
                                                  required by the GluonTS model.
            num_samples (int): The number of sample paths to generate from the forecast distribution.
                               Defaults to 100.

        Returns:
            DataSet[Samples]: A DataSet where each item contains `Samples` objects representing
                              the forecast distributions for a specific location.

        Raises:
            Exception: Can re-raise exceptions from GluonTS prediction or data adaptation.
        """
        logger.info(f"Preparing data for GluonTS prediction (num_samples: {num_samples}).")
        try:
            # DataSetAdaptor converts CHAP-core DataSet to GluonTS compatible format
            gluonts_dataset_iterable = DataSetAdaptor.to_gluonts_testinstances(
                history, future_data, self.gluonts_predictor.prediction_length
            )

            logger.info("Calling underlying GluonTS predictor...")
            # GluonTS predictor.predict expects an iterable of data entries
            gluonts_forecasts_iterable = self.gluonts_predictor.predict(
                gluonts_dataset_iterable, num_samples=num_samples
            )

            # Convert GluonTS Forecast objects back to CHAP-core DataSet[Samples]
            # The order of forecasts in gluonts_forecasts_iterable should match the order of locations in history.keys()
            output_data: Dict[str, Samples] = {}
            location_keys = list(history.keys())  # Ensure consistent ordering

            for i, forecast_obj in enumerate(gluonts_forecasts_iterable):
                location_id = location_keys[i]
                # GluonTS forecast.index is typically pandas PeriodIndex or DatetimeIndex
                # forecast.samples is usually a (num_samples, prediction_length) numpy array
                # We need (prediction_length, num_samples) for CHAP Samples constructor
                output_data[location_id] = Samples(
                    time_period=PeriodRange.from_pandas(forecast_obj.index),
                    samples=forecast_obj.samples.T,  # Transpose samples
                )
            logger.info("GluonTS prediction successful and data converted back to DataSet[Samples].")
            return DataSet(output_data)
        except Exception as e:
            logger.error(f"Error during GluonTS prediction or data adaptation: {e}", exc_info=True)
            raise

    def save(self, directory_path: str) -> None:
        """
        Serializes the wrapped GluonTS predictor to a specified directory.

        Args:
            directory_path (str): The path to the directory where the predictor
                                  should be saved. The directory will be created if it
                                  does not exist.

        Raises:
            IOError: If saving fails due to file system issues.
            Exception: Can re-raise exceptions from GluonTS `serialize`.
        """
        path_obj = Path(directory_path)
        try:
            path_obj.mkdir(exist_ok=True, parents=True)
            self.gluonts_predictor.serialize(path_obj)
            logger.info(f"GluonTSPredictor saved successfully to directory: {directory_path}")
        except IOError as e:
            logger.error(
                f"Failed to create directory or save GluonTS predictor to {directory_path}: {e}", exc_info=True
            )
            raise
        except Exception as e:
            logger.error(f"Error serializing GluonTS predictor: {e}", exc_info=True)
            raise

    @classmethod
    def load(cls: Type["GluonTSPredictor"], directory_path: str) -> "GluonTSPredictor":
        """
        Deserializes a GluonTS predictor from a specified directory.

        Args:
            cls (Type['GluonTSPredictor']): The class itself.
            directory_path (str): The path to the directory from which to load the predictor.

        Returns:
            GluonTSPredictor: An instance of `GluonTSPredictor` wrapping the deserialized
                              GluonTS predictor.

        Raises:
            FileNotFoundError: If the directory or necessary files do not exist.
            Exception: Can re-raise exceptions from GluonTS `deserialize`.
        """
        path_obj = Path(directory_path)
        if not path_obj.exists() or not path_obj.is_dir():
            logger.error(f"Directory for loading GluonTSPredictor not found: {directory_path}")
            raise FileNotFoundError(f"Predictor directory not found: {directory_path}")
        try:
            deserialized_predictor = GluonTSPredictorNative.deserialize(path_obj)
            logger.info(f"GluonTSPredictor loaded successfully from directory: {directory_path}")
            return cls(deserialized_predictor)
        except Exception as e:
            logger.error(f"Error deserializing GluonTS predictor from {directory_path}: {e}", exc_info=True)
            raise


@dataclass
class GluonTSEstimator:
    """
    A wrapper for a GluonTS `Estimator` object.

    This class adapts a GluonTS estimator to integrate with CHAP-core's
    `DataSet` objects for model training.

    Attributes:
        gluonts_estimator (GluonTSEstimatorNative): The underlying GluonTS estimator instance.
                                                    Note: Aliased from `Estimator` to avoid name clash.
    """

    gluonts_estimator: GluonTSEstimatorNative  # Renamed attribute for clarity due to aliasing

    def train(self, dataset: DataSet[TimeSeriesData]) -> GluonTSPredictor:
        """
        Trains the wrapped GluonTS estimator using data from a CHAP-core `DataSet`.

        Converts the input `DataSet` into a GluonTS `ListDataset`, then calls
        the estimator's `train` method. The `freq` for the `ListDataset` is
        currently hardcoded to "m" (monthly) and should be made dynamic.

        Args:
            dataset (DataSet[TimeSeriesData]): The training dataset. The specific
                                               `TimeSeriesData` type should contain fields
                                               expected by the GluonTS estimator (e.g., target, features).

        Returns:
            GluonTSPredictor: A `GluonTSPredictor` instance wrapping the trained
                              GluonTS predictor.

        Raises:
            ValueError: If the input dataset is empty or time period frequency cannot be determined.
            Exception: Can re-raise exceptions from GluonTS training or data adaptation.
        """
        if not dataset:
            raise ValueError("Input dataset for training cannot be empty.")

        logger.info("Preparing dataset for GluonTS estimator training.")
        try:
            # DataSetAdaptor converts CHAP-core DataSet to GluonTS ListDataset format
            gluonts_train_data_list = DataSetAdaptor.to_gluonts(dataset)

            # Determine frequency from the dataset's PeriodRange
            # This assumes PeriodRange has a method like `get_gluonts_freq_str()`
            # or that DataSetAdaptor.to_gluonts embeds freq info.
            # For now, using a placeholder or a hardcoded value as in original.
            # TODO: Make frequency dynamic based on dataset.time_period.
            dataset_freq = "M"  # Defaulting to Monthly as per original hardcoding
            if dataset.time_period and hasattr(dataset.time_period, "freq_str_gluonts"):
                dataset_freq = dataset.time_period.freq_str_gluonts()  # Hypothetical method
            else:
                logger.warning(
                    f"Could not determine frequency from dataset; defaulting to '{dataset_freq}'. This may affect model training."
                )

            gluonts_list_dataset = ListDataset(gluonts_train_data_list, freq=dataset_freq)

            logger.info(f"Training GluonTS estimator with frequency '{dataset_freq}'...")
            trained_predictor_native = self.gluonts_estimator.train(gluonts_list_dataset)
            logger.info("GluonTS estimator training successful.")
            return GluonTSPredictor(trained_predictor_native)
        except Exception as e:
            logger.error(f"Error during GluonTS estimator training or data adaptation: {e}", exc_info=True)
            raise
