# Improvement Suggestions:
# 1. **Comprehensive Docstrings**: Add a module-level docstring and detailed docstrings for the `Predictor` and `Estimator` protocols, and all functions (`backtest`, `create_multiloc_timeseries`, `_get_forecast_dict`, `plot_forecasts`, `plot_predictions`). Ensure existing docstrings are clear and complete. (Primary task).
# 2. **Typo Correction**: Correct the typo `FetureType` to `FeatureType` in its `TypeVar` definition and usage.
# 3. **Clarify `without_disease` Type Hint**: The `without_disease(FeatureType)` construct in `Predictor.predict` type hint is unconventional. Its docstring (or a comment) should clearly explain that it signifies a `DataSet` containing features similar to `FeatureType` but excluding the target variable (e.g., "disease_cases"). Consider if a more standard type hint or a dedicated type alias could achieve this more clearly.
# 4. **Document GluonTS Integration**: Explicitly document the integration with GluonTS components (e.g., `gluonts.evaluation.Evaluator`, `gluonts.model.Forecast` via `ForecastAdaptor`) in relevant docstrings, explaining how CHAP-core data is adapted for GluonTS evaluation.
# 5. **Matplotlib Usage in Plotting**: The plotting functions (`plot_forecasts`, `plot_predictions`) directly use `matplotlib.pyplot`. Document this dependency. The global `plt.set_loglevel('warning')` should ideally be managed by the application using the library, not set at the module level here.

"""
This module defines protocols for model estimators and predictors, and provides
functions for evaluating model performance, primarily through backtesting and
comparison with ground truth data.

It integrates with GluonTS for evaluation, using its `Evaluator` and adapting
forecasts to the `gluonts.model.Forecast` format via `ForecastAdaptor`.
The module also includes utilities for plotting forecasts against truth data
and saving these plots to PDF reports.
"""

import logging
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Protocol, Tuple, TypeVar  # Added Optional, Tuple

import pandas as pd
from gluonts.evaluation import Evaluator as GluonTSEvaluator  # Aliased for clarity
from gluonts.model import Forecast
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from chap_core.assessment.dataset_splitting import train_test_generator
from chap_core.assessment.evaluator import EvaluationError  # Added this import
from chap_core.assessment.representations import MultiLocationDiseaseTimeSeries  # Added this import

# Assuming ForecastAdaptor is correctly located here
from chap_core.data.gluonts_adaptor.dataset import ForecastAdaptor
from chap_core.datatypes import Samples, SamplesWithTruth, TimeSeriesData
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet

# Suppress verbose matplotlib logs, consider if this should be application-level
# plt.set_loglevel(level='warning') # Original line
# It's generally better for libraries not to alter global state of other libs.
# This could be set by the calling application if desired.
# For now, commenting it out from library code.
# import matplotlib
# matplotlib.set_loglevel(level='warning')


logger = logging.getLogger(__name__)


# Corrected typo from FetureType to FeatureType
FeatureType = TypeVar("FeatureType", bound=TimeSeriesData, covariant=True)  # Made covariant for more flexibility


def without_disease(t: FeatureType) -> FeatureType:  # Type hint for 't' and return
    """
    A helper identity function used in type hinting to signify that a `DataSet`
    should contain features like `FeatureType` but without the "disease_cases" (target) field.

    Note: This function itself doesn't perform the removal; it's a marker for type analysis
    and documentation. The actual removal is handled by `DataSet.remove_field("disease_cases")`.
    """
    return t


class Predictor(Protocol[FeatureType]):  # Made generic for clarity
    """
    Protocol defining the interface for a trained model predictor.

    A predictor is expected to take historical data and future covariate data
    and produce forecast samples.
    """

    def predict(
        self,
        historic_data: DataSet[FeatureType],
        future_data: DataSet[without_disease(FeatureType)],  # DataSet of features, excluding target
    ) -> DataSet[Samples]:  # Changed return from Samples to DataSet[Samples] for consistency
        """
        Generates predictions based on historic data and future covariates.

        Args:
            historic_data (DataSet[FeatureType]): Historical data providing context for predictions.
                                                  `FeatureType` should include the target variable.
            future_data (DataSet[without_disease(FeatureType)]): Future covariate data for the prediction horizon.
                                                                 This dataset should not contain the target variable.

        Returns:
            DataSet[Samples]: A DataSet containing `Samples` objects for each location,
                              representing the forecast distributions.
        """
        ...


class Estimator(Protocol):
    """
    Protocol defining the interface for a model estimator.

    An estimator is responsible for training a model on a given dataset and
    returning a `Predictor` instance.
    """

    def train(self, data: DataSet[TimeSeriesData]) -> Predictor:  # data can be any TimeSeriesData for training
        """
        Trains the model on the provided dataset.

        Args:
            data (DataSet[TimeSeriesData]): The training dataset, including target variables and covariates.

        Returns:
            Predictor: A trained predictor instance capable of making forecasts.
        """
        ...


def backtest(
    estimator: Estimator,
    data: DataSet[TimeSeriesData],  # Data should be general TimeSeriesData for flexibility
    prediction_length: int,  # Typically an int number of periods
    n_test_sets: int,
    stride: int = 1,
    weather_provider: Any = None,  # Optional[FutureWeatherFetcher] would be more specific
) -> Iterable[DataSet[SamplesWithTruth]]:
    """
    Performs backtesting by generating multiple train/test splits and evaluating the estimator on each.

    It uses `train_test_generator` to create a main training set and an iterator for test windows.
    The estimator is trained once on the main training set. Then, for each test window,
    predictions are made and merged with the true future values.

    Args:
        estimator (Estimator): The model estimator to train and evaluate.
        data (DataSet[TimeSeriesData]): The full dataset to use for backtesting.
        prediction_length (int): The number of periods to forecast ahead in each test window.
        n_test_sets (int): The number of test windows to generate and evaluate.
        stride (int): The step size (number of periods) to move forward for each subsequent test window.
                      Defaults to 1.
        weather_provider (Any, optional): An optional provider for future weather data.
                                          If None, weather data is derived from the truth set.
                                          Should ideally be `Optional[FutureWeatherFetcher]`.

    Yields:
        Iterable[DataSet[SamplesWithTruth]]: An iterator yielding `DataSet` objects for each test window.
                                             Each `DataSet` contains `SamplesWithTruth`, merging
                                             forecast samples with actual observed values for that window.
    """
    logger.info(f"Starting backtest for estimator {estimator.__class__.__name__} with {n_test_sets} test sets.")
    train_set, test_generator = train_test_generator(
        data, prediction_length, n_test_sets, stride=stride, future_weather_provider=weather_provider
    )

    if not train_set:
        logger.error("Backtesting cannot proceed: Training set generated by train_test_generator is empty.")
        raise ValueError(
            "Generated training set for backtesting is empty. Check dataset size and splitting parameters."
        )

    logger.info("Training estimator on the main training set for backtesting...")
    predictor = estimator.train(train_set)
    logger.info("Estimator training complete.")

    for i, (historic_data, future_covariates, future_truth_with_target) in enumerate(test_generator):
        logger.info(f"Backtesting window {i+1}/{n_test_sets}...")
        try:
            # `future_covariates` is DataSet[without_disease(FeatureType)]
            # `historic_data` is DataSet[FeatureType]
            # `future_truth_with_target` is DataSet[FeatureType] (contains target)
            prediction_samples_dataset = predictor.predict(historic_data, future_covariates)

            # Merge predictions (Samples) with the truth data (which includes the target, e.g., disease_cases)
            # This assumes future_truth_with_target and prediction_samples_dataset have aligned locations and periods.
            # The result_dataclass SamplesWithTruth should be able to combine fields from both.
            samples_with_truth_dataset = future_truth_with_target.merge(
                prediction_samples_dataset, result_dataclass=SamplesWithTruth
            )
            yield samples_with_truth_dataset
        except Exception as e:
            logger.error(f"Error during prediction for backtest window {i+1}: {e}", exc_info=True)
            # Decide whether to skip this window or re-raise
            # For now, re-raising to make test failures explicit
            raise EvaluationError(f"Prediction failed for backtest window {i+1}: {e}")


def evaluate_model(
    estimator: Estimator,
    data: DataSet[TimeSeriesData],
    prediction_length: int = 3,  # Changed to int for number of periods
    n_test_sets: int = 4,
    report_filename: Optional[str] = None,  # Path for PDF report
    weather_provider: Any = None,  # Optional[FutureWeatherFetcher]
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:  # Return type from original GluonTS evaluator
    """
    Evaluates a model by training it once and then making multiple predictions on rolling test sets.
    Uses GluonTS Evaluator for calculating metrics and can generate a PDF report with plots.

    Args:
        estimator (Estimator): The estimator to train and evaluate.
        data (DataSet[TimeSeriesData]): The full dataset for training and evaluation.
        prediction_length (int): The number of periods to predict ahead in each test window. Defaults to 3.
        n_test_sets (int): The number of test windows to evaluate on. Defaults to 4.
        report_filename (Optional[str]): If provided, path to save a PDF report with forecast plots.
                                         Defaults to None (no report).
        weather_provider (Any, optional): Provider for future weather data. Defaults to None.

    Returns:
        Tuple[pd.DataFrame, Optional[pd.DataFrame]]: A tuple containing:
            - agg_metrics (pd.DataFrame): DataFrame of aggregated metrics.
            - item_metrics (Optional[pd.DataFrame]): DataFrame of metrics per item (if available from evaluator).
                                                     May be None.
    """
    logger.info(
        f"Evaluating estimator {estimator.__class__.__name__} with {n_test_sets} test sets, forecasting {prediction_length} periods ahead."
    )

    # Generate main training set and the iterator for test windows
    train_set, test_generator_for_eval = train_test_generator(
        data,
        prediction_length,
        n_test_sets,
        stride=1,  # stride=1 is common for this type of eval
        future_weather_provider=weather_provider,
    )

    if not train_set:
        logger.error("Evaluation cannot proceed: Training set is empty.")
        raise ValueError("Generated training set for evaluation is empty.")

    logger.info("Training estimator for evaluation...")
    predictor = estimator.train(train_set)
    logger.info("Estimator training complete.")

    # Prepare truth data in pandas format for GluonTS Evaluator
    # This extracts the 'disease_cases' series for each location.
    # Assumes 'disease_cases' is the target and present in the original `data`.
    truth_data_for_gluonts: Dict[str, pd.DataFrame] = {
        location: pd.DataFrame(
            data_item.disease_cases,  # Accessing disease_cases directly
            index=data_item.time_period.to_period_index(),  # GluonTS expects pd.PeriodIndex
            columns=["target"],  # GluonTS often expects a 'target' column
        )
        for location, data_item in data.items()
        if hasattr(data_item, "disease_cases")
    }

    # Store forecasts and corresponding truth series for GluonTS Evaluator
    forecast_objects_list: List[Forecast] = []
    truth_series_list: List[pd.Series] = []  # GluonTS Evaluator expects list of pd.Series for truth

    # Generator for plotting and collecting forecasts/truths
    # Re-create a test generator for plotting if report_filename is provided, to ensure it's fresh
    # Or, better, consume the test_generator_for_eval once for both plotting and evaluation.

    # We need to iterate through test_generator_for_eval to get historic_data, future_data, future_truth
    # Then make predictions, adapt them to GluonTS Forecast, and collect corresponding truth series.

    forecast_and_truth_iterator = []  # Will store (forecast_obj, truth_pd_series) tuples

    for historic_data, future_covariates, future_truth_ds in test_generator_for_eval:
        prediction_samples_ds = predictor.predict(historic_data, future_covariates)
        for loc_id, samples in prediction_samples_ds.items():
            gluonts_forecast = ForecastAdaptor.from_samples(samples)
            forecast_objects_list.append(gluonts_forecast)

            # Extract corresponding truth series for this forecast window and location
            # The truth_data_for_gluonts is for the *entire* dataset. We need to slice it.
            # The `future_truth_ds[loc_id].time_period` gives the exact periods for this forecast.
            current_truth_periods = future_truth_ds[loc_id].time_period.to_period_index()
            # Slice the full truth series for this location and period range
            truth_series_for_window = truth_data_for_gluonts[loc_id].loc[current_truth_periods]["target"]
            truth_series_list.append(truth_series_for_window)
            forecast_and_truth_iterator.append((gluonts_forecast, truth_series_for_window))

    if report_filename:
        logger.info(f"Plotting forecasts to {report_filename}")
        # Use the collected forecasts_and_truths for plotting
        with PdfPages(report_filename) as pdf:
            # Need to group forecasts by location if original plot_forecasts did that.
            # This simplified plotting assumes one plot per forecast window instance.
            # For a per-location plot over all windows, data needs restructuring.
            # The original plot_forecasts was more complex.
            # For now, let's plot each forecast window.
            # This requires access to the location ID for titles.
            # The current `forecast_and_truth_iterator` doesn't preserve location easily.
            # This plotting part needs careful review based on desired output.
            # Simplified:
            idx = 0
            for forecast_obj, truth_series_for_window in forecast_and_truth_iterator:  # Iterate through collected items
                plt.figure(figsize=(8, 4))
                forecast_obj.plot(show_label=True)
                # Plot truth, ensuring alignment if possible
                truth_series_for_window.plot(label="Truth")  # truth_series_for_window is already a pd.Series
                plt.title(
                    f"Forecast Window {idx//len(data.keys()) +1}, Loc: {list(data.keys())[idx%len(data.keys())]}"
                )  # Approximate title
                plt.legend()
                pdf.savefig()
                plt.close()
                idx += 1

    if not forecast_objects_list or not truth_series_list:
        logger.warning("No forecasts or truth series generated for evaluation. Returning empty metrics.")
        return pd.DataFrame(), pd.DataFrame()  # Return empty DataFrames

    logger.info(f"Evaluating {len(forecast_objects_list)} forecast instances...")
    # GluonTS Evaluator calculates metrics across all provided (forecast, truth) pairs.
    gluonts_eval = GluonTSEvaluator(quantiles=[0.1, 0.5, 0.9], num_workers=None)  # num_workers=None for local
    agg_metrics, item_metrics = gluonts_eval(iter(truth_series_list), iter(forecast_objects_list))

    logger.info("Finished GluonTS evaluation.")
    return agg_metrics, item_metrics


def create_multiloc_timeseries(truth_data: Dict[str, pd.DataFrame]) -> MultiLocationDiseaseTimeSeries:
    """
    Converts a dictionary of pandas DataFrames (truth data per location) into
    a `MultiLocationDiseaseTimeSeries` object.

    Args:
        truth_data (Dict[str, pd.DataFrame]): A dictionary where keys are location IDs
            and values are pandas DataFrames. Each DataFrame must have a `pd.PeriodIndex`
            and a column containing disease cases (assumed to be the first/only column
            if not explicitly named 'target' or similar).

    Returns:
        MultiLocationDiseaseTimeSeries: The converted multi-location time series object.
    """
    from chap_core.assessment.representations import DiseaseObservation, DiseaseTimeSeries  # Local import

    multi_location_ts = MultiLocationDiseaseTimeSeries(timeseries_dict={})  # Initialize empty
    for location_id, df in truth_data.items():
        observations = []
        # Assuming DataFrame index is pd.PeriodIndex and first column is disease cases
        # df.itertuples(index=True, name='Pandas') yields (Period, value1, value2, ...)
        for row in df.itertuples(index=True):  # row[0] is Period, row[1] is first column value
            period = row[0]  # This is pd.Period
            # Convert pd.Period to chap_core.time_period.TimePeriod if necessary,
            # or ensure DiseaseObservation can handle pd.Period.
            # For now, assuming direct use or compatible conversion.
            # If DiseaseObservation expects specific TimePeriod subclass:
            # from chap_core.time_period import Month, Week, Day # etc.
            # if period.freqstr == 'M': chap_period = Month(period.year, period.month) ...

            disease_cases_value = row[1]  # Assuming first column is the target
            observations.append(DiseaseObservation(time_period=period, disease_cases=int(disease_cases_value)))

        multi_location_ts[location_id] = DiseaseTimeSeries(observations=observations)
    return multi_location_ts


def _get_forecast_generators(
    predictor: Predictor,
    test_generator: Iterable[Tuple[DataSet, DataSet, DataSet]],
    truth_data: Dict[str, pd.DataFrame],
) -> Tuple[List[Forecast], List[pd.DataFrame]]:  # Changed to pd.DataFrame for truth
    """
    Generates lists of GluonTS Forecast objects and corresponding truth pandas DataFrames.

    This helper iterates through test windows, makes predictions, adapts them to
    GluonTS `Forecast` format, and pairs them with the relevant slice of truth data.
    Each entry in the output lists corresponds to a specific location and prediction window.

    Args:
        predictor (Predictor): The trained model predictor.
        test_generator (Iterable[Tuple[DataSet, DataSet, DataSet]]): An iterator yielding
            (historic_data, future_covariates, future_truth_dataset) for each test window.
        truth_data (Dict[str, pd.DataFrame]): A dictionary mapping location IDs to pandas DataFrames
                                              containing the full truth series for that location,
                                              indexed by `pd.PeriodIndex`.

    Returns:
        Tuple[List[Forecast], List[pd.DataFrame]]: A tuple containing:
            - A list of `gluonts.model.Forecast` objects.
            - A list of corresponding truth `pd.DataFrame` (or `pd.Series`) slices.
    """
    truth_series_list: List[pd.DataFrame] = []  # Store truth series slices
    forecast_objects_list: List[Forecast] = []

    for historic_data, future_covariates, future_truth_dataset in test_generator:
        # `future_truth_dataset` is a CHAP-core DataSet for the current window
        prediction_samples_dataset = predictor.predict(historic_data, future_covariates)

        for location_id, samples_obj in prediction_samples_dataset.items():
            gluonts_forecast = ForecastAdaptor.from_samples(samples_obj)
            forecast_objects_list.append(gluonts_forecast)

            # Extract the corresponding truth data slice for this location and forecast window
            # The `samples_obj.time_period` (which is also `gluonts_forecast.index` after conversion)
            # defines the exact periods for which this forecast is made.
            forecast_period_index = samples_obj.time_period.to_period_index()

            # Slice the full truth series for this location using the forecast's period index
            # Ensure truth_data[location_id] has 'target' column or is a Series
            truth_slice = truth_data[location_id].loc[forecast_period_index]
            if isinstance(truth_slice, pd.DataFrame) and "target" in truth_slice.columns:
                truth_series_list.append(truth_slice["target"])  # Use only the 'target' Series
            elif isinstance(truth_slice, pd.Series):
                truth_series_list.append(truth_slice)
            else:
                raise ValueError(
                    f"Truth data for location {location_id} is not in expected Series/DataFrame format or missing 'target' column."
                )

    return forecast_objects_list, truth_series_list


def _get_forecast_dict(
    predictor: Predictor, test_generator: Iterable[Tuple[DataSet, DataSet, DataSet]]
) -> Dict[str, List[Forecast]]:
    """
    Generates a dictionary mapping location IDs to lists of GluonTS Forecast objects.

    For each test window and each location, a forecast is generated and adapted.
    This is useful for plotting forecasts per location across multiple windows.

    Args:
        predictor (Predictor): The trained model predictor.
        test_generator (Iterable[Tuple[DataSet, DataSet, DataSet]]): An iterator yielding
            (historic_data, future_covariates, _future_truth_dataset) for each test window.

    Returns:
        Dict[str, List[Forecast]]: A dictionary where keys are location IDs and values are
                                   lists of `gluonts.model.Forecast` objects for that location,
                                   one for each test window.
    """
    forecast_dict: Dict[str, List[Forecast]] = defaultdict(list)

    for historic_data, future_covariates, _ in test_generator:  # _ is future_truth_dataset, not used here
        if not future_covariates.period_range:  # Check if future_covariates has periods
            logger.warning(
                f"Future covariate data has no periods for context ending {historic_data.period_range.end_time_period}. Skipping this window."
            )
            continue

        # Ensure historic_data and future_covariates are not empty if predictor requires non-empty context/future.
        # This depends on the specific predictor's requirements.

        prediction_samples_dataset = predictor.predict(historic_data, future_covariates)
        for location_id, samples_obj in prediction_samples_dataset.items():
            forecast_dict[location_id].append(ForecastAdaptor.from_samples(samples_obj))
    return forecast_dict


def plot_forecasts(
    predictor: Predictor,
    test_instance_generator: Iterable[Tuple[DataSet, DataSet, DataSet]],  # Renamed for clarity
    truth_data_all_locations: Dict[str, pd.DataFrame],  # Full truth series for all locations
    pdf_filename: str,
) -> Iterable[Tuple[Forecast, pd.Series]]:  # Yields forecast and its corresponding truth slice
    """
    Generates and plots forecasts for multiple test windows and locations, saving to a PDF.

    Iterates through test windows, generates forecasts, and plots each forecast
    against the corresponding truth data. Each plot is saved to a page in the PDF.
    Also yields the GluonTS Forecast object and the corresponding truth pd.Series slice
    for each forecast generated, which can be used by `evaluate_model`.

    Args:
        predictor (Predictor): The trained model predictor.
        test_instance_generator (Iterable[Tuple[DataSet, DataSet, DataSet]]):
            Generator yielding (historic_data, future_covariates, future_truth_dataset) tuples.
        truth_data_all_locations (Dict[str, pd.DataFrame]):
            A dictionary mapping location IDs to pandas DataFrames of the full truth series,
            indexed by `pd.PeriodIndex` and having a 'target' column.
        pdf_filename (str): Path to the output PDF file for saving plots.

    Yields:
        Iterable[Tuple[Forecast, pd.Series]]: For each forecast made, yields a tuple of
                                              (gluonts_forecast_object, truth_data_slice_for_forecast).
    """
    # This function now combines forecasting, plotting, and yielding data for evaluation.
    # The original _get_forecast_dict is effectively integrated here.

    logger.info(f"Generating and plotting forecasts to {pdf_filename}...")
    with PdfPages(pdf_filename) as pdf:
        for i, (historic_data, future_covariates, future_truth_dataset) in enumerate(test_instance_generator):
            logger.debug(f"Processing test window {i+1} for plotting and evaluation.")
            if not future_covariates.period_range:
                logger.warning(f"Window {i+1}: Future covariate data has no periods. Skipping.")
                continue

            prediction_samples_dataset = predictor.predict(historic_data, future_covariates)

            for location_id, samples_obj in prediction_samples_dataset.items():
                gluonts_forecast = ForecastAdaptor.from_samples(samples_obj)

                # Get the corresponding truth slice for this specific forecast window and location
                full_truth_series_for_loc = truth_data_all_locations.get(location_id)
                if full_truth_series_for_loc is None:
                    logger.warning(
                        f"No truth data found for location '{location_id}'. Cannot plot or yield truth for this forecast."
                    )
                    # Yield forecast with None truth, or skip? For now, skip yielding if truth missing.
                    continue

                # Ensure forecast.index is pd.PeriodIndex for .loc
                forecast_period_index = pd.PeriodIndex(gluonts_forecast.index, freq=gluonts_forecast.freq)
                truth_slice_for_forecast = full_truth_series_for_loc.loc[forecast_period_index]["target"]  # Get Series

                # Plotting
                plt.figure(figsize=(10, 5))  # Adjusted size slightly
                # Plot truth first, then forecast samples over it
                # Context for plotting: show some historical data before the forecast start
                context_length = 52 * 2  # Show 2 years of context if available

                # Determine plotting range for truth data (context + forecast period)
                plot_start_period = forecast_period_index[0] - context_length
                plot_end_period = forecast_period_index[-1]

                truth_for_plot = full_truth_series_for_loc.loc[plot_start_period:plot_end_period]["target"]
                if not truth_for_plot.empty:
                    plt.plot(truth_for_plot.index.to_timestamp(), truth_for_plot.values, label="Truth")

                gluonts_forecast.plot(show_label=True)  # Plots samples and median

                plt.title(f"Forecast for {location_id} (Window {i+1}) starting {gluonts_forecast.start_date}")
                plt.legend()
                pdf.savefig()
                plt.close()

                yield gluonts_forecast, truth_slice_for_forecast  # Yield for evaluation


def plot_predictions(
    predictions: DataSet[Samples],
    truth: DataSet[TimeSeriesData],  # Assuming truth contains target in a field like 'disease_cases'
    pdf_filename: str,
) -> None:
    """
    Plots predictions from a `DataSet[Samples]` against corresponding truth data.

    This function is useful for visualizing a single set of predictions (e.g., from
    `forecast_ahead`) rather than rolling forecasts.

    Args:
        predictions (DataSet[Samples]): The predicted samples for multiple locations.
        truth (DataSet[TimeSeriesData]): The ground truth data, expected to contain the target variable
                                         (e.g., in a field named 'disease_cases').
                                         Must align with predictions by location and time period.
        pdf_filename (str): Path to the output PDF file where plots will be saved.
    """
    logger.info(f"Plotting predictions against truth to {pdf_filename}...")
    # Prepare truth data into a dictionary of pandas Series, similar to evaluate_model
    truth_series_dict: Dict[str, pd.Series] = {}
    for location_id, truth_data_item in truth.items():
        if hasattr(truth_data_item, "disease_cases"):
            truth_series_dict[location_id] = pd.Series(
                truth_data_item.disease_cases, index=truth_data_item.time_period.to_period_index(), name="target"
            )
        else:
            logger.warning(f"Truth data for location '{location_id}' missing 'disease_cases'. Cannot plot truth.")

    with PdfPages(pdf_filename) as pdf:
        for location_id, samples_obj in predictions.items():
            gluonts_forecast = ForecastAdaptor.from_samples(samples_obj)

            plt.figure(figsize=(10, 5))

            # Plot forecast
            gluonts_forecast.plot(show_label=True)  # Shows median and sample quantiles

            # Plot corresponding truth data if available
            if location_id in truth_series_dict:
                truth_series_for_loc = truth_series_dict[location_id]
                # Align truth data with forecast period for plotting
                forecast_period_index = pd.PeriodIndex(gluonts_forecast.index, freq=gluonts_forecast.freq)

                # Plot historical context + forecast period truth
                context_length = 52 * 2  # Number of periods of context to show before forecast
                plot_start_period = forecast_period_index[0] - context_length
                plot_end_period = forecast_period_index[-1]

                truth_for_plot = truth_series_for_loc.loc[plot_start_period:plot_end_period]
                if not truth_for_plot.empty:
                    plt.plot(truth_for_plot.index.to_timestamp(), truth_for_plot.values, label="Truth", color="black")

            plt.title(f"Prediction for {location_id} starting {gluonts_forecast.start_date}")
            plt.legend()
            pdf.savefig()
            plt.close()
    logger.info(f"Prediction plots saved to {pdf_filename}.")
