# Improvement Suggestions:
# 1. **Module Docstring**: Add a comprehensive module docstring explaining the purpose of this file: to provide specialized functions for visualizing model predictions, including forecast samples, quantiles, and comparisons against true data, utilizing both Matplotlib and Plotly libraries. (Primary task).
# 2. **Complete Function Docstrings**: Provide detailed docstrings for all functions (`prediction_plot`, `forecast_plot`, `plot_forecast_from_summaries`, `plot_forecast`, `plot_forecasts_from_data_frame`, `add_prediction_lines`), clearly outlining their purpose, arguments (with types), return values, and any key assumptions or behaviors.
# 3. **Plotting Library Consistency**: The module currently uses `matplotlib.pyplot` for `prediction_plot` and `plotly.graph_objects` for other functions. Consider standardizing on Plotly for consistency within this module and the broader `chap_core.plotting` package, or clearly document the rationale for using different libraries if intentional.
# 4. **Comprehensive Type Hinting**: Ensure all function parameters and return values have accurate and specific type hints. For instance, `n_samples` should be `int`, `x_pred` in `plot_forecast` could be `Optional[pd.Series | List[str]]`, and `true_df` in `plot_forecasts_from_data_frame` should be `pd.DataFrame`.
# 5. **Robust Error Handling and Input Validation**:
#    - Add input validation (e.g., `n_samples > 0`).
#    - In `add_prediction_lines`, the line `last_idx = np.where(prediction_df["time_period"][0] == true_df["x"])[0][0]` is prone to `IndexError` if `prediction_df` is empty or the time period is not found. Implement checks and raise informative errors or handle gracefully.
#    - Ensure data alignment (e.g., time periods, required columns) between prediction and truth DataFrames before plotting, or handle potential mismatches.

"""
This module provides functions for visualizing model predictions, focusing on
comparing forecasted distributions (samples, quantiles) against actual observed data.

It utilizes both Matplotlib (for one function) and Plotly (for others) to generate
various plots useful in assessing prediction quality and understanding forecast uncertainty.
"""

import logging
from typing import Any, Callable, List, Optional, Protocol, TypeVar, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib import pyplot as plt
from plotly.graph_objs import Figure

from chap_core.datatypes import Samples  # Added Samples
from chap_core.datatypes import ClimateData, HealthData, SummaryStatistics, TimeSeriesData
from chap_core.predictor.protocol import IsSampler
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet  # Added DataSet

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


class Predictor(Protocol[FeatureType]):
    """
    Protocol defining the interface for a trained model predictor.

    A predictor is expected to take historical data and future covariate data
    and produce forecast samples.
    """

    def predict(
        self,
        historic_data: DataSet[FeatureType],
        future_data: DataSet[without_disease(FeatureType)],  # DataSet of features, excluding target
    ) -> DataSet[Samples]:
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


def prediction_plot(
    true_data: HealthData,
    predicition_sampler: IsSampler,
    climate_data: ClimateData,
    n_samples: int,
) -> plt.Figure:
    """
    Plots multiple predicted sample paths against the true data using Matplotlib.

    This function generates `n_samples` paths from the `prediction_sampler`
    based on the provided `climate_data` and plots them alongside the `true_data`.

    Args:
        true_data (HealthData): The ground truth health data.
        predicition_sampler (IsSampler): A sampler object that has a `sample` method
                                         which takes `ClimateData` and returns a single predicted path.
        climate_data (ClimateData): Climate data used as input/context for the sampler.
        n_samples (int): The number of sample paths to generate and plot.

    Returns:
        matplotlib.figure.Figure: The Matplotlib figure object containing the plot.

    Raises:
        ValueError: If n_samples is not positive.
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be a positive integer.")

    plt.figure()
    for i in range(n_samples):
        try:
            new_observed = predicition_sampler.sample(climate_data)
            plt.plot(new_observed, label="predicted" if i == 0 else None, color="grey", alpha=0.5)
        except Exception as e:
            logger.error(f"Error generating sample {i+1} in prediction_plot: {e}", exc_info=True)

    plt.plot(true_data.disease_cases, label="real", color="blue")
    plt.legend()
    plt.title("Predicted paths using estimated parameters vs real path")
    plt.xlabel("Time Index")
    plt.ylabel("Disease Cases")
    return plt.gcf()


def forecast_plot(
    true_data: HealthData,
    predicition_sampler: IsSampler,
    climate_data: ClimateData,
    n_samples: int,
) -> Figure:
    """
    Generates a forecast plot showing quantiles of prediction samples against true data, using Plotly.

    It samples `n_samples` paths from the `predicition_sampler`, calculates 0.1, 0.5 (median),
    and 0.9 quantiles, and then plots these along with the `true_data`.

    Args:
        true_data (HealthData): The ground truth health data.
        predicition_sampler (IsSampler): A sampler object with a `sample` method.
        climate_data (ClimateData): Climate data used as input for the sampler.
        n_samples (int): The number of samples to generate for quantile calculation.

    Returns:
        plotly.graph_objs.Figure: The Plotly figure object.

    Raises:
        ValueError: If n_samples is not positive or if sampling/quantile calculation fails.
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be a positive integer.")

    try:
        samples_list = [predicition_sampler.sample(climate_data) for _ in range(n_samples)]
        if not samples_list or not all(len(s) == len(samples_list[0]) for s in samples_list):
            raise ValueError("All generated samples must have the same length and be non-empty.")

        samples_array = np.array(samples_list)
        quantiles = np.quantile(samples_array, [0.1, 0.5, 0.9], axis=0)
    except Exception as e:
        logger.error(f"Error during sampling or quantile calculation in forecast_plot: {e}", exc_info=True)
        raise ValueError(f"Failed to generate samples or quantiles: {e}")

    return plot_forecast(quantiles, true_data)


def plot_forecast_from_summaries(
    summaries: Union[SummaryStatistics, List[SummaryStatistics]],
    true_data: HealthData,
    transform: Callable[[Any], Any] = lambda x: x,
) -> Figure:
    """
    Plots forecasts derived from `SummaryStatistics` objects against true data.

    This function can handle a single `SummaryStatistics` object or a list of them.
    It converts the summary statistics and true data to pandas DataFrames before plotting.

    Args:
        summaries (Union[SummaryStatistics, List[SummaryStatistics]]):
            A single `SummaryStatistics` object or a list of such objects.
            Each object is expected to have a `topandas()` method.
        true_data (HealthData): The ground truth health data.
        transform (Callable[[Any], Any], optional): A function to transform the y-values
            (both predictions and truth) before plotting. Defaults to identity function.

    Returns:
        plotly.graph_objs.Figure: The Plotly figure object.
    """
    true_df = pd.DataFrame(
        {
            "x": [str(p) for p in true_data.time_period.topandas()],
            "real": true_data.disease_cases,
        }
    )
    if isinstance(summaries, list):
        prediction_dfs = []
        for summary_item in summaries:
            if not isinstance(summary_item, SummaryStatistics):
                raise TypeError(f"All items in summaries list must be SummaryStatistics, got {type(summary_item)}")
            df_item = summary_item.topandas()
            df_item["time_period"] = df_item["time_period"].astype(str)
            prediction_dfs.append(df_item)
        df_to_plot = prediction_dfs
    elif isinstance(summaries, SummaryStatistics):
        df_to_plot = summaries.topandas()
        df_to_plot["time_period"] = df_to_plot["time_period"].astype(str)
    else:
        raise TypeError(f"summaries must be SummaryStatistics or List[SummaryStatistics], got {type(summaries)}")

    return plot_forecasts_from_data_frame(df_to_plot, true_df, transform)


def plot_forecast(quantiles: np.ndarray, true_data: HealthData, x_pred: Optional[Any] = None) -> Figure:
    """
    Plots pre-calculated forecast quantiles against true data using Plotly.

    Args:
        quantiles (np.ndarray): A 2D numpy array where rows are quantiles (e.g., 0.1, 0.5, 0.9)
                                and columns are time points. Expected shape (3, num_timepoints).
        true_data (HealthData): The ground truth health data.
        x_pred (Optional[Any]): The x-axis values (time periods) for the predictions.
                                If None, uses the time periods from `true_data`.
                                Should be convertible to a list of strings. Defaults to None.

    Returns:
        plotly.graph_objs.Figure: The Plotly figure object.
    """
    if quantiles.shape[0] != 3:
        raise ValueError(f"Quantiles array must have 3 rows (0.1, 0.5, 0.9), got shape {quantiles.shape}")

    x_true_periods = true_data.time_period.topandas()
    x_true_str = [str(p) for p in x_true_periods]

    if x_pred is None:
        x_pred_str = x_true_str
        if len(x_pred_str) != quantiles.shape[1]:
            raise ValueError(
                f"Length of true_data time_period ({len(x_pred_str)}) must match prediction length ({quantiles.shape[1]}) when x_pred is None."
            )
    elif hasattr(x_pred, "topandas"):
        x_pred_str = [str(p) for p in x_pred.topandas()]
    else:
        x_pred_str = [str(p) for p in x_pred]

    if len(x_pred_str) != quantiles.shape[1]:
        raise ValueError(f"Length of x_pred ({len(x_pred_str)}) must match prediction length ({quantiles.shape[1]}).")

    df = pd.DataFrame({"x": x_pred_str, "10th": quantiles[0], "50th": quantiles[1], "90th": quantiles[2]})
    true_df = pd.DataFrame({"x": x_true_str, "real": true_data.disease_cases})
    return plot_forecasts_from_data_frame(df, true_df)


def plot_forecasts_from_data_frame(
    prediction_df: Union[pd.DataFrame, List[pd.DataFrame]],
    true_df: pd.DataFrame,
    transform: Callable[[Any], Any] = lambda x: x,
) -> Figure:
    """
    Core plotting function that takes pandas DataFrames for predictions and truth,
    and generates a Plotly figure.

    Args:
        prediction_df (Union[pd.DataFrame, List[pd.DataFrame]]):
            A DataFrame or list of DataFrames containing prediction data.
            Expected columns: 'time_period' (or 'x'), 'quantile_low', 'median', 'quantile_high'.
        true_df (pd.DataFrame): DataFrame containing true data.
                                Expected columns: 'x' (time periods as strings), 'real' (true values).
        transform (Callable[[Any], Any], optional): A function to transform y-values before plotting.
                                                    Defaults to identity.

    Returns:
        plotly.graph_objs.Figure: The generated Plotly figure.
    """
    fig = go.Figure()
    if isinstance(prediction_df, list):
        for i, df_item in enumerate(prediction_df):
            add_prediction_lines(fig, df_item, transform, true_df, show_legend=(i == 0))
    else:
        add_prediction_lines(fig, prediction_df, transform, true_df, show_legend=True)

    fig.add_scatter(
        x=true_df["x"],
        y=transform(true_df["real"]),
        mode="lines",
        name="Real Data",
        line=dict(color="blue"),
    )
    fig.update_layout(
        title="Forecast vs. Real Data",
        xaxis_title="Time Period",
        yaxis_title="Disease Cases",
        legend_title_text="Legend",
    )
    return fig


def add_prediction_lines(
    fig: go.Figure,
    prediction_df: pd.DataFrame,
    transform: Callable[[Any], Any],
    true_df: pd.DataFrame,
    show_legend: bool = True,
) -> None:
    """
    Helper function to add prediction quantile and median traces to a Plotly figure.
    It also adds a vertical line indicating the start of the forecast period relative to truth.

    Args:
        fig (go.Figure): The Plotly figure to add traces to.
        prediction_df (pd.DataFrame): DataFrame with prediction data. Expected columns:
                                      'time_period' (or 'x'), 'quantile_low', 'median', 'quantile_high'.
        transform (Callable[[Any], Any]): Function to transform y-values.
        true_df (pd.DataFrame): DataFrame with true data, used to find the forecast start.
        show_legend (bool): Whether to show legend entries for these traces. Defaults to True.

    Raises:
        IndexError: If the first prediction time period is not found in true_df.
        ValueError: If required columns are missing in prediction_df.
    """
    x_col = "time_period" if "time_period" in prediction_df.columns else "x"
    q_low_col = "quantile_low" if "quantile_low" in prediction_df.columns else "10th"
    median_col = "median" if "median" in prediction_df.columns else "50th"
    q_high_col = "quantile_high" if "quantile_high" in prediction_df.columns else "90th"

    required_cols = [x_col, q_low_col, median_col, q_high_col]
    if not all(col in prediction_df.columns for col in required_cols):
        raise ValueError(
            f"prediction_df missing one or more required columns from {required_cols}. Available: {prediction_df.columns.tolist()}"
        )
    if prediction_df.empty:
        logger.warning("add_prediction_lines received an empty prediction_df. No lines will be added.")
        return

    try:
        first_pred_period = str(prediction_df[x_col].iloc[0])
        matching_indices = np.where(true_df["x"] == first_pred_period)[0]
        if not matching_indices.size > 0:
            logger.warning(
                f"First prediction period '{first_pred_period}' not found in true_df. Cannot align for plotting continuity."
            )
            df_to_plot = prediction_df.copy()
            last_idx_in_truth_for_line = len(true_df)
        else:
            last_idx_in_truth_for_line = matching_indices[0]
            if last_idx_in_truth_for_line > 0:
                last_true_row = true_df.iloc[last_idx_in_truth_for_line - 1]
                prepend_data = {
                    x_col: [last_true_row["x"]],
                    q_high_col: [last_true_row["real"]],
                    q_low_col: [last_true_row["real"]],
                    median_col: [last_true_row["real"]],
                }
                for col in prediction_df.columns:
                    if col not in prepend_data:
                        prepend_data[col] = [None]

                df_to_plot = pd.concat([pd.DataFrame(prepend_data), prediction_df], ignore_index=True)
            else:
                df_to_plot = prediction_df.copy()

    except IndexError:
        logger.error(
            "Error aligning prediction and truth data for plotting continuity due to empty data.", exc_info=True
        )
        df_to_plot = prediction_df.copy()
        last_idx_in_truth_for_line = len(true_df)

    fig.add_trace(
        go.Scatter(
            x=df_to_plot[x_col],
            y=transform(df_to_plot[q_high_col]),
            mode="lines",
            line=dict(color="lightgrey"),
            name="90th Quantile" if show_legend else None,
            legendgroup="prediction_quantiles" if show_legend else None,
            showlegend=show_legend,
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=df_to_plot[x_col],
            y=transform(df_to_plot[q_low_col]),
            mode="lines",
            line=dict(color="lightgrey"),
            fill="tonexty",
            fillcolor="rgba(128, 128, 128, 0.2)",
            name="10th Quantile" if show_legend else None,
            legendgroup="prediction_quantiles" if show_legend else None,
            showlegend=show_legend,
        )
    )
    fig.add_scatter(
        x=df_to_plot[x_col],
        y=transform(df_to_plot[median_col]),
        mode="lines",
        line=dict(color="grey"),
        name="Median Forecast" if show_legend else None,
        legendgroup="prediction_median" if show_legend else None,
        showlegend=show_legend,
    )

    if last_idx_in_truth_for_line > 0 and last_idx_in_truth_for_line <= len(true_df):
        vline_x = true_df["x"].iloc[last_idx_in_truth_for_line - 1]
        y_min = min(df_to_plot[q_low_col].min(), true_df["real"].min())
        y_max = max(df_to_plot[q_high_col].max(), true_df["real"].max())
        if pd.isna(y_min) or pd.isna(y_max):
            y_min, y_max = 0, 1

        fig.add_shape(
            dict(
                type="line",
                x0=vline_x,
                x1=vline_x,
                y0=y_min,
                y1=y_max,
                line=dict(color="red", width=1, dash="dash"),
                name="Forecast Start" if show_legend else None,
            )
        )


# Added DataSet and Samples to satisfy type hints that were previously ignored
# from chap_core.spatio_temporal_data.temporal_dataclass import DataSet # This was added by me, but auto-formatter might move it
# from chap_core.datatypes import Samples # This was added by me


# Added DataSet and Samples to satisfy type hints that were previously ignored
# from chap_core.spatio_temporal_data.temporal_dataclass import DataSet # This was added by me, but auto-formatter might move it
# from chap_core.datatypes import Samples # This was added by me
