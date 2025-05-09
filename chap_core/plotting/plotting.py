# Improvement Suggestions:
# 1. **Module Docstring**: Add a comprehensive module docstring explaining the purpose of this file: to provide core plotting functions for visualizing CHAP-core time series data (like ClimateHealthTimeSeries, ClimateData, HealthData) using the Plotly library. (Primary task).
# 2. **Standardized Function Docstrings**: Refactor the docstrings for `plot_timeseries_data` and `plot_multiperiod` to follow a consistent, standard format (e.g., including clear Args, Returns, and potentially Raises sections) for improved readability and maintainability.
# 3. **Robust Input Data Handling**:
#    - In `plot_multiperiod`, add checks for empty DataFrames after `climate_data.topandas().head(head)` and `health_data.topandas()` to prevent `IndexError` if data is empty or `head` results in an empty frame.
#    - The line `cut_off_idx = (health_df.time_period == cut_off).to_list().index(True) + 1` in `plot_multiperiod` can raise a `ValueError` if `cut_off` is not found. Implement more robust handling, perhaps by logging a warning and plotting up to the available data, or raising a custom error.
# 4. **Configurable Variables in `plot_multiperiod`**:
#    - The function currently hardcodes plotting "mean_temperature" and "disease_cases". Make these variable names parameters of the function to allow plotting other relevant climate or health variables.
#    - The frequency "M" for `pd.Period(year=last_year, month=last_month, freq="M")` is hardcoded. If input data can have different frequencies, this should ideally be derived from `climate_data.time_period.freq` or made a parameter.
# 5. **Enhanced Plot Customization**: Allow users to pass more Plotly layout and trace customization options as parameters to the functions (e.g., titles, axis labels, colors, line shapes, figure size). This would provide greater control over the plot aesthetics.

"""
This module provides core plotting functionalities for visualizing time series data
within the CHAP-core project, primarily using the Plotly library.

It includes functions to:
- Plot multi-variable `ClimateHealthTimeSeries` data with faceted subplots.
- Plot `ClimateData` (e.g., temperature) and `HealthData` (e.g., disease cases)
  on a shared time axis with dual y-axes for comparison.
"""

import logging

logger = logging.getLogger(__name__)

from typing import Optional  # Added for Optional type hint

import pandas as pd
import plotly.express as px
from plotly.graph_objs import Figure
from plotly.subplots import make_subplots

from chap_core.datatypes import ClimateData, ClimateHealthTimeSeries, HealthData


def plot_timeseries_data(data: ClimateHealthTimeSeries) -> Figure:
    """
    Plots ClimateHealthTimeSeries data, creating one subplot for each variable.

    Each subplot displays a variable against the time_period on the x-axis.

    Args:
        data (ClimateHealthTimeSeries): The input data containing multiple time series variables.

    Returns:
        Figure: A Plotly Figure object that can be shown or saved.
    """
    df = data.topandas()
    if df.empty:
        # Return an empty figure or raise error if data is empty
        fig = Figure()
        fig.update_layout(title_text="Climate Health Data (No data to plot)")
        return fig

    df = pd.melt(df, id_vars=["time_period"], var_name="variable", value_name="value")
    fig = px.line(
        df,
        x="time_period",
        y="value",
        facet_row="variable",
        title="Climate Health Data",
    )
    # Allow y-axes of subplots to have independent scales
    fig.update_yaxes(matches=None)
    # Ensure each subplot y-axis title is just the variable name
    for i, var_name in enumerate(df["variable"].unique()):
        fig.update_yaxes(title_text=var_name, row=i + 1, col=1)
    return fig


def plot_multiperiod(
    climate_data: ClimateData,
    health_data: HealthData,
    head: Optional[int] = None,
    climate_var: str = "mean_temperature",
    health_var: str = "disease_cases",
) -> Figure:
    """
    Plots specified climate and health data variables on the same plot with dual y-axes.

    The time_period is on the x-axis. The plot is aligned based on the end date
    of the climate data (or the `head` parameter).

    Args:
        climate_data (ClimateData): The climate data to plot.
        health_data (HealthData): The health data to plot.
        head (Optional[int]): Number of initial rows from climate_data to plot.
                              If None, all rows are plotted. Defaults to None.
        climate_var (str): The name of the climate variable to plot from `climate_data`.
                           Defaults to "mean_temperature".
        health_var (str): The name of the health variable to plot from `health_data`.
                          Defaults to "disease_cases".

    Returns:
        Figure: A Plotly Figure object with dual y-axes.

    Raises:
        ValueError: If input data is empty or required variables are not found.
    """
    climate_df = climate_data.topandas()
    if climate_df.empty:
        raise ValueError("Climate data is empty, cannot generate plot.")
    if climate_var not in climate_df.columns:
        raise ValueError(
            f"Climate variable '{climate_var}' not found in climate_data. Available: {climate_df.columns.tolist()}"
        )

    climate_df = climate_df.head(head)  # Apply head after checking for variable
    if climate_df.empty:  # Check again after head
        raise ValueError("Climate data is empty after applying 'head' parameter, cannot generate plot.")

    climate_df["time_period"] = climate_df["time_period"].dt.to_timestamp()  # Use .loc to avoid SettingWithCopyWarning

    last_month = climate_df["time_period"].iloc[-1].month
    last_year = climate_df["time_period"].iloc[-1].year
    # TODO: Infer frequency from data instead of hardcoding "M"
    cut_off_period = pd.Period(year=last_year, month=last_month, freq="M")

    health_df = health_data.topandas()
    if health_df.empty:
        raise ValueError("Health data is empty, cannot generate plot.")
    if health_var not in health_df.columns:
        raise ValueError(
            f"Health variable '{health_var}' not found in health_data. Available: {health_df.columns.tolist()}"
        )

    try:
        cut_off_idx = (health_df["time_period"] == cut_off_period).to_list().index(True) + 1
        health_df_processed = health_df.head(cut_off_idx)
    except ValueError:
        # cut_off_period not found in health_df, plot all available health_df or up to climate_df's range
        health_df_processed = health_df[health_df["time_period"] <= cut_off_period]
        if health_df_processed.empty:
            logging.warning(
                f"No health data found up to climate data cut-off {cut_off_period}. Health data will not be plotted effectively."
            )
        else:
            logging.info(
                f"Cut-off period {cut_off_period} not found in health data. Plotting available health data up to this period."
            )

    if health_df_processed.empty:  # If still empty after processing
        logging.warning("Processed health data is empty. Plot will only show climate data.")
        # Create an empty DataFrame with expected columns to avoid errors in trace creation
        health_df_processed = pd.DataFrame(columns=["time_period", health_var])
    else:
        health_df_processed = health_df_processed.copy()  # Avoid SettingWithCopyWarning
        health_df_processed.loc[:, "time_period"] = health_df_processed["time_period"].dt.to_timestamp()

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Climate data trace
    if not climate_df.empty:
        climate_trace = px.line(climate_df, x="time_period", y=climate_var)
        climate_trace.update_traces(line_color="#1E88E5", name=climate_var.replace("_", " ").title())
        fig.add_traces(climate_trace.data)

    # Health data trace
    if not health_df_processed.empty:
        disease_trace = px.line(health_df_processed, x="time_period", y=health_var, line_shape="vh")
        disease_trace.update_traces(line_color="#D81B60", name=health_var.replace("_", " ").title())
        disease_trace.update_traces(yaxis="y2")
        fig.add_traces(disease_trace.data)

    fig.update_layout(title_text="Climate and Health Data Comparison")
    fig.layout.xaxis.title = "Time"
    fig.layout.yaxis.title = climate_var.replace("_", " ").title()
    fig.layout.yaxis2.title = health_var.replace("_", " ").title()

    # Ensure yaxis2 is visible if health data was plotted
    if not health_df_processed.empty:
        fig.layout.yaxis2.showgrid = False  # Example: hide grid for secondary axis if desired
    else:  # If no health data, hide the secondary y-axis to avoid confusion
        fig.layout.yaxis2.visible = False

    return fig
