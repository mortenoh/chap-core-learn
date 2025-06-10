import numpy as np
import pandas as pd

# Domain-specific dataclasses and utilities for health and climate datasets
from chap_core.datatypes import FullData, HealthData
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet
from chap_core.time_period import PeriodRange
from chap_core.time_period.date_util_wrapper import Month, convert_time_period_string


def hydromet(filename: str) -> DataSet:
    """
    Loads hydrometeorological dengue dataset and structures it into a DataSet.

    Parameters:
        filename (str): Path to the CSV file containing the raw data.

    Returns:
        DataSet: A mapping of 'micro_code' to FullData entries containing dengue and climate data.
    """
    df = pd.read_csv(filename)

    # Ensure data is sorted consistently by spatial and temporal attributes
    df = df.sort_values(by=["micro_code", "year", "month"])

    # Group by 'micro_code' to prepare for building spatially keyed records
    grouped = df.groupby("micro_code")

    data_dict = {}

    # Loop through each group to build FullData objects
    for name, group in grouped:
        # Construct a Month object from 'year' and 'month' series
        period = Month(group["year"], group["month"])

        # Extract tmax and tmin values and compute the average temperature
        tmax = group["tmax"].values
        tmin = group["tmin"].values
        tmean = (tmax + tmin) / 2

        # Create a FullData instance for this micro_code group
        data_dict[name] = FullData(
            period=period,
            tmin=np.zeros_like(tmean),  # Placeholder if tmin not separately tracked
            tmax=tmean,  # Storing tmean as tmax (possibly a design quirk)
            cases=group["dengue_cases"].values,
            population=group["population"].values,
        )

    # Wrap all spatially-keyed FullData into a unified DataSet
    return DataSet(data_dict)


def rwanda_data(filename: str) -> DataSet:
    """
    Cleans and loads Rwandan malaria case data from Excel.

    NOTE: The input `filename` is ignored and a hardcoded path is used instead.
    Consider refactoring to use the argument for production use.

    Returns:
        DataSet: A dataset of total malaria cases by sector and period.
    """
    # Hardcoded path (should be passed as an argument ideally)
    filename = "/home/knut/Downloads/data/Malaria Cases Final.xlsx"

    # Load the Excel sheet
    df = pd.read_excel(filename, sheet_name="Sheet1")

    # Optional: Save as CSV for caching or debugging
    df.to_csv("/home/knut/Downloads/data/malaria_cases.csv")

    # Columns containing age/sex-specific malaria case counts
    case_names = "Under5_F\tUnder5_M\t5-19_F\t5-19_M\t20 +_F\t20 +_M".split("\t")
    case_names = [name.strip() for name in case_names]

    # Sum cases across all relevant columns
    cases = sum([df[name].values for name in case_names])

    # Build pandas Periods (monthly) from Year and Month
    period = [pd.Period(f"{year}-{month}") for year, month in zip(df["Year"], df["Period"])]

    # Construct a cleaned DataFrame
    clean_df = pd.DataFrame(
        {
            "location": df["Sector"],
            "time_period": period,
            "disease_cases": cases,
        }
    )

    # Save cleaned output for inspection/debugging
    clean_df.to_csv("/home/knut/Downloads/data/malaria_clean.csv", index=False)

    # Convert to internal DataSet format
    return DataSet.from_pandas(clean_df, dataclass=HealthData)


def laos_data(filename: str) -> DataSet:
    """
    Loads Laos health data, converts period strings, and maps each column to HealthData.

    Parameters:
        filename (str): Path to the CSV file.

    Returns:
        DataSet: A mapping from data columns (e.g., disease names) to HealthData.
    """
    df = pd.read_csv(filename)

    # Ensure consistent order by 'periodid'
    df = df.sort_values(by=["periodid"])

    # Convert each period ID string into standardized format
    periods = [convert_time_period_string(str(row)) for row in df["periodid"]]

    # Debug print to inspect parsed periods (can be removed in production)
    print(periods)

    # Build a PeriodRange to apply across all time-series
    period_range = PeriodRange.from_strings(periods)

    # Map each column (excluding first 4, assumed to be metadata) to HealthData
    return DataSet({colname: HealthData(period_range, df[colname].values) for colname in df.columns[4:]})
