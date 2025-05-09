import pandas as pd

# External application-level imports — assumed to exist
from chap_core.api_types import FeatureCollectionModel
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet as _DataSet


def observations_to_dataset(dataclass, observations, fill_missing=False):
    """
    Converts a list of observation models into a structured DataSet.

    Args:
        dataclass: A data class defining schema for the resulting DataSet.
        observations: List of Pydantic models representing observations.
        fill_missing: Whether to fill in missing values in the DataSet.

    Returns:
        An instance of _DataSet constructed from the observation records.
    """
    # Convert each observation to a dictionary and create a DataFrame
    dataframe = pd.DataFrame([obs.model_dump() for obs in observations])

    # Rename columns to expected DataSet format
    dataframe = dataframe.rename(columns={"org_unit": "location", "period": "time_period"})

    # Set multi-index for later reshaping
    dataframe = dataframe.set_index(["location", "time_period"])

    # Pivot to get one column per feature_name, indexed by location and time_period
    pivoted = dataframe.pivot(columns="feature_name", values="value").reset_index()

    # Use the custom DataSet class to convert from pandas DataFrame
    new_dataset = _DataSet.from_pandas(pivoted, dataclass, fill_missing=fill_missing)

    return new_dataset


def dataset_model_to_dataset(dataclass, dataset, fill_missing=False):
    """
    Converts a full DatasetModel (observations + geojson) to a DataSet.

    Args:
        dataclass: The data schema class.
        dataset: A DatasetModel containing observations and geojson polygons.
        fill_missing: Whether to fill in missing features.

    Returns:
        A fully populated _DataSet instance with polygons assigned.
    """
    # First convert the observation records to a structured DataSet
    ds = observations_to_dataset(dataclass, dataset.observations, fill_missing=fill_missing)

    # Parse the geojson polygons using the Pydantic model
    polygons = FeatureCollectionModel.model_validate(dataset.geojson)

    # Set the polygons on the DataSet
    ds.set_polygons(polygons)

    return ds
