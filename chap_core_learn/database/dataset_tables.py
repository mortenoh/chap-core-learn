# Improvement Suggestions:
# 1. Module-Level Docstring: Add a comprehensive module docstring explaining the purpose of the file (defining dataset-related database tables).
# 2. Detailed Class and Field Docstrings: Provide thorough docstrings for all SQLModel classes and their fields, explaining their roles, data types, and relationships.
# 3. `DataSet.created` Default Timestamp: Set a `default_factory=datetime.utcnow` (or `datetime.now`) for the `DataSet.created` field to automatically record creation time.
# 4. GeoJSON Storage in `DataSet.geojson`: Clarify storage for `DataSet.geojson`. If structured GeoJSON, consider `sa_column=Column(JSON)` and type `Optional[FeatureCollectionModel]` or `Dict`.
# 5. `Observation.feature_name` Clarification: Clarify the purpose and typical values for `Observation.feature_name`, especially its relation to `DataSet.covariates` or other feature/target concepts.

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic_geojson import FeatureCollectionModel as _FeatureCollectionModelBase
from pydantic_geojson import FeatureModel
from sqlalchemy import JSON, Column
from sqlmodel import Field, Relationship

from chap_core.database.base_tables import DBModel, PeriodID


class FeatureCollectionModel(_FeatureCollectionModelBase):
    """
    Represents a GeoJSON FeatureCollection.

    This class extends the base FeatureCollectionModel from pydantic_geojson
    to potentially enforce a strict list type for features or for other customizations.
    """

    features: list[FeatureModel] = Field(..., description="A list of GeoJSON Feature objects.")


class ObservationBase(DBModel):
    """
    Base model for an observation, representing a single data point.
    It is not a table itself but provides common fields for Observation records.
    """

    period: PeriodID = Field(description="The time period identifier for this observation.")
    org_unit: str = Field(description="The organizational unit identifier associated with this observation.")
    value: Optional[float] = Field(default=None, description="The numerical value of the observation.")
    feature_name: Optional[str] = Field(
        default=None,
        description="Optional name of the feature this observation pertains to, e.g., a specific covariate or target variable name.",
    )


class Observation(ObservationBase, table=True):
    """
    Represents a single observation data point linked to a DataSet.
    This is a database table.
    """

    id: Optional[int] = Field(default=None, primary_key=True, description="Primary key for the observation.")
    dataset_id: int = Field(foreign_key="dataset.id", description="Foreign key linking to the DataSet table.")
    dataset: DataSet = Relationship(
        back_populates="observations", description="The DataSet this observation belongs to."
    )


class DataSetBase(DBModel):
    """
    Base model for a dataset, containing metadata about a collection of observations.
    It is not a table itself but provides common fields for DataSet records.
    """

    name: str = Field(description="The unique name of the dataset.")
    type: Optional[str] = Field(
        default=None, description="An optional type descriptor for the dataset (e.g., 'health', 'climate')."
    )
    geojson: Optional[str] = Field(
        default=None,
        description="Optional raw GeoJSON string representing the geographical features associated with this dataset. Consider storing as JSON object in DB for better querying.",
    )
    # Optional[FeatureCollectionModel] = Field(default=None, sa_type=AutoString) #fix from https://github.com/fastapi/sqlmodel/discussions/730#discussioncomment-7952622


class DataSet(DataSetBase, table=True):
    """
    Represents a dataset, which is a collection of observations and associated metadata.
    This is a database table.
    """

    id: Optional[int] = Field(default=None, primary_key=True, description="Primary key for the dataset.")
    observations: List[Observation] = Relationship(
        back_populates="dataset", description="A list of observations belonging to this dataset."
    )
    covariates: List[str] = Field(
        default_factory=list,
        sa_column=Column(JSON),
        description="A list of covariate names (strings) associated with this dataset.",
    )
    created: Optional[datetime] = Field(
        default=None,
        description="Timestamp of when the dataset record was created. Consider adding default_factory=datetime.utcnow.",
    )


class DataSetWithObservations(DataSetBase):
    """
    A Pydantic model representing a DataSet along with its observations.
    This model is typically used for API responses or data transfer, not as a database table.
    """

    id: int = Field(description="The ID of the dataset.")
    observations: List[ObservationBase] = Field(description="A list of observations associated with the dataset.")
    created: Optional[datetime] = Field(default=None, description="Timestamp of when the dataset record was created.")
