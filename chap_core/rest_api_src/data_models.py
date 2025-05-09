from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field

from chap_core.api_types import FeatureCollectionModel
from chap_core.database.base_tables import DBModel
from chap_core.database.dataset_tables import DataSetBase, ObservationBase
from chap_core.database.tables import BackTestBase, BackTestForecast, BackTestMetric

# ---------- PREDICTION MODELS ----------


class PredictionBase(BaseModel):
    """Base schema for a prediction"""

    orgUnit: str  # Organization unit ID
    dataElement: str  # Name or ID of the predicted variable
    period: str  # Time period for the prediction


class PredictionResponse(PredictionBase):
    """Single prediction response (point estimate)"""

    value: float  # Predicted value


class PredictionSamplResponse(PredictionBase):
    """Prediction response with uncertainty represented as a list of sample values"""

    values: List[float]  # Sampled prediction values


class FullPredictionResponse(BaseModel):
    """Response model for full prediction request"""

    diseaseId: str
    dataValues: List[PredictionResponse]  # List of predictions


class FullPredictionSampleResponse(BaseModel):
    """Response model for full prediction request with uncertainty samples"""

    diseaseId: str
    dataValues: List[PredictionSamplResponse]  # List of sampled predictions


# ---------- FETCH / DATASET REQUEST MODELS ----------


class FetchRequest(DBModel):
    """Request to fetch data from an external source"""

    feature_name: str  # Name of the feature to fetch
    data_source_name: str  # Data source identifier


class DatasetMakeRequest(DataSetBase):
    """
    Request for making a new dataset by combining provided observations and
    fetched external data, linked to a GeoJSON layer.
    """

    geojson: FeatureCollectionModel  # Polygons for spatial dimension
    provided_data: List[ObservationBase]  # Observations the user already provides
    data_to_be_fetched: List[FetchRequest]  # Features that should be fetched from external sources


# ---------- JOB MODELS ----------


class JobResponse(BaseModel):
    """Basic job response wrapper (e.g., for Celery job tracking)"""

    id: str  # Job/task ID


# ---------- BACKTEST MODELS ----------


class BackTestCreate(BackTestBase):
    """Request model to initiate a new backtest (inherits all fields from base)"""

    ...


class BackTestRead(BackTestBase):
    """
    Read model for querying an existing backtest
    Includes additional fields for ID and filtering options
    """

    id: int
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    org_unit_ids: List[str] = Field(default_factory=list)


class BackTestFull(BackTestRead):
    """
    Full backtest response including associated metrics and forecast results
    """

    metrics: List[BackTestMetric]  # Evaluation metrics for the backtest
    forecasts: List[BackTestForecast]  # Model outputs for the backtest
