# Improvement Suggestions:
# 1. Add `Field` descriptions for all model attributes to provide more context on each field (e.g., for `pe`, `ou` in `DataElement`).
# 2. Consider using `Enum` types for fields with a limited set of known values (e.g., if `estimator_id` in `RequestV2` has a predefined list of valid estimators).
# 3. Add examples of usage or expected data format in the docstrings for more complex models like `RequestV1` or `FeatureCollectionModel`.
# 4. Review naming consistency (e.g., `featureId` vs. `dataElement` vs. `orgUnit`). While this might be for external compatibility, internal consistency is good if possible.
# 5. For models like `DataElement` and `DataElementV2` which are very similar, consider if a base class or aliasing could reduce duplication if their evolution is linked.

"""
This module defines Pydantic models used for API request and response data structures,
including GeoJSON extensions, data upload formats, and prediction/evaluation outputs.
"""

from typing import Any, Optional

from pydantic import BaseModel, Field
from pydantic_geojson import FeatureCollectionModel as _FeatureCollectionModel
from pydantic_geojson import FeatureModel as _FeatureModel

# --- GeoJSON Wrapper Models ---


class FeatureModel(_FeatureModel):
    """Extended GeoJSON Feature model supporting optional `id` and flexible `properties`."""

    id: Optional[str] = None
    properties: Optional[dict[str, Any]] = Field(default_factory=dict)


class FeatureCollectionModel(_FeatureCollectionModel):
    """FeatureCollection using the extended FeatureModel."""

    features: list[FeatureModel]


# --- Data Upload Formats (V1 + V2) ---


class DataElement(BaseModel):
    """Single DHIS2-style data element."""

    pe: str = Field(..., description="Period of the data element (e.g., '202301').")
    ou: str = Field(..., description="Organisation unit ID for the data element.")
    value: Optional[float] = Field(None, description="Value of the data element.")


class DataList(BaseModel):
    """List of data elements tied to a specific feature."""

    featureId: str = Field(..., description="Identifier of the GeoJSON feature this data list belongs to.")
    dhis2Id: str = Field(..., description="DHIS2 data element or indicator ID.")
    data: list[DataElement] = Field(..., min_items=1, description="List of data observations.")


class DataElementV2(BaseModel):
    """V2-compatible data element format."""

    period: str = Field(..., description="Period of the data element (e.g., '202301').")
    orgUnit: str = Field(..., description="Organisation unit ID for the data element.")
    value: Optional[float] = Field(None, description="Value of the data element.")


class DataListV2(BaseModel):
    """V2-compatible data list format."""

    featureId: str = Field(..., description="Identifier of the GeoJSON feature this data list belongs to.")
    dataElement: str = Field(..., description="DHIS2 data element or indicator ID.")
    data: list[DataElementV2] = Field(..., min_items=1, description="List of data observations.")


# --- API Request Models ---


class RequestV1(BaseModel):
    """Base request format used for forecasting (V1)."""

    orgUnitsGeoJson: FeatureCollectionModel = Field(..., description="GeoJSON FeatureCollection of organisation units.")
    features: list[DataList] = Field(..., description="List of data associated with features.")


class RequestV2(RequestV1):
    """Extended request supporting model selection (V2). Inherits from RequestV1."""

    estimator_id: str = Field("chap_ewars_monthly", description="Identifier of the estimation model to use.")


class PredictionRequest(RequestV2):
    """Forecast prediction request, building on RequestV2."""

    n_periods: int = Field(3, description="Number of future periods to forecast.", gt=0)
    include_data: bool = Field(False, description="Whether to include input data in the response.")


# --- Prediction Output ---


class PredictionEntry(BaseModel):
    """A single forecasted value for an org unit and period."""

    orgUnit: str = Field(..., description="Organisation unit ID.")
    period: str = Field(..., description="Forecasted period.")
    quantile: float = Field(..., description="Quantile of the prediction (e.g., 0.5 for median).")
    value: float = Field(..., description="Predicted value for the quantile.")


class EvaluationEntry(PredictionEntry):
    """Prediction entry extended with the backtesting split period for evaluation purposes."""

    splitPeriod: str = Field(..., description="Period used as the split point for backtesting.")


class EvaluationResponse(BaseModel):
    """Final evaluation output returned by CHAP API, containing actual cases and predictions."""

    actualCases: DataList = Field(..., description="Actual observed cases for the evaluation period.")
    predictions: list[EvaluationEntry] = Field(..., description="List of prediction entries for evaluation.")


# --- Time-Aligned Data Class ---


class PeriodObservation(BaseModel):
    """
    Helper model for time series construction, representing an observation for a specific time period.
    This class is typically used internally during data processing.
    """

    time_period: str = Field(..., description="The time period of the observation (e.g., '2023W01', '202301').")
