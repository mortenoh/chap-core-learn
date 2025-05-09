# Improvement Suggestions:
# 1. Comprehensive Docstrings: Add module, class, and field docstrings throughout.
# 2. Relocate `test()` Function and TODOs: Move `test()` to a test file and TODOs to an issue tracker.
# 3. Automatic Timestamps: Use `default_factory=datetime.utcnow` for `created` fields.
# 4. Review `cascade_delete` Usage: Ensure `cascade_delete=True` is appropriate for all relevant relationships.
# 5. Consistency in Naming/Typing: Ensure consistent naming and typing, e.g., `org_unit` vs `org_unity`.

from __future__ import annotations

import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import JSON, Column
from sqlmodel import Field, Relationship

from chap_core.database.base_tables import DBModel, PeriodID


class BackTestBase(DBModel):
    """Base model for a backtest run, containing metadata about the evaluation."""

    dataset_id: int = Field(foreign_key="dataset.id", description="ID of the dataset used for this backtest.")
    model_id: str = Field(description="Identifier of the model that was backtested.")
    name: Optional[str] = Field(default=None, description="Optional user-defined name for this backtest run.")
    created: Optional[datetime.datetime] = Field(
        default_factory=datetime.datetime.utcnow,
        description="Timestamp of when this backtest record was created (UTC).",
    )


class BackTest(BackTestBase, table=True):
    """
    Database table model for a backtest run.

    A backtest involves evaluating a model over multiple historical periods.
    This table stores the overall metadata for a backtest and links to its
    constituent forecasts and calculated metrics.
    """

    id: Optional[int] = Field(default=None, primary_key=True, description="Primary key for the backtest.")
    forecasts: List[BackTestForecast] = Relationship(
        back_populates="backtest",
        cascade_delete=True,
        description="List of individual forecasts generated during this backtest.",
    )
    metrics: List[BackTestMetric] = Relationship(
        back_populates="backtest", cascade_delete=True, description="List of metrics calculated for this backtest."
    )


class ForecastBase(DBModel):
    """Base model for forecast data, specifying the period and organizational unit."""

    period: PeriodID = Field(description="The time period this forecast pertains to.")
    org_unit: str = Field(description="The organizational unit this forecast is for.")


class PredictionBase(DBModel):
    """Base model for a prediction run, containing metadata about the prediction task."""

    dataset_id: int = Field(foreign_key="dataset.id", description="ID of the dataset used as input for the prediction.")
    model_id: str = Field(description="Identifier of the model used for this prediction.")
    n_periods: int = Field(description="Number of periods forecasted ahead.")
    name: str = Field(description="User-defined name for this prediction run.")
    created: datetime.datetime = Field(
        default_factory=datetime.datetime.utcnow,
        description="Timestamp of when this prediction record was created (UTC).",
    )
    meta_data: Dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column(JSON),
        description="Arbitrary metadata associated with the prediction (stored as JSON).",
    )


class Prediction(PredictionBase, table=True):
    """
    Database table model for a prediction run.

    This table stores metadata for a specific prediction task and links to the
    generated forecast sample entries.
    """

    id: Optional[int] = Field(default=None, primary_key=True, description="Primary key for the prediction.")
    forecasts: List[PredictionSamplesEntry] = Relationship(
        back_populates="prediction",
        cascade_delete=True,
        description="List of forecast sample entries for this prediction.",
    )


class ForecastRead(ForecastBase):
    """Pydantic model for reading forecast data, including the list of forecast values."""

    values: List[float] = Field(
        default_factory=list, sa_column=Column(JSON), description="List of forecast values for the period and org_unit."
    )


class PredictionInfo(PredictionBase):
    """Pydantic model providing information about a prediction, including its ID."""

    id: int = Field(description="Primary key of the prediction.")


class PredictionRead(PredictionInfo):
    """Pydantic model for reading a full prediction, including its metadata and associated forecasts."""

    forecasts: List[ForecastRead] = Field(description="List of forecasts associated with this prediction.")


class PredictionSamplesEntry(ForecastBase, table=True):
    """
    Database table model for storing individual forecast sample entries for a prediction.

    Each entry corresponds to a specific period and organizational unit within a larger prediction task.
    """

    id: Optional[int] = Field(default=None, primary_key=True, description="Primary key for the forecast sample entry.")
    prediction_id: int = Field(foreign_key="prediction.id", description="Foreign key linking to the Prediction table.")
    prediction: Prediction = Relationship(
        back_populates="forecasts", description="The Prediction this forecast entry belongs to."
    )
    values: List[float] = Field(
        default_factory=list, sa_column=Column(JSON), description="List of forecast sample values for this entry."
    )


class BackTestForecast(ForecastBase, table=True):
    """
    Database table model for storing individual forecast entries generated during a backtest.
    """

    id: Optional[int] = Field(
        default=None, primary_key=True, description="Primary key for the backtest forecast entry."
    )
    backtest_id: int = Field(foreign_key="backtest.id", description="Foreign key linking to the BackTest table.")
    last_train_period: PeriodID = Field(description="The last period included in the training data for this forecast.")
    last_seen_period: PeriodID = Field(description="The last period of actual data seen before making this forecast.")
    backtest: BackTest = Relationship(
        back_populates="forecasts", description="The BackTest this forecast entry belongs to."
    )
    values: List[float] = Field(
        default_factory=list, sa_column=Column(JSON), description="List of forecast values for this backtest entry."
    )


class BackTestMetric(DBModel, table=True):
    """
    Database table model for storing metrics calculated during a backtest.
    """

    id: Optional[int] = Field(default=None, primary_key=True, description="Primary key for the backtest metric entry.")
    backtest_id: int = Field(foreign_key="backtest.id", description="Foreign key linking to the BackTest table.")
    metric_id: str = Field(description="Identifier for the type of metric (e.g., 'MAE', 'RMSE').")
    period: PeriodID = Field(description="The forecast period this metric pertains to.")
    last_train_period: PeriodID = Field(description="The last training period relevant to this metric calculation.")
    last_seen_period: PeriodID = Field(description="The last period of actual data seen relevant to this metric.")
    value: float = Field(description="The calculated value of the metric.")
    backtest: BackTest = Relationship(back_populates="metrics", description="The BackTest this metric belongs to.")
