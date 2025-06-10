# Improvement Suggestions:
# 1. Comprehensive Docstrings: Add module, class, and method docstrings throughout.
# 2. Correct `ModelSpec` Instantiation: Change `period=...` to `supported_period_types=...` in `seed_with_session_wrapper`.
# 3. Robust Seeding Logic: Make selection of `base_covariates` and `target_type` in seeding more robust than list indices.
# 4. Review Commented-Out Seed Data: Evaluate and either integrate or remove the commented-out `ModelSpec` instances.
# 5. Type Hint for `session_wrapper`: Add `SessionWrapper` type hint to `seed_with_session_wrapper` parameter.

from typing import List, Optional

from sqlalchemy import JSON, Column
from sqlmodel import Field, Relationship

from chap_core.database.base_tables import DBModel
from chap_core.database.database import SessionWrapper  # Added for type hinting
from chap_core.model_spec import PeriodType


class FeatureTypeBase(DBModel):
    """Base model for a feature type, defining its display name and description."""

    display_name: str = Field(description="User-friendly display name for the feature type.")
    description: str = Field(description="A detailed description of the feature type.")


class FeatureTypeRead(FeatureTypeBase):
    """Pydantic model for reading a FeatureType, including its unique name."""

    name: str = Field(description="The unique programmatic name of the feature type.")


class FeatureType(FeatureTypeBase, table=True):
    """
    Database table model for a feature type.

    Feature types define the kinds of data that can be used as covariates or targets
    in models (e.g., 'rainfall', 'disease_cases').
    """

    name: str = Field(primary_key=True, description="Unique programmatic name for the feature type (e.g., 'rainfall').")


class FeatureSource(DBModel, table=True):
    """
    Database table model for a feature source.

    Feature sources describe where and how specific feature types can be obtained
    (e.g., a specific API endpoint for 'rainfall' data from a particular provider).
    """

    name: str = Field(primary_key=True, description="Unique name for the feature source.")
    display_name: str = Field(description="User-friendly display name for the feature source.")
    feature_type: str = Field(
        foreign_key="featuretype.name", description="The name of the FeatureType this source provides."
    )
    provider: str = Field(description="Identifier for the data provider or mechanism (e.g., 'gee', 'dhis2').")
    supported_period_types: List[PeriodType] = Field(
        default_factory=list,
        sa_column=Column(JSON),
        description="List of period types (e.g., monthly, weekly) supported by this source.",
    )


class ModelFeatureLink(DBModel, table=True):
    """
    Link table for the many-to-many relationship between ModelSpec (covariates) and FeatureType.
    """

    model_id: Optional[int] = Field(
        default=None, foreign_key="modelspec.id", primary_key=True, description="Foreign key to the ModelSpec table."
    )
    feature_type: Optional[str] = Field(
        default=None,
        foreign_key="featuretype.name",
        primary_key=True,
        description="Foreign key to the FeatureType table (name of the covariate).",
    )


class ModelSpecBase(DBModel):
    """Base model for a model specification, containing common metadata fields."""

    name: str = Field(description="Unique programmatic name for the model specification.")
    display_name: str = Field(description="User-friendly display name for the model.")
    supported_period_types: PeriodType = Field(
        default=PeriodType.any,
        description="The time period granularity this model supports (e.g., monthly, weekly, any).",
    )
    description: str = Field(default="No Description yet", description="A detailed description of the model.")
    author: str = Field(default="Unknown Author", description="The author(s) of the model.")
    organization: Optional[str] = Field(default=None, description="The organization responsible for the model.")
    organization_logo_url: Optional[str] = Field(default=None, description="URL to the organization's logo.")
    source_url: Optional[str] = Field(default=None, description="URL to the model's source code or primary reference.")
    contact_email: Optional[str] = Field(default=None, description="Contact email for inquiries about the model.")
    citation_info: Optional[str] = Field(default=None, description="Information on how to cite the model.")


class ModelSpecRead(ModelSpecBase):
    """Pydantic model for reading a ModelSpec, including its ID and resolved covariate/target FeatureTypes."""

    id: int = Field(description="Primary key of the model specification.")
    covariates: List[FeatureTypeRead] = Field(description="List of feature types used as covariates by the model.")
    target: FeatureTypeRead = Field(description="The feature type used as the target variable by the model.")


class ModelSpec(ModelSpecBase, table=True):
    """
    Database table model for a model specification.

    This table stores metadata about different predictive models available in the system,
    including their supported features, target variable, and descriptive information.
    """

    id: Optional[int] = Field(default=None, primary_key=True, description="Primary key for the model specification.")
    covariates: List[FeatureType] = Relationship(
        link_model=ModelFeatureLink,
        description="List of feature types used as covariates, linked via ModelFeatureLink.",
    )
    target_name: str = Field(
        foreign_key="featuretype.name", description="Name of the feature type used as the target variable."
    )
    target: FeatureType = Relationship(description="The FeatureType instance representing the target variable.")


target_type = FeatureType(name="disease_cases", display_name="Disease Cases", description="Disease Cases")


def seed_with_session_wrapper(session_wrapper: SessionWrapper) -> None:
    """
    Seeds the database with default feature types and model specifications.

    This function populates the FeatureType and ModelSpec tables with a predefined
    set of common features (like rainfall, temperature, population) and a
    standard target ('disease_cases'). It also adds specifications for several
    default models available in the CHAP system.

    Args:
        session_wrapper: An active SessionWrapper instance for database interaction.
    """
    seeded_feature_types_data = [
        {"name": "rainfall", "display_name": "Precipitation", "description": "Precipitation in mm"},
        {
            "name": "mean_temperature",
            "display_name": "Mean Temperature",
            "description": "A measurement of mean temperature",
        },
        {"name": "population", "display_name": "Population", "description": "Population"},
        {"name": "disease_cases", "display_name": "Disease Cases", "description": "Disease Cases"},
    ]

    db_feature_types = {}
    for ft_data in seeded_feature_types_data:
        ft = FeatureType(**ft_data)
        db_feature_types[ft_data["name"]] = session_wrapper.create_if_not_exists(ft, id_name="name")

    # Ensure target_type is the one from the database after seeding/retrieval
    db_target_type = db_feature_types["disease_cases"]
    base_covariates = [
        db_feature_types["rainfall"],
        db_feature_types["mean_temperature"],
        db_feature_types["population"],
    ]

    seeded_models = [
        ModelSpec(
            name="naive_model",
            display_name="Naive model used for testing",
            # parameters={}, # Assuming parameters is not a direct field of ModelSpec table
            target=db_target_type,
            covariates=base_covariates,
            supported_period_types=PeriodType.month,
            description="A simple naive model only to be used for testing purposes.",
            author="CHAP team",
            organization="HISP Centre, University of Oslo",
            organization_logo_url="https://landportal.org/sites/default/files/2024-03/university_of_oslo_logo.png",
            source_url="NA",
            contact_email="knut.rand@dhis2.org",
            citation_info='Climate Health Analytics Platform. 2025. "Naive model used for testing". HISP Centre, University of Oslo. https://dhis2-chap.github.io/chap-core/external_models/overview_of_supported_models.html',
        ),
        ModelSpec(
            name="chap_ewars_weekly",
            display_name="Weekly CHAP-EWARS model",
            # parameters={},
            target=db_target_type,
            covariates=base_covariates,
            supported_period_types=PeriodType.week,
            description="Modified version of the World Health Organization (WHO) EWARS model. EWARS is a Bayesian hierarchical model implemented with the INLA library.",
            author="CHAP team",
            organization="HISP Centre, University of Oslo",
            organization_logo_url="https://landportal.org/sites/default/files/2024-03/university_of_oslo_logo.png",
            source_url="https://github.com/sandvelab/chap_auto_ewars_weekly@737446a7accf61725d4fe0ffee009a682e7457f6",
            contact_email="knut.rand@dhis2.org",
            citation_info='Climate Health Analytics Platform. 2025. "Weekly CHAP-EWARS model". HISP Centre, University of Oslo. https://dhis2-chap.github.io/chap-core/external_models/overview_of_supported_models.html',
        ),
        ModelSpec(
            name="chap_ewars_monthly",
            display_name="Monthly CHAP-EWARS",
            # parameters={},
            target=db_target_type,
            covariates=base_covariates,
            supported_period_types=PeriodType.month,
            description="Modified version of the World Health Organization (WHO) EWARS model. EWARS is a Bayesian hierarchical model implemented with the INLA library.",
            author="CHAP team",
            organization="HISP Centre, University of Oslo",
            organization_logo_url="https://landportal.org/sites/default/files/2024-03/university_of_oslo_logo.png",
            source_url="https://github.com/sandvelab/chap_auto_ewars@58d56f86641f4c7b09bbb635afd61740deff0640",
            contact_email="knut.rand@dhis2.org",
            citation_info='Climate Health Analytics Platform. 2025. "Monthly CHAP-EWARS model". HISP Centre, University of Oslo. https://dhis2-chap.github.io/chap-core/external_models/overview_of_supported_models.html',
        ),
        ModelSpec(
            name="auto_regressive_weekly",
            display_name="Weekly Deep Auto Regressive",
            # parameters={},
            target=db_target_type,
            covariates=base_covariates,
            supported_period_types=PeriodType.week,
            description="An experimental deep learning model based on an RNN architecture, focusing on predictions based on auto-regressive time series data.",
            author="Knut Rand",
            organization="HISP Centre, University of Oslo",
            organization_logo_url="https://landportal.org/sites/default/files/2024-03/university_of_oslo_logo.png",
            source_url="https://github.com/knutdrand/weekly_ar_model@1730b26996201d9ee0faf65695f44a2410890ea5",
            contact_email="knut.rand@dhis2.org",
            citation_info='Rand, Knut. 2025. "Weekly Deep Auto Regressive model". HISP Centre, University of Oslo. https://dhis2-chap.github.io/chap-core/external_models/overview_of_supported_models.html',
        ),
        ModelSpec(
            name="auto_regressive_monthly",
            display_name="Monthly Deep Auto Regressive",  # Corrected typo from displayName
            # parameters={},
            target=db_target_type,
            covariates=base_covariates,
            supported_period_types=PeriodType.month,
            description="An experimental deep learning model based on an RNN architecture, focusing on predictions based on auto-regressive time series data.",
            author="Knut Rand",
            organization="HISP Centre, University of Oslo",
            organization_logo_url="https://landportal.org/sites/default/files/2024-03/university_of_oslo_logo.png",
            source_url="https://github.com/sandvelab/monthly_ar_model@89f070dbe6e480d1e594e99b3407f812f9620d6d",
            contact_email="knut.rand@dhis2.org",
            citation_info='Rand, Knut. 2025. "Monthly Deep Auto Regressive model". HISP Centre, University of Oslo. https://dhis2-chap.github.io/chap-core/external_models/overview_of_supported_models.html',
        ),
    ]
    # Removed commented out ModelSpec block for brevity in this step.
    # It should be reviewed: either integrate valid models or delete obsolete ones.

    for model_spec_data in seeded_models:
        # Parameters field is not part of ModelSpec table, so it's removed from instantiation
        # It might be handled differently, e.g. via a separate table or JSON field if needed.
        session_wrapper.create_if_not_exists(model_spec_data, id_name="name")
