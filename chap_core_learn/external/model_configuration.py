# Improvement Suggestions:
# 1. Comprehensive Docstrings: Add module, class, and field docstrings for all Pydantic models.
# 2. Explicit Field Descriptions: Ensure all Pydantic model fields have clear descriptions.
# 3. Review `ModelTemplateConfig.required_fields` Default: Document or change the default for `required_fields`.
# 4. Clarify `ModelTemplateConfig.adapters`: Explain the purpose and structure of the `adapters` field.
# 5. Type Hinting for `CommandConfig.parameters`: Consider more specific typing or detailed docstring for `parameters`.

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class DockerEnvConfig(BaseModel):
    """Configuration for a Docker environment required by an external model."""

    image: str = Field(description="The Docker image name and tag (e.g., 'user/repository:tag').")


class CommandConfig(BaseModel):
    """Configuration for a command to be executed, including parameters."""

    command: str = Field(
        description="The command string to be executed (e.g., 'python train.py', 'Rscript predict.R')."
    )
    parameters: Optional[Dict[str, str]] = Field(
        default=None,
        description="Optional dictionary of parameters to be passed to the command. "
        "Keys are parameter names, values are their string representations.",
    )


class EntryPointConfig(BaseModel):
    """Defines the training and prediction entry points for an external model."""

    train: CommandConfig = Field(description="Configuration for the training command.")
    predict: CommandConfig = Field(description="Configuration for the prediction command.")


class UserOption(BaseModel):
    """Defines a user-configurable option for an external model."""

    name: str = Field(description="The programmatic name of the user option.")
    type: Literal["string", "integer", "float", "boolean"] = Field(description="The data type of the user option.")
    description: str = Field(description="A user-friendly description of what this option does.")
    default: Optional[str] = Field(
        default=None,
        description="The default value for this option, as a string. Must be convertible to the specified 'type'.",
    )


class ModelInfo(BaseModel):
    """Provides descriptive information about an external model."""

    author: str = Field(description="The author(s) of the model.")
    description: str = Field(description="A detailed description of the model, its purpose, and methodology.")


class ModelTemplateConfig(BaseModel):
    """
    Configuration schema for an external model template.

    This model defines all necessary parameters to describe and execute an external model,
    including its entry points, environment, required data, and user-configurable options.
    The `extra='forbid'` configuration ensures that no unspecified fields are allowed,
    promoting strict adherence to the defined schema.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Unique programmatic name for this model template.")
    entry_points: EntryPointConfig = Field(description="Defines the train and predict commands for the model.")
    docker_env: Optional[DockerEnvConfig] = Field(
        default=None, description="Optional Docker environment configuration if the model runs in Docker."
    )
    python_env: Optional[str] = Field(
        default=None,
        description="Optional path to a Python environment file (e.g., requirements.txt, environment.yml).",
    )
    required_fields: List[str] = Field(
        default_factory=lambda: ["rainfall", "mean_temperature"],
        description="List of feature names that are strictly required by the model. "
        "Default includes 'rainfall' and 'mean_temperature', adjust if model needs differ.",
    )
    allow_free_additional_continuous_covariates: bool = Field(
        default=False,
        description="If True, allows the model to accept additional continuous covariates beyond those in 'required_fields'.",
    )
    adapters: Optional[Dict[str, str]] = Field(
        default=None,
        description="Optional dictionary defining data adapters. Keys might represent internal CHAP-core feature names "
        "and values the names expected by the external model script, or vice-versa. "
        "Exact usage depends on the model runner implementation.",
    )
    user_options: List[UserOption] = Field(
        default_factory=list, description="List of user-configurable options for this model."
    )
    model_info: Optional[ModelInfo] = Field(
        default=None, description="Optional descriptive information about the model (author, description)."
    )
