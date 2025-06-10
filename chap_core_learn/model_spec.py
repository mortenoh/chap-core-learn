# Improvement Suggestions:
# 1. **Comprehensive Docstrings**: Add detailed docstrings for all classes (including Pydantic models and Enums like `PeriodType`, `ParameterSpec`, `ModelSpec`), methods, and functions to clarify their purpose, attributes, parameters, and return values. (This is the primary task of this refactoring).
# 2. **`ParameterSpec` Integration**: Clarify how `ParameterSpec` and its subclasses (e.g., `EwarsParamSpec`) are intended to be used with `ModelSpec.parameters`. Currently, `parameters` is a generic `dict` and initialized as empty. If specific parameter structures are expected, `ModelSpec.parameters` should be typed more specifically (e.g., `Union[ParameterSpec, Dict[str, Any]]`) and the creation functions updated.
# 3. **Error Handling in `model_spec_from_yaml`**: Implement robust error handling for file operations (e.g., `FileNotFoundError`), YAML parsing (`yaml.YAMLError`), and missing or malformed keys in the YAML data to prevent unexpected crashes.
# 4. **Clarity on Feature Exclusion**: Document the rationale behind the `_non_feature_names` set within the docstrings of functions that use it (e.g., `_get_feature_names`, `model_spec_from_yaml`), explaining why these names are typically excluded as model features.
# 5. **Robustness of `get_dataclass`**: The `get_dataclass` function relies on specific introspection patterns (`inspect.get_annotations`, `__args__[0]`). This can be brittle. Add error handling (e.g., for `IndexError`, `AttributeError`, `TypeError` if annotations are not as expected) and document the assumptions about the `model_class.train` method's signature.

"""
This module defines specifications for predictive models within the CHAP-core framework.

It includes Pydantic models for defining model parameters (`ParameterSpec`, `EwarsParamSpec`),
feature sets, and overall model metadata (`ModelSpec`). It also provides utility functions
to construct `ModelSpec` instances from YAML configuration files or directly from
model class definitions through introspection. The `PeriodType` enum categorizes
models by their operational time period (e.g., weekly, monthly).
"""

import dataclasses
import inspect
from enum import Enum
from typing import Any, Dict, List, Optional, Type  # Added List, Dict, Any, Type, Optional

import yaml
from pydantic import BaseModel, Field, PositiveInt  # Added Field

import chap_core.predictor.feature_spec as fs
from chap_core.datatypes import TimeSeriesData
from chap_core.exceptions import ModelConfigurationException  # For custom errors

# Set of names typically found in TimeSeriesData that are not considered model features.
_non_feature_names = {
    "disease_cases",  # Usually the target variable
    "week",  # Time-related, often used for feature engineering
    "month",  # Time-related
    "location",  # Identifier, not a feature
    "time_period",  # Core time information
    "year",  # Time-related
}


class PeriodType(Enum):
    """
    Enumeration for the typical operational time period of a model.
    """

    week = "week"
    month = "month"
    any = "any"  # Model is agnostic to period or handles various types
    year = "year"


class ParameterSpec(BaseModel):
    """
    Base Pydantic model for defining model-specific parameters.
    Subclasses should define their own fields for specific parameters.
    This class itself acts as a placeholder or for models with no typed parameters.
    """

    # Example: param_name: ParamType = Field(default=..., description="...")
    pass


class EwarsParamSpec(ParameterSpec):
    """
    Parameter specification for EWAS-style (Early Warning and Response System) models.
    """

    n_weeks: PositiveInt = Field(..., description="Number of weeks for a rolling window or similar parameter.")
    alpha: float = Field(..., description="A smoothing factor or learning rate, typically between 0 and 1.")


# Represents an empty parameter specification, used as a default.
EmptyParameterSpec: Dict[str, Any] = {}


class ModelSpec(BaseModel):
    """
    Pydantic model defining the specification of a predictive model.

    Attributes:
        name (str): The unique name or identifier for the model.
        parameters (dict): A dictionary of parameters for the model. Ideally, this would
                           hold an instance of a `ParameterSpec` subclass, but is currently
                           a generic dict. See improvement suggestion #2.
        features (list[fs.Feature]): A list of feature specifications (`chap_core.predictor.feature_spec.Feature`)
                                     that the model expects as input.
        period (PeriodType): The typical operational time period for the model (e.g., weekly, monthly).
                             Defaults to `PeriodType.any`.
        description (str): A human-readable description of the model.
        author (str): The author or maintainer of the model.
        targets (str): The name of the target variable the model predicts (e.g., "disease_cases").
                       Currently a single string; consider List[str] for multi-target models.
    """

    name: str = Field(..., description="Unique name or identifier for the model.")
    parameters: Dict[str, Any] = Field(
        default_factory=dict, description="Model-specific parameters."
    )  # Or ParameterSpec
    features: List[fs.Feature] = Field(..., description="List of feature specifications required by the model.")
    period: PeriodType = Field(PeriodType.any, description="Typical operational time period of the model.")
    description: str = Field("No Description yet", description="Human-readable description of the model.")
    author: str = Field("Unknown Author", description="Author or maintainer of the model.")
    targets: str = Field("disease_cases", description="Name of the target variable(s) the model predicts.")


def model_spec_from_yaml(filename: str) -> ModelSpec:
    """
    Loads a model specification from a YAML file.

    The YAML file is expected to contain keys like 'name', 'adapters' (for features),
    'period', 'description', and 'author'.

    Args:
        filename (str): Path to the YAML configuration file.

    Returns:
        ModelSpec: An instance of ModelSpec populated from the YAML data.

    Raises:
        FileNotFoundError: If the YAML file does not exist.
        yaml.YAMLError: If the YAML file is malformed.
        ModelConfigurationException: If required keys are missing or data is invalid.
    """
    try:
        with open(filename, "r", encoding="utf-8") as file:
            data = yaml.safe_load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Model specification YAML file not found: {filename}")
    except yaml.YAMLError as e:
        raise ModelConfigurationException(f"Error parsing YAML file {filename}: {e}")

    if not isinstance(data, dict):
        raise ModelConfigurationException(f"Invalid YAML format in {filename}: expected a dictionary at the root.")

    try:
        name = data["name"]
    except KeyError:
        raise ModelConfigurationException(f"Missing required 'name' field in model specification YAML: {filename}")

    # TODO: Implement proper loading of parameters based on model type or a 'parameters' section in YAML.
    # Currently, parameters are set to EmptyParameterSpec.
    parameters = EmptyParameterSpec

    adapters = data.get("adapters", {})
    if not isinstance(adapters, dict):
        raise ModelConfigurationException(f"Invalid 'adapters' format in {filename}: expected a dictionary.")

    features = []
    for feature_key, feature_name in adapters.items():
        if feature_name not in _non_feature_names:
            if feature_name in fs.feature_dict:
                features.append(fs.feature_dict[feature_name])
            else:
                logger.warning(
                    f"Feature '{feature_name}' (from adapter '{feature_key}') in {filename} not found in feature_dict. Skipping."
                )

    period_str = data.get("period", "any")
    try:
        period = PeriodType[period_str]
    except KeyError:
        logger.warning(
            f"Invalid period type '{period_str}' in {filename}. Defaulting to 'any'. Valid types: {[pt.name for pt in PeriodType]}"
        )
        period = PeriodType.any

    description = data.get("description", "No Description yet")
    author = data.get("author", "Unknown Author")
    targets = data.get("targets", "disease_cases")  # Allow specifying targets from YAML

    return ModelSpec(
        name=name,
        parameters=parameters,
        features=features,
        period=period,
        description=description,
        author=author,
        targets=targets,
    )


def model_spec_from_model(model_class: Type[Any]) -> ModelSpec:
    """
    Generates a ModelSpec by introspecting a given model class.

    It extracts the model name, attempts to infer feature names from the
    type hints of its `train` method's first argument (expected to be a TimeSeriesData subclass),
    and sets default values for other specification fields.

    Args:
        model_class (Type[Any]): The model class to introspect. Expected to have a `train` method.

    Returns:
        ModelSpec: An instance of ModelSpec derived from the model class.
    """
    name = model_class.__name__
    feature_names = _get_feature_names(model_class)

    features = []
    if feature_names:
        for feature_name in feature_names:
            if feature_name in fs.feature_dict:
                features.append(fs.feature_dict[feature_name])
            else:
                logger.warning(
                    f"Feature '{feature_name}' inferred from model {name} not found in feature_dict. Skipping."
                )

    # TODO: Introspect parameters if possible, or define a convention for models to declare them.
    parameters = EmptyParameterSpec

    # TODO: Infer period if possible (e.g., from expected data types or model attributes).
    period = PeriodType.any

    description = inspect.getdoc(model_class) or "Internally defined model"  # Use class docstring if available
    author = getattr(model_class, "__author__", "CHAP Team")  # Check for __author__ attribute

    return ModelSpec(
        name=name,
        parameters=parameters,
        features=features,
        period=period,
        description=description,
        author=author,
        # targets could also be introspected if a convention exists
    )


def _get_feature_names(model_class: Type[Any]) -> List[str]:
    """
    Infers feature names from a model class's `train` method signature.

    It expects the first argument of the `train` method to be type-hinted as a
    generic type (e.g., `DataSet[SomeTimeSeriesData]`) where `SomeTimeSeriesData`
    is a dataclass derived from `TimeSeriesData`. Fields of this dataclass,
    excluding those in `_non_feature_names`, are considered feature names.

    Args:
        model_class (Type[Any]): The model class to introspect.

    Returns:
        List[str]: A list of inferred feature names. Returns an empty list if
                   features cannot be inferred or if the `train` method signature
                   does not match expectations.
    """
    try:
        target_dataclass = get_dataclass(model_class)
        if target_dataclass is None or not dataclasses.is_dataclass(target_dataclass):
            logger.debug(
                f"Could not determine a valid dataclass for feature extraction from model: {model_class.__name__}"
            )
            return []

        # These are names of fields in the TimeSeriesData dataclass used for training
        # (e.g., rainfall, temperature from ClimateData)
        feature_names = [
            field.name for field in dataclasses.fields(target_dataclass) if field.name not in _non_feature_names
        ]
        return feature_names
    except Exception as e:  # pylint: disable=broad-except
        logger.warning(
            f"Error introspecting feature names for model {model_class.__name__}: {e}. Returning empty feature list."
        )
        return []


def get_dataclass(model_class: Type[Any]) -> Optional[Type[TimeSeriesData]]:
    """
    Attempts to extract the specific TimeSeriesData-derived dataclass used as
    input to a model's `train` method.

    It inspects the type annotation of the first parameter of `model_class.train()`.
    It expects this annotation to be a generic type like `SomeContainer[ActualDataClass]`,
    where `ActualDataClass` is what this function aims to return.

    Args:
        model_class (Type[Any]): The model class.

    Returns:
        Optional[Type[TimeSeriesData]]: The extracted dataclass type if successful,
                                        None otherwise.
    """
    if not hasattr(model_class, "train") or not callable(model_class.train):
        logger.debug(f"Model class {model_class.__name__} has no callable 'train' method.")
        return None

    try:
        train_annotations = inspect.get_annotations(model_class.train, eval_str=True)
        # Get the annotation of the first parameter (excluding 'self' if it's a method)
        param_names = list(inspect.signature(model_class.train).parameters.keys())
        if not param_names:
            logger.debug(f"'train' method of {model_class.__name__} has no parameters.")
            return None

        # Skip 'self' or 'cls' for instance/class methods
        first_param_name = param_names[0]
        if first_param_name in ("self", "cls") and len(param_names) > 1:
            first_param_name = param_names[1]
        elif first_param_name in ("self", "cls"):  # Only self/cls param
            logger.debug(f"'train' method of {model_class.__name__} only has '{first_param_name}'.")
            return None

        param_type = train_annotations.get(first_param_name)

        if param_type is None:
            logger.debug(
                f"No type annotation found for the first data parameter '{first_param_name}' of {model_class.__name__}.train."
            )
            return None

        # Expecting something like DataSet[MySpecificTimeSeriesData]
        if hasattr(param_type, "__origin__") and hasattr(param_type, "__args__") and param_type.__args__:
            # This gets the type argument of the generic, e.g., MySpecificTimeSeriesData
            inner_type = param_type.__args__[0]
            if isinstance(inner_type, type) and issubclass(inner_type, TimeSeriesData):
                return inner_type
            else:
                logger.debug(
                    f"First parameter's inner type '{inner_type}' is not a subclass of TimeSeriesData for {model_class.__name__}."
                )
                return None
        # Handle cases where the annotation is directly the TimeSeriesData subclass (not generic)
        elif isinstance(param_type, type) and issubclass(param_type, TimeSeriesData):
            return param_type
        else:
            logger.debug(
                f"Annotation '{param_type}' for first data parameter of {model_class.__name__}.train is not in the expected format (e.g., Generic[TimeSeriesDataSubclass] or TimeSeriesDataSubclass)."
            )
            return None

    except Exception as e:  # pylint: disable=broad-except
        logger.warning(f"Error introspecting dataclass for model {model_class.__name__}: {e}")
        return None


# Add logger if not already present at module level
import logging

logger = logging.getLogger(__name__)
