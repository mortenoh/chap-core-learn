# Improvement Suggestions:
# 1. **Comprehensive Docstrings**: Add a module-level docstring and a detailed docstring for `generate_app`. Critically review and correct the docstrings of the inner `train` and `predict` API endpoint functions to accurately reflect their intended REST API behavior, parameters (especially how data is received), and responses. (Primary task).
# 2. **Fix `train` Endpoint Logic**: The `train` endpoint implementation has critical issues: it defines `training_data: List[model]` as an input parameter (presumably from the request body) but then attempts to load data from a hardcoded file path and calls `DataSet.df_from_pydantic_observations()` without using the input `training_data`. This must be refactored to process the actual input data.
# 3. **Re-evaluate `predict` Endpoint Data Input**: The `predict` endpoint currently expects file paths as parameters. For a typical REST API, data is usually sent in the request body or referenced by IDs if managed by the API. Clarify if this file-path-based approach is intentional for a specific deployment context or if it should be adapted for standard REST practices.
# 4. **Configuration and State Management**: The use of `working_dir` and hardcoded filenames (`training_data.csv`, `model`) suggests a single-model or stateful instance. For a more general REST API, consider how multiple models, datasets, and concurrent requests would be managed. Paths and configurations should be more robustly handled.
# 5. **API Error Handling and Responses**: Implement proper error handling within the FastAPI endpoint functions. Catch exceptions from data loading, model operations, etc., and return appropriate HTTP error responses (e.g., using `fastapi.HTTPException`) with informative messages instead of letting exceptions propagate directly.

"""
This module provides functionality to dynamically generate a FastAPI application
for serving CHAP-core models via a REST API.

The `generate_app` function creates FastAPI endpoints (e.g., for training and prediction)
based on a provided model estimator. This allows for exposing model functionalities
over HTTP, potentially for integration with web applications or other services.
The current implementation has some inconsistencies in how data is handled by the
generated endpoints, particularly for the 'train' endpoint.
"""

import logging
from pathlib import Path  # Added Path
from typing import Any, List, Type  # Added Any, Type

import pandas as pd  # For error catching
import pydantic
from fastapi import FastAPI, HTTPException  # Added HTTPException
from starlette.middleware.cors import CORSMiddleware

from chap_core.datatypes import remove_field  # Removed create_tsdataclass as it's not used
from chap_core.model_spec import get_dataclass
from chap_core.models.model_template_interface import ModelPredictor  # For type hint
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet

logger = logging.getLogger(__name__)


# TODO: Similar to command_line_interface.py, clarify if 'estimator' should be an instance or class.
# Assuming 'estimator_class' for now.
def generate_app(estimator_class: Type[Any], working_dir: str) -> FastAPI:
    """
    Generates a FastAPI application with 'train' and 'predict' endpoints
    for a given model estimator class.

    The FastAPI app is configured with CORS middleware to allow cross-origin requests.
    It dynamically creates a Pydantic model for training data based on the
    estimator's input dataclass.

    Args:
        estimator_class (Type[Any]): The model estimator class. Expected to have `train`
                                     and `load_predictor` methods.
        working_dir (str): A path to a working directory where models and potentially
                           temporary data might be stored. Its usage needs clarification.

    Returns:
        FastAPI: A FastAPI application instance with generated endpoints.

    Raises:
        ValueError: If the input data structure (dataclass) cannot be inferred for the estimator.
    """
    app = FastAPI(
        title=f"CHAP-core Model API for {estimator_class.__name__}",
        description=f"Dynamically generated API for training and prediction with {estimator_class.__name__}.",
        version="0.1.0",  # Example version
    )

    # Configure CORS
    origins = [
        "*",  # Allow all origins - consider restricting for production
        "http://localhost:3000",  # Example frontend
        "localhost:3000",
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],  # Allows all methods
        allow_headers=["*"],  # Allows all headers
    )

    dc = get_dataclass(estimator_class)
    if dc is None:
        msg = f"Input data structure (dataclass) could not be inferred for estimator {estimator_class.__name__}."
        logger.error(msg)
        raise ValueError(msg)

    # Dynamically create a Pydantic model for the training data based on the dataclass's annotations
    # This model will be used for request body validation in the /train endpoint.
    pydantic_training_model = pydantic.create_model(f"{dc.__name__}InputModel", **dc.__annotations__)

    # These paths seem to imply a single, fixed location for data/models related to this app instance.
    # This might not be suitable for a multi-user or multi-model API without further namespacing.
    training_data_filename = Path(working_dir) / "training_data.csv"  # Converted to Path
    model_path = Path(working_dir) / "model"  # Converted to Path

    # NOTE: The @app.command() decorator is from cyclopts, not FastAPI.
    # For FastAPI, endpoints are defined with @app.post(), @app.get(), etc.
    # The original code seems to mix cyclopts CLI generation with FastAPI.
    # Assuming the intent was to create FastAPI routes.

    # @app.command() # This is for cyclopts, should be @app.post("/train/") for FastAPI
    @app.post("/train/")  # Corrected decorator for FastAPI
    async def train(training_data_payload: List[pydantic_training_model]):  # Parameter name changed for clarity
        """
        Train the model using historic data provided in the request body.

        (Original docstring was for a CLI, updated for API context)
        This endpoint expects a list of data records conforming to the dynamically
        generated Pydantic model based on the estimator's input dataclass.
        The trained model (predictor) is saved to a path derived from `working_dir`.

        Args:
            training_data_payload (List[PydanticModel]): A list of data records for training.
                                                        The PydanticModel is dynamically created.

        Returns:
            dict: A message indicating training status.

        Raises:
            HTTPException: If training fails or input data is problematic.
        """
        logger.info(f"API: Training model {estimator_class.__name__} with {len(training_data_payload)} records.")

        # CRITICAL ISSUE: The original code below does not use `training_data_payload`.
        # It attempts to load from `training_data_filename` and uses
        # `DataSet.df_from_pydantic_observations()` without arguments.
        # This needs to be corrected to process `training_data_payload`.

        # Corrected (conceptual) logic:
        if not training_data_payload:
            raise HTTPException(status_code=400, detail="Training data payload cannot be empty.")

        try:
            # Convert Pydantic models to a format suitable for DataSet.from_pandas or similar
            # This step depends heavily on how DataSet expects its input from Pydantic models.
            # Assuming DataSet.df_from_pydantic_observations can take a list of Pydantic models.
            # Or, convert to list of dicts:
            data_as_dicts = [item.model_dump() for item in training_data_payload]

            # Further processing might be needed to structure this into a DataFrame that
            # DataSet.from_pandas can use with the inferred dataclass `dc`.
            # This part is highly dependent on the structure of `dc` and `DataSet` internals.
            # For demonstration, assuming a direct way or placeholder:
            # dataset = DataSet.from_pydantic_list(training_data_payload, dc) # Hypothetical method

            # Placeholder: If `dc` represents rows, a DataFrame could be made:
            df = pd.DataFrame(data_as_dicts)
            # Ensure DataFrame columns match fields in `dc` and `time_period` is handled.
            # This is a complex step. For now, we'll assume `DataSet.from_pandas` can handle it.
            # A more robust solution would be a dedicated method in DataSet or an adaptor.
            if "time_period" not in df.columns and all(hasattr(item, "time_period") for item in training_data_payload):
                # This is a simplification; time_period needs careful handling for DataSet
                df["time_period"] = [str(item.time_period) for item in training_data_payload]

            dataset = DataSet.from_pandas(df, dc)  # This assumes from_pandas can map df to dc structure

            estimator_instance = estimator_class()
            predictor = estimator_instance.train(dataset)

            model_path.mkdir(parents=True, exist_ok=True)  # Ensure model directory exists
            predictor.save(str(model_path))
            logger.info(f"Training complete. Model saved to '{model_path}'.")
            return {"message": f"Model {estimator_class.__name__} trained successfully.", "model_path": str(model_path)}
        except ValueError as e:  # Catch specific errors from data processing
            logger.error(f"ValueError during training data processing: {e}", exc_info=True)
            raise HTTPException(status_code=400, detail=f"Invalid training data: {e}")
        except Exception as e:
            logger.error(f"An error occurred during API training: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Internal server error during training: {e}")

    # @app.command() # This is for cyclopts, should be @app.post("/predict/") for FastAPI
    @app.post("/predict/")  # Corrected decorator for FastAPI
    async def predict(
        model_filename: str, historic_data_filename: str, future_data_filename: str, output_filename: str
    ):
        """
        Predict using a trained model. (Original docstring seems for CLI)

        This API endpoint currently expects file paths for model and data.
        For a typical REST API, data would be part of the request body or
        referenced by IDs if managed by the API. This implementation might
        be for specific local/batch processing scenarios via API.

        Parameters (as per current implementation, not typical REST):
        ----------
        model_filename: str
            The path (relative to `working_dir` or absolute if handled by server) to the model file.
        historic_data_filename: str
            Path to the historic data CSV file.
        future_data_filename: str
            Path to the future data CSV file.
        output_filename: str
            Path where the output CSV should be saved by the server.

        Returns:
            dict: Path to the output file or prediction results.

        Raises:
            HTTPException: If prediction fails.
        """
        # These paths are relative to where the FastAPI app is running or need careful handling.
        # For security and robustness, direct file path parameters in APIs are generally discouraged.
        # Consider using IDs to refer to server-managed resources or upload/download data.

        # Construct full paths based on working_dir for safety if these are relative names
        full_model_path = Path(working_dir) / model_filename
        full_historic_path = Path(working_dir) / historic_data_filename
        full_future_path = Path(working_dir) / future_data_filename
        full_output_path = Path(working_dir) / output_filename

        logger.info(f"API: Predicting with model from '{full_model_path}'...")
        try:
            estimator_instance = estimator_class()
            predictor: ModelPredictor = estimator_instance.load_predictor(str(full_model_path))

            logger.info(f"Loading historic data from '{full_historic_path}'.")
            dataset = DataSet.from_csv(str(full_historic_path), dc)

            if "disease_cases" in [f.name for f in dataclasses.fields(dc)]:  # Added dataclasses import
                future_dc = remove_field(dc, "disease_cases")
            else:
                future_dc = dc
                logger.warning(
                    "Target 'disease_cases' not found in inferred dataclass for future data. Using full dataclass."
                )

            logger.info(f"Loading future data from '{full_future_path}' as type '{future_dc.__name__}'.")
            future_data = DataSet.from_csv(str(full_future_path), future_dc)

            forecasts = predictor.predict(dataset, future_data)

            full_output_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure output dir exists
            forecasts.to_csv(str(full_output_path))
            logger.info(f"Predictions saved to '{full_output_path}'.")
            # API should ideally return the data, or a URL to get it, not just save to server disk.
            return {"message": "Prediction successful.", "output_file": str(full_output_path)}
        except FileNotFoundError as e:
            logger.error(f"File not found during API prediction: {e.filename or e}")
            raise HTTPException(status_code=404, detail=f"Required file not found: {e.filename or e}")
        except (pd.errors.ParserError, ValueError) as e:
            logger.error(f"Error loading or parsing data during API prediction: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid data format: {e}")
        except Exception as e:
            logger.error(f"An error occurred during API prediction: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Internal server error during prediction: {e}")

    return app


# Need to import dataclasses for dataclasses.fields(dc)
import dataclasses
