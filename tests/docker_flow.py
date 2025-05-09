# Improvement Suggestions:
# 1. **Comprehensive Docstrings**: Add detailed docstrings for all functions (`dataset`, `main`, `ensure_up`, `evaluate_model`) to explain their purpose, arguments, return values, and any important side effects or assumptions. (Primary task of this refactoring).
# 2. **Configuration Management**: Make hardcoded values like the default `hostname` ('chap') and the path in `dataset()` configurable (e.g., via environment variables or command-line arguments) for greater test flexibility and portability.
# 3. **Refined Error Handling**: In `main()` and `evaluate_model()`, replace bare `except:` clauses with more specific exception catching (e.g., `requests.exceptions.RequestException`, `json.JSONDecodeError`). Improve error messages and logging for clarity.
# 4. **Task-Specific Status Polling**: The `evaluate_model` function polls a generic `/v1/status`. If the `/v1/predict/` endpoint initiates an asynchronous task and returns a task ID, polling should ideally be directed at a task-specific status endpoint using that ID. This makes the test more robust, especially in concurrent environments. The timeout mechanism should also be configurable.
# 5. **Clarity of Assertions**: Enhance assertions. For example, in `evaluate_model`, the assertion `assert len(results['dataValues']) == 45` should be accompanied by a comment or docstring note explaining why 45 dataValues are expected for the given test data and model.

"""
This is meant to be a standalone python file for testing the flow with docker compose.

This script tests an API flow, likely involving Docker Compose for service deployment.
It fetches a list of models, then iterates through them (or a subset) to
trigger an evaluation (prediction) via an API endpoint. It polls a status
endpoint to monitor completion and finally retrieves results.
"""

import json
import logging
import os  # For environment variable access
import sys  # Moved import sys to top
import time
from typing import Any, Dict  # For type hints
from urllib.parse import urljoin  # For better URL construction

import requests

# Configure logger for the script
logger = logging.getLogger(__name__)
if not logger.handlers:  # Avoid adding multiple handlers if script is re-imported or logger is already configured
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
else:
    logger.setLevel(logging.DEBUG)


# Determine hostname from environment variable or default to 'chap'
HOSTNAME = os.getenv("CHAP_TEST_HOSTNAME", "chap")  # Consistent with docker_db_flow.py
CHAP_BASE_URL = f"http://{HOSTNAME}:8000"
DEFAULT_TIMEOUT_SECONDS = 10  # Default timeout for requests


def dataset() -> Dict[str, Any]:
    """
    Loads a predefined dataset from a JSON file to be used in API requests.

    The dataset is sourced from '../example_data/anonymous_chap_request.json'.

    Returns:
        Dict[str, Any]: The dataset loaded as a Python dictionary.

    Raises:
        FileNotFoundError: If the dataset JSON file cannot be found.
        json.JSONDecodeError: If the file content is not valid JSON.
    """
    # Path should be relative to this script file or made configurable
    # Assuming this script is in 'tests/' and 'example_data' is a sibling to 'tests/' parent.
    # For robustness, construct path relative to this file's location.
    base_path = Path(__file__).parent.parent  # Goes up to project root if tests/ is child of project root
    dataset_file_path = base_path / "example_data" / "anonymous_chap_request.json"

    logger.info(f"Loading dataset from: {dataset_file_path}")
    try:
        with open(dataset_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        logger.error(f"Dataset file not found at {dataset_file_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {dataset_file_path}: {e}")
        raise


def ensure_up(base_url: str, status_endpoint: str = "/v1/status", retries: int = 5, delay: int = 5) -> None:
    """
    Ensures that the CHAP API service is up and responsive before proceeding.

    Args:
        base_url (str): The base URL of the CHAP API service.
        status_endpoint (str): Relative path to the status check endpoint.
        retries (int): Number of connection attempts.
        delay (int): Delay in seconds between retries.

    Raises:
        requests.exceptions.ConnectionError: If the service is not up after all retries.
    """
    status_url = urljoin(base_url, status_endpoint)
    logger.info(f"Ensuring service is up at {status_url} (retries={retries}, delay={delay}s)...")
    for i in range(retries):
        try:
            response = requests.get(status_url, timeout=DEFAULT_TIMEOUT_SECONDS)
            response.raise_for_status()  # Good practice to check for HTTP errors
            # Could add a check for response content if status endpoint returns specific "up" message
            logger.info(f"Service is up! Status: {response.status_code}")
            return
        except requests.exceptions.RequestException as e:
            logger.warning(f"Attempt {i+1}/{retries}: Failed to connect to {status_url} ({e}). Retrying in {delay}s...")
            time.sleep(delay)
    logger.error(f"Service at {status_url} is not responding after {retries} retries.")
    raise requests.exceptions.ConnectionError(f"Service at {status_url} not up after {retries} retries.")


def evaluate_model(
    base_url: str, data_payload: Dict[str, Any], model_info: Dict[str, Any], timeout_seconds: int = 600
) -> None:
    """
    Triggers a model evaluation (prediction) via the API and polls for its completion.

    Args:
        base_url (str): The base URL of the CHAP API service.
        data_payload (Dict[str, Any]): The dataset payload for the prediction request.
        model_info (Dict[str, Any]): A dictionary containing model information, must include 'name'.
        timeout_seconds (int): Maximum time in seconds to wait for the evaluation to complete.

    Raises:
        AssertionError: If API responses are not as expected (e.g., status codes, content).
        ValueError: If the model evaluation job fails based on status polling.
        TimeoutError: If the model evaluation does not complete within the specified timeout.
        requests.exceptions.RequestException: For network or HTTP errors during API calls.
    """
    model_name = model_info.get("name")
    if not model_name:
        logger.error("Model information missing 'name' field.")
        raise ValueError("Model information must include a 'name'.")

    logger.info(f"Evaluating model: {model_name}")
    ensure_up(base_url)  # Ensure service is up before each evaluation if desired, or once globally

    data_payload["estimator_id"] = model_name  # Set the estimator ID in the payload

    predict_url = urljoin(base_url, "/v1/predict/")
    status_url = urljoin(base_url, "/v1/status")  # Generic status, see suggestion #4
    exception_url = urljoin(base_url, "/v1/get-exception")
    results_url = urljoin(base_url, "/v1/get-results")

    logger.info(f"Posting to predict_url: {predict_url} for model {model_name}")
    response = requests.post(predict_url, json=data_payload, timeout=DEFAULT_TIMEOUT_SECONDS)
    response.raise_for_status()  # Will raise HTTPError for bad status (4xx or 5xx)

    # Assuming the initial POST to /v1/predict/ might return a task ID if it's async
    # For now, following original logic of polling global /v1/status
    # If a task_id is returned: task_id = response.json().get("task_id")
    # And status_url should become something like urljoin(base_url, f"/v1/status/{task_id}")

    logger.info(f"Initial response for {model_name}: {response.json()}")  # Log initial response
    # Original code asserted success on initial response, which might be for synchronous part
    assert (
        response.json().get("status") == "success"
    ), f"Initial prediction request for {model_name} did not return 'success' status. Response: {response.json()}"

    poll_interval = 5  # seconds
    max_polls = timeout_seconds // poll_interval
    logger.info(
        f"Polling status for {model_name} at {status_url} (max_polls={max_polls}, interval={poll_interval}s)..."
    )

    for i in range(max_polls):
        try:
            status_response = requests.get(status_url, timeout=DEFAULT_TIMEOUT_SECONDS)
            status_response.raise_for_status()
            job_status_data = status_response.json()
            logger.info(f"Poll {i+1}/{max_polls} for {model_name}: Status is {job_status_data}")

            if job_status_data.get("status") == "failed":
                logger.error(f"Model evaluation for {model_name} failed.")
                try:
                    exception_info = requests.get(exception_url, timeout=DEFAULT_TIMEOUT_SECONDS).json()
                    logger.error(f"Service exception info: {exception_info}")
                    if "Earth Engine client library not initialized" in str(exception_info):  # Check as string
                        raise Exception("Evaluation failed: Earth Engine client library not initialized.")
                    raise ValueError(
                        f"Model evaluation failed for {model_name}. Exception from service: {exception_info}"
                    )
                except requests.exceptions.RequestException as e_ex:
                    logger.error(f"Could not fetch exception info after failure: {e_ex}")
                    raise ValueError(f"Model evaluation failed for {model_name}, and could not fetch details.")

            if job_status_data.get(
                "ready"
            ):  # Assuming 'ready' means task completion (success or failure handled above)
                logger.info(f"Model {model_name} evaluation job indicates 'ready'. Fetching results.")
                break

            time.sleep(poll_interval)
        except requests.exceptions.RequestException as e_poll:
            logger.error(f"Error polling status for {model_name}: {e_poll}. Retrying...")
            time.sleep(poll_interval)  # Wait before retrying poll
    else:  # Loop completed without break (i.e., not ready)
        logger.error(f"Model evaluation for {model_name} timed out after {timeout_seconds} seconds.")
        raise TimeoutError(f"Model evaluation for {model_name} took too long.")

    logger.info(f"Fetching results for {model_name} from {results_url}")
    results_response = requests.get(results_url, timeout=DEFAULT_TIMEOUT_SECONDS)
    results_response.raise_for_status()  # Check for HTTP errors

    logger.info(f"Results status code for {model_name}: {results_response.status_code}")
    logger.debug(f"Results response content for {model_name}: {results_response.text[:500]}...")  # Log snippet

    results_data = results_response.json()
    # Example assertion: Check if 'dataValues' key exists and has the expected number of items.
    # The number 45 is specific to the example data and model behavior.
    assert "dataValues" in results_data, f"Results for {model_name} missing 'dataValues' key."
    assert (
        len(results_data["dataValues"]) == 45
    ), f"Expected 45 dataValues for {model_name}, got {len(results_data['dataValues'])}."
    logger.info(f"Model {model_name} evaluation successful with {len(results_data['dataValues'])} dataValues.")


def main() -> None:
    """
    Main function to run the Docker flow integration test.

    It fetches available models from the CHAP API and then triggers an
    evaluation for each model using a predefined dataset.
    """
    logger.info(f"Starting Docker flow test against CHAP API at {CHAP_BASE_URL}")
    ensure_up(CHAP_BASE_URL)

    models_url = urljoin(CHAP_BASE_URL, "/v1/list-models")
    try:
        logger.info(f"Fetching list of models from {models_url}...")
        models_response = requests.get(models_url, timeout=DEFAULT_TIMEOUT_SECONDS)
        models_response.raise_for_status()
        model_list = models_response.json()
        if not model_list:
            logger.warning("No models returned from API. Cannot proceed with evaluations.")
            return
        logger.info(f"Received {len(model_list)} models: {[m.get('name') for m in model_list]}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch models from {models_url}: {e}")
        try:
            exception_info_url = urljoin(CHAP_BASE_URL, "/v1/get-exception")
            err_response = requests.get(exception_info_url, timeout=DEFAULT_TIMEOUT_SECONDS)
            logger.error(f"Service exception info: {err_response.json()}")
        except Exception:  # pylint: disable=broad-except
            logger.error("Could not retrieve additional exception info from service.")
        return  # Cannot proceed without model list
    except json.JSONDecodeError:
        logger.error(f"Failed to decode JSON model list from {models_url}. Response: {models_response.text}")
        return

    test_dataset = dataset()  # Load the dataset once

    for model_info in model_list:
        if isinstance(model_info, dict) and "name" in model_info:
            try:
                evaluate_model(CHAP_BASE_URL, test_dataset.copy(), model_info)  # Pass a copy of dataset
            except Exception as e:  # Catch errors from evaluate_model to continue with other models
                logger.error(f"Evaluation failed for model '{model_info.get('name', 'Unknown')}': {e}", exc_info=True)
        else:
            logger.warning(f"Skipping invalid model entry in list: {model_info}")

    logger.info("Docker flow test script finished.")


if __name__ == "__main__":
    # Allow overriding hostname via command line argument for standalone execution
    # This part is specific to __main__ execution and doesn't affect pytest fixtures.
    if len(sys.argv) > 1:
        HOSTNAME_OVERRIDE = sys.argv[1]
        CHAP_BASE_URL = f"http://{HOSTNAME_OVERRIDE}:8000"
        logger.info(f"Overriding CHAP_BASE_URL from command line: {CHAP_BASE_URL}")

    main()

# Path needs to be imported if used directly, e.g. for dataset path construction
from pathlib import Path
