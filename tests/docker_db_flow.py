# Improvement Suggestions:
# 1. **Comprehensive Docstrings**: Add detailed docstrings for the `IntegrationTest` class, all its methods, and helper functions (`make_prediction_request`, `make_dataset_request`) to explain their purpose, arguments, return values, and any important side effects or assumptions. (Primary task of this refactoring).
# 2. **Configuration Management**: Centralize and make configurable hardcoded values like API paths (e.g., `/v1/health`), default hostname (`'chap'`), and file paths (`'../example_data/...'`). Use environment variables, a config file, or command-line arguments for better test environment management.
# 3. **Refined Exception Handling**: In `_get` and `_post` methods, replace bare `except:` clauses with more specific exception catching (e.g., `requests.exceptions.RequestException`). Consolidate and improve error logging to avoid redundancy and provide clearer diagnostics.
# 4. **Enhanced Test Assertions**: While basic assertions exist (e.g., for status codes), expand them to verify specific expected values in API responses where critical for confirming correctness (e.g., asserting properties of a prediction result or evaluation entry).
# 5. **Configurable Timeouts and Retries**: Make polling parameters in `ensure_up` and `wait_for_db_id` (like number of retries, sleep intervals, total timeout) configurable to adapt to different test environments or task execution times.

"""
This script is designed for standalone integration testing of the CHAP API,
particularly focusing on flows that involve Docker Compose, database interactions,
and potentially asynchronous task processing (e.g., via Celery).

It defines an `IntegrationTest` class that encapsulates various test scenarios,
such as making predictions and running model evaluations against a running
CHAP API instance. The script can be run directly and expects the CHAP service
to be accessible at a configurable URL.
"""

import json
import logging
import os  # For environment variable access
import time
from typing import Any, Dict, List, Tuple  # For type hints
from urllib.parse import urljoin  # For better URL construction

import requests

# Configure logger for the script
logger = logging.getLogger(__name__)
# Ensure basicConfig is called only if no handlers are configured for the root logger
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
else:
    logger.setLevel(logging.INFO)  # Set level for this specific logger if root is already configured


# --- Helper functions to load request data ---


def make_prediction_request(model_name: str) -> Dict[str, Any]:
    """
    Loads a template prediction request from a JSON file and sets the model ID.

    Args:
        model_name (str): The name of the model to be used in the prediction request.

    Returns:
        Dict[str, Any]: The prediction request data as a dictionary.

    Raises:
        FileNotFoundError: If the template JSON file is not found.
        json.JSONDecodeError: If the template file contains invalid JSON.
    """
    # Consider making this path configurable or relative to a known base path
    filename = "../example_data/anonymous_make_prediction_request.json"
    logger.debug(f"Loading prediction request template from: {filename}")
    try:
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)
        data["modelId"] = model_name
        logger.debug(f"Prediction request data prepared for model: {model_name}. Keys: {list(data.keys())}")
        return data
    except FileNotFoundError:
        logger.error(f"Prediction request template file not found: {filename}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {filename}: {e}")
        raise


def make_dataset_request() -> Dict[str, Any]:
    """
    Loads a template dataset creation request from a JSON file.

    Returns:
        Dict[str, Any]: The dataset creation request data as a dictionary.

    Raises:
        FileNotFoundError: If the template JSON file is not found.
        json.JSONDecodeError: If the template file contains invalid JSON.
    """
    # Consider making this path configurable
    filename = "../example_data/anonymous_make_dataset_request.json"
    logger.debug(f"Loading dataset request template from: {filename}")
    try:
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        logger.error(f"Dataset request template file not found: {filename}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {filename}: {e}")
        raise


class IntegrationTest:
    """
    A class encapsulating integration tests for the CHAP API.

    It provides methods to interact with various API endpoints, manage test flows
    for predictions and evaluations, and wait for asynchronous job completion.
    """

    DEFAULT_TIMEOUT = 10  # seconds for requests

    def __init__(self, chap_base_url: str, run_all_models: bool):
        """
        Initializes the IntegrationTest suite.

        Args:
            chap_base_url (str): The base URL of the CHAP API service to test against.
            run_all_models (bool): A flag indicating whether to run tests for all available
                                   models or a predefined subset (e.g., 'naive_model').
        """
        self._chap_base_url: str = chap_base_url.rstrip("/")
        self._run_all_models: bool = run_all_models
        logger.info(
            f"IntegrationTest initialized for URL: {self._chap_base_url}, Run all models: {self._run_all_models}"
        )

    def _build_url(self, path: str) -> str:
        """Helper to build full URLs."""
        return urljoin(self._chap_base_url, path)

    def ensure_up(self, retries: int = 20, delay: int = 5) -> None:
        """
        Checks if the CHAP API service is responsive at its health endpoint.

        Args:
            retries (int): Number of times to attempt connection.
            delay (int): Delay in seconds between retries.

        Raises:
            AssertionError: If the service is not up or health check fails after all retries.
        """
        health_url = self._build_url("/v1/health")
        logger.info(f"Ensuring service is up at {health_url} (retries={retries}, delay={delay}s)...")
        response = None
        for i in range(retries):
            try:
                response = requests.get(health_url, timeout=self.DEFAULT_TIMEOUT)
                response.raise_for_status()  # Raises HTTPError for 4xx/5xx status
                if response.json().get("status") == "success":
                    logger.info(f"Service is up and healthy! Status: {response.json()}")
                    return
                else:
                    logger.warning(f"Attempt {i+1}/{retries}: Health check status was not 'success': {response.json()}")
            except requests.exceptions.RequestException as e:
                logger.warning(
                    f"Attempt {i+1}/{retries}: Failed to connect to {health_url} ({e}). Retrying in {delay}s..."
                )
            except json.JSONDecodeError:
                logger.warning(
                    f"Attempt {i+1}/{retries}: Health check response from {health_url} was not valid JSON. Response: {response.text if response else 'No response'}"
                )

            time.sleep(delay)

        # Final assertion after all retries
        if response is not None:
            assert (
                response.status_code == 200
            ), f"Final health check failed with status {response.status_code}: {response.text}"
            assert (
                response.json().get("status") == "success"
            ), f"Final health check status was not 'success': {response.json()}"
        else:
            raise AssertionError(f"Service at {health_url} did not respond after {retries} retries.")

    def _get(self, url_path: str) -> Dict[str, Any]:
        """
        Performs a GET request to a specified API path and returns the JSON response.

        Args:
            url_path (str): The relative path of the API endpoint.

        Returns:
            Dict[str, Any]: The JSON response as a dictionary.

        Raises:
            requests.exceptions.RequestException: If the request fails (e.g., connection error, timeout).
            AssertionError: If the response status code is not 200.
            json.JSONDecodeError: If the response is not valid JSON.
        """
        full_url = self._build_url(url_path)
        logger.debug(f"GET request to: {full_url}")
        try:
            response = requests.get(full_url, timeout=self.DEFAULT_TIMEOUT)
            response.raise_for_status()  # Check for HTTP errors
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"GET request to {full_url} failed: {e}")
            # Attempt to get more specific exception info from the service if available
            try:
                exception_info_url = self._build_url("/v1/get-exception")
                err_response = requests.get(exception_info_url, timeout=self.DEFAULT_TIMEOUT)
                logger.error(f"Service exception info: {err_response.json()}")
            except Exception:  # pylint: disable=broad-except
                logger.error("Could not retrieve additional exception info from service.")
            raise
        except json.JSONDecodeError:
            logger.error(
                f"Failed to decode JSON response from {full_url}. Status: {response.status_code}, Text: {response.text}"
            )
            raise

    def _post(self, url_path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Performs a POST request with a JSON payload to a specified API path.

        Args:
            url_path (str): The relative path of the API endpoint.
            payload (Dict[str, Any]): The JSON payload to send with the request.

        Returns:
            Dict[str, Any]: The JSON response as a dictionary.

        Raises:
            requests.exceptions.RequestException: If the request fails.
            AssertionError: If the response status code is not 200.
            json.JSONDecodeError: If the response is not valid JSON.
        """
        full_url = self._build_url(url_path)
        logger.debug(
            f"POST request to: {full_url} with payload: {json.dumps(payload)[:200]}..."
        )  # Log truncated payload
        try:
            response = requests.post(full_url, json=payload, timeout=self.DEFAULT_TIMEOUT)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"POST request to {full_url} failed: {e}")
            raise
        except json.JSONDecodeError:
            logger.error(
                f"Failed to decode JSON response from {full_url}. Status: {response.status_code}, Text: {response.text}"
            )
            raise

    def get_models(self) -> List[Dict[str, Any]]:
        """Retrieves the list of available models from the API."""
        return self._get("/v1/crud/models")

    def make_prediction(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Initiates a prediction task and waits for its completion, then fetches the result.

        Args:
            data (Dict[str, Any]): The prediction request payload.

        Returns:
            Dict[str, Any]: The prediction result.
        """
        logger.info(f"Making prediction for model: {data.get('modelId', 'Unknown')}")
        response = self._post("/v1/analytics/make-prediction", payload=data)
        job_id = response["id"]
        logger.info(f"Prediction job initiated. Job ID: {job_id}")
        db_id = self.wait_for_db_id(job_id)
        logger.info(f"Job {job_id} completed. Resulting DB ID: {db_id}")
        prediction_result = self._get(f"/v1/crud/predictions/{db_id}")
        assert prediction_result["modelId"] == data["modelId"]
        return prediction_result

    def prediction_flow(self) -> None:
        """
        Executes the full prediction flow: ensures service is up, gets models,
        and makes predictions (either for all models or just 'naive_model').
        """
        logger.info("Starting prediction flow test...")
        self.ensure_up()
        model_list = self.get_models()
        logger.info(f"Available models: {[model['name'] for model in model_list]}")
        assert "naive_model" in {model["name"] for model in model_list}, "naive_model not found in model list."

        if self._run_all_models:
            logger.info("Running prediction for all available models.")
            for model in model_list:
                logger.info(f"--- Testing model: {model['name']} ---")
                self.make_prediction(make_prediction_request(model["name"]))
        else:
            logger.info("Running prediction for 'naive_model' only.")
            self.make_prediction(make_prediction_request("naive_model"))
        logger.info("Prediction flow test completed.")

    def evaluation_flow(self) -> None:
        """
        Executes the model evaluation flow: ensures service is up, creates a dataset,
        runs a backtest evaluation, and checks results.
        """
        logger.info("Starting evaluation flow test...")
        self.ensure_up()
        model_list = self.get_models()  # Ensure models are available
        model_name = "naive_model"
        assert model_name in {model["name"] for model in model_list}, f"{model_name} not found for evaluation."

        dataset_payload = make_dataset_request()
        logger.info("Creating dataset for evaluation...")
        dataset_id = self.make_dataset(dataset_payload)
        logger.info(f"Dataset created with ID: {dataset_id}")

        logger.info(f"Evaluating model '{model_name}' on dataset '{dataset_id}'...")
        evaluation_entries, backtest_id = self.evaluate_model(dataset_id, model_name)
        logger.info(f"Evaluation completed. Backtest ID: {backtest_id}")

        actual_cases = self._get(f"/v1/analytics/actualCases/{backtest_id}")

        result_org_units = {e["orgUnit"] for e in evaluation_entries}
        actual_org_units = {de["ou"] for de in actual_cases["data"]}

        assert (
            result_org_units == actual_org_units
        ), f"Org unit mismatch. Result: {result_org_units}, Actual: {actual_org_units}"
        logger.info("Evaluation flow test completed successfully.")

    def make_dataset(self, data: Dict[str, Any]) -> str:
        """
        Creates a dataset via the API and waits for the associated job to complete.

        Args:
            data (Dict[str, Any]): The dataset creation payload.

        Returns:
            str: The database ID of the created dataset.
        """
        logger.info("Submitting request to create dataset...")
        response = self._post("/v1/analytics/make-dataset", payload=data)
        job_id = response["id"]
        logger.info(f"Dataset creation job initiated. Job ID: {job_id}")
        db_id = self.wait_for_db_id(job_id)
        logger.info(f"Dataset creation job {job_id} completed. Resulting DB ID: {db_id}")
        # Optionally, fetch and verify the created dataset from /v1/crud/datasets/{db_id}
        return db_id

    def evaluate_model(self, dataset_id: str, model_id: str) -> Tuple[List[Dict[str, Any]], str]:
        """
        Initiates a model backtest evaluation and retrieves the results.

        Args:
            dataset_id (str): The ID of the dataset to use for evaluation.
            model_id (str): The ID of the model to evaluate.

        Returns:
            Tuple[List[Dict[str, Any]], str]: A tuple containing the list of evaluation entries
                                             and the backtest ID.
        """
        logger.info(f"Requesting model evaluation for model '{model_id}' on dataset '{dataset_id}'.")
        backtest_payload = {"modelId": model_id, "datasetId": dataset_id, "name": "integration_test_backtest"}
        response = self._post("/v1/crud/backtests/", payload=backtest_payload)
        job_id = response["id"]  # This is the job ID for the backtest creation itself
        # The 'id' in the response is likely the backtest DB ID directly if the endpoint is synchronous for creation.
        # If backtest creation is async, then wait_for_db_id is needed for the backtest job.
        # Assuming the POST to /crud/backtests/ is synchronous for creating the backtest record,
        # and returns the backtest DB ID, not a job ID for *creating* the backtest record.
        # The actual evaluation (running the model) might be an async job triggered by this.
        # The original code implies job_id from POST is for the evaluation task.

        logger.info(f"Backtest evaluation job initiated. Job ID: {job_id}")
        backtest_db_id = self.wait_for_db_id(job_id)  # This job_id should be for the evaluation task
        logger.info(f"Backtest job {job_id} completed. Resulting Backtest DB ID: {backtest_db_id}")

        evaluation_result_meta = self._get(f"/v1/crud/backtests/{backtest_db_id}")
        assert evaluation_result_meta["modelId"] == model_id
        assert evaluation_result_meta["datasetId"] == dataset_id
        assert evaluation_result_meta["name"] == "integration_test_backtest"
        assert evaluation_result_meta["created"] is not None

        url_string = f"/v1/analytics/evaluation-entry?backtestId={backtest_db_id}&quantiles=0.5"
        evaluation_entries = self._get(url_string)
        return evaluation_entries, backtest_db_id

    def wait_for_db_id(self, job_id: str, timeout_seconds: int = 400) -> str:
        """
        Polls the job status endpoint until the job succeeds or fails, or timeout is reached.

        Args:
            job_id (str): The ID of the job to monitor.
            timeout_seconds (int): Maximum time in seconds to wait for job completion.

        Returns:
            str: The database ID of the result associated with the completed job.

        Raises:
            ValueError: If the job status indicates failure.
            TimeoutError: If the job does not complete within the timeout period.
        """
        job_url_base = self._build_url(f"/v1/jobs/{job_id}")
        logger.info(f"Waiting for job {job_id} to complete (timeout: {timeout_seconds}s)...")

        start_time = time.time()
        while time.time() - start_time < timeout_seconds:
            try:
                job_status_response = self._get(f"/v1/jobs/{job_id}")  # Re-use _get for consistency
                job_status = job_status_response.get("status", "").lower()  # Make case-insensitive
                logger.info(f"Job {job_id} status: {job_status}. Response: {job_status_response}")

                if job_status == "failure":
                    logger.error(f"Job {job_id} failed. Full status: {job_status_response}")
                    # Optionally fetch more detailed error info if available from job_status_response
                    raise ValueError(
                        f"Job {job_id} failed. Status details: {job_status_response.get('result', 'No details')}"
                    )
                if job_status == "success":
                    # Assuming the result of a successful job contains the database ID
                    # The original code fetches /database_result/ from the job URL.
                    db_result_url = self._build_url(f"/v1/jobs/{job_id}/database_result/")
                    db_result_response = self._get(f"/v1/jobs/{job_id}/database_result/")  # Use relative path for _get

                    if "id" not in db_result_response:
                        logger.error(
                            f"Job {job_id} succeeded but database_result response missing 'id': {db_result_response}"
                        )
                        raise ValueError(f"Job {job_id} succeeded but database_result response missing 'id'")
                    logger.info(f"Job {job_id} succeeded. DB result ID: {db_result_response['id']}")
                    return db_result_response["id"]

                time.sleep(max(1, int(timeout_seconds / 100)))  # Sleep for 1% of timeout, min 1s
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 404:  # Job might not be registered immediately
                    logger.warning(f"Job {job_id} not found yet (404), retrying...")
                else:
                    raise  # Re-raise other HTTP errors
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request error while polling job {job_id}: {e}. Retrying...")

            time.sleep(1)  # Fallback sleep if inner sleep not hit

        logger.error(f"Job {job_id} timed out after {timeout_seconds} seconds.")
        raise TimeoutError(f"Job {job_id} did not complete within {timeout_seconds} seconds.")


if __name__ == "__main__":
    import sys

    # Determine hostname from command line arg or environment, then default
    arg_hostname = sys.argv[1] if len(sys.argv) > 1 else None
    effective_hostname = arg_hostname or os.getenv("CHAP_TEST_HOSTNAME", "localhost")

    current_chap_url = f"http://{effective_hostname}:8000"
    logger.info(f"Running integration tests against: {current_chap_url}")

    # Determine if all models should be run (e.g., based on another arg or env var)
    run_all_flag = os.getenv("CHAP_TEST_RUN_ALL_MODELS", "false").lower() == "true"

    test_suite = IntegrationTest(current_chap_url, run_all=run_all_flag)

    try:
        logger.info("--- Starting Prediction Flow ---")
        test_suite.prediction_flow()
        logger.info("--- Prediction Flow Completed ---")

        logger.info("--- Starting Evaluation Flow ---")
        test_suite.evaluation_flow()
        logger.info("--- Evaluation Flow Completed ---")

        logger.info("All integration test flows completed successfully.")
    except Exception as e:
        logger.critical(f"Integration test suite failed: {e}", exc_info=True)
        sys.exit(1)
    sys.exit(0)
