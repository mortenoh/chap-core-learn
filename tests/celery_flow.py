# Improvement Suggestions:
# 1. **Configurable Hostname/URL**: Make `hostname` (and thus `chap_url`) configurable via environment variables or command-line arguments for flexibility in testing different deployments.
# 2. **Refined Error Handling**: In `main()`, catch more specific exceptions (e.g., `requests.exceptions.RequestException`) instead of a bare `except:`. Consolidate logging and printing of error messages to avoid redundancy.
# 3. **Robust Status Polling**: The status polling loop in `main()` has a fixed number of retries and sleep time. Implement a more robust polling mechanism with configurable total timeout, retry attempts, and potentially exponential backoff. Also, the `task_id` is commented out in the status URL; if status is task-specific, this needs to be fixed.
# 4. **Explicit Test Assertions**: For an automated test, add explicit assertions to verify expected outcomes (e.g., HTTP status codes, specific values in JSON responses) rather than just printing information.
# 5. **URL Construction**: Use `urllib.parse.urljoin` or f-strings for constructing URLs to improve readability and reduce potential errors from string concatenation.

"""
This is meant to be a standalone python file for testing the flow with docker compose.

This script tests an asynchronous task flow (likely involving Celery) by making
HTTP requests to a CHAP API service. It initiates a task (e.g., adding numbers)
and then polls a status endpoint to check for completion. It's designed to be
run in an environment where the CHAP service is accessible, often via Docker Compose.
"""

import json
import logging
import os  # For environment variable access
import time
from urllib.parse import urljoin  # For better URL construction

import requests

logger = logging.getLogger(__name__)
# Configure logger basic settings if not configured by a higher-level setup
if not logger.handlers:
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
else:
    logger.setLevel(logging.DEBUG)


# Determine hostname from environment variable or default to 'chap'
HOSTNAME = os.getenv("CHAP_HOSTNAME", "chap")
CHAP_BASE_URL = f"http://{HOSTNAME}:8000"


def ensure_up(base_url: str, status_endpoint: str = "/v1/status", retries: int = 5, delay: int = 5) -> None:
    """
    Ensures that the CHAP API service is up and responsive before proceeding with tests.

    It attempts to connect to the specified status endpoint multiple times with delays.

    Args:
        base_url (str): The base URL of the CHAP API service.
        status_endpoint (str): The relative path to the status check endpoint.
        retries (int): The number of times to retry connecting.
        delay (int): The delay in seconds between retries.

    Raises:
        requests.exceptions.ConnectionError: If the service is not up after all retries.
    """
    status_url = urljoin(base_url, status_endpoint)
    logger.info(f"Ensuring service is up at {status_url} (retries={retries}, delay={delay}s)...")
    for i in range(retries):
        try:
            response = requests.get(status_url, timeout=5)  # Added timeout
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
            logger.info(f"Service is up! Status code: {response.status_code}")
            return  # Service is up
        except requests.exceptions.RequestException as e:
            logger.warning(f"Attempt {i+1}/{retries}: Failed to connect to {status_url} ({e}). Retrying in {delay}s...")
            time.sleep(delay)
    logger.error(f"Service at {status_url} is not responding after {retries} retries.")
    raise requests.exceptions.ConnectionError(f"Service at {status_url} not up after {retries} retries.")


def main() -> None:
    """
    Main function to execute the Celery flow test.

    It ensures the service is up, triggers an asynchronous task via the
    '/v1/debug/add-numbers' endpoint, and then polls the '/v1/debug/get-status'
    endpoint to check for the task's successful completion.
    """
    add_url = urljoin(CHAP_BASE_URL, "/v1/debug/add-numbers?a=1&b=2")
    status_base_url = urljoin(CHAP_BASE_URL, "/v1/debug/get-status")
    exception_url = urljoin(CHAP_BASE_URL, "/v1/get-exception")

    try:
        ensure_up(CHAP_BASE_URL)
    except requests.exceptions.ConnectionError as e:
        logger.error(f"Prerequisite check failed: CHAP service is not up. Aborting test. Error: {e}")
        return  # Exit if service is not up

    task_id = None
    try:
        logger.info(f"Requesting task execution at: {add_url}")
        response = requests.get(add_url, timeout=10)
        response.raise_for_status()  # Check for HTTP errors
        response_data = response.json()
        task_id = response_data.get("task_id")  # Assuming the response contains a task_id
        if not task_id:
            logger.error(f"Response from {add_url} did not contain a 'task_id'. Response: {response_data}")
            return
        logger.info(f"Task initiated successfully. Task ID: {task_id}")

    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to connect or request {add_url}: {e}")
        try:
            exception_info_response = requests.get(exception_url, timeout=5)
            exception_info = exception_info_response.json()
            logger.error(f"Exception info from service: {exception_info}")
        except requests.exceptions.RequestException as ie:
            logger.error(f"Failed to fetch exception info from service: {ie}")
        return  # Exit if initial request fails
    except json.JSONDecodeError:
        logger.error(f"Failed to decode JSON response from {add_url}. Response text: {response.text}")
        return

    # Polling for status
    # TODO: Fix task_id usage if status is per-task. Currently commented out in original.
    # status_url_with_task = f"{status_base_url}?task_id={task_id}" if task_id else status_base_url
    status_url_with_task = status_base_url  # Using base status URL as per original commented logic

    max_polls = 10  # Increased polling attempts
    poll_interval = 3  # Seconds
    logger.info(
        f"Polling status at {status_url_with_task} for task {task_id} (max_polls={max_polls}, interval={poll_interval}s)..."
    )

    for i in range(max_polls):
        time.sleep(poll_interval)
        try:
            status_response = requests.get(status_url_with_task, timeout=5)
            status_response.raise_for_status()  # Check for HTTP errors
            status_data = status_response.json()
            current_status = status_data.get("status")
            logger.info(
                f"Poll {i+1}/{max_polls}: Status for task {task_id} is '{current_status}'. Full response: {status_data}"
            )

            if current_status == "SUCCESS":
                logger.info(f"Task {task_id} completed successfully! Final status: {status_data}")
                # Add assertions here, e.g., assert "result" in status_data and status_data["result"] == 3
                return  # Test success
            elif current_status == "FAILURE" or current_status == "REVOKED":  # Handle terminal failure states
                logger.error(f"Task {task_id} failed or was revoked. Final status: {status_data}")
                # Add assertions for failure if needed
                return  # Test failure

        except requests.exceptions.HTTPError as e:
            if (
                e.response.status_code == 404 and task_id
            ):  # Task ID might not be found if polling too soon or wrong endpoint
                logger.warning(
                    f"Poll {i+1}/{max_polls}: Status for task {task_id} returned 404 (Not Found). Task might not exist or polling wrong URL."
                )
            else:
                logger.error(
                    f"Poll {i+1}/{max_polls}: HTTP error when polling status for task {task_id}: {e}. Response: {e.response.text if e.response else 'No response'}"
                )
            # Decide if to break or continue on HTTP errors other than 404 for task status
        except requests.exceptions.RequestException as e:
            logger.error(f"Poll {i+1}/{max_polls}: Request error when polling status for task {task_id}: {e}")
            # Potentially break or implement backoff on repeated connection errors
        except json.JSONDecodeError:
            logger.error(
                f"Poll {i+1}/{max_polls}: Failed to decode JSON status response for task {task_id}. Response text: {status_response.text}"
            )

    logger.error(f"Task {task_id} did not reach SUCCESS state after {max_polls} polls.")
    # Add assertion for timeout/failure here


if __name__ == "__main__":
    logger.info("Starting Celery flow test script...")
    main()
    logger.info("Celery flow test script finished.")
    #
# evaluate_model(chap_url, dataset(), {"name": "naive_model"})
