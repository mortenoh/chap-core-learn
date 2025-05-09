# Improvement Suggestions:
# 1. **Comprehensive Docstrings**: Add clear and concise docstrings to all pytest fixtures, hooks (`pytest_addoption`, `pytest_configure`, `pytest_collection_modifyitems`), and helper classes/functions defined in this file. Explain what each fixture provides, its scope, and any important context. (Primary task of this refactoring).
# 2. **`local_data_path` Portability**: The `local_data_path` fixture uses a hardcoded absolute path (`/home/knut/Data/ch_data/`), making tests dependent on it non-portable. This should be refactored to use relative paths, environment variables for configuration, or by including necessary test data within the repository and accessing it via `data_path` or `importlib.resources`.
# 3. **Celery Worker Skip Logic for Windows**: The conditional skip for `celery_session_worker` on Windows is commented out. If `pytest-celery` has issues on Windows, this logic should be reinstated and clarified. The current passthrough fixture might not behave as expected if the underlying plugin is problematic on Windows.
# 4. **`GEEMock` Implementation Details**: The `GEEMock` class is a basic stub. Its docstring and implementation (especially `get_historical_era5`) should clearly define its mocked behavior, what data it returns (currently random), and how it simulates the real `GoogleEarthEngine` for tests.
# 5. **Database Seeding Documentation (`clean_engine`)**: The `clean_engine` fixture seeds the database using `seed_with_session_wrapper`. Document what initial data this seeding function populates, as this is critical for understanding the baseline state for database-dependent tests.

"""
This `conftest.py` file provides shared fixtures, hooks, and plugins for the
pytest test suite of the CHAP-core application.

It includes:
- Configuration for marking and skipping slow tests.
- Fixtures for providing paths to example data, test outputs, and external models.
- Session-scoped fixtures for managing test caches and database engines.
- Fixtures for Celery worker setup and mocking external services like Google Earth Engine.
- Data loading fixtures for specific test datasets.
"""

import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Generator, List, Tuple  # Added Generator, Any, Dict, List
from unittest.mock import patch

# ignore showing plots in tests
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from sqlmodel import SQLModel, create_engine  # Added create_engine

from chap_core.api_types import RequestV1
from chap_core.assessment.dataset_splitting import train_test_generator
from chap_core.database.database import SessionWrapper
from chap_core.database.dataset_tables import ObservationBase  # Removed DataSet as it conflicts with fixture
from chap_core.database.model_spec_tables import seed_with_session_wrapper

# Import specific tables if needed, avoid `from chap_core.database.tables import *`
# For example: from chap_core.database.tables import SomeTable, AnotherTable
# Assuming FullData is from datatypes, not tables
from chap_core.datatypes import FullData, HealthPopulationData, SimpleClimateData
from chap_core.geometry import Polygons
from chap_core.rest_api_src.data_models import FetchRequest
from chap_core.rest_api_src.v1.routers.crud import DatasetCreate, PredictionCreate

# from chap_core.rest_api_src.v1.routers.analytics import MakePredictionRequest # Not used directly in this file
from chap_core.rest_api_src.worker_functions import WorkerConfig
from chap_core.services.cache_manager import get_cache

from .data_fixtures import *  # Imports fixtures defined in data_fixtures.py

# Don't use pytest-celery if on windows
IS_WINDOWS = os.name == "nt"


@pytest.fixture(scope="session")
def redis_available() -> bool:
    """
    Session-scoped fixture to check if Redis is available and responsive.
    Skips tests that depend on it if Redis is not found or connectable.

    Returns:
        bool: True if Redis is available, otherwise skips the test.
    """
    import redis  # Import here to avoid making redis a hard dependency for all tests

    try:
        r = redis.Redis()  # Default connection: localhost:6379
        r.ping()
        return True
    except redis.exceptions.ConnectionError:
        pytest.skip("Redis not available or connection refused.")
    except ModuleNotFoundError:
        pytest.skip("Python 'redis' library not installed.")


# Conditionally load pytest-celery plugin if not on Windows,
# as it might have compatibility issues.
if not IS_WINDOWS:
    pytest_plugins = ("celery.contrib.pytest",)

    @pytest.fixture(scope="session")
    def celery_session_worker(redis_available, celery_session_worker_instance):  # Renamed to avoid conflict
        """
        Provides a session-scoped Celery worker if Redis is available.
        This fixture depends on the `celery_session_worker` provided by `pytest-celery`.
        """
        # redis_available fixture will skip if Redis is not up.
        return celery_session_worker_instance
else:

    @pytest.fixture(scope="session")
    def celery_session_worker():
        """
        Placeholder fixture for Windows environments where pytest-celery might be skipped.
        """
        pytest.skip("pytest-celery worker is not supported/enabled on Windows for this test suite.")
        return None


plt.ion()  # Turns on interactive mode for matplotlib, plots may show and not block.


def pytest_addoption(parser: Any):
    """
    Pytest hook to add custom command-line options.
    Adds `--run-slow` option to include tests marked as 'slow'.
    """
    parser.addoption("--run-slow", action="store_true", default=False, help="Run slow tests (e.g., integration tests)")


def pytest_configure(config: Any):
    """
    Pytest hook for initial configuration.
    Registers the 'slow' marker.
    """
    config.addinivalue_line("markers", "slow: mark a test as a slow test to run.")


def pytest_collection_modifyitems(config: Any, items: List[Any]):
    """
    Pytest hook to modify collected test items.
    Skips tests marked with 'slow' if `--run-slow` option is not provided.
    """
    if not config.getoption("--run-slow"):
        skip_slow_marker = pytest.mark.skip(reason="need --run-slow option to run this test")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow_marker)


@pytest.fixture(scope="module")  # Changed to module scope for better path consistency per test module
def data_path() -> Path:
    """
    Fixture providing the absolute path to the 'example_data' directory.
    Assumes 'example_data' is a sibling to the 'tests' directory.
    Scope: module.
    """
    return Path(__file__).parent.parent / "example_data"


@pytest.fixture(scope="module")
def output_path() -> Path:
    """
    Fixture providing a path to a 'test_outputs' directory within the 'tests' directory.
    Creates the directory if it doesn't exist.
    Scope: module.
    """
    path = Path(__file__).parent / "test_outputs"
    path.mkdir(exist_ok=True, parents=True)
    return path


@pytest.fixture(scope="module")
def models_path() -> Path:
    """
    Fixture providing the path to the 'external_models' directory.
    Assumes 'external_models' is a sibling to the 'tests' directory.
    Scope: module.
    """
    return Path(__file__).parent.parent / "external_models"


@pytest.fixture(scope="module")
def local_data_path() -> Path:
    """
    Fixture providing a path to a local data directory (`/home/knut/Data/ch_data/`).
    WARNING: This path is user-specific and will cause tests to be skipped if not found.
    Consider refactoring for portability (see improvement suggestion #2).
    Scope: module.
    """
    path = Path("/home/knut/Data/ch_data/")  # User-specific path
    if not path.exists():
        pytest.skip(f"Local data path '{path}' does not exist. Skipping tests dependent on it.")
    return path


@pytest.fixture(scope="module")
def tests_path() -> Path:
    """
    Fixture providing the path to the root of the 'tests' directory.
    Scope: module.
    """
    return Path(__file__).parent


@pytest.fixture(scope="session", autouse=True)
def use_test_cache() -> Generator[None, None, None]:
    """
    Session-scoped, autouse fixture to set a test environment variable
    and clean up the test cache directory after the test session.
    """
    original_test_env = os.environ.get("TEST_ENV")
    os.environ["TEST_ENV"] = "true"
    yield
    if original_test_env is None:
        del os.environ["TEST_ENV"]
    else:
        os.environ["TEST_ENV"] = original_test_env

    # Cache cleanup
    try:
        cache = get_cache()  # Assuming get_cache() respects TEST_ENV or has test mode
        if cache and hasattr(cache, "directory") and cache.directory:
            shutil.rmtree(cache.directory, ignore_errors=True)
            # logger.info(f"Test cache directory '{cache.directory}' removed.") # Requires logger
    except Exception as e:
        # logger.warning(f"Error during test cache cleanup: {e}")
        print(f"Warning: Error during test cache cleanup: {e}")


@pytest.fixture()
def health_population_data(data_path: Path) -> DataSet:  # DataSet from database.dataset_tables
    """
    Fixture loading health and population data from 'health_population_data.csv'.
    Returns a `DataSet` instance (from `chap_core.database.dataset_tables`).
    """
    file_name = (data_path / "health_population_data").with_suffix(".csv")
    # Note: The original DataSet.from_pandas was for chap_core.spatio_temporal_data.temporal_dataclass.DataSet
    # Assuming this should be the spatio-temporal DataSet for consistency with other fixtures.
    from chap_core.spatio_temporal_data.temporal_dataclass import DataSet as SpatioTemporalDataSet

    return SpatioTemporalDataSet.from_pandas(pd.read_csv(file_name), HealthPopulationData)


@pytest.fixture
def nicaragua_path(data_path: Path) -> Path:
    """Fixture providing the path to 'nicaragua_weekly_data.csv'."""
    return (data_path / "nicaragua_weekly_data").with_suffix(".csv")


@pytest.fixture()
def weekly_full_data(nicaragua_path: Path) -> DataSet:  # DataSet from spatio_temporal_data
    """
    Fixture loading weekly data from the Nicaragua dataset into a `FullData` typed `DataSet`.
    """
    from chap_core.spatio_temporal_data.temporal_dataclass import DataSet as SpatioTemporalDataSet

    return SpatioTemporalDataSet.from_pandas(pd.read_csv(nicaragua_path), FullData)


@pytest.fixture
def dumped_weekly_data_paths(weekly_full_data: DataSet, tmp_path: Path) -> Tuple[Path, Path, Path]:
    """
    Splits `weekly_full_data` into train/test, saves them to CSVs in `tmp_path`,
    and returns the paths to these CSV files.
    """
    train, tests = train_test_generator(weekly_full_data, prediction_length=12)
    training_path = tmp_path / "training_data.csv"
    train.to_csv(training_path)

    historic, masked, _ = next(tests)  # Get first test split
    historic_path = tmp_path / "historic_data.csv"
    historic.to_csv(historic_path)
    future_path = tmp_path / "future_data.csv"  # 'masked' usually contains future climate data
    masked.to_csv(future_path)

    return training_path, historic_path, future_path


@pytest.fixture()
def google_earth_engine() -> Any:  # Should be 'GoogleEarthEngine' type if defined
    """
    Fixture providing an instance of `GoogleEarthEngine`.
    Skips tests if GEE is not available or fails to initialize.
    """
    from chap_core.google_earth_engine.gee_era5 import GoogleEarthEngine  # Local import

    try:
        return GoogleEarthEngine()
    except Exception as e:  # Catch broad exception as GEE init can fail for various reasons
        pytest.skip(f"Google Earth Engine not available or initialization failed: {e}")


@pytest.fixture()
def mocked_gee(gee_mock: Type["GEEMock"]) -> Generator[None, None, None]:
    """
    Fixture that patches `Era5LandGoogleEarthEngine` with `GEEMock` for testing purposes.
    """
    # Ensure the target path for patching is correct relative to where it's used.
    with patch("chap_core.rest_api_src.worker_functions.Era5LandGoogleEarthEngine", gee_mock):
        yield


@pytest.fixture()
def gee_mock() -> Type["GEEMock"]:
    """Provides the GEEMock class for mocking Google Earth Engine interactions."""
    return GEEMock


@pytest.fixture
def request_json(data_path: Path) -> str:
    """Fixture providing the content of 'v1_api/request.json' as a string."""
    with open(data_path / "v1_api/request.json", "r", encoding="utf-8") as f:
        return f.read()


@pytest.fixture
def big_request_json(data_path: Path) -> str:
    """
    Fixture providing the content of 'anonymous_chap_request.json'.
    Skips if the file does not exist.
    """
    filepath = data_path / "anonymous_chap_request.json"
    if not os.path.exists(filepath):  # Keep os.path.exists for direct path check
        pytest.skip(f"Required test data file not found: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


@pytest.fixture
def laos_request(local_data_path: Path) -> str:
    """
    Fixture loading a Laos-specific request JSON from the `local_data_path`.
    Modifies 'estimator_id' to 'naive_model'.
    WARNING: Depends on user-specific `local_data_path`.
    """
    filepath = local_data_path / "laos_requet.json"  # Typo in original: "laos_requet.json"
    if not filepath.exists():
        pytest.skip(f"Laos request file not found at {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
    data_dict = json.loads(text)
    data_dict["estimator_id"] = "naive_model"
    return json.dumps(data_dict)


@pytest.fixture
def laos_request_2(local_data_path: Path) -> str:
    """
    Fixture loading a second Laos-specific request JSON from `local_data_path`.
    Modifies 'estimator_id' to 'naive_model'.
    WARNING: Depends on user-specific `local_data_path`.
    """
    filepath = local_data_path / "laos_request_2.json"
    if not filepath.exists():
        pytest.skip(f"Laos request file (2) not found at {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
    data_dict = json.loads(text)
    data_dict["estimator_id"] = "naive_model"
    return json.dumps(data_dict)


@pytest.fixture
def laos_request_3(local_data_path: Path) -> str:
    """
    Fixture loading a third Laos-specific request JSON from `local_data_path`.
    Modifies 'estimator_id' to 'naive_model'.
    WARNING: Depends on user-specific `local_data_path`.
    """
    filepath = local_data_path / "laos_request_3.json"
    if not filepath.exists():
        pytest.skip(f"Laos request file (3) not found at {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
    data_dict = json.loads(text)
    data_dict["estimator_id"] = "naive_model"
    return json.dumps(data_dict)


@pytest.fixture
def dataset_create(big_request_json: str) -> DatasetCreate:
    """
    Fixture creating a `DatasetCreate` Pydantic model instance
    based on the 'big_request_json' data.
    """
    data = RequestV1.model_validate_json(big_request_json)
    observations = []
    for f_item in data.features:
        # Use 'disease_cases' as feature_name if original featureId is 'diseases'
        feature_name_val = "disease_cases" if f_item.featureId == "diseases" else f_item.featureId
        for d_item in f_item.data:
            observations.append(
                ObservationBase(feature_name=feature_name_val, period=d_item.pe, orgUnit=d_item.ou, value=d_item.value)
            )

    return DatasetCreate(
        name="test_dataset_from_big_request",
        type="evaluation",  # Example type
        geojson=data.orgUnitsGeoJson.model_dump(),
        observations=observations,
    )


@pytest.fixture()
def example_polygons(data_path: Path) -> DFeatureCollectionModel:  # Return type from api_types
    """Fixture loading example polygons from 'example_polygons.geojson'."""
    return Polygons.from_file(data_path / "example_polygons.geojson").data


@pytest.fixture
def make_prediction_request(dataset_create: DatasetCreate) -> PredictionCreate:
    """
    Fixture creating a `PredictionCreate` Pydantic model instance.
    Uses 'naive_model' and data from the `dataset_create` fixture.
    """
    # dataset_create.dict() might be deprecated for Pydantic V2, use model_dump()
    return PredictionCreate(
        model_id="naive_model",
        metaData={"test_meta_key": "test_meta_value"},  # Changed to avoid keyword 'test'
        **dataset_create.model_dump(),
    )


# Commented out Celery app fixture - if needed, ensure it's correctly configured.
# @pytest.fixture
# def celery_app():
#     from celery import Celery # Local import
#     app = Celery(
#         broker="memory://", # Simple in-memory broker for tests
#         backend="cache+memory://", # Simple in-memory backend
#         include=['chap_core.rest_api_src.celery_tasks'] # Ensure tasks are discoverable
#     )
#     app.conf.update(
#         task_always_eager=True,      # Execute tasks locally and synchronously
#         task_eager_propagates=True,  # Propagate exceptions from eager tasks
#         task_serializer="pickle",    # Ensure compatibility with task data
#         accept_content=["pickle"],
#         result_serializer="pickle",
#     )
#     return app


class GEEMock:
    """
    A mock class for `GoogleEarthEngine` (or `Era5LandGoogleEarthEngine`)
    to simulate its behavior in tests without making actual GEE calls.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        """Initializes the GEEMock. Accepts any arguments but does not use them."""
        logger.info(f"GEEMock initialized with args: {args}, kwargs: {kwargs}")
        pass  # ... is a valid stub, but pass is more common for empty blocks

    def get_historical_era5(
        self,
        features: Dict[str, Any],  # Typically a FeatureCollection dict
        periodes: Any,  # Typically a PeriodRange or similar
        fetch_requests: Optional[List[FetchRequest]] = None,
    ) -> DataSet:  # DataSet from spatio_temporal_data
        """
        Mocks the `get_historical_era5` method.

        Returns a DataSet populated with random data for temperature and precipitation
        for each location ID found in the input `features`.

        Args:
            features (Dict[str, Any]): A GeoJSON-like FeatureCollection dictionary.
            periodes (Any): The period range for which data is requested.
            fetch_requests (Optional[List[FetchRequest]]): Optional list of specific fetch requests.

        Returns:
            DataSet: A DataSet containing mocked climate data.
        """
        logger.info(f"GEEMock.get_historical_era5 called for {len(features.get('features', []))} features.")
        locations = [f["id"] for f in features.get("features", []) if f.get("id")]

        # Determine length of periodes if it's a PeriodRange or list-like
        num_periods = len(periodes) if hasattr(periodes, "__len__") else 10  # Default length if unknown

        from chap_core.spatio_temporal_data.temporal_dataclass import DataSet as SpatioTemporalDataSet

        return SpatioTemporalDataSet(
            {
                location: SimpleClimateData(
                    time_period=periodes,  # Pass through periodes
                    rainfall=np.random.rand(num_periods),
                    mean_temperature=np.random.rand(num_periods),
                )
                for location in locations
            }
        )


@pytest.fixture(scope="session")
def database_url() -> str:
    """
    Session-scoped fixture providing the URL for the test database.
    Uses an SQLite database file named 'test.db' in the tests directory.
    """
    cur_dir = Path(__file__).parent
    db_path = cur_dir / "test.db"
    return f"sqlite:///{db_path.resolve()}"


@pytest.fixture(scope="session")
def clean_engine(database_url: str) -> Any:  # Return type should be sqlalchemy.engine.Engine
    """
    Session-scoped fixture that creates a new, clean SQLite database engine.
    It drops all existing tables, recreates the schema based on SQLModel metadata,
    and seeds the database with initial data using `seed_with_session_wrapper`.

    Args:
        database_url (str): The URL for the test database.

    Returns:
        sqlalchemy.engine.Engine: The created SQLAlchemy engine.
    """
    logger.info(f"Setting up clean test database engine for URL: {database_url}")
    engine = create_engine(database_url, connect_args={"check_same_thread": False})  # For SQLite

    # Ensure tables are dropped and created in a controlled manner
    SQLModel.metadata.drop_all(engine)
    SQLModel.metadata.create_all(engine)

    logger.info("Database schema created. Seeding initial data...")
    with SessionWrapper(engine) as session:
        seed_with_session_wrapper(session)  # Document what this seeds
    logger.info("Database seeded.")
    return engine


@pytest.fixture(scope="session")
def celery_config(database_url: str, redis_available: bool) -> Dict[str, str]:
    """
    Session-scoped fixture providing Celery configuration for tests.
    Uses Redis as broker and backend if available, otherwise might need adjustment
    or tests relying on this might be skipped by `redis_available`.

    Args:
        database_url (str): The test database URL, included in the config.
        redis_available (bool): Indicates if Redis is available.

    Returns:
        Dict[str, str]: A dictionary of Celery configuration settings.
    """
    # redis_available fixture handles skipping if Redis is not up.
    logger.info(f"Celery config using database_url: {database_url} and Redis for broker/backend.")
    return {
        "broker_url": "redis://localhost:6379/0",  # Added /0 for default Redis DB
        "result_backend": "redis://localhost:6379/0",
        "task_serializer": "pickle",
        "accept_content": ["pickle"],
        "result_serializer": "pickle",
        "database_url": database_url,  # For tasks that might need DB access
    }


@pytest.fixture(scope="session")
def celery_worker_pool() -> str:
    """
    Session-scoped fixture specifying the Celery worker pool type.
    'prefork' is a common default.
    """
    return "prefork"  # Options: prefork, eventlet, gevent, solo (for debugging)


@pytest.fixture
def test_config() -> WorkerConfig:
    """
    Fixture providing a `WorkerConfig` instance configured for testing
    (e.g., `is_test=True`).
    """
    return WorkerConfig(is_test=True)
