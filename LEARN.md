# CHAP-Core Learning Document

This document contains detailed summaries and insights generated during a file-by-file exploration of the CHAP-core project.

## `tests/api/test_json.py`

**Purpose**:
This test file focuses on validating the parsing of JSON data into Pydantic models, specifically `chap_core.api_types.RequestV1`, and testing a utility function `chap_core.rest_api_src.worker_functions.v1_conversion`. This is crucial for ensuring that API request data is correctly handled and transformed within the system.

**Key Components**:

- **`request_model` fixture**:

  - Parses a JSON string (expected to be provided by another fixture named `request_json`, likely from `conftest.py`) into a `RequestV1` Pydantic model.
  - This fixture provides a validated model instance for use in other tests.

- **`test_validate_json(request_json)`**:

  - Directly validates that a given JSON string can be successfully parsed into the `RequestV1` model.
  - Asserts that the parsed object is of the correct type (`RequestV1`).
  - Asserts that key attributes like `orgUnitsGeoJson` and `features` are present in the parsed model.

- **`test_convert_pydantic(request_model)`**:
  - Tests the `v1_conversion` function using data from a `RequestV1` model instance (provided by the `request_model` fixture).
  - It passes the data from the first feature of the `request_model` to `v1_conversion`.
  - Currently includes a placeholder assertion that the result (`st`) is not `None`. More specific assertions would depend on the detailed behavior of `v1_conversion`.
  - Includes `pytest.skip` conditions if the `request_model` has no features or if the first feature contains no data elements, preventing tests from running with unsuitable data.

**Integration**:
This file tests components integral to the API's data intake and initial processing layer.

- `RequestV1` (from `chap_core.api_types`) likely defines the expected structure for incoming V1 API requests.
- `v1_conversion` (from `chap_core.rest_api_src.worker_functions`) appears to be a utility function. It's likely used by API handlers or background workers to transform parts of the incoming request data (specifically, the `data` attribute of a feature) into an internal format suitable for further processing or model input.

**Dependencies**:

- `pytest`: For the testing framework and fixtures.
- `chap_core.api_types.RequestV1`: The Pydantic model defining the structure of V1 API requests.
- `chap_core.rest_api_src.worker_functions.v1_conversion`: The function being tested for data conversion.
- Implicitly depends on a `request_json` fixture (likely defined in `tests/conftest.py`) which should provide the raw JSON string for testing.

---

## `tests/climate_data/__init__.py`

**Purpose**:
This `__init__.py` file serves to mark the `tests/climate_data/` directory as a Python sub-package. This allows modules within this directory to use relative imports and be part of the larger `tests` package structure.

**Key Components**:

- It contains a single import: `from ..mocks import ClimateDataBaseMock`. This imports `ClimateDataBaseMock` from a `mocks` module or package located in the parent directory (i.e., `tests/mocks.py` or `tests/mocks/__init__.py`).

**Integration**:

- By importing `ClimateDataBaseMock`, this `__init__.py` potentially makes this mock class available to other modules within the `tests.climate_data` sub-package if they were to import it directly from `tests.climate_data` (e.g., `from . import ClimateDataBaseMock`).
- The file itself doesn't execute any code beyond the import but plays a structural role in the test suite's organization. An `__all__` list could be added to explicitly re-export `ClimateDataBaseMock` if that is the primary intention.

**Functionality**:

- Primarily package initialization.
- The imported `ClimateDataBaseMock` is likely a base class or a utility for creating mock objects that simulate climate database interactions. These mocks would be used by tests within the `tests/climate_data/` sub-package to isolate them from actual database dependencies or to provide controlled test data for climate-related functionalities.

**Dependencies**:

- Depends on a `mocks` module or package located in the parent `tests/` directory, which is expected to contain the `ClimateDataBaseMock` definition.

---

## `tests/climate_data/test_gee_database.py`

**Purpose**:
This test file is intended for testing the fetching and processing of ERA5 climate data, presumably from Google Earth Engine (GEE). It includes tests for both monthly and daily data resolutions and a placeholder for testing climate data fetching for a broader dataset.

**Current Status**:

- **All tests in this file are currently marked with `@pytest.mark.skip`.**
- The primary reason cited is "ee not supported," indicating potential issues with Google Earth Engine access, setup, or that the specific GEE functionalities being tested are not currently enabled or available in the test environment.
- An import for `ERA5DataBase` from a `chap_core.climate_data.gee_legacy` module is commented out, suggesting that the tests might be targeting a deprecated or legacy GEE interface.

**Key Components (as originally intended)**:

- **`test_era5()`**:

  - Aimed to test fetching monthly ERA5 climate data using the (presumably legacy) `ERA5DataBase` class.
  - It was designed to check if the data fetched for different time period ranges (e.g., 6 months, 18 months) has the expected number of data points.
  - Included an operation to write the fetched data to a file named `"climate_data.csv"` (this practice is generally discouraged in automated tests in favor of using temporary paths).

- **`test_era5_daily()`**:

  - Similar to `test_era5()`, but focused on fetching daily ERA5 data.
  - It intended to verify the length of fetched daily data series for different date ranges.
  - Also included an operation to write data to `"climate_data_daily.csv"`.

- **`test_get_climate_data_for_dataset(google_earth_engine)`**:
  - This test is a placeholder and appears incomplete.
  - It uses a `google_earth_engine` fixture (likely defined in `tests/conftest.py`), suggesting it might be intended to test a newer or different GEE interface than the legacy `ERA5DataBase`.
  - To be functional, it would require implementation to define a sample dataset, call a GEE data fetching method using the fixture, and assert the properties of the returned climate data.

**Observations**:

- The tests rely on `Location`, `Month`, and `Day` objects from `chap_core.datatypes` and `chap_core.time_period` for defining test parameters.
- The hardcoded output file paths in the skipped tests (`climate_data.csv`, `climate_data_daily.csv`) should be replaced with pytest's `tmp_path` fixture if these tests are revived, to ensure proper test isolation and cleanup.
- Assertions in the skipped tests primarily focus on the length of the returned data. If reactivated, these assertions should be expanded to include more detailed checks on data content, types, or statistical properties.
- The file includes a minimal mock class definition for `ERA5DataBase` to allow the file to be parsed even with the original import commented out. This mock would need to be more sophisticated if the tests were unskipped without access to the actual `ERA5DataBase`.

**Dependencies (if tests were active)**:

- `pytest` for the testing framework.
- `chap_core.datatypes.Location`.
- `chap_core.time_period.Month`, `chap_core.time_period.Day`.
- A Google Earth Engine interface:
  - Originally `ERA5DataBase` from an apparently legacy module (`chap_core.climate_data.gee_legacy`).
  - Potentially a newer interface provided by the `google_earth_engine` fixture.
- A properly configured and authenticated Google Earth Engine environment.

---

## `tests/climate_data/test_gee_era5.py`

**Purpose**:
This test module is dedicated to testing functionalities related to Google Earth Engine (GEE), specifically for ERA5 Land climate data. It covers a range of operations including unit conversions for climate variables, parsing of GEE data properties, manipulation of GEE objects (Image, Feature), fetching daily climate data, harmonizing GEE data with other application-specific data types, and testing direct GEE API calls.

**Key Features & Components Tested**:

- **Unit Conversions**:

  - `test_kelvin_to_celsium()`: Verifies the conversion from Kelvin to Celsius.
  - `test_meter_to_mm()`: Verifies the conversion from meters (common for precipitation in GEE) to millimeters.
  - `test_round_two_decimal()`: Tests rounding functionality (using built-in `round`).

- **GEE Property Parsing**:

  - `test_parse_gee_properties()`: Uses a `property_dicts` fixture (simulating raw GEE output) to test the `Era5LandGoogleEarthEngineHelperFunctions.parse_gee_properties` method. It ensures that these raw properties are correctly transformed into the project's internal `DataSet` structure, with correct data for different locations and indicators.

- **GEE Object Manipulation Helpers** (testing methods of `Era5LandGoogleEarthEngineHelperFunctions`):

  - `test_get_period()`: Tests `get_image_for_period` for its ability to correctly filter or create a GEE Image corresponding to a specific `Band` (e.g., temperature, precipitation) and time period. Asserts properties of the resulting GEE Image like type, band ID, and time range.
  - `test_create_ee_dict()`: Tests `create_ee_dict` for its correctness in generating GEE-compatible dictionary representations of time periods (e.g., a `Month` object).
  - `test_creat_ee_feature()`: Tests `creat_ee_feature` for constructing a GEE Feature, ensuring properties from a source GEE Image and a specific value are correctly embedded into the new feature's properties.
  - `test_convert_value_by_band_converter()`: Verifies that band-specific unit conversion functions (defined in `Band` objects) are correctly applied to data values based on their 'indicator' type.
  - `test_feature_collection_to_list()`: Ensures that a GEE `FeatureCollection` can be correctly converted into a Python list of dictionaries.

- **Daily Data Fetching & Processing**:

  - `test_get_daily_data()`: An integration test (requires live GEE access, uses `era5_land_gee_instance` fixture) for `Era5LandGoogleEarthEngine.get_daily_data`. It fetches daily climate data for a subset of input polygons and a specified time range, then checks for data presence and structure. Output is written to a temporary CSV.
  - `test_pack_daily_data()` (commented out): Was intended to test the `pack_daily_data` function, which likely reshapes or aggregates flat daily data records into a more structured format (e.g., daily values nested under monthly periods within a `DataSet`).
  - `test_harmonize_daily_data()`: Tests the `harmonize_with_daily_data` function. This is a key integration test that combines application-specific `HealthPopulationData` with daily GEE climate data. It uses a mock for the GEE data fetching part (`mock_get_daily_data`) to ensure deterministic behavior. It verifies that the resulting harmonized dataset has the correct structure, time periods, and that the climate data (e.g., `temperature_2m`, `total_precipitation_sum`) is correctly shaped (e.g., as arrays of daily values per month).

- **Direct GEE API Calls** (tests for `chap_core.google_earth_engine.gee_raw.fetch_era5_data`):
  - `test_gee_api()`: An integration test (skipped by default) that uses GEE credentials and Pydantic `FeatureCollectionModel` to fetch ERA5 data directly. Asserts that data is returned and has the expected number of entries based on features, time range, and bands.
  - `test_gee_api_simple()`: Similar to `test_gee_api`, but passes credentials as a dictionary and features as a JSON string to test flexibility in input types for `fetch_era5_data`. Also skipped by default.

**Fixtures**:
The file defines and uses numerous pytest fixtures to provide:

- Initialized GEE API object (`ee`).
- An instance of `Era5LandGoogleEarthEngine` (`era5_land_gee_instance`), which handles GEE initialization and skips tests if GEE is unavailable.
- Sample GEE objects like `ImageCollection` (`collection`), `Dictionary` for periods (`periode`), `Feature` (`ee_feature`), and `Image` (`ee_image`).
- Test data for property parsing (`property_dicts`), band definitions (`list_of_bands`, `band`), and data conversion (`data_for_conversion`).
- GEE service account credentials (`gee_credentials`), loaded from environment variables.
- Polygon data from GeoJSON files (`polygons`, `polygon_json_str`), using `data_path` fixture from `conftest.py`.

**GEE Interaction and Mocking**:

- Several tests are designed for live GEE interaction and are marked with `@pytest.mark.skip` by default or rely on fixtures like `era5_land_gee_instance` and `gee_credentials` that skip if GEE is not set up.
- The `test_harmonize_daily_data` function demonstrates effective mocking of GEE data fetching (`era5_land_gee_instance.get_daily_data`) to allow for deterministic testing of the harmonization logic without live GEE calls.

**Overall**:
This module is critical for ensuring the reliability of climate data integration from Google Earth Engine, covering low-level utilities, helper functions for GEE object manipulation, and higher-level data processing and harmonization workflows. The distinction between tests requiring live GEE access (and thus skipped by default) and those that can run with mocks is important for CI/CD environments.

---

## `tests/climate_data/test_mocking.py`

**Purpose**:
This test file is designed to verify the behavior of mock objects used for climate data, specifically `ClimateDataBaseMock`. The primary goal is to ensure that this mock, when its `get_data` method is called, returns data of the expected length for given time periods. This allows other components that depend on climate data to be tested in isolation without needing a live climate database or GEE connection.

**Key Components**:

- **`test_mock()`**:
  - This is the only test function in the file.
  - It instantiates `ClimateDataBaseMock` (imported from `..mocks`, implying it's defined in `tests/mocks.py` or a `tests/mocks` package).
  - It calls the `get_data` method of the mock with predefined `Location` and `Month` objects representing different time period ranges.
  - It asserts that the length of the data returned by the mock matches expected values (7 for a 7-month period, 19 for a 19-month period). This implicitly tests the mock's logic for generating data based on the duration between the start and end months.

**Integration**:

- This test directly validates the `ClimateDataBaseMock`. The mock itself is intended to be used in other tests throughout the `tests/climate_data` sub-package (or potentially other test modules) to simulate interactions with a climate data source.
- By ensuring the mock behaves predictably (at least in terms of data length), other tests can reliably use it as a stand-in for actual climate data fetching components.

**Functionality Tested**:

- The core functionality tested is the `get_data` method of `ClimateDataBaseMock`, specifically its ability to return a dataset (presumably a list or array-like structure) whose length corresponds correctly to the number of months in the requested period range.

**Dependencies**:

- `pytest` (implicitly, for test execution).
- `chap_core.datatypes.Location`: For creating `Location` objects.
- `chap_core.time_period.Month`: For creating `Month` objects to define time periods.
- `..mocks.ClimateDataBaseMock`: The mock class being tested.

---

## `chap_core/adaptors/__init__.py`

**Purpose**:
This `__init__.py` file's primary role is to designate the `chap_core/adaptors/` directory as a Python package. This enables the use of modules within this directory as part of the `chap_core.adaptors` namespace and allows for relative imports between them.

**Key Components**:

- The file was originally empty.
- The updated version now includes a module docstring explaining the general purpose of the `adaptors` package.
- It also includes commented-out examples of how one might re-export commonly used classes/functions from other modules within the `adaptors` package using an `__all__` list for easier access by external code.

**Integration**:

- As an `__init__.py` file, it's fundamental to Python's package system. It allows Python's import mechanism to recognize `chap_core/adaptors` as a package.
- The `adaptors` package itself, as described in its new docstring, is intended to contain modules that bridge CHAP-core's internal systems with external interfaces, libraries (like GluonTS), or specific data formats (e.g., for command-line tools or REST APIs).

**Functionality**:

- Currently, its main functionality is to enable the `adaptors` directory to be treated as a package.
- It does not define or execute any code itself beyond comments and the (commented-out) example re-exports.

**Dependencies**:

- None directly within this file, but the commented-out examples suggest potential dependencies on other modules within the `chap_core.adaptors` package (e.g., `command_line_interface.py`, `gluonts.py`, `rest_api.py`).

---

## `chap_core/adaptors/command_line_interface.py`

**Purpose**:
This module provides functionality to dynamically generate command-line interfaces (CLIs) for CHAP-core models using the `cyclopts` library. It allows creating `train` and `predict` subcommands for a given model estimator class or a `ModelTemplate`. This facilitates model interaction via the command line, useful for scripting, automation, or environments without a full API deployment.

**Key Functions**:

- **`generate_app(estimator_class: Type[Any]) -> App`**:

  - **Input**: A model estimator class.
  - **Functionality**:
    - Infers the input data structure (dataclass `dc`) for the model by introspecting the type hints of the estimator's `train` method (via `get_dataclass`). Raises a `ValueError` if this inference fails.
    - Creates a `cyclopts.App` instance.
    - Defines two subcommands for this app:
      - `train(training_data_filename: Path, model_path: Path)`:
        - Loads training data from a CSV using `DataSet.from_csv(..., dc)`.
        - Instantiates `estimator_class`, calls its `train` method, and saves the trained predictor.
        - Includes basic error handling for file operations and training.
      - `predict(model_filename: Path, historic_data_filename: Path, future_data_filename: Path, output_filename: Path)`:
        - Loads a trained predictor, historic data (using `dc`), and future data (using `future_dc` where the target variable "disease_cases" is removed from `dc`).
        - Calls the predictor's `predict` method and saves forecasts to a CSV.
        - Includes basic error handling.
  - **Output**: Returns the configured `cyclopts.App` object.

- **`generate_template_app(model_template: ModelTemplate) -> Tuple[App, Any, Any]`**:
  - **Input**: A `ModelTemplate` instance.
  - **Functionality**:
    - Similar to `generate_app`, creates a `cyclopts.App` with `train` and `predict` subcommands, but tailored for models defined via `ModelTemplate`.
    - **`train` command**:
      - Accepts an optional `model_config_path` for model-specific configurations.
      - Instantiates the estimator via `model_template.get_model(model_config)`.
      - Dynamically creates the input dataclass (`dc`) using `create_tsdataclass` based on `estimator.covariate_names` and an assumed target name (e.g., "disease_cases").
      - Loads data, trains, and saves the model.
    - **`predict` command**:
      - Also accepts `model_config_path`.
      - Loads predictor and data similarly, using dynamically created dataclasses.
    - Contains a large commented-out block related to `ModelTemplateConfig`, suggesting potential future structured configuration for entry points.
  - **Output**: Returns the `App` object along with the inner `train` and `predict` functions (noted as an unconventional return signature).

**Integration**:

- Acts as an adaptor layer, exposing core model functionalities (`train`, `predict`) through a CLI.
- Relies on `chap_core.datatypes` (for `DataSet`, `create_tsdataclass`, `remove_field`), `chap_core.model_spec` (for `get_dataclass` to infer data types), and `chap_core.models` (for `ModelTemplate` and `ModelPredictor` interfaces).
- The generated CLIs interact with the file system by reading CSV data and model files, and writing output CSVs and model files.

**Key Technologies/Patterns**:

- **`cyclopts`**: Library used for declarative CLI creation.
- **Dynamic CLI Generation**: CLIs are not hardcoded but generated based on model/template introspection.
- **Type Hint Introspection**: `get_dataclass` is used to infer data structures from model method type hints.
- **Dynamic Dataclass Creation**: `create_tsdataclass` builds data structures on-the-fly.

**Error Handling**:

- Basic error handling for `FileNotFoundError`, `pandas` parsing errors, and general exceptions is present within the generated command functions, with errors logged.

**TODOs/Notes from Code**:

- A `TODO` in `generate_app` notes a need for clarity on whether `estimator_class` should be an instance or a class type.
- A `TODO` in `generate_template_app`'s `train` command indicates a need for a more robust method within `ModelTemplate` to define the fields for dynamic dataclass creation.

---

## `chap_core/adaptors/gluonts.py`

**Purpose**:
This module serves as an adaptor layer to integrate the GluonTS time series forecasting library with CHAP-core's internal data structures and workflows. It provides wrappers around GluonTS's `Estimator` and `Predictor` objects, allowing CHAP-core to use GluonTS models for training and prediction while working with its native `DataSet` objects.

**Key Classes**:

- **`GluonTSPredictor` (dataclass)**:

  - Wraps a pre-trained GluonTS `Predictor` object (`GluonTSPredictorNative`).
  - **`predict(history, future_data, num_samples)` method**:
    - Takes CHAP-core `DataSet` objects for historical data and future covariates.
    - Uses `DataSetAdaptor.to_gluonts_testinstances` to convert these into a format suitable for GluonTS.
    - Calls the underlying GluonTS predictor's `predict` method.
    - Converts the GluonTS forecast results (which include sample paths and time indices) back into a CHAP-core `DataSet` where each item is a `Samples` object (from `chap_core.datatypes`). The samples are transposed to match the expected shape.
  - **`save(directory_path)` method**: Serializes the wrapped GluonTS predictor to a specified directory using the predictor's `serialize` method. Creates the directory if it doesn't exist.
  - **`load(cls, directory_path)` class method**: Deserializes a GluonTS predictor from a directory and returns a new `GluonTSPredictor` instance wrapping it.

- **`GluonTSEstimator` (dataclass)**:
  - Wraps a GluonTS `Estimator` object (`GluonTSEstimatorNative`).
  - **`train(dataset)` method**:
    - Takes a CHAP-core `DataSet` (containing `TimeSeriesData`) as input for training.
    - Uses `DataSetAdaptor.to_gluonts` to convert the input `DataSet` into a list of GluonTS-compatible data entries.
    - Creates a `gluonts.dataset.common.ListDataset` from this list, specifying a frequency (`freq`). (Currently hardcoded to "M" - monthly, with a TODO to make it dynamic).
    - Calls the underlying GluonTS estimator's `train` method with the `ListDataset`.
    - Returns a `GluonTSPredictor` instance that wraps the trained GluonTS predictor.

**Integration**:

- This module is a key component for leveraging GluonTS models within the CHAP-core ecosystem.
- It relies heavily on `chap_core.data.gluonts_adaptor.dataset.DataSetAdaptor` for the crucial step of converting data between CHAP-core's `DataSet` format and the formats expected by GluonTS (typically lists of dictionaries with specific keys like 'start', 'target', 'feat_dynamic_real').
- The output `DataSet[Samples]` from `GluonTSPredictor.predict` can then be used by other CHAP-core components for evaluation, visualization, or further processing.

**Key Technologies/Patterns**:

- **Adapter Pattern**: The classes in this module are clear examples of the adapter pattern, making the GluonTS library's interface compatible with CHAP-core's interface.
- **Dataclasses**: Used for simple, structured wrappers.
- **Serialization/Deserialization**: GluonTS predictors have built-in `serialize` and `deserialize` methods, which are leveraged here for model persistence.

**Error Handling**:

- Basic error handling for file I/O and exceptions during GluonTS operations (training, prediction, serialization) has been added, typically logging the error and re-raising.

**Dependencies**:

- `gluonts`: The core external library being adapted.
- `chap_core.data.DataSet` and `chap_core.data.gluonts_adaptor.dataset.DataSetAdaptor`: For CHAP-core data representation and conversion.
- `chap_core.datatypes.Samples`, `chap_core.datatypes.TimeSeriesData`: For representing forecast outputs and typing inputs.
- `chap_core.time_period.PeriodRange`: For handling time period information.
- `pathlib` for path manipulation.

---

## `chap_core/adaptors/rest_api.py`

**Purpose**:
This module is designed to dynamically generate a FastAPI web application that exposes CHAP-core model functionalities (training and prediction) via a REST API. It takes a model estimator class and a working directory to create API endpoints.

**Key Function**:

- **`generate_app(estimator_class: Type[Any], working_dir: str) -> FastAPI`**:
  - **Input**:
    - `estimator_class`: The Python class of the model estimator to be served. This class is expected to have `train` and `load_predictor` methods.
    - `working_dir`: A string path to a directory used for storing models and potentially data files.
  - **Functionality**:
    - Initializes a `FastAPI` application instance with a title and description derived from the estimator class name.
    - Configures CORS (Cross-Origin Resource Sharing) middleware, allowing all origins by default (this is permissive and should be restricted in production environments).
    - Infers the input data structure (`dc`) for the model by introspecting the type hints of the `estimator_class.train` method using `get_dataclass` (from `chap_core.model_spec`). Raises a `ValueError` if this inference fails.
    - Dynamically creates a Pydantic model (`pydantic_training_model`) based on `dc`'s annotations. This model is intended for validating the request body of the `/train/` endpoint.
    - Defines two API endpoints using FastAPI decorators (`@app.post`):
      - **`POST /train/` (async)**:
        - Intended to receive a list of training data records (`training_data_payload`) in the request body, conforming to the dynamically created `pydantic_training_model`.
        - **Critical Implementation Note**: The current implementation of this endpoint has a significant issue. It defines `training_data_payload` as an input parameter but then attempts to load data from a hardcoded file path (`training_data_filename`) and calls `DataSet.df_from_pydantic_observations()` without arguments. This logic needs to be refactored to correctly process the `training_data_payload` received in the request body into a `DataSet`.
        - The conceptual flow involves instantiating the `estimator_class`, calling its `train` method with the (correctly processed) dataset, and saving the trained predictor to a path derived from `working_dir`.
        - Includes error handling that returns FastAPI `HTTPException` on failure.
      - **`POST /predict/` (async)**:
        - This endpoint currently expects several file paths (`model_filename`, `historic_data_filename`, `future_data_filename`, `output_filename`) as request parameters. This is an unconventional design for a REST API, where data is typically part of the request body or referenced by server-managed IDs. This might be intended for specific local or batch processing scenarios invoked via the API.
        - It loads a trained model, historic data, and future data from these paths (interpreted relative to `working_dir`).
        - Performs prediction and saves the forecasts to `output_filename` on the server.
        - Returns a success message including the path to the output file.
        - Includes error handling returning `HTTPException`.
  - **Output**: Returns the configured `FastAPI` application instance.

**Integration**:

- Acts as an adaptor to expose CHAP-core models over an HTTP REST API.
- Uses `FastAPI` for the web framework and `pydantic` for data validation of request bodies (via the dynamically created model).
- Interacts with `chap_core.datatypes` (for `DataSet`, `remove_field`), `chap_core.model_spec` (`get_dataclass`), and model interfaces (`ModelPredictor`).
- The generated API endpoints, as currently designed, interact heavily with the file system based on the `working_dir` and provided filenames.

**Key Technologies/Patterns**:

- **FastAPI**: A modern Python web framework for building APIs.
- **Pydantic**: Used for data validation (implicitly by FastAPI for request bodies based on type hints, and explicitly for dynamic model creation).
- **Dynamic API Generation**: API endpoints are, to some extent, generated based on the input `estimator_class`.
- **CORS**: Configured to allow cross-origin requests.

**Error Handling**:

- Endpoint functions include `try-except` blocks that catch exceptions and return FastAPI `HTTPException` objects with appropriate status codes (e.g., 400 for bad request/value error, 404 for file not found, 500 for internal server errors).

**Configuration/State**:

- The `working_dir` parameter is crucial as it dictates where models are saved/loaded and where data files are expected/written by the current implementation of the `predict` endpoint and the flawed `train` endpoint. This suggests that an API instance generated by this function might be tied to a specific model or dataset context defined by this directory, which has implications for managing multiple models or datasets through a single API instance.

---

## `chap_core/assessment/prediction_evaluator.py`

**Purpose**:
This module defines core components and workflows for evaluating the performance of predictive models, particularly time series forecasting models. It establishes `Predictor` and `Estimator` protocols, provides a `backtest` function for robust model evaluation over multiple time windows, and includes a comprehensive `evaluate_model` function that integrates with GluonTS for metrics calculation and reporting (including plotting forecasts).

**Key Components & Functionality**:

- **Protocols (`Predictor`, `Estimator`)**:
  - `Predictor`: Defines an interface for a trained model, requiring a `predict` method that takes historical data and future covariates to produce forecast samples (`DataSet[Samples]`). Uses a `FeatureType` TypeVar and a `without_disease` helper in type hints to signify that future data should exclude the target variable.
  - `Estimator`: Defines an interface for a model estimator, requiring a `train` method that takes a `DataSet` and returns a `Predictor`.
- **`backtest(...)`**:
  - Performs backtesting by training an `Estimator` once on an initial training set and then generating predictions for multiple subsequent test windows.
  - Uses `train_test_generator` (from `dataset_splitting`) to create the training set and test window iterator.
  - For each test window, it calls `predictor.predict()` and then merges the resulting `Samples` with the `future_truth` data (which includes the actual target values) into a `DataSet[SamplesWithTruth]`.
  - Yields these `DataSet[SamplesWithTruth]` objects, allowing for iterative processing of backtest results.
- **`evaluate_model(...)`**:
  - A comprehensive evaluation function. It trains an `Estimator` once.
  - It then generates forecasts for multiple test windows (similar to `backtest` but collects all forecasts).
  - Prepares truth data and forecasts in a format suitable for `gluonts.evaluation.Evaluator`. This involves using `ForecastAdaptor.from_samples` to convert CHAP-core `Samples` into GluonTS `Forecast` objects.
  - Optionally generates a PDF report with plots of forecasts against truth data using `plot_forecasts`.
  - Uses `gluonts.evaluation.Evaluator` to calculate aggregated metrics and per-item metrics.
  - Returns these metrics as pandas DataFrames.
- **`create_multiloc_timeseries(...)`**: A helper function to convert a dictionary of pandas DataFrames (truth data per location) into CHAP-core's `MultiLocationDiseaseTimeSeries` format, which is used by some evaluators.
- **`_get_forecast_generators(...)`** (private helper): Iterates through test windows, generates predictions, adapts them to GluonTS `Forecast` objects, and collects corresponding truth data slices. Returns lists of forecasts and truth series.
- **`_get_forecast_dict(...)`** (private helper): Generates forecasts for multiple test windows and organizes them into a dictionary mapping location IDs to lists of GluonTS `Forecast` objects. Used by `plot_forecasts`.
- **`plot_forecasts(...)`**: Generates plots of forecasts against truth data for multiple test windows and locations, saving them to a multi-page PDF. It also yields the (forecast, truth_slice) pairs for use in `evaluate_model`.
- **`plot_predictions(...)`**: Plots a single set of predictions (from `DataSet[Samples]`) against truth data, saving to PDF. Useful for visualizing forecasts from `forecast_ahead`.

**Integration**:

- Central to the model assessment workflow in CHAP-core.
- Integrates with `dataset_splitting` for creating evaluation windows.
- Relies on `Estimator` and `Predictor` protocols that model classes are expected to implement.
- Uses `ForecastAdaptor` from `chap_core.data.gluonts_adaptor.dataset` to bridge CHAP-core's `Samples` format with GluonTS `Forecast` objects.
- Leverages `gluonts.evaluation.Evaluator` for metrics calculation.
- Uses `matplotlib` for plotting.
- Interacts with various data types from `chap_core.datatypes` and `chap_core.spatio_temporal_data`.

**Key Concepts**:

- **Protocols**: Defines expected interfaces for models (`Estimator`, `Predictor`).
- **Backtesting**: Implemented via `backtest` and `evaluate_model` using `train_test_generator`.
- **GluonTS Integration**: Uses GluonTS for its robust evaluation metrics and forecast object representation.
- **Visualization**: Provides utilities for plotting forecasts to PDF for visual inspection.

**Error Handling**:

- The `backtest` function includes a `try-except` block for prediction errors within a window and re-raises them as `EvaluationError`.
- Other functions rely on exceptions from underlying calls (e.g., `requests` in plotting if data was remote, though not the case here; `pandas` errors; GluonTS errors).

**Dependencies**:

- `logging`, `collections.defaultdict`, `typing`.
- `pandas`, `gluonts`, `matplotlib`.
- Multiple `chap_core` submodules for data types, dataset splitting, adaptors, etc.

---

## `chap_core/assessment/representations.py`

**Purpose**:
This module defines a set of dataclasses that serve as standardized structures for representing data related to model assessment and evaluation. These include representations for disease observations, error metrics, and forecast samples, often designed to handle data from multiple geographic locations and across time series.

**Key Dataclasses**:

- **Disease Observation Data**:
  - `DiseaseObservation`: Represents a single data point for disease cases at a specific `time_period` (string).
  - `DiseaseTimeSeries`: A list of `DiseaseObservation` objects, forming a time series for one location.
  - `MultiLocationDiseaseTimeSeries`: A dictionary mapping location IDs (strings) to `DiseaseTimeSeries` objects. It includes dict-like methods (`__setitem__`, `__getitem__`, `locations`, `timeseries`, `items`, `values`, `__len__`, `__iter__`).
- **Error Representation Data**:
  - `Error`: Represents a single calculated error `value` (float) for a specific `time_period` (string).
  - `ErrorTimeSeries`: A list of `Error` objects, forming a time series of errors for one location or an aggregated series.
  - `MultiLocationErrorTimeSeries`: A dictionary mapping location IDs (or "Full_region") to `ErrorTimeSeries`. It includes dict-like methods and several utility methods:
    - `num_locations()`: Returns the count of distinct series.
    - `num_timeperiods()`: Returns the count of unique time periods (assumes alignment).
    - `get_the_only_location()`/`get_the_only_timeseries()`: Retrieve data when only one series is expected (e.g., after global aggregation). Raises `ValueError` if not exactly one.
    - `get_all_timeperiods()`: Returns a list of time period strings, asserting consistency across all series.
    - `timeseries_length()`: Returns the common length of observations in series, asserting consistency.
    - `items_grouped_by_timeperiod_str()`: A new method added to restructure data, grouping errors by time period first, then by location, useful for certain aggregations.
- **Forecast Sample Data**:
  - `Samples`: Represents forecast samples (`disease_case_samples`: `List[float]`) for a single `time_period` (string) at one location.
  - `Forecast`: Represents a time series of predictions for a single location, containing a list of `Samples` objects.
  - `MultiLocationForecast`: A dictionary mapping location IDs to `Forecast` objects. Includes dict-like methods.

**Integration**:

- These dataclasses are fundamental data structures used throughout the `chap_core.assessment` package (e.g., by `evaluator.py`, `prediction_evaluator.py`).
- They provide a standardized way to pass around truth data, model forecasts, and calculated errors during the evaluation pipeline.

**Key Concepts**:

- **Dataclasses**: Python's `dataclass` decorator is used for concise class definitions.
- **Structured Data Representation**: Provides clear, typed structures for complex assessment data involving multiple locations and time series.
- **Aggregation Support**: The `Error` related classes and methods in `MultiLocationErrorTimeSeries` are designed to support various levels of error aggregation (per timepoint, per location over time, global).

**Error Handling**:

- Methods in `MultiLocationErrorTimeSeries` like `get_the_only_location` and `get_all_timeperiods` use `ValueError` or `AssertionError` to handle cases of unexpected data structure (e.g., not exactly one location when one is expected, or inconsistent time periods).

**Dependencies**:

- `dataclasses` (for `dataclass`, `field`).
- `typing` (for `Dict`, `List`, `Optional`, `Tuple`, `Iterator`).
- `collections.defaultdict` (used in `items_grouped_by_timeperiod_str`).

---

## `chap_core/data/__init__.py`

**Purpose**:
This `__init__.py` file serves to initialize the `chap_core.data` package. Its primary functions are to make the `data` directory a Python package (allowing modules within it to be imported using the `chap_core.data` namespace) and to define the public API of this package by re-exporting selected classes.

**Key Components**:

- **Imports**:
  - `from ..spatio_temporal_data.temporal_dataclass import DataSet`: Imports the `DataSet` class from a sibling package `spatio_temporal_data`. `DataSet` is likely a core data structure for handling multi-location time series data.
  - `from ..api_types import PeriodObservation`: Imports the `PeriodObservation` Pydantic model from another sibling package `api_types`. This model is often used as a helper for constructing time series data, particularly when dealing with API inputs or outputs.
- **`__all__ = ["DataSet", "PeriodObservation"]`**: This list explicitly defines which names are imported when a client uses `from chap_core.data import *`. It signifies that `DataSet` and `PeriodObservation` are considered part of the public interface of the `chap_core.data` package, providing a convenient, centralized access point for these commonly used types.

**Integration**:

- This file acts as the entry point for the `chap_core.data` package.
- By re-exporting `DataSet` and `PeriodObservation`, it abstracts their original locations, potentially making the `chap_core.data` package's API more stable even if the internal structure of `chap_core` changes.
- The `chap_core.data` package, as described in its new docstring, is intended to house modules related to core data handling, adaptors for external libraries (like the `gluonts_adaptor` sub-package), specific dataset loaders (e.g., `open_dengue.py`), and example dataset utilities (`datasets.py`).

**Functionality**:

- Marks the directory as a package.
- Provides a simplified import path for `DataSet` and `PeriodObservation` (e.g., `from chap_core.data import DataSet`).

**Dependencies**:

- `chap_core.spatio_temporal_data.temporal_dataclass.DataSet`
- `chap_core.api_types.PeriodObservation`

---

## `chap_core/data/adaptors.py`

**Purpose**:
This module acts as a convenient access point for pre-instantiated data adaptors. Its primary role is to simplify the usage of common data adaptors by providing ready-to-use instances.

**Key Components**:

- It imports `DataSetAdaptor` from the submodule `.gluonts_adaptor.dataset`.
- It creates a single, module-level instance of `DataSetAdaptor` named `gluonts`.

**Integration**:

- This `gluonts` instance can be imported by other parts of the CHAP-core system (e.g., `from chap_core.data.adaptors import gluonts`) to perform conversions between CHAP-core's `DataSet` objects and GluonTS-compatible data formats.
- This centralizes the creation of a default `DataSetAdaptor` for GluonTS.

**Functionality**:

- The main functionality is provided by the `DataSetAdaptor` class itself (defined in the `gluonts_adaptor` submodule). This module simply makes an instance of it readily available.

**Dependencies**:

- `.gluonts_adaptor.dataset.DataSetAdaptor`: The class that is instantiated.

---

## `chap_core/database/__init__.py`

**Purpose**:
This `__init__.py` file marks the `chap_core/database/` directory as a Python package. This allows modules within this directory to be imported as part of the `chap_core.database` namespace and enables relative imports among them.

**Key Components**:

- The file was originally empty.
- The updated version includes a module docstring that explains the intended purpose of the `database` package: to handle all database interactions for CHAP-core, including schema definitions (using SQLModel), session management, caching, and other database utilities.
- It also contains commented-out examples showing how key components (like `SessionWrapper`, `get_engine` from `database.py`, or core table models) could be re-exported using an `__all__` list. This would make them directly accessible from the `chap_core.database` namespace, simplifying imports for users of this package.

**Integration**:

- As a standard Python package initializer, it enables the modular organization of database-related code.
- The `database` package, as outlined in its docstring, is central to data persistence in CHAP-core. It likely interacts with many other parts of the application that need to store or retrieve data.

**Functionality**:

- Its primary function is to enable the `database` directory to be treated as a package.
- Currently, it does not define or execute any functional code itself beyond comments and the (commented-out) example re-exports.

**Dependencies**:

- None directly within this file in its current state.
- The commented-out examples suggest potential internal dependencies on other modules within the `chap_core.database` package (e.g., `database.py`, `tables.py`).

---

## `chap_core/database/base_tables.py`

**Purpose**:
This module provides foundational elements for defining database table models using `SQLModel` within the CHAP-core project. Its main contribution is a base class `DBModel` that standardizes Pydantic configurations for all table models, particularly for API interactions.

**Key Components**:

- **`PeriodID = str`**: A type alias for period identifiers, with a comment suggesting expected string formats (e.g., "YYYYMM" or "YYYYWW").
- **`DBModel(SQLModel)` class**:
  - Serves as a common base class for all SQLModel table definitions in the project.
  - Includes a Pydantic `model_config` (using `ConfigDict`) with two key settings:
    - `alias_generator=to_camel`: Automatically generates camelCase aliases for field names (e.g., a Python field `my_field_name` would have a JSON/API alias `myFieldName`).
    - `populate_by_name=True`: Allows Pydantic models to be populated using either the original Python field name or its alias when parsing input data (e.g., from JSON).
  - The primary purpose of this configuration is to ensure consistent JSON serialization/deserialization with camelCase keys, which is a common convention for REST APIs.
  - The docstring clarifies that subclasses intended to be actual database tables must include `table=True` in their definition (e.g., `class MyTable(DBModel, table=True): ...`).
  - Includes a commented-out example of how fields might be defined in a subclass.

**Integration**:

- This module is fundamental for the database layer of CHAP-core if SQLModel is used for ORM and Pydantic for data validation/serialization.
- All other modules defining database tables (e.g., `dataset_tables.py`, `model_spec_tables.py`) are expected to have their table models inherit from `DBModel`.
- The camelCase aliasing directly impacts how data is structured when interacting with the REST API.

**Key Technologies/Patterns**:

- **SQLModel**: Used as the base for ORM and data validation.
- **Pydantic (`ConfigDict`, `alias_generator`)**: Leveraged for configuring data serialization and aliasing behavior.
- **Base Class Pattern**: `DBModel` provides a common foundation for all table models.

**Dependencies**:

- `pydantic` (for `ConfigDict`, `alias_generators.to_camel`).
- `sqlmodel` (for `SQLModel`).

---

## `chap_core/database/database.py`

**Purpose**:
This module is central to database interactions in CHAP-core. It handles the initialization of the database engine (typically PostgreSQL, based on `psycopg2` error handling, or SQLite), manages database sessions via a `SessionWrapper` context manager, and provides high-level functions for creating the database schema and seeding initial data. It also includes methods within `SessionWrapper` for common CRUD-like operations and specific data insertion tasks related to datasets, predictions, and evaluation results.

**Key Components & Functionality**:

- **Global Engine Initialization**:

  - Attempts to create a global SQLAlchemy `engine` based on the `CHAP_DATABASE_URL` environment variable.
  - Includes a retry loop (30 attempts with 1-second delays) to handle initial connection issues, catching `sqlalchemy.exc.OperationalError` and `psycopg2.OperationalError`.
  - Logs connection attempts and success/failure. Raises a `ValueError` if connection ultimately fails.
  - If `CHAP_DATABASE_URL` is not set, it logs a warning, and the `engine` remains `None`.

- **`SessionWrapper` Class**:

  - A context manager (`__enter__`, `__exit__`) for SQLModel/SQLAlchemy sessions.
  - Can use the global `engine` or a `local_engine` passed during initialization. It can also accept an externally managed `session`.
  - **Methods**:
    - `list_all(model_class)`: Retrieves all records for a given SQLModel table.
    - `create_if_not_exists(model_instance, id_name='id')`: Adds a model instance only if one with the same ID doesn't already exist. Refreshes the instance after commit.
    - `add_evaluation_results(evaluation_results, last_train_period, info)`: Creates a `BackTest` record and associated `BackTestForecast` entries from evaluation results. Handles `Iterable[SpatioTemporalDataSet[Samples]]`.
    - `add_predictions(predictions, dataset_id, model_id, name, metadata)`: Adds `Prediction` and `PredictionSamplesEntry` records. Assumes `predictions` is `SpatioTemporalDataSet[chap_core.datatypes.Forecast]`.
    - `add_dataset(dataset_name, orig_dataset, polygons, dataset_type)`: Adds a new `DBDataSet` (database table model for datasets) and its associated `Observation` records. Infers `covariates` from the input `SpatioTemporalDataSet`.
    - `get_dataset(dataset_id, dataclass_type)`: Retrieves a dataset by ID and converts its observations into a `SpatioTemporalDataSet` of the specified `dataclass_type` using `observations_to_dataset`. Handles JSON parsing for polygons.
    - `add_debug()`: Adds a `DebugEntry` with a timestamp, useful for simple database checks.

- **`create_db_and_tables()`**:
  - If the global `engine` is initialized, this function creates all database tables defined by `SQLModel.metadata.create_all(engine)`.
  - Includes retry logic for table creation, similar to engine initialization.
  - After table creation, it calls `seed_with_session_wrapper` (from `model_spec_tables`) to populate the database with initial/default data, using a `SessionWrapper`.
  - Logs warnings if the engine is not set.

**Integration**:

- This module is the primary interface to the database for most of CHAP-core.
- It's used by API endpoints, worker tasks, CLI commands, and any part of the application that needs to persist or retrieve structured data.
- It relies on table definitions from sibling modules: `.tables`, `.model_spec_tables`, `.dataset_tables`, and `.debug`.
- It uses `SpatioTemporalDataSet` from `chap_core.spatio_temporal_data` as the in-memory representation for datasets being added or retrieved, and `observations_to_dataset` for conversion.
- Interacts with `chap_core.time_period.TimePeriod` and Pydantic models from `chap_core.rest_api_src.data_models` (like `BackTestCreate`).

**Key Technologies/Patterns**:

- **SQLModel**: Used for ORM and defining database schemas.
- **SQLAlchemy**: The underlying toolkit for database interaction, used by SQLModel.
- **Environment Variable Configuration**: `CHAP_DATABASE_URL` for database connection string.
- **Context Manager (`SessionWrapper`)**: Ensures database sessions are properly managed and closed.
- **Retry Logic**: Implemented for initial database connection and table creation to handle transient issues, common in containerized environments.
- **Data Seeding**: Provides a mechanism to populate the database with initial data.

**Error Handling**:

- Specific exceptions like `sqlalchemy.exc.OperationalError` and `psycopg2.OperationalError` are caught during engine/table creation.
- `SessionWrapper` methods generally expect a session to be available and may raise `RuntimeError` if not.
- `get_dataset` raises `ValueError` if a dataset ID is not found.
- `add_evaluation_results` and `add_predictions` have internal logic that depends on the structure of input data, and mismatches could lead to errors.

**Dependencies**:

- Standard libraries: `dataclasses`, `datetime`, `json`, `logging`, `os`, `time`, `typing`.
- Third-party: `psycopg2` (implicitly, for PostgreSQL), `sqlalchemy`, `sqlmodel`.
- Internal `chap_core` modules: for table definitions, data types (`Samples`, `TimeSeriesData`), time period utilities, API data models, and spatio-temporal data structures/converters.

---

## `chap_core/plotting/__init__.py`

**Purpose**:
This `__init__.py` file initializes the `chap_core.plotting` package, making it a recognizable Python package. Its primary role is to define the public API of this package by importing and re-exporting key plotting functions from its submodules.

**Key Components**:

- **Imports**: It imports `plot_timeseries_data` and `plot_multiperiod` from the local submodule `.plotting`.
- **`__all__ = ["plot_timeseries_data", "plot_multiperiod"]`**: This list explicitly defines the names that are imported when `from chap_core.plotting import *` is used. It signifies that these two functions are the primary public interface of the `plotting` package.

**Integration**:

- This file serves as the entry point for the `chap_core.plotting` package.
- By re-exporting these functions, it provides a convenient and potentially more stable way for other parts of the CHAP-core system to access common plotting utilities without needing to know their exact submodule location.

**Functionality**:

- Marks the `plotting` directory as a Python package.
- Provides a simplified import path for `plot_timeseries_data` and `plot_multiperiod` (e.g., `from chap_core.plotting import plot_timeseries_data`).

**Dependencies**:

- `.plotting.plot_timeseries_data`
- `.plotting.plot_multiperiod`

---

## `chap_core/plotting/plotting.py`

**Purpose**:
This module provides core plotting functions for visualizing various time series datasets within the CHAP-core project, primarily utilizing the Plotly library for interactive charts.

**Key Functions**:

- **`plot_timeseries_data(data: ClimateHealthTimeSeries) -> Figure`**:
  - Takes a `ClimateHealthTimeSeries` object.
  - Converts it to a pandas DataFrame and melts it to a long format.
  - Creates a Plotly line plot with faceted subplots, where each variable in the input data is displayed in a separate subplot against time.
  - Y-axes of subplots have independent scales and are titled with the variable name.
  - Handles empty input data by returning an empty figure.
- **`plot_multiperiod(climate_data: ClimateData, health_data: HealthData, head: Optional[int] = None, climate_var: str = "mean_temperature", health_var: str = "disease_cases") -> Figure`**:
  - Plots a specified climate variable (default "mean_temperature") and a health variable (default "disease_cases") on the same chart with dual y-axes.
  - Takes `ClimateData` and `HealthData` objects.
  - Optionally plots only the first `head` data points from climate data.
  - Aligns the health data to the time range of the (potentially truncated) climate data.
  - Handles cases where data might be empty or alignment cut-off is not found, logging warnings.
  - Converts `time_period` columns to timestamps for plotting.
  - Uses `plotly.subplots.make_subplots` for dual y-axes.
  - Returns a Plotly `Figure` object.
  - Raises `ValueError` for empty input or missing specified variables.

**Integration**:

- This module is a key utility for data visualization in CHAP-core.
- It consumes `ClimateHealthTimeSeries`, `ClimateData`, and `HealthData` objects from `chap_core.datatypes`.
- The generated Plotly figures can be used in various contexts (notebooks, web apps).

**Key Technologies/Patterns**:

- **Plotly Express (`px`) and Plotly Graph Objects (`Figure`, `make_subplots`)**: Used for creating interactive charts.
- **Pandas**: For data manipulation (`topandas()`, `melt()`, `head()`) prior to plotting.
- **Dual Y-Axes Plotting**: Demonstrated in `plot_multiperiod`.
- **Faceted Plots**: Used in `plot_timeseries_data`.

**Dependencies**:

- `logging`, `typing` (`Optional`).
- `pandas` as `pd`.
- `plotly.express` as `px`, `plotly.graph_objs.Figure`, `plotly.subplots.make_subplots`.
- `chap_core.datatypes` (for `ClimateHealthTimeSeries`, `ClimateData`, `HealthData`).

---

## `chap_core/plotting/prediction_plot.py`

**Purpose**:
This module provides specialized functions for visualizing model predictions, focusing on comparing forecasted distributions (samples, quantiles) against actual observed data. It uses both Matplotlib for one type of plot and Plotly for others, offering different ways to inspect prediction quality.

**Key Components & Functionality**:

- **Type Definitions**:
  - `FeatureType = TypeVar("FeatureType", bound=TimeSeriesData, covariant=True)`: A generic type variable for features.
  - `without_disease(t: FeatureType) -> FeatureType`: An identity function used as a type hint marker to indicate a `DataSet` should exclude the target variable.
- **Protocols**:
  - `Predictor(Protocol[FeatureType])`: Defines an interface for a trained model, requiring a `predict(historic_data, future_data) -> DataSet[Samples]` method.
  - `Estimator(Protocol)`: Defines an interface for a model estimator, requiring a `train(data) -> Predictor` method.
- **Plotting Functions**:
  - `prediction_plot(...) -> plt.Figure`: Plots multiple raw predicted sample paths (from `IsSampler`) against true data using Matplotlib.
  - `forecast_plot(...) -> go.Figure`: Generates a Plotly figure showing 0.1, 0.5 (median), and 0.9 quantiles of prediction samples (from `IsSampler`) against true data. It internally calls `plot_forecast`.
  - `plot_forecast_from_summaries(...) -> go.Figure`: Plots forecasts derived from `SummaryStatistics` objects (which likely contain pre-calculated quantiles/median) against true data using Plotly. Handles single or list of `SummaryStatistics`.
  - `plot_forecast(quantiles, true_data, x_pred=None) -> go.Figure`: A core Plotly function that takes pre-calculated quantiles (as a NumPy array) and true data to generate a forecast plot.
  - `plot_forecasts_from_data_frame(...) -> go.Figure`: A generic Plotly plotting function that takes prediction and truth data as pandas DataFrames. It uses `add_prediction_lines` to draw the forecast bands and median.
  - `add_prediction_lines(fig, prediction_df, transform, true_df, show_legend=True) -> None`: A helper function to add prediction quantile bands (10th-90th) and a median line to an existing Plotly figure. It also attempts to connect the forecast to the last true data point and adds a vertical line to mark the forecast start.

**Integration**:

- This module is crucial for visual assessment of model predictions.
- It consumes `HealthData`, `ClimateData`, `SummaryStatistics`, `Samples`, and `DataSet` objects from `chap_core.datatypes` and `chap_core.spatio_temporal_data`.
- It interacts with models conforming to the `IsSampler`, `Predictor`, and `Estimator` protocols.
- Uses `ForecastAdaptor` (from `chap_core.data.gluonts_adaptor.dataset`) in `evaluate_model` (which calls `plot_forecasts`) to convert CHAP-core samples to GluonTS `Forecast` objects for plotting with GluonTS's native plot methods.

**Key Technologies/Patterns**:

- **Matplotlib & Plotly**: Uses both libraries for different visualization styles.
- **Protocol-Based Interfaces**: Defines `Predictor` and `Estimator` protocols for model interaction.
- **Quantile-Based Visualization**: Several functions focus on plotting prediction intervals (quantiles) to represent forecast uncertainty.

**Error Handling**:

- Includes `ValueError` for invalid inputs (e.g., non-positive `n_samples`, mismatched data lengths).
- `add_prediction_lines` has logic to handle cases where the first prediction period might not align perfectly with the true data.

**Dependencies**:

- `logging`, `typing` (various types including `Callable`, `Protocol`, `TypeVar`).
- `numpy` as `np`.
- `pandas` as `pd`.
- `matplotlib.pyplot` as `plt`.
- `plotly.graph_objects` as `go`.
- `chap_core.datatypes`, `chap_core.predictor.protocol`, `chap_core.spatio_temporal_data.temporal_dataclass`.
- `chap_core.assessment.evaluator.EvaluationError`.

---
