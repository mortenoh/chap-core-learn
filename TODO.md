# CHAP-Core Improvement Suggestions (TODO)

This document compiles improvement suggestions generated during a code review and documentation pass. Each file processed has five suggestions listed, aimed at enhancing clarity, robustness, and maintainability.

---

## `chap_core/__init__.py`

1.  **Descriptive Module Docstring**: Add a more descriptive module-level docstring explaining the purpose and high-level overview of the `chap-core` package.
2.  **Structured Metadata**: Consider using `importlib.metadata` (Python 3.8+) or a similar standard mechanism for managing package metadata (`__author__`, `__email__`, `__version__`) instead of direct string assignments.
3.  **Review `__all__` List**: Ensure the `__all__` list accurately reflects the intended public API of the `chap_core` package. Add or remove names as necessary for clarity.
4.  **Documentation Links**: Include a link to the project's main documentation or repository within the module docstring for easy reference.
5.  **Type Hinting for Metadata**: Add type hints for `__author__`, `__email__`, and `__version__` (e.g., `__version__: str = "1.0.7"`) for better static analysis and code clarity.

---

## `chap_core/alarms.py`

1.  **Pydantic Field Validation**: In `OutbreakParameters`, add Pydantic `Field` validators for `endemic_factor` (e.g., must be positive) and `probability_threshold` (e.g., must be between 0 and 1).
2.  **Detailed Error Logging**: Enhance the `outbreak_prediction` function with more detailed logging or a more specific custom exception when `ValueError` is raised for invalid `case_samples`.
3.  **Alternative Outbreak Detection Methods**: The module docstring or comments could briefly mention or suggest exploring alternative statistical methods for outbreak detection beyond the current threshold-based approach for potentially more robust predictions.
4.  **Unit Test Edge Cases**: Ensure unit tests for `outbreak_prediction` cover edge cases such as empty `case_samples`, all samples being equal, all samples below/above the calculated threshold, and non-finite values.
5.  **Minimum Sample Size**: Consider if `case_samples` in `outbreak_prediction` should enforce a minimum length for statistical significance in the proportion calculation, and document this requirement.

---

## `chap_core/api_types.py`

1.  **Pydantic Field Descriptions**: Add `description` attributes to all Pydantic `Field` definitions within the models (e.g., for `pe`, `ou` in `DataElement`, `featureId` in `DataList`) to provide more context directly in the schema.
2.  **Enum for Limited Choices**: For fields with a known, limited set of valid string values (e.g., `estimator_id` in `RequestV2` if there's a predefined list of estimators), consider using Python `Enum` types for better type safety and clarity.
3.  **Usage Examples in Docstrings**: For more complex Pydantic models like `RequestV1`, `RequestV2`, or `FeatureCollectionModel`, include examples of their expected JSON structure or usage in their class docstrings.
4.  **Naming Consistency Review**: Review field naming conventions across different models (e.g., `featureId` vs. `dataElement` vs. `orgUnit`). While some might be due to external API compatibility, strive for internal consistency where possible, or document the reasons for differences.
5.  **Model Inheritance/Aliasing**: For very similar models like `DataElement` and `DataElementV2`, evaluate if using a common base class or type aliasing could reduce code duplication if their structures are closely related and expected to evolve together.

---

## `chap_core/api.py`

1.  **Clarify `DummyControl` and `AreaPolygons`**: These classes are placeholders. Their docstrings should clearly state their purpose (even if just as stubs for future features) or they should be removed if obsolete. If `AreaPolygons` is intended for actual use, its attributes (e.g., `shape_file`) need more definition.
2.  **Robust `extract_disease_name`**: The `extract_disease_name` function's assumption about `health_data["rows"][0][0]` is brittle. Consider a more robust method for identifying the disease name, such as requiring it as explicit metadata or a dedicated field in the input structure.
3.  **Structured Return from `train_with_validation` / `forecast`**: These functions return `list[Figure]`. For programmatic use, consider returning structured data (e.g., a dictionary of metrics, prediction data objects) in addition to, or instead of, just plot figures.
4.  **Comprehensive Error Handling**: Improve error handling for operations like dataset loading (invalid `dataset_name`), model instantiation (`get_model`, `get_model_from_directory_or_github_url`), and model training/forecasting failures. Raise specific, informative exceptions.
5.  **Configuration of Model Parameters**: In `train_with_validation`, parameters like `n_iter=32000` and `n_samples=100` are hardcoded. Make these configurable via function arguments or a configuration object for greater flexibility. Similarly, `logging.basicConfig` in `forecast` should ideally be handled at the application entry point.

---

## `chap_core/chap_cli.py`

1.  **Reinforce Deprecation in Commands**: Add prominent warnings (e.g., using `logger.warning` or `click.secho` with `fg='yellow'`) at the beginning of each CLI command function (`harmonize`, `evaluate`, `predict`) reiterating that the `chap-cli` tool is deprecated.
2.  **Guidance on Alternatives**: In the module docstring, provide clear pointers or references to the current, non-deprecated tools or API endpoints that supersede the functionality of this CLI.
3.  **Specific Error Handling for File/Model Operations**: Enhance error handling for file inputs (e.g., `input_filename` not found, permission issues in `harmonize`) and for model loading/registry lookups (e.g., `model_id` not found in `evaluate` and `predict`) using specific exceptions.
4.  **Standardize Logging**: Ensure consistent logging setup. While `logging.basicConfig` is called in `main()`, individual commands also use `logger.info/error`. Consolidate configuration or ensure all commands use the logger appropriately.
5.  **Input Parameter Validation**: Add more explicit validation for input parameters beyond what `cyclopts` might provide, especially for file paths (check existence, readability) and choices like `model_id`.

---

## `chap_core/cli.py`

1.  **Consistent Logging Initialization**: Ensure `initialize_logging` is called consistently at the start of all relevant CLI command functions (or once globally in `main`) for uniform log handling and formatting.
2.  **Comprehensive Error Handling**: Expand error handling within each command for file I/O (e.g., `dataset_csv` not found, unreadable `polygons_json`), model loading issues (invalid `model_name` or path), and failures during core operations (e.g., `evaluate_model`, `forecast`, `backtest`). Return non-zero exit codes on error.
3.  **Robust Parameter Validation**: Implement more thorough validation for command-line arguments beyond basic type checking by `cyclopts`. For example, check that `prediction_length` and `n_splits` are positive, or that `dataset_name` and `dataset_csv` are not specified simultaneously if mutually exclusive.
4.  **Refactor `evaluate` Command Complexity**: The model loading and configuration loop within the `evaluate` command (handling `model_list` and `model_configuration_yaml_list`) is complex. Review for potential refactoring to improve clarity and reduce redundancy, perhaps by moving some logic into helper functions.
5.  **Clarify `AreaPolygons` and `main_function`**: The `AreaPolygons` dataclass and `main_function` are placeholders. Their docstrings should clearly state their intended purpose or status (e.g., "Placeholder for future schema integration," "Reserved for future programmatic entry point"). If obsolete, they should be removed.

---

## `chap_core/climate_predictor.py`

1.  **Specific Type Hints**: Enhance type hints, especially for `self._models` in `MonthlyClimatePredictor` (e.g., `Dict[str, Dict[str, RegressorMixin]]`) and ensure all methods have explicit return type hints (e.g., `-> DataSet[SimpleClimateData]` for `predict`).
2.  **Robust Error Handling in Training**: Add more specific error handling in `MonthlyClimatePredictor.train`, such as for empty `train_data`, insufficient data points for regression, or if `data_item.time_period` is empty. Raise informative exceptions.
3.  **Complete `SeasonalForecastFetcher`**: The `SeasonalForecastFetcher.get_future_weather` method is a stub and raises `NotImplementedError`. If this class is intended for use, its implementation for loading and providing actual seasonal forecasts needs to be completed.
4.  **Configurable Regression Models**: `MonthlyClimatePredictor` and `WeeklyClimatePredictor` hardcode `sklearn.linear_model.LinearRegression`. Consider allowing the type of regression model or its parameters to be passed during initialization for greater flexibility.
5.  **Safe Slicing in `FetcherNd`**: The slicing logic in `FetcherNd.get_future_weather` (`getattr(data, field.name)[-len(period_range) :]`) could fail or produce unexpected results if `len(period_range)` exceeds the length of available historical data for a field. Implement checks or alternative logic (e.g., repeat the last known value, tile available data, or raise an error).

---

## `chap_core/datatypes.py`

1.  **Standardize `to_pandas` vs `topandas`**: The `TimeSeriesData` class has both `topandas` and an alias `to_pandas = topandas`. Standardize to one, preferably `to_pandas`, for consistency with the pandas library itself, and update all internal calls.
2.  **Specific Error Handling in `from_pandas` / `from_csv`**: Refine error handling in `TimeSeriesData.from_pandas`. The current broad `except Exception:` for time period parsing should be made more specific (e.g., catch `KeyError` for missing 'time_period' column, `ValueError` for unparseable periods) to provide better diagnostics.
3.  **Clarify `BNPDataClass` Implications**: For classes decorated with `tsdataclass` (which uses `bionumpy.bnpdataclass`), their docstrings should briefly mention key implications, such as data being stored internally as numpy arrays and potential performance characteristics related to bionumpy.
4.  **Role of `ClimateHealthTimeSeriesModel` (Pydantic)**: The Pydantic model `ClimateHealthTimeSeriesModel` seems to duplicate some structure from the dataclasses. Its docstring should clearly explain its specific role (e.g., for API request/response validation, serialization for external systems) and how it differs from or complements the `tsdataclass` versions.
5.  **Review `TimeSeriesData.__setstate__`**: The current `__setstate__` uses `self.__dict__.update(state)`. If `__getstate__` (which uses `self.todict()`) serializes `PeriodRange` in a special way, `__setstate__` might need to use `cls.from_dict(state)` logic to correctly reconstruct `PeriodRange` objects during unpickling, especially if bnpdataclass default pickling isn't sufficient.

---

## `chap_core/docker_helper_functions.py`

1.  **Parse Docker Build Response in `docker_image_from_fo`**: Instead of just printing the stream, parse the Docker build response to explicitly check for build success or failure. Raise `docker.errors.BuildError` with detailed error messages if the build fails.
2.  **Command Injection Warning for `run_command_through_docker_container`**: Add a more prominent warning in the docstring for `run_command_through_docker_container` regarding the `command` parameter. If this string can come from untrusted input, it's a security risk. Advise on sanitization or using list-form commands if the Docker SDK supports it for `client.containers.run`.
3.  **Explicit Docker Client Management**: While `docker.from_env()` is convenient, for applications making many Docker calls or in long-running services, consider explicitly creating and closing the client (e.g., `client = docker.DockerClient(...)`; `client.close()`) using a context manager or try/finally block to ensure resources are released.
4.  **Configurable Mount Point in Container**: The `working_directory` is mounted to a fixed `/home/run`. Document why this path is chosen (e.g., common user in base images, permissions). Consider making the internal mount point configurable if different images have different user/directory structures.
5.  **Structured Output for `run_command_through_docker_container`**: The function returns combined stdout/stderr. For better programmatic use, return a structured object (e.g., a dataclass or tuple) containing `stdout`, `stderr`, and `exit_code` separately.

---

## `chap_core/exceptions.py`

1.  **Contextual Information in Exceptions**: Enhance custom exceptions (e.g., `ModelFailedException`, `InvalidModelException`) to accept and store more context about the error. For example, `InvalidModelException` could store the model name or path that was invalid. This can be done by adding parameters to their `__init__` methods and storing them as attributes.
2.  **Granular Exception Hierarchy**: For broad exceptions like `ModelFailedException` or `ModelConfigurationException`, consider if more specific subclasses would be beneficial for finer-grained error handling downstream (e.g., `ModelTrainingFailedException`, `ModelPredictionFailedException` inheriting from `ModelFailedException`).
3.  **Error Codes or Documentation Links**: If the application uses specific error codes or has detailed documentation for troubleshooting certain exceptions, consider adding these codes as attributes to the exception classes or linking to the documentation in their docstrings.
4.  **Review Exception Usage**: Ensure that these custom exceptions are consistently used throughout the `chap-core` codebase where appropriate, instead of generic `Exception` or `ValueError` when a more specific `ChapCoreException` subclass would be more informative.
5.  **Consistency in Exception Messages**: Establish a consistent style for exception messages, perhaps including what operation failed, why, and any relevant parameters, to aid debugging.

---

## `chap_core/geojson.py`

1.  **Error Handling for GeoPandas Operations**: Functions `geojson_to_shape` and `geojson_to_graph` (and `NeighbourGraph.from_geojson_file`) use `gpd.read_file()` and other GeoPandas operations. Wrap these in more specific `try-except` blocks to catch potential GeoPandas/Fiona errors (e.g., `gpd.io.file.DriverError`, `fiona.errors.DriverError`) and raise custom `GeoJSONProcessingError` with more context.
2.  **Configurable `id_column` in `NeighbourGraph`**: The `NeighbourGraph.from_geojson_file` method attempts to find alternative ID columns if the specified `id_column` (default "id") is not found. This is good, but the `NeighbourGraph.__init__` still defaults to "id". Ensure consistency or make `id_column` a more central, configurable attribute of the class, clearly documented.
3.  **Standardize Graph File Format**: The custom graph file format in `NeighbourGraph.to_graph_file` is specific. For broader interoperability, consider adding an option to export the graph to standard formats like Adjacency List (CSV), GML, GraphML, or JSON Graph Format.
4.  **Type Hint for `IO` in GeoJSON Functions**: Clarify the type hint for `geojson_filename: Union[str, TextIO]` in `geojson_to_graph`. Since GeoJSON is text, `IO[str]` (which `TextIO` is an alias for) is correct. Ensure this is consistently applied if other functions accept file-like objects for GeoJSON.
5.  **`LocationMapping` Exception Type**: In `LocationMapping.name_to_index` and `index_to_name`, change `KeyError` to a more specific custom exception (e.g., `LocationNameNotFoundError`, `LocationIndexNotFoundError` inheriting from `KeyError` or `ChapCoreException`) for more targeted error handling by callers.

---

## `chap_core/geometry.py`

1.  **GADM URL Configuration**: The `GADM_DATA_URL_TEMPLATE` is hardcoded. Make this a configurable constant at the module level or load it from an external configuration. Document the GADM version it points to.
2.  **Error Handling in `pooch.retrieve`**: The `get_data_file` function uses `pooch.retrieve`. Enhance error handling to catch specific `pooch` exceptions (e.g., for download failures, hash mismatches if `known_hash` were used) and wrap them in a custom application exception if needed.
3.  **Clarify `PFeatureModel` vs. `DFeatureModel`**: The module defines `PFeatureModel` (with open `properties`) and imports `DFeatureModel` (from `api_types`, likely with stricter properties). The module docstring or class docstrings should clearly explain when and why each model is used (e.g., `PFeatureModel` for parsing raw GADM data, `DFeatureModel` for internal standardized representation).
4.  **Robustness of `add_id`**: The `add_id` function assumes `feature.properties[f"NAME_{admin_level}"]` exists. This should use `.get()` or a `try-except KeyError` to handle missing name properties gracefully, perhaps logging a warning or raising a specific error.
5.  **`Polygons._add_ids` Error Handling**: The `try-except Exception` in `Polygons._add_ids` is too broad. It should catch more specific exceptions (e.g., `KeyError` if `id_property` is missing and `feature.id` is `None`) and provide more context. Clarify behavior if an ID cannot be assigned.

---

## `chap_core/geoutils.py`

1.  **Type Hint for `collection` in `inspect_feature_collection`**: The `collection` parameter in `inspect_feature_collection` should be explicitly typed, likely as `FeatureCollectionModel` (from `chap_core.api_types` or a local Pydantic model if different).
2.  **Handle "GeometryCollection" in `feature_bbox`**: The `feature_bbox` function currently raises a `ValueError` for unhandled geometry types. Add explicit handling for "GeometryCollection" by iterating through its `geometries`, calculating their individual bounding boxes, and then computing the overall extent.
3.  **Document Optional Dependencies for `render`**: The `render` function imports `geopandas`, `matplotlib.pyplot`, and `PIL.Image`. Clearly document these as optional dependencies required only for this visualization utility, perhaps suggesting how to install them (e.g., `pip install chap-core[plotting]`).
4.  **`simplify_topology` Advanced Parameters**: The comments in `simplify_topology` about `simplify_algorithm`, `simplify_with`, etc., are useful. If these are not intended for current use or are remnants of older `topojson` versions, remove them to avoid confusion. If they represent valid tuning options, briefly explain their effect or link to relevant `topojson` documentation.
5.  **Error Handling for Invalid Geometries**: Functions like `buffer_feature` and `simplify_topology` that perform geometric operations using `shapely` or `topojson` should include `try-except` blocks to catch potential errors arising from invalid or unexpected geometry inputs (e.g., `shapely.errors.TopologicalError`).

---

## `chap_core/internal_state.py`

1.  **Clarify `Control._controls` Dictionary Structure**: The `Control.__init__` docstring should detail the expected structure of the `controls` dictionary argument: what types of keys (status strings) and values (control objects conforming to a certain interface, e.g., having `cancel`, `get_status`, `get_progress` methods) it expects.
2.  **Context for `asyncio.CancelledError`**: The use of `asyncio.CancelledError` in `Control.set_status` implies an asynchronous context. The class docstring for `Control` should clarify if it's primarily designed for `asyncio`-based applications. If not, consider defining and using a custom `OperationCancelledError` for broader applicability.
3.  **Specific Type Hint for `Control._controls` Values**: Instead of `Dict[str, Any]`, if the control objects passed in the `controls` dictionary are expected to adhere to a specific interface (protocol), define that protocol and use it in the type hint (e.g., `Dict[str, ControlInterfaceProtocol]`).
4.  **`InternalState.current_data` Specificity**: The `current_data` field in `InternalState` is typed as `Dict[str, Any]`. If this data typically follows a more specific structure (e.g., a Pydantic model, a specific set of keys), using a `TypedDict` or a Pydantic model for this field would improve clarity and type safety.
5.  **Thread Safety/Concurrency**: If `Control` or `InternalState` instances are intended for concurrent access (e.g., from multiple threads or asyncio tasks modifying status or cancellation flags), document whether they are thread-safe or if external locking mechanisms are required by the user. If thread safety is intended, implement necessary synchronization (e.g., `threading.Lock`).

---

## `chap_core/log_config.py`

1.  **Logger Specificity**: Configure a logger specific to `chap_core` (e.g., `logging.getLogger('chap_core')`) in `initialize_logging` instead of the root logger. Modifying the root logger is generally discouraged for libraries, as it can interfere with the application's own logging setup.
2.  **Handler Idempotency**: Ensure `initialize_logging` is idempotent, especially regarding `FileHandler`. If called multiple times with the same `log_file`, it should not add duplicate handlers, which would cause log messages to be repeated in the file. Check if a handler for that file already exists before adding.
3.  **Robust File Operation Error Handling**: Wrap file system operations (directory creation `log_file_p.parent.mkdir`, file touching `log_file_p.touch()`, `os.chmod`, and file reading in `get_logs`) in `try-except` blocks to handle potential `IOError` or `OSError` exceptions (e.g., permission denied) more gracefully.
4.  **Consistent Use of Logging Framework**: Replace `print()` statements used for status messages within `initialize_logging` (if any remain after refactoring) with appropriate `logger` calls (e.g., `logger.info`, `logger.warning`) for consistent log output once the logger is configured.
5.  **Global State Management (`_global_log_file`)**: The global variable `_global_log_file` can make testing and concurrent usage complex. Consider alternatives: `initialize_logging` could return the log file path, or logging configuration could be managed by a dedicated configuration object passed around or accessed via a context.

---

## `chap_core/model_spec.py`

1.  **`ParameterSpec` Integration with `ModelSpec`**: Clarify how `ParameterSpec` and its subclasses (e.g., `EwarsParamSpec`) are intended to be used with `ModelSpec.parameters`. Currently, `ModelSpec.parameters` is a generic `Dict[str, Any]` and is initialized as empty by `model_spec_from_yaml` and `model_spec_from_model`. If specific parameter structures are expected, `ModelSpec.parameters` should be typed more specifically (e.g., `Union[ParameterSpec, Dict[str, Any]]`), and the creation functions updated to populate it accordingly.
2.  **Error Handling in `model_spec_from_yaml`**: Implement more robust error handling for file operations (`FileNotFoundError`), YAML parsing (`yaml.YAMLError`), and missing or malformed keys (e.g., `KeyError` for "name", type checks for "adapters") in the YAML data. Raise `ModelConfigurationException` with specific messages.
3.  **Clarity on Feature Exclusion Rationale**: Document the rationale behind the `_non_feature_names` set within the docstrings of functions that use it (e.g., `_get_feature_names`, `model_spec_from_yaml`), explaining why these specific names (like "disease_cases", "time_period") are typically excluded as model input features.
4.  **Robustness of `get_dataclass`**: The `get_dataclass` function relies on specific introspection patterns (`inspect.get_annotations`, `__args__[0]`). This can be brittle. Add more comprehensive error handling (e.g., for `IndexError` if `__args__` is empty, `AttributeError` if not a generic type, `TypeError` if annotations are not as expected) and clearly document the assumptions about the `model_class.train` method's signature.
5.  **Default Values in `ModelSpec`**: Review default values in `ModelSpec` (e.g., `description`, `author`, `targets`). Ensure these defaults are sensible or consider making more fields required if they are essential for all model specifications.

---

## `chap_core/pandas_adaptors.py`

1.  **Implement Full Functionality for `get_time_period`**: Complete the `get_time_period` function to correctly handle daily (`day_name`) and weekly (`week_name`) period constructions as indicated by its parameters, instead of raising `NotImplementedError`. This includes defining how year/week or year/month/day combinations are converted to `pd.Period` objects with appropriate frequencies.
2.  **Refined Error Handling and Parameter Validation**:
    - Replace `NotImplementedError` with `ValueError` if mutually exclusive parameters are provided (e.g., both `month_name` and `week_name`).
    - Validate that provided column names (`year_name`, `month_name`, etc.) actually exist in the input DataFrame `df` before attempting to access them, raising `KeyError` if not.
    - Handle potential `ValueError` or `TypeError` during `pd.Period` creation if data in columns is not in the expected format (e.g., non-numeric years/months).
3.  **Type Hinting**: Ensure all parameters (`df`, `year_name`, etc.) and the return type (`List[pd.Period]`) have accurate type hints.
4.  **Flexibility in Period String Construction**: For monthly data, the period string is `f"{year}-{month}"`. Document the expected format of year and month columns (e.g., integer years, 1-12 for months). Consider if more flexibility (e.g., handling string months, different date part separators) is needed.
5.  **Support for Different Period Frequencies**: When implementing daily and weekly support, ensure the `freq` parameter for `pd.Period` is correctly set (e.g., "D" for daily, "W-SUN" or "W-MON" etc. for weekly, depending on desired convention).

---

## `chap_core/rbased_docker.py`

1.  **Specific Docker Exception Handling**: In `create_image`, catch more specific exceptions from `client.images.build` (e.g., `docker.errors.BuildError`, `docker.errors.APIError`) instead of a generic `Exception`. This allows for more targeted error messages and potentially different retry/handling logic.
2.  **Explicit Temporary Directory and Dockerfile Cleanup**: While `tempfile.TemporaryDirectory` handles cleanup on exit, ensure the temporary Dockerfile itself is explicitly closed before `client.images.build` is called with the directory path, especially on Windows where open files can cause issues. The `with open(...)` handles this for the file object, but ensure the directory context manager correctly cleans up if `build` raises an error.
3.  **Base Image Versioning**: The Dockerfile uses `FROM rocker/r-base:latest`. Using `:latest` can lead to non-reproducible builds if the base image changes. Consider pinning to a specific version of `rocker/r-base` (e.g., `rocker/r-base:4.3.2`) for more stable and predictable environments, or make the base image tag configurable.
4.  **Security of R Package Names**: If the `r_packages` list can be influenced by untrusted external input, there's a minor risk if package names could be crafted to inject commands into the `RUN R -e "install.packages(...)"` line. While `install.packages` itself is generally robust with package names, this should be noted. Parameterizing the R command more safely could be an option if this is a concern.
5.  **Build Context Optimization**: The `client.images.build(path=temp_dir_name, ...)` sends the entire `temp_dir_name` as build context. Since only the `Dockerfile` is needed, creating a minimal context (e.g., using `fileobj` parameter of `client.api.build` with the Dockerfile content directly, as was commented out) can be more efficient, especially if `temp_dir_name` could inadvertently contain other files.

---

## `chap_core/training_control.py`

1.  **Safe Division in `get_progress`**: In `TrainingControl.get_progress`, add an explicit check for `self._total_samples == 0` (in addition to `None`) before division to prevent `ZeroDivisionError`. Return a sensible default like 0.0 in such cases, or 1.0 if `_n_finished` is also 0 and `_total_samples` is 0 (implying an empty task completed).
2.  **Complete Type Hinting**: Ensure all method parameters (e.g., `total_samples: int`, `n_sampled: int` in `TrainingControl`) and return values (e.g., `-> None` for setters, `-> bool` for `is_cancelled`) have explicit type hints.
3.  **Logging in `PrintingTrainingControl`**: The `PrintingTrainingControl` uses `print()`. Replace these with calls to the `logging` module (e.g., `logger.info()`). This allows applications using this class to control message verbosity and output streams through standard logging configuration. The `print_to_logger` flag is a good step in this direction.
4.  **Clarity on `asyncio.CancelledError`**: The docstrings should clearly state that `asyncio.CancelledError` is used for cancellation. If these control objects are not exclusively for `asyncio` contexts, explain why this specific error is chosen or consider defining a custom `OperationCancelledError` for broader applicability and to avoid confusion.
5.  **Idempotency of `cancel()`**: Clarify if `cancel()` is idempotent (i.g., can be called multiple times without adverse effects). Currently, it sets `_is_cancelled = True`, which is idempotent. If the sub-control's `cancel()` method is not idempotent, this could be an issue.

---

## `chap_core/util.py`

1.  **Robustness of `interpolate_nans` for Edge NaNs**: Document that `np.interp` (used in `interpolate_nans`) does not extrapolate. NaNs at the very beginning or end of the array will remain NaNs if there are no valid data points on both sides. Consider adding an option or strategy for handling leading/trailing NaNs (e.g., fill with first/last valid value, or a specified constant).
2.  **Refined `redis_available` Error Handling**: In `redis_available`, when catching the generic `Exception as e`, log the specific type and message of `e` (e.g., `logger.error(f"Unexpected error checking Redis: {type(e).__name__}: {e}")`) before returning `False` or re-raising. This aids debugging for unexpected Redis client issues.
3.  **Clarity on Availability Check Scope**: The docstrings for `conda_available`, `docker_available`, and `pyenv_available` should explicitly state that they check for the presence of command-line executables in the system's PATH, distinguishing them from checks for Python library importability (which `redis_available` does).
4.  **Return Value of `nan_helper`**: The second element returned by `nan_helper` is `lambda z: z.nonzero()[0]`. The docstring explains its use but could be clearer that `x_func(~nans)` provides the indices of non-NaN values, and `x_func(nans)` provides the indices where NaNs exist (which are then used as the x-coordinates for interpolation).
5.  **Consider `ConnectionRefusedError` for `redis_available`**: In addition to `redis.exceptions.ConnectionError`, Python's built-in `ConnectionRefusedError` (a subclass of `OSError`) might also be raised if the Redis server is down but the port is active but refusing connections. Consider catching this as well for a more robust check.

---

## `chap_core/validators.py`

1.  **Usage of `estimator` Parameter**: The `estimator` parameter in `validate_training_data` is currently unused. If it's intended for future use (e.g., checking if dataset features match estimator's requirements), this should be implemented or the TODO for it should be more explicit. If not planned, remove the parameter to simplify the function signature.
2.  **Specific Exception Types**: Replace `TypeError` for incorrect `dataset` type with a more specific check at the beginning. The `ValueError` for insufficient time coverage should be changed to the custom `InvalidTrainingDataError` defined in the module for consistency and better error categorization.
3.  **Comprehensive Validation Logic**: Expand `validate_training_data` beyond just dataset duration. Consider adding checks for:
    - Presence of NaNs or non-finite values in critical columns (target, key features).
    - Minimum number of observations per location or overall.
    - Consistency of time periods across different locations within the `DataSet`.
    - If `estimator` is used, compatibility of dataset features with `estimator.expected_features`.
4.  **Clarity on Time Coverage Check**: The check `dataset.end_timestamp < min_duration` assumes `min_duration` is calculated correctly based on `dataset.start_timestamp + (2 * delta_year)`. Ensure `delta_year` and timestamp arithmetic handle edge cases like leap years correctly if very precise "two full years" is critical. The error message is good.
5.  **Logging Validation Steps**: Add `logger.debug` or `logger.info` messages for each validation step performed and its outcome. This can be very helpful for users to understand why their data might be failing validation.

---

## `chap_core/adaptors/__init__.py`

1.  **Module Docstring**: Ensure a clear module docstring explains the role of the `adaptors` package (e.g., "Contains modules for adapting CHAP-core data and functionalities to various external interfaces, libraries, or use cases."). (Primary task).
2.  **Selective Namespace Exports**: If specific classes or functions from modules within `adaptors` (like `CommandLineInterfaceAdapter` from `command_line_interface.py` or `RestApiAdapter` from `rest_api.py`) are intended as the primary public interface of this package, import them here and list them in `__all__` for cleaner access by users of the `adaptors` package.
3.  **Package-Level Initialization**: If there's any setup or configuration common to all adaptors in this package (e.g., initializing a common service, setting up a specific logger for adaptors), this `__init__.py` could be a place for it.
4.  **Consider Sub-Package Structure**: If the number of adaptors grows significantly, evaluate if further sub-packaging within `adaptors` (e.g., `adaptors.data_format_adaptors`, `adaptors.library_wrappers`) would improve organization.
5.  **Documentation Entry Point**: This `__init__.py`'s docstring can serve as an entry point for developers looking to understand the `adaptors` package. It could briefly list the key adaptors available and their main purpose.

---

## `chap_core/adaptors/command_line_interface.py`

1.  **Comprehensive Docstrings**: Add a module-level docstring and detailed docstrings for the main generator functions (`generate_app`, `generate_template_app`). Review and correct the docstrings of the inner `train` and `predict` CLI command functions, ensuring parameter descriptions are accurate and complete.
2.  **Robust Error Handling**: Implement error handling for file I/O operations (e.g., `DataSet.from_csv`, model loading/saving, config parsing) using `try-except` blocks to catch `FileNotFoundError`, `pd.errors.ParserError`, `yaml.YAMLError`, and other potential exceptions, providing informative error messages to the CLI user.
3.  **Configuration Handling in `generate_template_app`**:
    - Clarify the role and necessity of `model_config_path` in both `train` and `predict` commands of `generate_template_app`. If a trained model (predictor) is self-contained, the config might not be needed for prediction.
    - Handle cases where `model_config_path` might be `None` or invalid when `parse_file` is called.
4.  **Dynamic Dataclass and Feature Determination**: The dynamic creation of dataclasses and determination of `covariate_names` (related to the `TODO` comment) should be made more robust and clearly documented. Ensure that the assumptions about how `estimator.covariate_names` are obtained are reliable.
5.  **Review Commented-Out Code**: Evaluate the large block of commented-out code at the end of `generate_template_app`. If it represents an obsolete approach or an unfinished feature not currently planned, it should be removed to improve code clarity. If it's relevant for future work, it should be documented as such or moved to an issue tracker.

---

## `chap_core/adaptors/gluonts.py`

1.  **Comprehensive Docstrings**: Add a module-level docstring and detailed docstrings for the `GluonTSPredictor` and `GluonTSEstimator` classes and all their methods. Explain their roles as adaptors, the purpose of their parameters, what they return, and how they interact with the GluonTS library.
2.  **Specific Type Hinting**: Enhance type hints for `DataSet` arguments (e.g., `DataSet[SomeSpecificDataType]`) and clarify the types returned by `DataSetAdaptor` methods if possible. Ensure all methods have explicit return type hints.
3.  **Robust Error Handling**: Implement `try-except` blocks for operations involving the GluonTS library (training, prediction, serialization, deserialization) and file system interactions (`save`, `load`) to catch and handle potential errors gracefully (e.g., GluonTS-specific exceptions, `FileNotFoundError`, `IOError`, pickling/serialization errors).
4.  **Dynamic `freq` in `ListDataset`**: The frequency (`freq="M"`) for `gluonts.dataset.common.ListDataset` in `GluonTSEstimator.train` is hardcoded. This should be made dynamic, ideally inferred from the input `dataset.time_period.freq_str` (if available) or passed as a parameter, as GluonTS models are sensitive to this.
5.  **Context for `DataSetAdaptor`**: Add a brief comment explaining the role of `DataSetAdaptor` (imported from `..data.gluonts_adaptor.dataset`) in converting between CHAP-core's `DataSet` format and the format expected by GluonTS.

---

## `chap_core/adaptors/rest_api.py`

1.  **Comprehensive Docstrings**: Add a module-level docstring and a detailed docstring for `generate_app`. Critically review and correct the docstrings of the inner `train` and `predict` API endpoint functions to accurately reflect their intended REST API behavior, parameters (especially how data is received), and responses.
2.  **Fix `train` Endpoint Logic**: The `train` endpoint implementation has critical issues: it defines `training_data: List[model]` as an input parameter (presumably from the request body) but then attempts to load data from a hardcoded file path and calls `DataSet.df_from_pydantic_observations()` without using the input `training_data`. This must be refactored to process the actual input data.
3.  **Re-evaluate `predict` Endpoint Data Input**: The `predict` endpoint currently expects file paths as parameters. For a typical REST API, data is usually sent in the request body or referenced by IDs if managed by the API. Clarify if this file-path-based approach is intentional for a specific deployment context or if it should be adapted for standard REST practices.
4.  **Configuration and State Management**: The use of `working_dir` and hardcoded filenames (`training_data.csv`, `model`) suggests a single-model or stateful instance. For a more general REST API, consider how multiple models, datasets, and concurrent requests would be managed. Paths and configurations should be more robustly handled.
5.  **API Error Handling and Responses**: Implement proper error handling within the FastAPI endpoint functions. Catch exceptions from data loading, model operations, etc., and return appropriate HTTP error responses (e.g., using `fastapi.HTTPException`) with informative messages instead of letting exceptions propagate directly.

---

## `chap_core/assessment/__init__.py`

1.  **Module Docstring**: Ensure a clear module docstring explains the role of the `assessment` package (e.g., "Contains modules for model evaluation, performance assessment, dataset splitting, and forecast analysis.").
2.  **Selective Namespace Exports**: If key classes or functions from modules within `assessment` (e.g., `evaluate_model` from `prediction_evaluator.py`, `train_test_split` from `dataset_splitting.py`) are intended as the primary public interface of this package, import them here and list them in `__all__` for cleaner access.
3.  **Common Assessment Utilities**: If there are common utility functions, constants, or base classes used across different assessment modules (e.g., a base class for evaluation metrics, common plotting utilities for assessment results), this `__init__.py` could be a place to define or expose them.
4.  **Sub-Package Structure Overview (in Docstring)**: The module docstring could provide a brief overview of the modules contained within the `assessment` package and their respective roles (e.g., "Includes modules for: `dataset_splitting` for creating train/test sets, `prediction_evaluator` for calculating performance metrics, `forecast` for generating evaluation forecasts.").
5.  **Package-Level Logging Setup**: If the assessment processes generate significant logs or require specific log formatting, consider initializing a package-level logger here (e.g., `logger = logging.getLogger(__name__)`, potentially with a `NullHandler` for library use).

---

## `chap_core/assessment/dataset_splitting.py`

1.  **Module Docstring**: Add a comprehensive module docstring that outlines the purpose of this module: providing various strategies for splitting time series datasets for model training, testing, and backtesting.
2.  **`IsTimeDelta` Protocol Definition**: The `IsTimeDelta` protocol is defined with `pass`. For it to be a useful structural subtype, it should define the expected attributes or methods of a "time delta object" (e.g., methods to add to a `TimePeriod` or represent a duration). If it's just a conceptual marker, this should be documented.
3.  **Parameter Consistency (`extension` vs. `future_length`)**: The parameters `extension` and `future_length` (both typed as `IsTimeDelta`) seem to serve similar purposes (defining the length of a future/test period). Ensure their naming and usage are consistent and clearly documented across all relevant functions to avoid confusion.
4.  **Robust Edge Case Handling & Input Validation**:
    - In `train_test_split` and `train_test_split_with_weather`, add checks for `prediction_start_period` being within the bounds of `data_set.period_range`.
    - In `train_test_generator`, validate that `prediction_length`, `n_test_sets`, and `stride` are compatible with the dataset length to prevent indexing errors. The calculation of `split_idx` should be robust.
    - In `get_split_points_for_data_set`, handle the case where `data_set.values()` might be empty to avoid `StopIteration`.
5.  **Clarify `future_weather_provider` Interface**: The `future_weather_provider` parameter in `train_test_generator` is typed as `Optional[FutureWeatherFetcher]`. The usage `future_weather_provider(hd).get_future_weather(...)` implies `future_weather_provider` is a callable that takes historical data (`hd`) and returns an object with a `get_future_weather` method. This interface contract should be clearly documented.

---

## `chap_core/assessment/evaluator.py`

1.  **Module Docstring**: Add a comprehensive module docstring that describes the purpose of this module: defining an evaluation framework for comparing model forecasts against ground truth data, including base classes and a flexible component-based evaluator.
2.  **Specific Callable Type Hints**: In `ComponentBasedEvaluator.__init__`, refine the type hints for `errorFunc`, `timeAggregationFunc`, and `regionAggregationFunc`. Instead of just `Callable`, use more specific signatures like `Callable[[float, List[float]], float]` for `errorFunc` (truth, samples -> error) and `Callable[[List[float]], float]` for aggregation functions, to clarify their expected inputs and outputs.
3.  **Robust Error Handling in `evaluate`**: Replace `assert` statements in `ComponentBasedEvaluator.evaluate` (e.g., for length and time period mismatches) with proper error handling that raises informative exceptions (e.g., `DataAlignmentError`) to clearly indicate issues with input data alignment.
4.  **Documentation of Aggregation Output Structure**: Clearly document the structure of the output `MultiLocationErrorTimeSeries` when time and/or region aggregations are applied. Specifically, explain how the `time_period` and location keys (e.g., "Full_period", "Full_region") are set for aggregated results.
5.  **Extensibility of `Error` Object for Multiple Metrics**: The `Error` object currently stores a single `value`. If there's a need to compute and store multiple error metrics simultaneously (e.g., MAE, RMSE, bias) for each evaluation point or aggregation, consider whether the `Error` object should support a dictionary of metric values, or if the framework expects separate `ErrorTimeSeries` for each metric.

---

## `chap_core/assessment/evaluator_suites.py`

1.  **Module Docstring**: Add a comprehensive module docstring that describes the purpose of this file: to define common error functions, aggregation functions, pre-configured `ComponentBasedEvaluator` instances, and suites of these evaluators for standardized model assessment.
2.  **Correct `ComponentBasedEvaluator` Parameter Names**: Ensure the parameter names used when instantiating `ComponentBasedEvaluator` match the actual parameter names in its `__init__` method (`errorFunc`, `timeAggregationFunc`, `regionAggregationFunc`). (This was a functional correction made).
3.  **Clarify `predictions` Argument in Error Functions**: The docstrings for `mae_error` and `mse_error` should explicitly state what the `predictions` list represents (e.g., multiple samples from a probabilistic forecast, where the mean of samples is compared to the truth).
4.  **Comprehensive Type Hinting**: Add specific type hints for all function parameters and return values (e.g., `truth: float`, `predictions: List[float]`), and for complex structures like `evaluator_suite_options` (e.g., `Dict[str, List[ComponentBasedEvaluator]]`).
5.  **Extensibility and Configuration of Suites**: The `evaluator_suite_options` dictionary is hardcoded. For greater flexibility, consider if these suites could be defined in a configuration file (e.g., YAML) and loaded dynamically, or if a registration mechanism for custom evaluators and suites would be beneficial.

---

## `chap_core/assessment/forecast.py`

1.  **Module Docstring**: Add a comprehensive module docstring that describes the purpose of this module: providing functions to generate forecasts under various scenarios, primarily for model assessment, backtesting, and future prediction with synthesized weather.
2.  **Typo Correction (`prediction_lenght`)**: Correct the typo in the `prediction_lenght` parameter in `multi_forecast` to `prediction_length`. (This was a functional correction made).
3.  **Model Parameter Configuration in `forecast`**: The direct setting of `model._num_warmup` and `model._num_samples` in the `forecast` function is specific to certain model types. This should be documented clearly, or ideally, such parameters should be configured via the model's own interface. The dead code `if False and hasattr(model, "diagnose"):` was removed.
4.  **Specific Type Hints for Models**: Refine type hints for `model`, `Estimator`, and `Predictor` parameters. If there's a common base class or protocol, use it for more precise typing.
5.  **Error Handling and Input Validation**: Enhance error handling for model operations (train, predict) and data processing steps. Document expected exceptions from model methods.

---

## `chap_core/assessment/prediction_evaluator.py`

1.  **Comprehensive Docstrings**: Add a module-level docstring and detailed docstrings for the `Predictor` and `Estimator` protocols, and all functions (`backtest`, `create_multiloc_timeseries`, `_get_forecast_dict`, `plot_forecasts`, `plot_predictions`).
2.  **Typo Correction**: Correct `FetureType` to `FeatureType` in its `TypeVar` definition and usage.
3.  **Clarify `without_disease` Type Hint**: The `without_disease(FeatureType)` construct in `Predictor.predict` type hint is unconventional. Its docstring should clearly explain its purpose as a marker.
4.  **Document GluonTS Integration**: Explicitly document the integration with GluonTS components (e.g., `gluonts.evaluation.Evaluator`, `ForecastAdaptor`) in relevant docstrings.
5.  **Matplotlib Usage in Plotting**: Document `matplotlib.pyplot` dependency for plotting functions. Advise against global log level changes in library code.

---

## `chap_core/assessment/representations.py`

1.  **Module Docstring**: Add a comprehensive module docstring explaining that this file defines dataclasses for model assessment components like observations, errors, and forecasts.
2.  **`time_period` Type Hint**: The `time_period` attribute in `DiseaseObservation`, `Error`, and `Samples` is `str`. Clarify if this should be a more specific type (e.g., `TimePeriod`) or if string identifiers (like "Full_period") are intentional and document their usage.
3.  **Complete Method Docstrings**: Add docstrings for all methods in `MultiLocationDiseaseTimeSeries` and `MultiLocationErrorTimeSeries` that were missing them (e.g., `__setitem__`, `__getitem__`, `locations`, utility methods).
4.  **Robust Error Handling in Utility Methods**: Replace `assert` statements in `MultiLocationErrorTimeSeries` utility methods with specific exceptions (e.g., `ValueError`) and informative messages.
5.  **Consistency and Naming**: Review `timeseries_dict` initialization consistency. Clarify naming for `Forecast` vs. `MultiLocationForecast` in docstrings to emphasize single vs. multi-location.

---

## `chap_core/climate_data/__init__.py`

1.  **Module Docstring**: Add a clear module docstring explaining the role of the `climate_data` package (e.g., "Modules for accessing, processing, and representing climate data.").
2.  **Review and Action Commented Code**: Evaluate the commented-out imports and `__all__` list. If intended for the package's public API, uncomment and verify. If obsolete, remove.
3.  **Clarify `Shape` Datatype**: If the commented `Shape` import is relevant for defining geographical areas for climate data, its role should be documented, and re-export considered.
4.  **Expose Key Components**: If modules like `seasonal_forecasts.py` provide central components, consider re-exporting them here via `__all__`.
5.  **Package-Specific Utilities**: If common base classes or utilities for climate data handling are needed across this package, this `__init__.py` could define or expose them.

---

## `chap_core/climate_data/seasonal_forecasts.py`

1.  **Comprehensive Docstrings**: Add a module-level docstring, and detailed docstrings for the `DataElement` Pydantic model, the `SeasonalForecast` class, and all its methods (`__init__`, `add_json`, `get_forecasts`). Clearly explain the structure of `data_dict` in `SeasonalForecast`.
2.  **Refined Type Hinting**:
    - Use `typing.Dict` for `data_dict` in `SeasonalForecast.__init__`.
    - Type `json_data` in `add_json` as `List[Dict[str, Any]]`.
    - Type hint `period_range` in `get_forecasts` as `chap_core.time_period.PeriodRange`.
3.  **Robust Error Handling**:
    - In `add_json`, wrap `DataElement(**data)` in `try-except pydantic.ValidationError`.
    - In `get_forecasts`, replace `assert` statements with `KeyError` or `ValueError` for missing items.
4.  **Unused `start_date` Parameter**: Remove the unused `start_date` parameter from `SeasonalForecast.get_forecasts` or implement its functionality.
5.  **Clarity of `data_dict` Structure**: The docstring for `SeasonalForecast` should explicitly describe its nested dictionary structure: `Dict[field_name, Dict[org_unit_id, Dict[period_str, value_float]]]`.

---

## `chap_core/data/__init__.py`

1.  **Module Docstring**: Add a comprehensive module docstring that clearly defines the scope and purpose of the `chap_core.data` package, noting its role in core data structures, adaptors, and dataset utilities.
2.  **Rationale for Re-exports**: In the module docstring or comments, explain why `DataSet` and `PeriodObservation` are specifically chosen for re-export (e.g., convenience, defining a stable public API).
3.  **Completeness of `__all__`**: Review submodules (`adaptors.py`, `datasets.py`, `open_dengue.py`, `gluonts_adaptor/`). If other components are key to this package's public interface, add them to `__all__`.
4.  **API Stability Note**: If re-exporting is an intentional design choice for API stability, this benefit could be mentioned in the docstring.
5.  **Overview of Sub-components in Docstring**: The module docstring could briefly outline the roles of key modules and sub-packages within `chap_core.data` to guide developers.

---

## `chap_core/data/adaptors.py`

1.  **Module Docstring**: Add a clear module docstring explaining that this module provides pre-instantiated data adaptors, currently focusing on an adaptor for GluonTS datasets.
2.  **Rationale for Singleton Instance**: Document the reason for providing `gluonts` as a pre-instantiated singleton of `DataSetAdaptor`. Is it for global default configuration or convenience?
3.  **Naming Clarity**: While `gluonts` is concise, consider if a more descriptive name like `default_gluonts_dataset_adaptor` would improve clarity, especially if other adaptors are introduced.
4.  **Future Extensibility**: If more data adaptors are anticipated, briefly mention in the docstring how this module might evolve or how new adaptors could be added/registered.
5.  **Configuration of `DataSetAdaptor`**: If `DataSetAdaptor` has configurable parameters, document how users can obtain a differently configured instance if the default `gluonts` instance is not suitable.

---

## `chap_core/database/__init__.py`

1.  **Module Docstring**: Add a comprehensive module docstring that clearly defines the scope and purpose of the `chap_core.database` package, outlining its role in managing database interactions, schemas, and sessions.
2.  **Expose Key Components via `__all__`**: Identify core classes and functions from the submodules (e.g., `SessionWrapper` from `database.py`, main ORM base classes from `base_tables.py`, or key table models) that form the public API of this package. Import and list them in `__all__` for convenient access.
3.  **Centralized Database Initialization/Setup**: Consider providing a utility function here (e.g., `get_engine()`, `create_db_session()`) if there's a standard way the application initializes its database connection or sessions, abstracting away the details from other parts of the application.
4.  **Overview of Database Schema Modules**: The module docstring could briefly describe the different schema modules within this package (e.g., `dataset_tables.py`, `model_spec_tables.py`, `tables.py`) and the types of data they represent in the database.
5.  **Database-Related Constants**: If there are any package-wide constants related to the database (e.g., default schema names, common string lengths for certain fields, specific database dialect options if applicable), this `__init__.py` could be a suitable place to define them.

---

## `chap_core/database/base_tables.py`

1.  **Module Docstring**: Add a comprehensive module docstring explaining that this file provides foundational elements for database table models, including a base SQLModel class with Pydantic configurations for API interaction and common type aliases.
2.  **`PeriodID` Documentation**: Add a comment to the `PeriodID = str` type alias to specify the expected format or conventions for period identifiers (e.g., "YYYYMM" for monthly, "YYYY-WW" for weekly, or if it's a generic string).
3.  **Elaborate `DBModel` Docstring**: Expand the docstring for `DBModel` to clearly state its purpose as a common base for all SQLModel table definitions in the project. Emphasize that its Pydantic `ConfigDict` (using `to_camel` for aliases and `populate_by_name`) standardizes JSON serialization/deserialization, particularly for REST API interactions.
4.  **Clarify `table=True` Expectation**: The `DBModel` itself is not a table. Its docstring or a comment could note that subclasses intended to be database tables must include `table=True` in their class definition (e.g., `class MyTable(DBModel, table=True): ...`).
5.  **Illustrative Example (Optional)**: Consider adding a brief, commented-out example of a simple model inheriting from `DBModel` to illustrate how the camelCase aliasing works with field definitions and Pydantic, e.g., `my_field: str = Field(default=None, alias="myField")`.

---

## `chap_core/database/database.py`

1.  **Module Docstring**: Add a comprehensive module docstring explaining the purpose of this module, focusing on database session management, engine creation, and utility functions for interacting with the database.
2.  **Refined Engine Initialization Error**: In the engine initialization block, when all connection retries fail, raise a more specific custom exception (e.g., `DatabaseConnectionError(ChapCoreException)`) instead of a generic `ValueError`.
3.  **Clarify `SessionWrapper.__init__`**: Correct the logic `self.engine = local_engine # or engine` to `self.engine = local_engine or engine` if the global `engine` is intended as a fallback. Clearly document the purpose of the `session` parameter (e.g., for allowing external session management or testing).
4.  **Robustness to Schema Changes in `add_*` Methods**: The `add_dataset_metadata`, `add_model_spec`, and `add_model_run` methods are tightly coupled to specific table schemas. Document this coupling and consider using constants for field names or a more abstract way to handle data insertion to improve resilience to schema changes.
5.  **Transaction Management Review**: Review the transaction management in `SessionWrapper` methods. Ensure that `commit()` is called appropriately (e.g., after a logical unit of work) and that `rollback()` is handled in case of errors to maintain data integrity. Consider if context management (`__enter__`, `__exit__`) for sessions should explicitly handle commit/rollback.

---

## `chap_core/plotting/__init__.py`

1.  **Module Docstring**: Add a comprehensive module docstring that clearly defines the scope and purpose of the `chap_core.plotting` package, highlighting its role in providing visualization utilities for time series data, forecasts, and model evaluations.
2.  **Rationale for Re-exports**: In the module docstring or via comments, explain that `plot_timeseries_data` and `plot_multiperiod` are re-exported to provide a convenient and stable public API for common plotting tasks from this package.
3.  **Completeness of `__all__`**: Review other functions in `plotting.py` and `prediction_plot.py`. If any of those are also considered core, frequently used plotting utilities, consider importing and adding them to the `__all__` list here.
4.  **External Dependencies Note**: The module docstring could briefly mention key external dependencies for this package, such as `matplotlib`, which is essential for the plotting functions.
5.  **Usage Examples (Optional)**: If these plotting functions have common usage patterns, a brief example could be included in the module docstring or linked to more detailed documentation/examples.

---

## `chap_core/plotting/plotting.py`

1.  **Module Docstring**: Add a comprehensive module docstring explaining the purpose of this file: to provide core plotting functions for visualizing CHAP-core time series data (like ClimateHealthTimeSeries, ClimateData, HealthData) using the Plotly library.
2.  **Standardized Function Docstrings**: Refactor the docstrings for `plot_timeseries_data` and `plot_multiperiod` to follow a consistent, standard format (e.g., including clear Args, Returns, and potentially Raises sections) for improved readability and maintainability.
3.  **Robust Input Data Handling**:
    - In `plot_multiperiod`, add checks for empty DataFrames after `climate_data.topandas().head(head)` and `health_data.topandas()` to prevent `IndexError` if data is empty or `head` results in an empty frame.
    - The line `cut_off_idx = (health_df.time_period == cut_off).to_list().index(True) + 1` in `plot_multiperiod` can raise a `ValueError` if `cut_off` is not found. Implement more robust handling, perhaps by logging a warning and plotting up to the available data, or raising a custom error.
4.  **Configurable Variables in `plot_multiperiod`**:
    - The function currently hardcodes plotting "mean_temperature" and "disease_cases". Make these variable names parameters of the function to allow plotting other relevant climate or health variables.
    - The frequency "M" for `pd.Period(year=last_year, month=last_month, freq="M")` is hardcoded. If input data can have different frequencies, this should ideally be derived from `climate_data.time_period.freq` or made a parameter.
5.  **Enhanced Plot Customization**: Allow users to pass more Plotly layout and trace customization options as parameters to the functions (e.g., titles, axis labels, colors, line shapes, figure size). This would provide greater control over the plot aesthetics.

---

## `chap_core/plotting/prediction_plot.py`

1.  **Module Docstring**: Add a module docstring explaining that this file provides functions for visualizing model predictions, including forecast samples, quantiles, and comparisons against true data, using both Matplotlib and Plotly.
2.  **Comprehensive Function Docstrings**: Add detailed docstrings for all functions.
3.  **Consistent Plotting Library**: Consider refactoring `prediction_plot` to use Plotly for consistency, or document the rationale for using different libraries.
4.  **Type Hinting**: Add or complete type hints for all function parameters and return values.
5.  **Error Handling and Input Validation**: Add input validation (e.g., `n_samples > 0`). Improve error handling in `add_prediction_lines` for `IndexError` or missing columns. Ensure data alignment.

---

<environment_details>

# VSCode Visible Files

chap_core/plotting/prediction_plot.py

# VSCode Open Tabs

chap_core/plotting/**init**.py
chap_core/plotting/plotting.py
LEARN.md
TODO.md
chap_core/plotting/prediction_plot.py

# Current Time

5/9/2025, 5:29:24 PM (Europe/Oslo, UTC+2:00)

# Context Window Usage

773,659 / 1,048.576K tokens used (74%)

# Current Mode

ACT MODE
</environment_details>
