# CHAP-Core Learning Roadmap (with Climate Focus)

This 20-step roadmap is designed to help you understand the CHAP-core project, with a particular emphasis on its climate-related aspects.

## Phase 1: Project Overview & Setup (Steps 1-4)

1.  **Understand Project Goals & Scope**:

    - Read the project's main `README.md`.
    - Explore any high-level documentation in the `docs/` directory (e.g., `index.rst` or architecture overviews).
    - Identify the primary objectives of CHAP-core, its key features, and how it addresses climate and health analytics.

2.  **Explore Core Architecture**:

    - Familiarize yourself with the top-level directory structure: `chap_core/` (core library), `scripts/` (utility/example scripts), `tests/` (automated tests), `config/` (configuration files), `example_data/` (sample datasets), `external_models/` (externally defined models).
    - Note how "climate" related modules or data might be organized within this structure.

3.  **Setup Development Environment**:

    - Follow any setup instructions in `README.md` or `docs/installation/`.
    - Ensure you have the correct Python version (check `.python-version` if present).
    - Install dependencies (likely from `pyproject.toml` using a tool like Poetry or PDM, or `uv`).
    - Set up Docker and Docker Compose, as the project seems to use them (based on `Dockerfile` and `compose.*.yml` files).
    - Verify the environment by running any preliminary checks or simple commands suggested in the documentation.

4.  **Run Basic Examples/CLI**:
    - Look for and execute example scripts in `example_data/` or `scripts/` (e.g., `scripts/prediction_example.py`, `scripts/evaluation_example.py`).
    - Try using the command-line interface if available (likely defined in `chap_core/cli.py` or `chap_core/chap_cli.py`).
    - This will help you see the project's basic input/output flow.

## Phase 2: Data Handling & Climate Data Integration (Steps 5-9)

5.  **Data Structures (`chap_core/datatypes.py`)**:

    - Study how different data types are defined, especially `TimeSeriesData` and its derivatives.
    - Pay close attention to `ClimateData`, `SimpleClimateData`, and how they store climate variables (e.g., rainfall, temperature).
    - Understand how health data (`HealthData`) and population data (`HealthPopulationData`) are structured.

6.  **GeoJSON and Geometry (`chap_core/geojson.py`, `chap_core/geometry.py`, `chap_core/geoutils.py`)**:

    - Learn how the project handles geographic information (polygons, features).
    - Examine `chap_core/geometry.py` for how GADM data (administrative boundaries) is fetched and processed.
    - Review `chap_core/geoutils.py` for utility functions operating on these geometries (e.g., buffering, bounding box calculation). This is crucial for spatially explicit climate data.

7.  **Climate Data Sources & Fetching**:

    - Investigate the `chap_core/climate_data/` subdirectory (if it contains relevant modules for specific climate datasets or APIs).
    - Explore `chap_core/google_earth_engine/` (e.g., `gee_era5.py` if present) to understand how climate datasets like ERA5 might be accessed from Google Earth Engine.
    - Study `chap_core/climate_predictor.py`, focusing on `FutureWeatherFetcher` and its subclasses (`SeasonalForecastFetcher`, `QuickForecastFetcher`) to see how future or historical climate data is obtained or simulated for models.

8.  **Data Harmonization**:

    - Look for modules or scripts responsible for combining different data types (health, population, climate). The `harmonize` command in the (deprecated) `chap_core/chap_cli.py` or scripts like `scripts/harmonization_example.py` might offer insights.
    - Understand how different datasets are aligned temporally and spatially.

9.  **Example Climate Data**:
    - Analyze the structure and content of climate-related files in `example_data/`, such as:
      - `climate_data.csv` / `climate_data_daily.csv`
      - `precipitation_seasonal_forecast.json`
      - `temperature_seasonal_forecast.json`
    - This will provide concrete examples of expected climate data formats.

## Phase 3: Modeling & Prediction with Climate Factors (Steps 10-14)

10. **Model Specifications (`chap_core/model_spec.py`)**:

    - Understand the `ModelSpec` Pydantic model and how it defines a model's characteristics: parameters, input features (including climate variables), operational period, and metadata.
    - See how these specifications can be loaded from YAML files.

11. **Core Predictor Logic (`chap_core/predictor/`)**:

    - Dive into the `chap_core/predictor/` directory.
    - Look for base model classes (e.g., `ModelTemplateInterface` from `chap_core/models/model_template_interface.py`) and how specific models inherit from them.
    - Study `model_registry.py` (if present) to see how different predictive models are registered and made available.

12. **Feature Engineering & Transformations (`chap_core/predictor/feature_spec.py`, `chap_core/transformations/`)**:

    - Examine `feature_spec.py` to understand how input features for models are defined and categorized.
    - Explore the `chap_core/transformations/` directory for any modules that pre-process or transform data, including climate variables (e.g., creating lags, calculating anomalies, temporal aggregation).

13. **External & Climate-Sensitive Models (`external_models/`)**:

    - Browse the `external_models/` directory. Identify models that might be particularly relevant to climate (e.g., `hydromet_dengue` sounds promising).
    - Understand their structure (R scripts, Python code, Dockerfiles, `MLproject` files for MLflow integration).
    - Note how climate data is used as input or features in these models.

14. **Prediction & Forecasting Workflow (`chap_core/api.py`, `chap_core/cli.py`, `chap_core/predictor/forecast.py` if exists)**:
    - Trace the typical workflow for making a prediction. This involves data loading, preprocessing, model training (or loading a pre-trained model), feature generation, and finally, generating a forecast.
    - Identify how climate data (historical or future) feeds into this workflow.

## Phase 4: API, Deployment & Advanced Topics (Steps 15-20)

15. **REST API (`chap_core/rest_api_src/`)**:

    - Study the API structure, focusing on endpoints related to:
      - Data ingestion (including climate data).
      - Model configuration and selection.
      - Triggering predictions and evaluations.
      - Retrieving results.
    - Note the request/response schemas defined (likely using Pydantic, see `chap_core/api_types.py`).

16. **Asynchronous Tasks & Workers (`chap_core/worker/`, Celery/RQ files)**:

    - Learn how the project handles potentially long-running operations like model training or complex data processing. This often involves task queues like Celery or RQ.
    - Look for task definitions and worker configurations.

17. **Dockerization & Deployment (`Dockerfile`, `compose.*.yml`, `Makefile`)**:

    - Examine the main `Dockerfile` to understand how the CHAP-core application is containerized.
    - Review `compose.*.yml` files (e.g., `compose.yml`, `compose.dev.yml`) to see how different services (API, database, workers) are orchestrated for development and deployment.
    - Check the `Makefile` for common build, run, and test commands.

18. **Model Evaluation & Validation (`chap_core/assessment/`, `chap_core/validators.py`)**:

    - Understand the methodologies used for model evaluation (e.g., backtesting, metrics calculation in `chap_core/assessment/prediction_evaluator.py`).
    - Review `chap_core/validators.py` to see how input data, including training data that might contain climate variables, is validated.

19. **Configuration System (`config/`)**:

    - Explore files in the `config/` directory (e.g., `model_templates.yaml`).
    - Understand how models, system parameters, or other aspects of the application are configured.

20. **Contribute & Extend**:
    - Once you have a good grasp of the project, identify areas where you could contribute. This might involve:
      - Enhancing climate data integration (new sources, improved processing).
      - Developing or refining climate-sensitive predictive models.
      - Improving documentation or adding more examples related to climate applications.
    - Review any contributor guidelines (e.g., in `docs_source/contributor/` or `CONTRIBUTING.md`).

This roadmap provides a structured approach. Feel free to adjust the order or depth of exploration based on your specific interests and the project's evolving structure. Good luck!
