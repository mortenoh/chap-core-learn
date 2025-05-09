# External Services, Data Sources, and Dependencies

This document provides an overview of key external services, data sources, and Python libraries used by or relevant to the CHAP-Core project.

## External Services & Data Sources

This section details external platforms, datasets, and services that CHAP-Core interacts with or is designed to utilize.

### Google Earth Engine (GEE)

- **What it is**: Google Earth Engine is a cloud-based platform for planetary-scale geospatial analysis. It combines a multi-petabyte catalog of satellite imagery and geospatial datasets with planetary-scale analysis capabilities.
- **Purpose in CHAP-Core**: Used for accessing and processing various climate and environmental data (e.g., temperature, precipitation, land cover) at regional or national scales.
- **Key Features**:
  - Access to a vast archive of publicly available geospatial datasets (Landsat, Sentinel, MODIS, climate data, etc.).
  - Parallel processing capabilities for large-scale computations.
  - JavaScript and Python APIs for scripting and analysis.
  - Web-based Code Editor for interactive development.
- **Data Formats**: Primarily raster data (images) and vector data (feature collections).
- **Authentication**: Requires a registered Google account and authentication, often via service accounts for non-interactive use.
- **CHAP-Core Interaction**: Likely uses the `earthengine-api` Python client library (see dependencies) to query, retrieve, and process data. Modules like `chap_core.google_earth_engine` and `chap_core.climate_data` would handle these interactions.
- **Learning URL**: [https://earthengine.google.com/](https://earthengine.google.com/)
- **API Documentation**: [https://developers.google.com/earth-engine/guides/python_install](https://developers.google.com/earth-engine/guides/python_install)

### ERA5

- **What it is**: ERA5 is the fifth generation ECMWF (European Centre for Medium-Range Weather Forecasts) atmospheric reanalysis of the global climate. It provides hourly estimates of a large number of atmospheric, land, and oceanic climate variables.
- **Purpose in CHAP-Core**: Serves as a primary source for historical gridded climate data (e.g., temperature, precipitation) used as input features for disease models.
- **Key Features**:
  - Global coverage.
  - Hourly temporal resolution (though often aggregated to daily or monthly for modeling).
  - Covers the period from 1940 to near real-time.
  - Provides uncertainty information for its variables.
- **Data Access**: Can be accessed via the Copernicus Climate Data Store (CDS) API, or through platforms like Google Earth Engine which host ERA5 datasets.
- **Variables**: Includes temperature, precipitation, wind speed, humidity, soil moisture, etc.
- **CHAP-Core Interaction**: Likely fetched via Google Earth Engine (using `earthengine-api`) or directly from CDS using a Python client. Processed by modules in `chap_core.climate_data`.
- **Learning URL (ECMWF)**: [https://www.ecmwf.int/en/forecasts/dataset/ecmwf-reanalysis-v5](https://www.ecmwf.int/en/forecasts/dataset/ecmwf-reanalysis-v5)
- **Learning URL (Copernicus CDS)**: [https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels)

### GADM (Database of Global Administrative Areas)

- **What it is**: GADM is a high-resolution database of global administrative boundaries (countries, provinces, districts, etc.).
- **Purpose in CHAP-Core**: Used to obtain administrative boundary vector data (polygons) for specific countries or regions. This is essential for spatially aggregating data, defining areas of interest, and visualization.
- **Key Features**:
  - Provides multiple administrative levels for most countries.
  - Available in various formats (e.g., GeoPackage, Shapefile, R data files).
  - Versioned, allowing for reproducible research.
- **Data Access**: Can be downloaded directly from the GADM website or accessed via tools/libraries that wrap GADM data (e.g., `pooch` in CHAP-Core for fetching).
- **CHAP-Core Interaction**: The `chap_core.geometry` module uses `pooch` to download GADM data files (likely GeoPackage) and `geopandas` to read and process these shapefiles into polygon data.
- **Learning URL**: [https://gadm.org/](https://gadm.org/)

### yr.no (Norwegian Meteorological Institute & NRK)

- **What it is**: A weather service provided by the Norwegian Meteorological Institute (MET Norway) and the Norwegian Broadcasting Corporation (NRK).
- **Purpose in CHAP-Core**: Likely used as a source for weather forecasts or recent observational weather data, potentially for specific locations.
- **Key Features**:
  - Provides weather forecasts, textual forecasts, and weather maps.
  - Offers a free, public API for accessing weather data (though terms of use should be checked for high-volume applications).
  - Data includes temperature, precipitation, wind, symbols for weather conditions, etc.
- **Data Access**: Typically via their Locationforecast API.
- **CHAP-Core Interaction**: If used, it would likely be through HTTP requests (e.g., using `requests` or `httpx` libraries) to their API endpoints. Modules in `chap_core.climate_data` or `chap_core.fetch` might handle this.
- **Learning URL**: [https://www.yr.no/](https://www.yr.no/)
- **API Info (Developer Portal)**: [https://developer.yr.no/](https://developer.yr.no/)

---

## Python Dependencies

This section lists Python libraries used in CHAP-Core, as specified in `pyproject.toml`.

### Main Dependencies

- **`annotated_types`**: Provides utilities for adding metadata to types, often used with Pydantic for more expressive validation.
  - URL: [https://pypi.org/project/annotated-types/](https://pypi.org/project/annotated-types/)
- **`bionumpy`**: Efficiently handles biological sequence data using NumPy arrays. Its `bnpdataclass` is used in `chap_core.datatypes`.
  - URL: [https://pypi.org/project/bionumpy/](https://pypi.org/project/bionumpy/)
- **`cyclopts`**: A library for creating command-line interfaces with type annotations. Used in `chap_core.cli` and `chap_core.adaptors.command_line_interface`.
  - URL: [https://pypi.org/project/cyclopts/](https://pypi.org/project/cyclopts/)
- **`diskcache`**: A disk-backed cache library, useful for memoizing expensive function calls or storing temporary results.
  - URL: [https://pypi.org/project/diskcache/](https://pypi.org/project/diskcache/)
- **`docker`**: The official Python library for interacting with the Docker Engine API. Used in `chap_core.docker_helper_functions`.
  - URL: [https://pypi.org/project/docker/](https://pypi.org/project/docker/)
- **`earthengine-api==1.4.6`**: Python client library for Google Earth Engine.
  - URL: [https://pypi.org/project/earthengine-api/](https://pypi.org/project/earthengine-api/)
- **`fastapi`**: A modern, fast (high-performance) web framework for building APIs with Python, based on standard Python type hints. Used for the REST API.
  - Purpose: Building the REST API interface for CHAP-Core.
  - Key Features: Automatic data validation (via Pydantic), interactive API documentation (Swagger UI, ReDoc), dependency injection, asynchronous support.
  - CHAP-Core Usage: `chap_core.adaptors.rest_api` uses it to generate API endpoints.
  - Learning URL: [https://fastapi.tiangolo.com/](https://fastapi.tiangolo.com/)
- **`sqlmodel`**: A library for interacting with SQL databases from Python code, with Python objects. It's built on Pydantic and SQLAlchemy.
  - Purpose: Defines database table schemas and serves as an ORM.
  - Key Features: Combines SQLAlchemy's power with Pydantic's data validation, type hints for table models, automatic schema generation.
  - CHAP-Core Usage: Used extensively in `chap_core.database` modules (`base_tables.py`, `dataset_tables.py`, etc.).
  - Learning URL: [https://sqlmodel.tiangolo.com/](https://sqlmodel.tiangolo.com/)
- **`psycopg2-binary`**: PostgreSQL database adapter for Python. Required if CHAP-Core connects to a PostgreSQL database.
  - URL: [https://pypi.org/project/psycopg2-binary/](https://pypi.org/project/psycopg2-binary/)
- **`geopandas`**: Extends pandas to allow spatial operations on geometric types. Used for handling GeoJSON and shapefiles.
  - URL: [https://pypi.org/project/geopandas/](https://pypi.org/project/geopandas/)
- **`geopy`**: Python client for several popular geocoding web services.
  - URL: [https://pypi.org/project/geopy/](https://pypi.org/project/geopy/)
- **`gitpython`**: Python library to interact with Git repositories.
  - URL: [https://pypi.org/project/GitPython/](https://pypi.org/project/GitPython/)
- **`gluonts`**: A Python toolkit for probabilistic time series modeling, focusing on deep learning-based models.
  - Purpose: Used for time series forecasting and evaluation.
  - Key Features: Pre-built models (DeepAR, Transformer, etc.), tools for data processing, evaluation metrics.
  - CHAP-Core Usage: `chap_core.adaptors.gluonts` provides wrappers. `chap_core.assessment.prediction_evaluator` uses its evaluation tools.
  - Learning URL: [https://ts.gluon.ai/](https://ts.gluon.ai/)
- **`httpx`**: A fully featured HTTP client for Python 3, which provides sync and async APIs.
  - URL: [https://pypi.org/project/httpx/](https://pypi.org/project/httpx/)
- **`libpysal`**: Library for spatial data analysis, including spatial weights, spatial autocorrelation, etc.
  - URL: [https://pypi.org/project/libpysal/](https://pypi.org/project/libpysal/)
- **`matplotlib`**: A comprehensive library for creating static, animated, and interactive visualizations in Python.
  - URL: [https://pypi.org/project/matplotlib/](https://pypi.org/project/matplotlib/)
- **`meteostat`**: Python library for accessing open weather and climate data from sources like NOAA, DWD.
  - URL: [https://pypi.org/project/meteostat/](https://pypi.org/project/meteostat/)
- **`mlflow-skinny`**: A lightweight version of MLflow, an open-source platform for the machine learning lifecycle.
  - URL: [https://pypi.org/project/mlflow-skinny/](https://pypi.org/project/mlflow-skinny/)
- **`numpy`**: Fundamental package for scientific computing with Python, providing support for large, multi-dimensional arrays and matrices.
  - URL: [https://pypi.org/project/numpy/](https://pypi.org/project/numpy/)
- **`pandas`**: A fast, powerful, flexible, and easy-to-use open-source data analysis and manipulation tool.
  - Purpose: Core library for data handling, manipulation, and analysis in tabular form (DataFrames, Series).
  - Key Features: DataFrame and Series objects, tools for reading/writing data (CSV, Excel, SQL, etc.), data cleaning, reshaping, merging, time series functionality.
  - CHAP-Core Usage: Used extensively for data processing, especially in `datatypes.py` (`topandas`), plotting functions, and data adaptors.
  - Learning URL: [https://pandas.pydata.org/](https://pandas.pydata.org/)
- **`plotly`**: An interactive graphing library for Python. Used for creating web-based visualizations.
  - URL: [https://pypi.org/project/plotly/](https://pypi.org/project/plotly/)
- **`pooch`**: A Python library to download and cache data files from the web. Used in `chap_core.geometry` for GADM data.
  - URL: [https://pypi.org/project/pooch/](https://pypi.org/project/pooch/)
- **`pycountry`**: Provides ISO country, subdivision, language, currency, and script definitions.
  - URL: [https://pypi.org/project/pycountry/](https://pypi.org/project/pycountry/)
- **`pydantic-geojson<2`**: Pydantic models for GeoJSON objects.
  - URL: [https://pypi.org/project/pydantic-geojson/](https://pypi.org/project/pydantic-geojson/)
- **`pydantic>=2.0`**: Data validation and settings management using Python type annotations.
  - URL: [https://pypi.org/project/pydantic/](https://pypi.org/project/pydantic/)
- **`python-dateutil`**: Provides powerful extensions to the standard datetime module.
  - URL: [https://pypi.org/project/python-dateutil/](https://pypi.org/project/python-dateutil/)
- **`python-dotenv`**: Reads key-value pairs from a `.env` file and can set them as environment variables.
  - URL: [https://pypi.org/project/python-dotenv/](https://pypi.org/project/python-dotenv/)
- **`python-multipart`**: A streaming multipart parser for Python. Often used with FastAPI for file uploads.
  - URL: [https://pypi.org/project/python-multipart/](https://pypi.org/project/python-multipart/)
- **`pyyaml`**: YAML parser and emitter for Python. Used for configuration files.
  - URL: [https://pypi.org/project/PyYAML/](https://pypi.org/project/PyYAML/)
- **`requests`**: An elegant and simple HTTP library for Python, built for human beings.
  - URL: [https://pypi.org/project/requests/](https://pypi.org/project/requests/)
- **`rq`**: (Redis Queue) A simple Python library for queueing jobs and processing them asynchronously with workers.
  - URL: [https://pypi.org/project/rq/](https://pypi.org/project/rq/)
- **`scikit-learn`**: Machine learning library for Python, featuring various classification, regression, and clustering algorithms.
  - URL: [https://pypi.org/project/scikit-learn/](https://pypi.org/project/scikit-learn/)
- **`scipy`**: A Python-based ecosystem of open-source software for mathematics, science, and engineering.
  - URL: [https://pypi.org/project/scipy/](https://pypi.org/project/scipy/)
- **`topojson`**: An extension of GeoJSON that encodes topology. This library likely provides tools for working with TopoJSON data.
  - URL: [https://pypi.org/project/topojson/](https://pypi.org/project/topojson/)
- **`unidecode`**: Transliterates Unicode text into US-ASCII.
  - URL: [https://pypi.org/project/Unidecode/](https://pypi.org/project/Unidecode/)
- **`uvicorn`**: An ASGI server, used for running FastAPI applications.
  - URL: [https://pypi.org/project/uvicorn/](https://pypi.org/project/uvicorn/)
- **`virtualenv`**: A tool to create isolated Python environments.
  - URL: [https://pypi.org/project/virtualenv/](https://pypi.org/project/virtualenv/)
- **`xarray`**: N-D labeled arrays and datasets in Python, inspired by pandas. Useful for multi-dimensional scientific data.
  - URL: [https://pypi.org/project/xarray/](https://pypi.org/project/xarray/)
- **`orjson>=3.10.7`**: A fast JSON library for Python.
  - URL: [https://pypi.org/project/orjson/](https://pypi.org/project/orjson/)
- **`celery[pytest]`**: Distributed task queue. `[pytest]` extra installs pytest integration.
  - URL: [https://pypi.org/project/celery/](https://pypi.org/project/celery/)

### Development Dependencies

- **`build`**: A simple, correct PEP 517 build frontend.
  - URL: [https://pypi.org/project/build/](https://pypi.org/project/build/)
- **`bump2version`**: A tool to simplify version string bumping.
  - URL: [https://pypi.org/project/bump2version/](https://pypi.org/project/bump2version/)
- **`coverage>=7.6.2`**: Code coverage measurement for Python.
  - URL: [https://pypi.org/project/coverage/](https://pypi.org/project/coverage/)
- **`furo>=2024.8.6`**: A clean, customizable Sphinx documentation theme.
  - URL: [https://pypi.org/project/furo/](https://pypi.org/project/furo/)
- **`myst-parser>=4.0.0`**: A Sphinx parser for MyST Markdown (a superset of CommonMark).
  - URL: [https://pypi.org/project/myst-parser/](https://pypi.org/project/myst-parser/)
- **`pre-commit>=4.0.1`**: A framework for managing and maintaining multi-language pre-commit hooks.
  - URL: [https://pypi.org/project/pre-commit/](https://pypi.org/project/pre-commit/)
- **`pytest<8`**: A mature, full-featured Python testing tool.
  - Purpose: Used for writing and running unit, integration, and functional tests.
  - Key Features: Fixtures, parameterized testing, rich plugin architecture, detailed test reports.
  - CHAP-Core Usage: The `tests/` directory contains pytest-style tests.
  - Learning URL: [https://docs.pytest.org/](https://docs.pytest.org/)
- **`pytest-cov>=5.0.0`**: Pytest plugin for measuring code coverage with Coverage.py.
  - URL: [https://pypi.org/project/pytest-cov/](https://pypi.org/project/pytest-cov/)
- **`pytest-mock>=3.14.0`**: Pytest wrapper for the `unittest.mock` patching library.
  - URL: [https://pypi.org/project/pytest-mock/](https://pypi.org/project/pytest-mock/)
- **`ruff>=0.6.9`**: An extremely fast Python linter and code formatter, written in Rust.
  - Purpose: Used for static code analysis (linting) to find errors and enforce code style, and for auto-formatting code.
  - Key Features: Speed, extensive rule set, auto-fix capabilities, integration with pre-commit.
  - CHAP-Core Usage: Configuration is present in `pyproject.toml` under `[tool.ruff]`.
  - Learning URL: [https://docs.astral.sh/ruff/](https://docs.astral.sh/ruff/)
- **`sphinx>=8.1.0`**: A tool that makes it easy to create intelligent and beautiful documentation.
  - URL: [https://pypi.org/project/Sphinx/](https://pypi.org/project/Sphinx/)
- **`wheel>=0.44.0`**: A built-package format for Python.
  - URL: [https://pypi.org/project/wheel/](https://pypi.org/project/wheel/)
- **`typer~=0.9.0`**: A library for building CLI applications with Python type hints, based on Click.
  - URL: [https://pypi.org/project/typer/](https://pypi.org/project/typer/)

---

This list provides a starting point. More details (10+ bullet points) can be added for each dependency if needed, focusing on their specific use within CHAP-Core as it becomes clearer through further exploration.
