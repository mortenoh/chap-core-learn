# CHAP-Core High-Level Architecture (Component Diagram)

This document contains a textual description of the CHAP-core project's high-level architecture in Mermaid syntax. You can render this into an image using a Mermaid-compatible renderer (e.g., online editors like mermaid.live, or integrated tools).

## Mermaid Diagram Code

```mermaid
graph TD;
    %% Define major user-facing interfaces
    subgraph UserInterfaces ["User Interfaces"]
        direction LR
        CLI["Command Line Interface\n(chap_core.cli, chap_core.adaptors.command_line_interface)"]
        REST_API["REST API\n(chap_core.rest_api_src, chap_core.adaptors.rest_api)"]
    end

    %% Define the core library and its main internal components
    subgraph CoreLogic ["CHAP Core Library (chap_core)"]
        direction TB
        DataHandling["Data Handling\n(datatypes, data, spatio_temporal_data)"]
        ClimateIntegration["Climate Data Integration\n(climate_data, google_earth_engine)"]
        ModelingEngine["Modeling Engine\n(models, predictor, model_spec)"]
        Assessment["Assessment & Evaluation\n(assessment)"]
        WorkerSystem["Asynchronous Worker System\n(worker, Celery/Redis)"]
        DatabaseInterface["Database Interface\n(database)"]
        CoreUtilities["Core Utilities\n(util, exceptions, log_config, geojson, geometry, geoutils)"]
    end

    %% Define external systems and configurations
    subgraph ExternalSystemsAndConfig ["External Systems & Configuration"]
        direction LR
        ExternalModels["External Models Execution\n(external_models/, Docker)"]
        Configuration["Project Configuration\n(config/)"]
        Database["Persistent Storage\n(SQL Database)"]
        GEEService["Google Earth Engine\n(External Service)"]
        DockerEnv["Docker Environment"]
    end

    %% Define relationships
    CLI --> CoreLogic;
    REST_API --> CoreLogic;

    CoreLogic --> DatabaseInterface;
    DatabaseInterface --> Database;

    CoreLogic --> ExternalModels;
    ExternalModels -- uses --> DockerEnv;

    DataHandling -.-> ClimateIntegration;
    ClimateIntegration --> GEEService;

    ModelingEngine --> DataHandling;
    ModelingEngine -.-> ClimateIntegration; % Models might use climate features
    ModelingEngine -.-> ExternalModels; % Core might trigger external models

    Assessment --> ModelingEngine;
    Assessment --> DataHandling;

    WorkerSystem -.-> ModelingEngine; % Workers run modeling tasks
    WorkerSystem -.-> DatabaseInterface; % Workers update status/results
    REST_API -.-> WorkerSystem; % API offloads tasks to workers

    CoreLogic -- reads --> Configuration;
    CoreLogic -- uses --> CoreUtilities;

    %% Styling (optional, makes it visually distinct)
    classDef userInterface fill:#c9f,stroke:#333,stroke-width:2px;
    classDef coreComponent fill:#lightgrey,stroke:#333,stroke-width:2px;
    classDef externalSystem fill:#ccf,stroke:#333,stroke-width:2px;
    classDef dataStore fill:#cfc,stroke:#333,stroke-width:2px;

    class CLI,REST_API userInterface;
    class DataHandling,ClimateIntegration,ModelingEngine,Assessment,WorkerSystem,DatabaseInterface,CoreUtilities coreComponent;
    class ExternalModels,Configuration,GEEService,DockerEnv externalSystem;
    class Database dataStore;
```

## Explanation of Components:

- **User Interfaces**:

  - `Command Line Interface`: Allows users to interact with CHAP-core functionalities via terminal commands. Built using `cyclopts` and defined in `chap_core.cli` and `chap_core.adaptors.command_line_interface`.
  - `REST API`: Exposes CHAP-core functionalities over HTTP. Built using `FastAPI` and defined in `chap_core.rest_api_src` and `chap_core.adaptors.rest_api`.

- **CHAP Core Library (`chap_core`)**: The central part of the application containing the main logic.

  - `Data Handling`: Modules responsible for defining and managing core data structures, especially time series and geospatial data (`chap_core.datatypes`, `chap_core.data`, `chap_core.spatio_temporal_data`).
  - `Climate Data Integration`: Modules for fetching, processing, and representing climate data from sources like Google Earth Engine (`chap_core.climate_data`, `chap_core.google_earth_engine`).
  - `Modeling Engine`: Components for defining model specifications, training predictive models, and making predictions (`chap_core.models`, `chap_core.predictor`, `chap_core.model_spec`).
  - `Assessment & Evaluation`: Modules for evaluating model performance, including dataset splitting, metrics calculation, and backtesting (`chap_core.assessment`).
  - `Asynchronous Worker System`: Manages long-running tasks like model training or complex data processing, likely using Celery and Redis (`chap_core.worker`).
  - `Database Interface`: Handles interactions with the persistent storage for data, model metadata, results, etc. (`chap_core.database`).
  - `Core Utilities`: General utility functions, custom exceptions, logging configuration, and geospatial utilities (`chap_core.util`, `chap_core.exceptions`, `chap_core.log_config`, `chap_core.geojson`, `chap_core.geometry`, `chap_core.geoutils`).

- **External Systems & Configuration**:
  - `External Models Execution`: Manages the execution of models defined externally to the core Python library, often using Docker containers (`external_models/`).
  - `Project Configuration`: Stores system-wide and model-specific configurations (`config/`).
  - `Persistent Storage (Database)`: The actual database (e.g., PostgreSQL, SQLite) used by the `Database Interface`.
  - `Google Earth Engine (External Service)`: An external service for accessing satellite imagery and geospatial datasets, including climate data.
  - `Docker Environment`: The containerization platform used to run CHAP-core components and external models.

## Key Interactions:

- User Interfaces (CLI, REST API) interact with the `CHAP Core Library` to trigger actions.
- The `REST API` often offloads computationally intensive tasks to the `Asynchronous Worker System`.
- The `Core Logic` (especially `Modeling Engine`, `Data Handling`, `Worker System`) interacts with the `Database Interface` to read and write data.
- `Climate Data Integration` components fetch data from external services like `Google Earth Engine`.
- The `Modeling Engine` can trigger `External Models Execution`, which often run within the `Docker Environment`.
- All core components may read from `Project Configuration`.

This diagram provides a high-level overview. Each component can be further broken down into more detailed diagrams.
