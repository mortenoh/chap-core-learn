# Running `chap-core` with Docker

This document outlines various methods for building, running, and testing the `chap-core` project using Docker and Docker Compose. This project utilizes several Dockerfiles and Docker Compose configurations to manage different environments and tasks.

## Prerequisites

- Docker Engine installed (e.g., Docker Desktop).
- Docker Compose installed (usually included with Docker Desktop).
- Access to a terminal or command prompt.
- The `chap-core` project code checked out.

## Core Docker Files

The project contains the following key Docker-related files:

- **`Dockerfile`**: The primary Dockerfile for building the main application image (API, workers).
- **`Dockerfile.inla`**: A specialized Dockerfile, likely for running models requiring R and the INLA package.
- **`Dockerfile.test`**: Dockerfile for a testing environment.
- **`Dockerfile.integrationtest`**: Dockerfile for an integration testing environment.
- **`compose.yml`**: Docker Compose file for a production-like or default deployment.
- **`compose.dev.yml`**: Docker Compose file for the development environment.
- **`compose.test.yml`**: Docker Compose file for running general tests.
- **`compose.integration.test.yml`**: Docker Compose file for running integration tests.
- **`compose.db.yml`**: Docker Compose file likely defining the database service (PostgreSQL), often included by other compose files.
- **`external_models/Dockerfile`**: Dockerfile for containerizing external models.
- **`.dockerignore`**: Specifies files and directories to exclude from Docker image builds.

## 1. Development Environment

The development environment is typically set up for local development, often with features like volume mounting for live code reloading.

**To run the development environment:**

It's common to use an override file like `compose.dev.yml` on top of a base `compose.yml` and `compose.db.yml`.

```bash
# Ensure .env file is configured if needed by the application

# Start services defined in compose.yml, compose.db.yml, and overridden by compose.dev.yml
docker-compose -f compose.yml -f compose.db.yml -f compose.dev.yml up --build
```

- `--build`: Forces a rebuild of the images if they are outdated or Dockerfiles have changed.
- This command will typically start the API service, a database (e.g., PostgreSQL), Redis (for Celery), and Celery workers.
- The API might be accessible at `http://localhost:8000` (or as configured).
- Code changes in mounted volumes (if configured in `compose.dev.yml`) might trigger automatic reloading of the FastAPI application.

**To stop the development environment:**

```bash
docker-compose -f compose.yml -f compose.db.yml -f compose.dev.yml down
```

## 2. Production-like Environment

This setup is intended to simulate a production deployment.

**To run the production-like environment:**

```bash
# Ensure .env file is configured with production settings if applicable

# Start services defined in compose.yml and compose.db.yml
docker-compose -f compose.yml -f compose.db.yml up --build -d
```

- `-d`: Runs containers in detached mode (in the background).
- This setup usually doesn't mount local code volumes to ensure the container runs exactly what was built into the image.

**To stop the production-like environment:**

```bash
docker-compose -f compose.yml -f compose.db.yml down
```

## 3. Running Tests

The project has dedicated configurations for running tests in isolated Docker environments.

### a. General Tests (Unit/Component Tests)

These tests likely use `compose.test.yml`.

```bash
# Build and run test services
docker-compose -f compose.test.yml up --build --abort-on-container-exit
```

- `--abort-on-container-exit`: Stops all services if any container exits. This is useful for test runs, as the test runner container will exit once tests are complete. The exit code of the test runner will be the exit code of the `docker-compose` command.

### b. Integration Tests

Integration tests might require a more complex setup, potentially involving multiple services. The `Makefile` references `./tests/test_docker_compose_integration_flow.sh`.

**To run integration tests (using the script):**

```bash
# Navigate to the tests directory if needed, or run from project root
./tests/test_docker_compose_integration_flow.sh
```

This script likely uses `compose.integration.test.yml` and possibly other files to:

1.  Build necessary images.
2.  Start dependent services (e.g., a test database, Redis).
3.  Run the integration tests (e.g., `pytest` commands within a test container).
4.  Tear down the services.

Consult the content of `test_docker_compose_integration_flow.sh` for specific steps.

## 4. Building and Running Individual Services Manually

While Docker Compose is preferred for multi-container setups, you can build and run individual images.

### a. Building the Main Application Image

```bash
docker build -t chap-core-app .
# or specify a different Dockerfile:
# docker build -t chap-core-inla -f Dockerfile.inla .
```

### b. Running an Individual Container (e.g., API)

This is more complex as you need to manually manage networks, environment variables, and links to other services (like a database).

```bash
# Example (highly dependent on actual application needs and .env variables):
# First, ensure dependent services like PostgreSQL and Redis are running (e.g., via a separate docker-compose up for them)

# docker run -p 8000:8000 \
#   -e DATABASE_URL="postgresql://user:password@host.docker.internal:5432/chapdb" \
#   -e REDIS_URL="redis://host.docker.internal:6379/0" \
#   --name chap_api chap-core-app
```

_Note: `host.docker.internal` is a special DNS name for Docker Desktop that resolves to the internal IP address of the host._
_This manual approach is generally not recommended for `chap-core` due to its multi-service nature. Use Docker Compose._

## 5. Running External Models

External models, located in the `external_models/` directory, often have their own `Dockerfile` and/or `MLproject` files.

- **If an external model has a `Dockerfile`**:
  ```bash
  cd external_models/your_specific_model/
  docker build -t external-model-name .
  # Run command would be specific to the model's Dockerfile CMD or ENTRYPOINT
  docker run external-model-name <args_if_any>
  ```
- **If an external model uses an `MLproject` file (for MLflow)**:
  MLflow can run projects in Docker containers.
  ```bash
  # Example: Running an MLflow project that might use Docker
  mlflow run external_models/your_specific_model/ -P parameter1=value1
  ```
  The `MLproject` file itself might specify a Docker environment, which MLflow would then build and use.

## 6. Using Makefile for Docker Operations

The `Makefile` mentions scripts for running test flows with Docker Compose:

- `./tests/test_docker_compose_integration_flow.sh`

While the `Makefile` doesn't provide generic `docker-build` or `docker-up` targets, these scripts are the primary entry points for Docker-based testing workflows defined in the project. Always refer to these scripts for the most accurate way to run specific test environments.

---

_This document provides a general guide. Always refer to the specific `Dockerfile`, `compose._.yml` files, and any scripts (`.sh`, `Makefile`) in the project for the most accurate and up-to-date instructions.\*
