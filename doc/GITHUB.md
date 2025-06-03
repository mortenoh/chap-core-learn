# GitHub CI Overview

This document provides an overview of the GitHub Actions workflows used in this project. These workflows automate various tasks such as building, testing, and publishing the project.

## Workflows

### 1. `build_chap_docker_image.yml`

- **Purpose**: Builds and publishes the `dhis2-chap/chap-core` Docker image.
- **Trigger**: Pushes to the `dev` branch.
- **Key Steps**:
  - Checks out the repository.
  - Logs in to GitHub Container Registry (ghcr.io).
  - Extracts metadata (tags, labels) for the Docker image.
  - Builds the Docker image using `Dockerfile`.
  - Pushes the image to ghcr.io.
  - Generates an artifact attestation for supply chain security.

### 2. `build_chap_worker_docker_image.yml`

- **Purpose**: Builds and publishes the `dhis2-chap/chap-worker-with-inla` Docker image.
- **Trigger**: Pushes to the `dev` branch.
- **Key Steps**:
  - Similar to `build_chap_docker_image.yml` but uses `Dockerfile.inla` for building the worker image.

### 3. `build_sphinx_website.yml`

- **Purpose**: Builds the Sphinx documentation and deploys it to GitHub Pages.
- **Trigger**: Pushes to `master` or `dev` branches.
- **Key Steps**:
  - Checks out the repository.
  - Sets up Python 3.10.
  - Installs project dependencies using `uv sync --dev`.
  - Builds HTML documentation using `sphinx-build`.
  - Commits the generated documentation in the `docs/` directory.
  - Pushes the changes to the `gh-pages` branch.

### 4. `bumpversion.yml`

- **Purpose**: Bumps the project version and pushes a new tag.
- **Trigger**: Successful completion of the "Build and test" workflow on the `master` branch.
- **Key Steps**:
  - Checks out the repository.
  - Uses `jasonamyers/github-bumpversion-action` to increment the version.
  - Pushes the new tag and version changes to the `master` branch.

### 5. `push-to-pypi.yml`

- **Purpose**: Builds and publishes the Python package to PyPI.
- **Trigger**: Successful completion of the "Bump version" workflow on the `master` branch.
- **Key Steps**:
  - Checks out the repository.
  - Sets up Python 3.10.
  - Installs project dependencies using `uv sync --dev`.
  - Builds the package using `python -m build`.
  - Publishes the package to PyPI using `pypa/gh-action-pypi-publish`.

### 6. `python-install-and-test.yml` (Named: "Build and test")

- **Purpose**: Installs Python dependencies and runs tests.
- **Trigger**: Pushes or pull requests to `master` or `dev` branches.
- **Key Steps**:
  - Checks out the repository.
  - Sets up multiple Python versions (3.10, 3.11, 3.12) on different operating systems (Ubuntu, Windows, macOS).
  - Installs project dependencies using `uv sync --dev`.
  - Runs tests using `pytest`.
  - Utilizes Google Cloud service account secrets for tests requiring them.

### 7. `ruff.yml`

- **Purpose**: Runs the Ruff linter to check code style and quality.
- **Trigger**: Pushes or pull requests.
- **Key Steps**:
  - Checks out the repository.
  - Uses `astral-sh/ruff-action` to run Ruff.

### 8. `test-external-models.yml` (Named: "Test-all (docker, external models, etc)")

- **Purpose**: Runs comprehensive tests including Docker builds and external model evaluations.
- **Trigger**: Pushes or pull requests to `master` or `dev` branches.
- **Key Steps**:
  - Checks out the repository.
  - Sets up Python 3.10 on Ubuntu.
  - Installs project dependencies using `uv sync --dev`.
  - Installs `pyenv`.
  - Executes `make test-all` which likely runs a series of tests defined in the Makefile, covering Docker, external models, etc.
  - Utilizes Google Cloud service account secrets.

## Disabled Workflows

- `manuscript.yml.disabled`: Purpose and functionality unknown as it is currently disabled.
- `r-models-evaluation.yml.disabled`: Likely for evaluating R models, but currently disabled.

This overview should help in understanding the CI/CD processes automated via GitHub Actions for this project.
