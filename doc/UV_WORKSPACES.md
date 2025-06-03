# Migration Plan to `uv` Workspaces for GitHub CI

This document outlines a plan to transition the existing GitHub CI workflows to leverage `uv` workspaces. The goal is to enhance dependency management, potentially simplify build processes, and improve the overall organization of our Python projects within the CI/CD pipeline.

## General Principles for Migration

1.  **Centralized Dependency Management**: Utilize `uv` workspace features (e.g., a root `pyproject.toml` with workspace declarations, shared lock files where appropriate) to manage dependencies across different components or packages.
2.  **`uv` Commands**: Replace existing dependency installation and tool execution commands (e.g., `pip install`, `python -m build`, direct script calls) with their `uv` equivalents (`uv sync`, `uv run`, `uv build`).
3.  **Workspace Structure**: Assume the project will be organized into a `uv` workspace, potentially with multiple packages (e.g., core library, worker, external models, docs). CI workflows will need to be aware of this structure.
4.  **Incremental Adoption**: The migration can be phased, starting with workflows that benefit most or are easiest to adapt.

## Workflow-Specific Migration Plan

### 1. Docker Image Builds (`build_chap_docker_image.yml`, `build_chap_worker_docker_image.yml`)

- **Current**: Uses `Dockerfile` and `Dockerfile.inla`. Dependency installation within Docker is not explicitly detailed in the workflow YAML but assumed to be part of the Dockerfiles.
- **Migration Plan**:
  - Ensure `uv` is installed in the base Docker images.
  - Modify `Dockerfile` and `Dockerfile.inla` to use `uv sync` or `uv pip install --system` (if appropriate for the Docker context) to install dependencies based on the workspace's `pyproject.toml` and `uv.lock`.
  - If building specific packages from the workspace, adjust `uv` commands to target those packages.
  - The build context for Docker might need to include the relevant workspace configuration files.

### 2. Sphinx Documentation Build (`build_sphinx_website.yml`)

- **Current**: Uses `uv sync --dev` and `uv run sphinx-build`.
- **Migration Plan**:
  - This workflow is already `uv`-centric.
  - Verify that `uv run sphinx-build` correctly operates within the context of a `uv` workspace, especially if the documentation source and the main library are separate packages within the workspace.
  - Ensure `uv sync --dev` installs all necessary dependencies for the documentation build, potentially from a dedicated `[project.optional-dependencies]` group for docs within the workspace `pyproject.toml`.

### 3. Version Bumping (`bumpversion.yml`)

- **Current**: Uses `jasonamyers/github-bumpversion-action`.
- **Migration Plan**:
  - `uv` workspaces are unlikely to directly impact this workflow's core logic.
  - However, if versioning becomes per-package within the workspace, the `bumpversion` configuration might need to be adapted to target the correct `pyproject.toml` files or version strings. For a unified version, no significant change is expected.

### 4. PyPI Publishing (`push-to-pypi.yml`)

- **Current**: Uses `uv sync --dev` and `python -m build`.
- **Migration Plan**:
  - Replace `python -m build` with `uv build` (or the equivalent `uv` command for building distributions like wheels and sdists).
  - If the workspace contains multiple publishable packages, the workflow will need to be parameterized or duplicated to build and publish each package individually, targeting it with `uv build --package <package-name>`.
  - Ensure `uv sync` installs build dependencies.

### 5. Python Installation and Testing (`python-install-and-test.yml`)

- **Current**: Uses `uv python install`, `uv sync --dev`, and `uv run pytest`.
- **Migration Plan**:
  - This workflow is already heavily reliant on `uv`.
  - With `uv` workspaces, `uv sync --dev` should correctly install dependencies for all packages in the workspace or for a specific package being tested.
  - `uv run pytest` might need to be adjusted to target tests for specific packages within the workspace (e.g., `uv run pytest packages/my_package/tests`) or run tests across the entire workspace.
  - The matrix strategy for Python versions and OS should be maintained.

### 6. Ruff Linting (`ruff.yml`)

- **Current**: Uses `astral-sh/ruff-action@v1`.
- **Migration Plan**:
  - The `ruff-action` should continue to work.
  - If `ruff` configuration (`ruff.toml` or `pyproject.toml [tool.ruff]`) is defined at the workspace root and applies to all packages, no major changes are needed.
  - If individual packages within the workspace have distinct `ruff` configurations, ensure the action or `ruff` itself correctly discovers and applies these. `uv` might offer ways to run `ruff` across all workspace packages.

### 7. Comprehensive Testing (`test-external-models.yml`)

- **Current**: Uses `uv sync --dev` and `make test-all`.
- **Migration Plan**:
  - Audit the `make test-all` script. Any steps involving Python environment setup or package installation should be transitioned to `uv` commands.
  - If "external models" are treated as separate packages within the `uv` workspace, this can help isolate their dependencies and testing environments.
  - `uv sync` should handle the dependency setup for these tests.
  - `uv run` can be used to execute test scripts or commands defined in the `Makefile`, ensuring they operate within the `uv`-managed environment.

## Next Steps

1.  **Research `uv` Workspaces**: Gain a deeper understanding of `uv`'s specific features and best practices for workspace management.
2.  **Proof of Concept**: Implement `uv` workspaces for a small subset of the project or a single CI workflow to identify challenges.
3.  **Update `pyproject.toml`**: Restructure the project's `pyproject.toml` (and potentially create new ones for sub-packages) to define the workspace structure.
4.  **Iterative Rollout**: Gradually update each CI workflow according to this plan, testing thoroughly at each step.
5.  **Documentation**: Update project documentation to reflect the new `uv` workspace structure and CI processes.
