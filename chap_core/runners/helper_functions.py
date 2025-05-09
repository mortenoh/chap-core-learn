import logging
from pathlib import Path
from typing import Literal, Optional

import yaml

from chap_core.external.model_configuration import ModelTemplateConfig
from chap_core.models.model_template import ModelConfiguration
from chap_core.runners.command_line_runner import CommandLineRunner, CommandLineTrainPredictRunner
from chap_core.runners.docker_runner import DockerRunner, DockerTrainPredictRunner
from chap_core.runners.mlflow_runner import MlFlowTrainPredictRunner
from chap_core.runners.runner import TrainPredictRunner

logger = logging.getLogger(__name__)


def get_train_predict_runner_from_model_template_config(
    model_template_config: ModelTemplateConfig,
    working_dir: Path,
    skip_environment: bool = False,
    model_configuration: Optional[ModelConfiguration] = None,
) -> TrainPredictRunner:
    """
    Constructs a TrainPredictRunner from a model template config.

    Supports Docker, MLflow, or raw command-line execution depending on the template.

    Args:
        model_template_config: The model template config object.
        working_dir: Path to working directory.
        skip_environment: If True, forces CommandLineRunner (no Docker or MLflow).
        model_configuration: Optional model parameters to serialize and inject into commands.

    Returns:
        A TrainPredictRunner instance.
    """
    # Determine which environment type is specified
    if model_template_config.docker_env is not None:
        runner_type = "docker"
    elif model_template_config.python_env is not None:
        runner_type = "mlflow"
    else:
        runner_type = ""
        skip_environment = True  # No environment config = fall back to command line

    logger.info(f"skip_environment: {skip_environment}, runner_type: {runner_type}")

    # Extract train/predict command strings
    train_command = model_template_config.entry_points.train.command
    predict_command = model_template_config.entry_points.predict.command

    # If model configuration is passed in, write it to a YAML file and append to command args
    if model_configuration is not None:
        model_configuration_file = working_dir / "model_configuration_for_run.yaml"
        with open(model_configuration_file, "w") as file:
            yaml.dump(model_configuration.model_dump(), file)

        train_command += f" --model_configuration {model_configuration_file}"
        predict_command += f" --model_configuration {model_configuration_file}"

    if skip_environment:
        # Run locally using CommandLineRunner
        return CommandLineTrainPredictRunner(CommandLineRunner(working_dir), train_command, predict_command)

    # If runner is Docker-based
    if runner_type == "docker":
        docker_image = model_template_config.docker_env.image
        logger.info(f"Docker image is {docker_image}")
        command_runner = DockerRunner(docker_image, working_dir)
        return DockerTrainPredictRunner(command_runner, train_command, predict_command)

    # Only MLflow is left — model_configuration not supported here
    assert model_configuration is None, "ModelConfiguration not supported with mlflow runner"
    assert runner_type == "mlflow"
    return MlFlowTrainPredictRunner(working_dir)


def get_train_predict_runner(
    mlproject_file: Path, runner_type: Literal["mlflow", "docker"], skip_environment: bool = False
) -> TrainPredictRunner:
    """
    Load an MLproject file and construct the appropriate TrainPredictRunner.

    Args:
        mlproject_file: Path to MLproject YAML.
        runner_type: Type of runner ("mlflow" or "docker").
        skip_environment: If True, ignore environment fields and run locally.

    Returns:
        A TrainPredictRunner.
    """
    logger.info(f"skip_environment: {skip_environment}, runner_type: {runner_type}")

    # Extract working directory from file
    working_dir = mlproject_file.parent

    # Load YAML from MLproject
    with open(mlproject_file, "r") as file:
        data = yaml.load(file, Loader=yaml.FullLoader)

    train_command = data["entry_points"]["train"]["command"]
    predict_command = data["entry_points"]["predict"]["command"]

    if skip_environment:
        # Use local shell commands
        return CommandLineTrainPredictRunner(CommandLineRunner(working_dir), train_command, predict_command)

    if runner_type == "docker":
        # Ensure docker_env section exists
        assert "docker_env" in data, "Runner type is docker, but no docker_env in mlproject file"
        docker_image = data["docker_env"]["image"]
        logger.info(f"Docker image is {docker_image}")
        command_runner = DockerRunner(docker_image, working_dir)
        return DockerTrainPredictRunner(command_runner, train_command, predict_command)

    # Default fallback: mlflow
    assert runner_type == "mlflow"
    return MlFlowTrainPredictRunner(working_dir)
