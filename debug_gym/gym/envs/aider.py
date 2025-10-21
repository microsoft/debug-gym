import os
import subprocess
import tempfile
from pathlib import Path

import debug_gym.gym.utils as utils
from debug_gym.constants import DEBUG_GYM_CACHE_DIR
from debug_gym.gym.entities import EvalOutput
from debug_gym.gym.envs.env import RepoEnv
from debug_gym.gym.terminals.docker import DockerTerminal
from debug_gym.gym.terminals.terminal import Terminal

DOCKER_AIDER_IMAGE_NAME = "debug-gym:aider"


def build_docker_image(logger):
    """
    Build a Docker image for the Mini Nightmare environment.
    """
    # Check if Docker image is built.
    import docker

    docker_client = docker.from_env(timeout=600)
    try:
        docker_client.images.get(DOCKER_AIDER_IMAGE_NAME)
        return
    except docker.errors.ImageNotFound:
        pass

    logger.info(f"Docker image {DOCKER_AIDER_IMAGE_NAME} not found. Building it...")

    # Starts from the official Python 3.12 slim image
    base_image = "python:3.12-slim"
    # Then install git and the required Python packages
    setup_commands = [
        "apt update",
        "apt install -y git tree",
        "pip install pytest",
    ]
    # Create a temporary Dockerfile
    with tempfile.TemporaryDirectory() as build_dir:
        dockerfile_path = Path(build_dir) / "Dockerfile"
        with open(dockerfile_path, "w") as dockerfile:
            dockerfile.write(f"FROM {base_image}\n")
            for command in setup_commands:
                dockerfile.write(f"RUN {command}\n")

        # Build the Docker image using docker client
        image, build_logs = docker_client.images.build(
            path=str(build_dir),
            dockerfile="Dockerfile",
            tag=DOCKER_AIDER_IMAGE_NAME,
            rm=True,
        )

    logger.info(f"Docker image {DOCKER_AIDER_IMAGE_NAME} built successfully.")


class AiderBenchmarkEnv(RepoEnv):
    REPO_URL = "https://github.com/exercism/python"
    REPO_PATH = DEBUG_GYM_CACHE_DIR / "exercism"

    def __init__(
        self,
        entrypoint: str = "python -m pytest --tb=no -s .",
        terminal: Terminal | None = None,
        **kwargs,
    ):
        terminal = terminal or DockerTerminal(
            base_image=DOCKER_AIDER_IMAGE_NAME,
            logger=kwargs.get("logger"),
        )
        if hasattr(terminal, "base_image") and terminal.base_image is None:
            terminal.base_image = DOCKER_AIDER_IMAGE_NAME

        super().__init__(entrypoint=entrypoint, terminal=terminal, **kwargs)

    @property
    def instructions(self) -> str:
        return self.current_task["instructions"]

    def calculate_max_score(self, eval_output: EvalOutput) -> int:
        return utils.extract_max_score_from_pytest_output(eval_output.output)

    def calculate_score(self, eval_output: EvalOutput) -> int:
        return utils.extract_reward_from_pytest_output(eval_output.output)

    def eval(self, **kwargs) -> EvalOutput:
        success, output = self.terminal.run(self.entrypoint, timeout=self.run_timeout)
        output = utils.cleanup_pytest_output(output)
        self.last_eval = EvalOutput(success, output)
        return self.last_eval

    def setup_task(self, task_name: str, options: dict = None):
        if task_name not in self.dataset:
            raise ValueError(f"Task {task_name} not found in the dataset.")
        self.current_task = self.dataset[task_name]

    def setup_workspace(self):
        self.workspace.reset()

        self.logger.info("Copying files..")
        self.workspace.copy_content(
            src=self.current_task["codebase"], target=self.workspace.working_dir
        )
        self.workspace.setup_file_filters()  # Use codebase's .debugignore and .debugreadonly.

    def setup_terminal(self):
        self.logger.info(f"Configuring {self.terminal}...")

        self.terminal.run("git init")
        self.terminal.run("git config user.name 'debug-gym'")
        self.terminal.run("git config user.email '<>'")

        self.terminal.run(
            "git add *.py *.txt"
        )  # Aider tasks only have Python and text files.
        self.terminal.run("git commit -am 'Init'")

        self.terminal.run(
            "git add .debugignore .debugreadonly"
        )  # Aider tasks come with those.
        self.terminal.run("git commit -am 'Add debug-gym ignore and read-only files'")

    def load_dataset(self, problems: str | list[str] | None = None):
        if isinstance(self.terminal, DockerTerminal):
            build_docker_image(self.logger)

        if not os.path.exists(self.REPO_PATH):
            subprocess.run(["git", "clone", self.REPO_URL, self.REPO_PATH], check=True)

        practice_path = self.REPO_PATH / "exercises" / "practice"
        directories = [d for d in practice_path.iterdir() if d.is_dir()]

        dataset = {}
        for directory in directories:
            task_name = directory.name.replace("-", "_")

            docs = directory / ".docs"
            intro_md = docs / "introduction.md"
            instr_md = docs / "instructions.md"
            instr_more_md = docs / "instructions.append.md"
            instructions = ""
            instructions += intro_md.read_text() if intro_md.exists() else ""
            instructions += instr_md.read_text() if instr_md.exists() else ""
            instructions += instr_more_md.read_text() if instr_more_md.exists() else ""

            # Add .debugignore so all files are ignored except Python files.
            utils.create_ignore_file(
                directory / ".debugignore",
                patterns=[
                    ".?*",  # Ignore hidden files and directories but not current dir "."
                    "__pycache__/",
                    "*.pyc",
                ],
            )
            # Add .debugreadonly so tests are readonly.
            utils.create_ignore_file(
                directory / ".debugreadonly", patterns=["*test*.py"]
            )

            dataset[task_name] = {
                "codebase": directory,
                "instructions": instructions,
                "filename": task_name + ".py",
            }

        problems = utils.filter_problems(dataset, problems)
        dataset = {id: i for id, i in dataset.items() if id in problems}
        return dataset
