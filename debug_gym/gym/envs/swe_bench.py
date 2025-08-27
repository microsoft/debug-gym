import json
import os
import subprocess

import datasets
import docker
from swebench.harness.constants import MAP_REPO_VERSION_TO_SPECS, TestStatus
from swebench.harness.docker_build import (
    build_env_images,
    build_instance_image,
    get_env_configs_to_build,
)
from swebench.harness.log_parsers import MAP_REPO_TO_PARSER
from swebench.harness.test_spec.python import get_test_directives
from swebench.harness.test_spec.test_spec import make_test_spec
from swebench.harness.utils import load_swebench_dataset

from debug_gym.constants import DEBUG_GYM_CACHE_DIR
from debug_gym.gym.entities import EvalOutput
from debug_gym.gym.envs.env import RepoEnv
from debug_gym.gym.terminal import DockerTerminal, Terminal
from debug_gym.gym.utils import create_ignore_file, filter_problems


class SWEBenchEnv(RepoEnv):
    CACHE = DEBUG_GYM_CACHE_DIR / "swe-bench"
    DUMMY_DIR = DEBUG_GYM_CACHE_DIR / "swe-bench" / "empty"

    def __init__(
        self,
        dataset_id: str = "princeton-nlp/SWE-bench_Verified",
        # dataset_id: str = "princeton-nlp/SWE-bench_lite",
        split: str = "test",
        terminal: Terminal | None = None,
        **kwargs,
    ):
        terminal = terminal or DockerTerminal(logger=kwargs.get("logger"))
        if not isinstance(terminal, DockerTerminal):
            raise ValueError("SWEBenchEnv only supports DockerTerminal.")

        self.DUMMY_DIR.mkdir(parents=True, exist_ok=True)
        self.dataset_id = dataset_id
        self.split = split
        self.session_commands = []
        self.test_directives = []

        super().__init__(terminal=terminal, **kwargs)

    @property
    def instructions(self) -> str:
        return self.ds_row["problem_statement"]

    def load_dataset(self, problems: str | list[str] | None = None):
        self.ds = datasets.load_dataset(self.dataset_id)[self.split]
        dataset = {id: i for i, id in enumerate(self.ds["instance_id"])}
        problems = filter_problems(dataset, problems)
        dataset = {id: i for id, i in dataset.items() if id in problems}

        swebench_instances = load_swebench_dataset(
            name=self.dataset_id, instance_ids=list(dataset)
        )
        docker_client = docker.from_env()

        try:
            env_configs_to_build = get_env_configs_to_build(
                docker_client, swebench_instances
            )
        # swe-bench catches docker.errors.ImageNotFound and raises Exception
        except BaseException:
            env_configs_to_build = True

        if env_configs_to_build:
            self.logger.debug("Building Docker env-level images for SWE-Bench...")
            build_env_images(
                docker_client,
                swebench_instances,
                force_rebuild=False,
                max_workers=24,
            )

        return dataset

    def setup_task(self, task_name: str, options: dict = None):
        if task_name not in self.dataset:
            raise ValueError(
                f"Task `{task_name}` was not found in dataset. The available tasks are: {self.dataset}.\n"
                "Please provide a valid task or initialize the environment without problems to load all tasks."
            )

        self.task_name = task_name
        self.ds_row = self.ds[self.dataset[self.task_name]]
        self.repo = self.ds_row["repo"]
        self.package_name = self.repo.split("/")[1]
        self.version = self.ds_row["version"]
        self.install_configs = MAP_REPO_VERSION_TO_SPECS[self.repo][self.version]
        self.gold_patch = self.ds_row["patch"]
        self.test_spec = make_test_spec(self.ds_row)
        self.base_image = self.test_spec.instance_image_key
        self.base_commit = self.ds_row["base_commit"]
        self.test_patch = self.ds_row["test_patch"]
        self.fail_to_pass = json.loads(self.ds_row["FAIL_TO_PASS"])
        self.pass_to_pass = json.loads(self.ds_row["PASS_TO_PASS"])
        self.test_cmd = self.install_configs["test_cmd"]
        self.test_directives = get_test_directives(self.ds_row)

        self.entrypoint = " ".join([self.test_cmd, *self.test_directives])

        if self.package_name == "sphinx" or self.package_name == "sympy":
            # use pytest instead of `sympy bin/test` and `sphinx tox` so pdb breakpoints work
            expression = " ".join(self.test_directives)
            self.debug_entrypoint = f"python -m pytest {expression}"
            # Install pytest if not already installed
            self.install_configs["install"] += " && python -m pip install pytest"

            if self.entrypoint.startswith("PYTHONWARNINGS"):
                # Move PYTHONWARNINGS from the entrypoint to the session commands
                export, remaining = self.entrypoint.split(" ", 1)
                self.session_commands.append(f"export {export}")
                self.entrypoint = remaining

        # -s (capture=no) with pytest allows for debugging with pdb
        # -q (quiet) with pytest avoids long pytest output
        self.debug_entrypoint = self.entrypoint.replace("pytest", "pytest -sq")

        # --tb=short with pytest keeps the output concise
        self.entrypoint = self.entrypoint.replace("--tb=no", "--tb=short")

        self.git_apply_cmd = f"git apply -"

        # Use SWE-Bench's test spec to build the instance image.
        build_instance_image(
            self.test_spec, docker.from_env(), logger=None, nocache=False
        )

    def setup_workspace(self):
        self.terminal.base_image = self.base_image
        self.workspace.reset()
        self.set_entrypoints(self.entrypoint, self.debug_entrypoint)

    def setup_terminal(self):
        self.logger.info(f"Configuring {self.terminal}...")

        # Install tree for listdir.
        self.terminal.run("apt update && apt install -y tree")

        self.terminal.session_commands.append("source /opt/miniconda3/bin/activate")
        self.terminal.session_commands.append("conda activate testbed")

        # Apply the test patch directly.
        self.terminal.run(f"git apply - <<'EOF'\n{self.test_patch}\nEOF")

        self.terminal.run("git config user.name 'debug-gym'")
        self.terminal.run("git config user.email '<>'")
        self.terminal.run(f"git commit -am 'Applying test patch for {self.task_name}'")

        # Remove the remote so the agent won't see newer commits.
        self.terminal.run("git remote remove origin")

    def apply_gold_patch(self):
        self.logger.info(f"Applying gold patch to {self.working_dir}.")
        command = self.git_apply_cmd + f" <<'EOF'\n{self.gold_patch}\nEOF"
        self.terminal.run(command, raises=True)
        self.logger.info("Patch applied successfully.")

    def calculate_max_score(self, eval_output: EvalOutput) -> int:
        return len(self.fail_to_pass)

    def calculate_score(self, eval_output: EvalOutput) -> int:
        test_status_map = MAP_REPO_TO_PARSER[self.repo](
            eval_output.output, self.test_spec
        )
        self.logger.debug(f"fail_to_pass: {self.fail_to_pass}")
        self.logger.debug(f"Test status map: {test_status_map}")
        score = sum(
            1
            for test in self.fail_to_pass
            # *Do not* assume silent success for now as done in SWE-Bench grading.py
            if test_status_map.get(test, TestStatus.ERROR.value)
            in (TestStatus.PASSED.value, TestStatus.XFAIL.value)
        )
        assert score <= self.max_score
        return score
