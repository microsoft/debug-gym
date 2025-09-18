import json
import re
from importlib.resources import files as importlib_files
from pathlib import Path

import datasets
import docker
import yaml
from datasets import load_from_disk

from debug_gym.constants import DEBUG_GYM_CACHE_DIR
from debug_gym.gym.entities import EvalOutput
from debug_gym.gym.envs.env import RepoEnv
from debug_gym.gym.terminals.docker import DockerTerminal
from debug_gym.gym.terminals.kubernetes import KubernetesTerminal
from debug_gym.gym.terminals.terminal import Terminal
from debug_gym.gym.utils import filter_problems


def decolor_dict_keys(key):
    """Remove ANSI escape codes"""
    # Ref: https://github.com/R2E-Gym/R2E-Gym/blob/main/src/r2egym/repo_analysis/execution_log_parser.py#L68
    decolor = lambda key: re.sub(r"\u001b\[\d+m", "", key)
    return {decolor(k): v for k, v in key.items()}


def parse_log_pytest(log: str | None) -> dict[str, str]:
    """
    Parser for test logs generated with pytest framework

    Args:
        log (str): log content
    Returns:
        dict: test case to test status mapping
    """
    # Ref: https://github.com/R2E-Gym/R2E-Gym/blob/main/src/r2egym/repo_analysis/execution_log_parser.py#L4
    if log is None:
        return {}
    test_status_map = {}
    if "short test summary info" not in log:
        return test_status_map
    log = log.split("short test summary info")[1]
    log = log.strip()
    log = log.split("\n")
    for line in log:
        if "PASSED" in line:
            test_name = ".".join(line.split("::")[1:])
            test_status_map[test_name] = "PASSED"
        elif "FAILED" in line:
            test_name = ".".join(line.split("::")[1:]).split(" - ")[0]
            test_status_map[test_name] = "FAILED"
        elif "ERROR" in line:
            try:
                test_name = ".".join(line.split("::")[1:])
            except IndexError:
                test_name = line
            test_name = test_name.split(" - ")[0]
            test_status_map[test_name] = "ERROR"
    return test_status_map


class R2EGymEnv(RepoEnv):
    CACHE = DEBUG_GYM_CACHE_DIR / "r2e-gym"
    CONFIG = importlib_files("debug_gym") / "gym" / "envs" / "configs" / "r2egym.yaml"

    def __init__(
        self,
        dataset_id: str = "R2E-Gym/R2E-Gym-Lite",
        dataset_revision: str = "8d3163011f01f9393bb3dc7700497a79a8686ae5",
        split: str = "train",
        terminal: Terminal | None = None,
        **kwargs,
    ):
        terminal = terminal or DockerTerminal(logger=kwargs.get("logger"))
        if not isinstance(terminal, (DockerTerminal, KubernetesTerminal)):
            raise ValueError(
                "R2EGymEnv only supports DockerTerminal and KubernetesTerminal."
            )

        self.dataset_id = dataset_id
        self.dataset_revision = dataset_revision
        self.split = split
        self.session_commands = []
        self.test_directives = []

        super().__init__(terminal=terminal, **kwargs)

    @property
    def instructions(self) -> str:
        # try getting the content inside of [ISSUE] [/ISSUE] using regex tags for ds['problem_statement'] else return ds['problem_statement']
        # ref: https://github.com/R2E-Gym/R2E-Gym/blob/main/src/r2egym/agenthub/runtime/docker.py#L592
        try:
            content = self.ds_row["problem_statement"]
            return re.search(r"\[ISSUE\](.*)\[/ISSUE\]", content, re.DOTALL).group(1)
        except Exception as e:
            return self.ds_row["problem_statement"]

    def load_dataset(self, problems: str | list[str] | None = None):
        if Path(self.dataset_id).is_file() and self.dataset_id.endswith(".json"):
            # Loading from local JSON file.
            self.ds = datasets.load_dataset("json", data_files=self.dataset_id)[
                self.split
            ]
        elif Path(self.dataset_id).is_dir():
            # Loading from local folder.
            self.ds = load_from_disk(self.dataset_id)[self.split]
        else:
            # Loading from HuggingFace or a folder.
            self.ds = datasets.load_dataset(
                self.dataset_id, revision=self.dataset_revision
            )[self.split]

        # Load custom dataset splits from config.
        with open(R2EGymEnv.CONFIG) as f:
            custom_splits = yaml.safe_load(f)
            excluded_ids = custom_splits.get("excluded", [])

        dataset = {
            id.split("/", 1)[-1]: i for i, id in enumerate(self.ds["docker_image"])
        }
        problems = filter_problems(dataset, problems, custom_splits, excluded_ids)
        dataset = {id: i for id, i in dataset.items() if id in problems}

        image_names = set(self.ds[dataset[id]]["docker_image"] for id in dataset)
        self.logger.debug(
            f"Loaded {len(dataset)} tasks accross {len(image_names)} Docker images from {self.dataset_id}."
        )

        if not isinstance(self.terminal, KubernetesTerminal):
            # Download all images needed for R2E-Gym.
            client = docker.from_env()

            existing_images = set(
                tag for image in client.images.list() for tag in image.tags
            )
            missing_images = image_names - existing_images
            if missing_images:
                self.logger.warning(
                    f"Found {len(missing_images)} missing Docker images."
                )
                for i, image_name in enumerate(missing_images):
                    self.logger.warning(
                        f"Pulling Docker image {i + 1}/{len(missing_images)} `{image_name}`."
                    )
                    client.images.pull(image_name)

        return dataset

    def setup_task(self, task_name: str, options: dict = None):
        if task_name not in self.dataset:
            raise ValueError(
                f"Task `{task_name}` was not found in dataset. The available tasks are: {self.dataset}.\n"
                "Please provide a valid task or initialize the environment without problems to load all tasks."
            )

        self.task_name = task_name
        self.ds_row = self.ds[self.dataset[self.task_name]]
        self.base_image = self.ds_row["docker_image"]
        self.package_name = self.ds_row["repo_name"]
        self.expected_output = json.loads(self.ds_row["expected_output_json"])
        self.expected_output = decolor_dict_keys(self.expected_output)
        self.expected_output = {
            k.split(" - ")[0]: self.expected_output[k]
            for k in sorted(self.expected_output.keys())
        }

        self.commit_hash = self.ds_row["commit_hash"]

        self.entrypoint = "python -m pytest -W ignore -rA r2e_tests"
        if self.package_name == "pillow":
            test_file_codes = json.loads(self.ds_row["execution_result_content"])[
                "test_file_codes"
            ]
            if any(["unittest" in test_code for test_code in test_file_codes]):
                self.entrypoint = "python r2e_tests/unittest_custom_runner.py -W ignore"
        elif self.package_name == "orange3":
            self.entrypoint = "xvfb-run --auto-servernum .venv/bin/python -m pytest -rA r2e_tests -W ignore"
            self.terminal.env_vars["QT_QPA_PLATFORM"] = "minimal"

        elif self.package_name == "tornado":
            self.entrypoint = "python r2e_tests/tornado_unittest_runner.py -W ignore"

        elif self.package_name == "scrapy":
            # Reduce socket's timeout to speed up tests.
            # Ref: https://github.com/scrapy/scrapy/blob/master/tests/__init__.py#L27
            self.terminal.env_vars["RES_OPTIONS"] = "timeout:1 attempts:1"

        self.terminal.env_vars["PYTHONWARNINGS"] = (
            "ignore::UserWarning,ignore::SyntaxWarning"
        )

        # -s (capture=no) with pytest allows for debugging with pdb
        # -q (quiet) with pytest avoids long pytest output
        self.debug_entrypoint = self.entrypoint.replace("pytest", "pytest -sq")

        self.git_apply_cmd = f"git apply -"

    def setup_workspace(self):
        self.terminal.task_name = self.task_name
        self.terminal.base_image = self.base_image
        # Ignore hidden files (dotfiles) and any contents under hidden directories
        self.workspace.reset(ignore_patterns=["**/.*"])
        self.set_entrypoints(self.entrypoint, self.debug_entrypoint)

    def setup_terminal(self):
        self.logger.info(f"Configuring {self.terminal}...")

        # Install tree for listdir.
        self.terminal.run("apt update && apt install -y tree")

        # Follow r2egym setup for non- swe-bench/swe-smith tasks.
        # Ref: https://github.com/R2E-Gym/R2E-Gym/blob/main/src/r2egym/agenthub/runtime/docker.py#L545

        self.repo_path = "/testbed"
        self.alt_path = "/root"

        # create a symlink from repo_path/.venv to /root/.venv
        self.terminal.run(f"ln -s {self.repo_path}/.venv {self.alt_path}/.venv")

        self.terminal.run(
            f"ln -s {self.repo_path}/.venv/bin/python {self.alt_path}/.local/bin/python"
        )
        self.terminal.run(
            f"ln -s {self.repo_path}/.venv/bin/python {self.alt_path}/.local/bin/python3"
        )
        self.terminal.run(
            f"find {self.repo_path}/.venv/bin -type f -executable -exec ln -sf {{}} {self.alt_path}/.local/bin/ \\;"
        )

        self.terminal.run("uv pip install chardet")

        self.terminal.run("find . -name '*.pyc' -delete")
        self.terminal.run("find . -name '__pycache__' -exec rm -rf {} +")

        # also delete pycache and pyc from /r2e_tests
        self.terminal.run("find /r2e_tests -name '*.pyc' -delete")
        self.terminal.run("find /r2e_tests -name '__pycache__' -exec rm -rf {} +")

        # move all skip files (if present) to /root
        SKIP_FILES_NEW = [
            "run_tests.sh",
            "r2e_tests",
        ]
        for skip_file in SKIP_FILES_NEW:
            self.terminal.run(
                f"mv {self.repo_path}/{skip_file} {self.alt_path}/{skip_file}"
            )

        # r2e_tests are in the / directory, move them to /root
        self.terminal.run(f"mv /r2e_tests {self.alt_path}/r2e_tests")

        # make a softlink for /root/r2e_tests (if present)
        self.terminal.run(f"ln -s {self.alt_path}/r2e_tests {self.repo_path}/r2e_tests")

        self.terminal.session_commands.append("source .venv/bin/activate")

        self.terminal.run("git config user.name 'debug-gym'")
        self.terminal.run("git config user.email '<>'")

        # Get the gold patch.
        _, self.gold_patch = self.terminal.run(
            f"git diff HEAD {self.commit_hash}", raises=True
        )

        # Remove the remote so the agent won't see newer commits.
        # TODO: remove .git/ entirely?
        self.terminal.run("git remote remove origin")

    def apply_gold_patch(self):
        self.logger.info(f"Applying gold patch to {self.working_dir}.")
        command = self.git_apply_cmd + f" <<'EOF'\n{self.gold_patch}\nEOF"
        self.terminal.run(command, raises=True)
        self.logger.info("Patch applied successfully.")

    def eval(self, **kwargs) -> EvalOutput:
        """Evaluates the current code using the provided entrypoint.
        Sets the last_eval and returns it.
        Override in subclasses for different behavior."""
        success, output = self.terminal.run(self.entrypoint, timeout=self.run_timeout)

        # success, output = self.terminal.run(f"bash {self.alt_path}/run_tests.sh", timeout=self.run_timeout)
        # Remove ANSI escape codes and \r characters
        output = re.sub(r"\x1b\[[0-9;]*m|\r", "", output)
        self.last_eval = EvalOutput(success, output)
        return self.last_eval

    def calculate_max_score(self, eval_output: EvalOutput) -> int:
        # return len([1 for k, v in self.expected_output.items() if v])
        return 1

    def calculate_score(self, eval_output: EvalOutput) -> int:
        parse = parse_log_pytest(eval_output.output)
        parse = decolor_dict_keys(parse)
        parse = {k.split(" - ")[0]: parse[k] for k in sorted(parse.keys())}

        # Compare
        if len(parse) != len(self.expected_output):
            reward = 0
        else:
            # If ANY mismatch, reward = 0, else = 1
            match = True
            for k in parse.keys():
                if not k:
                    continue
                if k not in self.expected_output:
                    match = False
                    break
                if parse[k] != self.expected_output[k]:
                    match = False
                    break
            reward = 1 if match else 0

        return reward
