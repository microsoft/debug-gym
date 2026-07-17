import json
import re
import shlex
from importlib.resources import files as importlib_files
from pathlib import Path

import docker
import yaml
from datasets import load_dataset, load_from_disk

from debug_gym.constants import DEBUG_GYM_CACHE_DIR
from debug_gym.gym.entities import EvalDetails, EvalOutput
from debug_gym.gym.envs.env import RepoEnv
from debug_gym.gym.terminals.docker import DockerTerminal
from debug_gym.gym.terminals.kubernetes import KubernetesTerminal
from debug_gym.gym.terminals.terminal import Terminal
from debug_gym.gym.utils import filter_problems
from debug_gym.logger import DebugGymLogger


def decolor_dict_keys(key):
    """Remove ANSI escape codes"""
    # Ref: https://github.com/R2E-Gym/R2E-Gym/blob/main/src/r2egym/repo_analysis/execution_log_parser.py#L68
    decolor = lambda key: re.sub(r"\u001b\[\d+m", "", key)
    return {decolor(k): v for k, v in key.items()}


def _parse_log_pytest_with_optional_passes(
    log: str | None,
) -> tuple[dict[str, str], set[str]]:
    """
    Parser for test logs generated with pytest framework

    Args:
        log (str): log content
    Returns:
        tuple: test statuses and expected-failure tests that may be absent from
            legacy expected-output maps
    """
    # Ref: https://github.com/R2E-Gym/R2E-Gym/blob/main/src/r2egym/repo_analysis/execution_log_parser.py#L4
    if log is None:
        return {}, set()
    test_status_map = {}
    optional_passes = set()
    if "short test summary info" not in log:
        return test_status_map, optional_passes
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
        elif "XFAIL" in line or "XPASS" in line:
            test_name = ".".join(line.split("::")[1:]).split(" - ")[0]
            test_status_map[test_name] = "PASSED"
            optional_passes.add(test_name)
    return test_status_map, optional_passes


def parse_log_pytest(log: str | None) -> dict[str, str]:
    test_status_map, _ = _parse_log_pytest_with_optional_passes(log)
    return test_status_map


class R2EGymEnv(RepoEnv):
    CACHE = DEBUG_GYM_CACHE_DIR / "r2e-gym"
    CONFIG = importlib_files("debug_gym") / "gym" / "envs" / "configs" / "r2egym.yaml"
    EVAL_COMMAND = "bash /root/run_tests.sh"

    def __init__(
        self,
        task_data: dict,
        terminal: Terminal | None = None,
        **kwargs,
    ):
        terminal = terminal or DockerTerminal(logger=kwargs.get("logger"))
        if not isinstance(terminal, (DockerTerminal, KubernetesTerminal)):
            raise ValueError(
                "R2EGymEnv only supports DockerTerminal and KubernetesTerminal."
            )

        super().__init__(task_data=task_data, terminal=terminal, **kwargs)
        self.session_commands = []

    @property
    def task_name(self) -> str:
        return self.task_data["instance_id"]

    @property
    def instructions(self) -> str:
        # try getting the content inside of [ISSUE] [/ISSUE] using regex tags for ds['problem_statement'] else return ds['problem_statement']
        # ref: https://github.com/R2E-Gym/R2E-Gym/blob/main/src/r2egym/agenthub/runtime/docker.py#L592
        try:
            content = self.task_data["problem_statement"]
            return re.search(r"\[ISSUE\](.*)\[/ISSUE\]", content, re.DOTALL).group(1)
        except Exception:
            return self.task_data["problem_statement"]

    def setup_task(self):
        self.base_image = self.task_data["docker_image"]
        self.package_name = self.task_data.get("repo_name", "")
        self.expected_output = None
        if "expected_output_json" in self.task_data:
            self.expected_output = self._parse_expected_output(
                self.task_data["expected_output_json"],
                "task_data['expected_output_json']",
            )

        self.commit_hash = self.task_data["commit_hash"]

        self.entrypoint = "python -m pytest -W ignore -rA r2e_tests"
        if self.package_name == "pillow":
            test_file_codes = json.loads(self.task_data["execution_result_content"])[
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

    @staticmethod
    def _parse_expected_output(expected_json: str, source: str) -> dict[str, str]:
        try:
            expected_output = json.loads(expected_json)
        except (json.JSONDecodeError, TypeError) as exc:
            raise ValueError(
                f"Invalid R2E expected output JSON from {source}."
            ) from exc

        expected_output = decolor_dict_keys(expected_output)
        return {
            key.split(" - ")[0]: expected_output[key] for key in sorted(expected_output)
        }

    def load_expected_output(self) -> None:
        if self.expected_output is not None:
            return

        expected_output_path = f"{self.alt_path}/expected_test_output.json"
        success, expected_json = self.terminal.run(
            f"cat {shlex.quote(expected_output_path)}", timeout=300
        )
        if not success:
            raise ValueError(
                "R2E task has no expected_output_json and "
                f"{expected_output_path} could not be read."
            )
        self.expected_output = self._parse_expected_output(
            expected_json, expected_output_path
        )

    def setup_workspace(self):
        self.terminal.task_name = self.task_name
        self.terminal.base_image = self.base_image
        self.workspace.reset()
        self.set_entrypoints(self.entrypoint, self.debug_entrypoint)

    def setup_terminal(self):
        self.logger.debug(f"Configuring {self.terminal}...")

        # Follow r2egym setup for non- swe-bench/swe-smith tasks.
        # Ref: https://github.com/R2E-Gym/R2E-Gym/blob/main/src/r2egym/agenthub/runtime/docker.py#L545

        self.repo_path = "/testbed"
        self.alt_path = "/root"

        # Quote paths for shell safety
        repo_path_q = shlex.quote(self.repo_path)
        alt_path_q = shlex.quote(self.alt_path)

        setup_commands = [
            f"ln -s {repo_path_q}/.venv {alt_path_q}/.venv",
            f"ln -s {repo_path_q}/.venv/bin/python {alt_path_q}/.local/bin/python",
            f"ln -s {repo_path_q}/.venv/bin/python {alt_path_q}/.local/bin/python3",
            f"find {repo_path_q}/.venv/bin -type f -executable -exec ln -sf {{}} {alt_path_q}/.local/bin/ \\;",
            "uv pip install chardet",
            "find . -name '*.pyc' -delete",
            "find . -name '__pycache__' -exec rm -rf {} +",
            (
                "if [ -d /r2e_tests ]; then "
                "find /r2e_tests -name '*.pyc' -delete; "
                "find /r2e_tests -name '__pycache__' -exec rm -rf {} +; "
                "fi"
            ),
            (
                f"if [ -f {repo_path_q}/run_tests.sh ]; then "
                f"mv {repo_path_q}/run_tests.sh {alt_path_q}/run_tests.sh; "
                "fi"
            ),
            (
                "if [ -d /r2e_tests ]; then "
                f"rm -rf {alt_path_q}/r2e_tests {repo_path_q}/r2e_tests; "
                f"mv /r2e_tests {alt_path_q}/r2e_tests; "
                f"ln -s {alt_path_q}/r2e_tests {repo_path_q}/r2e_tests; "
                "fi"
            ),
            "git config user.name 'debug-gym'",
            "git config user.email '<>'",
        ]
        self.terminal.run(setup_commands, timeout=300, raises=True)

        self.terminal.session_commands.append("source .venv/bin/activate")

        # Get the gold patch.
        _, self.gold_patch = self.terminal.run(
            f"git diff HEAD {self.commit_hash}", timeout=300, raises=True
        )

        # Remove the remote so the agent won't see newer commits.
        # TODO: remove .git/ entirely?
        self.terminal.run("git remote remove origin", timeout=300)
        self.load_expected_output()

    def apply_gold_patch(self):
        self.logger.debug(f"Applying gold patch to {self.working_dir}.")
        command = self.git_apply_cmd + f" <<'EOF'\n{self.gold_patch}\nEOF"
        self.terminal.run(command, raises=True)
        self.logger.debug("Patch applied successfully.")

    def eval(self, **kwargs) -> EvalOutput:
        """Evaluates the current code using the provided entrypoint.
        Sets the last_eval and returns it.
        Override in subclasses for different behavior."""
        success, output = self.terminal.run(self.EVAL_COMMAND, timeout=self.run_timeout)

        # Remove ANSI escape codes and \r characters
        output = re.sub(r"\x1b\[[0-9;]*m|\r", "", output)
        details = self._calculate_eval_details(output)
        self.last_eval = EvalOutput(success, output, details=details)
        return self.last_eval

    def calculate_max_score(self, eval_output: EvalOutput) -> int:
        # return len([1 for k, v in self.expected_output.items() if v])
        return 1

    def calculate_score(self, eval_output: EvalOutput) -> int:
        if eval_output.details is not None:
            return int(eval_output.details.reward)
        return int(self._calculate_eval_details(eval_output.output).reward)

    def _calculate_eval_details(self, output: str) -> EvalDetails:
        if self.expected_output is None:
            raise ValueError("R2E expected output has not been loaded.")

        parse, optional_passes = _parse_log_pytest_with_optional_passes(output)
        parse = decolor_dict_keys(parse)
        parse = {k.split(" - ")[0]: parse[k] for k in sorted(parse.keys())}
        optional_passes = set(decolor_dict_keys({key: None for key in optional_passes}))
        optional_passes = {key.split(" - ")[0] for key in optional_passes}
        graded_parse = {
            key: status
            for key, status in parse.items()
            if key in self.expected_output or key not in optional_passes
        }

        # Compare
        if len(graded_parse) != len(self.expected_output):
            reward = 0
        else:
            # If ANY mismatch, reward = 0, else = 1
            match = True
            for k in graded_parse.keys():
                if not k:
                    continue
                if k not in self.expected_output:
                    match = False
                    break
                if graded_parse[k] != self.expected_output[k]:
                    match = False
                    break
            reward = 1 if match else 0

        n_passed = sum(status == "PASSED" for status in parse.values())
        return EvalDetails(
            parsed_tests=parse,
            n_parsed=len(parse),
            n_passed=n_passed,
            n_failed=len(parse) - n_passed,
            reward=reward,
        )

    @classmethod
    def load_dataset(
        cls,
        dataset_id: str = "R2E-Gym/R2E-Gym-Lite",
        dataset_revision: str = "8d3163011f01f9393bb3dc7700497a79a8686ae5",
        split: str = "train",
        problems: list | None = None,
        prepull_images: bool = False,
        logger: DebugGymLogger | None = None,
        **kwargs,
    ) -> dict:
        logger = logger or DebugGymLogger("debug_gym")
        data_path = Path(dataset_id)

        if data_path.is_file():
            # Loading from local file.
            if data_path.suffix.lower() == ".json":
                ds = load_dataset("json", data_files=dataset_id)
            elif data_path.suffix.lower() == ".parquet":
                ds = load_dataset("parquet", data_files=dataset_id)
        elif data_path.is_dir():
            # Loading from local folder.
            ds = load_from_disk(dataset_id)
        else:
            # Loading from HuggingFace or a folder.
            ds = load_dataset(dataset_id, revision=dataset_revision)

        # Select the split.
        ds = ds[split]

        # Load custom dataset splits from config.
        with open(R2EGymEnv.CONFIG) as f:
            custom_splits = yaml.safe_load(f)
            excluded_ids = custom_splits.get("excluded", [])

        def extract_instance_id(docker_image: str) -> str:
            return docker_image.split("/", 1)[-1]

        id2idx = {
            extract_instance_id(docker_image): i
            for i, docker_image in enumerate(ds["docker_image"])
        }
        problems = filter_problems(id2idx, problems, custom_splits, excluded_ids)
        dataset = {problem: ds[id2idx[problem]] for problem in problems}

        # Add instance_id (name of the image) and env_type to each task_data.
        for instance_id, task_data in dataset.items():
            task_data["instance_id"] = instance_id
            task_data["env_type"] = "r2egym"

        image_names = set(task_data["docker_image"] for task_data in dataset.values())
        logger.debug(
            f"Loaded {len(dataset)} tasks across {len(image_names)} Docker images from {dataset_id}."
        )

        if prepull_images:
            # Download all images needed for R2E-Gym.
            client = docker.from_env()

            existing_images = set(
                tag for image in client.images.list() for tag in image.tags
            )
            missing_images = image_names - existing_images
            if missing_images:
                logger.warning(f"Found {len(missing_images)} missing Docker images.")
                for i, image_name in enumerate(missing_images):
                    logger.warning(
                        f"Pulling Docker image {i + 1}/{len(missing_images)} `{image_name}`."
                    )
                    client.images.pull(image_name)
        return dataset
