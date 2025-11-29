from importlib.resources import files as importlib_files
from pathlib import Path
from typing import List

import docker
import yaml
from datasets import load_dataset, load_from_disk
from swesmith.build_repo.download_images import DOCKER_ORG, TAG
from swesmith.constants import MAP_REPO_TO_SPECS
from swesmith.harness.grading import TestStatus
from swesmith.harness.log_parsers import MAP_REPO_TO_PARSER, parse_log_pytest
from swesmith.harness.utils import get_test_command
from swesmith.utils import get_repo_commit_from_image_name

from debug_gym.constants import DEBUG_GYM_CACHE_DIR
from debug_gym.gym.entities import EvalOutput
from debug_gym.gym.envs.swe_bench import SWEBenchEnv
from debug_gym.gym.terminals.terminal import DebugGymLogger, Terminal
from debug_gym.gym.utils import filter_problems

from .swe_smith_constants import DOCKER_ORG, MAP_REPO_TO_SPECS, TAG
from .swe_smith_utils import get_repo_commit_from_image_name, get_test_command


class SWESmithEnv(SWEBenchEnv):
    CACHE = DEBUG_GYM_CACHE_DIR / "swe-smith"
    CONFIG = (
        importlib_files("debug_gym") / "gym" / "envs" / "configs" / "swe_smith.yaml"
    )

    def setup_task(self):
        self.base_commit = (
            self.task_data["base_commit"] if "base_commit" in self.task_data else "main"
        )
        self.branch_name = self.task_data["instance_id"]
        self.bug_patch = self.task_data["patch"]
        self.image_name = self.task_data["image_name"]
        self.repo, self.commit = get_repo_commit_from_image_name(self.image_name)
        self.install_configs = MAP_REPO_TO_SPECS[self.repo][self.commit]
        self.base_image = f"{DOCKER_ORG}/{self.image_name}:{TAG}"
        self.package_name = self.repo.split("/")[1]
        self.test_cmd, self.test_directives = get_test_command(self.task_data)
        self.fail_to_pass = self.task_data["FAIL_TO_PASS"]
        self.pass_to_pass = self.task_data["PASS_TO_PASS"]
        self.log_parser = MAP_REPO_TO_PARSER.get(self.repo, parse_log_pytest)

        if self.package_name == "python-colorlog":
            self.package_name = "colorlog"
        elif self.package_name == "MONAI":
            self.package_name = "monai"
        elif self.package_name == "mido":
            # ../dev doesn't exist in docker image
            self.test_cmd = self.test_cmd.replace("/dev/null", "/dev")
            self.test_cmd = self.test_cmd.replace("../dev/", "./")
            self.test_directives = [
                directive.replace("../dev/", "") for directive in self.test_directives
            ]
            self.fail_to_pass = [
                test.replace("../dev/", "") for test in self.fail_to_pass
            ]
            self.pass_to_pass = [
                test.replace("../dev/", "") for test in self.pass_to_pass
            ]
        elif self.package_name == "pydantic":
            self.test_cmd = self.test_cmd.replace("/root/", "$HOME/")
            self.install_configs["install"] = ["pip install uv"] + self.install_configs[
                "install"
            ]
        elif self.package_name == "alive-progress":
            # Removing pdbpp as it creates conflicts, i.e. we read until "(Pdb)" in the pdb tool.
            self.install_configs["install"].append("pip uninstall -y pdbpp")
        elif self.package_name == "conan":
            # Skip system packages installation (they are already installed in the Docker image).
            self.install_configs["install"] = ["python -m pip install ."]
        elif self.package_name == "autograd":
            # Disable pytest-xdist which interfers with pytest output.
            self.test_cmd = self.test_cmd.replace("--verbose", "-n 0 --verbose")
            # Since disabling pytest-xdist, no need for a special log parser.
            self.log_parser = parse_log_pytest

        # Filter out the command that removes tests files.
        self.install_configs["install"] = [
            cmd for cmd in self.install_configs["install"] if "rm tests/" not in cmd
        ]

        # Convert all "pip update" to normal "pip install" without dependencies.
        self.install_configs["install"] = [
            cmd.replace("pip install -U", "pip install --no-deps")
            for cmd in self.install_configs["install"]
        ]

        # Filter out the command that adds the upstream remote.
        self.install_configs["install"] = [
            cmd
            for cmd in self.install_configs["install"]
            if "git remote add upstream" not in cmd
        ]

        # allow traceback to be printed in the output.
        self.entrypoint = self.test_cmd.replace("--tb=no", "--tb=short")
        # -s (capture=no) from pytest, allows for debugging with pdb
        # -q (quiet) from pytest, to avoid long pytest output
        self.debug_entrypoint = self.test_cmd.replace("pytest", "pytest -sq")
        # Note that the `gold_patch` is the same as the `test_patch` but will
        # be used in conjunction with --reverse.
        self.git_apply_cmd = f"git apply --reverse -"
        self.gold_patch = self.bug_patch

    def setup_terminal(self):
        super().setup_terminal()

        # Apply bug patch.
        self.terminal.run(f"git apply - <<'EOF'\n{self.bug_patch}\nEOF", raises=True)
        self.terminal.run(f"git commit -am 'Applying bug patch for {self.task_name}'")

    def calculate_score(self, eval_output: EvalOutput) -> int:
        test_status_map = self.log_parser(eval_output.output)
        score = sum(
            1
            for test in self.fail_to_pass
            # *Do not* assume silent success for now as done in SWE-Smith grading.py
            # Ref: https://github.com/SWE-bench/SWE-smith/blob/main/swesmith/harness/grading.py#L154
            if test_status_map.get(test, TestStatus.ERROR.value)
            in (TestStatus.PASSED.value, TestStatus.XFAIL.value)
        )

        # Getting not passed tests.
        not_passed_tests = {
            test: status
            for test, status in test_status_map.items()
            if status not in (TestStatus.PASSED.value, TestStatus.XFAIL.value)
        }
        if not_passed_tests:
            self.logger.debug(f"Not passed tests: {not_passed_tests}")

        assert score <= self.max_score
        self.logger.debug(
            f"Score: {score}/{self.max_score} ({score/self.max_score:.1%})"
        )
        return score

    def eval(self, **kwargs) -> EvalOutput:
        success, output = self.terminal.run(self.entrypoint, timeout=self.run_timeout)
        self.last_eval = EvalOutput(success, output)
        return self.last_eval

    @classmethod
    def load_dataset(
        cls,
        dataset_id: str = "SWE-bench/SWE-smith",
        dataset_revision: str = "699b53400d3855206a0fbf3ff4beaf1a52f4f232",
        split: str = "train",
        problems: list | None = None,
        prepull_images: bool = False,
        logger: DebugGymLogger | None = None,
    ) -> dict:
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
        with open(SWESmithEnv.CONFIG) as f:
            custom_splits = yaml.safe_load(f)
            excluded_ids = custom_splits.get("excluded", [])

        dataset = {d["instance_id"]: d for d in ds}
        problems = filter_problems(dataset, problems, custom_splits, excluded_ids)
        dataset = {pid: dataset[pid] for pid in problems}

        image_names = set([problem["image_name"] for problem in dataset.values()])
        if logger is not None:
            logger.debug(
                f"Loaded {len(dataset)} tasks across {len(image_names)} Docker images from {dataset_id}."
            )

        if prepull_images:
            # Download all images needed for SWE-Smith.
            client = docker.from_env()
            tagged_image_names = set(
                f"{DOCKER_ORG}/{name}:{TAG}" for name in image_names
            )

            existing_images = set(
                tag for image in client.images.list() for tag in image.tags
            )
            missing_images = tagged_image_names - existing_images
            if missing_images:
                if logger is not None:
                    logger.info(f"Found {len(missing_images)} missing Docker images.")

                for image_name in missing_images:
                    docker_hub_image = image_name.replace("__", "_1776_")
                    if logger is not None:
                        logger.info(
                            f"Pulling Docker image `{docker_hub_image}` to `{image_name}`."
                        )
                    client.images.pull(docker_hub_image)
                    # Rename images via tagging
                    client.images.get(docker_hub_image).tag(image_name)
        return dataset
