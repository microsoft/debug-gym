from concurrent.futures import ThreadPoolExecutor
from importlib.resources import files as importlib_files
from pathlib import Path

import datasets
import docker
import yaml
from datasets import load_from_disk
from swesmith.build_repo.download_images import DOCKER_ORG, TAG
from swesmith.harness.grading import TestStatus
from swesmith.profiles import global_registry

from debug_gym.constants import DEBUG_GYM_CACHE_DIR
from debug_gym.gym.entities import EvalOutput
from debug_gym.gym.envs.swe_bench import SWEBenchEnv
from debug_gym.gym.terminal import Terminal


class SWESmithEnv(SWEBenchEnv):
    CACHE = DEBUG_GYM_CACHE_DIR / "swe-smith"
    DUMMY_DIR = DEBUG_GYM_CACHE_DIR / "swe-smith" / "empty"
    CONFIG = (
        importlib_files("debug_gym") / "gym" / "envs" / "configs" / "swe_smith.yaml"
    )

    def __init__(
        self,
        dataset_id: str = "SWE-bench/SWE-smith",
        split: str = "train",
        instance_ids: list[str] | None = None,
        terminal: Terminal | None = None,
        **kwargs,
    ):
        super().__init__(
            dataset_id=dataset_id,
            split=split,
            instance_ids=instance_ids,
            terminal=terminal,
            **kwargs,
        )

    def load_dataset(self):
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
            self.ds = datasets.load_dataset(self.dataset_id)[self.split]

        self.dataset = {id: i for i, id in enumerate(self.ds["instance_id"])}

        # To avoid concurrency issues, we will clone all the repos in the dataset.
        swesmith_repos = set(self.ds["repo"])
        image_names = set(self.ds["image_name"])
        if self.instance_ids:
            # If instance_ids are provided, filter the dataset to only include those repos.
            swesmith_repos = set(
                self.ds[self.dataset[id]]["repo"] for id in self.instance_ids
            )
            image_names = set(
                self.ds[self.dataset[id]]["image_name"] for id in self.instance_ids
            )
        self.logger.debug(
            f"Loaded {len(self.ds)} tasks accross {len(swesmith_repos)} repos from {self.dataset_id}."
        )

        # Download all images needed for SWE-Smith.
        client = docker.from_env()
        tagged_image_names = set(f"{name}:{TAG}" for name in image_names)

        existing_images = set(
            tag for image in client.images.list() for tag in image.tags
        )
        missing_images = tagged_image_names - existing_images
        if missing_images:
            self.logger.info(f"Found {len(missing_images)} missing Docker images.")
            for image_name in missing_images:
                self.logger.info(f"Pulling Docker image `{image_name}`.")
                client.images.pull(image_name)

        # Load dataset splits.
        with open(SWESmithEnv.CONFIG) as f:
            self.dataset_splits = yaml.safe_load(f)
            self.excluded_ids = self.dataset_splits.get("excluded", [])

    def get_problem_ids(self, split_or_problem_id):
        if split_or_problem_id == "all":
            return sorted(
                k for k in self.dataset.keys() if k not in self.excluded_ids
            )  # all tasks
        elif split_or_problem_id in self.dataset:
            return [split_or_problem_id]  # Single task
        elif split_or_problem_id in self.dataset_splits:
            return self.dataset_splits[split_or_problem_id]
        else:
            raise ValueError(
                f"Invalid split or problem id: '{split_or_problem_id}'. Available splits are: {['all'] + sorted(self.dataset_splits.keys())}"
            )

    def setup_task(self, task_name):
        if self.instance_ids:
            if task_name not in self.instance_ids:
                raise ValueError(
                    f"Task `{task_name}` was not found in instance_ids. The available tasks are: {self.instance_ids}.\n"
                    "Please provide a valid task or initialize the environment without instance_ids to load all tasks."
                )

        self.task_name = task_name
        self.ds_row = self.ds[self.dataset[self.task_name]]
        self.base_commit = self.ds_row["base_commit"]
        self.branch_name = self.ds_row["instance_id"]
        self.test_patch = self.ds_row["patch"]
        self.image_name = self.ds_row["image_name"]

        self.repo_profile = global_registry[task_name]
        self.commit = self.repo_profile.commit
        self.package_name = self.repo_profile.repo

        self.install_configs = {"install": self.repo_profile.install_cmds}
        self.base_image = f"{DOCKER_ORG}/{self.image_name}:{TAG}"
        self.test_cmd = self.repo_profile.test_cmd
        self.test_directives = self.repo_profile._get_f2p_test_files(self.ds_row)
        self.fail_to_pass = self.ds_row["FAIL_TO_PASS"]
        self.pass_to_pass = self.ds_row["PASS_TO_PASS"]
        self.log_parser = self.repo_profile.log_parser

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

        # The following will create the temporary working directory.
        self.setup_workspace(
            # Empty folder. The actual codebase will come from the docker image.
            path=SWESmithEnv.DUMMY_DIR,
            # allow traceback to be printed in the output.
            entrypoint=self.test_cmd.replace("--tb=no", "--tb=short"),
            # -s (capture=no) from pytest, allows for debugging with pdb
            # -q (quiet) from pytest, to avoid long pytest output
            debug_entrypoint=self.test_cmd.replace("pytest", "pytest -sq"),
        )

        # Those changes depend on the working directory created by setup_workspace.
        if self.package_name == "gunicorn":
            self.fail_to_pass = [
                test.replace("/testbed/", f"{self.working_dir}/")
                for test in self.fail_to_pass
            ]
            self.pass_to_pass = [
                test.replace("/testbed/", f"{self.working_dir}/")
                for test in self.pass_to_pass
            ]

        self.git_apply_cmd = f"git -C {self.working_dir} apply --reverse -"
        # Note that the `gold_patch` is the same as the `test_patch` but will
        # be used in conjunction with --reverse.
        self.gold_patch = self.test_patch

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

    def run_post_install(self):
        pass  # SWE-Smith does not have post-install commands.
