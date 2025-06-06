import os
import subprocess
from importlib.resources import files as importlib_files
from pathlib import Path

import datasets
import docker
import yaml
from datasets import load_from_disk
from swesmith.build_repo.download_images import DOCKER_ORG, TAG
from swesmith.constants import MAP_REPO_TO_SPECS
from swesmith.harness.grading import TestStatus
from swesmith.harness.log_parsers import MAP_REPO_TO_PARSER, parse_log_pytest
from swesmith.harness.utils import get_test_command
from swesmith.utils import get_repo_commit_from_image_name
from tqdm import tqdm

from debug_gym.constants import DEBUG_GYM_CACHE_DIR
from debug_gym.gym.entities import EvalOutput
from debug_gym.gym.envs.env import RepoEnv
from debug_gym.gym.terminal import DockerTerminal, Terminal
from debug_gym.gym.utils import create_ignore_file


class SWESmithEnv(RepoEnv):
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
        terminal = terminal or DockerTerminal(logger=kwargs.get("logger"))
        if not isinstance(terminal, DockerTerminal):
            raise ValueError("SWESmithEnv only supports DockerTerminal.")

        super().__init__(terminal=terminal, **kwargs)

        self.dataset_id = dataset_id
        self.split = split
        self.instance_ids = instance_ids
        SWESmithEnv.DUMMY_DIR.mkdir(parents=True, exist_ok=True)

        self.load_dataset()
        self.session_commands = []
        self.test_directives = []

    @property
    def instructions(self):
        return {
            **super().instructions,
            "Problem description": self.ds_row["problem_statement"],
        }

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
        self.logger.debug(
            f"Loaded {len(self.ds)} tasks accross {len(swesmith_repos)} repos from {self.dataset_id}."
        )

        # Download all images needed for SWE-Smith.
        client = docker.from_env()
        image_names = set(self.ds["image_name"])
        tagged_image_names = set(f"{DOCKER_ORG}/{name}:{TAG}" for name in image_names)

        existing_images = set(
            tag for image in client.images.list() for tag in image.tags
        )
        missing_images = tagged_image_names - existing_images
        if missing_images:
            self.logger.debug(f"Found {len(missing_images)} missing Docker images.")
            for image_name in tqdm(missing_images, desc="Pulling images for SWE-Smith"):
                docker_hub_image = image_name.replace("__", "_1776_")
                client.images.pull(docker_hub_image)
                # Rename images via tagging
                client.images.get(docker_hub_image).tag(image_name)

        # Load dataset splits.
        with open(SWESmithEnv.CONFIG) as f:
            self.dataset_splits = yaml.safe_load(f)

    def get_dataset_split(self, split):
        if split == "all":
            return sorted(self.dataset.keys())  # all tasks
        elif split in self.dataset:
            return [split]  # Single task
        elif split in self.dataset_splits:
            return self.dataset_splits[split]["ids"]
        else:
            raise ValueError(
                f"Invalid split '{split}'. Available splits are: {['all'] + sorted(self.dataset_splits.keys())}"
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
        self.swesmith_repo_name = self.ds_row["repo"].split("/")[1]
        self.base_commit = self.ds_row["base_commit"]
        self.branch_name = self.ds_row["instance_id"]
        self.bug_patch = self.ds_row["patch"]
        self.gold_patch = self.ds_row[
            "patch"
        ]  # Buggy code patch but will be used in conjunction with --reverse.
        self.git_apply_args = "--reverse"
        self.image_name = self.ds_row["image_name"]
        self.repo, self.commit = get_repo_commit_from_image_name(self.image_name)
        self.install_configs = MAP_REPO_TO_SPECS[self.repo][self.commit]
        self.base_image = f"{DOCKER_ORG}/{self.image_name}:{TAG}"
        self.package_name = self.repo.split("/")[1]
        self.test_cmd, self.test_directives = get_test_command(self.ds_row)
        self.fail_to_pass = self.ds_row["FAIL_TO_PASS"]
        self.pass_to_pass = self.ds_row["PASS_TO_PASS"]

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
        elif self.package_name == "alive-progress":
            self.install_configs["install"].append("pip uninstall -y pdbpp")

        # The following will create the temporary working directory.
        self.setup_workspace(
            # Empty folder. The actual codebase will come from the docker image.
            path=SWESmithEnv.DUMMY_DIR,
            entrypoint=self.test_cmd,
            # -q (quiet) from pytest, to avoid long pytest output
            debug_entrypoint=self.test_cmd.replace("pytest", "pytest -q"),
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

    @property
    def patch(self):
        command = "git diff"
        result = subprocess.run(
            command.split(), cwd=self.working_dir, text=True, capture_output=True
        )
        patch = result.stdout.replace(str(self.working_dir), str(self.path))
        return patch

    def calculate_score(self, eval_output: EvalOutput) -> int:
        log_parser = MAP_REPO_TO_PARSER.get(self.repo, parse_log_pytest)
        test_status_map = log_parser(eval_output.output)
        score = sum(
            1
            for test in self.fail_to_pass
            # Like in SWE-Smith, we assume silent success.
            # Ref: https://github.com/SWE-bench/SWE-smith/blob/main/swesmith/harness/grading.py#L154
            if test_status_map.get(test, TestStatus.PASSED.value)
            in (TestStatus.PASSED.value, TestStatus.XFAIL.value)
        )
        # Getting not passed tests.
        not_passed_tests = {
            test: status
            for test, status in test_status_map.items()
            if status not in (TestStatus.PASSED.value, TestStatus.XFAIL.value)
        }
        self.logger.debug(f"Not passed tests: {not_passed_tests}")
        assert score <= self.max_score
        return score

    def reset(self, *, options: dict | None = None):
        # TODO: support reset current task, i.e. no options provided.
        options = options or {}

        # Clean up the previous task, if any.
        self.close()

        self.setup_task(options["task_name"])

        # Start the terminal
        self.terminal.base_image = self.base_image

        self.logger.info(f"Configuring docker container: {self.terminal.container}")

        # Create new group (if needed) and user.
        uid = os.getuid()
        group_id = os.getgid()
        self.terminal.run(f"groupadd -g {group_id} debug_gym_group", user="root")
        self.terminal.run(
            f"useradd -m -u {uid} -g {group_id} -G sudo debug_gym_user", user="root"
        )

        # Install sudo.
        self.terminal.run(f"apt update && apt install -y sudo", user="root")
        # Add the user to sudoers.
        self.terminal.run(
            f"echo 'debug_gym_user ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/debug_gym_user",
            user="root",
        )

        # Allow for the user to pip install in the env. TODO: This is still slow.
        # self.terminal.run(f"chmod -R o+rwX /opt/miniconda3/envs/testbed", user="root")

        # Alternatively, we can use the following to specifically allow read/write/execute permissions on certain directories.
        self.terminal.run(
            f"chmod -R o+rwX /opt/miniconda3/envs/testbed/bin", user="root"
        )
        self.terminal.run(
            f"chmod o+rwX /opt/miniconda3/envs/testbed/lib/python*/site-packages",
            user="root",
        )
        self.terminal.run(
            f"chmod o+rwX /opt/miniconda3/envs/testbed/lib/python*/site-packages/*",
            user="root",
        )
        self.terminal.run(
            f"chmod -R o+rwX /opt/miniconda3/envs/testbed/lib/python*/site-packages/{self.package_name}*",
            user="root",
        )
        self.terminal.run(
            f"chmod -R o+rwX /opt/miniconda3/envs/testbed/lib/python*/site-packages/{self.package_name.title()}*",
            user="root",
        )
        self.terminal.run(
            f"chmod -R o+rwX /opt/miniconda3/envs/testbed/lib/python*/site-packages/*pdb*",
            user="root",
        )

        # Delete the content in the working directory.
        self.terminal.run(f"rm -rf {self.working_dir / '*'} {self.working_dir / '.*'}")

        # Copy the initial code to the working directory.
        self.terminal.run(f"cp -r /testbed/. {self.working_dir}")
        self.terminal.run(f"chmod -R a+rw {self.working_dir}")

        self.terminal.session_commands.append("source /opt/miniconda3/bin/activate")
        self.terminal.session_commands.append(f"conda activate testbed")

        self.terminal.run(f"pip install uv")
        self.run_install()

        ## Checkout the branch for the current task.
        # self.terminal.run(f"git fetch origin {self.branch_name}")
        # self.terminal.run(f"git checkout {self.branch_name}")

        # Apply the bug patch directly.
        self.terminal.run(f"git apply - <<'EOF'\n{self.bug_patch}\nEOF")

        self.terminal.run(f"git config user.name 'debug-gym'")
        self.terminal.run(f"git config user.email '<>'")
        self.terminal.run(f"git commit -am 'Applying buggy patch {self.branch_name}'")

        # Rebuild the debug ignore and read-only files.
        create_ignore_file(
            self.working_dir / ".debugignore", patterns=self.ignore_files
        )

        # # Get test directives from test patch and remove non-test files
        # test_files = re.findall(r"diff --git a/.* b/(.*)", self.gold_patch)
        # test_files = [
        #     f for f in test_files if not any(f.endswith(ext) for ext in NON_TEST_EXTS)
        # ]
        # # Add test/ to readonly files if not already present
        # if "test/" not in test_files:
        #     test_files.append("test/")
        # TODO: do we want to add all test/ ?, if so check that the gold_patch is not about fixing a bug in such file.
        create_ignore_file(
            self.working_dir / ".debugreadonly", patterns=self.test_directives
        )
        self.setup_file_filters()  # Need to refresh the file filters after re-creating ignore files.

        self.terminal.run(f"git add .debugignore")
        self.terminal.run(f"git add .debugreadonly")
        self.terminal.run(f"git commit -am 'Add debug-gym ignore and read-only files'")

        # Reset RepoEnv
        self.max_score = len(self.fail_to_pass)
        infos = super().reset(options=options)
        assert not self.done, "Tests should be failing before debugging."

        return infos

    @property
    def ignore_files(self):
        return [
            ".?*",  # Hidden files and directories. It also ignores the parent directory.
        ]

    def run_command_with_raise(self, command):
        try:
            command = command.replace("apt-get", "sudo apt-get").replace(
                "sudo sudo", "sudo"
            )
            command = command.replace("pip install -U", "pip install --no-deps")
            status, output = self.terminal.run(command, raises=True)
            return status, output
        except ValueError as e:
            if "error: remote upstream already exists." in str(e):
                pass
            # else:
            #     raise

    def run_install(self):
        install_cmds = self.install_configs.get("install", [])
        if install_cmds:
            self.logger.debug("Running install commands...")
            for install_cmd in install_cmds:
                self.run_command_with_raise(install_cmd)
