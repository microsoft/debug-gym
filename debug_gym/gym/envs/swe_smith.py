import os
import subprocess
from importlib.resources import files as importlib_files
from pathlib import Path

import datasets
import docker
import yaml
from swebench.harness.constants import NON_TEST_EXTS
from swesmith.build_repo.download_images import DOCKER_ORG, TAG
from swesmith.constants import MAP_REPO_TO_SPECS
from swesmith.harness.grading import TestStatus
from swesmith.harness.log_parsers import MAP_REPO_TO_PARSER, parse_log_pytest
from swesmith.harness.utils import get_test_command
from swesmith.utils import get_repo_commit_from_image_name
from tqdm import tqdm

from debug_gym.gym.entities import EvalOutput
from debug_gym.gym.envs.env import RepoEnv
from debug_gym.gym.terminal import DockerTerminal, Terminal
from debug_gym.gym.utils import create_ignore_file
from debug_gym.utils import DEBUG_GYM_CACHE_DIR, DEBUG_GYM_CONFIG_DIR


class SWESmithEnv(RepoEnv):
    CACHE = DEBUG_GYM_CACHE_DIR / "swe-smith"
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
        SWESmithEnv.CACHE.mkdir(parents=True, exist_ok=True)

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
        self.ds = datasets.load_dataset(self.dataset_id)[self.split]
        self.dataset = {
            instance_id: i for i, instance_id in enumerate(self.ds["instance_id"])
        }

        # To avoid concurrency issues, we will clone all the repos in the dataset.
        swesmith_repos = set(self.ds["repo"])
        self.logger.debug(
            f"Loaded {len(self.ds)} tasks accross {len(swesmith_repos)} repos from {self.dataset_id}."
        )
        swesmith_repo_names = [repo.split("/")[1] for repo in swesmith_repos]
        missing_repos = [
            repo
            for repo in swesmith_repo_names
            if not Path.exists(SWESmithEnv.CACHE / repo)
        ]
        if missing_repos:
            self.logger.debug("Cloning all repos needed for SWE-Smith...")
            for repo in tqdm(missing_repos, desc="Cloning repos needed for SWE-Smith"):
                self.clone_repo(repo_address=repo)

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
                client.images.get(docker_hub_image).tag(
                    image_name
                )  # Rename images via tagging

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

    def setup_local_repo(self):
        self.fail_to_pass = self.ds_row["FAIL_TO_PASS"]
        self.pass_to_pass = self.ds_row["PASS_TO_PASS"]
        self.test_cmd, self.test_directives = get_test_command(self.ds_row)

        local_repo_path = SWESmithEnv.CACHE / self.swesmith_repo_name
        assert local_repo_path.exists()

        entrypoint = " ".join([self.test_cmd, *self.test_directives])

        # -s (capture=no) from pytest, allows for debugging with pdb
        # -q (quiet) from pytest, to avoid long pytest output
        debug_entrypoint = entrypoint.replace("pytest", "pytest -sq")

        self.setup_workspace(
            # path=local_branch_path,
            path=local_repo_path,
            entrypoint=entrypoint,
            debug_entrypoint=debug_entrypoint,
        )

        # Checkout to base commit.
        try:
            command = f"git -C {self.working_dir} checkout {self.base_commit} -f"
            self.logger.info(f"Checking out to {self.base_commit}")
            cmd_output = subprocess.run(
                command.split(), check=True, capture_output=True
            )
            self.logger.debug(cmd_output)
        except subprocess.CalledProcessError as e:
            self.logger.debug(e)
            self.logger.debug(e.stderr)
            self.logger.debug(e.stdout)
            raise

        # create_ignore_file(
        #     self.working_dir / ".debugignore", patterns=self.ignore_files
        # )
        # create_ignore_file(
        #     self.working_dir / ".debugreadonly", patterns=self.test_directives
        # )

    def setup_task_info(self, task_name):
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
        self.gold_patch = self.ds_row["patch"]
        self.git_apply_args = "--reverse"
        self.image_name = self.ds_row["image_name"]
        self.repo, self.commit = get_repo_commit_from_image_name(self.image_name)
        self.install_configs = MAP_REPO_TO_SPECS[self.repo][self.commit]
        self.base_image = f"{DOCKER_ORG}/{self.image_name}:{TAG}"
        self.repo_name = self.repo.split("/")[1]

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

    def reset(self, *, options: dict | None = None):
        # TODO: support reset current task, i.e. no options provided.
        options = options or {}

        # Clean up the previous task, if any.
        self.close()

        self.setup_task_info(options["task_name"])
        self.setup_local_repo()

        # spec = make_test_spec(self.ds_row)
        # docker_client = docker.from_env()
        # build_instance_image(spec, docker_client, logger=None, nocache=False)

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
        # Allow for the user to pip install in the env. TODO: This is still slow.
        # self.terminal.run(f"chmod -R o+rwX /opt/miniconda3/envs/testbed", user="root")
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
            f"chmod -R o+rwX /opt/miniconda3/envs/testbed/lib/python*/site-packages/{self.repo_name}*",
            user="root",
        )
        self.terminal.run(
            f"chmod -R o+rwX /opt/miniconda3/envs/testbed/lib/python*/site-packages/{self.repo_name.title()}*",
            user="root",
        )

        # Delete the content in the working directory.
        self.terminal.run(f"rm -rf {self.working_dir / '*'}")
        self.terminal.run(f"rm -rf {self.working_dir / '.*'}")
        # Copy the initial code to the working directory.
        self.terminal.run(f"cp -r /testbed/. {self.working_dir}")

        self.terminal.run(f"git fetch origin {self.branch_name}")
        self.terminal.run(f"git checkout {self.branch_name}")

        self.terminal.session_commands.append("source /opt/miniconda3/bin/activate")
        self.terminal.session_commands.append(f"conda activate testbed")

        self.run_install()
        # self.run_post_install()

        # Apply test patch
        # command = f"git apply -"
        # subprocess.run(
        #     command.split(),
        #     cwd=self.working_dir,
        #     input=self.ds_row["test_patch"],
        #     stdout=subprocess.PIPE,
        #     stderr=subprocess.PIPE,
        #     text=True,
        #     check=True,
        # )

        # Need to recreate those files after copying the initial code.
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
        self._index_files()

        self.terminal.run(f"git config user.name 'debug-gym'")
        self.terminal.run(f"git config user.email '<>'")
        self.terminal.run(f"git add .debugignore")
        self.terminal.run(f"git add .debugreadonly")
        self.terminal.run(f"git commit -am 'Add debug-gym ignore files'")

        # Reset RepoEnv
        self.max_score = len(self.fail_to_pass)
        infos = super().reset(options=options)
        assert not self.done, "Tests should be failing before debugging."

        return infos

    def clone_repo(self, repo_address):
        org_name, repo_name = repo_address.split("/")
        repo_url = f"https://github.com/{repo_address.lstrip('/')}"
        local_repo_path = SWESmithEnv.CACHE / repo_name

        if not local_repo_path.exists():
            self.logger.info(f"Cloning {repo_url} into {local_repo_path}")
            subprocess.run(["git", "clone", repo_url, local_repo_path], check=True)

        return local_repo_path

    @property
    def ignore_files(self):
        return [
            ".*/",
            # ".pytest_cache/",
            # "*test*.py",
            # "*.pyc",
            # "*.md",
            # ".*",
        ]

    def run_command_with_raise(self, command):
        command = command.replace("apt-get", "sudo apt-get").replace(
            "sudo sudo", "sudo"
        )
        status, output = self.terminal.run(command, raises=True)
        return status, output

    def run_install(self):
        install_cmds = self.install_configs.get("install", [])
        if install_cmds:
            self.logger.debug("Running install commands...")
            for install_cmd in install_cmds:
                self.run_command_with_raise(install_cmd)
