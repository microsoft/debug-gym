import io
import logging
import os
import re
import shutil
import subprocess
import tempfile
from ast import literal_eval
from os.path import join as pjoin
from pathlib import Path

import datasets
import docker

from swebench.harness.constants import TestStatus
from swebench.harness.log_parsers import MAP_REPO_TO_PARSER
from swebench.harness.constants import MAP_REPO_VERSION_TO_SPECS
from swebench.harness.utils import load_swebench_dataset
from swebench.harness.test_spec import make_test_spec
from swebench.harness.docker_build import build_instance_image, build_env_images, get_env_configs_to_build
from tqdm import tqdm

import froggy.utils as utils
from froggy.envs.env import RepoEnv


class SWEBenchEnv(RepoEnv):

    def __init__(
        self,
        dataset_id: str = "princeton-nlp/SWE-bench_Verified",
        split: str = "test",
        base_image: str = "python:3.12",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.dataset_id = dataset_id
        self.split = split
        self.swe_bench_base_image = base_image
        self.swe_bench_repo_paths = Path.joinpath(Path.home(), ".cache", "froggy", "swe-bench")
        self.swe_bench_repo_paths.mkdir(parents=True, exist_ok=True)

        self.load_dataset()
        self.setup_commands = []

    @property
    def instructions(self):
        _instruction = {
            "Problem description": self.ds_row["problem_statement"],
            "Available tools to solve the problem": self.tool_instructions,
            "Available commands": self.actions_str,
        }
        return _instruction

    def load_dataset(self):
        self.ds = datasets.load_dataset(self.dataset_id)["test"]
        self.dataset = {row["instance_id"]: row for row in self.ds.sort("instance_id")}

        # To avoid concurrency issues, we will clone all the repos in the dataset.
        repos = sorted({task["repo"] for task in self.dataset.values()})
        repo_names = [repo.split("/")[1] for repo in repos]
        missing_repos = [repo for repo in repo_names if not Path.exists(self.swe_bench_repo_paths / repo)]
        if missing_repos:
            self.logger.debug("Cloning all repos needed for SWE-Bench...")
            for repo in tqdm(repos, desc="Cloning repos needed for SWE-Bench"):
                self.clone_repo(repo_address=repo)

        swebench_instances = load_swebench_dataset(name=self.dataset_id)
        docker_client = docker.from_env()
        if get_env_configs_to_build(docker_client, swebench_instances):
            self.logger.debug("Building Docker env-level images for SWE-Bench...")
            build_env_images(docker.from_env(), swebench_instances, force_rebuild=False, max_workers=24)

    def setup_local_repo(self):
        repo_address = self.ds_row["repo"]
        base_commit = self.ds_row["base_commit"]
        test_patch = self.ds_row["test_patch"]
        # TODO: use fail_to_pass and pass_to_pass
        self.fail_to_pass = literal_eval(self.ds_row["FAIL_TO_PASS"])
        self.pass_to_pass = literal_eval(self.ds_row["PASS_TO_PASS"])

        # Clone repository (should already be cloned)
        assert (self.swe_bench_repo_paths / repo_address.split("/")[1]).exists()
        local_repo_path = self.clone_repo(repo_address=repo_address)
        local_branch_path = local_repo_path.parent / self.ds_row["instance_id"]

        if not local_branch_path.exists():
            # Duplicate the repo to avoid changing the current branch.
            self.logger.info(f"Copying {local_repo_path} to {local_branch_path}")
            shutil.copytree(local_repo_path, local_branch_path, symlinks=True)

            # Checkout to base commit.
            command = f"git -C {local_branch_path} checkout {base_commit} -f"
            self.logger.info(f"Checking out to {base_commit}")
            subprocess.run(command.split(), check=True)

            # Apply test patch
            if test_patch != "":
                command = f"git -C {local_branch_path} apply -"
                subprocess.run(command.split(), input=test_patch, text=True, check=True)
                self.logger.info("Patch applied successfully.")

            # Make the pdb ignore
            self.make_froggyignore(local_repo_path=local_branch_path)
        else:
            self.logger.debug(
                f"Local checked out branch {local_branch_path} already exists."
            )

        # Depending on the SWE-Bench task instance, we need to patch fail_to_pass and pass_to_pass.
        if "django" in self.ds_row["instance_id"]:
            # test_zero_ip_addr (admin_scripts.tests.ManageRunserver) -> admin_scripts.tests.ManageRunserver.test_zero_ip_addr
            # Make regex to perform the replacement
            self.fail_to_pass = [
                re.sub(r"(.*) \((.*)\)", r"\2.\1", test) for test in self.fail_to_pass
            ]

        entrypoint = (
            self.install_configs["test_cmd"] + " " + " ".join(self.fail_to_pass)
        )  # + pass_to_pass)
        # # TODO: Find another way to extract inline env vars from entrypoint. Move to env_vars instead of setup_commands
        if entrypoint.startswith("PYTHONWARNINGS"):
            export, remaining = entrypoint.split(" ", 1)
            self.setup_commands.append(f"export {export}")
            entrypoint = remaining

        self.setup_workspace(local_branch_path, entrypoint)
        return local_branch_path, entrypoint

    def setup_task_info(self, task_name):
        self.task_name = task_name
        self.ds_row = self.dataset[self.task_name]
        self.repo = self.ds_row["repo"]
        self.version = self.ds_row["version"]
        self.install_configs = self.get_configs(self.repo, self.version)
        self.gold_patch = self.ds_row["patch"]

    def reset(self, *, seed=None, options: dict | None = None):
        options = options or {}
        self.setup_task_info(options["task_name"])
        self.setup_local_repo()

        spec = make_test_spec(self.ds_row)
        docker_client = docker.from_env()
        build_instance_image(spec, docker_client, logger=None, nocache=False)

        # Start the terminal
        self.terminal.base_image = spec.instance_image_key
        self.terminal.container

        # Create new group (if needed) and user.
        uid = os.getuid()
        group_id = os.getgid()
        self.terminal.run(f"groupadd -g {group_id} froggy_group", user="root")
        self.terminal.run(f"useradd -m -u {uid} -g {group_id} -G sudo froggy_user", user="root")

        # Delete the content in the working directory.
        self.terminal.run(f"rm -rf {self.working_dir / '*'}")
        # Copy the initial code to the working directory.
        self.terminal.run(f"cp -r /testbed/. {self.working_dir}")

        self.terminal.setup_commands.append("source /opt/miniconda3/bin/activate")
        self.terminal.setup_commands.append(f"conda activate testbed")
        self.terminal.setup_commands.append(f"echo testbed activated")

        self.run_install()
        self.run_post_install()

        # Reset RepoEnv
        obs, infos = super().reset()
        # TODO: probably needed cleanup specific to each SWE-Bench repo.
        # infos["last_run_obs"] = utils.cleanup_pytest_output(infos["last_run_obs"])

        self.max_score = len(self.fail_to_pass) #+ len(self.pass_to_pass)
        infos["max_score"] = self.max_score

        test_status_map = MAP_REPO_TO_PARSER[self.repo](infos["last_run_obs"])
        infos["score"] = sum(
            1
            for test in test_status_map
            if test_status_map[test] == TestStatus.PASSED.value
        )

        return infos["obs"], infos

    def step(self, action: str):
        obs, score, done, infos = super().step(action)

        # TODO: probably needed cleanup specific to each SWE-Bench repo.
        # infos["last_run_obs"] = utils.cleanup_pytest_output(infos["last_run_obs"])
        #infos["score"] = utils.extract_reward_from_pytest_output(infos["last_run_obs"])
        test_status_map = MAP_REPO_TO_PARSER[self.repo](infos["last_run_obs"])
        infos["score"] = sum(
            1
            for test in test_status_map
            if test_status_map[test] == TestStatus.PASSED.value
        )

        return obs, score, done, infos

    def clone_repo(self, repo_address):
        org_name, repo_name = repo_address.split("/")
        repo_url = f"https://github.com/{repo_address.lstrip('/')}"
        local_repo_path = self.swe_bench_repo_paths / repo_name

        # clone
        if not local_repo_path.exists():
            self.logger.info(f"Cloning {repo_url} into {local_repo_path}")
            subprocess.run(["git", "clone", repo_url, local_repo_path], check=True)

        return local_repo_path

    def make_froggyignore(self, local_repo_path, include_gitignore: bool = True):
        # Add an ignore file
        froggyignore_contents = "\n".join(
            [
                "*/tests/",
                ".froggyignore",
            ]
        )
        if include_gitignore and ".gitignore" in os.listdir(local_repo_path):
            with open(local_repo_path / ".gitignore", "r") as f:
                gitignore_content = f.read()
                froggyignore_contents += "\n"
                froggyignore_contents += gitignore_content

        with open(local_repo_path / ".froggyignore", "w") as f:
            f.write(froggyignore_contents)

    def run_command_with_raise(self, command):
        command = command.replace("apt-get", "sudo apt-get").replace(
            "sudo sudo", "sudo"
        )
        status, output = self.terminal.run(command, raises=True)
        return status, output

    # def run_pre_install(self):
    #     pre_install_cmds = self.install_configs.get("pre_install")
    #     if pre_install_cmds:
    #         self.logger.info("Running pre-install commands...")
    #         for pre_install_cmd in pre_install_cmds:
    #             self.run_command_with_raise(pre_install_cmd)

    def prepare_eval_commands(self):
        """Add eval_cmd to be executed every time the terminal is called"""
        for eval_cmd in self.install_configs.get("eval_commands", []):
            self.setup_commands.append(eval_cmd)

    def run_install(self):
        install_cmd = self.install_configs.get("install", "")
        if install_cmd:
            self.logger.info("Running install commands...")
            install_cmd = install_cmd.replace("--verbose", "").replace("-v", "").strip()
            self.run_command_with_raise(install_cmd)

    def run_post_install(self):
        post_install_cmds = self.install_configs.get("post_install", [])
        if post_install_cmds:
            self.logger.info("Running post-install commands...")
            for post_install_cmd in post_install_cmds:
                self.run_command_with_raise(post_install_cmd)

    def get_configs(self, repo, version):
        return MAP_REPO_VERSION_TO_SPECS[repo][version]

    def repo_name(self, repo):
        return repo.replace("/", "__").replace(" ", "--").replace("'", "")
