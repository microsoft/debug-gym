import os
import re
import shutil
import subprocess
from ast import literal_eval
from pathlib import Path

import datasets
import docker
from swebench.harness.constants import (
    MAP_REPO_VERSION_TO_SPECS,
    NON_TEST_EXTS,
    TestStatus,
)
from swebench.harness.docker_build import (
    build_env_images,
    build_instance_image,
    get_env_configs_to_build,
)
from swebench.harness.log_parsers import MAP_REPO_TO_PARSER
from swebench.harness.test_spec import make_test_spec
from swebench.harness.utils import get_test_directives, load_swebench_dataset
from tqdm import tqdm

from debug_gym.gym.entities import EvalOutput
from debug_gym.gym.envs.env import RepoEnv
from debug_gym.gym.terminal import DockerTerminal, Terminal
from debug_gym.gym.utils import create_ignore_file


class SWEBenchEnv(RepoEnv):
    CACHE = Path.joinpath(Path.home(), ".cache", "debug_gym", "swe-bench")

    def __init__(
        self,
        dataset_id: str = "princeton-nlp/SWE-bench_Verified",
        # dataset_id: str = "princeton-nlp/SWE-bench_lite",
        split: str = "test",
        instance_ids: list[str] | None = None,
        terminal: Terminal | None = None,
        **kwargs,
    ):
        terminal = terminal or DockerTerminal(logger=kwargs.get("logger"))
        if not isinstance(terminal, DockerTerminal):
            raise ValueError("SWEBenchEnv only supports DockerTerminal.")

        super().__init__(terminal=terminal, **kwargs)

        self.dataset_id = dataset_id
        self.split = split
        self.instance_ids = instance_ids
        SWEBenchEnv.CACHE.mkdir(parents=True, exist_ok=True)

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
        self.ds = datasets.load_dataset(self.dataset_id)["test"]
        self.dataset = {row["instance_id"]: row for row in self.ds.sort("instance_id")}

        # To avoid concurrency issues, we will clone all the repos in the dataset.
        repos = sorted({task["repo"] for task in self.dataset.values()})
        repo_names = [repo.split("/")[1] for repo in repos]
        missing_repos = [
            repo for repo in repo_names if not Path.exists(SWEBenchEnv.CACHE / repo)
        ]
        if missing_repos:
            self.logger.debug("Cloning all repos needed for SWE-Bench...")
            for repo in tqdm(repos, desc="Cloning repos needed for SWE-Bench"):
                self.clone_repo(repo_address=repo)

        swebench_instances = load_swebench_dataset(
            name=self.dataset_id, instance_ids=self.instance_ids
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

    def setup_local_repo(self):
        repo_address = self.ds_row["repo"]
        base_commit = self.ds_row["base_commit"]
        test_patch = self.ds_row["test_patch"]
        # TODO: use fail_to_pass and pass_to_pass
        self.fail_to_pass = literal_eval(self.ds_row["FAIL_TO_PASS"])
        self.pass_to_pass = literal_eval(self.ds_row["PASS_TO_PASS"])
        self.test_directives = get_test_directives(self.ds_row)

        local_repo_path = SWEBenchEnv.CACHE / self.repo_name
        assert local_repo_path.exists()
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

            create_ignore_file(
                local_branch_path / ".debugignore", patterns=self.ignore_files
            )
            create_ignore_file(
                local_branch_path / ".debugreadonly", patterns=self.test_directives
            )
        else:
            self.logger.debug(
                f"Local checked out branch {local_branch_path} already exists."
            )

        entrypoint = " ".join([self.install_configs["test_cmd"], *self.test_directives])

        if (
            "sphinx" in self.ds_row["instance_id"]
            or "sympy" in self.ds_row["instance_id"]
        ):
            # use pytest instead of `sympy bin/test` and `sphinx tox` so pdb breakpoints work
            expression = " ".join(self.test_directives)
            debug_entrypoint = f"python -m pytest {expression}"
            # Install pytest if not already installed
            self.install_configs["install"] += " && python -m pip install pytest"

            if entrypoint.startswith("PYTHONWARNINGS"):
                # Move PYTHONWARNINGS from the entrypoint to the session commands
                export, remaining = entrypoint.split(" ", 1)
                self.session_commands.append(f"export {export}")
                entrypoint = remaining

        # -s (capture=no) from pytest, allows for debugging with pdb
        # -q (quiet) from pytest, to avoid long pytest output
        debug_entrypoint = entrypoint.replace("pytest", "pytest -sq")

        self.setup_workspace(
            path=local_branch_path,
            entrypoint=entrypoint,
            debug_entrypoint=debug_entrypoint,
        )

        return local_branch_path, entrypoint

    def setup_task_info(self, task_name):
        if self.instance_ids:
            if task_name not in self.instance_ids:
                raise ValueError(
                    f"Task `{task_name}` was not found in instance_ids. The available tasks are: {self.instance_ids}.\n"
                    "Please provide a valid task or initialize the environment without instance_ids to load all tasks."
                )
        self.task_name = task_name
        self.ds_row = self.dataset[self.task_name]
        self.repo = self.ds_row["repo"]
        self.repo_name = self.repo.split("/")[1]
        self.version = self.ds_row["version"]
        self.install_configs = self.get_configs(self.repo, self.version)
        self.gold_patch = self.ds_row["patch"]

    @property
    def patch(self):
        command = "git diff"
        result = subprocess.run(
            command.split(), cwd=self.working_dir, text=True, capture_output=True
        )
        patch = result.stdout.replace(str(self.working_dir), str(self.path))
        return patch

    def calculate_score(self, eval_output: EvalOutput) -> int:
        test_status_map = MAP_REPO_TO_PARSER[self.repo](eval_output.output)
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

        spec = make_test_spec(self.ds_row)
        docker_client = docker.from_env()
        build_instance_image(spec, docker_client, logger=None, nocache=False)

        # Start the terminal
        self.terminal.base_image = spec.instance_image_key

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

        self.terminal.session_commands.append("source /opt/miniconda3/bin/activate")
        self.terminal.session_commands.append(f"conda activate testbed")

        self.run_install()
        self.run_post_install()

        # Apply test patch
        command = f"git apply -"
        subprocess.run(
            command.split(),
            cwd=self.working_dir,
            input=self.ds_row["test_patch"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )

        # Need to recreate those files after copying the initial code.
        create_ignore_file(
            self.working_dir / ".debugignore", patterns=self.ignore_files
        )

        # Get test directives from test patch and remove non-test files
        test_files = re.findall(r"diff --git a/.* b/(.*)", self.ds_row["test_patch"])
        test_files = [
            f for f in test_files if not any(f.endswith(ext) for ext in NON_TEST_EXTS)
        ]
        # Add test/ to readonly files if not already present
        if "test/" not in test_files:
            test_files.append("test/")
        create_ignore_file(self.working_dir / ".debugreadonly", patterns=test_files)
        self._index_files()

        self.terminal.run(f"git config user.name 'SWE-Bench'")
        self.terminal.run(f"git config user.email '<>'")
        self.terminal.run(f"git add .debugignore")
        self.terminal.run(f"git commit -am 'Applied test patch'")

        # Reset RepoEnv
        # TODO: Create a RepoEnv per task and set max_score at initialization.
        self.max_score = len(self.fail_to_pass)
        infos = super().reset(options=options)
        assert not self.done, "Tests should be failing before debugging."

        return infos

    def clone_repo(self, repo_address):
        org_name, repo_name = repo_address.split("/")
        repo_url = f"https://github.com/{repo_address.lstrip('/')}"
        local_repo_path = SWEBenchEnv.CACHE / repo_name

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

    def prepare_eval_commands(self):
        """Add eval_cmd to be executed every time the terminal is called"""
        for eval_cmd in self.install_configs.get("eval_commands", []):
            self.session_commands.append(eval_cmd)

    def run_install(self):
        install_cmd = self.install_configs.get("install", "")
        if install_cmd:
            self.logger.debug("Running install commands...")
            install_cmd = install_cmd.replace("--verbose", "").replace("-v", "").strip()
            self.run_command_with_raise(install_cmd)

    def run_post_install(self):
        post_install_cmds = self.install_configs.get("post_install", [])
        if post_install_cmds:
            self.logger.debug("Running post-install commands...")
            for post_install_cmd in post_install_cmds:
                self.run_command_with_raise(post_install_cmd)

    def get_configs(self, repo, version):
        return MAP_REPO_VERSION_TO_SPECS[repo][version]

    def repo_name(self, repo):
        return repo.replace("/", "__").replace(" ", "--").replace("'", "")
