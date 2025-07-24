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

    def setup_task(self, task_name):
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

        entrypoint = " ".join([self.test_cmd, *self.test_directives])

        if self.package_name == "sphinx" or self.package_name == "sympy":
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
            # Empty folder. The actual codebase will come from the docker image.
            path=SWEBenchEnv.DUMMY_DIR,
            # allow traceback to be printed in the output.
            entrypoint=entrypoint.replace("--tb=no", "--tb=short"),
            debug_entrypoint=debug_entrypoint,
        )

        self.git_apply_cmd = f"git -C {self.working_dir} apply -"

        # Use SWE-Bench's test spec to build the instance image.
        build_instance_image(
            self.test_spec, docker.from_env(), logger=None, nocache=False
        )

    @property
    def patch(self):
        command = "git diff"
        result = subprocess.run(
            command.split(), cwd=self.working_dir, text=True, capture_output=True
        )
        # patch = result.stdout.replace(str(self.working_dir), str(self.path))
        return result.stdout

    def apply_gold_patch(self):
        self.logger.info(f"Applying gold patch to {self.working_dir}.")
        command = self.git_apply_cmd + f" <<'EOF'\n{self.gold_patch}\nEOF"
        self.terminal.run(command, raises=True)
        self.logger.info("Patch applied successfully.")

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

    def setup_terminal(self):
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
        self.terminal.run("apt update && apt install -y sudo", user="root")
        # Add the user to sudoers.
        self.terminal.run(
            "echo 'debug_gym_user ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/debug_gym_user",
            user="root",
        )

        # Allow for the user to pip install in the env. TODO: This is still slow.
        # self.terminal.run(f"chmod -R o+rwX /opt/miniconda3/envs/testbed", user="root")

        # Alternatively, we can use the following to specifically allow read/write/execute permissions on certain directories.
        venv_path = "/opt/miniconda3/envs/testbed"
        venv_packages = f"{venv_path}/lib/python*/site-packages"
        self.terminal.run(f"chmod -R o+rwX {venv_path}/bin", user="root")
        self.terminal.run(f"chmod o+rwX {venv_packages}", user="root")
        self.terminal.run(f"chmod o+rwX {venv_packages}/*", user="root")
        self.terminal.run(
            f"chmod -R o+rwX {venv_packages}/{self.package_name}*", user="root"
        )
        self.terminal.run(
            f"chmod -R o+rwX {venv_packages}/{self.package_name.title()}*", user="root"
        )
        self.terminal.run(f"chmod -R o+rwX {venv_packages}/*pdb*", user="root")

        # Delete the content in the working directory.
        self.terminal.run(f"rm -rf {self.working_dir / '*'} {self.working_dir / '.*'}")

        # Copy the initial code to the working directory.
        self.terminal.run(f"cp -r /testbed/. {self.working_dir}")
        self.terminal.run(f"chmod -R a+rw {self.working_dir}")

        self.terminal.session_commands.append("source /opt/miniconda3/bin/activate")
        self.terminal.session_commands.append("conda activate testbed")

        self.run_install()
        self.run_post_install()

        # Apply the test patch directly.
        self.terminal.run(f"git apply - <<'EOF'\n{self.test_patch}\nEOF")

        self.terminal.run("git config user.name 'debug-gym'")
        self.terminal.run("git config user.email '<>'")
        self.terminal.run(f"git commit -am 'Applying test patch for {self.task_name}'")

        # Rebuild the debug ignore and read-only files.
        create_ignore_file(
            self.working_dir / ".debugignore", patterns=self.ignore_files
        )
        create_ignore_file(
            self.working_dir / ".debugreadonly", patterns=self.test_directives
        )
        self.setup_file_filters()  # Need to refresh the file filters after re-creating ignore files.

        self.terminal.run("git add .debugignore")
        self.terminal.run("git add .debugreadonly")
        self.terminal.run("git commit -am 'Add debug-gym ignore and read-only files'")

        # Remove the remote so the agent won't see newer commits.
        self.terminal.run("git remote remove origin")

    def reset(self, *, options: dict | None = None):
        # TODO: support reset current task, i.e. no options provided.
        options = options or {}

        # Clean up the previous task, if any.
        self.close()

        self.setup_task(options["task_name"])

        self.setup_terminal()

        # Reset RepoEnv
        # TODO: Create a RepoEnv per task and set max_score at initialization.
        self.max_score = len(self.fail_to_pass)
        infos = super().reset(options=options)
        assert not self.done, "Tests should be failing before debugging."

        return infos

    @property
    def ignore_files(self):
        return [
            ".?*",  # Ignore hidden files and directories but not current dir "."
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
            if type(install_cmds) is str:
                install_cmds = [install_cmds]

            self.logger.debug("Running install commands...")
            for install_cmd in install_cmds:
                # install_cmd = install_cmd.replace("--verbose", "").replace("-v", "").strip()
                self.run_command_with_raise(install_cmd)

    def run_post_install(self):
        post_install_cmds = self.install_configs.get("post_install", [])
        if post_install_cmds:
            self.logger.debug("Running post-install commands...")
            for post_install_cmd in post_install_cmds:
                self.run_command_with_raise(post_install_cmd)
