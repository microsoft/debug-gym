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
from swebench.harness.constants import MAP_REPO_VERSION_TO_SPECS
from swebench.harness.utils import get_environment_yml, get_requirements
from tqdm import tqdm

import froggy.utils as utils
from froggy.envs.env import RepoEnv

logger = logging.getLogger("froggy")


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
        self.swe_bench_repo_paths = Path.home() / ".cache/froggy/swe-bench"
        self.swe_bench_repo_paths.mkdir(parents=True, exist_ok=True)

        self.load_dataset()
        self.setup_commands = []
        # atexit.register(self.cleanup)  # cleanup the containers and images

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
        logger.info("Cloning all repos needed for SWE-Bench...")
        repos = {task["repo"] for task in self.dataset.values()}
        for repo in tqdm(repos, desc="Cloning repos needed for SWE-Bench"):
            self.clone_repo(repo_address=repo)

    def setup_local_repo(self):
        repo_address = self.ds_row["repo"]
        base_commit = self.ds_row["base_commit"]
        test_patch = self.ds_row["test_patch"]
        # TODO: use fail_to_pass and pass_to_pass
        fail_to_pass = literal_eval(self.ds_row["FAIL_TO_PASS"])
        pass_to_pass = literal_eval(self.ds_row["PASS_TO_PASS"])

        # Clone repository (should already be cloned)
        assert (self.swe_bench_repo_paths / repo_address.split("/")[1]).exists()
        local_repo_path = self.clone_repo(repo_address=repo_address)
        local_branch_path = local_repo_path.parent / self.ds_row["instance_id"]

        if not local_branch_path.exists():
            # Duplicate the repo to avoid changing the current branch.
            logger.info(f"Copying {local_repo_path} to {local_branch_path}")
            shutil.copytree(local_repo_path, local_branch_path)

            # Checkout to base commit.
            command = f"git -C {local_branch_path} checkout {base_commit} -f"
            logger.info(f"Checking out to {base_commit}")
            subprocess.run(command.split(), check=True)

            # Apply test patch
            if test_patch != "":
                command = f"git -C {local_branch_path} apply -"
                subprocess.run(command.split(), input=test_patch, text=True, check=True)
                logger.info("Patch applied successfully.")

            # Make the pdb ignore
            self.make_froggyignore(local_repo_path=local_branch_path)
        else:
            logger.info(f"Local checked out branch {local_branch_path} already exists.")

        entrypoint = self.install_configs["test_cmd"]
        # # TODO: Find another way to extract inline env vars from entrypoint. Move to env_vars instead of setup_commands
        if entrypoint.startswith("PYTHONWARNINGS"):
            export, remaining = entrypoint.split(" ", 1)
            self.setup_commands.append(f"export {export}")
            entrypoint = remaining

        # # For swebench, we must pass the fail_to_pass and pass_to_pass unit tests.
        # entrypoint = "python -m pytest " + " ".join(fail_to_pass + pass_to_pass)

        self.setup_workspace(local_branch_path, entrypoint)

    def setup_docker_image(self):
        base_image = "python:3.12"
        docker_client = docker.from_env()

        repo_address = self.ds_row["repo"]
        base_commit = self.ds_row["base_commit"]
        test_patch = self.ds_row["test_patch"]
        # TODO: use fail_to_pass and pass_to_pass
        fail_to_pass = literal_eval(self.ds_row["FAIL_TO_PASS"])
        pass_to_pass = literal_eval(self.ds_row["PASS_TO_PASS"])

        # Clone repository
        #local_repo_path = self.clone_repo(repo_address=repo_address)
        org_name, repo_name = repo_address.split("/")
        repo_url = f"https://github.com/{repo_address.lstrip('/')}"
        #local_repo_path = self.swe_bench_repo_paths / repo_name
        local_repo_path = "/code"

        # # Checkout to base commit
        # command = f"git -C {local_repo_path} checkout {base_commit} -f"
        # subprocess.run(command.split(), check=True)
        # logger.info(f"Checked out to {base_commit}")

        dockerfile = f"""
            FROM {base_image}
            # Clone task repo
            RUN git clone {repo_url} {local_repo_path}
            # Checkout to base commit
            RUN git -C {local_repo_path} checkout {base_commit} -f
            """

        # Apply test patch
        if test_patch != "":
            dockerfile += f"""
            # Apply test patch
            RUN git -C {local_repo_path} apply {test_patch}
            """
            # command = f"git -C {local_repo_path} apply -"
            # subprocess.run(command.split(), input=test_patch, text=True, check=True)
            # logger.info("Patch applied successfully.")

        image_tag = f"{base_image}-{self.ds_row['instance_id']}"
        try:
            self.docker_client.images.get(image_tag)
            logger.debug(f"Image {image_tag} already exists.")
        except docker.errors.ImageNotFound:
            logger.debug(f"Building image {image_tag}.")
            self.docker_client.images.build(
                fileobj=io.BytesIO(dockerfile.encode("utf-8")),
                tag=image_tag,
                rm=True,
                custom_context=False,
            )

        self.terminal.base_image = image_tag
        self.setup_terminal()
        return self.terminal.patched_image

        # # Start the container.
        # #return image_tag

        # # Make the pdb ignore
        # self.make_froggyignore(local_repo_path=local_repo_path)

        # entrypoint = self.install_configs["test_cmd"]
        # # # TODO: Find another way to extract inline env vars from entrypoint. Move to env_vars instead of setup_commands
        # if entrypoint.startswith("PYTHONWARNINGS"):
        #     export, remaining = entrypoint.split(" ", 1)
        #     self.setup_commands.append(f"export {export}")
        #     entrypoint = remaining

        # # # For swebench, we must pass the fail_to_pass and pass_to_pass unit tests.
        # # entrypoint = "python -m pytest " + " ".join(fail_to_pass + pass_to_pass)

        # # TODO: one workspace per task?
        # self.setup_workspace(local_repo_path, entrypoint)

    def setup_task_info(self, task_name):
        self.task_name = task_name
        self.ds_row = self.dataset[self.task_name]
        self.repo = self.ds_row["repo"]
        self.version = self.ds_row["version"]
        self.install_configs = self.get_configs(self.repo, self.version)

    def reset(self, *, seed=None, options: dict | None = None):
        options = options or {}
        self.setup_task_info(options["task_name"])
        self.setup_local_repo()

        from ipdb import set_trace; set_trace()
        self.setup_terminal()

        # Reset RepoEnv
        obs, infos = super().reset()
        infos["last_run_obs"] = utils.cleanup_pytest_output(infos["last_run_obs"])

        self.max_score = utils.extract_max_score_from_pytest_output(
            infos["last_run_obs"]
        )
        infos["max_score"] = self.max_score
        infos["score"] = utils.extract_reward_from_pytest_output(infos["last_run_obs"])

        return infos["obs"], infos

    def step(self, action: str):
        obs, score, done, infos = super().step(action)
        infos["last_run_obs"] = utils.cleanup_pytest_output(infos["last_run_obs"])
        infos["score"] = utils.extract_reward_from_pytest_output(infos["last_run_obs"])
        return obs, score, done, infos

    def clone_repo(self, repo_address):
        org_name, repo_name = repo_address.split("/")
        repo_url = f"https://github.com/{repo_address.lstrip('/')}"
        local_repo_path = self.swe_bench_repo_paths / repo_name

        # clone
        if not local_repo_path.exists():
            logger.info(f"Cloning {repo_url} into {local_repo_path}")
            subprocess.run(["git", "clone", repo_url, local_repo_path], check=True)
        else:
            logger.info(f"Repo {repo_url} already cloned at {local_repo_path}")

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

    def setup_terminal(self):
        host_uid = os.getuid()
        host_gid = os.getgid()
        cached_image = f"{self.ds_row['instance_id']}-{host_uid}-{host_gid}"

        try:
            docker_client = docker.from_env()
            docker_client.images.get(cached_image)
            logger.debug(f"Used cached image: {cached_image}.")
            self.terminal._patched_image = cached_image
            self.setup_base_image()
            #self.terminal.container = self.terminal.setup_container()

        except docker.errors.ImageNotFound:
            logger.debug(f"Building cached image {cached_image}.")
            # TODO: set base_image and conda_env per task
            self.run_pre_install()
            self.setup_base_image()
            self.run_install()
            self.run_post_install()
            # Commit the container to a new image with the same name
            self.terminal.container.commit(cached_image)

    def setup_base_image(self):
        self.prepare_eval_commands()

        self.conda_env = self.create_conda_env()

        entrypoint = self.install_configs["test_cmd"]

        # TODO: Find another way to extract inline env vars from entrypoint. Move to env_vars instead of setup_commands
        if entrypoint.startswith("PYTHONWARNINGS"):
            export, remaining = entrypoint.split(" ", 1)
            self.setup_commands.append(f"export {export}")
            entrypoint = remaining

        # Commit the container to a new image with the same name
        # self.terminal.container.commit(repository=container_name)
        # return self.conda_env

    def run_pre_install(self):
        pre_install_cmds = self.install_configs.get("pre_install")
        if pre_install_cmds:
            logger.info("Running pre-install commands...")
            for pre_install_cmd in pre_install_cmds:
                self.run_command_with_raise(pre_install_cmd)

    def prepare_eval_commands(self):
        """Add eval_cmd to be executed every time the terminal is called"""
        for eval_cmd in self.install_configs.get("eval_commands", []):
            self.setup_commands.append(eval_cmd)

    def run_install(self):
        install_cmd = self.install_configs.get("install", "")
        if install_cmd:
            logger.info("Running install commands...")
            install_cmd = install_cmd.replace("--verbose", "").replace("-v", "").strip()
            self.run_command_with_raise(install_cmd)

    def run_post_install(self):
        post_install_cmds = self.install_configs.get("post_install", [])
        if post_install_cmds:
            logger.info("Running post-install commands...")
            for post_install_cmd in post_install_cmds:
                self.run_command_with_raise(post_install_cmd)

    def get_configs(self, repo, version):
        return MAP_REPO_VERSION_TO_SPECS[repo][version]

    def install_conda(self):
        install_commands = (
            "sudo apt update && sudo apt install -y wget git && "
            "mkdir -p ~/miniconda3 && "
            "wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh && "
            "bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3 && "
            "rm ~/miniconda3/miniconda.sh && "
            "source ~/miniconda3/bin/activate"
        )
        return self.run_command_with_raise(install_commands)

    def conda_environment_exists(self, env_name):
        conda_env_exists = False
        command = f"conda env list | grep {env_name}"
        status, output = self.terminal.run(command)
        if env_name in output:
            conda_env_exists = True
        elif "conda: command not found" in output:
            self.install_conda()
        return conda_env_exists

    def repo_name(self, repo):
        return repo.replace("/", "__").replace(" ", "--").replace("'", "")

    def _create_conda_env(self, cmd, env_name):
        logger.info(f"Creating conda environment {env_name}")
        self.run_command_with_raise(cmd)
        self.terminal.setup_commands.append(f"conda activate {env_name}")

    def create_conda_env(self):
        # try to activate conda environment without failing if activation fails
        self.terminal.setup_commands += ["source ~/miniconda3/bin/activate || true"]
        # Create environment if does not exist yet
        python = self.install_configs["python"]
        repo_name = self.repo_name(self.repo)
        env_name = f"{repo_name}__{self.version}"

        if self.conda_environment_exists(env_name):
            logger.info("Conda env `{env_name}` already exists, activating...")
            self.terminal.setup_commands.append(f"conda activate {env_name}")
        else:
            logger.info(f"Conda env `{env_name}` not found, creating...")
            packages = self.install_configs.get("packages", "")
            pip_packages = self.install_configs.get("pip_packages")
            if packages == "requirements.txt":
                logger.info("Installing from requirements.txt")
                self._create_conda_env(
                    f"conda create -n {env_name} python={python} -y", env_name
                )
                requirements = get_requirements(self.ds_row)
                tmp_requirements_file = (
                    Path(self.terminal.working_dir) / "tmp_froggy_requirements.txt"
                )
                with open(tmp_requirements_file, "w") as f:
                    f.write(requirements)
                self.run_command_with_raise(f"pip install -r {tmp_requirements_file}")
                self.run_command_with_raise(f"rm {tmp_requirements_file}")
            elif packages == "environment.yml":
                logger.info("Installing from environment.yml")
                content_env_yml = get_environment_yml(self.ds_row, env_name)
                no_use_env = self.install_configs.get("no_use_env")
                if no_use_env:
                    pattern = r"(python=)([^\s]+)"
                    content_env_yml = re.sub(
                        pattern, f"python={python}", content_env_yml
                    )
                tmp_environment_file = (
                    Path(self.terminal.working_dir) / "tmp_froggy_environment.yml"
                )
                with open(tmp_environment_file, "w") as f:
                    f.write(content_env_yml)

                if no_use_env:
                    self._create_conda_env(
                        f"conda create -c conda-forge -n {env_name} python={python} -y",
                        env_name,
                    )
                    self.run_command_with_raise(
                        f"conda env update -n {env_name} -f {tmp_environment_file}"
                    )
                else:
                    self._create_conda_env(
                        f"conda env create -n {env_name} -f {tmp_environment_file} -y",
                        env_name,
                    )
                self.run_command_with_raise(f"rm {tmp_environment_file}")
            else:
                self._create_conda_env(
                    f"conda create -n {env_name} python={python} -y", env_name
                ),
                if packages.strip():
                    self.run_command_with_raise(f"conda install {packages} -y")
            if pip_packages:
                self.run_command_with_raise(f"pip install {' '.join(pip_packages)}")
        return env_name
