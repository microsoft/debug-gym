import os
import subprocess
import tempfile
from ast import literal_eval
from os.path import join as pjoin
from pathlib import Path

import datasets
from swebench.harness.constants import MAP_REPO_VERSION_TO_SPECS
from swebench.harness.utils import get_environment_yml, get_requirements

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
        self.swe_bench_repo_paths = Path(pjoin(tempfile.gettempdir(), "swe-bench"))

        self.load_dataset()
        self.setup_commands = []
        # # atexit.register(self.cleanup)  # cleanup the containers and images

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

    def setup_local_repo(self):
        repo_address = self.ds_row["repo"]
        base_commit = self.ds_row["base_commit"]
        test_patch = self.ds_row["test_patch"]
        # TODO: use fail_to_pass and pass_to_pass
        fail_to_pass = literal_eval(self.ds_row["FAIL_TO_PASS"])
        pass_to_pass = literal_eval(self.ds_row["PASS_TO_PASS"])

        # Clone repository
        local_repo_path = self.clone_repo(repo_address=repo_address)

        # TODO: create a unique workspace per task
        # Checkout to base commit
        command = f"git -C {local_repo_path} checkout {base_commit} -f"
        subprocess.run(command.split(), check=True)
        print(f"Checked out to {base_commit}")

        # Apply test patch
        if test_patch != "":
            command = f"git -C {local_repo_path} apply -"
            subprocess.run(command.split(), input=test_patch, text=True, check=True)
            print("Patch applied successfully.")

        # Make the pdb ignore
        self.make_froggyignore(local_repo_path=local_repo_path)

        entrypoint = self.install_configs["test_cmd"]
        # # TODO: Find another way to extract inline env vars from entrypoint. Move to env_vars instead of setup_commands
        if entrypoint.startswith("PYTHONWARNINGS"):
            export, remaining = entrypoint.split(" ", 1)
            self.setup_commands.append(f"export {export}")
            entrypoint = remaining

        # # For swebench, we must pass the fail_to_pass and pass_to_pass unit tests.
        # entrypoint = "python -m pytest " + " ".join(fail_to_pass + pass_to_pass)

        # TODO: one workspace per task?
        self.setup_workspace(local_repo_path, entrypoint)

    def reset(self, *, seed=None, options: dict | None = None):
        options = options or {}
        self.task_name = options["task_name"]
        self.ds_row = self.dataset[self.task_name]
        self.repo = self.ds_row["repo"]
        self.version = self.ds_row["version"]
        self.install_configs = self.get_configs(self.repo, self.version)

        self.setup_local_repo()
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
            print(f"Cloning {repo_url} into {local_repo_path}")
            subprocess.run(["git", "clone", repo_url, local_repo_path], check=True)
        else:
            print(f"Repo {repo_url} already cloned at {local_repo_path}")

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
        print(f"Running command: {command}")
        status, output = self.terminal.run(command)
        if not status:
            raise ValueError(f"Failed to run command: {command} ", output)
        print(f"{output}\n\n")
        return status, output

    def setup_terminal(self):
        # TODO: set base_image and conda_env per task
        self.run_pre_install()
        self.setup_base_image()
        self.run_install()
        self.run_post_install()

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
            print("Running pre-install commands...")
            for pre_install_cmd in pre_install_cmds:
                self.run_command_with_raise(pre_install_cmd)

    def prepare_eval_commands(self):
        """Add eval_cmd to be executed every time the terminal is called"""
        for eval_cmd in self.install_configs.get("eval_commands", []):
            self.setup_commands.append(eval_cmd)

    def run_install(self):
        install_cmd = self.install_configs.get("install", "")
        if install_cmd:
            install_cmd = install_cmd.replace("--verbose", "").replace("-v", "").strip()
            self.run_command_with_raise(install_cmd)

    def run_post_install(self):
        post_install_cmds = self.install_configs.get("post_install", [])
        if post_install_cmds:
            print("Running post-install commands...")
            for post_install_cmd in post_install_cmds:
                self.run_command_with_raise(post_install_cmd)
            print("Ran post-install commands")

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

    def create_conda_env(self):
        # try to activate conda environment without failing if activation fails
        self.terminal.setup_commands += ["source ~/miniconda3/bin/activate || true"]
        # Create environment if does not exist yet
        python = self.install_configs["python"]
        repo_name = self.repo_name(self.repo)
        env_name = f"{repo_name}__{self.version}"

        if not self.conda_environment_exists(env_name):
            print(f"{env_name} conda env not found, creating...")
            packages = self.install_configs.get("packages", "")
            pip_packages = self.install_configs.get("pip_packages")
            if packages == "requirements.txt":
                # Create conda environment
                self.run_command_with_raise(
                    f"conda create -n {env_name} python={python} -y"
                )
                self.terminal.setup_commands.append(f"conda activate {env_name}")
                print("Created conda environment")
                requirements = get_requirements(self.ds_row)
                tmp_requirements_file = (
                    Path(self.terminal.working_dir) / "tmp_froggy_requirements.txt"
                )
                with open(tmp_requirements_file, "w") as f:
                    f.write(requirements)
                self.run_command_with_raise(f"pip install -r {tmp_requirements_file}")
                print("Installed requirements from requirements.txt")
                self.run_command_with_raise(f"rm {tmp_requirements_file}")
            elif packages == "environment.yml":
                content_env_yml = get_environment_yml(self.ds_row, env_name)
                no_use_env = self.install_configs.get("no_use_env")
                if not no_use_env:
                    content_env_yml += f"\n  - python={python}\n"
                tmp_environment_file = (
                    Path(self.terminal.working_dir) / "tmp_froggy_environment.yml"
                )
                with open(tmp_environment_file, "w") as f:
                    f.write(content_env_yml)

                if no_use_env:
                    self.run_command_with_raise(
                        f"conda create -c conda-forge -n {env_name} python={python} -y"
                    )
                    self.terminal.setup_commands.append(f"conda activate {env_name}")
                    print("Created conda environment")

                    self.run_command_with_raise(
                        f"conda env update -f {tmp_environment_file}"
                    )
                    print("Installed packages from environment.yml")
                else:
                    # Create environment + install packages
                    self.run_command_with_raise(
                        f"conda env create --file {tmp_environment_file}"
                    )
                    self.run_command_with_raise(f"rm {tmp_environment_file}")
                    self.terminal.setup_commands.append(f"conda activate {env_name}")
                    print("Installed packages from environment.yml")
            else:
                self.run_command_with_raise(
                    f"conda create --name {env_name} python={python} -y"
                ),
                print(f"Created conda environment {env_name} with python {python}")
                self.terminal.setup_commands.append(f"conda activate {env_name}")
                if packages.strip():
                    self.run_command_with_raise(f"conda install {packages} -y")
                    print("Installed conda packages")
            if pip_packages:
                self.run_command_with_raise(f"pip install {' '.join(pip_packages)}")
                print("Installed extra pip dependencies")
        else:
            self.terminal.setup_commands.append(f"conda activate {env_name}")
        return env_name
