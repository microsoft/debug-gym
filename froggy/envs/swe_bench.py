import os
import subprocess
import tempfile
from ast import literal_eval
from os.path import join as pjoin
from pathlib import Path

from datasets import load_dataset as load_hf_dataset
from termcolor import colored

from froggy.envs.env import RepoEnv
from froggy.utils import (
    cleanup_pytest_output,
    extract_max_score_from_pytest_output,
    extract_reward_from_pytest_output,
)


class SWEBenchEnv(RepoEnv):
    HF_SWE_BENCH_VERIFIED = "princeton-nlp/SWE-bench_Verified"
    SWE_BENCH_REPO_PATHS = Path(pjoin(tempfile.gettempdir(), "swe-bench"))

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.load_dataset()

    @property
    def instructions(self):
        _instruction = {
            "Problem description": self.ds_row["problem_statement"],
            "Available tools to solve the problem": self.tool_instructions,
            "Available commands": self.actions_str,
        }
        return _instruction

    def load_dataset(self):
        self.ds = load_hf_dataset(self.HF_SWE_BENCH_VERIFIED)["test"]
        instance_id_list = self.ds["instance_id"]
        self.dataset = {}
        for instance_id in instance_id_list:
            self.dataset[instance_id] = self.ds.filter(
                lambda x: x["instance_id"] == instance_id
            )[0]

    def reset(self, *, seed=None, options={}):
        assert "task_name" in options, "task_name must be provided in options"
        assert (
            options["task_name"] in self.dataset
        ), f"task_name {options['task_name']} not found in dataset"
        self.ds_row = self.dataset[options["task_name"]]
        repo_address = self.ds_row["repo"]
        base_commit = self.ds_row["base_commit"]
        test_patch = self.ds_row["test_patch"]
        fail_to_pass = literal_eval(self.ds_row["FAIL_TO_PASS"])
        pass_to_pass = literal_eval(self.ds_row["PASS_TO_PASS"])

        # Clone repository
        local_repo_path = self.clone_repo(repo_address=repo_address)

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

        # For swebench, we must pass the fail_to_pass and pass_to_pass unit tests.
        entrypoint = "python -m pytest " + " ".join(fail_to_pass + pass_to_pass)
        self.setup_workspace(local_repo_path, entrypoint)

        # Reset RepoEnv
        obs, infos = super().reset()
        infos["last_run_obs"] = cleanup_pytest_output(infos["last_run_obs"])

        self.max_score = extract_max_score_from_pytest_output(infos["last_run_obs"])
        infos["max_score"] = self.max_score
        infos["score"] = extract_reward_from_pytest_output(infos["last_run_obs"])

        return infos["obs"], infos

    def step(self, action: str):
        obs, score, done, infos = super().step(action)
        infos["last_run_obs"] = cleanup_pytest_output(infos["last_run_obs"])
        infos["score"] = extract_reward_from_pytest_output(infos["last_run_obs"])
        return obs, score, done, infos

    def clone_repo(self, repo_address):
        org_name, repo_name = repo_address.split("/")
        repo_url = f"https://github.com/{repo_address.lstrip('/')}"
        local_repo_path = self.SWE_BENCH_REPO_PATHS / repo_name

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
            with open(pjoin(local_repo_path, ".gitignore"), "r") as f:
                gitignore_content = f.read()
                froggyignore_contents += "\n"
                froggyignore_contents += gitignore_content

        with open(local_repo_path / ".froggyignore", "w") as f:
            f.write(froggyignore_contents)
