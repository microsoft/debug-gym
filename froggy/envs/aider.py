import os
import re
import subprocess
import tempfile
from os.path import join as pjoin
from pathlib import Path

from froggy.envs.env import RepoEnv
import froggy.utils as utils


class AiderBenchmarkEnv(RepoEnv):
    REPO_URL = "https://github.com/exercism/python"
    REPO_PATH = Path(pjoin(tempfile.gettempdir(), "exercism"))

    @property
    def instructions(self):
        _instruction = {
            "Problem description": self.current_sample["instructions"],
            "Available tools to solve the problem": self.tool_instructions,
            "Available commands": self.actions_str,
        }
        return _instruction

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.load_dataset()

    def reset(self, *, seed=None, options: dict = None):
        options = options or {}
        self.current_sample = self.dataset[options["task_name"]]

        directory = self.current_sample["base_directory"]
        self.setup_workspace(directory)

        obs, infos = super().reset()
        infos["instructions"] = self.instructions
        infos["last_run_obs"] = utils.cleanup_pytest_output(infos["last_run_obs"])

        self.max_score = utils.extract_max_score_from_pytest_output(infos["last_run_obs"])
        infos["max_score"] = self.max_score
        infos["score"] = utils.extract_reward_from_pytest_output(infos["last_run_obs"])

        # By default, open the only modifiable file.
        self.load_current_file(self.current_sample["filename"])
        # an update the infos related to current code.
        infos["current_code_with_line_number"] = self.current_code_with_line_number()
        return infos["obs"], infos

    def step(self, action: str):
        obs, score, done, infos = super().step(action)
        infos["last_run_obs"] = utils.cleanup_pytest_output(infos["last_run_obs"])
        infos["score"] = utils.extract_reward_from_pytest_output(infos["last_run_obs"])
        return obs, score, done, infos

    def load_dataset(self):
        if not os.path.exists(self.REPO_PATH):
            subprocess.run(["git", "clone", self.REPO_URL, self.REPO_PATH], check=True)

        practice_path = self.REPO_PATH / "exercises" / "practice"
        directories = [d for d in practice_path.iterdir() if d.is_dir()]

        self.dataset = {}
        for directory in directories:
            task_name = directory.name.replace("-", "_")

            docs = directory / ".docs"
            intro_md = docs / "introduction.md"
            instr_md = docs / "instructions.md"
            instr_more_md = docs / "instructions.append.md"
            instructions = ""
            instructions += intro_md.read_text() if intro_md.exists() else ""
            instructions += instr_md.read_text() if instr_md.exists() else ""
            instructions += instr_more_md.read_text() if instr_more_md.exists() else ""

            # Add an ignore file
            self.make_froggyignore(directory, include_gitignore=True)

            self.dataset[task_name] = {
                "base_directory": directory,
                "instructions": instructions,
                "filename": task_name + ".py",
            }

    def make_froggyignore(self, directory: str, include_gitignore: bool = True):
        froggyignore_contents = "\n".join(
                        [
                            ".DS_Store",
                            "__pycache__/",
                            ".approaches/",
                            ".docs/",
                            ".meta/",
                            ".pytest_cache/",
                            "*test*.py",
                            "*.pyc",
                            "*.md",
                            ".froggyignore",
                            "log/",
                            "data/",
                        ]
                    )

        if include_gitignore and ".gitignore" in os.listdir(directory):
            with open(pjoin(directory, ".gitignore"), "r") as f:
                gitignore_content = f.read()
                froggyignore_contents += "\n"
                froggyignore_contents += gitignore_content

        with open(directory / ".froggyignore", "w") as f:
                f.write(froggyignore_contents)




