import os
import subprocess
from pathlib import Path

import froggy.utils as utils
from froggy.envs.env import RepoEnv


class AiderBenchmarkEnv(RepoEnv):
    REPO_URL = "https://github.com/exercism/python"
    REPO_PATH = Path.joinpath(Path.home(), ".cache", "froggy", "exercism")

    @property
    def instructions(self):
        _instruction = {
            "Problem description": self.current_sample["instructions"],
            "Available tools to solve the problem": self.tool_instructions,
            "Available commands": self.tool_names,
        }
        return _instruction

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.load_dataset()

    def eval(self, **kwargs):
        # if not self.done:  # Skip evaluation if the task is already solved.
        success, output = self.terminal.run(self.entrypoint, timeout=self.run_timeout)
        self.max_score = utils.extract_max_score_from_pytest_output(output)
        self.score = utils.extract_reward_from_pytest_output(output)
        self.done = self.score == self.max_score
        self.last_eval_obs = utils.cleanup_pytest_output(output)
        return self.last_eval_obs

    def reset(self, *, seed=None, options: dict = None):
        options = options or {}
        self.current_sample = self.dataset[options["task_name"]]

        directory = self.current_sample["base_directory"]
        self.setup_workspace(directory, entrypoint="python -m pytest -s .")
        infos = super().reset()
        infos.instructions = self.instructions  # TODO: is this needed?

        # By default, open the only modifiable file.
        self.load_current_file(self.current_sample["filename"])
        # an update the infos related to current code.
        infos.current_code_with_line_number = self.current_code_with_line_number()
        return infos

    def step(self, action: str):
        return super().step(action)

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

            # Add .froggyignore so all files are ignored except Python files.
            utils.create_ignore_file(
                directory / ".froggyignore",
                patterns=[
                    ".*/",
                    "__pycache__/",
                    "*.pyc",
                    # "*.md",
                    # "log/",
                    # "data/",
                ],
            )
            # Add .froggyreadonly so tests are readonly.
            utils.create_ignore_file(
                directory / ".froggyreadonly", patterns=["*test*.py"]
            )

            self.dataset[task_name] = {
                "base_directory": directory,
                "instructions": instructions,
                "filename": task_name + ".py",
            }
