import os
import subprocess
from pathlib import Path

import froggy.pond.utils as utils
from froggy.pond.entities import EvalOutput
from froggy.pond.envs.env import RepoEnv


class AiderBenchmarkEnv(RepoEnv):
    REPO_URL = "https://github.com/exercism/python"
    REPO_PATH = Path.joinpath(Path.home(), ".cache", "froggy", "exercism")

    @property
    def instructions(self):
        return {
            **super().instructions,
            "Problem description": self.current_sample["instructions"],
        }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.load_dataset()

    def calculate_max_score(self, eval_output: EvalOutput) -> int:
        return utils.extract_max_score_from_pytest_output(eval_output.output)

    def calculate_score(self, eval_output: EvalOutput) -> int:
        return utils.extract_reward_from_pytest_output(eval_output.output)

    def eval(self, **kwargs) -> EvalOutput:
        success, output = self.terminal.run(self.entrypoint, timeout=self.run_timeout)
        output = utils.cleanup_pytest_output(output)
        self.last_eval = EvalOutput(success, output)
        return self.last_eval

    def reset(self, *, options: dict = None):
        options = options or {}
        self.current_sample = self.dataset[options["task_name"]]

        directory = self.current_sample["base_directory"]
        self.setup_workspace(directory, entrypoint=self.entrypoint)
        infos = super().reset(options=options)

        # By default, open the only modifiable file.
        self.load_current_file(self.current_sample["filename"])
        # an update the infos related to current code.
        infos.current_code_with_line_number = self.current_code_with_line_number()
        return infos

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
