import os
import subprocess

import debug_gym.gym.utils as utils
from debug_gym.constants import DEBUG_GYM_CACHE_DIR
from debug_gym.gym.entities import EvalOutput
from debug_gym.gym.envs.env import RepoEnv


class AiderBenchmarkEnv(RepoEnv):
    REPO_URL = "https://github.com/exercism/python"
    REPO_PATH = DEBUG_GYM_CACHE_DIR / "exercism"

    @property
    def instructions(self) -> str:
        return self.current_sample["instructions"]

    def __init__(self, entrypoint: str = "python -m pytest -s .", **kwargs):
        super().__init__(entrypoint=entrypoint, **kwargs)

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
        return infos

    def load_dataset(self, problems: str | list[str] | None = None):
        if not os.path.exists(self.REPO_PATH):
            subprocess.run(["git", "clone", self.REPO_URL, self.REPO_PATH], check=True)

        practice_path = self.REPO_PATH / "exercises" / "practice"
        directories = [d for d in practice_path.iterdir() if d.is_dir()]

        dataset = {}
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

            # Add .debugignore so all files are ignored except Python files.
            utils.create_ignore_file(
                directory / ".debugignore",
                patterns=[
                    ".?*",  # Ignore hidden files and directories but not current dir "."
                    "__pycache__/",
                    "*.pyc",
                    # "*.md",
                    # "log/",
                    # "data/",
                ],
            )
            # Add .debugreadonly so tests are readonly.
            utils.create_ignore_file(
                directory / ".debugreadonly", patterns=["*test*.py"]
            )

            dataset[task_name] = {
                "base_directory": directory,
                "instructions": instructions,
                "filename": task_name + ".py",
            }

        problems = utils.filter_problems(dataset, problems)
        dataset = {id: i for id, i in dataset.items() if id in problems}
        return dataset
