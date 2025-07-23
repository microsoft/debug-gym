import os
import subprocess

import debug_gym.gym.utils as utils
from debug_gym.constants import DEBUG_GYM_CACHE_DIR
from debug_gym.gym.entities import EvalOutput
from debug_gym.gym.envs.env import RepoEnv
from debug_gym.gym.terminal import DockerTerminal, Terminal


class AiderBenchmarkEnv(RepoEnv):
    REPO_URL = "https://github.com/exercism/python"
    REPO_PATH = DEBUG_GYM_CACHE_DIR / "exercism"

    def __init__(
        self,
        terminal: Terminal | None = None,
        **kwargs,
    ):
        terminal = terminal or DockerTerminal(logger=kwargs.get("logger"))
        if not isinstance(terminal, DockerTerminal):
            raise ValueError("AiderBenchmarkEnv only supports DockerTerminal.")

        super().__init__(terminal=terminal, **kwargs)

    @property
    def instructions(self) -> str:
        return self.current_sample["instructions"]

    def __init__(self, entrypoint: str = "python -m pytest -s .", **kwargs):
        super().__init__(entrypoint=entrypoint, **kwargs)
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

    def setup_terminal(self):
        self.logger.info(f"Configuring docker container: {self.terminal.container}")

        self.terminal.run("git init")
        self.terminal.run("git config user.name 'debug-gym'")
        self.terminal.run("git config user.email '<>'")

        self.terminal.run("git add *.py")
        self.terminal.run("git commit -am 'Init'")

        self.terminal.run("git add .debugignore")
        self.terminal.run("git add .debugreadonly")
        self.terminal.run("git commit -am 'Add debug-gym ignore and read-only files'")

    def reset(self, *, options: dict = None):
        options = options or {}
        self.current_sample = self.dataset[options["task_name"]]
        directory = self.current_sample["base_directory"]
        self.setup_workspace(directory, entrypoint=self.entrypoint)
        self.setup_terminal()
        infos = super().reset(options=options)
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

            self.dataset[task_name] = {
                "base_directory": directory,
                "instructions": instructions,
                "filename": task_name + ".py",
            }

    def get_problem_ids(self, split_or_problem_id):
        if split_or_problem_id == "all":
            return sorted(self.dataset.keys())  # all tasks
        elif split_or_problem_id in self.dataset:
            return [split_or_problem_id]  # Single task
        else:
            raise ValueError(
                f"Invalid split or problem id: '{split_or_problem_id}'.\nChoose from: {['all'] + sorted(self.dataset.keys())}"
            )
