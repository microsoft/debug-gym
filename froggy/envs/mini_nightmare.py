import os
from os.path import join as pjoin

import froggy.utils as utils
from froggy.envs.env import RepoEnv


class MiniNightmareEnv(RepoEnv):
    DATA_PATH = "data/mini_nightmare"
    TASK_NAMES = [
        "config",
        "counter",
        "grader",
        "pandas_dataframe",
        "patcher",
        "purr",
        "scientific_calculator",
        "shopping_cart",
        "sum_tree",
        "tomorrow_date",
    ]

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
        self.setup_workspace(
            directory,
            entrypoint="python -m pytest -s test.py",
            debug_entrypoint="pytest --pdb -s test.py",
        )

        infos = super().reset()
        infos.instructions = self.instructions
        infos.last_run_obs = utils.cleanup_pytest_output(infos.last_run_obs)

        self.max_score = utils.extract_max_score_from_pytest_output(infos.last_run_obs)
        infos.max_score = self.max_score
        self.score = utils.extract_reward_from_pytest_output(infos.last_run_obs)
        self.done = self.score == self.max_score
        infos.score = self.score
        infos.done = self.done

        # By default, open the only modifiable file.
        self.load_current_file(self.current_sample["filename"])
        # an update the infos related to current code.
        infos.current_code_with_line_number = self.current_code_with_line_number()
        return infos

    def step(self, action: str):
        infos = super().step(action)

        self.score = utils.extract_reward_from_pytest_output(infos.last_run_obs)
        self.done = self.score == self.max_score
        infos.score = self.score
        infos.done = self.done
        return infos

    def load_dataset(self):
        assert os.path.exists(
            self.DATA_PATH
        ), f"Data path {self.DATA_PATH} does not exist."
        for task_name in self.TASK_NAMES:
            assert os.path.exists(
                pjoin(self.DATA_PATH, task_name, "test.py")
            ), f"Task {task_name} missing test.py file."
            assert os.path.exists(
                pjoin(self.DATA_PATH, task_name, "code.py")
            ), f"Task {task_name} missing code.py file."
            assert os.path.exists(
                pjoin(self.DATA_PATH, task_name, ".froggyignore")
            ), f"Task {task_name} missing .froggyignore file."
            assert os.path.exists(
                pjoin(self.DATA_PATH, task_name, ".froggyreadonly")
            ), f"Task {task_name} missing .froggyreadonly file."

        self.dataset = {}
        for task_name in self.TASK_NAMES:
            task_path = pjoin(self.DATA_PATH, task_name)

            self.dataset[task_name] = {
                "instructions": "The program doesn't behave as intended. Investigate the repository, figure out the root cause, then rewrite the code to fix the issue. Beaware that the bug may not be in the code you initially see.",
                "base_directory": task_path,
                "filename": "code.py",
            }
