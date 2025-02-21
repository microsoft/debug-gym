import os
from os.path import join as pjoin

import froggy.pond.utils as utils
from froggy.pond.envs.env import RepoEnv


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
        return {
            **super().instructions,
            "Problem description": self.current_sample["instructions"],
        }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.load_dataset()

    def eval(self, **kwargs):
        success, output = self.terminal.run(self.entrypoint, timeout=self.run_timeout)
        self.max_score = utils.extract_max_score_from_pytest_output(output)
        self.score = utils.extract_reward_from_pytest_output(output)
        self.done = self.score == self.max_score
        self.last_eval_obs = utils.cleanup_pytest_output(output)
        return self.last_eval_obs

    def reset(self, *, options: dict = None):
        options = options or {}
        self.current_sample = self.dataset[options["task_name"]]

        directory = self.current_sample["base_directory"]
        self.setup_workspace(
            directory,
            entrypoint="python -m pytest -s test.py",
            debug_entrypoint="python -m pdb -m pytest -s test.py",
        )

        infos = super().reset(options=options)

        # By default, open the only modifiable file.
        self.load_current_file(self.current_sample["filename"])
        # an update the infos related to current code.
        infos.current_code_with_line_number = self.current_code_with_line_number()
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
                pjoin(self.DATA_PATH, task_name, task_name + "_code.py")
            ), f"Task {task_name} missing {task_name}_code.py file."
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
                "filename": task_name + "_code.py",
            }
