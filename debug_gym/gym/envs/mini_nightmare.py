import os
from os.path import join as pjoin

import debug_gym.gym.utils as utils
from debug_gym.gym.entities import EvalOutput
from debug_gym.gym.envs.env import RepoEnv


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
    def instructions(self) -> str:
        return self.current_sample["instructions"]

    def __init__(self, entrypoint: str = "python -m pytest -s test.py", **kwargs):
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
                pjoin(self.DATA_PATH, task_name, ".debugignore")
            ), f"Task {task_name} missing .debugignore file."
            assert os.path.exists(
                pjoin(self.DATA_PATH, task_name, ".debugreadonly")
            ), f"Task {task_name} missing .debugreadonly file."

        dataset = {}
        for task_name in self.TASK_NAMES:
            task_path = pjoin(self.DATA_PATH, task_name)

            dataset[task_name] = {
                "instructions": "The program doesn't behave as intended. Investigate the repository, figure out the root cause, then rewrite the code to fix the issue. Beaware that the bug may not be in the code you initially see.",
                "base_directory": task_path,
                "filename": task_name + "_code.py",
            }

        problems = utils.filter_problems(dataset, problems)
        dataset = {id: i for id, i in dataset.items() if id in problems}
        return dataset
