import json
import os
import shutil
import tempfile
from os.path import join as pjoin
from pathlib import Path

from froggy.envs.env import RepoEnv


class TerminalSimulatorEnv(RepoEnv):

    @property
    def instructions(self):
        _instruction = {
            "Problem description": self.current_sample["instructions"],
            "Available tools to solve the problem": self.tool_instructions,
            "Available commands": self.actions_str,
        }
        return _instruction

    def __init__(self, bug_free_code_path, buggy_code_path, **kwargs):
        super().__init__(**kwargs)
        self.bug_free_code_path = bug_free_code_path
        self.buggy_code_path = buggy_code_path
        self.load_dataset()
        self.tempdir = tempfile.TemporaryDirectory(prefix="TerminalSimulatorEnv-")
        self.tempdir_name = self.tempdir.name

    def reset(self, *, seed=None, options: dict = None):
        options = options or {}
        self.tempdir.cleanup()
        self.current_sample = self.dataset[options["task_name"]]
        shutil.copytree(self.bug_free_code_path, self.tempdir_name, dirs_exist_ok=True)
        for change_id in range(len(self.current_sample["original path"])):
            ori_path = self.current_sample["original path"][change_id]
            new_code = self.current_sample["new code"][change_id]
            assert os.path.exists(pjoin(self.tempdir_name, ori_path))
            with open(pjoin(self.tempdir_name, ori_path), "w") as f:
                f.write(new_code)

        self.setup_workspace(
            self.tempdir_name, entrypoint=self.current_sample["entry_point"]
        )

        infos = super().reset()
        infos.instructions = self.instructions

        # By default, open the only modifiable file.
        self.load_current_file(self.current_sample["default_file_name"])
        # an update the infos related to current code.
        infos.current_code_with_line_number = self.current_code_with_line_number()
        return infos

    def load_dataset(self):
        assert os.path.exists(
            self.bug_free_code_path
        ), f"Bug free code path {self.bug_free_code_path} does not exist."
        assert os.path.exists(
            pjoin(self.bug_free_code_path, ".froggyignore")
        ), f"Bug free code path {self.bug_free_code_path} does not contain .froggyignore file."
        assert os.path.exists(
            self.buggy_code_path
        ), f"Buggy code path {self.buggy_code_path} does not exist."

        self.dataset = {}
        with open(self.buggy_code_path, "r") as f:
            buggy_code_info = json.load(f)
        buggy_code_info = buggy_code_info["data"]
        dataset_size = len(buggy_code_info)
        for i in range(dataset_size):
            self.dataset[buggy_code_info[i]["id"]] = {
                "original path": buggy_code_info[i]["original_code_paths"],  # list
                "new code": buggy_code_info[i]["buggy_code_list"],  # list
                "entry_point": "python -m pytest -sq test.py",
                "instructions": "The program doesn't behave as intended. Investigate the repository, figure out the root cause, then rewrite the code to fix the issue. Beaware that the bug may not be in the code you initially see.",
                "default_file_name": "code/run_terminal_simulator.py",
            }
