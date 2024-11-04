import json
import os
import subprocess
import uuid
from os.path import join as pjoin

import numpy as np
from termcolor import colored

from froggy.agents.llm_api import instantiate_llm


class AgentBase:
    name: str = "base"

    def __init__(self, config_dict, env, verbose=False, _uuid=None):
        self.config = config_dict
        self.env = env
        self.llm = instantiate_llm(self.config, verbose=verbose)
        self._uuid = str(uuid.uuid4()) if _uuid is None else _uuid
        self._output_path = pjoin(self.config["output_path"], self._uuid)
        os.makedirs(self._output_path, exist_ok=True)
        print(colored(f"Output will be saved in {self._output_path}", "magenta"))
        self.set_seed(self.config["random_seed"])

    def set_seed(self, seed):
        np.random.seed(seed)

    def build_prompt(self):
        raise NotImplementedError(
            "build_prompt should be implemented in the child class"
        )

    def run(self):
        raise NotImplementedError("run should be implemented in the child class")

    def apply_patch(self, patch_path: str) -> bool:
        patch_command = ["patch", "-p1"]
        try:
            # Open the patch file
            with open(patch_path, "r") as patch:
                # Run the patch command
                result = subprocess.run(
                    patch_command,
                    stdin=patch,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=True,
                )
            print("Patch applied successfully.")
            print("Output:", result.stdout)
            return True
        except subprocess.CalledProcessError as e:
            print("Failed to apply patch.")
            print("Error:", e.stderr)
            return False

    def save_patch(self, task_name="custom"):
        os.makedirs(pjoin(self._output_path, task_name), exist_ok=True)
        patch_path = pjoin(self._output_path, task_name, "froggy.patch")
        with open(patch_path, "w") as f:
            f.write(self.env.patch)
        print(f"Patch saved in {pjoin(self._output_path, task_name, 'froggy.patch')}")

    def log(self, task_name="custom"):
        jsonl_output = {
            "problem": task_name,
            "config": self.config,
            "uuid": self._uuid,
            "success": self.env.done,
            "log": [],
        }
        for step_id in range(len(self.history)):
            step_json = self.history.json(
                step_id,
                include_prompt_response_pairs=self.config["log_prompt_response_pairs"],
            )
            jsonl_output["log"].append(step_json)
        os.makedirs(pjoin(self._output_path, task_name), exist_ok=True)
        with open(pjoin(self._output_path, task_name, "froggy.jsonl"), "w") as f:
            json.dump(jsonl_output, f, indent=4)
        print(f"Log saved in {pjoin(self._output_path, task_name, 'froggy.jsonl')}")
