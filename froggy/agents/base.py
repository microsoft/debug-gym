import json
import os
import subprocess
import uuid
from os.path import join as pjoin

import numpy as np
from termcolor import colored

from froggy.agents.llm_api import instantiate_llm
from froggy.utils import unescape


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

    def _build_history_conversation(self):
        _history = self.history.get()
        # Find the latest rewrite step
        if (
            len(_history) == 0
            or self.config["reset_prompt_history_after_rewrite"] is False
        ):
            latest_rewrite_step = 0
        else:
            for i in range(len(_history)):
                if _history[i]["rewrite_counter"] == _history[-1]["rewrite_counter"]:
                    latest_rewrite_step = i
                    break
        _messages = []
        for history_info in _history[latest_rewrite_step:]:
            if history_info["action"] is not None:
                _messages.append(
                    {"role": "assistant", "content": f"{history_info["action"]}"}
                )
            _messages.append({"role": "user", "content": f"{history_info["obs"]}"})
        return _messages

    def _build_history_non_conversation(self):
        _history = self.history.get()
        # Find the latest rewrite step
        if (
            len(_history) == 0
            or self.config["reset_prompt_history_after_rewrite"] is False
        ):
            latest_rewrite_step = 0
        else:
            for i in range(len(_history)):
                if _history[i]["rewrite_counter"] == _history[-1]["rewrite_counter"]:
                    latest_rewrite_step = i
                    break
        _history_prompt = []
        for _i, history_info in enumerate(_history):
            if _i < latest_rewrite_step:
                continue
            _m = {
                "step": _i,
                "command": (
                    None if history_info["action"] is None else history_info["action"]
                ),
                "stdout": history_info["obs"],
            }
            _history_prompt.append(_m)
        return _history_prompt

    def build_history_prompt(self):
        messages = []
        if self.config["use_conversational_prompt"] is True:
            conversation_history = self._build_history_conversation()
            messages.append(
                {
                    "role": "user",
                    "content": f"History of command and terminal outputs (the last {(len(conversation_history) + 1) // 2} steps):",
                }
            )
            messages.extend(conversation_history)
        else:
            history_prompt = self._build_history_non_conversation()
            prompt = [
                f"History of command and terminal outputs (the last {len(history_prompt)} steps):"
            ]
            prompt += ["\n" + unescape(json.dumps(history_prompt, indent=4)) + "\n"]
            messages.append({"role": "user", "content": "\n".join(prompt)})
        return messages

    def build_system_prompt(self, info):
        system_prompt = {}
        system_prompt["Overall task"] = (
            "Your goal is to debug a Python program to make sure it can pass a set of test functions. You have access to the pdb debugger tools, you can use them to investigate the code, set breakpoints, and print necessary values to identify the bugs. Once you have gained enough information, propose a rewriting patch to fix the bugs. Avoid rewriting the entire code, focus on the bugs only."
        )
        system_prompt["Instructions"] = info["instructions"]
        system_prompt["Repo directory tree"] = info["dir_tree"]
        system_prompt["Editable files"] = info["editable_files"]
        system_prompt["Current code in view"] = info["current_code_with_line_number"]
        system_prompt["Current breakpoints"] = info["current_breakpoints"]
        system_prompt["Last execution output"] = info["last_run_obs"]
        system_prompt = unescape(json.dumps(system_prompt, indent=4))
        messages = [
            {
                "role": "system",
                "content": system_prompt,
            }
        ]
        return messages

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
