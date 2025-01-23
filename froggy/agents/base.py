import json
import logging
import os
import subprocess
import uuid
from os.path import join as pjoin

import numpy as np
from termcolor import colored
from tqdm import tqdm

from froggy.agents.llm_api import instantiate_llm
from froggy.agents.utils import HistoryTracker, build_history_prompt
from froggy.utils import unescape

# logger = logging.getLogger("froggy")


class AgentBase:
    name: str = "base"

    def __init__(
        self,
        config_dict,
        env,
        verbose=False,
        _uuid=None,
        logger=logging.getLogger("froggy"),
    ):
        self.config = config_dict
        self.env = env
        self.llm = instantiate_llm(self.config, verbose=verbose)
        _uuid = self.config.get("uuid", _uuid)
        self._uuid = str(uuid.uuid4()) if _uuid is None else _uuid
        self._output_path = pjoin(self.config["output_path"], self._uuid)
        self.logger = logger

        os.makedirs(self._output_path, exist_ok=True)

        self.set_seed(self.config["random_seed"])
        self.history = HistoryTracker(self.config["memory_size"])

    def set_seed(self, seed):
        np.random.seed(seed)

    def build_history_prompt(self):
        messages = build_history_prompt(
            self.history,
            self.config["use_conversational_prompt"],
            self.config["reset_prompt_history_after_rewrite"],
        )
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

        self.logger.debug(
            f"Patch saved in {pjoin(self._output_path, task_name, 'froggy.patch')}"
        )

    def log(self, task_name="custom"):
        jsonl_output = {
            "problem": task_name,
            "config": self.config,
            "uuid": self._uuid,
            "success": self.env.done,
            "log": [],
            "agent_type": self.__class__.__name__,
            "logger": str(self.logger.log_file),
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

        self.logger.debug(
            f"Log saved in {pjoin(self._output_path, task_name, 'froggy.jsonl')}"
        )


class AgentSolution(AgentBase):
    name: str = "solution"

    def __init__(self, config_dict, env, verbose=False, _uuid=None, **kwargs):
        super().__init__(config_dict, env, verbose, _uuid, **kwargs)

    def run(self, task_name=None, debug=False):
        self.history.reset()
        _, info = self.env.reset(options={"task_name": task_name})
        self.history.step(info)

        if info["done"] is True:
            return True

        done = False
        highscore = info["score"]

        pbar = tqdm(
            total=self.config["max_steps"],
            desc=f"Debugging inside {self.env.working_dir} - Task: {task_name}",
            leave=True,
            file=open(os.devnull, "w"),
        )
        for step in range(self.config["max_steps"]):
            highscore = max(highscore, info["score"])
            pbar.set_postfix_str(
                f"Score: {info['score']}/{info['max_score']} ({info['score']/info['max_score']:.1%}) [Best: {highscore}]".format(
                    info["score"]
                )
            )
            self.logger.info(str(pbar))

            self.logger.info(f"Applying gold patch to {self.env.working_dir}.")
            command = f"git -C {self.env.working_dir} apply -"
            subprocess.run(
                command.split(), input=self.env.gold_patch, text=True, check=True
            )
            self.logger.info("Patch applied successfully.")

            if debug:
                breakpoint()

            _, _, done, info = self.env.step("```eval```")
            info["token_usage"] = [0]
            self.history.step(info)

            pbar.update()
            self.logger.info(str(pbar))
            if done or info["rewrite_counter"] >= self.config["max_rewrite_steps"]:
                pbar.set_postfix_str(
                    f"Score: {info['score']}/{info['max_score']} ({info['score']/info['max_score']:.1%})".format(
                        info["score"]
                    )
                )
                self.logger.info(str(pbar))
                break

        return done
