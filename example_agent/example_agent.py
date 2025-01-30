import json
import os
import subprocess
import uuid
from os.path import join as pjoin

import numpy as np

from example_agent.llm_api import instantiate_llm
from example_agent.utils import HistoryTracker, build_history_prompt
from froggy.envs.env import RepoEnv
from froggy.logger import FroggyLogger
from froggy.utils import unescape


class PdbAgent:
    name: str = "pdb agent"

    def __init__(
        self,
        config: dict,
        env: RepoEnv,
        logger: FroggyLogger | None = None,
    ):
        self.config = config
        self.env = env
        self.logger = logger or FroggyLogger("froggy")
        self.llm = instantiate_llm(self.config, logger=self.logger)
        self._uuid = self.config.get("uuid", str(uuid.uuid4()))
        self._output_path = pjoin(self.config["output_path"], self._uuid)

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

    def build_question_prompt(self):
        messages = []
        question = "Based on the instruction, the current code, the last execution output, and the history information, "
        question += "continue your debugging process using pdb commands or to propose a patch using rewrite command. "
        question += "Output a single command, nothing else. Do not repeat your previous commands unless they can provide more information."
        messages.append({"role": "user", "content": question})
        return messages

    def build_prompt(self, info):
        messages = self.build_system_prompt(info)
        messages.extend(self.build_history_prompt())
        messages.extend(self.build_question_prompt())
        return messages

    def run(self, task_name=None, debug=False):
        self.history.reset()
        _, info = self.env.reset(options={"task_name": task_name})
        self.history.step(info)

        if info["done"] is True:
            # msg = "Environment started with entrypoint passing without errors."
            return True

        done = False
        highscore = info["score"]

        for step in self.logger.tqdm(range(self.config["max_steps"])):
            highscore = max(highscore, info["score"])
            self.logger.info(
                f"Score: {info['score']}/{info['max_score']} ({info['score']/info['max_score']:.1%}) [Best: {highscore}]".format(
                    info["score"]
                )
            )

            prompt = self.build_prompt(info)
            answer, token_usage = self.llm(
                prompt, info, temperature=self.config["llm_temperature"][0]
            )

            if debug:
                breakpoint()

            _, _, done, info = self.env.step(answer)
            info["token_usage"] = [
                token_usage
            ]  # in some other agents this is a list because of multi-step llm calls
            self.history.step(info)
            self.history.save_prompt_response_pairs(
                prompt_response_pairs=[(prompt, answer)]
            )

            if done or info["rewrite_counter"] >= self.config["max_rewrite_steps"]:
                self.logger.info(
                    f"Score: {info['score']}/{info['max_score']} ({info['score']/info['max_score']:.1%})".format(
                        info["score"]
                    )
                )
                break
        return done

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


class RewriteOnly(PdbAgent):
    name: str = "rewrite only"

    def build_system_prompt(self, info):
        system_prompt = {}
        system_prompt["Overall task"] = (
            "Your goal is to debug a Python program to make sure it can pass a set of test functions. You need to propose a rewriting patch to fix the bugs. Avoid rewriting the entire code, focus on the bugs only."
        )
        system_prompt["Instructions"] = info["instructions"]
        system_prompt["Repo directory tree"] = info["dir_tree"]
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

    def build_question_prompt(self):
        messages = []
        question = "Based on the instruction, the current code, the last execution output, and the history information, "
        question += (
            "continue your debugging process to propose a patch using rewrite command. "
        )
        question += "Output a single command, nothing else. Do not repeat your previous commands unless they can provide more information."
        messages.append({"role": "user", "content": question})
        return messages


class PdbAfterRewrites(PdbAgent):
    name: str = "pdb after rewrites"

    def run(self, task_name=None, debug=False):
        # remove the pdb tool from the environment
        assert "pdb" in self.env.tools, "pdb not found in env tools"
        pdb_tool = self.env.tools.pop("pdb")

        self.history.reset()
        _, info = self.env.reset(options={"task_name": task_name})
        self.history.step(info)

        if info["done"] is True:
            # msg = "Environment started with entrypoint passing without errors."
            return True

        done = False
        highscore = info["score"]

        for step in self.logger.tqdm(range(self.config["max_steps"])):
            highscore = max(highscore, info["score"])
            self.logger.info(
                f"Score: {info['score']}/{info['max_score']} ({info['score']/info['max_score']:.1%}) [Best: {highscore}]".format(
                    info["score"]
                )
            )

            prompt = self.build_prompt(info)
            answer, token_usage = self.llm(
                prompt, info, temperature=self.config["llm_temperature"][0]
            )

            if debug:
                breakpoint()

            _, _, done, info = self.env.step(answer)
            info["token_usage"] = [
                token_usage
            ]  # in some other agents this is a list because of multi-step llm calls

            # re-introduce pdb tool at the right time
            if (
                info["rewrite_counter"] >= self.config["n_rewrites_before_pdb"]
                and pdb_tool.name not in self.env.tools
            ):
                self.env.add_tool(pdb_tool)
                self.env.tools["pdb"].start_pdb()
                info["instructions"] = self.env.instructions
                info["obs"] += "\nThe pdb tool has been added."

            self.history.step(info)
            self.history.save_prompt_response_pairs(
                prompt_response_pairs=[(prompt, answer)]
            )

            if done or info["rewrite_counter"] >= self.config["max_rewrite_steps"]:
                self.logger.info(
                    f"Score: {info['score']}/{info['max_score']} ({info['score']/info['max_score']:.1%})".format(
                        info["score"]
                    )
                )
                break

        return done


class AgentSolution:
    name: str = "solution"

    def __init__(
        self,
        config: dict,
        env: RepoEnv,
        logger: FroggyLogger | None = None,
    ):
        self.config = config
        self.env = env
        self.logger = logger or FroggyLogger("froggy")
        self.llm = instantiate_llm(self.config, logger=self.logger)
        self._uuid = self.config.get("uuid", str(uuid.uuid4()))
        self._output_path = pjoin(self.config["output_path"], self._uuid)

        os.makedirs(self._output_path, exist_ok=True)

    def run(self, task_name=None, debug=False):
        self.history.reset()
        _, info = self.env.reset(options={"task_name": task_name})
        self.history.step(info)

        if info["done"] is True:
            return True

        done = False
        highscore = info["score"]

        for step in self.logger.tqdm(range(self.config["max_steps"])):
            highscore = max(highscore, info["score"])
            self.logger.info(
                f"Score: {info['score']}/{info['max_score']} ({info['score']/info['max_score']:.1%}) [Best: {highscore}]".format(
                    info["score"]
                )
            )

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

            if done or info["rewrite_counter"] >= self.config["max_rewrite_steps"]:
                self.logger.info(
                    f"Score: {info['score']}/{info['max_score']} ({info['score']/info['max_score']:.1%})".format(
                        info["score"]
                    )
                )
                break

        return done
