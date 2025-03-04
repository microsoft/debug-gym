import json
import os
import subprocess
import uuid
from os.path import join as pjoin

import numpy as np

from debug_gym.agents.llm_api import instantiate_llm
from debug_gym.agents.utils import HistoryTracker, build_history_prompt
from debug_gym.gym.envs.env import RepoEnv
from debug_gym.gym.utils import unescape
from debug_gym.logger import DebugGymLogger

AGENT_REGISTRY = {}


def register_agent(cls):
    if not issubclass(cls, BaseAgent):
        raise ValueError("agent_class must be a subclass of BaseAgent")
    if cls.name is None:
        raise ValueError("agent_class must have a name attribute")
    AGENT_REGISTRY[cls.name.lower()] = cls
    return cls


class BaseAgent:
    name: str = None
    system_prompt: str = None
    action_prompt: str = None

    def __init__(
        self,
        config: dict,
        env: RepoEnv,
        logger: DebugGymLogger | None = None,
    ):
        self.config = config
        self.env = env
        self.logger = logger or DebugGymLogger("debug-gym")
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

    def parse_reasoning_model_response(self, response, reasoning_end_token):
        # Strip the reasoning, e.g., in Deepseek r1, between <think> and </think>.
        reasoning_end = response.find(reasoning_end_token)
        if reasoning_end != -1:
            reasoning_end += len(reasoning_end_token)
            response = response[reasoning_end:].strip()
        return response

    def build_system_prompt(self, info):
        system_prompt = {}
        system_prompt["Overall task"] = self.system_prompt
        system_prompt["Instructions"] = info.instructions
        system_prompt["Repo directory tree"] = info.dir_tree
        system_prompt["Current code in view"] = info.current_code_with_line_number
        system_prompt["Current breakpoints"] = info.current_breakpoints
        system_prompt["Last evaluation output"] = info.eval_observation.observation

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
        messages.append({"role": "user", "content": self.action_prompt})
        return messages

    def build_prompt(self, info):
        messages = self.build_system_prompt(info)
        messages.extend(self.build_history_prompt())
        messages.extend(self.build_question_prompt())
        return messages

    def run(self, task_name=None, debug=False):
        self.history.reset()
        info = self.env.reset(options={"task_name": task_name})
        # initial state does not have prompt and response
        self.history.step(info, None)

        if info.done is True:
            # msg = "Environment started with entrypoint passing without errors."
            return True

        highscore = info.score

        for step in self.logger.tqdm(range(self.config["max_steps"])):
            highscore = max(highscore, info.score)
            self.logger.info(
                f"Score: {info.score}/{info.max_score} ({info.score/info.max_score:.1%}) [Best: {highscore}]"
            )

            prompt = self.build_prompt(info)
            llm_response = self.llm(
                prompt, info, temperature=self.config["llm_temperature"][0]
            )
            if self.llm.reasoning_end_token is not None:
                llm_response.response = self.parse_reasoning_model_response(
                    llm_response.response,
                    reasoning_end_token=self.llm.reasoning_end_token,
                )

            if debug:
                breakpoint()

            info = self.env.step(llm_response.response)
            self.history.step(info, llm_response)

            if info.done or info.rewrite_counter >= self.config["max_rewrite_steps"]:
                self.logger.info(
                    f"Score: {info.score}/{info.max_score} ({info.score/info.max_score:.1%})"
                )
                break
        return info.done

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
        patch_path = pjoin(self._output_path, task_name, "debug_gym.patch")
        with open(patch_path, "w") as f:
            f.write(self.env.patch)

        self.logger.debug(
            f"Patch saved in {pjoin(self._output_path, task_name, 'debug_gym.patch')}"
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
        with open(pjoin(self._output_path, task_name, "debug_gym.jsonl"), "w") as f:
            json.dump(jsonl_output, f, indent=4)

        self.logger.debug(
            f"Log saved in {pjoin(self._output_path, task_name, 'debug_gym.jsonl')}"
        )


def create_agent(agent_type: str, **agent_kwargs):
    if agent_type in AGENT_REGISTRY:
        agent_class = AGENT_REGISTRY[agent_type]
    elif "." in agent_type:
        # try to import agent_type module
        import importlib

        parts = agent_type.split(".")
        module_name = ".".join(parts[:-1])
        class_name = parts[-1]

        module = importlib.import_module(module_name)
        agent_class = getattr(module, class_name)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

    agent = agent_class(**agent_kwargs)
    return agent
