import json
import os
import subprocess
import uuid
from collections import namedtuple
from copy import copy
from dataclasses import MISSING, asdict, dataclass, field, fields
from typing import Any, Dict

import numpy as np
from jinja2 import Environment, Template

from debug_gym.agents.history_tracker import HistoryTracker, build_history_prompt
from debug_gym.gym.envs.env import EnvInfo, RepoEnv
from debug_gym.gym.utils import filter_non_utf8
from debug_gym.llms.base import LLM
from debug_gym.llms.utils import trim
from debug_gym.logger import DebugGymLogger

AGENT_REGISTRY = {}


@dataclass
class AgentArgs:
    random_seed: int = 42
    max_rewrite_steps: int = -1
    max_steps: int = 100
    system_prompt: str | None = None
    problem_prompt: str | None = None
    uuid: str = field(default_factory=lambda: str(uuid.uuid4()))
    extras: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "AgentArgs":
        # Get all field names from the dataclass
        field_names = {f.name for f in fields(cls)}

        # Check for required fields (those without defaults)
        required_fields = {
            f.name
            for f in fields(cls)
            if f.default is MISSING and f.default_factory is MISSING
        }
        missing = required_fields - config.keys()
        if missing:
            raise ValueError(
                f"Missing required agent config keys: {', '.join(sorted(missing))}"
            )

        # Separate known fields from extras
        known_values = {k: v for k, v in config.items() if k in field_names}
        extras = {k: v for k, v in config.items() if k not in field_names}

        # Add extras if that field exists
        if "extras" in field_names:
            known_values["extras"] = extras

        return cls(**known_values)

    def get(self, key: str, default=None):
        if key in self.__dataclass_fields__:
            return getattr(self, key)
        return self.extras.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        extras = data.pop("extras", {})
        data.update(extras)
        return data


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
    problem_prompt: str = None

    def __init__(
        self,
        agent_args: AgentArgs | Dict[str, Any],
        llm: LLM | None = None,
        logger: DebugGymLogger | None = None,
    ):
        self.args = (
            AgentArgs.from_dict(agent_args)
            if isinstance(agent_args, dict)
            else agent_args
        )
        self.logger = logger or DebugGymLogger("debug-gym")
        self.llm = llm
        self._uuid = self.args.uuid
        self.env = None
        self.system_prompt = self.args.system_prompt or self.system_prompt
        self.problem_prompt = self.args.problem_prompt or self.problem_prompt

        self.set_seed(self.args.random_seed)
        self.history = HistoryTracker()

    def set_seed(self, seed):
        np.random.seed(seed)

    def parse_reasoning_model_response(self, response, reasoning_end_token):
        # Strip the reasoning, e.g., in Deepseek r1, between <think> and </think>.
        reasoning_end = response.find(reasoning_end_token)
        if reasoning_end != -1:
            reasoning_end += len(reasoning_end_token)
            response = response[reasoning_end:].strip()
        return response

    def build_system_prompt(self):
        return [
            {"role": "system", "content": self.system_prompt},
            self.llm.convert_observation_to_message(self.problem_prompt),
        ]

    def build_history_prompt(self):
        messages = []
        for observation, response in zip(
            self.history.env_observations, self.history.llm_responses
        ):
            # environment observation
            messages.extend(
                self.llm.convert_observation_to_message(
                    observation.step_observation.observation,
                    (
                        observation.action_tool_call.id
                        if observation.action_tool_call
                        else None
                    ),
                    (
                        observation.action_tool_call.name
                        if observation.action_tool_call
                        else None
                    ),
                )
            )
            # llm response
            messages.extend(self.llm.convert_response_to_message(response))
        return messages

    def build_prompt(self, info: EnvInfo = None):
        messages = []
        messages.extend(self.build_system_prompt())
        messages.extend(self.build_history_prompt())
        return messages

    def run(self, env: RepoEnv, debug=False):
        step = 0
        info = None
        self.env = env
        max_steps = self.args.max_steps

        try:
            info = self.env.reset()

            # initial state does not have prompt and response
            self.history.init(self.system_prompt, self.problem_prompt, info)

            if info.resolved is True:
                self.logger.report_progress(
                    problem_id=env.task_name,
                    step=1,
                    total_steps=1,
                    score=info.score,
                    max_score=info.max_score,
                    status="resolved",
                )
                return self._build_trajectory()

            self.logger.info(
                "Available tools (in LLM's tool calling format):\n"
                f"{json.dumps(self.llm.define_tools(info.tools), indent=4)}\n"
            )

            highscore = info.score
            current_status = "running"

            while step < max_steps or current_status not in ["resolved", "unresolved"]:
                self.logger.info(f"\n{'='*20} STEP {step+1} {'='*20}\n")
                highscore = max(highscore, info.score)
                msg = f"[{env.task_name[:10]:<10}] Step {step} | Score: {info.score}/{info.max_score or '-'} [Best: {highscore}]"
                self.logger.info(msg)

                messages = self.build_prompt(info)
                llm_response = self.llm(messages, info.tools)

                if debug:
                    breakpoint()

                info = self.env.step(
                    llm_response.tool,
                    llm_response.response,
                    llm_response.reasoning_response,
                )
                self.history.step(info, llm_response)

                if info.terminated or (
                    self.args.max_rewrite_steps >= 0
                    and info.rewrite_counter >= self.args.max_rewrite_steps
                ):
                    reason = (
                        "terminated" if info.resolved else "max_rewrite_steps reached"
                    )
                    self.logger.info(
                        f"Step: {step} | Score: {info.score}/{info.max_score if info.max_score else '-'} | Reason: {reason}"
                    )
                    current_status = "resolved" if info.resolved else "unresolved"
                else:
                    current_status = "running"

                # keep progress bar running until max_steps is reached
                self.logger.report_progress(
                    problem_id=env.task_name,
                    step=step + 1,
                    total_steps=max_steps + 1,
                    score=info.score,
                    max_score=info.max_score,
                    status=current_status,
                )
            return self._build_trajectory()
        except Exception:
            # report any error that happens during the run
            self.logger.report_progress(
                problem_id=env.task_name,
                step=step + 1,
                total_steps=step + 1,
                score=info.score if info else 0,
                max_score=info.max_score,
                status="error",
            )
            raise

    def _build_trajectory(self) -> Dict[str, Any]:
        """Return the trajectory as a JSON-serializable dict without writing it."""
        tools = [f"{tool.name}({tool.arguments})" for tool in self.env.tools]
        json_output = {
            "problem": self.env.task_name,
            "config": self.args.to_dict(),
            "tools": self.llm.define_tools(self.env.tools) if self.llm else tools,
            "uuid": self._uuid,
            "success": self.env.resolved,
            "log": [],
            "agent_type": self.__class__.__name__,
            "logger": str(self.logger.log_file),
        }
        for step_id in range(len(self.history)):
            step_json = self.history.json(step_id)
            json_output["log"].append(step_json)
        return json_output


def create_agent(
    agent_type: str,
    *,
    agent_args: AgentArgs | Dict[str, Any] | None = None,
    config: Dict[str, Any] | None = None,
    **agent_kwargs,
):
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

    agent_args = agent_args or config
    if agent_args is None:
        raise ValueError("Either agent_args or config must be provided.")

    agent = agent_class(args=agent_args, **agent_kwargs)
    return agent
