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
    system_prompt: str | None = None
    instance_prompt: str | None = None
    random_seed: int = 42
    max_steps: int = 100
    max_rewrite_steps: int = -1
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
    instance_prompt: str = None

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

        self.env = None
        self._uuid = self.args.uuid
        if self.args.system_prompt:
            self.system_prompt = self.args.system_prompt
        if self.args.instance_prompt:
            self.instance_prompt = self.args.instance_prompt
        self.logger = logger or DebugGymLogger("debug-gym")
        self.llm = llm
        self.set_seed(self.args.random_seed)

        self.history = HistoryTracker()

    def set_seed(self, seed):
        np.random.seed(seed)

    def build_system_prompt(self):
        return [
            {"role": "system", "content": self.system_prompt},
            self.llm.convert_observation_to_message(self.instance_prompt),
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

    def should_stop(self, step: int, info: EnvInfo):
        should_stop, reason = False, None
        max_steps_reached = step + 1 >= self.args.max_steps
        if info.terminated:
            should_stop = True
            reason = "terminated"
        elif max_steps_reached:
            should_stop = True
            reason = "max_steps reached"
        return should_stop, reason

    def run(self, env: RepoEnv, debug=False):
        step = 0
        info = None
        self.env = env

        try:
            info = self.env.reset()
            self.history.init(self.system_prompt, self.instance_prompt, info)

            if info.resolved:
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
            for step in range(self.args.max_steps):
                self.logger.info(f"\n{'='*20} STEP {step+1} {'='*20}\n")

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
                should_stop, reason = self.should_stop(step, info)

                highscore = max(highscore, info.score)
                msg = f"[{env.task_name[:10]:<10}] Step {step} | Score: {info.score}/{info.max_score or '-'} [Best: {highscore}]"
                if should_stop:
                    msg += f" | Stopping Reason: {reason}"
                self.logger.info(msg)

                # keep progress bar running until max_steps is reached
                self.logger.report_progress(
                    problem_id=env.task_name,
                    step=step + 1,
                    total_steps=self.args.max_steps + 1,
                    score=info.score,
                    max_score=info.max_score,
                    status=(
                        "resolved"
                        if info.resolved
                        else ("unresolved" if should_stop else "running")
                    ),
                )
                if should_stop:
                    break

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
