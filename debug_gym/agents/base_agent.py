import json
import os
import uuid
from dataclasses import MISSING, asdict, dataclass, field, fields
from typing import Any, Dict, List

from jinja2 import Environment, Template

from debug_gym.agents.history_tracker import HistoryTracker
from debug_gym.gym.envs.env import EnvInfo, RepoEnv
from debug_gym.gym.utils import filter_non_utf8
from debug_gym.llms.base import LLM, LLMResponse
from debug_gym.llms.utils import trim
from debug_gym.logger import DebugGymLogger

AGENT_REGISTRY = {}


def register_agent(cls):
    if not issubclass(cls, BaseAgent):
        raise ValueError("agent_class must be a subclass of BaseAgent")
    if cls.name is None:
        raise ValueError("agent_class must have a name attribute")
    AGENT_REGISTRY[cls.name.lower()] = cls
    return cls


@dataclass
class AgentArgs:
    system_prompt: str = ""
    instance_prompt: str = "Instructions: {{ info.instructions }}"
    max_steps: int = 100
    max_history_token_cutoff: int = -1
    max_history_steps_cutoff: int = -1
    uuid: str = field(default_factory=lambda: str(uuid.uuid4()))
    extras: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def make(cls, config: Dict[str, Any] | "AgentArgs") -> "AgentArgs":
        if isinstance(config, cls):
            return config
        return cls.from_dict(config)

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


class BaseAgent:
    name: str = None
    args_class = AgentArgs

    def __init__(
        self,
        agent_args: AgentArgs | Dict[str, Any] | None = None,
        llm: LLM | None = None,
        logger: DebugGymLogger | None = None,
    ):
        self.args = self.args_class.make(agent_args or {})
        self.history = HistoryTracker()
        self.logger = logger or DebugGymLogger("debug-gym")
        self.llm = llm
        self.env = None
        self.system_prompt = str(self.args.system_prompt)
        self.instance_prompt = str(self.args.instance_prompt)

    @staticmethod
    def to_pretty_json(value):
        """Convert a value to a pretty JSON string."""
        return json.dumps(value, indent=2, sort_keys=False)

    def trim_message(
        self,
        message: str,
        count_tokens=None,
        max_length=None,
        max_length_percentage=0,
        where="middle",
    ):
        """Filter non utf8 and trim the message to fit within the token limit.
        If the message exceeds the max_length, it will be trimmed to fit.
        The `max_length` can be specified as an absolute value or a percentage
        of the LLM's context length, if any."""
        message = filter_non_utf8(message)
        count_tokens = count_tokens or self.llm.count_tokens
        if self.llm.context_length is not None:
            max_length = (
                max_length
                or (max_length_percentage * self.llm.context_length)
                or self.llm.context_length
            )

        if count_tokens is None or max_length is None or max_length <= 0:
            return message

        return trim(message, max_length, count_tokens=count_tokens, where=where)

    def _load_prompt_template(self, template: str) -> Template:
        """Load a prompt template and register custom filters.

        Args:
            template (str): The prompt template as a string or
                            a Jinja file path with a `.jinja` extension.
        """
        if template.endswith(".jinja"):
            if not os.path.isfile(template):
                error_msg = f"Prompt template file `{template}` not found."
                self.logger.error(error_msg)
                raise FileNotFoundError(error_msg)

            with open(template, "r") as f:
                template = f.read()

        # Add custom filter to Jinja2 environment
        env = Environment()
        env.filters["to_pretty_json"] = self.to_pretty_json
        env.filters["trim_message"] = self.trim_message
        return env.from_string(template)

    def build_system_prompt(self, info: EnvInfo | None = None) -> dict:
        """Build system prompt using the default template or one provided in args."""
        system_prompt_template = self._load_prompt_template(self.system_prompt)
        self.logger.debug(f"Loaded system prompt template:\n{self.system_prompt}")
        system_prompt = system_prompt_template.render(agent=self, info=info)
        self.logger.debug(f"Rendered system prompt:\n{system_prompt}")

        # TODO: should we call self.llm.convert_observation_to_message(system_prompt) ?
        return {"role": "system", "content": filter_non_utf8(system_prompt)}

    def build_instance_prompt(self, info: EnvInfo | None = None) -> dict:
        """Build instance prompt using the default template or one provided in args."""
        instance_prompt_template = self._load_prompt_template(self.instance_prompt)
        self.logger.debug(f"Loaded instance prompt template:\n{self.instance_prompt}")
        instance_prompt = instance_prompt_template.render(agent=self, info=info)
        self.logger.debug(f"Rendered instance prompt:\n{instance_prompt}")
        return self.llm.convert_observation_to_message(instance_prompt)

    def build_history_prompt(self) -> list[dict]:
        """Here, we rebuild the history prompt from scratch at each time."""
        messages = []
        for llm_response, next_observation in zip(
            self.history.llm_responses, self.history.env_observations, strict=True
        ):
            # llm response
            messages.append(self.llm.convert_response_to_message(llm_response))
            # next environment observation
            kwargs = {
                "observation": next_observation.step_observation.observation,
            }
            if next_observation.action_tool_call:
                kwargs["action_tool_call_id"] = next_observation.action_tool_call.id
                kwargs["action_tool_call_name"] = next_observation.action_tool_call.name
            messages.append(self.llm.convert_observation_to_message(**kwargs))
        return messages

    def build_prompt(self, info: EnvInfo = None):
        messages = []
        messages.append(self.build_system_prompt(info))
        messages.append(self.build_instance_prompt(info))
        messages.extend(self.build_history_prompt())

        if self.args.max_history_steps_cutoff > 0:
            # keep at most max_history_steps_cutoff history messages including the first 2 messages (system and instance prompts)
            if len(messages) > self.args.max_history_steps_cutoff + 2:
                messages = (
                    messages[:2] + messages[-(self.args.max_history_steps_cutoff) :]
                )

        if self.args.max_history_token_cutoff > 0:
            first_two = messages[:2]
            remaining = messages[2:]

            # Calculate tokens for the first two messages
            first_two_tokens = sum(self.llm.count_tokens(msg) for msg in first_two)
            available_tokens = self.args.max_history_token_cutoff - first_two_tokens

            # Add messages from the end until we exceed the token limit
            kept_messages = []
            current_tokens = 0
            for msg in reversed(remaining):
                msg_tokens = self.llm.count_tokens(msg)
                if current_tokens + msg_tokens <= available_tokens:
                    kept_messages.insert(0, msg)
                    current_tokens += msg_tokens
                else:
                    break
            messages = first_two + kept_messages
        return messages

    def should_stop(self, step: int, info: EnvInfo):
        should_stop, reason = False, None
        max_steps_reached = step >= self.args.max_steps
        if info.terminated:
            should_stop = True
            reason = "terminated"
        elif max_steps_reached:
            should_stop = True
            reason = "max_steps reached"
        return should_stop, reason

    def init(self, info: EnvInfo) -> None:
        """Initialize the agent with environment

        Args:
            info: The environment info to interact with.
        """
        self.history.init(
            self.build_system_prompt(info), self.build_instance_prompt(info), info
        )

        self.logger.info(
            "Available tools (in LLM's tool calling format):\n"
            f"{json.dumps(self.llm.define_tools(info.tools), indent=4)}\n"
        )

    def step(self, info: EnvInfo) -> LLMResponse | List[LLMResponse]:
        """Execute a single agent step (LLM decision only).

        Args:
            info: Current environment info.

        Returns:
            LLMResponse with the agent's decision.
        """
        messages = self.build_prompt(info)
        return self.llm(messages, info.tools)

    def execute_action(self, llm_response: LLMResponse | List[LLMResponse]) -> EnvInfo:
        next_info = self.env.step(
            llm_response.tool,
            llm_response.response,
            llm_response.reasoning_response,
        )
        self.history.step(next_info, llm_response)
        return next_info

    def build_trajectory(self) -> Dict[str, Any]:
        """Return the trajectory as a JSON-serializable dict without writing it."""
        tools = [f"{tool.name}({tool.arguments})" for tool in self.env.tools]
        json_output = {
            "problem": self.env.task_name,
            "config": self.args.to_dict(),
            "tools": self.llm.define_tools(self.env.tools) if self.llm else tools,
            "uuid": self.args.uuid,
            "success": self.env.resolved,
            "log": [],
            "agent_type": self.__class__.__name__,
            "logger": str(self.logger.log_file),
        }
        for step_id in range(len(self.history)):
            step_json = self.history.json(step_id)
            json_output["log"].append(step_json)
        return json_output

    def run(
        self,
        env: RepoEnv,
        debug: bool = False,
        reset_env: bool = True,
    ) -> Dict[str, Any]:
        """Run the agent loop until termination or max steps.

        Args:
            env: The environment to interact with.
            debug: Whether to drop into debugger after each LLM call.
            reset_env: Whether to reset the environment (default True).

        Returns:
            The trajectory as a JSON-serializable dict.
        """
        info = None
        step = 0

        # assign the env
        self.env = env

        try:
            if reset_env:
                info = env.reset()
            else:
                info = env.info

            self.init(info)

            if info.resolved:
                self.logger.report_progress(
                    problem_id=env.task_name,
                    step=0,
                    total_steps=self.args.max_steps,
                    score=info.score,
                    max_score=info.max_score,
                    status="resolved",
                )
                return self.build_trajectory()

            highscore = info.score
            should_stop = False
            step = 1

            while not should_stop:
                self.logger.info(f"\n{'='*20} STEP {step} {'='*20}\n")

                agent_response = self.step(info)
                next_info = self.execute_action(agent_response)

                if debug:
                    breakpoint()

                should_stop, reason = self.should_stop(step + 1, next_info)
                status = (
                    "resolved"
                    if info.resolved
                    else ("unresolved" if should_stop else "running")
                )

                highscore = max(highscore, next_info.score)
                msg = f"[{env.task_name[:10]:<10}] Step {step} | Score: {next_info.score}/{next_info.max_score or '-'} [Best: {highscore}]"
                if should_stop:
                    msg += f" | Stopping Reason: {reason}"
                self.logger.info(msg)
                step += 1

                self.logger.report_progress(
                    problem_id=env.task_name,
                    step=step,
                    total_steps=self.args.max_steps,
                    score=next_info.score,
                    max_score=next_info.max_score,
                    status=status,
                )
            return self.build_trajectory()
        except Exception as e:
            self.logger.report_progress(
                problem_id=env.task_name,
                step=step,
                total_steps=step,
                score=getattr(next_info, "score", 0),
                max_score=getattr(next_info, "max_score", None),
                status="error",
            )
            raise e


def create_agent(config: Dict[str, Any], **kwargs) -> BaseAgent:
    """Create an agent from the config dictionary."""

    agent_type = config.get("type", "froggy")
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

    agent = agent_class(agent_args=config, **kwargs)
    return agent
