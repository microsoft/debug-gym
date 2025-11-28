import json
import os
import subprocess
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict

import numpy as np
from jinja2 import Environment, Template

from debug_gym.agents.history_tracker import HistoryTracker, build_history_prompt
from debug_gym.gym.envs.env import RepoEnv
from debug_gym.gym.utils import filter_non_utf8
from debug_gym.llms.base import LLM
from debug_gym.llms.utils import trim
from debug_gym.logger import DebugGymLogger

AGENT_REGISTRY = {}


@dataclass
class AgentArgs:
    random_seed: int
    memory_size: int
    max_steps: int
    max_rewrite_steps: int
    show_directory_tree: int = 0
    show_current_breakpoints: bool = False
    reset_prompt_history_after_rewrite: bool = False
    system_prompt_template_file: str | None = None
    n_rewrites_before_pdb: int = 0
    uuid: str = field(default_factory=lambda: str(uuid.uuid4()))
    extras: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "AgentArgs":
        required_keys = ["random_seed", "memory_size", "max_steps", "max_rewrite_steps"]
        missing = [key for key in required_keys if key not in config]
        if missing:
            raise ValueError(
                f"Missing required agent config keys: {', '.join(sorted(missing))}"
            )

        known_keys = {
            "random_seed",
            "memory_size",
            "max_steps",
            "max_rewrite_steps",
            "show_directory_tree",
            "show_current_breakpoints",
            "reset_prompt_history_after_rewrite",
            "system_prompt_template_file",
            "n_rewrites_before_pdb",
            "uuid",
        }
        extras = {k: v for k, v in config.items() if k not in known_keys}

        return cls(
            random_seed=config["random_seed"],
            memory_size=config["memory_size"],
            max_steps=config["max_steps"],
            max_rewrite_steps=config["max_rewrite_steps"],
            show_directory_tree=config.get("show_directory_tree", 0),
            show_current_breakpoints=config.get("show_current_breakpoints", False),
            reset_prompt_history_after_rewrite=config.get(
                "reset_prompt_history_after_rewrite", False
            ),
            system_prompt_template_file=config.get("system_prompt_template_file"),
            n_rewrites_before_pdb=config.get("n_rewrites_before_pdb", 0),
            uuid=config.get("uuid", str(uuid.uuid4())),
            extras=extras,
        )

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
    action_prompt: str = None

    def __init__(
        self,
        args: AgentArgs | Dict[str, Any],
        llm: LLM | None = None,
        logger: DebugGymLogger | None = None,
    ):
        self.args = AgentArgs.from_dict(args) if isinstance(args, dict) else args
        self.logger = logger or DebugGymLogger("debug-gym")
        self.llm = llm
        self._uuid = self.args.uuid
        self.env = None

        self.set_seed(self.args.random_seed)
        self.history = HistoryTracker(self.args.memory_size)

    def set_seed(self, seed):
        np.random.seed(seed)

    def build_history_prompt(self):
        messages = build_history_prompt(
            self.history,
            self.llm,
            self.args.reset_prompt_history_after_rewrite,
        )
        return messages

    def parse_reasoning_model_response(self, response, reasoning_end_token):
        # Strip the reasoning, e.g., in Deepseek r1, between <think> and </think>.
        reasoning_end = response.find(reasoning_end_token)
        if reasoning_end != -1:
            reasoning_end += len(reasoning_end_token)
            response = response[reasoning_end:].strip()
        return response

    def _auto_eval_on_rewrite(self):
        """Check if auto eval on rewrite is enabled."""
        try:
            return self.env.get_tool("eval").auto_eval_on_rewrite
        except KeyError:
            return False  # no eval tool

    def shortcut_features(self):
        features = []
        if self._auto_eval_on_rewrite():
            features.append(
                "After successful rewrites, the environment will automatically "
                "call the Eval tool to evaluate the rewritten code. Therefore, "
                "you do not need to call the Eval tool yourself. The evaluation "
                "output will be updated automatically in the system prompt."
            )
        if self.args.show_directory_tree:
            features.append(
                "The environment will show the directory tree of the repository in the system prompt."
            )
        if self.env.has_tool("pdb"):
            if self.args.show_current_breakpoints:
                features.append(
                    "The environment will show the current breakpoints in the system prompt."
                )
            if self.env.get_tool("pdb").persistent_breakpoints:
                features.append(
                    "The environment will automatically restore existing breakpoints "
                    "when a new PDB session is started (e.g., after a rewrite)."
                )
            if self.env.get_tool("pdb").auto_list:
                features.append(
                    "After every valid PDB tool calling, the environment will "
                    "automatically call the PDB tool again with a `list .` command, "
                    "which will show the code around the current frame."
                )
        return features

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

    def _load_system_prompt_template(self) -> Template | None:
        """Load system prompt template from config if specified and register custom filters.
        If no template is specified, return None.
        """
        system_prompt_template = self.args.system_prompt_template_file
        if system_prompt_template:
            if not os.path.isfile(system_prompt_template):
                error_msg = (
                    f"System prompt template file `{system_prompt_template}` not found."
                )
                self.logger.error(error_msg)
                raise FileNotFoundError(error_msg)
            with open(system_prompt_template, "r") as f:
                system_prompt_template = f.read()
            # Add custom filter to Jinja2 environment
            env = Environment()
            env.filters["to_pretty_json"] = self.to_pretty_json
            env.filters["trim_message"] = self.trim_message
            return env.from_string(system_prompt_template)
        return None

    def _default_system_prompt(self, info) -> str:
        """Return the default system prompt as pretty JSON.
        Trimmed to fit within the token limit."""

        system_prompt_dict = {
            "Overall task": self.system_prompt,
            "Instructions": info.instructions,
        }

        if self.args.show_directory_tree > 0:
            system_prompt_dict["Repo directory tree"] = self.trim_message(
                self.env.workspace.display_files(self.args.show_directory_tree),
                max_length_percentage=0.1,
                where="end",
            )

        if self.args.show_current_breakpoints:
            system_prompt_dict["Current breakpoints"] = info.current_breakpoints

        if self._auto_eval_on_rewrite():
            system_prompt_dict["Evaluation output of current code"] = self.trim_message(
                info.eval_observation.observation,
                max_length_percentage=0.8,
                where="middle",
            )

        shortcut_features = self.shortcut_features()
        if shortcut_features:
            system_prompt_dict["Shortcut features"] = shortcut_features

        return self.to_pretty_json(system_prompt_dict)

    def build_system_prompt(self, info):
        """Build system prompt using jinja template from config or default template."""
        system_prompt_template = self._load_system_prompt_template()
        if system_prompt_template is not None:
            system_prompt = system_prompt_template.render(agent=self, info=info)
        else:
            system_prompt = self._default_system_prompt(info)
        messages = [{"role": "system", "content": filter_non_utf8(system_prompt)}]
        return messages

    def build_question_prompt(self):
        messages = []
        if self.action_prompt is not None:
            messages.append({"role": "user", "content": self.action_prompt})
        return messages

    def build_prompt(self, info):
        messages = []
        messages.extend(self.build_system_prompt(info))
        messages.extend(self.build_history_prompt())
        messages.extend(self.build_question_prompt())
        return messages

    def run(self, env: RepoEnv, debug=False):
        self.env = env
        task_name = self.env.task_name or "custom"  # Used for reporting
        step = 0
        info = None
        max_steps = self.args.max_steps
        try:
            self.history.reset()
            info = self.env.reset()
            # initial state does not have prompt and response
            self.history.step(info, None)

            if info.resolved is True:
                self.logger.report_progress(
                    problem_id=task_name,
                    step=1,
                    total_steps=1,
                    score=info.score,
                    max_score=info.max_score,
                    status="resolved",
                )
                return True

            self.logger.info(
                "Available tools (in LLM's tool calling format):\n"
                f"{json.dumps(self.llm.define_tools(info.tools), indent=4)}\n"
            )

            highscore = info.score
            for step in range(max_steps):
                self.logger.info(f"\n{'='*20} STEP {step+1} {'='*20}\n")
                highscore = max(highscore, info.score)
                msg = f"[{task_name[:10]:<10}] Step {step} | Score: {info.score}/{info.max_score or '-'} [Best: {highscore}]"
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

                if (
                    info.terminated
                    or info.rewrite_counter >= self.args.max_rewrite_steps
                ):
                    reason = (
                        "terminated" if info.resolved else "max_rewrite_steps reached"
                    )
                    self.logger.info(
                        f"Step: {step} | Score: {info.score}/{info.max_score if info.max_score else '-'} | Reason: {reason}"
                    )
                    # early stop, set current step and total steps to be the same
                    self.logger.report_progress(
                        problem_id=task_name,
                        step=step + 1,
                        total_steps=step + 1,
                        score=info.score,
                        max_score=info.max_score,
                        status="resolved" if info.resolved else "unresolved",
                    )
                    break
                # keep progress bar running until max_steps is reached
                self.logger.report_progress(
                    problem_id=task_name,
                    step=step + 1,
                    total_steps=max_steps + 1,
                    score=info.score,
                    max_score=info.max_score,
                    status="running",
                )
            # max_steps was reached, task was either resolved or unresolved
            self.logger.report_progress(
                problem_id=task_name,
                step=step + 1,
                total_steps=step + 1,
                score=info.score,
                max_score=info.max_score,
                status="resolved" if info.resolved else "unresolved",
            )
            return info.resolved
        except Exception:
            # report any error that happens during the run
            self.logger.report_progress(
                problem_id=task_name,
                step=step + 1,
                total_steps=step + 1,
                score=info.score if info else 0,
                max_score=info.max_score,
                status="error",
            )
            raise

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

    def build_trajectory(self, task_name="custom"):
        """Return the trajectory as a JSON-serializable dict without writing it."""
        tools = [f"{tool.name}({tool.arguments})" for tool in self.env.tools]
        json_output = {
            "problem": task_name,
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

    if agent_args is None:
        if config is None:
            raise ValueError("Either agent_args or config must be provided.")
        agent_args = config

    if isinstance(agent_args, dict):
        agent_args = AgentArgs.from_dict(agent_args)

    agent = agent_class(args=agent_args, **agent_kwargs)
    return agent
