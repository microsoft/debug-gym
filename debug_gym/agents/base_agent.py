import json
import os
import subprocess
import uuid
from os.path import join as pjoin

import numpy as np
from jinja2 import Environment, Template

from debug_gym.agents.history_tracker import HistoryTracker, build_history_prompt
from debug_gym.agents.utils import trim
from debug_gym.gym.envs.env import RepoEnv
from debug_gym.gym.utils import filter_non_utf8
from debug_gym.llms.base import LLM
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
        self.llm = LLM.instantiate(
            llm_name=self.config["llm_name"],
            llm_config_file_path=self.config.get("llm_config_file_path"),
            logger=self.logger,
        )
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
            self.llm,
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

    def _auto_eval_on_rewrite(self):
        """Check if auto eval on rewrite is enabled."""
        return self.config.get("env_kwargs", {}).get("auto_eval_on_rewrite", False)

    def _show_current_breakpoints(self):
        """Check if current breakpoints should be shown in the system prompt."""
        return self.config.get("env_kwargs", {}).get("show_current_breakpoints", False)

    def _show_directory_tree(self):
        """Check if directory tree should be shown in the system prompt."""
        return self.config.get("env_kwargs", {}).get("show_directory_tree", False)

    def shortcut_features(self):
        features = []
        if self._auto_eval_on_rewrite():
            features.append(
                "After successful rewrites, the environment will automatically "
                "call the Eval tool to evaluate the rewritten code. Therefore, "
                "you do not need to call the Eval tool yourself. The evaluation "
                "output will be updated automatically in the system prompt."
            )
        if self._show_directory_tree():
            features.append(
                "The environment will show the directory tree of the repository in the system prompt."
            )
        if self.env.has_tool("pdb"):
            if self._show_current_breakpoints():
                features.append(
                    "The environment will show the current breakpoints in the system prompt."
                )
            if self.config.get("env_kwargs", {}).get("persistent_breakpoints"):
                features.append(
                    "The environment will automatically restore existing breakpoints "
                    "when a new PDB session is started (e.g., after a rewrite)."
                )
            if self.config.get("env_kwargs", {}).get("auto_list"):
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
        message,
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
        tokens = count_tokens(message)
        if tokens > max_length:
            return trim(message, max_length, count_tokens=count_tokens, where=where)
        return message

    def _load_system_prompt_template(self) -> Template | None:
        """Load system prompt template from config if specified and register custom filters.
        If no template is specified, return None.
        """
        system_prompt_template = self.config.get("system_prompt_template_file")
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

        if self._show_directory_tree():
            system_prompt_dict["Repo directory tree"] = self.trim_message(
                info.dir_tree, max_length_percentage=0.1, where="end"
            )

        if self._show_current_breakpoints():
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

    def run(self, task_name=None, debug=False):
        self.history.reset()
        info = self.env.reset(options={"task_name": task_name})
        # initial state does not have prompt and response
        self.history.step(info, None)

        if info.done is True:
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
        max_steps = self.config["max_steps"]
        for step in range(max_steps):
            self.logger.info(f"\n{'='*20} STEP {step+1} {'='*20}\n")
            highscore = max(highscore, info.score)
            self.logger.info(
                f"[{task_name[:10]:<10}] | Step: {step:<4} | Score: {info.score:>4}/{info.max_score:<4} ({info.score/info.max_score:.1%}) [Best: {highscore}]"
            )

            messages = self.build_prompt(info)
            llm_response = self.llm(messages, info.tools)

            if debug:
                breakpoint()

            info = self.env.step(llm_response.tool, llm_response.response)
            self.history.step(info, llm_response)

            if info.done or info.rewrite_counter >= self.config["max_rewrite_steps"]:
                reason = "done" if info.done else "max_rewrite_steps reached"
                self.logger.info(
                    f"Step: {step} | Score: {info.score}/{info.max_score} ({info.score/info.max_score:.1%}) | Reason: {reason}"
                )
                self.logger.report_progress(
                    problem_id=task_name,
                    step=step + 1,
                    total_steps=step + 1,  # early stop, current step is total steps
                    score=info.score,
                    max_score=info.max_score,
                    status="resolved" if info.done else "unresolved",
                )
                break
            self.logger.report_progress(
                problem_id=task_name,
                step=step + 1,
                total_steps=max_steps + 1,  # keep progress bar running until max_steps
                score=info.score,
                max_score=info.max_score,
                status="running",
            )
        self.logger.report_progress(
            problem_id=task_name,
            step=step + 1,
            total_steps=step + 1,
            score=info.score,
            max_score=info.max_score,
            status="resolved" if info.done else "unresolved",
        )
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
            "tools": self.llm.define_tools(self.env.tools),
            "uuid": self._uuid,
            "success": self.env.done,
            "log": [],
            "agent_type": self.__class__.__name__,
            "logger": str(self.logger.log_file),
        }
        for step_id in range(len(self.history)):
            step_json = self.history.json(step_id)
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
