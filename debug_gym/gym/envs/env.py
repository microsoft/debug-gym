from dataclasses import dataclass
from pathlib import Path

import numpy as np

from debug_gym.gym.entities import EvalOutput, Event, Observation
from debug_gym.gym.terminals.local import LocalTerminal
from debug_gym.gym.terminals.terminal import Terminal
from debug_gym.gym.tools.tool import EnvironmentTool, ToolCall
from debug_gym.gym.workspace import Workspace
from debug_gym.logger import DebugGymLogger


@dataclass
class EnvInfo:
    # obs from tool triggered by `env.step` or eval if `env.reset`
    step_observation: Observation
    all_observations: list[Observation]  #  env.step + triggered tools obs
    eval_observation: Observation | None  # last eval observation
    current_breakpoints: str
    action_reasoning: str | None
    action_content: str | None
    action_tool_call: ToolCall | None
    instructions: dict
    score: int
    max_score: int
    terminated: bool  # Whether the task has finished running
    resolved: bool  # Whether the task was successfully solved
    rewrite_counter: int
    tools: list[EnvironmentTool]

    def __str__(self) -> str:
        """Pretty print the environment information."""
        lines = []
        lines.append("=" * 70)
        lines.append("DEBUG GYM ENVIRONMENT INFO".center(70))
        lines.append("=" * 70)

        # Status section
        lines.append(
            f"📊 Status: {('✅' if self.resolved else '❌') + ' (TERMINATED)' if self.terminated else '🔄 (IN PROGRESS)'}\t"
            f"🎯 Score: {self.score}/{self.max_score or '?'}\t"
            f"✏️ Rewrites: {self.rewrite_counter}"
        )

        # Action section
        if self.action_tool_call:
            lines.append("🔧 Last Action:")
            lines.append(f"   Tool: {self.action_tool_call.name}")
            if self.action_content:
                lines.append(f"   Explanation: {self.action_content}")
            if self.action_reasoning:
                lines.append(f"   Reasoning: {self.action_reasoning}")
            lines.append("")

        # Observations section
        lines.append(f"👁️ Observation:")
        lines.append(f"```\n{self.step_observation}\n```")
        lines.append("")

        # Tools section
        lines.append(f"🛠️  Available Tools ({len(self.tools)}):")
        # tool_names = [tool.name for tool in self.tools]
        lines.append(f"   {'\n   '.join(map(str, self.tools))}")
        lines.append("")

        # Breakpoints section
        lines.append("🔴 Breakpoints:")
        if self.current_breakpoints == "No breakpoints are set.":
            lines.append("   None set")
        else:
            for bp in self.current_breakpoints.split("\n")[:5]:  # Show first 5
                lines.append(f"   • {bp}")
            if len(self.current_breakpoints.split("\n")) > 5:
                lines.append(
                    f"   ... and {len(self.current_breakpoints.split('\n')) - 5} more"
                )
        lines.append("")

        return "\n".join(lines)


class EventHooks:
    def __init__(self):
        self.event_listeners = {event: [] for event in Event}

    def subscribe(self, event: Event, tool: "Tool"):
        if event not in self.event_listeners:
            raise ValueError(f"Unknown event type: {event}")
        if not hasattr(tool, event.handler_name):
            raise ValueError(f"Tool does not implement method {event.handler_name}")
        if tool in self.event_listeners[event]:
            raise ValueError(f"Tool already subscribed to event: {event}")
        self.event_listeners[event].append(tool)

    def unsubscribe(self, event: Event, tool):
        self.event_listeners[event].remove(tool)

    def notify(
        self, environment, event: Event, source=None, **kwargs
    ) -> list[Observation]:
        """Notify all tools that are subscribed to the event.
        Returns a list of observations from all tools that are triggered by the event.
        If error occurs while handling the event, an error observation is returned.
        """
        observations = []
        for tool in self.event_listeners[event]:
            if tool == source:
                continue  # skip the source tool to avoid infinite loop
            try:
                observation = getattr(tool, event.handler_name)(environment, **kwargs)
                if observation:
                    observations.append(observation)
            except Exception as e:
                error_message = f"Error in tool {tool.name} handling {event}:\n{e}"
                observations.append(Observation(tool.name, error_message))
        return observations


class TooledEnv:
    def __init__(self):
        self._tools = {}
        self.event_hooks = EventHooks()
        self.event_queue = []
        self.all_observations = []

    @property
    def tool_names(self):
        return ", ".join([t.name for t in self._tools.values()])

    def add_tool(self, tool):
        if tool.name in self._tools:
            raise ValueError(f"Tool {tool.name} already exists!")

        self._tools[tool.name] = tool
        tool.register(self)

    def has_tool(self, tool_name):
        return tool_name in self._tools

    def get_tool(self, tool_name):
        return self._tools[tool_name]

    def remove_tool(self, tool_name):
        if tool_name not in self._tools:
            raise ValueError(f"Tool {tool_name} not found!")
        removed_tool = self._tools.pop(tool_name)
        removed_tool.unregister(self)  # Unsubscribe from all events
        return removed_tool

    def get_triggered_tools(self, action: ToolCall):
        try:
            tool_name = action.name
            tool_kwargs = action.arguments
        except Exception as e:
            # parse error
            return str(e), None
        if tool_name == "empty_tool_response":
            # the agent did not generate any tool call
            return (
                "No tool call was generated by the agent. This may be due to the reasoning being too long and exceeding the context length. Make sure to keep the reasoning concise.",
                None,
            )
        if tool_name not in self._tools:
            # failed to find tool
            return f"Unregistered tool: {tool_name}", None
        tool = self._tools[tool_name]
        return None, [tool, tool_kwargs]

    @property
    def tools(self):
        return list(self._tools.values())

    def clear_all_observations(self):
        self.all_observations = []

    def empty_event_queue(self):
        self.event_queue = []

    def queue_event(self, event: Event, source=None, **kwargs) -> None:
        """Add an event to the queue for processing later."""
        self.event_queue.append((event, source, kwargs))

    def process_events(self) -> list[Observation]:
        """Process all queued events and handle their observations."""
        while self.event_queue:
            event, source, kwargs = self.event_queue.pop(0)
            observations = self.event_hooks.notify(
                environment=self, event=event, source=source, **kwargs
            )
            self.all_observations.extend(observations)
            self.post_process_event(event, source, kwargs, observations)
        return self.all_observations

    def post_process_event(self, event: Event, source, kwargs, observations):
        """Post-process the event after it has been handled by the tools."""
        pass


class RepoEnv(TooledEnv):

    def __init__(
        self,
        path: str | None = None,
        entrypoint: str = "python -m pytest -sq .",
        debug_entrypoint: str | None = None,
        max_score: int | None = None,
        readonly_patterns: list[str] | None = None,  # TODO: remove
        run_timeout: int | None = None,
        terminal: Terminal | None = None,
        logger: DebugGymLogger | None = None,
        problems: str | list[str] | None = None,
        **kwargs,
    ):
        super().__init__()

        self.path = path
        self.max_score = max_score
        self.run_timeout = run_timeout
        self.terminal = terminal or LocalTerminal()  # TODO: default to DockerTerminal
        self._entrypoint = entrypoint
        self._debug_entrypoint = debug_entrypoint
        self.logger = logger or DebugGymLogger("debug-gym")
        self.infos: EnvInfo | None = None
        self.rng = None
        self.additional_kwargs = kwargs
        self.task_name: str | None = None
        self.options: dict = {}

        if "auto_eval_on_rewrite" in kwargs:
            raise ValueError(
                "The 'auto_eval_on_rewrite' parameter is no longer supported. "
                "Please remove it from your initialization arguments."
                "Instead, set 'auto_eval_on_rewrite' in the EvalTool instance."
            )

        self.workspace = Workspace(self.terminal, logger=self.logger)
        self.dataset = self.load_dataset(problems)
        self.set_entrypoints(self._entrypoint, self._debug_entrypoint)

    def _reset_env_state(self):
        """Reset the environment state to the initial state."""
        # reset all state variables
        self.current_breakpoints_state = {}
        self.rewrite_counter = 0
        self.last_eval: EvalOutput = None
        self.score = 0
        self.terminated = False
        self.resolved = False
        # clear all observations and event queue (queue should be empty already)
        self.clear_all_observations()
        self.empty_event_queue()

    def set_entrypoints(self, entrypoint: str, debug_entrypoint: str | None = None):
        self.entrypoint = self._prepare_entrypoint(entrypoint)
        self.debug_entrypoint = self._prepare_entrypoint(debug_entrypoint or entrypoint)

        if "python -m pdb" not in self.debug_entrypoint:
            self.debug_entrypoint = self.debug_entrypoint.replace(
                "python ", "python -m pdb "
            )

    @staticmethod
    def _prepare_entrypoint(entrypoint):
        entrypoint_list = entrypoint.split()
        # Handle uv package manager's run command by ensuring the correct interpreter path
        # and explicitly adding 'python' to the execution chain for consistency.
        if entrypoint_list[0].endswith("uv") and entrypoint_list[1] == "run":
            entrypoint_list[2] = f"$(which {entrypoint_list[2]})"
            entrypoint_list = entrypoint_list[:2] + ["python"] + entrypoint_list[2:]

        elif "xvfb" in entrypoint:
            # parse "xvfb-run --auto-servernum .venv/bin/python -W ignore -m pytest -rA r2e_tests"
            return entrypoint

        # For non-python commands, ensure we have the absolute path to the Python executable
        # and explicitly run it through Python for consistent execution behavior.
        elif entrypoint_list[0] != "python":
            entrypoint_list[0] = f"$(which {entrypoint_list[0]})"
            entrypoint_list = ["python"] + entrypoint_list

        entrypoint = " ".join(entrypoint_list)
        return entrypoint

    @property
    def working_dir(self) -> Path:
        return self.workspace.working_dir

    @property
    def instructions(self) -> str:
        """Instructions for the current task.
        Override in subclasses for different behavior."""
        return ""

    def setup_task(self, options: dict = None) -> None:
        """Setup the task information.
        Override in subclasses for different behavior. Called once at reset."""
        pass

    def setup_workspace(self) -> None:
        """Setup the workspace.
        Override in subclasses for different behavior. Called once at reset."""
        self.workspace.reset()
        self.workspace.copy_content(self.path)
        self.workspace.setup_file_filters()

    def setup_terminal(self) -> None:
        """Setup the terminal.
        Override in subclasses for different behavior. Called once at reset."""

        self.logger.debug(f"Configuring {self.terminal}...")

        self.terminal.run("git init -b main")
        self.terminal.run("git config user.name 'debug-gym'")
        self.terminal.run("git config user.email '<>'")

        self.terminal.run("git add *")
        self.terminal.run("git commit -am 'Init'")

        self.terminal.run("git add .debugignore .debugreadonly")
        self.terminal.run("git commit -am 'Add debug-gym ignore and read-only files'")

    def reset(self, *, options: dict = None):
        """Resets the environment and returns eval as the initial observation."""
        self.options = options if options is not None else self.options
        self.logger.debug("Resetting environment")
        self.close()  # Clean up previous workspace and terminal.
        self.setup_task(options=self.options)
        self.setup_workspace()
        self.setup_terminal()
        self._reset_env_state()

        # Notify all tools that the environment is reset and get their observations
        self.queue_event(Event.ENV_RESET, source="env")
        self.all_observations = self.process_events()

        # First observation always include the task instructions.
        self.step_observation = Observation("env", self.instructions)
        self.all_observations.insert(0, self.step_observation)

        if self.last_eval:
            self.max_score = self.calculate_max_score(self.last_eval)
            self.score = self.calculate_score(self.last_eval)
            self.resolved = self.calculate_resolved(self.last_eval)
            self.terminated = self.calculate_terminated(self.last_eval)

        self.infos = EnvInfo(
            step_observation=self.step_observation,
            all_observations=self.all_observations,
            eval_observation=(
                Observation("env", self.last_eval.output) if self.last_eval else None
            ),
            current_breakpoints=self.current_breakpoints(),
            action_reasoning=None,
            action_content=None,
            action_tool_call=None,
            terminated=self.terminated,
            resolved=self.resolved,
            score=self.score,
            max_score=self.max_score,
            instructions=self.instructions,
            rewrite_counter=self.rewrite_counter,
            tools=self.tools,
        )
        return self.infos

    def seed(self, seed=None):
        if seed is not None:
            self.rng = np.random.RandomState(seed)

    def calculate_max_score(self, eval_output: EvalOutput) -> int:
        """Calculate the maximum score. Called once at reset.
        Override in subclasses for different behavior."""
        return self.max_score

    def calculate_score(self, eval_output: EvalOutput) -> int:
        """Calculate the score from the eval output.
        Override in subclasses for different behavior."""
        return eval_output.success

    def calculate_resolved(self, eval_output: EvalOutput) -> bool:
        """Determine if the task has been resolved.
        Override in subclasses for different behavior."""
        return self.score == self.max_score

    def calculate_terminated(self, eval_output: EvalOutput) -> bool:
        """Determine if the task is terminated.
        Override in subclasses for different behavior."""
        return self.calculate_resolved(eval_output)

    def eval(self, **kwargs) -> EvalOutput:
        """Evaluates the current code using the provided entrypoint.
        Sets the last_eval and returns it.
        Override in subclasses for different behavior."""
        success, output = self.terminal.run(self.entrypoint, timeout=self.run_timeout)
        self.last_eval = EvalOutput(success, output)
        return self.last_eval

    def has_breakpoint(self, file_path: str, line_number: int) -> bool:
        """Check if a breakpoint is set at the given file and line number."""
        key = f"{self.workspace.resolve_path(file_path)}|||{line_number}"
        return key in self.current_breakpoints_state

    def current_breakpoints(self):
        if len(self.current_breakpoints_state) == 0:
            return "No breakpoints are set."
        else:
            # print the breakpoints sorted by file names and line number
            breakpoints = []
            for _key in self.current_breakpoints_state.keys():
                _file_path, _line_number = _key.split("|||")
                _line_number = int(_line_number)
                breakpoints.append([_file_path, _line_number])
            # sort by file name, if file names are same, sort by line number
            breakpoints = sorted(breakpoints, key=lambda x: (x[0], x[1]))
            breakpoints = [
                f"line {_line_number} in {_file_path}"
                for _file_path, _line_number in breakpoints
            ]
            return "\n".join(breakpoints)

    @property
    def patch(self):
        success, output = self.terminal.run("git diff", strip_output=False, raises=True)
        return output

    def apply_gold_patch(self):
        raise NotImplementedError(
            f"apply_gold_patch is not implemented for {self.__class__.__name__}."
        )

    def step(
        self,
        action_tool_call: ToolCall,
        action_content: str | None = None,
        action_reasoning: str | None = None,
    ) -> EnvInfo:
        # given action, return new obs, and update infos
        # the action space is composed of a few smaller action spaces
        self.clear_all_observations()
        self.empty_event_queue()
        message, tool_info = self.get_triggered_tools(action_tool_call)
        if message:
            self.step_observation = Observation("env", message)
        else:
            triggered_tool, tool_kwargs = tool_info
            try:
                # tool_kwargs is a dict, so we need to unpack it
                self.step_observation = triggered_tool(self, **tool_kwargs)
            except KeyboardInterrupt:
                self.logger.error("Step was interrupted by user.")
                raise
            except BaseException as e:
                error_message = (
                    f"Error while using tool {triggered_tool.name} "
                    f"with action: {action_tool_call}.\n{e}"
                )
                self.step_observation = Observation("env", error_message)
                self.logger.debug(error_message)

        # Process any events that were queued during tool execution
        self.all_observations = self.process_events()
        # prepend step_observation to all_observations
        self.all_observations.insert(0, self.step_observation)

        # Calculate score and done based on the last eval output
        if self.last_eval:
            self.max_score = self.calculate_max_score(self.last_eval)
            self.score = self.calculate_score(self.last_eval)
            self.terminated = self.calculate_terminated(self.last_eval)
            self.resolved = self.calculate_resolved(self.last_eval)

        self.infos = EnvInfo(
            step_observation=self.step_observation,
            all_observations=self.all_observations,
            eval_observation=(
                Observation("env", self.last_eval.output) if self.last_eval else None
            ),
            current_breakpoints=self.current_breakpoints(),
            action_reasoning=action_reasoning,
            action_content=action_content,
            action_tool_call=action_tool_call,
            instructions=self.instructions,
            score=self.score,
            max_score=self.max_score,
            terminated=self.terminated,
            resolved=self.resolved,
            rewrite_counter=self.rewrite_counter,
            tools=self.tools,
        )

        return self.infos

    def post_process_event(self, event: Event, source, kwargs, observations):
        """Post-process the event after it has been handled by the tools."""
        if event in (Event.REWRITE_SUCCESS, Event.REWRITE_FAIL):
            self.rewrite_counter += 1

    def close(self):
        self.workspace.cleanup()
        if self.terminal:
            self.terminal.close()

    def __del__(self):
        self.close()

    def load_dataset(self, problems: str | list[str] | None = None):
        return {"custom": None}
