import atexit
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from glob import glob
from pathlib import Path

import numpy as np

from debug_gym.gym.entities import EvalOutput, Event, Observation
from debug_gym.gym.terminal import Terminal
from debug_gym.gym.tools.tool import EnvironmentTool, ToolCall
from debug_gym.gym.utils import _walk, make_file_matcher
from debug_gym.logger import DebugGymLogger


@dataclass
class EnvInfo:
    # obs from tool triggered by `env.step` or eval if `env.reset`
    step_observation: Observation
    all_observations: list[Observation]  #  env.step + triggered tools obs
    eval_observation: Observation  # last eval observation
    dir_tree: str
    current_breakpoints: str
    action_reasoning: str | None
    action: ToolCall | None
    instructions: dict
    score: int
    max_score: int
    done: bool
    rewrite_counter: int
    tools: list[EnvironmentTool]


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
        readonly_patterns: list[str] | None = None,
        auto_eval_on_rewrite: bool = True,
        run_timeout: int | None = None,
        dir_tree_depth: int = 1,
        persistent_breakpoints: bool = True,
        auto_list: bool = True,
        terminal: Terminal | None = None,
        logger: DebugGymLogger | None = None,
        problems: str | list[str] | None = None,
        **kwargs,
    ):
        super().__init__()

        self.path = None
        self.max_score = max_score
        self.auto_eval_on_rewrite = auto_eval_on_rewrite
        self.run_timeout = run_timeout
        self.dir_tree_depth = dir_tree_depth
        self.terminal = terminal or Terminal()
        self.entrypoint = entrypoint
        self.debug_entrypoint = debug_entrypoint or entrypoint
        self.persistent_breakpoints = persistent_breakpoints
        self.auto_list = auto_list
        self.logger = logger or DebugGymLogger("debug-gym")
        self.infos: EnvInfo | None = None
        self.rng = None
        self._tempdir = None
        self.additional_kwargs = kwargs

        self.setup_workspace(
            path=path,
            entrypoint=entrypoint,
            debug_entrypoint=debug_entrypoint,
            readonly_patterns=readonly_patterns,
        )
        self._reset_env_state()
        self.dataset = self.load_dataset(problems)

    def _reset_env_state(self):
        """Reset the environment state to the initial state."""
        # reset all state variables
        self.current_breakpoints_state = {}
        self.rewrite_counter = 0
        self.last_eval: EvalOutput = None
        self.score = 0
        self.done = False
        # clear all observations and event queue (queue should be empty already)
        self.clear_all_observations()
        self.empty_event_queue()

    def setup_workspace(
        self,
        path: str,
        entrypoint: str | None = None,
        debug_entrypoint: str | None = None,
        readonly_patterns: list[str] | None = None,
        ignore_patterns: list[str] | None = None,
    ):
        readonly_patterns = readonly_patterns or []
        ignore_patterns = ignore_patterns or []
        if self.path:
            self.cleanup_workspace()
            self.path = None

        if path is None:
            return

        self.path = Path(path)

        # Create a random temporary folder for storing a backup of the repo.
        self._tempdir = tempfile.TemporaryDirectory(prefix="RepoEnv-")
        self.working_dir = Path(self._tempdir.name).resolve()

        # Make sure to cleanup that folder once done.
        atexit.register(self._tempdir.cleanup)

        self.logger.debug(f"Working directory: {self.working_dir}")
        shutil.copytree(
            self.path,
            self.working_dir,
            dirs_exist_ok=True,
            symlinks=True,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )

        self.setup_file_filters(readonly_patterns, ignore_patterns)

        # override entrypoint as it might be task dependent
        self.set_entrypoints(entrypoint, debug_entrypoint)

        # Set up the terminal working dir
        self.terminal.working_dir = str(self.working_dir)
        self._reset_env_state()

    def set_entrypoints(self, entrypoint, debug_entrypoint):
        if entrypoint:
            self.entrypoint = self._prepare_entrypoint(entrypoint)
            debug_entrypoint = debug_entrypoint or entrypoint.replace(
                "python", "python -m pdb"
            )
            self.debug_entrypoint = self._prepare_entrypoint(debug_entrypoint)
        if self.debug_entrypoint is not None and "-m pdb" not in self.debug_entrypoint:
            self.debug_entrypoint = self.debug_entrypoint.replace(
                "python", "python -m pdb"
            )
        self.entrypoint = "PYTHONPATH=$PYTHONPATH:$PWD " + self.entrypoint
        self.debug_entrypoint = "PYTHONPATH=$PYTHONPATH:$PWD " + self.debug_entrypoint

    @staticmethod
    def _prepare_entrypoint(entrypoint):
        entrypoint_list = entrypoint.split()
        # Handle uv package manager's run command by ensuring the correct interpreter path
        # and explicitly adding 'python' to the execution chain for consistency.
        if entrypoint_list[0].endswith("uv") and entrypoint_list[1] == "run":
            entrypoint_list[2] = f"$(which {entrypoint_list[2]})"
            entrypoint_list = entrypoint_list[:2] + ["python"] + entrypoint_list[2:]

        # For non-python commands, ensure we have the absolute path to the Python executable
        # and explicitly run it through Python for consistent execution behavior.
        elif entrypoint_list[0] != "python":
            entrypoint_list[0] = f"$(which {entrypoint_list[0]})"
            entrypoint_list = ["python"] + entrypoint_list

        entrypoint = " ".join(entrypoint_list)
        return entrypoint

    def cleanup_workspace(self):
        if self._tempdir:
            self._tempdir.cleanup()

    @property
    def instructions(self) -> str:
        return ""

    def display_files(self):
        msg = (
            "Listing files in the current working directory."
            " (read-only) indicates read-only files."
            f" Max depth: {str(self.dir_tree_depth)}.\n"
        )
        msg += self.directory_tree()
        return msg

    def restore(self, *filepaths):
        filepaths = filepaths or glob(
            f"{self.path}/**",
            root_dir=self.path,
            recursive=True,
        )
        relative_filepaths = [os.path.relpath(f, self.path) for f in filepaths]
        for filepath in relative_filepaths:
            if os.path.isdir(self.path / filepath):
                os.makedirs(self.working_dir / filepath, exist_ok=True)
                continue

            shutil.copy2(self.path / filepath, self.working_dir / filepath)

    def reset(self, *, options: dict = None):
        """Resets the environment and returns eval as the initial observation."""
        self.logger.info("Resetting environment")
        options = options or {}

        self._reset_env_state()

        # Notify all tools that the environment is reset and get their observations
        self.queue_event(Event.ENV_RESET, source="env")
        self.all_observations = self.process_events()

        # Gets eval (initial observation) from cache or by running env.eval
        if self.last_eval:  # if eval tool was triggered by Event.ENV_RESET
            self.step_observation = Observation("env", self.last_eval.output)
        else:  # if eval tool was not triggered by Event.ENV_RESET
            self.last_eval = self.eval()
            self.step_observation = Observation("env", self.last_eval.output)
            self.all_observations.insert(0, self.step_observation)

        self.max_score = self.calculate_max_score(self.last_eval)
        self.score = self.calculate_score(self.last_eval)
        self.done = self.calculate_done(self.last_eval)

        self.infos = EnvInfo(
            step_observation=self.step_observation,
            all_observations=self.all_observations,
            eval_observation=Observation("env", self.last_eval.output),
            dir_tree=self.display_files(),
            current_breakpoints=self.current_breakpoints(),
            action_reasoning=None,
            action=None,
            done=self.done,
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
        # Default to 1 (eval) if max_score is not set
        return self.max_score or 1

    def calculate_score(self, eval_output: EvalOutput) -> int:
        """Calculate the score from the eval output.
        Override in subclasses for different behavior."""
        return eval_output.success

    def calculate_done(self, eval_output: EvalOutput) -> bool:
        """Determine if the task is done.
        Override in subclasses for different behavior."""
        return self.score == self.max_score

    def eval(self, **kwargs) -> EvalOutput:
        """Evaluates the current code using the provided entrypoint.
        Sets the last_eval and returns it.
        Override in subclasses for different behavior."""
        success, output = self.terminal.run(self.entrypoint, timeout=self.run_timeout)
        self.last_eval = EvalOutput(success, output)
        return self.last_eval

    def resolve_path(self, filepath: str | Path, raises=False) -> Path:
        """Convert a relative filepath to absolute based on the working_dir.
        If the path is already absolute, it is returned as is.
        If raises is True, raises FileNotFoundError if the file does not exist,
        is not in the working directory or is ignored by the ignore patterns.
        If raises is False, returns the absolute path regardless of the file existence.
        """
        abs_filepath = Path(filepath)
        if not abs_filepath.is_absolute():
            abs_filepath = (Path(self.working_dir) / abs_filepath).resolve()
        if (
            raises
            and abs_filepath != self.working_dir
            and not (
                abs_filepath.is_relative_to(self.working_dir)
                and abs_filepath.exists()
                and not self._is_ignored_func(abs_filepath)
            )
        ):
            # raises error with original path
            raise FileNotFoundError(
                f"`{filepath}` does not exist or is not in "
                f"the working directory `{self.working_dir}`."
            )
        return abs_filepath

    def has_file(self, filepath: str) -> bool:
        """Checks if a file exists in the working directory.
        Shortcut for `resolve_path` with raises=True.
        """
        try:
            self.resolve_path(filepath, raises=True)
            return True
        except FileNotFoundError:
            return False

    def read_file(self, filepath: str) -> str:
        """Reads a file from the working directory.
        Raises value error if the file does not exist"""
        abs_filepath = self.resolve_path(filepath, raises=True)
        return abs_filepath.read_text()

    def is_editable(self, filepath):
        return not self._is_readonly_func(self.resolve_path(filepath, raises=True))

    def setup_file_filters(
        self,
        readonly_patterns: list[str] | None = None,
        ignore_patterns: list[str] | None = None,
    ):
        """Indexes files and subdir in the working
        directory, applying ignore and readonly patterns."""
        readonly_patterns = readonly_patterns or []
        ignore_patterns = ignore_patterns or []

        # Ignore debug gym hidden files
        ignore_patterns += [".debugignore", ".debugreadonly"]

        # create a matcher function for ignored files, .debugignore has precedence over .gitignore
        self._is_ignored_func = make_file_matcher(
            base_dir=self.working_dir,
            pattern_files=[
                self.resolve_path(".gitignore"),
                self.resolve_path(".debugignore"),
            ],
            patterns=ignore_patterns,
        )

        # create a matcher function for readonly files
        self._is_readonly_func = make_file_matcher(
            base_dir=self.working_dir,
            pattern_files=self.resolve_path(".debugreadonly"),
            patterns=readonly_patterns,
        )

    def directory_tree(self, root: str | Path = None, max_depth: int | None = None):
        root = self.resolve_path(root or self.working_dir, raises=True)
        max_depth = max_depth or self.dir_tree_depth

        # initalize with root directory
        result = [f"{root}/"]

        # get all paths with correct depth
        for path in _walk(root, max_depth, skip=self._is_ignored_func):
            rel_path = path.relative_to(root)  # relative path from root
            depth = len(rel_path.parts) - 1  # depth of current path
            indent = "  " * depth  # 2 spaces per level for indent

            # file vs direcrory formatting
            result.append(f"{indent}|-- {path.name}")

            if path.is_dir():
                result[-1] += "/"

            if not self.is_editable(path):
                result[-1] += " (read-only)"

        return "\n".join(result)

    def has_breakpoint(self, file_path: str, line_number: int) -> bool:
        """Check if a breakpoint is set at the given file and line number."""
        key = f"{self.resolve_path(file_path)}|||{line_number}"
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
        command = ["git", "diff", "--no-index", self.path, self.working_dir]
        result = subprocess.run(command, text=True, capture_output=True)
        patch = result.stdout.replace(str(self.working_dir), str(self.path))
        return patch

    def apply_gold_patch(self):
        raise NotImplementedError(
            f"apply_gold_patch is not implemented for {self.__class__.__name__}."
        )

    def step(self, action: ToolCall, action_reasoning: str = "") -> EnvInfo:
        # given action, return new obs, and update infos
        # the action space is composed of a few smaller action spaces
        self.clear_all_observations()
        self.empty_event_queue()
        message, tool_info = self.get_triggered_tools(action)
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
                    f"with action: {action}.\n{e}"
                )
                self.step_observation = Observation("env", error_message)
                self.logger.debug(error_message)

        # Process any events that were queued during tool execution
        self.all_observations = self.process_events()
        # prepend step_observation to all_observations
        self.all_observations.insert(0, self.step_observation)

        # Calculate score and done based on the last eval output
        self.score = self.calculate_score(self.last_eval)
        self.done = self.calculate_done(self.last_eval)

        self.infos = EnvInfo(
            step_observation=self.step_observation,
            all_observations=self.all_observations,
            eval_observation=Observation("env", self.last_eval.output),
            dir_tree=self.display_files(),
            current_breakpoints=self.current_breakpoints(),
            action_reasoning=action_reasoning,
            action=action,
            instructions=self.instructions,
            score=self.score,
            max_score=self.max_score,
            done=self.done,
            rewrite_counter=self.rewrite_counter,
            tools=self.tools,
        )

        return self.infos

    def post_process_event(self, event: Event, source, kwargs, observations):
        """Post-process the event after it has been handled by the tools."""
        if event in (Event.REWRITE_SUCCESS, Event.REWRITE_FAIL):
            self.rewrite_counter += 1

    def close(self):
        self.cleanup_workspace()
        if self.terminal:
            self.terminal.close()

    def load_dataset(self, problems: str | list[str] | None = None):
        return {"custom": None}
