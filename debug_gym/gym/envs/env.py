import atexit
import glob
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from glob import glob
from os.path import join as pjoin
from pathlib import Path

import numpy as np

from debug_gym.gym.entities import EvalOutput, Event, Observation
from debug_gym.gym.terminal import Terminal
from debug_gym.gym.utils import _walk, make_file_matcher, parse_action, show_line_number
from debug_gym.logger import DebugGymLogger


@dataclass
class EnvInfo:
    # obs from tool triggered by `env.step` or eval if `env.reset`
    step_observation: Observation
    all_observations: list[Observation]  #  env.step + triggered tools obs
    eval_observation: Observation  # last eval observation
    dir_tree: str
    current_code_with_line_number: dict | str
    current_breakpoints: str
    action: str
    instructions: dict
    score: int
    max_score: int
    done: bool
    rewrite_counter: int
    tools: dict  # TODO: return some tool dataclass


class EventHooks:
    def __init__(self):
        self.event_listeners = {event: [] for event in Event}

    def subscribe(self, event: Event, tool: "Tool"):
        if event not in self.event_listeners:
            raise ValueError(f"Unknown event type: {event}")
        if not hasattr(tool, event.handler_name):
            raise ValueError(f"Tool does not implement method {event.handler_name}")
        self.event_listeners[event].append(tool)

    def unsubscribe(self, event: Event, tool):
        self.event_listeners[event].remove(tool)

    def notify(self, event: Event, source=None, **kwargs) -> list[Observation]:
        """Notify all tools that are subscribed to the event.
        Returns a list of observations from all tools that are triggered by the event.
        If error occurs while handling the event, an error observation is returned.
        """
        observations = []
        for tool in self.event_listeners[event]:
            if tool == source:
                continue  # skip the source tool to avoid infinite loop
            try:
                observation = getattr(tool, event.handler_name)(**kwargs)
                if observation:
                    observations.append(observation)
            except Exception as e:
                error_message = f"Error in tool {tool.name} handling {event}:\n{e}"
                observations.append(Observation(tool.name, error_message))
        return observations


class TooledEnv:
    def __init__(self):
        self.tools = {}
        self.event_hooks = EventHooks()
        self.event_queue = []
        self.all_observations = []

    @property
    def tool_names(self):
        return ", ".join([f"```{t.name}```" for t in self.tools.values()])

    def add_tool(self, tool):
        if tool.name in self.tools:
            raise ValueError(f"Tool {tool.name} already exists!")

        self.tools[tool.name] = tool
        tool.register(self)

    def has_tool(self, tool_name):
        return tool_name in self.tools

    def get_tool(self, tool_name):
        return self.tools[tool_name]

    def get_triggered_tools(self, action):
        try:
            tool_name, tool_args = parse_action(action)
        except Exception as e:
            # parse error
            return str(e), None
        if tool_name not in self.tools:
            # failed to find tool
            return f"Unregistered tool: {tool_name}", None
        tool = self.tools[tool_name]
        return None, [tool, tool_args]

    @property
    def tool_instructions(self):
        return {name: tool.instructions for name, tool in self.tools.items()}

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
            observations = self.event_hooks.notify(event=event, source=source, **kwargs)
            self.all_observations.extend(observations)
        return self.all_observations


class RepoEnv(TooledEnv):

    def __init__(
        self,
        path: str | None = None,
        entrypoint: str = "python -m pytest -sq .",
        debug_entrypoint: str | None = None,
        max_score: int | None = None,
        readonly_patterns: list[str] | None = None,
        run_on_rewrite: bool = True,
        run_timeout: int | None = None,
        dir_tree_depth: int | None = None,
        auto_view_change: bool = True,
        terminal: Terminal | None = None,
        logger: DebugGymLogger | None = None,
    ):
        super().__init__()

        self.path = None
        self.max_score = max_score
        self.run_on_rewrite = run_on_rewrite
        self.run_timeout = run_timeout
        self.dir_tree_depth = dir_tree_depth
        self.auto_view_change = auto_view_change
        self.terminal = terminal or Terminal()
        self.entrypoint = entrypoint
        self.debug_entrypoint = debug_entrypoint or entrypoint
        self.logger = logger or DebugGymLogger("debug-gym")
        self.infos: EnvInfo | None = None
        self.rng = None
        self.tempdir = None

        self.setup_workspace(
            path=path,
            entrypoint=entrypoint,
            debug_entrypoint=debug_entrypoint,
            readonly_patterns=readonly_patterns,
        )
        self._reset_env_state()

    def _reset_env_state(self):
        """Reset the environment state to the initial state."""
        # reset all state variables
        self.current_file = None
        self.current_file_content = None
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
    ):
        readonly_patterns = readonly_patterns or []
        if self.path:
            self.cleanup_workspace()
            self.path = None

        if path is None:
            return

        self.path = Path(path)

        # Create a random temporary folder for storing a backup of the repo.
        self.tempdir = tempfile.TemporaryDirectory(prefix="RepoEnv-")
        self.working_dir = Path(self.tempdir.name)
        # Make sure to cleanup that folder once done.
        atexit.register(self.tempdir.cleanup)

        self.logger.debug(f"Working directory: {self.working_dir}")
        shutil.copytree(self.path, self.working_dir, dirs_exist_ok=True, symlinks=True)

        self._index_files(readonly_patterns)

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

    @staticmethod
    def _prepare_entrypoint(entrypoint):
        entrypoint_list = entrypoint.split()

        if entrypoint_list[0] != "python":
            entrypoint_list[0] = f"$(which {entrypoint_list[0]})"
            entrypoint_list = ["python"] + entrypoint_list
            entrypoint = entrypoint_list

        entrypoint = " ".join(entrypoint_list)
        return entrypoint

    def cleanup_workspace(self):
        if self.tempdir:
            self.tempdir.cleanup()

    @property
    def instructions(self):
        _instruction = {
            "Available tools to solve the problem": self.tool_instructions,
            "Available commands": self.tool_names,
        }
        return _instruction

    def display_files(self):
        msg = (
            "Listing files in the current working directory."
            " (ro) indicates read-only files."
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
        self.logger.info(f"Resetting environment")
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
            current_code_with_line_number=self.current_code_with_line_number(),
            current_breakpoints=self.current_breakpoints(),
            action=None,
            done=self.done,
            score=self.score,
            max_score=self.max_score,
            instructions=self.instructions,
            rewrite_counter=self.rewrite_counter,
            tools=self.tool_instructions,
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

    def load_current_file(self, filepath: str) -> bool:
        self.current_file = filepath
        self.current_file_content = self.load_file(filepath)

    def load_file(self, filepath: str) -> str:
        return (self.working_dir / filepath).read_text()

    def _index_files(self, readonly_patterns: list[str] | None = None):
        # get all file paths relative to the working directory
        self._is_ignored = make_file_matcher(
            self.working_dir / ".debugignore", patterns=readonly_patterns
        )
        self.all_files = sorted(
            os.path.relpath(path, self.working_dir)
            for path in _walk(self.working_dir, skip=self._is_ignored)
        )

        # get list of editable files
        self._is_readonly = make_file_matcher(
            self.working_dir / ".debugreadonly", patterns=readonly_patterns
        )
        self.editable_files = [
            p for p in self.all_files if not self._is_readonly(self.working_dir / p)
        ]

    def directory_tree(self, root: str = None, max_depth: int | None = None):
        root = Path(root or self.working_dir).absolute()
        max_depth = max_depth or self.dir_tree_depth

        if not root.exists() or root.is_file():
            return (
                f"Could not display directory tree because {root} is not a directory."
            )

        # initalize with root directory
        result = [str(root) + "/"]

        # get all paths with correct depth
        for path in _walk(root, max_depth, skip=self._is_ignored):
            rel_path = path.relative_to(root)  # relative path from root
            depth = len(rel_path.parts) - 1  # depth of current path
            indent = "  " * depth  # 2 spaces per level for indent

            # file vs direcrory formatting
            result.append(f"{indent}|-- {path.name}")

            if path.is_dir():
                result[-1] += "/"

            if str(path.relative_to(self.working_dir)) not in self.editable_files:
                result[-1] += " (ro)"

        return "\n".join(result)

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

    def current_code_with_line_number(self):
        if self.current_file is None or self.current_file_content is None:
            return "You are currently not working in a file. You can use ```view path/to/file.py``` to navigate to a file first."

        output = {
            "File name": self.current_file,
            "Content": "\n"
            + show_line_number(
                self.current_file_content,
                self.current_file,
                self.current_breakpoints_state,
            )
            + "\n",
        }
        if self.current_breakpoints_state:
            output["Note"] = (
                "B indicates breakpoint before a certain line of code, this can be changed using pdb commands such as b, cl, etc."
            )
        return output

    def overwrite_file(self, filepath: str, content: str):
        assert isinstance(content, str), "content should be a string."
        with open(pjoin(self.working_dir, filepath), "w") as f:
            f.write(content)

    @property
    def patch(self):
        command = ["git", "diff", "--no-index", self.path, self.working_dir]
        result = subprocess.run(command, text=True, capture_output=True)
        patch = result.stdout.replace(str(self.working_dir), str(self.path))
        return patch

    def step(self, action: str):
        # given action, return new obs, and update infos
        # the action space is composed of a few smaller action spaces
        self.clear_all_observations()
        self.empty_event_queue()
        message, tool_info = self.get_triggered_tools(action)
        if message:
            self.step_observation = Observation("env", message)
        else:
            triggered_tool, tool_args = tool_info
            try:
                self.step_observation = triggered_tool(tool_args)
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
            current_code_with_line_number=self.current_code_with_line_number(),
            current_breakpoints=self.current_breakpoints(),
            action=action,
            instructions=self.instructions,
            score=self.score,
            max_score=self.max_score,
            done=self.done,
            rewrite_counter=self.rewrite_counter,
            tools=self.tool_instructions,
        )

        return self.infos

    def close(self):
        self.cleanup_workspace()
        if self.terminal:
            self.terminal.close()
