import ast
import atexit
import copy
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

from froggy.logger import FroggyLogger
from froggy.terminal import Terminal
from froggy.tools.pdb import PDBTool
from froggy.tools.reasoning import ReasoningTool
from froggy.tools.rewrite import RewriteTool
from froggy.utils import _walk, make_file_matcher, show_line_number


@dataclass
class EnvInfo:
    obs: str
    last_run_obs: str
    dbg_obs: str
    dir_tree: str
    current_code_with_line_number: dict | str
    current_breakpoints: str
    action: str
    instructions: dict
    score: int
    max_score: int
    done: bool
    rewrite_counter: int
    tools: dict


class TooledEnv:
    def __init__(self):
        self.tools = {}

    @property
    def tool_names(self):
        return ", ".join([t.name for t in self.tools.values()])

    def seed(self, seed):
        self.rng = np.random.RandomState(seed)

    def add_tool(self, tool):
        if tool.name in self.tools:
            raise ValueError(f"Tool {tool.name} already exists!")

        self.tools[tool.name] = tool
        tool.register(self)

    def has_tool(self, tool_name):
        return tool_name in self.tools

    def get_tool(self, tool_name):
        return self.tools[tool_name]

    def parse_args(self, args):
        args = "f({})".format(args)
        tree = ast.parse(args)
        funccall = tree.body[0].value
        args = [ast.literal_eval(arg) for arg in funccall.args]
        kwargs = {arg.arg: ast.literal_eval(arg.value) for arg in funccall.keywords}
        return args, kwargs

    def parse_action(self, action):
        action = action.strip()
        # remove ``` in case LLM generates ```tool_name(args, kwargs)```
        if action.startswith("```"):
            action = action[3:]
        if action.endswith("```"):
            action = action[:-3]
        action = action.strip()
        assert "(" in action and action.endswith(")"), "Syntax Error: {}".format(action)
        tool_name, args = action.split("(", 1)
        args = args[:-1]
        tool_name, args = tool_name.strip(), args.strip()
        assert tool_name is not None, "Syntax Error: {}".format(action)
        try:
            args, kwargs = self.parse_args(args)
        except Exception as e:
            raise Exception("Syntax Error: {}\n{}".format(action, str(e)))
        return tool_name, args, kwargs

    def get_triggered_tools(self, action):
        try:
            tool_name, args, kwargs = self.parse_action(action)
        except Exception as e:
            # parse error
            return str(e), None
        if tool_name not in self.tools:
            # failed to find tool
            return f"Unregistered tool: {tool_name}", None
        tool = self.tools[tool_name]
        return None, [tool, args, kwargs]

    @property
    def tool_instructions(self):
        return {name: tool.instructions for name, tool in self.tools.items()}


class RepoEnv(TooledEnv):

    DEFAULT_MAX_SCORE = 1

    def __init__(
        self,
        path: str | None = None,
        entrypoint: str = "python -m pytest -sq .",
        debug_entrypoint: str | None = None,
        readonly_patterns: list[str] | None = None,
        run_on_rewrite: bool = True,
        run_timeout: int | None = None,
        dir_tree_depth: int | None = None,
        auto_view_change: bool = True,
        terminal: Terminal | None = None,
        logger: FroggyLogger | None = None,
    ):
        super().__init__()

        self.path = None
        self.max_score = RepoEnv.DEFAULT_MAX_SCORE
        self.run_on_rewrite = run_on_rewrite
        self.run_timeout = run_timeout
        self.dir_tree_depth = dir_tree_depth
        self.auto_view_change = auto_view_change
        self.terminal = terminal or Terminal()
        self.entrypoint = entrypoint
        self.debug_entrypoint = debug_entrypoint or entrypoint
        self.logger = logger or FroggyLogger("froggy")
        self.infos: EnvInfo | None = None

        self.setup_workspace(
            path=path,
            entrypoint=entrypoint,
            debug_entrypoint=debug_entrypoint,
            readonly_patterns=readonly_patterns,
        )
        self.last_run_obs = None
        self.score = 0
        self.done = False
        self.rewrite_counter = 0

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

        self.current_file = None
        self.current_file_content = None
        self.current_breakpoints_state = {}

        # override entrypoint as it might be task dependent
        self.set_entrypoints(entrypoint, debug_entrypoint)

        # Set up the terminal working dir
        self.terminal.working_dir = str(self.working_dir)

    def set_entrypoints(self, entrypoint, debug_entrypoint):
        if entrypoint:
            entrypoint = entrypoint or ""
            debug_entrypoint = debug_entrypoint or entrypoint
            self.entrypoint = self._prepare_entrypoint(entrypoint)
            self.debug_entrypoint = self._prepare_entrypoint(debug_entrypoint)

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

    def reset(self, *, seed=None, options: dict = None, restore_code=True):
        self.logger.info(f"Resetting environment")
        options = options or {}
        self.current_file = None
        self.current_file_content = None
        self.current_breakpoints_state = {}
        self.rewrite_counter = 0

        if restore_code:
            self.restore()

        # Run the initial code. This will set self.last_run_obs, self.done and self.score.
        self.logger.info(f"Running initial evaluation")
        self.run()

        self.obs = ""
        if self.has_tool("pdb"):
            self.get_tool("pdb").start_pdb()
            self.dbg_obs = self.get_tool("pdb").pdb_obs
            # self.obs += "Debugging terminal started:\n" f"{self.dbg_obs}\n"

        self.infos = EnvInfo(
            obs=self.obs,
            dbg_obs=self.dbg_obs if hasattr(self, "dbg_obs") else "",
            last_run_obs=self.last_run_obs,
            dir_tree=self.display_files(),
            current_breakpoints=(
                self.tools["pdb"].current_breakpoints()
                if self.has_tool("pdb")
                else "No breakpoints are set."
            ),
            current_code_with_line_number=self.current_code_with_line_number(),
            action=None,
            done=self.done,
            score=self.score,
            max_score=self.max_score,
            instructions=self.instructions,
            rewrite_counter=self.rewrite_counter,
            tools=self.tool_instructions,
        )

        return self.infos

    def run(self):
        success, output = self.terminal.run(self.entrypoint, timeout=self.run_timeout)
        self.last_run_obs = output
        self.score = int(success)
        self.done = success

        return self.last_run_obs, self.done

    def load_current_file(self, filepath: str) -> bool:
        self.current_file = filepath
        self.current_file_content = self.load_file(filepath)

    def load_file(self, filepath: str) -> str:
        return (self.working_dir / filepath).read_text()

    def _index_files(self, readonly_patterns: list[str] | None = None):
        # get all file paths relative to the working directory
        self._is_ignored = make_file_matcher(
            self.working_dir / ".froggyignore", patterns=readonly_patterns
        )
        self.all_files = sorted(
            os.path.relpath(path, self.working_dir)
            for path in _walk(self.working_dir, skip=self._is_ignored)
        )

        # get list of editable files
        self._is_readonly = make_file_matcher(
            self.working_dir / ".froggyreadonly", patterns=readonly_patterns
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

    def current_code_with_line_number(self):
        if self.current_file is None or self.current_file_content is None:
            return """You are currently not working in a file. You can use view(path="path/to/file.py") to navigate to a file first."""

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
        if self.has_tool("pdb"):
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
        message, tool_info = self.get_triggered_tools(action)
        if message:
            self.obs = message
        else:
            triggered_tool, args, kwargs = tool_info
            try:
                self.obs = triggered_tool.use(*args, **kwargs)
            except Exception as e:
                self.obs = f"Error while using tool {triggered_tool.name} with action: \n{action}"

            if isinstance(triggered_tool, RewriteTool):
                self.rewrite_counter += 1
                if self.get_tool(triggered_tool.name).rewrite_success:
                    if self.run_on_rewrite:
                        self.obs += "\nNew code has been run."
                        self.run()
                    if self.has_tool("pdb"):
                        # Restart pdb to take into account recent changes.
                        self.get_tool("pdb").restart_pdb()
                        self.dbg_obs = self.get_tool("pdb").pdb_obs
                        self.obs += (
                            "\nDebugging terminal started:\n" f"{self.dbg_obs}\n"
                        )
            elif isinstance(triggered_tool, PDBTool):
                if self.auto_view_change:
                    current_frame_file = self.get_tool("pdb").current_frame_file
                    if current_frame_file in self.all_files:
                        self.load_current_file(self.get_tool("pdb").current_frame_file)
            elif isinstance(triggered_tool, ReasoningTool):
                reasoning_tool = self.get_tool(triggered_tool.name)
                if (
                    reasoning_tool.success_chain_action is True
                    and reasoning_tool.infos_cache is not None
                    and reasoning_tool.infos_cache.done is not None
                ):
                    # use done, score and info from the tool that was executed after reasoning
                    self.done = reasoning_tool.infos_cache.done
                    self.score = reasoning_tool.infos_cache.score
                    self.infos = copy.deepcopy(reasoning_tool.infos_cache)
                    # update obs and action in info
                    self.infos.obs = self.obs
                    self.infos.action = action
                    return self.infos

        self.infos = EnvInfo(
            obs=self.obs,
            last_run_obs=self.last_run_obs,
            dbg_obs=self.dbg_obs if hasattr(self, "dbg_obs") else "",
            dir_tree=self.display_files(),
            current_code_with_line_number=self.current_code_with_line_number(),
            current_breakpoints=(
                self.tools["pdb"].current_breakpoints()
                if self.has_tool("pdb")
                else "No breakpoints are set."
            ),
            action=action,
            instructions=self.instructions,
            score=self.score,
            max_score=self.max_score,
            done=self.done,
            rewrite_counter=self.rewrite_counter,
            tools=self.tool_instructions,
        )

        return self.infos
