import atexit
import copy
import glob
import os
import shutil
import subprocess
import tempfile
from glob import glob
from os.path import join as pjoin
from pathlib import Path

import numpy as np

from froggy.logger import FroggyLogger
from froggy.terminal import Terminal
from froggy.tools.patchers import CodePatcher
from froggy.tools.pdb import PDBTool
from froggy.tools.reasoning import ReasoningTool
from froggy.utils import _walk, make_is_readonly, show_line_number


class TooledEnv:
    def __init__(self):
        self.tools = {}

    @property
    def actions(self):
        return [t.action for t in self.tools.values()]

    @property
    def actions_str(self):
        return ", ".join([item.strip("`") for item in self.actions])

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

    def get_triggered_tools(self, action):
        return [tool for tool in self.tools.values() if tool.is_triggered(action)]

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
        self.logger = logger or FroggyLogger("froggy")

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

    def _index_files(self, readonly_patterns: list[str] | None = None):
        # get list of all the files
        self.all_files = sorted(
            os.path.relpath(pjoin(path, name), self.working_dir)
            for path, _, files in os.walk(self.working_dir)
            for name in files
        )

        # get list of editable files
        froggyignore = (
            self.working_dir / ".froggyignore"
        )  # By default look for .froggyignore.
        self.is_readonly = make_is_readonly(froggyignore, patterns=readonly_patterns)
        self.editable_files = [
            p for p in self.all_files if not self.is_readonly(self.working_dir / p)
        ]

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
            "Available commands": self.actions_str,
        }
        return _instruction

    def display_files(self, editable_only: bool = False):
        msg_prefix = "\nEditable" if editable_only else "\nAll"
        if self.dir_tree_depth is not None:
            msg = f"{msg_prefix} files up to depth {self.dir_tree_depth}:"
        else:
            msg = f"{msg_prefix} files:"
        msg += self.directory_tree(editable_only=editable_only)
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
            self.obs += "Debugging terminal started:\n" f"{self.dbg_obs}\n"

        self.infos = {
            "obs": self.obs,
            "dbg_obs": self.dbg_obs if hasattr(self, "dbg_obs") else "",
            "last_run_obs": self.last_run_obs,
            "dir_tree": self.display_files(editable_only=False),
            "editable_files": self.display_files(editable_only=True),
            "current_breakpoints": (
                self.tools["pdb"].current_breakpoints()
                if self.has_tool("pdb")
                else "No breakpoints are set."
            ),
            "current_code_with_line_number": self.current_code_with_line_number(),
            "action": None,
            "done": self.done,
            "score": self.score,
            "max_score": self.max_score,
            "instructions": self.instructions,
            "rewrite_counter": self.rewrite_counter,
        }
        return self.obs, self.infos

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

    def directory_tree(self, root: str = None, editable_only: bool = False):
        root = Path(root or self.path).absolute()

        if not root.exists() or root.is_file():
            return (
                f"\nCould not display directory tree because {root} is not a directory."
            )

        # initalize with root directory
        result = ["\n", str(self.working_dir) + "/"]

        # get all paths with correct depth
        for path in _walk(root, self.dir_tree_depth):
            path = Path(path)

            rel_path = path.relative_to(root)  # relative path from root
            depth = len(rel_path.parts)  # depth of current path
            indent = "  " * depth  # 2 spaces per level for indent

            if editable_only and self.is_readonly(self.working_dir / rel_path):
                continue

            # file vs direcrory formatting
            if path.is_dir():
                result.append(f"{indent}|-- {path.name}/")
            else:
                if (str(rel_path) in self.editable_files) or (not editable_only):
                    result.append(f"{indent}|-- {path.name}")

        result.append("\n")
        # join with newlines
        return "\n".join(result)

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
        triggered_tools = self.get_triggered_tools(action)
        assert (
            len(triggered_tools) <= 1
        ), f"Multiple tools are triggered by the same action! {action}"

        self.obs = f"Invalid action: {action}."
        if triggered_tools:
            triggered_tool = triggered_tools[0]
            try:
                self.obs = triggered_tool.use(action)
            except:
                self.obs = f"Error while using tool {triggered_tool.name} with action: \n{action}"

            if isinstance(triggered_tool, CodePatcher):
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
                if (
                    self.get_tool(triggered_tool.name).success_chain_action is True
                    and self.get_tool(triggered_tool.name).done_cache is not None
                    and self.get_tool(triggered_tool.name).infos_cache is not None
                ):
                    # use done, score and info from the tool that was executed after reasoning
                    self.done = self.get_tool(triggered_tool.name).done_cache
                    self.score = self.get_tool(triggered_tool.name).score_cache
                    self.infos = copy.copy(
                        self.get_tool(triggered_tool.name).infos_cache
                    )
                    # update obs and action in info
                    self.infos["obs"] = self.obs
                    self.infos["action"] = action
                    return self.obs, self.score, self.done, self.infos

        self.infos = {
            "obs": self.obs,
            "last_run_obs": self.last_run_obs,
            "dbg_obs": self.dbg_obs if hasattr(self, "dbg_obs") else "",
            "dir_tree": self.display_files(editable_only=False),
            "editable_files": self.display_files(editable_only=True),
            "current_code_with_line_number": self.current_code_with_line_number(),
            "current_breakpoints": (
                self.tools["pdb"].current_breakpoints()
                if self.has_tool("pdb")
                else "No breakpoints are set."
            ),
            "action": action,
            "instructions": self.instructions,
            "score": self.score,
            "max_score": self.max_score,
            "done": self.done,
            "rewrite_counter": self.rewrite_counter,
        }

        return self.obs, self.score, self.done, self.infos
