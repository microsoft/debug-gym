import atexit
import os
import tempfile
from pathlib import Path

from debug_gym.gym.terminal import Terminal
from debug_gym.gym.utils import make_file_matcher
from debug_gym.logger import DebugGymLogger


class Workspace:

    def __init__(self, terminal: Terminal, logger: DebugGymLogger | None = None):
        self._tempdir = None
        self.working_dir = None
        self.logger = logger or DebugGymLogger("debug-gym")
        self.terminal = terminal

    def cleanup(self):
        self.working_dir = None
        if self._tempdir:
            self._tempdir.cleanup()
            self._tempdir = None

    def reset(
        self,
        readonly_patterns: list[str] | None = None,
        ignore_patterns: list[str] | None = None,
    ):
        self.cleanup()

        self.working_dir = Path("/testbed")
        if type(self.terminal) is Terminal:
            self._tempdir = tempfile.TemporaryDirectory(prefix="DebugGym-")
            atexit.register(self._tempdir.cleanup)
            self.working_dir = Path(self._tempdir.name).resolve()

        self.logger.debug(f"Working directory: {self.working_dir}")
        self.terminal.working_dir = str(self.working_dir)
        self.setup_file_filters(readonly_patterns, ignore_patterns)

    def setup_file_filters(
        self,
        readonly_patterns: list[str] | None = None,
        ignore_patterns: list[str] | None = None,
    ):
        """Indexes files and subdir in the working
        directory, applying ignore and readonly patterns."""
        self._is_readonly_func = lambda f: False
        self._is_ignored_func = lambda f: False

        readonly_patterns = readonly_patterns or []
        ignore_patterns = ignore_patterns or []

        # Ignore debug gym hidden files
        ignore_patterns += [".debugignore", ".debugreadonly"]

        ignore_patterns += (
            self.read_file(".gitignore").splitlines()
            if self.has_file(".gitignore")
            else []
        )
        ignore_patterns += (
            self.read_file(".debugignore").splitlines()
            if self.has_file(".debugignore")
            else []
        )

        readonly_patterns += (
            self.read_file(".debugreadonly").splitlines()
            if self.has_file(".debugreadonly")
            else []
        )

        # create a matcher function for ignored files, .debugignore has precedence over .gitignore
        self._is_ignored_func = make_file_matcher(
            base_dir=self.working_dir,
            pattern_files=[],
            patterns=ignore_patterns,
        )

        # create a matcher function for readonly files
        self._is_readonly_func = make_file_matcher(
            base_dir=self.working_dir,
            pattern_files=[],
            patterns=readonly_patterns,
        )

    def copy_content(self, src: str | Path, target: str | Path | None = None):
        """Copy files contained in src to a target directory."""
        src = Path(src).resolve()
        target = Path(target or self.working_dir).resolve()
        self.terminal.copy_content(src, target)

    def resolve_path(self, filepath: str | Path, raises=False) -> Path:
        """Convert a relative filepath to absolute based on the working_dir.
        If the path is already absolute, it is returned as is.
        If raises is True, raises FileNotFoundError if the file does not exist,
        is not in the working directory or is ignored by the ignore patterns.
        If raises is False, returns the absolute path regardless of the file existence.
        """
        abs_filepath = Path(filepath)
        if not abs_filepath.is_absolute():
            abs_filepath = Path(self.working_dir) / abs_filepath
        abs_filepath_str = str(abs_filepath)

        if raises and abs_filepath != self.working_dir:
            # Check if file exists, is within working_dir and is not ignored.
            check_cmd = (
                f'abs_path=$(realpath "{abs_filepath_str}"); '
                f'test -e "$abs_path" && [[ "$abs_path" == {self.working_dir}* ]]'
            )
            success, result = self.terminal.run(
                f"{check_cmd} && echo OK || echo MISSING"
            )
            if result.strip() != "OK" or self._is_ignored_func(abs_filepath):
                raise FileNotFoundError(
                    f"`{filepath}` does not exist or is not in "
                    f"the working directory `{self.working_dir}`."
                )

        return Path(abs_filepath_str)

    def read_file(self, filepath: str) -> str:
        """Reads a file from the working directory.
        Raises value error if the file does not exist"""
        abs_filepath = self.resolve_path(filepath, raises=True)
        success, output = self.terminal.run(
            f"cat {abs_filepath}", raises=True, strip_output=False
        )
        return output

    def write_file(self, filepath: str, content: str):
        """Writes `content` to `filepath` exactly as-is, preserving any trailing newlines."""
        abs_filepath = self.resolve_path(filepath)

        # create parent directories via the terminal if needed
        self.terminal.run(f'mkdir -p "{str(abs_filepath.parent)}"', raises=True)

        # In the following command we:
        # - use a single-quoted heredoc (cat <<'nDEBUGGYM_EOF' ... nDEBUGGYM_EOF) so the heredoc body is taken literally (no shell expansion)
        # - append a sentinel character DEBUGGYM_DEL inside the heredoc so we can detect/restore trailing newlines later
        # - capture the heredoc output into shell variable CONTENT since command substitution strips trailing newlines
        # - "${CONTENT%DEBUGGYM_DEL}" removes the trailing sentinel DEBUGGYM_DEL (restoring the original trailing-newline state)
        # - echo -n writes the result without adding an extra newline
        cmd = f"CONTENT=$(cat <<'DEBUGGYM_EOF'\n{content}DEBUGGYM_DEL\nDEBUGGYM_EOF\n); echo -n \"${{CONTENT%DEBUGGYM_DEL}}\" > {abs_filepath}"
        self.terminal.run(cmd, raises=True)

    def directory_tree(self, root: str | Path = None, max_depth: int = 1):
        root = self.resolve_path(root or self.working_dir, raises=True)
        # Use the terminal to run a bash command to list files
        tree_cmd = f"tree --charset=ASCII --noreport -a -v -F -f -L {max_depth} {root} "
        success, output = self.terminal.run(tree_cmd, raises=True)

        first, *rest = output.splitlines()
        lines = [first]
        for line in rest:
            assert "-- " in line
            prefix, path = line.split("-- ", 1)
            prefix += "-- "

            if self._is_ignored_func(path):
                continue

            lines.append(f"{prefix}{os.path.basename(path.rstrip('/'))}")

            if path.endswith("/"):
                # i.e. a directory
                lines[-1] += "/"

            if self._is_readonly_func(path):
                lines[-1] += " (read-only)"

        output = "\n".join(lines)

        # To maintain backward compatibility with previous version of debug-gym.
        output = output.replace("`", "|").replace("    ", "  ")
        return output

    def is_editable(self, filepath):
        return not self._is_readonly_func(self.resolve_path(filepath, raises=True))

    def display_files(self, dir_tree_depth: int = 1) -> str:
        msg = (
            "Listing files in the current working directory."
            " (read-only) indicates read-only files."
            f" Max depth: {str(dir_tree_depth)}.\n"
        )
        msg += self.directory_tree(max_depth=dir_tree_depth)
        return msg

    def has_file(self, filepath: str) -> bool:
        """Checks if a file exists in the working directory.
        Shortcut for `resolve_path` with raises=True.
        """
        try:
            self.resolve_path(filepath, raises=True)
            return True
        except FileNotFoundError:
            return False
