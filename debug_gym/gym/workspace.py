import atexit
import os
import shlex
import tempfile
from pathlib import Path

from debug_gym.gym.terminals.local import LocalTerminal
from debug_gym.gym.terminals.terminal import Terminal
from debug_gym.logger import DebugGymLogger


class WorkspaceError(Exception):
    """Base class for workspace-related errors."""


class WorkspaceReadError(WorkspaceError):
    """Raised when a file cannot be read or is missing from the workspace."""


class WorkspaceWriteError(WorkspaceError):
    """Raised when a file cannot be written."""


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

    def reset(self):
        self.cleanup()

        self.working_dir = self.working_dir or Path("/testbed")
        # only create temp dir for local terminal
        if type(self.terminal) is LocalTerminal:
            self._tempdir = tempfile.TemporaryDirectory(prefix="DebugGym-")
            atexit.register(self._tempdir.cleanup)
            self.working_dir = Path(self._tempdir.name).resolve()

        self.logger.debug(f"Working directory: {self.working_dir}")
        self.terminal.working_dir = str(self.working_dir)

    def copy_content(self, src: str | Path, target: str | Path | None = None):
        """Copy files contained in src to a target directory."""
        src = Path(src).resolve()
        target = Path(target or self.working_dir).resolve()
        self.terminal.copy_content(src, target)

    def resolve_path(self, filepath: str | Path, raises: bool = False) -> Path:
        """Convert a relative filepath to absolute based on the working_dir.
        If the path is already absolute, it is returned as is.
        If raises is True, raises FileNotFoundError if the file does not exist
        or is not in the working directory.
        If raises is False, returns the absolute path regardless of the file existence.
        """
        abs_filepath = Path(filepath)
        if not abs_filepath.is_absolute():
            abs_filepath = Path(self.working_dir) / abs_filepath

        # Normalize the path (resolve .., . without resolving symlinks)
        # This is done in Python for cross-platform compatibility
        abs_filepath = Path(os.path.normpath(abs_filepath))
        abs_filepath_str = str(abs_filepath)

        if raises and abs_filepath != self.working_dir:
            # Check if file is within working_dir (security check)
            # Use os.path.commonpath to safely check if path is under working_dir
            try:
                common = os.path.commonpath([str(self.working_dir), abs_filepath_str])
                if common != str(self.working_dir):
                    raise FileNotFoundError(
                        f"`{filepath}` does not exist or is not in "
                        f"the working directory `{self.working_dir}`."
                    )
            except ValueError:
                # commonpath raises ValueError for paths on different drives (Windows)
                raise FileNotFoundError(
                    f"`{filepath}` does not exist or is not in "
                    f"the working directory `{self.working_dir}`."
                )

            # Check if file exists via terminal
            check_cmd = f"test -e {shlex.quote(abs_filepath_str)}"
            success, _ = self.terminal.run(check_cmd, raises=False)
            if not success:
                raise FileNotFoundError(
                    f"`{filepath}` does not exist or is not in "
                    f"the working directory `{self.working_dir}`."
                )

        return Path(abs_filepath_str)

    def read_file(self, filepath: str, raises: bool = True) -> str:
        """Reads a file from the working directory.
        By default, raises WorkspaceReadError if the file does not exist or cannot be read.
        """
        try:
            abs_filepath = self.resolve_path(filepath, raises=raises)
        except FileNotFoundError as exc:
            raise WorkspaceReadError(
                f"Failed to read `{filepath}` because it does not exist in the working directory `{self.working_dir}`."
            ) from exc

        success_read, output = self.terminal.run(
            f"cat {shlex.quote(str(abs_filepath))}", raises=False, strip_output=False
        )

        if not success_read:
            message = output.strip() or "Unknown error"
            raise WorkspaceReadError(
                f"Failed to read `{filepath}`. Command output:\n{message}"
            )

        return output

    def write_file(self, filepath: str, content: str):
        """Writes `content` to `filepath` exactly as-is, preserving any trailing newlines."""
        abs_filepath = self.resolve_path(filepath, raises=False)

        # Security check: ensure path is within workspace
        try:
            common = os.path.commonpath([str(self.working_dir), str(abs_filepath)])
            if common != str(self.working_dir):
                raise WorkspaceWriteError(
                    f"Failed to write `{filepath}` because it is outside the workspace."
                )
        except ValueError:
            # commonpath raises ValueError for paths on different drives (Windows)
            raise WorkspaceWriteError(
                f"Failed to write `{filepath}` because it is outside the workspace."
            )

        def _run_or_raise(command: str):
            success, output = self.terminal.run(
                command, raises=False, strip_output=False
            )
            if not success:
                message = output.strip() or "Unknown error"
                raise WorkspaceWriteError(
                    f"Failed to write `{filepath}`. Command output:\n{message}"
                )

        # create parent directories via the terminal if needed
        _run_or_raise(f"mkdir -p {shlex.quote(str(abs_filepath.parent))}")

        # We will split content in chunks of 32kB to avoid hitting command length limits.
        chunk_size = 32 * 1024  # 32kB
        first_chunk = content[:chunk_size]
        rest = content[chunk_size:]

        # In the following command we:
        # - use a single-quoted heredoc (cat <<'nDEBUGGYM_EOF' ... nDEBUGGYM_EOF) so the heredoc body is taken literally (no shell expansion)
        # - append a sentinel character DEBUGGYM_DEL inside the heredoc so we can detect/restore trailing newlines later
        # - capture the heredoc output into shell variable CONTENT since command substitution strips trailing newlines
        # - "${CONTENT%DEBUGGYM_DEL}" removes the trailing sentinel DEBUGGYM_DEL (restoring the original trailing-newline state)
        # - echo -n writes the result without adding an extra newline
        quoted_filepath = shlex.quote(str(abs_filepath))
        cmd = (
            "CONTENT=$(cat <<'DEBUGGYM_EOF'\n"
            f"{first_chunk}DEBUGGYM_DEL\nDEBUGGYM_EOF\n); "
            'echo -n "${CONTENT%DEBUGGYM_DEL}" > '
            f"{quoted_filepath}"
        )
        _run_or_raise(cmd)

        for i in range(0, len(rest), chunk_size):
            chunk = rest[i : i + chunk_size]
            cmd = (
                "CONTENT=$(cat <<'DEBUGGYM_EOF'\n"
                f"{chunk}DEBUGGYM_DEL\nDEBUGGYM_EOF\n); "
                'echo -n "${CONTENT%DEBUGGYM_DEL}" >> '
                f"{quoted_filepath}"
            )
            _run_or_raise(cmd)

    def directory_tree(self, root: str | Path = None, max_depth: int = 1):
        """List the directory tree using the `tree` command.
        Requires the `tree` package to be installed in the terminal.
        """
        root = self.resolve_path(root or self.working_dir, raises=True)
        # Validate max_depth to prevent abuse
        max_depth = max(1, min(int(max_depth), 20))
        # Use the terminal to run a bash command to list files
        tree_cmd = f"tree --charset=ASCII --noreport -a -v -F -f -l -L {max_depth} {shlex.quote(str(root))} "
        success, output = self.terminal.run(tree_cmd, raises=False)
        if not success:
            raise WorkspaceReadError(
                f"Failed to list directory '{root}'. Command output:\n{output}"
            )

        first, *rest = output.splitlines()
        lines = [first]
        for line in rest:
            assert "-- " in line
            prefix, path = line.split("-- ", 1)
            prefix += "-- "

            # Remove trailing / and symbolic link details.
            clean_path = path.split(" -> ")[0].rstrip("/")
            lines.append(f"{prefix}{os.path.basename(clean_path)}")

            if path.endswith("/"):
                # i.e. a directory
                lines[-1] += "/"

        output = "\n".join(lines)

        # To maintain backward compatibility with previous version of debug-gym.
        output = output.replace("`", "|").replace("    ", "  ")
        return output

    def has_file(self, filepath: str) -> bool:
        """Checks if a file exists in the working directory.
        Shortcut for `resolve_path` with raises=True.
        """
        try:
            self.resolve_path(filepath, raises=True)
            return True
        except FileNotFoundError:
            return False
