import os
import shlex
from pathlib import Path

from debug_gym.gym.terminals.terminal import Terminal, UnrecoverableTerminalError
from debug_gym.logger import DebugGymLogger


class WorkspaceError(Exception):
    """Base class for workspace-related errors."""


class WorkspaceReadError(WorkspaceError):
    """Raised when a file cannot be read or is missing from the workspace."""


class WorkspaceWriteError(WorkspaceError):
    """Raised when a file cannot be written."""


class Workspace:

    def __init__(self, terminal: Terminal, logger: DebugGymLogger | None = None):
        self.working_dir = None
        self.logger = logger or DebugGymLogger("debug-gym")
        self.terminal = terminal

    def cleanup(self):
        self.working_dir = None

    def reset(self):
        self.cleanup()

        self.working_dir = self.working_dir or Path("/testbed")

        self.logger.debug(f"Working directory: {self.working_dir}")
        self.terminal.working_dir = str(self.working_dir)

    def copy_content(self, src: str | Path, target: str | Path | None = None):
        """Copy files contained in src to a target directory."""
        src = Path(src).resolve()
        target = Path(target or self.working_dir).resolve()
        self.terminal.copy_content(src, target)

    def _workspace_root(self) -> Path:
        return Path(os.path.normpath(self.working_dir))

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

        # Normalize traversal components in Python for cross-platform compatibility.
        abs_filepath = Path(os.path.normpath(abs_filepath))
        abs_filepath_str = str(abs_filepath)

        workspace_root = self._workspace_root()
        if raises and abs_filepath != workspace_root:
            try:
                abs_filepath.relative_to(workspace_root)
            except ValueError as exc:
                raise FileNotFoundError(
                    f"`{filepath}` does not exist or is not in "
                    f"the working directory `{self.working_dir}`."
                ) from exc

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

        try:
            abs_filepath.relative_to(self._workspace_root())
        except ValueError as exc:
            raise WorkspaceWriteError(
                f"Failed to write `{filepath}` because it is outside the workspace."
            ) from exc

        try:
            self.terminal.write_text(abs_filepath, content)
        except UnrecoverableTerminalError:
            raise
        except (OSError, RuntimeError, ValueError) as exc:
            raise WorkspaceWriteError(f"Failed to write `{filepath}`.") from exc

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
