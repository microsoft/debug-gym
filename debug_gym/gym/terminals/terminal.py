import atexit
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path

from debug_gym.gym.terminals.shell_session import DEFAULT_PS1
from debug_gym.logger import DebugGymLogger


class TerminalError(RuntimeError):
    """Base exception for terminal-related failures."""


class UnrecoverableTerminalError(TerminalError):
    """Raised when the terminal becomes unusable and the episode must stop."""

    def __init__(self, message: str, env_info=None):
        super().__init__(message)
        self.env_info = env_info


DISABLE_ECHO_COMMAND = "stty -echo"

# Default cap on command output to prevent unbounded memory/disk usage.
# Commands producing more output than this will have their output truncated.
DEFAULT_MAX_OUTPUT_BYTES = 1_000_000  # 1 MB


class Terminal(ABC):

    def __init__(
        self,
        working_dir: str | None = None,
        session_commands: list[str] | None = None,
        env_vars: dict[str, str] | None = None,
        logger: DebugGymLogger | None = None,
        max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
        **kwargs,
    ):
        self.logger = logger or DebugGymLogger("debug-gym")
        self.session_commands = session_commands or []
        self.env_vars = env_vars or {}
        # Clean up output by disabling terminal prompt and colors
        self.env_vars["NO_COLOR"] = "1"  # disable colors
        self.env_vars["PYTHONSTARTUP"] = ""  # prevent Python from loading startup files
        # use a sentinel to know when to stop reading
        self.env_vars["PS1"] = DEFAULT_PS1
        self.env_vars["PYTHONDONTWRITEBYTECODE"] = "1"  # prevent creation of .pyc files

        self._working_dir = working_dir
        self.max_output_bytes = max_output_bytes
        self.sessions = []

        kwargs.pop("type", None)  # remove 'type' if present
        if kwargs:
            self.logger.debug(f"Ignoring unknown parameters: {kwargs}")

    @property
    def working_dir(self):
        """Lazy initialization of the working directory."""
        if self._working_dir is None:
            _tempdir = tempfile.TemporaryDirectory(prefix="Terminal-")
            atexit.register(_tempdir.cleanup)
            self._working_dir = str(Path(_tempdir.name).resolve())
            self.logger.debug(f"Using temporary working directory: {self._working_dir}")
        return self._working_dir

    @working_dir.setter
    def working_dir(self, value):
        self._working_dir = value

    @abstractmethod
    def prepare_command(self, entrypoint: str | list[str]) -> list[str]:
        """Prepares a shell command by combining session commands and entrypoint commands.
        Then wraps the command in a shell (self.default_shell_command) call."""
        pass

    @abstractmethod
    def run(
        self,
        entrypoint: str | list[str],
        timeout: int = None,
        raises: bool = False,
        strip_output: bool = True,
    ) -> tuple[bool, str]:
        """Run a list of commands in the terminal. Return command status and output."""
        pass

    @property
    @abstractmethod
    def default_shell_command(self) -> str:
        pass

    @abstractmethod
    def new_shell_session(self):
        pass

    def close_shell_session(self, session):
        session.close()
        self.sessions.remove(session)

    def close(self):
        for session in self.sessions:
            self.close_shell_session(session)

    def _truncate_output(self, output: str) -> str:
        """Truncate command output to max_output_bytes to prevent unbounded memory/disk usage."""
        if self.max_output_bytes > 0 and len(output) > self.max_output_bytes:
            original_len = len(output)
            output = (
                output[: self.max_output_bytes]
                + f"\n\n[OUTPUT TRUNCATED: {original_len} bytes -> {self.max_output_bytes} bytes]"
            )
        return output

    def __str__(self):
        return f"Terminal[{self.working_dir}]"

    @abstractmethod
    def copy_content(self, src: str | Path, target: str | Path | None = None) -> None:
        """Copy files contained in src on the host to target on the host."""
        pass
