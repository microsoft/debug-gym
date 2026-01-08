import os
import shlex
import subprocess
from pathlib import Path

from debug_gym.gym.terminals.shell_session import ShellSession
from debug_gym.gym.terminals.terminal import Terminal
from debug_gym.logger import DebugGymLogger


class LocalTerminal(Terminal):

    def __init__(
        self,
        working_dir: str | None = None,
        session_commands: list[str] | None = None,
        env_vars: dict[str, str] | None = None,
        logger: DebugGymLogger | None = None,
        # Local-specific parameters
        include_os_env_vars: bool = True,
        command_timeout: int = 300,
        **kwargs,
    ):
        """
        Args:
            working_dir: Working directory for command execution.
            session_commands: Commands to run at the start of each session.
            env_vars: Environment variables to set.
            logger: Logger instance.
            include_os_env_vars: Whether to include current OS environment variables.
            command_timeout: Default timeout in seconds for individual command execution
                (default: 300 = 5 minutes). This is NOT the terminal session lifetime.
                Commands that exceed this timeout will be killed.
            **kwargs: Additional arguments (ignored with debug log).
        """
        env_vars = env_vars or {}
        if include_os_env_vars:
            env_vars = env_vars | dict(os.environ)

        super().__init__(
            working_dir=working_dir,
            session_commands=session_commands,
            env_vars=env_vars,
            logger=logger,
            **kwargs,
        )
        self.command_timeout = command_timeout

    @property
    def working_dir(self):
        """Lazy initialization of the working directory."""
        return super().working_dir

    @working_dir.setter
    def working_dir(self, value):
        self._working_dir = value

    def prepare_command(self, entrypoint: str | list[str]) -> list[str]:
        """Prepares a shell command by combining session commands and entrypoint commands.
        Then wraps the command in a shell (self.default_shell_command) call."""
        if isinstance(entrypoint, str):
            entrypoint = [entrypoint]
        if self.session_commands:
            entrypoint = self.session_commands + entrypoint
        entrypoint = " && ".join(entrypoint)
        command = shlex.split(self.default_shell_command) + ["-c", entrypoint]
        return command

    def run(
        self,
        entrypoint: str | list[str],
        timeout: int = None,
        raises: bool = False,
        strip_output: bool = True,
        background: bool = False,
    ) -> tuple[bool, str]:
        """Run a list of commands in the terminal. Return command status and output.

        Args:
            entrypoint: Command(s) to run.
            timeout: Timeout in seconds for this command. If the command exceeds this
                time, it will be killed and the method returns (False, timeout_message).
                If None, uses self.command_timeout.
            raises: If True, raise ValueError on command failure.
            strip_output: If True, strip trailing newlines from output.
            background: If True, run the command in the background.

        Returns:
            Tuple of (success, output). Success is False if command failed or timed out.
        """
        # Use command_timeout if not specified per-call
        effective_timeout = timeout if timeout is not None else self.command_timeout
        command = self.prepare_command(entrypoint)
        self.logger.debug(
            f"Running command in terminal (timeout={effective_timeout}s): {command}"
        )
        process = subprocess.Popen(
            command,
            env=self.env_vars,
            cwd=self.working_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            assert (
                background is False
            ), "TODO: Background execution is not yet supported for LocalTerminal."
            stdout, stderr = process.communicate(timeout=effective_timeout)
            success = process.returncode == 0
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()  # Collect any partial output
            self.logger.warning(
                f"Command timed out after {effective_timeout}s: {entrypoint}"
            )
            timeout_msg = f"Command timed out after {effective_timeout} seconds"
            partial = (stdout + stderr).strip()
            if partial:
                output = f"{timeout_msg}\nPartial output:\n{partial}"
            else:
                output = timeout_msg
            return False, output

        if raises and not success:
            # Command includes the entrypoint + session commands
            self.logger.debug(f"Failed to run command: {command}")
            raise ValueError(f"Failed to run command: {entrypoint}")

        output = stdout + stderr
        if strip_output:
            output = output.strip("\r\n").strip("\n")

        self.logger.debug(
            f"Output from terminal with status {process.returncode}:\n{output}"
        )
        return success, output

    @property
    def default_shell_command(self) -> str:
        """Starts a new bash session exporting the current python executable as 'python'.
        Flags --noprofile and --norc are used to avoid loading any bash profile or rc file,
        which could interfere with the terminal setup (clean outputs).
        Flag --noediting disables readline editing features including bracketed paste mode.
        """
        return "/bin/bash --noprofile --norc --noediting"

    def new_shell_session(self):
        session = ShellSession(
            shell_command=self.default_shell_command,
            session_commands=self.session_commands,
            working_dir=self.working_dir,
            env_vars=self.env_vars,
            logger=self.logger,
        )
        self.sessions.append(session)
        return session

    def close_shell_session(self, session):
        session.close()
        self.sessions.remove(session)

    def close(self):
        for session in self.sessions:
            self.close_shell_session(session)

    def __str__(self):
        return f"LocalTerminal[{self.working_dir}]"

    def copy_content(self, src: str | Path, target: str | Path | None = None) -> None:
        """Copy files contained in src on the host to target on the host."""
        src = str(src)
        target = str(target or self.working_dir)

        if not os.path.isdir(src):
            raise ValueError(f"Source {src} must be a directory.")

        self.logger.debug(f"[{self}] Copying {src} to {target}.")
        # Use cp to copy files, including hidden files (dotfiles)
        self.run(f"cp -r {shlex.quote(src)}/. {shlex.quote(target)}", raises=True)
