import os
import shlex
import subprocess
from pathlib import Path

from debug_gym.gym.terminals.shell_session import ShellSession
from debug_gym.gym.terminals.terminal import Terminal


class LocalTerminal(Terminal):

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
    ) -> tuple[bool, str]:
        """Run a list of commands in the terminal. Return command status and output."""
        command = self.prepare_command(entrypoint)
        self.logger.debug(f"Running command in terminal: {command}")
        process = subprocess.Popen(
            command,
            env=self.env_vars,
            cwd=self.working_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            stdout, stderr = process.communicate(timeout=timeout)
            success = process.returncode == 0
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = "", "Timeout expired."
            success = False

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
        which could interfere with the terminal setup (clean outputs)"""
        return "/bin/bash --noprofile --norc"

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
        self.run(f"cp -r {src}/. {target}", raises=True)
