import os
import shlex
import subprocess
from pathlib import Path

from debug_gym.gym.envs.local import LocalEnv as ProductionLocalEnv
from debug_gym.gym.terminals.shell_session import ShellSession
from debug_gym.gym.terminals.terminal import Terminal


class HostTerminal(Terminal):
    """Host-backed terminal for unit tests only."""

    uses_host_filesystem = True

    def __init__(
        self,
        working_dir: str | None = None,
        session_commands: list[str] | None = None,
        env_vars: dict[str, str] | None = None,
        command_timeout: int = 300,
        **kwargs,
    ):
        env_vars = (env_vars or {}) | dict(os.environ)
        super().__init__(
            working_dir=working_dir,
            session_commands=session_commands,
            env_vars=env_vars,
            **kwargs,
        )
        self.command_timeout = command_timeout

    def prepare_command(self, entrypoint: str | list[str]) -> list[str]:
        if isinstance(entrypoint, str):
            entrypoint = [entrypoint]
        commands = [*self.session_commands, *entrypoint]
        return shlex.split(self.default_shell_command) + ["-c", " && ".join(commands)]

    def run(
        self,
        entrypoint: str | list[str],
        timeout: int = None,
        raises: bool = False,
        strip_output: bool = True,
    ) -> tuple[bool, str]:
        effective_timeout = timeout if timeout is not None else self.command_timeout
        try:
            process = subprocess.run(
                self.prepare_command(entrypoint),
                env=self.env_vars,
                cwd=self.working_dir,
                capture_output=True,
                text=True,
                timeout=effective_timeout,
            )
        except subprocess.TimeoutExpired as exc:
            output = (exc.stdout or "") + (exc.stderr or "")
            return False, (
                f"Command timed out after {effective_timeout} seconds"
                + (f"\nPartial output:\n{output}" if output else "")
            )
        success = process.returncode == 0
        output = process.stdout + process.stderr
        if raises and not success:
            raise ValueError(f"Failed to run command: {entrypoint}")
        if self.max_output_bytes > 0 and len(output) > self.max_output_bytes:
            self._raise_output_limit_exceeded(len(output), output)
        return success, output.strip("\r\n") if strip_output else output

    @property
    def default_shell_command(self) -> str:
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

    def copy_content(self, src: str | Path, target: str | Path | None = None) -> None:
        src = Path(src)
        if not src.is_dir():
            raise ValueError(f"Source {src} must be a directory.")
        target = Path(target or self.working_dir)
        self.run(
            f"cp -r {shlex.quote(str(src))}/. {shlex.quote(str(target))}",
            raises=True,
        )


class LocalEnv(ProductionLocalEnv):
    """Local-path environment using the test-only host terminal."""

    def __init__(self, path: str | Path, terminal: Terminal | None = None, **kwargs):
        super().__init__(
            path=str(path),
            terminal=terminal or HostTerminal(),
            **kwargs,
        )
