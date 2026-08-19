import os
import shlex
import stat
import subprocess
import sys
import uuid
from pathlib import Path

from debug_gym.gym.terminals.shell_session import ShellSession
from debug_gym.gym.terminals.terminal import Terminal, TerminalError
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
        if os.environ.get("ALLOW_LOCAL_TERMINAL", "").strip().lower() != "true":
            raise TerminalError(
                "Local terminal execution is disabled. Set "
                "ALLOW_LOCAL_TERMINAL=true to explicitly allow commands to run "
                "on the host."
            )

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
    ) -> tuple[bool, str]:
        """Run a list of commands in the terminal. Return command status and output.

        Args:
            entrypoint: Command(s) to run.
            timeout: Timeout in seconds for this command. If the command exceeds this
                time, it will be killed and the method returns (False, timeout_message).
                If None, uses self.command_timeout.
            raises: If True, raise ValueError on command failure.
            strip_output: If True, strip trailing newlines from output.

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
            if self.max_output_bytes > 0 and len(partial) > self.max_output_bytes:
                preview = partial[:2000]
                self._raise_output_limit_exceeded(len(partial), preview)
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
        if self.max_output_bytes > 0 and len(output) > self.max_output_bytes:
            preview = output[:2000]
            self._raise_output_limit_exceeded(len(output), preview)
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

    def _effective_umask(self) -> int:
        success, output = self.run("umask", raises=False)
        if not success:
            raise TerminalError("Failed to determine the local terminal umask")
        return int(output.splitlines()[-1], 8)

    @staticmethod
    def _is_symbolic_link(path: Path) -> bool:
        return path.is_symlink() or (
            hasattr(path, "is_junction") and path.is_junction()
        )

    def _normalize_write_target(self, filepath: str | Path) -> tuple[Path, Path]:
        root = Path(os.path.abspath(os.path.normpath(self.working_dir)))
        current = Path(root.anchor)
        for part in root.parts[1:]:
            current /= part
            if self._is_symbolic_link(current):
                raise TerminalError(
                    "Terminal working directory must not contain symbolic links"
                )
        root = root.resolve(strict=True)

        target = Path(filepath)
        if not target.is_absolute():
            target = root / target
        target = Path(os.path.abspath(os.path.normpath(target)))
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise TerminalError(
                "Write target is outside the terminal working directory"
            ) from exc
        if target == root:
            raise TerminalError("Write target must be a file")
        return root, target

    def _reject_symlink_ancestors(self, root: Path, target: Path) -> None:
        current = root
        for part in target.relative_to(root).parts:
            current /= part
            if self._is_symbolic_link(current):
                raise TerminalError("Write target must not contain symbolic links")
            if not current.exists():
                break

    def write_bytes(self, filepath: str | Path, content: bytes) -> None:
        """Write bytes directly with host filesystem APIs."""
        if not isinstance(content, bytes):
            raise TypeError("content must be bytes")
        root, target = self._normalize_write_target(filepath)
        self._reject_symlink_ancestors(root, target)

        effective_umask = self._effective_umask()
        directory_mode = 0o777 & ~effective_umask
        new_file_mode = 0o666 & ~effective_umask
        use_dir_fd = all(
            function in os.supports_dir_fd
            for function in (
                os.chmod,
                os.mkdir,
                os.open,
                os.rename,
                os.stat,
                os.unlink,
            )
        )
        parent_descriptor = None
        if use_dir_fd:
            parent_descriptor = os.open(
                root,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
            try:
                for part in target.parent.relative_to(root).parts:
                    try:
                        next_descriptor = os.open(
                            part,
                            os.O_RDONLY
                            | getattr(os, "O_DIRECTORY", 0)
                            | getattr(os, "O_NOFOLLOW", 0),
                            dir_fd=parent_descriptor,
                        )
                    except FileNotFoundError:
                        os.mkdir(
                            part,
                            directory_mode,
                            dir_fd=parent_descriptor,
                        )
                        os.chmod(
                            part,
                            directory_mode,
                            dir_fd=parent_descriptor,
                            follow_symlinks=False,
                        )
                        next_descriptor = os.open(
                            part,
                            os.O_RDONLY
                            | getattr(os, "O_DIRECTORY", 0)
                            | getattr(os, "O_NOFOLLOW", 0),
                            dir_fd=parent_descriptor,
                        )
                    os.close(parent_descriptor)
                    parent_descriptor = next_descriptor
            except OSError:
                os.close(parent_descriptor)
                raise
        else:
            missing_parents = []
            current_parent = target.parent
            while current_parent != root and not current_parent.exists():
                missing_parents.append(current_parent)
                current_parent = current_parent.parent
            for parent in reversed(missing_parents):
                parent.mkdir(mode=directory_mode)
                os.chmod(parent, directory_mode)

        temporary_name = f".debug-gym-write-{uuid.uuid4().hex}.tmp"
        temporary_path = target.parent / temporary_name
        descriptor = None
        try:
            try:
                if parent_descriptor is not None:
                    target_stat = os.stat(
                        target.name,
                        dir_fd=parent_descriptor,
                        follow_symlinks=False,
                    )
                else:
                    target_stat = target.lstat()
            except FileNotFoundError:
                target_stat = None

            if target_stat is not None and not stat.S_ISREG(target_stat.st_mode):
                raise TerminalError("Write target must be a regular file")

            create_mode = 0o600 if target_stat is not None else new_file_mode
            if parent_descriptor is not None:
                descriptor = os.open(
                    temporary_name,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                    create_mode,
                    dir_fd=parent_descriptor,
                )
            else:
                descriptor = os.open(
                    temporary_path,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                    create_mode,
                )

            with os.fdopen(os.dup(descriptor), "wb") as temporary_file:
                temporary_file.write(content)
            if target_stat is not None and hasattr(os, "fchown"):
                try:
                    os.fchown(descriptor, target_stat.st_uid, target_stat.st_gid)
                except PermissionError as exc:
                    raise TerminalError(
                        "Cannot preserve destination file ownership"
                    ) from exc
            if target_stat is not None:
                desired_mode = stat.S_IMODE(target_stat.st_mode) & ~0o6000
            else:
                desired_mode = new_file_mode
            if hasattr(os, "fchmod"):
                os.fchmod(descriptor, desired_mode)
            else:
                os.chmod(temporary_path, desired_mode)
            os.close(descriptor)
            descriptor = None

            if parent_descriptor is not None:
                os.replace(
                    temporary_name,
                    target.name,
                    src_dir_fd=parent_descriptor,
                    dst_dir_fd=parent_descriptor,
                )
            else:
                os.replace(temporary_path, target)
        finally:
            original_error = sys.exception()
            cleanup_errors = []
            if descriptor is not None:
                try:
                    os.close(descriptor)
                except OSError as exc:
                    cleanup_errors.append(exc)
            try:
                if parent_descriptor is not None:
                    os.unlink(temporary_name, dir_fd=parent_descriptor)
                else:
                    temporary_path.unlink(missing_ok=True)
            except FileNotFoundError:
                pass
            except OSError as exc:
                cleanup_errors.append(exc)
            if parent_descriptor is not None:
                try:
                    os.close(parent_descriptor)
                except OSError as exc:
                    cleanup_errors.append(exc)
            if cleanup_errors:
                details = "; ".join(str(error) for error in cleanup_errors)
                if original_error is None:
                    raise TerminalError(
                        f"Failed to clean up temporary write state: {details}"
                    ) from cleanup_errors[0]
                self.logger.warning(
                    f"Failed to clean up temporary write state after an error: {details}"
                )

    def copy_content(self, src: str | Path, target: str | Path | None = None) -> None:
        """Copy files contained in src on the host to target on the host."""
        src = str(src)
        target = str(target or self.working_dir)

        if not os.path.isdir(src):
            raise ValueError(f"Source {src} must be a directory.")

        self.logger.debug(f"[{self}] Copying {src} to {target}.")
        # Use cp to copy files, including hidden files (dotfiles)
        self.run(f"cp -r {shlex.quote(src)}/. {shlex.quote(target)}", raises=True)
