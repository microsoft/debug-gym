import atexit
import os
import posixpath
import shlex
import tarfile
import time
import uuid
from io import BytesIO
from pathlib import Path, PurePosixPath

import docker
from docker import errors as docker_errors

from debug_gym.gym.terminals.shell_session import ShellSession
from debug_gym.gym.terminals.terminal import (
    DISABLE_ECHO_COMMAND,
    Terminal,
    TerminalError,
    UnrecoverableTerminalError,
)
from debug_gym.logger import DebugGymLogger


class DockerTerminal(Terminal):

    def __init__(
        self,
        working_dir: str | None = None,
        session_commands: list[str] | None = None,
        env_vars: dict[str, str] | None = None,
        logger: DebugGymLogger | None = None,
        # Docker-specific parameters
        base_image: str | None = None,
        registry: str = "",
        setup_commands: list[str] | None = None,
        command_timeout: int = 300,
        extra_labels: dict[str, str] | None = None,
        # Container resource limits
        mem_limit: str = "16G",
        pids_limit: int | None = 4096,
        cpu_limit: float | None = None,
        **kwargs,
    ):
        """
        Args:
            working_dir: Working directory inside the container.
            session_commands: Commands to run at the start of each session.
            env_vars: Environment variables to set in the container.
            logger: Logger instance.
            base_image: Docker image to use.
            registry: Docker registry URL.
            setup_commands: Commands to run once when setting up the container.
            command_timeout: Default timeout in seconds for individual command execution
                (default: 300 = 5 minutes). This is NOT the terminal session lifetime.
                Commands that exceed this timeout will be killed. Can be configured via YAML:
                    terminal_config:
                        type: docker
                        command_timeout: 60
            extra_labels: Additional labels to add to the container (e.g., {"run-id": "my-run"}).
                Useful for identifying containers during cleanup.
            mem_limit: Container memory limit (default: "16G"). Uses Docker's memory limit
                format (e.g., "8G", "32G", "512M").
            pids_limit: Maximum number of PIDs in the container (default: 4096).
                Prevents fork bombs and runaway thread creation. Set to None for unlimited.
            cpu_limit: Maximum number of CPU cores the container can use (e.g., 4.0).
                Default is None (unlimited). Passed to Docker as nano_cpus.
            **kwargs: Additional arguments (ignored with debug log).
        """
        super().__init__(
            working_dir=working_dir,
            session_commands=session_commands,
            env_vars=env_vars,
            logger=logger,
            **kwargs,
        )
        self.base_image = base_image
        self.registry = registry.rstrip("/") + "/" if registry else ""
        self.setup_commands = setup_commands or []
        self.command_timeout = command_timeout
        self.extra_labels = extra_labels or {}
        self.mem_limit = mem_limit
        self.pids_limit = pids_limit
        self.cpu_limit = cpu_limit
        self._docker_client = None  # Lazily initialized
        self._container = None

    @property
    def docker_client(self):
        """Lazy initialization of Docker client."""
        if self._docker_client is None:
            self._docker_client = docker.from_env(timeout=600)
        return self._docker_client

    def _ensure_container_running(self):
        """Verify that the container exists and is running."""
        container = self.container
        try:
            container.reload()
        except docker_errors.NotFound as exc:
            raise UnrecoverableTerminalError(
                "Docker container is not available. It may have been removed."
            ) from exc
        except docker_errors.DockerException as exc:
            raise UnrecoverableTerminalError(
                "Failed to refresh Docker container state."
            ) from exc

        if container.status != "running":
            raise UnrecoverableTerminalError(
                "Docker container is not running. Cannot continue execution."
            )

    @property
    def working_dir(self):
        """Lazy initialization of the working directory."""
        return super().working_dir

    @working_dir.setter
    def working_dir(self, value):
        if self._container is not None:
            raise ValueError(
                "Cannot change working directory while container is running."
            )

        self._working_dir = value

    @property
    def container(self):
        """Lazy initialization of the container."""
        if self._container is None:
            self._container = self.setup_container()
        return self._container

    @property
    def default_shell_command(self) -> list[str]:
        """Expects the container to have bash installed and python executable available."""
        entrypoint = f"docker exec -t -i {self.container.name} /bin/bash --noprofile --norc --noediting"
        return entrypoint

    def new_shell_session(self):
        self._ensure_container_running()
        session = ShellSession(
            shell_command=self.default_shell_command,
            session_commands=[DISABLE_ECHO_COMMAND] + self.session_commands,
            working_dir=".",
            env_vars=self.env_vars,
            logger=self.logger,
        )
        self.sessions.append(session)
        return session

    def prepare_command(
        self, entrypoint: str | list[str], timeout: int | None = None
    ) -> list[str]:
        """Prepares a shell command by combining session commands and entrypoint commands.
        Then wraps the command in a shell call with optional timeout.

        Args:
            entrypoint: Command(s) to run.
            timeout: Optional timeout in seconds. If provided, the command is wrapped
                with the Unix `timeout` command to ensure it doesn't block forever.
        """
        if isinstance(entrypoint, str):
            entrypoint = [entrypoint]
        if self.session_commands:
            entrypoint = self.session_commands + entrypoint
        entrypoint_str = " && ".join(entrypoint)

        # Wrap with timeout command if specified
        if timeout is not None:
            # Use timeout command to kill the process if it exceeds the limit
            # Exit code 124 indicates timeout was reached
            entrypoint_str = (
                f"timeout {timeout} /bin/bash -c {shlex.quote(entrypoint_str)}"
            )
            command = ["/bin/bash", "-c", entrypoint_str]
        else:
            command = ["/bin/bash", "-c", entrypoint_str]

        return command

    def run(
        self,
        entrypoint: str | list[str],
        timeout: int = None,
        raises: bool = False,
        strip_output: bool = True,
    ) -> tuple[bool, str]:
        """Run a command in the terminal. Return command status and output.

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
        command = self.prepare_command(entrypoint, timeout=effective_timeout)

        self.logger.debug(f"Exec run (timeout={effective_timeout}s): {command}")

        self._ensure_container_running()

        try:
            status, output = self.container.exec_run(
                command,
                workdir=self.working_dir,
                environment=self.env_vars,
                stdout=True,
                stderr=True,
            )
        except docker_errors.APIError as exc:
            raise UnrecoverableTerminalError(
                "Docker exec encountered an API error."
            ) from exc
        except docker_errors.DockerException as exc:
            raise UnrecoverableTerminalError(
                "Docker exec failed due to an unexpected container error."
            ) from exc

        # Check raw byte length before decoding to prevent OOM during string allocation
        if self.max_output_bytes > 0 and len(output) > self.max_output_bytes:
            preview = output[:2000].decode(errors="replace")
            self._raise_output_limit_exceeded(len(output), preview)

        output = output.decode()
        if strip_output:
            output = output.strip("\r\n").strip("\n")

        # Check for timeout (exit code 124 from the timeout command)
        if status == 124:
            self.logger.warning(
                f"Command timed out after {effective_timeout}s: {entrypoint}"
            )
            timeout_msg = f"Command timed out after {effective_timeout} seconds"
            if output:
                output = f"{timeout_msg}\nPartial output:\n{output}"
            else:
                output = timeout_msg
            return False, output

        success = status == 0

        if raises and not success:
            # Command includes the entrypoint + session commands
            self.logger.debug(f"Failed to run command `{command}`:\n{output}")
            raise ValueError(f"Failed to run command `{entrypoint}`:\n{output}")

        self.logger.debug(f"Output from terminal with status `{status}`:\n{output}")
        return success, output

    def setup_container(self) -> docker.models.containers.Container:
        # Create and start a container mounting volumes and setting environment variables
        self.logger.debug(
            f"Setting up container with image: {self.registry}{self.base_image}"
        )

        # Generate a unique container name
        container_name = f"debug_gym_{uuid.uuid4()}"

        # Build labels: always include app=debug-gym for identification
        labels = {"app": "debug-gym"}
        if self.extra_labels:
            labels.update(self.extra_labels)

        container_kwargs = dict(
            name=container_name,
            image=f"{self.registry}{self.base_image}",
            command="sleep infinity",  # Keep the container running
            working_dir=self.working_dir,
            environment=self.env_vars,
            labels=labels,
            detach=True,
            auto_remove=True,
            remove=True,
            tty=True,
            stdin_open=True,
            network_mode="host",
            mem_limit=self.mem_limit,
        )
        if self.pids_limit is not None:
            container_kwargs["pids_limit"] = self.pids_limit
        if self.cpu_limit is not None:
            container_kwargs["nano_cpus"] = int(self.cpu_limit * 1e9)

        container = self.docker_client.containers.run(**container_kwargs)
        container.reload()  # Refresh container attributes (e.g., status="running")
        self._run_setup_commands(container)
        self.logger.debug(f"{container} ({container_name}) started successfully.")
        atexit.register(self.clean_up)
        return container

    def _run_setup_commands(self, container):
        """Run setup commands if any. If commands fail, stop the container."""
        if self.setup_commands:
            setup_commands = " && ".join(self.setup_commands)
            self.logger.debug(f"{container} Running setup commands: {setup_commands}")
            try:
                status, output = container.exec_run(
                    ["/bin/bash", "-c", setup_commands],
                    # user="root",  # Run as root to allow installations
                    workdir=self.working_dir,
                    environment=self.env_vars,
                )
            except docker_errors.APIError as exc:
                container.stop()
                raise UnrecoverableTerminalError(
                    "Docker setup commands failed with an API error."
                ) from exc
            except docker_errors.DockerException as exc:
                container.stop()
                raise UnrecoverableTerminalError(
                    "Docker setup commands encountered an unexpected error."
                ) from exc
            if status != 0:
                container.stop()
                raise UnrecoverableTerminalError(
                    f"Failed to run setup command: {setup_commands}\n"
                    f"Output: {output.decode()}"
                )
            self.logger.debug("Setup commands ran successfully.")

    def clean_up(self):
        """Clean up the Docker container."""
        if self._container is not None:
            try:
                self.container.stop(timeout=1)
            except docker_errors.NotFound:
                self.logger.debug(
                    f"Container {self.container.name} not found. "
                    "It might have already been removed."
                )
            except docker_errors.DockerException as exc:
                self.logger.debug(
                    f"Failed to stop container {self.container.name}: {exc}"
                )
            self._container = None

    def close(self):
        super().close()
        self.clean_up()
        # Close the Docker client to release connection pool resources
        if self._docker_client is not None:
            try:
                self._docker_client.close()
            except Exception as exc:
                self.logger.debug(f"Failed to close Docker client: {exc}")
            self._docker_client = None

    def __str__(self):
        return f"DockerTerminal[{self.container}, {self.working_dir}]"

    def _normalize_write_target(
        self, filepath: str | Path
    ) -> tuple[PurePosixPath, PurePosixPath]:
        root = PurePosixPath(posixpath.normpath(str(self.working_dir)))
        target = PurePosixPath(str(filepath))
        if not target.is_absolute():
            target = root / target
        target = PurePosixPath(posixpath.normpath(str(target)))
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise TerminalError(
                "Write target is outside the terminal working directory"
            ) from exc
        if target == root:
            raise TerminalError("Write target must be a file")
        return root, target

    def _exec_write_command(self, command: list[str]) -> tuple[int, bytes]:
        self._ensure_container_running()
        try:
            return self.container.exec_run(
                command,
                workdir="/",
                environment=self.env_vars,
                stdout=True,
                stderr=True,
            )
        except docker_errors.APIError as exc:
            raise UnrecoverableTerminalError(
                "Docker file operation encountered an API error."
            ) from exc
        except docker_errors.DockerException as exc:
            raise UnrecoverableTerminalError(
                "Docker file operation failed unexpectedly."
            ) from exc

    def _reject_symlink_ancestors(
        self, root: PurePosixPath, target: PurePosixPath
    ) -> None:
        current = root
        ancestors = [root]
        for part in target.parent.relative_to(root).parts:
            current /= part
            ancestors.append(current)
        ancestors.append(target)
        for ancestor in ancestors:
            status, _ = self._exec_write_command(["test", "-L", str(ancestor)])
            if status == 0:
                raise TerminalError("Write target contains a symbolic-link directory")

    def _runtime_identity(self) -> tuple[int, int, set[int]]:
        user_status, user_output = self._exec_write_command(["id", "-u"])
        group_status, group_output = self._exec_write_command(["id", "-g"])
        groups_status, groups_output = self._exec_write_command(["id", "-G"])
        if user_status != 0 or group_status != 0 or groups_status != 0:
            raise TerminalError("Failed to determine the container user")
        return (
            int(user_output.strip()),
            int(group_output.strip()),
            {int(group) for group in groups_output.split()},
        )

    def _destination_metadata(self, target: PurePosixPath) -> tuple[int, int, int]:
        runtime_user_id, runtime_group_id, runtime_group_ids = self._runtime_identity()
        exists, _ = self._exec_write_command(["test", "-e", str(target)])
        if exists == 0:
            is_regular, _ = self._exec_write_command(["test", "-f", str(target)])
            if is_regular != 0:
                raise TerminalError("Write target must be a regular file")
            status, output = self._exec_write_command(
                ["stat", "-c", "%a %u %g", "--", str(target)]
            )
            if status != 0:
                raise TerminalError("Failed to inspect destination file")
            mode, user_id, group_id = output.decode().strip().split()
            user_id = int(user_id)
            group_id = int(group_id)
            if runtime_user_id != 0 and (
                user_id != runtime_user_id
                or (
                    group_id not in runtime_group_ids
                    and not self._inherits_parent_group(target.parent, group_id)
                )
            ):
                raise TerminalError("Cannot preserve destination file ownership")
            return int(mode, 8) & ~0o6000, user_id, group_id

        umask_success, umask_output = self.run("umask", raises=False)
        parent_status, parent_output = self._exec_write_command(
            ["stat", "-c", "%a %g", "--", str(target.parent)]
        )
        if not umask_success or parent_status != 0:
            raise TerminalError("Failed to determine destination file metadata")
        parent_mode, parent_group_id = parent_output.decode().strip().split()
        mode = 0o666 & ~int(umask_output.splitlines()[-1], 8)
        group_id = (
            int(parent_group_id) if int(parent_mode, 8) & 0o2000 else runtime_group_id
        )
        return mode, runtime_user_id, group_id

    def _inherits_parent_group(self, parent: PurePosixPath, group_id: int) -> bool:
        status, output = self._exec_write_command(
            ["stat", "-c", "%a %g", "--", str(parent)]
        )
        if status != 0:
            raise TerminalError("Failed to inspect destination directory")
        parent_mode, parent_group_id = output.decode().strip().split()
        return bool(int(parent_mode, 8) & 0o2000 and int(parent_group_id) == group_id)

    def write_bytes(self, filepath: str | Path, content: bytes) -> None:
        """Transfer bytes through Docker without exposing content to a shell."""
        if not isinstance(content, bytes):
            raise TypeError("content must be bytes")

        root, target = self._normalize_write_target(filepath)
        self._reject_symlink_ancestors(root, target)

        success, output = self.run(
            f"mkdir -p -- {shlex.quote(str(target.parent))}",
            raises=False,
        )
        if not success:
            raise TerminalError(f"Failed to create destination directory: {output}")
        mode, user_id, group_id = self._destination_metadata(target)

        temporary_name = f".debug-gym-write-{uuid.uuid4().hex}.tmp"
        temporary_target = target.parent / temporary_name
        archive = BytesIO()
        with tarfile.open(fileobj=archive, mode="w") as tar:
            info = tarfile.TarInfo(name=temporary_name)
            info.size = len(content)
            info.mode = mode
            info.uid = user_id
            info.gid = group_id
            info.mtime = int(time.time())
            tar.addfile(info, BytesIO(content))
        archive.seek(0)

        try:
            self._ensure_container_running()
            self.container.put_archive(str(target.parent), archive.getvalue())
            status, output = self._exec_write_command(
                ["mv", "-f", "--", str(temporary_target), str(target)]
            )
            if status != 0:
                raise TerminalError(
                    f"Failed to replace destination file: {output.decode(errors='replace')}"
                )
        except docker_errors.APIError as exc:
            raise UnrecoverableTerminalError(
                "Docker file transfer encountered an API error."
            ) from exc
        except docker_errors.DockerException as exc:
            raise UnrecoverableTerminalError(
                "Docker file transfer failed unexpectedly."
            ) from exc
        finally:
            try:
                self._exec_write_command(["rm", "-f", "--", str(temporary_target)])
            except UnrecoverableTerminalError:
                pass

    def copy_content(self, src: str | Path, target: str | Path | None = None) -> None:
        """Copy files contained in src on the host to target in the container."""
        src = str(src)
        target = str(target or self.working_dir)

        if not os.path.isdir(src):
            raise ValueError(f"Source {src} must be a directory.")

        self.logger.debug(f"[{self}] Copying {src} to {target}.")

        # Create a tar archive of the file
        tar_stream = BytesIO()
        with tarfile.open(fileobj=tar_stream, mode="w") as tar:
            if os.path.isdir(src):
                for item in Path(src).iterdir():
                    self.logger.debug(f"Adding {item} to tar")
                    tar.add(str(item), arcname=os.path.basename(item))
            else:
                self.logger.debug(f"Adding {src} to tar")
                tar.add(src, arcname=os.path.basename(src))

        tar_stream.seek(0)

        # Get the container object and copy the archive
        self._ensure_container_running()
        try:
            self.container.put_archive(target, tar_stream)
        except docker_errors.APIError as exc:
            raise UnrecoverableTerminalError(
                "Docker copy failed with an API error."
            ) from exc
        except docker_errors.DockerException as exc:
            raise UnrecoverableTerminalError(
                "Docker copy encountered an unexpected error."
            ) from exc
