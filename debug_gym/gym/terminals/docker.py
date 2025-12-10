import atexit
import os
import tarfile
import uuid
from io import BytesIO
from pathlib import Path

import docker
from docker import errors as docker_errors

from debug_gym.gym.terminals.shell_session import ShellSession
from debug_gym.gym.terminals.terminal import (
    DISABLE_ECHO_COMMAND,
    Terminal,
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
        **kwargs,
    ):
        """
        volumes (dict or list): A dictionary to configure volumes mounted
                inside the container. The key is either the host path or a
                volume name, and the value is a dictionary with the keys:

                - ``bind`` The path to mount the volume inside the container
                - ``mode`` Either ``rw`` to mount the volume read/write, or
                  ``ro`` to mount it read-only.
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
        self.docker_client = docker.from_env(timeout=600)
        self._container = None

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

    def prepare_command(self, entrypoint: str | list[str]) -> list[str]:
        """Prepares a shell command by combining session commands and entrypoint commands.
        Then wraps the command in a shell call."""
        if isinstance(entrypoint, str):
            entrypoint = [entrypoint]
        if self.session_commands:
            entrypoint = self.session_commands + entrypoint
        entrypoint = " && ".join(entrypoint)
        command = ["/bin/bash", "-c", entrypoint]
        return command

    def run(
        self,
        entrypoint: str | list[str],
        timeout: int = None,
        raises: bool = False,
        strip_output: bool = True,
    ) -> tuple[bool, str]:
        """Run a command in the terminal. Return command status and output."""
        command = self.prepare_command(entrypoint)

        self.logger.debug(f"Exec run: {command}")

        self._ensure_container_running()

        # TODO: docker exec_run timeout?
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
        success = status == 0

        output = output.decode()
        if strip_output:
            output = output.strip("\r\n").strip("\n")

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
        container = self.docker_client.containers.run(
            name=container_name,
            image=f"{self.registry}{self.base_image}",
            command="sleep infinity",  # Keep the container running
            working_dir=self.working_dir,
            environment=self.env_vars,
            detach=True,
            auto_remove=True,
            remove=True,
            tty=True,
            stdin_open=True,
            network_mode="host",
            mem_limit="16G",
        )
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
                raise ValueError(
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

    def __str__(self):
        return f"DockerTerminal[{self.container}, {self.working_dir}]"

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
