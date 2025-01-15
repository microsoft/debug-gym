import atexit
import errno
import fcntl
import io
import logging
import os
import pty
import shlex
import signal
import subprocess
import sys
import tempfile
import termios
import time

import docker

# logger = logging.getLogger("froggy")


class Terminal:

    def __init__(
        self,
        working_dir: str = None,
        setup_commands: list[str] = None,
        env_vars: dict[str, str] = None,
        include_os_env_vars: bool = True,
        logger=logging.getLogger("froggy"),
        **kwargs,
    ):
        self.setup_commands = setup_commands or []
        self.env_vars = env_vars or {}
        if include_os_env_vars:
            self.env_vars = self.env_vars | dict(os.environ)
        # Clean up output by disabling terminal prompt and colors
        self.env_vars["NO_COLOR"] = "1"  # disable colors
        self.env_vars["PS1"] = ""  # disable prompt
        self._working_dir = working_dir
        self._master = None  # PTY master file descriptor
        self.logger = logger

    @property
    def working_dir(self):
        """Lazy initialization of the working directory."""
        if self._working_dir is None:
            temp_dir = tempfile.TemporaryDirectory(prefix="Terminal-")
            atexit.register(lambda: temp_dir.cleanup())
            self._working_dir = temp_dir.name
            self.logger.debug(f"Using temporary working directory: {self._working_dir}")
        return self._working_dir

    @working_dir.setter
    def working_dir(self, value):
        self._working_dir = value

    def prepare_command(self, entrypoint: str | list[str]) -> list[str]:
        """Prepares a shell command by combining setup commands and entrypoint commands.
        Then wraps the command in a shell (self.default_entrypoint) call."""
        if isinstance(entrypoint, str):
            entrypoint = [entrypoint]
        if self.setup_commands:
            entrypoint = self.setup_commands + entrypoint
        entrypoint = " && ".join(entrypoint)
        command = shlex.split(
            f'{shlex.join(self.default_entrypoint)} -c "{entrypoint}"'
        )
        return command

    def run(
        self, entrypoint: str | list[str], timeout: int = None, raises: bool = False,
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
            # Command includes the entrypoint + setup commands
            self.logger.debug(f"Failed to run command: {command} {output}")
            raise ValueError(f"Failed to run command: {entrypoint} ", output)

        output = (stdout + stderr).strip("\r\n").strip("\n")
        self.logger.debug(
            f"Output from terminal with status {process.returncode}: {output}"
        )
        return success, output

    def run_interactive(
        self, entrypoint: str, expected_output: str = "", timeout: int = 30
    ):
        """Run a command in the interactive terminal and return the output.
        Requires a PTY. The terminal stays open after the command is executed.
        """
        if not self.has_pseudo_terminal():
            self.start_pseudo_terminal()
        return self.interact_with_pseudo_terminal(entrypoint, expected_output, timeout)

    @property
    def default_entrypoint(self) -> list[str]:
        """Starts a new bash session exporting the current python executable as 'python'.
        Flags --noprofile and --norc are used to avoid loading any bash profile or rc file,
        which could interfere with the terminal setup (clean outputs)"""
        return shlex.split("/bin/bash --noprofile --norc")

    def clone(self) -> "Terminal":
        return Terminal(
            working_dir=self.working_dir,
            setup_commands=self.setup_commands,
            env_vars=self.env_vars,
            logger=self.logger,
        )

    def has_pseudo_terminal(self):
        return self._master is not None

    def start_pseudo_terminal(self, timeout=300, no_output_timeout=30):
        if self.has_pseudo_terminal():
            self.close_pseudo_terminal()

        self.logger.debug(f"Starting PTY with entrypoint: {self.default_entrypoint}")

        self._master, slave = pty.openpty()

        # set_fd_nonblocking
        flags = fcntl.fcntl(self._master, fcntl.F_GETFL)
        fcntl.fcntl(self._master, fcntl.F_SETFL, flags | os.O_NONBLOCK)

        # Turn off ECHO on the slave side
        attrs = termios.tcgetattr(slave)
        attrs[3] = attrs[3] & ~termios.ECHO  # lflags
        termios.tcsetattr(slave, termios.TCSANOW, attrs)

        process = subprocess.Popen(
            self.default_entrypoint,
            env=self.env_vars,
            cwd=self.working_dir,
            stdin=slave,
            stdout=slave,
            stderr=slave,
            text=True,
            close_fds=True,
            start_new_session=True,
        )

        # close slave, end in the parent process
        os.close(slave)
        atexit.register(self.close_pseudo_terminal)

        initial_output = ""
        commands = " && ".join(self.setup_commands)
        if commands:
            initial_output = self.interact_with_pseudo_terminal(
                commands, timeout=timeout, no_output_timeout=no_output_timeout
            )

        self.logger.debug(f"Initial output from interactive terminal: {initial_output}")

        return initial_output

    def close_pseudo_terminal(self):
        if self._master is not None:
            self.logger.debug("Closing PTY.")
            os.close(self._master)
            self._master = None

    def read_pseudo_terminal_output(
        self,
        expected_output: str = "",
        timeout: int = 300,
        no_output_timeout: int = 60,
        read_length: int = 1024,
    ) -> str:
        """Read from PTY until expected_output is found, timeout is reached,
        or no output change for no_output_timeout seconds.
        """
        output = ""
        start_time = time.time()
        last_change_time = time.time()
        while True:
            if time.time() - start_time > timeout:
                self.logger.debug("Timeout reached while reading from PTY.")
                break
            if time.time() - last_change_time > no_output_timeout:
                self.logger.debug(f"No output change for {no_output_timeout} seconds.")
                break
            try:
                data = os.read(self._master, read_length).decode(
                    "utf-8", errors="ignore"
                )
                if data:
                    output += data
                    last_change_time = time.time()
                    if expected_output and expected_output in output:
                        break
            except BlockingIOError:
                time.sleep(0.1)
                continue
            except OSError as e:
                if e.errno == errno.EIO:
                    self.logger.debug("End of file reached while reading from PTY.")
                    break
                if e.errno != errno.EAGAIN:
                    raise
        return output

    def interact_with_pseudo_terminal(
        self,
        command: str,
        expected_output: str = "",
        timeout: int = 300,
        no_output_timeout: int = 30,
    ):
        self.logger.debug(f"Sending command to interactive terminal: {command}")
        os.write(self._master, command.encode("utf-8") + b"\n")

        output = self.read_pseudo_terminal_output(
            expected_output=expected_output,
            timeout=timeout,
            no_output_timeout=no_output_timeout,
        )

        output = output.strip().strip("\r\n").strip("\n")

        self.logger.debug(f"Output from interactive terminal: {output}")
        return output


class DockerTerminal(Terminal):

    def __init__(
        self,
        working_dir: str = None,
        setup_commands: list[str] = None,
        env_vars: dict[str, str] = None,
        base_image: str = "ubuntu:latest",
        volumes: dict[str, dict[str:str]] = None,
        include_os_env_vars: bool = False,
        **kwargs,
        # TODO: dockerfile and/or docker-compose file?
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
            setup_commands=setup_commands,
            env_vars=env_vars,
            include_os_env_vars=include_os_env_vars,
            **kwargs,
        )
        self.base_image = base_image
        self.volumes = volumes or {}
        self.docker_client = docker.from_env()
        self.host_uid = os.getuid()
        self.host_gid = os.getgid()
        self._patched_image = None
        self._container = None

    # def __del__(self):
    #     self.logger.debug(f"Object destroyed, cleanup container.")
    #     self.clean_up()

    @property
    def working_dir(self):
        """Lazy initialization of working_dir and volume."""
        if self._working_dir is not None:
            self.volumes.pop(self._working_dir, None)
        working_dir = super().working_dir
        self.volumes[working_dir] = {"bind": working_dir, "mode": "rw"}
        # self.volumes[working_dir] = {"bind": "/tmp/code", "mode": "rw"}
        return working_dir

    @working_dir.setter
    def working_dir(self, value):
        if self._working_dir is not None:
            self.volumes.pop(self._working_dir, None)
        self._working_dir = value
        self.volumes[self._working_dir] = {"bind": self._working_dir, "mode": "rw"}
        # self.volumes[self._working_dir] = {"bind": "/tmp/code", "mode": "rw"}

    @property
    def patched_image(self):
        """Lazy initialization of the patched image."""
        if self._patched_image is None:
            self._patched_image = self.patch_base_image(self.base_image)
        return self._patched_image

    @property
    def container(self):
        """Lazy initialization of the container."""
        if self._container is None:
            self._container = self.setup_container()
        return self._container

    @property
    def default_entrypoint(self) -> list[str]:
        """Expects the container to have bash installed and python executable available."""
        return shlex.split(
            f"docker exec -i {self.container.name} /bin/bash --noprofile --norc"
        )

    def prepare_command(self, entrypoint: str | list[str]) -> list[str]:
        """Prepares a shell command by combining setup commands and entrypoint commands.
        Then wraps the command in a shell call."""
        if isinstance(entrypoint, str):
            entrypoint = [entrypoint]
        if self.setup_commands:
            entrypoint = self.setup_commands + entrypoint
        entrypoint = " && ".join(entrypoint)
        command = ["/bin/bash", "-c", entrypoint]
        return command

    def run(
        self, entrypoint: str | list[str], timeout: int = None, raises: bool = False, user: str = None
    ) -> tuple[bool, str]:
        """Run a command in the terminal. Return command status and output."""
        command = self.prepare_command(entrypoint)

        # TODO: docker exec_run timeout?
        status, output = self.container.exec_run(
            command,
            workdir=self.working_dir,
            # workdir="/tmp/code",
            environment=self.env_vars,
            user=f"{self.host_uid}:{self.host_gid}" if user is None else user,
            stdout=True,
            stderr=True,
        )
        success = status == 0

        if raises and not success:
            # Command includes the entrypoint + setup commands
            self.logger.debug(f"Failed to run command: {command} {output}")
            raise ValueError(f"Failed to run command: {entrypoint} ", output)

        self.logger.debug(f"Output from terminal with status {status}: {output}")
        return success, output.decode().strip("\r\n").strip("\n")

    def clone(self) -> "DockerTerminal":
        terminal = DockerTerminal(
            base_image=self.base_image,
            setup_commands=self.setup_commands,
            volumes=self.volumes,
            env_vars=self.env_vars,
            working_dir=self.working_dir,
            logger=self.logger,
        )
        return terminal

    def setup_container(self) -> docker.models.containers.Container:
        # Create and start a container mounting volumes and setting environment variables
        self.logger.debug(f"Setting up container with base image: {self.patched_image}")
        container = self.docker_client.containers.run(
            image=self.patched_image,
            command="sleep infinity",  # Keep the container running
            working_dir=self.working_dir,
            volumes=self.volumes,
            environment=self.env_vars,
            user=f"{self.host_uid}:{self.host_gid}",
            detach=True,
            auto_remove=True,
            remove=True,
        )
        container_name = f"froggy_{container.name}"
        container.rename(container_name)
        container.reload()
        self.logger.debug(f"Container {container_name} started successfully.")
        atexit.register(self.clean_up)
        return container

    def clean_up(self):
        """Clean up the Docker container."""
        if self.container:
            try:
                self.container.stop(timeout=1)
            except docker.errors.NotFound:
                self.logger.debug(
                    f"Container {self.container.name} not found. "
                    "It might have already been removed."
                )
            self._container = None

    def patch_base_image(self, base_image: str) -> str:
        """Patch the base image creating a user and group with
        the same UID and GID as the host. This allows the container
        to write to the host filesystem with the same permissions.
        Inside the container, the user has root privileges."""
        try:
            self.docker_client.images.get(base_image)
        except docker.errors.ImageNotFound:
            self.logger.debug(f"Pulling base image: {base_image}")
            self.docker_client.images.pull(base_image)

        patch_version = "v1"
        dockerfile = f"""
            FROM {base_image}
            # Install sudo
            RUN apt-get update && apt-get install -y sudo
            # Create group with GID if it does not exist
            RUN if ! getent group {self.host_gid} > /dev/null; then \\
                groupadd -g {self.host_gid} froggy_group; \\
            fi
            # Create a user with UID if it does not exist
            RUN useradd -m -u {self.host_uid} -g {self.host_gid} -G sudo froggy_user
            # Allow passwordless sudo for froggy_user
            RUN echo 'froggy_user ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
            """

        image_tag = f"{base_image}-{self.host_uid}-{self.host_gid}-{patch_version}"
        try:
            self.docker_client.images.get(image_tag)
            self.logger.debug(f"Image {image_tag} already exists.")
        except docker.errors.ImageNotFound:
            self.logger.debug(f"Building image {image_tag}.")
            self.docker_client.images.build(
                fileobj=io.BytesIO(dockerfile.encode("utf-8")),
                tag=image_tag,
                rm=True,
                custom_context=False,
            )
        return image_tag


def select_terminal(
    terminal_config: dict | None = None, logger=logging.getLogger("froggy")
) -> Terminal:
    terminal_config = terminal_config or {"type": "local"}
    terminal_type = terminal_config["type"]
    match terminal_type:
        case "docker":
            from froggy.terminal import DockerTerminal as terminal_class
        case "local":
            from froggy.terminal import Terminal as terminal_class
        case _:
            raise ValueError(f"Unknown terminal {terminal_type}")

    return terminal_class(**terminal_config, logger=logger)
