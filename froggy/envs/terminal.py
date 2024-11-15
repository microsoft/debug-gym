import atexit
import errno
import fcntl
import logging
import os
import pty
import random
import string
import subprocess
import time

import docker

logger = logging.getLogger("froggy")


class Terminal:

    def __init__(
        self,
        working_dir: str = "/tmp/Froggy",
        setup_commands: list[str] = None,
        env_vars: dict[str, str] = None,
    ):
        self._setup_commands = setup_commands if setup_commands else []
        self.env_vars = env_vars if env_vars else {}
        # self.env_vars["NO_COLOR"] = "1"
        self.working_dir = working_dir
        self._master = None  # PTY master file descriptor

    def run(self, entrypoint, working_dir=None):
        """Run a command in the terminal. Return command status and output."""

        command = [entrypoint] if isinstance(entrypoint, str) else entrypoint

        if self.setup_commands:
            command = self.setup_commands + ["&&"] + entrypoint

        logger.debug(f"Running command in terminal: {command}\n")
        process = subprocess.Popen(
            command,
            env=dict(os.environ, NO_COLOR="1"),
            cwd=working_dir or self.working_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        stdout, stderr = process.communicate()
        output = stdout + stderr
        success = process.returncode == 0
        logger.debug(
            f"Output from terminal with status {process.returncode}: {output}\n"
        )
        return success, output

    def run_interactive(
        self, entrypoint: str, expected_output: str = "", timeout: int = 30
    ):
        """Run a command in the interactive terminal and return the output.
        Requires a PTY. The terminal stays open after the command is executed.
        """
        # TODO: include working_dir parameter to align with run()?
        if not self.has_pseudo_terminal():
            raise ValueError(
                "Interactive terminal not available. Please start the terminal first."
            )
        return self.interact_with_pseudo_terminal(entrypoint, expected_output, timeout)

    @property
    def default_entrypoint(self) -> list[str]:
        return ["/bin/bash"]

    @property
    def setup_commands(self):
        return self._setup_commands

    def clone(self) -> "Terminal":
        return self.__class__(
            working_dir=self.working_dir,
            setup_commands=self.setup_commands,
            env_vars=self.env_vars,
        )

    def _set_fd_nonblocking(self):
        flags = fcntl.fcntl(self._master, fcntl.F_GETFL)
        fcntl.fcntl(self._master, fcntl.F_SETFL, flags | os.O_NONBLOCK)

    def has_pseudo_terminal(self):
        return self._master is not None

    def start_pseudo_terminal(self, timeout=120):
        if self.has_pseudo_terminal():
            self.close_pseudo_terminal()

        logger.debug("Starting PTY.")
        # _env = os.environ.copy()  # TODO: use self.env_vars
        # _env["NO_COLOR"] = "1"

        self._master, slave = pty.openpty()
        self._set_fd_nonblocking()

        process = subprocess.Popen(
            self.default_entrypoint,
            env=self.env_vars,
            cwd=self.working_dir,
            stdin=slave,
            stdout=slave,
            stderr=slave,
            text=True,
            close_fds=True,
        )

        # close slave, end in the parent process
        os.close(slave)
        atexit.register(self.close_pseudo_terminal)

        commands = " && ".join(self.setup_commands)
        initial_output = self.interact_with_pseudo_terminal(commands, timeout=timeout)
        logger.debug(f"Initial output from interactive terminal: {initial_output}\n")

        return initial_output

    def close_pseudo_terminal(self):
        if self._master is not None:
            logger.debug("Closing PTY.")
            os.close(self._master)
            self._master = None

    def read_pseudo_terminal_output(
        self, expected_output="", timeout=30, read_length=1024
    ):
        """Read from PTY until expected_output is found.
        If no expected_output is provided, read for timeout seconds.
        """
        output = ""
        start_time = time.time()
        while True:
            if not expected_output and time.time() - start_time > timeout:
                logger.debug("Timeout reached while reading from PTY.")
                break
            try:
                # [TODO] 1024? We might need more than that.
                data = os.read(self._master, read_length).decode(
                    "utf-8", errors="ignore"
                )
                if data:
                    output += data
                    if expected_output and expected_output in output:
                        break

            except BlockingIOError:
                time.sleep(0.1)
                continue

            except OSError as e:
                if e.errno == errno.EIO:
                    # end of file
                    logger.debug("End of file reached while reading from PTY.")
                    break

                if e.errno != errno.EAGAIN:
                    raise

        return output

    def interact_with_pseudo_terminal(
        self, command: str, expected_output: str = "", timeout: int = 30
    ):
        if not isinstance(command, str):
            command = " ".join(command)

        logger.debug(f"Sending command to interactive terminal: {command}\n")
        os.write(self._master, command.encode("utf-8") + b"\n")
        # get output back:
        output = self.read_pseudo_terminal_output(
            expected_output=expected_output, timeout=timeout
        )

        # when success, the output always repeats the command, we can remove it
        output = output.strip()
        if output.startswith(command):
            output = output[len(command) :].strip("\r\n")

        logger.debug(f"Output from interactive terminal: {output}\n")
        return output


class DockerTerminal(Terminal):

    def __init__(
        self,
        working_dir: str = "/",
        setup_commands: list[str] = None,
        env_vars: dict[str, str] = None,
        base_image: str = "ubuntu:latest",
        volumes: dict[str, dict[str:str]] = None,
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
        )
        self.base_image = base_image
        self.volumes = volumes if volumes else {}
        self.container = None
        self.docker_client = docker.from_env()
        self.setup_container()

    @property
    def default_entrypoint(self) -> list[str]:
        return f"docker exec -i {self.container.name} /bin/bash".split()

    def run(self, entrypoint, working_dir=None):
        """Run a command in the terminal. Return command status and output."""

        command = [entrypoint] if isinstance(entrypoint, str) else entrypoint

        if self.setup_commands:
            command = self.setup_commands + ["&&"] + entrypoint

        command = " ".join(command)

        command = f'/bin/bash -c "{command}"'
        status, output = self.container.exec_run(
            command,
            workdir=working_dir or self.working_dir,
            environment=self.env_vars,
            stdout=True,
            stderr=True,
        )
        success = status == 0
        return success, output.decode()

    def run_interactive(
        self, entrypoint: str, expected_output: str = "", timeout: int = 30
    ):
        if not self.has_pseudo_terminal():
            raise ValueError(
                "Interactive terminal not available. Please start the terminal first."
            )
        return self.interact_with_pseudo_terminal(entrypoint, expected_output, timeout)

    def clone(self) -> Terminal:
        terminal = self.__class__(
            base_image=self.base_image,
            setup_commands=self.setup_commands,
            volumes=self.volumes,
            env_vars=self.env_vars,
            working_dir=self.working_dir,
        )
        return terminal

    def setup_container(self):
        # Create and start a container mounting volumes and setting environment variables
        suffix = "".join(random.choices(string.ascii_lowercase, k=8))
        container_name = f"froggy-container-{suffix}"
        logger.debug(
            f"Setting up container: {container_name} "
            f"with base image: {self.base_image}"
        )
        try:
            self.docker_client.images.get(self.base_image)
        except docker.errors.ImageNotFound:
            logger.debug(f"Pulling base image: {self.base_image}")
            self.docker_client.images.pull(self.base_image)

        self.container = self.docker_client.containers.run(
            image=self.base_image,
            command="sleep infinity",  # Keep the container running
            working_dir=self.working_dir,
            volumes=self.volumes,
            environment=self.env_vars,
            name=container_name,
            detach=True,
            auto_remove=True,
            remove=True,
        )
        atexit.register(self.clean_up)
        logger.debug("Container setup complete")

    def clean_up(self):
        if self.container:
            logger.debug(f"Cleaning up container: {self.container.name}")
            self.container.stop()
