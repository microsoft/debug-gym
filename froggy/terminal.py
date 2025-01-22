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
import uuid

import docker

class ShellSession:
    def __init__(self, filedescriptor, session_id=None, logger=logging.getLogger("froggy")):
        self.filedescriptor = filedescriptor
        self.session_id = session_id or str(uuid.uuid4())
        self.logger = logger
        self._terminal = None
        atexit.register(self.close)

    def close(self):
        if self.filedescriptor is not None:
            self.logger.debug(f"Closing {self}.")
            os.close(self.filedescriptor)
            self.filedescriptor = None

            if self._terminal is not None:
                self._terminal.sessions.remove(self)

    def read(
        self,
        read_until: str = "",
        timeout: int = 300,
        no_output_timeout: int = 60,
        read_length: int = 1024,
    ) -> str:
        """Read from this Shell session until read_until is found, timeout is reached,
        or no output change for no_output_timeout seconds.
        """
        output = ""
        start_time = time.time()
        last_change_time = time.time()
        while True:
            if time.time() - start_time > timeout:
                self.logger.debug(f"Timeout reached while reading from {self}.")
                break
            if time.time() - last_change_time > no_output_timeout:
                self.logger.debug(f"No output change for {no_output_timeout} seconds.")
                break
            try:
                data = os.read(self.filedescriptor, read_length).decode(
                    "utf-8", errors="ignore"
                )
                if data:
                    output += data
                    last_change_time = time.time()
                    if read_until and read_until in output:
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

    def run(
        self,
        command: str,
        read_until: str = "",
        timeout: int = 300,
        no_output_timeout: int = 30,
    ):
        """Run a command in the Shell session and return the output."""
        self.logger.debug(f"Sending command to {self}: {command}")
        os.write(self.filedescriptor, command.encode("utf-8") + b"\n")

        output = self.read(
            read_until=read_until,
            timeout=timeout,
            no_output_timeout=no_output_timeout,
        )

        output = output.strip().strip("\r\n").strip("\n")

        self.logger.debug(f"Output from {self}: {output}")
        return output

    def __str__(self):
        return f"ShellSession {self.session_id}"


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
        # self._master = None  # PTY master file descriptor
        self.logger = logger
        self.sessions = []

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
        self,
        entrypoint: str | list[str],
        timeout: int = None,
        raises: bool = False,
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

    # def run_interactive(
    #     self, entrypoint: str, expected_output: str = "", timeout: int = 30
    # ):
    #     """Run a command in the interactive terminal and return the output.
    #     Requires a PTY. The terminal stays open after the command is executed.
    #     """
    #     if not self.has_pseudo_terminal():
    #         self.start_pseudo_terminal()
    #     return self.interact_with_pseudo_terminal(entrypoint, expected_output, timeout)

    @property
    def default_entrypoint(self) -> list[str]:
        """Starts a new bash session exporting the current python executable as 'python'.
        Flags --noprofile and --norc are used to avoid loading any bash profile or rc file,
        which could interfere with the terminal setup (clean outputs)"""
        return shlex.split("/bin/bash --noprofile --norc")

    # def clone(self) -> "Terminal":
    #     return Terminal(
    #         working_dir=self.working_dir,
    #         setup_commands=self.setup_commands,
    #         env_vars=self.env_vars,
    #         logger=self.logger,
    #     )

    # def has_pseudo_terminal(self):
    #     return self._master is not None

    def start_shell_session(self, timeout=30, no_output_timeout=0.1):
        self.logger.debug(f"Starting ShellSession with entrypoint: {self.default_entrypoint}")

        master, slave = pty.openpty()

        # set_fd_nonblocking
        flags = fcntl.fcntl(master, fcntl.F_GETFL)
        fcntl.fcntl(master, fcntl.F_SETFL, flags | os.O_NONBLOCK)

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
        #atexit.register(self.close_pseudo_terminal)
        session = ShellSession(master, logger=self.logger)
        self.sessions.append(session)

        initial_output = ""
        commands = " && ".join(self.setup_commands)
        if commands:
            # initial_output = self.interact_with_pseudo_terminal(
            #     commands, timeout=timeout, no_output_timeout=no_output_timeout
            # )
            initial_output = session.run(commands, timeout=timeout, no_output_timeout=no_output_timeout)

        self.logger.debug(f"Initial output from {session}: {initial_output}")

        return session

    # def close_pseudo_terminal(self):
    #     if self._master is not None:
    #         self.logger.debug("Closing PTY.")
    #         os.close(self._master)
    #         self._master = None

    # def read_pseudo_terminal_output(
    #     self,
    #     expected_output: str = "",
    #     timeout: int = 300,
    #     no_output_timeout: int = 60,
    #     read_length: int = 1024,
    # ) -> str:
    #     """Read from PTY until expected_output is found, timeout is reached,
    #     or no output change for no_output_timeout seconds.
    #     """
    #     output = ""
    #     start_time = time.time()
    #     last_change_time = time.time()
    #     while True:
    #         if time.time() - start_time > timeout:
    #             self.logger.debug("Timeout reached while reading from PTY.")
    #             break
    #         if time.time() - last_change_time > no_output_timeout:
    #             self.logger.debug(f"No output change for {no_output_timeout} seconds.")
    #             break
    #         try:
    #             data = os.read(self._master, read_length).decode(
    #                 "utf-8", errors="ignore"
    #             )
    #             if data:
    #                 output += data
    #                 last_change_time = time.time()
    #                 if expected_output and expected_output in output:
    #                     break
    #         except BlockingIOError:
    #             time.sleep(0.1)
    #             continue
    #         except OSError as e:
    #             if e.errno == errno.EIO:
    #                 self.logger.debug("End of file reached while reading from PTY.")
    #                 break
    #             if e.errno != errno.EAGAIN:
    #                 raise
    #     return output

    # def interact_with_pseudo_terminal(
    #     self,
    #     command: str,
    #     expected_output: str = "",
    #     timeout: int = 300,
    #     no_output_timeout: int = 30,
    # ):
    #     self.logger.debug(f"Sending command to interactive terminal: {command}")
    #     os.write(self._master, command.encode("utf-8") + b"\n")

    #     output = self.read_pseudo_terminal_output(
    #         expected_output=expected_output,
    #         timeout=timeout,
    #         no_output_timeout=no_output_timeout,
    #     )

    #     output = output.strip().strip("\r\n").strip("\n")

    #     self.logger.debug(f"Output from interactive terminal: {output}")
    #     return output


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
        self._container = None

    @property
    def working_dir(self):
        """Lazy initialization of working_dir and volume."""
        if self._working_dir is not None:
            self.volumes.pop(self._working_dir, None)
        working_dir = super().working_dir
        self.volumes[working_dir] = {"bind": working_dir, "mode": "rw"}
        return working_dir

    @working_dir.setter
    def working_dir(self, value):
        if self._working_dir is not None:
            self.volumes.pop(self._working_dir, None)
        self._working_dir = value
        self.volumes[self._working_dir] = {"bind": self._working_dir, "mode": "rw"}

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
            f"docker exec -i --user {self.host_uid}:{self.host_gid} {self.container.name} /bin/bash --noprofile --norc"
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
        self,
        entrypoint: str | list[str],
        timeout: int = None,
        raises: bool = False,
        user: str = None,
    ) -> tuple[bool, str]:
        """Run a command in the terminal. Return command status and output."""
        command = self.prepare_command(entrypoint)

        self.logger.debug(f"Exec run: {command}")

        # TODO: docker exec_run timeout?
        status, output = self.container.exec_run(
            command,
            workdir=self.working_dir,
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

    # def clone(self) -> "DockerTerminal":
    #     terminal = DockerTerminal(
    #         base_image=self.base_image,
    #         setup_commands=self.setup_commands,
    #         volumes=self.volumes,
    #         env_vars=self.env_vars,
    #         working_dir=self.working_dir,
    #         logger=self.logger,
    #     )
    #     return terminal

    def setup_container(self) -> docker.models.containers.Container:
        # Create and start a container mounting volumes and setting environment variables
        self.logger.debug(f"Setting up container with base image: {self.base_image}")
        container = self.docker_client.containers.run(
            image=self.base_image,
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
