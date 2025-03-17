import atexit
import errno
import fcntl
import os
import pty
import shlex
import subprocess
import tempfile
import termios
import time
import uuid

import docker

from debug_gym.logger import DebugGymLogger
from debug_gym.utils import strip_ansi

DEFAULT_TIMEOUT = 300
DEFAULT_PS1 = "DEBUG_GYM_PS1"
DISABLE_ECHO_COMMAND = "stty -echo"


class ShellSession:

    def __init__(
        self,
        shell_command: str,
        working_dir: str,
        session_commands: list[str] | None = None,
        env_vars: dict[str, str] | None = None,
        logger: DebugGymLogger | None = None,
    ):
        self._session_id = str(uuid.uuid4()).split("-")[0]
        self.shell_command = shell_command
        self.working_dir = working_dir
        self.session_commands = list(session_commands or [])
        self.env_vars = dict(env_vars or {})
        self.logger = logger or DebugGymLogger("debug-gym")
        self.filedescriptor = None
        self.process = None

        # Make sure session can read the output until the given sentinel or PS1
        if not self.env_vars.get("PS1"):
            self.env_vars["PS1"] = DEFAULT_PS1

        self.default_read_until = self.env_vars["PS1"]

        atexit.register(self.close)

    @property
    def is_running(self):
        return self.process is not None and self.process.poll() is None

    def start(self, command=None, read_until=None):
        self.close()  # Close any existing session

        # Prepare entrypoint, combining session commands and command if provided
        # For example: `bin/bash -c "session_command1 && session_command2 && pdb"`
        entrypoint = self.shell_command
        if command:
            command = " && ".join(self.session_commands + [command])
            entrypoint = f'{self.shell_command} -c "{command}"'

        self.logger.debug(f"Starting {self} with entrypoint: {entrypoint}")

        # Prepare the file descriptor
        _server, _client = pty.openpty()
        self.filedescriptor = _server

        # set_fd_nonblocking
        flags = fcntl.fcntl(_server, fcntl.F_GETFL)
        fcntl.fcntl(_server, fcntl.F_SETFL, flags | os.O_NONBLOCK)

        # Turn off ECHO on the _client side
        attrs = termios.tcgetattr(_client)
        attrs[3] = attrs[3] & ~termios.ECHO  # lflags
        termios.tcsetattr(_client, termios.TCSANOW, attrs)

        self.process = subprocess.Popen(
            shlex.split(entrypoint),
            env=self.env_vars,
            cwd=self.working_dir,
            stdin=_client,
            stdout=_client,
            stderr=_client,
            text=True,
            close_fds=True,
            start_new_session=True,
        )

        # close _client, end in the parent process
        os.close(_client)

        # Read the output until the sentinel or PS1
        output = self.read(read_until=read_until)

        # Run session commands after starting the session if command was not provided
        if not command and self.session_commands:
            command = " && ".join(self.session_commands)
            output += self.run(command, read_until)

        return output

    def close(self):
        if self.filedescriptor is not None:
            self.logger.debug(f"Closing {self}.")
            os.close(self.filedescriptor)
            self.filedescriptor = None

        if self.process:
            self.process.terminate()
            self.process = None

    def read(
        self,
        read_until: str | None = None,
        timeout: int | None = None,
        read_length: int = 1024,
    ) -> str:
        """Read from this Shell session until read_until is found, timeout is reached"""
        read_until = read_until or self.default_read_until
        timeout = timeout or DEFAULT_TIMEOUT

        output = ""
        start_time = time.time()
        while True:
            if time.time() - start_time > timeout:
                raise TimeoutError(
                    f"{self}: Read timeout after {timeout} secs. Read so far: {output!r}"
                )

            try:
                data = os.read(self.filedescriptor, read_length).decode(
                    "utf-8", errors="ignore"
                )
                if data:
                    output += data
                    if read_until and read_until in output:
                        break
                else:
                    time.sleep(0.01)
            except BlockingIOError:
                time.sleep(0.1)
                continue
            except OSError as e:
                if e.errno == errno.EIO:
                    self.is_closed = True
                    self.logger.debug("End of file reached while reading from PTY.")
                    break
                if e.errno != errno.EAGAIN:
                    raise

        # Strip out ANSI escape codes.
        output = strip_ansi(output)
        output = output.replace(read_until, "").strip().strip("\r\n")
        return output

    def run(
        self,
        command: str,
        read_until: str | None = None,
        timeout: int | None = None,
    ):
        """Run a command in the Shell session and return the output."""
        output = ""
        if not self.is_running:
            output += self.start()
            self.logger.debug(f"{self}: Initial output: {output!r}")

        self.logger.debug(f"{self}: Running {command!r}")
        os.write(self.filedescriptor, command.encode("utf-8") + b"\n")

        try:
            output += self.read(read_until=read_until, timeout=timeout)
        except TimeoutError as e:
            self.close()
            self.logger.debug(f"{e!r}")
            raise

        self.logger.debug(f"{self}: Output: {output!r}")
        return output

    def __str__(self):
        return f"Shell[{self._session_id}]"

    def __del__(self):
        self.close()


class Terminal:

    def __init__(
        self,
        working_dir: str = None,
        session_commands: list[str] = None,
        env_vars: dict[str, str] = None,
        include_os_env_vars: bool = True,
        logger: DebugGymLogger | None = None,
        **kwargs,
    ):
        self.logger = logger or DebugGymLogger("debug-gym")
        self.session_commands = session_commands or []
        self.env_vars = env_vars or {}
        if include_os_env_vars:
            self.env_vars = self.env_vars | dict(os.environ)
        # Clean up output by disabling terminal prompt and colors
        self.env_vars["NO_COLOR"] = "1"  # disable colors
        self.env_vars["PS1"] = (
            DEFAULT_PS1  # use a sentinel to know when to stop reading
        )
        self._working_dir = working_dir
        self.sessions = []

    @property
    def working_dir(self):
        """Lazy initialization of the working directory."""
        if self._working_dir is None:
            temp_dir = tempfile.TemporaryDirectory(prefix="Terminal-")
            atexit.register(temp_dir.cleanup)
            self._working_dir = temp_dir.name
            self.logger.debug(f"Using temporary working directory: {self._working_dir}")
        return self._working_dir

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
        command = shlex.split(f'{self.default_shell_command} -c "{entrypoint}"')
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
            # Command includes the entrypoint + session commands
            self.logger.debug(f"Failed to run command: {command} {output}")
            raise ValueError(f"Failed to run command: {entrypoint} ", output)

        output = (stdout + stderr).strip("\r\n").strip("\n")
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


class DockerTerminal(Terminal):

    def __init__(
        self,
        working_dir: str | None = None,
        session_commands: list[str] | None = None,
        env_vars: dict[str, str] | None = None,
        include_os_env_vars: bool = False,
        logger: DebugGymLogger | None = None,
        # Docker-specific parameters
        base_image: str = "ubuntu:latest",
        setup_commands: list[str] | None = None,
        volumes: dict[str, dict[str:str]] | None = None,
        map_host_uid_gid: bool = True,
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
            session_commands=session_commands,
            env_vars=env_vars,
            include_os_env_vars=include_os_env_vars,
            logger=logger,
            **kwargs,
        )
        self.base_image = base_image
        self.setup_commands = setup_commands or []
        self.volumes = volumes or {}
        self.map_host_uid_gid = map_host_uid_gid
        self.docker_client = docker.from_env()
        self.host_uid = os.getuid()
        self.host_gid = os.getgid()
        self._container = None

    def user_map(self):
        _user = ""
        if self.map_host_uid_gid:
            _user = f"{self.host_uid}:{self.host_gid}"
        return _user

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
    def default_shell_command(self) -> list[str]:
        """Expects the container to have bash installed and python executable available."""
        user_map = self.user_map()
        if user_map:
            user_map = f"--user {user_map}"
        entrypoint = f"docker exec -t -i {user_map} {self.container.name} /bin/bash --noprofile --norc"
        return entrypoint

    def new_shell_session(self):
        session = ShellSession(
            shell_command=self.default_shell_command,
            session_commands=[DISABLE_ECHO_COMMAND] + self.session_commands,
            working_dir=self.working_dir,
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
            user=self.user_map() if user is None else user,
            stdout=True,
            stderr=True,
        )
        success = status == 0

        if raises and not success:
            # Command includes the entrypoint + session commands
            self.logger.debug(f"Failed to run command: {command} {output}")
            raise ValueError(f"Failed to run command: {entrypoint} ", output)

        self.logger.debug(
            f"Output from terminal with status {status}:\n{output.decode()}"
        )
        return success, output.decode().strip("\r\n").strip("\n")

    def setup_container(self) -> docker.models.containers.Container:
        # Create and start a container mounting volumes and setting environment variables
        self.logger.debug(f"Setting up container with base image: {self.base_image}")
        container = self.docker_client.containers.run(
            image=self.base_image,
            command="sleep infinity",  # Keep the container running
            working_dir=self.working_dir,
            volumes=self.volumes,
            environment=self.env_vars,
            user=self.user_map(),
            detach=True,
            auto_remove=True,
            remove=True,
            tty=True,
            stdin_open=True,
        )
        container_name = f"debug_gym_{container.name}"
        container.rename(container_name)
        container.reload()
        self._run_setup_commands(container)
        self.logger.debug(f"Container {container_name} started successfully.")
        atexit.register(self.clean_up)
        return container

    def _run_setup_commands(self, container):
        """Run setup commands if any. If commands fail, stop the container."""
        if self.setup_commands:
            setup_commands = " && ".join(self.setup_commands)
            self.logger.debug(f"Running setup commands: {setup_commands}")
            status, output = container.exec_run(
                ["/bin/bash", "-c", setup_commands],
                user="root",  # Run as root to allow installations
                workdir=self.working_dir,
                environment=self.env_vars,
            )
            if status != 0:
                container.stop()
                raise ValueError(
                    f"Failed to run setup command: {setup_commands}\n"
                    f"Output: {output.decode()}"
                )
            self.logger.debug(f"Setup commands ran successfully.")

    def clean_up(self):
        """Clean up the Docker container."""
        if self._container is not None:
            try:
                self.container.stop(timeout=1)
            except docker.errors.NotFound:
                self.logger.debug(
                    f"Container {self.container.name} not found. "
                    "It might have already been removed."
                )
            self._container = None

    def close(self):
        super().close()
        self.clean_up()


def select_terminal(
    terminal_config: dict | None = None, logger: DebugGymLogger | None = None
) -> Terminal:
    logger = logger or DebugGymLogger("debug-gym")
    terminal_config = terminal_config or {"type": "local"}
    terminal_type = terminal_config["type"]
    docker_only = ["base_image", "setup_commands", "volumes", "map_host_uid_gid"]
    match terminal_type:
        case "docker":
            terminal_class = DockerTerminal
        case "local":
            terminal_class = Terminal
            if any(cfg in terminal_config for cfg in docker_only):
                logger.warning("Ignoring Docker-only parameters for local terminal.")
        case _:
            raise ValueError(f"Unknown terminal {terminal_type}")

    return terminal_class(**terminal_config, logger=logger)
