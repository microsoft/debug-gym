import atexit
import errno
import fcntl
import io
import logging
import os
import pty
import random
import shlex
import string
import subprocess
import sys
import tarfile
import termios
import time

import docker

logger = logging.getLogger("froggy")


class Terminal:

    def __init__(
        self,
        working_dir: str = None,
        setup_commands: list[str] = None,
        env_vars: dict[str, str] = None,
    ):
        if working_dir is None:
            working_dir = "/tmp/Froggy"
            os.makedirs(working_dir, exist_ok=True)
        self.setup_commands = setup_commands if setup_commands else []
        self.env_vars = env_vars if env_vars else {}
        # Clean up output by disabling terminal prompt and colors
        self.env_vars["NO_COLOR"] = "1"  # disable colors
        self.env_vars["PS1"] = ""  # disable prompt
        self.working_dir = working_dir
        self._master = None  # PTY master file descriptor

    @property
    def path_env(self):
        # TODO: find a better way to set PATH
        return {
            "PATH": f"{os.path.dirname(sys.executable)}:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
        }

    def prepare_command(self, entrypoint: list[str]) -> list[str]:
        """Prepares a shell command by combining setup commands and entrypoint commands.
        Then wraps the command in a shell (self.default_entrypoint) call."""
        if self.setup_commands:
            entrypoint = " && ".join(self.setup_commands + entrypoint)
        else:
            entrypoint = " && ".join(entrypoint)
        command = shlex.split(
            f'{shlex.join(self.default_entrypoint)} -c "{entrypoint}"'
        )
        return command

    def run(self, entrypoint: list[str], working_dir=None, timeout=None):
        """Run a command in the terminal. Return command status and output."""
        command = self.prepare_command(entrypoint)
        logger.debug(f"Running command in terminal: {command}")
        process = subprocess.Popen(
            command,
            env=self.env_vars | self.path_env,
            cwd=working_dir or self.working_dir,
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
        output = (stdout + stderr).strip("\r\n").strip("\n")
        logger.debug(f"Output from terminal with status {process.returncode}: {output}")
        return success, output

    def run_interactive(
        self, entrypoint: str, expected_output: str = "", timeout: int = 30
    ):
        """Run a command in the interactive terminal and return the output.
        Requires a PTY. The terminal stays open after the command is executed.
        """
        # TODO: include working_dir parameter to align with run()?
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
        return self.__class__(
            working_dir=self.working_dir,
            setup_commands=self.setup_commands,
            env_vars=self.env_vars,
        )

    def has_pseudo_terminal(self):
        return self._master is not None

    def start_pseudo_terminal(self, timeout=120):
        if self.has_pseudo_terminal():
            self.close_pseudo_terminal()

        logger.debug(f"Starting PTY with entrypoint: {self.default_entrypoint}")

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
            env=self.env_vars | self.path_env,
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
            initial_output = self.interact_with_pseudo_terminal(commands, timeout=timeout)

        logger.debug(f"Initial output from interactive terminal: {initial_output}")

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

        logger.debug(f"Sending command to interactive terminal: {command}")
        os.write(self._master, command.encode("utf-8") + b"\n")
        # get output back:
        output = self.read_pseudo_terminal_output(
            expected_output=expected_output, timeout=timeout
        )

        output = output.strip().strip("\r\n").strip("\n")

        logger.debug(f"Output from interactive terminal: {output}")
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
        self.docker_client = docker.from_env()
        self.host_uid = os.getuid()
        self.host_gid = os.getgid()
        self.patched_image = self.patch_base_image(base_image)
        self.container = self.setup_container()

    @property
    def default_entrypoint(self) -> list[str]:
        """Expects the container to have bash installed and python executable available."""
        return shlex.split(
            f"docker exec -i {self.container.name} /bin/bash --noprofile --norc"
        )

    @property
    def path_env(self):
        return {}

    def run(self, entrypoint, working_dir=None, timeout=None):
        """Run a command in the terminal. Return command status and output."""

        command = [entrypoint] if isinstance(entrypoint, str) else entrypoint

        if self.setup_commands:
            command = self.setup_commands + ["&&"] + entrypoint

        command = " ".join(command)

        # TODO: docker exec_run timeout?
        command = f'/bin/bash -c "{command}"'
        status, output = self.container.exec_run(
            command,
            workdir=working_dir or self.working_dir,
            environment=self.env_vars,
            stdout=True,
            stderr=True,
        )
        success = status == 0
        return success, output.decode().strip("\r\n").strip("\n")

    def clone(self) -> Terminal:
        terminal = self.__class__(
            base_image=self.base_image,
            setup_commands=self.setup_commands,
            volumes=self.volumes,
            env_vars=self.env_vars,
            working_dir=self.working_dir,
        )
        return terminal

    def setup_container(self) -> docker.models.containers.Container:
        # Create and start a container mounting volumes and setting environment variables

        suffix = "".join(random.choices(string.ascii_lowercase, k=8))
        container_name = f"froggy-container-{suffix}"
        logger.debug(
            f"Setting up container: {container_name} "
            f"with base image: {self.patched_image}"
        )
        container = self.docker_client.containers.run(
            image=self.patched_image,
            command="sleep infinity",  # Keep the container running
            working_dir=self.working_dir,
            volumes=self.volumes,
            environment=self.env_vars,
            name=container_name,
            user=f"{self.host_uid}:{self.host_gid}",
            detach=True,
            auto_remove=True,
            remove=True,
        )
        atexit.register(self.clean_up)
        logger.debug("Container setup complete")
        return container

    def clean_up(self):
        if self.container:
            logger.debug(f"Cleaning up container: {self.container.name}")
            self.container.stop()

    def patch_base_image(self, base_image: str) -> str:
        """Patch the base image creating a user and group with
        the same UID and GID as the host. This allows the container
        to write to the host filesystem with the same permissions.
        Inside the container, the user has root privileges."""
        try:
            self.docker_client.images.get(base_image)
        except docker.errors.ImageNotFound:
            logger.debug(f"Pulling base image: {base_image}")
            self.docker_client.images.pull(base_image)

        dockerfile = f"""
            FROM {base_image}
            # Ensure a group with GID exists; create it if necessary
            RUN if ! getent group 100 > /dev/null; then \\
                groupadd -g {self.host_gid} froggy_group; \\
                fi && \\
                # Create the user with UID, assign to GID, and add to root group, -m to create home dir
                useradd -m -u {self.host_uid} -g {self.host_gid} -G sudo froggy_user
            """

        image_tag = f"{base_image}-{self.host_uid}-{self.host_gid}"
        try:
            self.docker_client.images.get(image_tag)
            logger.debug(f"Image {image_tag} already exists.")
        except docker.errors.ImageNotFound:
            logger.debug(f"Building image {image_tag}.")
            dockerfile_bytes = dockerfile.encode("utf-8")
            context = io.BytesIO()
            with tarfile.open(fileobj=context, mode="w") as tar:
                tarinfo = tarfile.TarInfo("Dockerfile")
                tarinfo.size = len(dockerfile_bytes)
                tar.addfile(tarinfo, io.BytesIO(dockerfile_bytes))
            context.seek(0)
            self.docker_client.images.build(
                fileobj=context, custom_context=True, tag=image_tag, rm=True
            )
        return image_tag
