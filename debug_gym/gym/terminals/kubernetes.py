import atexit
import json
import os
import subprocess
import time
import uuid
from pathlib import Path

from debug_gym.gym.terminals.shell_session import ShellSession
from debug_gym.gym.terminals.terminal import DISABLE_ECHO_COMMAND, Terminal
from debug_gym.logger import DebugGymLogger
from kubernetes import client, config, stream
from kubernetes.client.rest import ApiException
from kubernetes.stream.ws_client import ERROR_CHANNEL


def _clean_pod_name(name: str) -> str:
    """Clean pod name to conform to Kubernetes naming conventions."""
    # Replace any non-alphanumeric characters with hyphens and convert to lowercase.
    # regex used for validation is
    regex = r"[a-z0-9]([-a-z0-9]*[a-z0-9])?(\\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*"
    # replace any characters not in the regex with hyphens
    cleaned = "".join(c if c.isalnum() or c in "-." else "-" for c in name).lower()
    # ensure it starts and ends with alphanumeric character
    cleaned = cleaned.strip("-").strip(".")
    # truncate to 253 characters
    return cleaned[:253]


class KubernetesTerminal(Terminal):

    def __init__(
        self,
        working_dir: str | None = None,
        session_commands: list[str] | None = None,
        env_vars: dict[str, str] | None = None,
        include_os_env_vars: bool = False,
        logger: DebugGymLogger | None = None,
        # Kubernetes-specific parameters
        base_image: str = "ubuntu:latest",
        setup_commands: list[str] | None = None,
        namespace: str = "default",
        kube_config: str | None = None,
        **kwargs,
    ):
        """
        Kubernetes Terminal that manages a pod instead of a Docker container.

        Args:
            base_image: The container image to use for the pod
            setup_commands: Commands to run after pod creation
            namespace: Kubernetes namespace to create the pod in
            pod_name: Custom pod name (auto-generated if None)
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
        self._task_name = base_image
        self.setup_commands = setup_commands or []
        self.namespace = namespace
        self._pod_name = None

        # Initialize Kubernetes client
        self.kube_config = kube_config or os.getenv(
            "KUBECONFIG", str(Path.home() / ".kube" / "config")
        )  # TODO: debugging
        if self.kube_config:
            config.load_kube_config(self.kube_config)
        else:
            config.load_incluster_config()

        self.k8s_client = client.CoreV1Api()

    @property
    def task_name(self):
        return self._task_name

    @task_name.setter
    def task_name(self, value):
        if self._pod_name is not None:
            raise ValueError("Cannot change task_name while pod is running.")

        self._task_name = value

    @property
    def working_dir(self):
        """Lazy initialization of the working directory."""
        return super().working_dir

    @working_dir.setter
    def working_dir(self, value):
        if self._pod_name:
            raise ValueError("Cannot change working directory while pod is running.")

        self._working_dir = value

    @property
    def pod_name(self):
        """Lazy initialization of the pod."""
        if self._pod_name is None:
            self.setup_pod()

        return self._pod_name

    @property
    def default_shell_command(self) -> list[str]:
        """Expects the pod to have bash installed and python executable available."""
        # TODO: if self.kubeconfig is None, remove the --kubeconfig argument
        entrypoint = [
            "kubectl",
            "--kubeconfig",
            self.kube_config,
            "exec",
            "-it",
            f"{self.pod_name}",
            "-n",
            self.namespace,
            "--",
            "/bin/bash",
            "--noprofile",
            "--norc",
        ]
        return entrypoint

    def new_shell_session(self):
        if not self.is_running():
            raise ValueError("Pod is not running. Cannot create shell session.")

        session = ShellSession(
            shell_command=" ".join(self.default_shell_command),
            session_commands=[DISABLE_ECHO_COMMAND] + self.session_commands,
            working_dir=".",
            env_vars=self.env_vars,
            logger=self.logger,
        )
        self.sessions.append(session)
        return session

    def run(
        self,
        entrypoint: str | list[str],
        timeout: int = None,
        raises: bool = False,
        strip_output: bool = True,
    ) -> tuple[bool, str]:
        """Run a command in the pod. Return command status and output."""
        if isinstance(entrypoint, str):
            entrypoint = [entrypoint]
        if self.session_commands:
            entrypoint = self.session_commands + entrypoint
        entrypoint_str = " && ".join(entrypoint)

        if not self.is_running():
            raise ValueError("Pod is not running. Cannot run commands.")

        self.logger.debug(f"[{self.pod_name}] Kubernetes exec run: {entrypoint_str}")

        try:
            # Set environment variables by prefixing the command
            env_prefix = ""
            if self.env_vars:
                env_vars_str = " ".join(
                    [f'{k}="{v}"' for k, v in self.env_vars.items()]
                )
                env_prefix = f"export {env_vars_str} && "

            # Build the full command with environment variables and working directory
            full_command = entrypoint_str
            if self.working_dir and self.working_dir != "/":
                full_command = f"cd {self.working_dir} && {env_prefix}{full_command}"
            elif env_prefix:
                full_command = f"{env_prefix}{full_command}"

            # Execute command using Kubernetes stream API
            resp = stream.stream(
                self.k8s_client.connect_get_namespaced_pod_exec,
                name=self.pod_name,
                namespace=self.namespace,
                command=["/bin/bash", "-c", full_command],
                stderr=True,
                stdin=False,
                stdout=True,
                tty=False,
                _preload_content=False,
            )

            output = ""
            while resp.is_open():
                resp.update(timeout=1)
                if resp.peek_stdout():
                    output += resp.read_stdout()
                if resp.peek_stderr():
                    output += resp.read_stderr()

            # Get the exit code
            error_channel = resp.read_channel(ERROR_CHANNEL)  # Error channel
            self.logger.debug(f"[{self.pod_name}] error channel: {error_channel}")
            status = json.loads(error_channel)
            success = status["status"] == "Success"

        except Exception as e:
            success = False
            output = f"Command execution failed: {str(e)}"

        if strip_output:
            output = output.strip("\r\n").strip("\n")

        if raises and not success:
            self.logger.debug(f"Failed to run command `{entrypoint_str}`:\n{output}")
            raise ValueError(f"Failed to run command `{entrypoint}`:\n{output}")

        self.logger.debug(f"Output from pod with success `{success}`:\n{output}")
        return success, output

    def setup_pod(self) -> None:
        """Create and start a Kubernetes pod."""

        self._pod_name = _clean_pod_name(
            f"debug-gym.{self.task_name}.{str(uuid.uuid4())[:8]}"
        )
        self.logger.debug(
            f"Setting up pod {self.pod_name} with base image: {self.base_image}"
        )

        # Create pod specification using Kubernetes Python client objects
        pod_spec = client.V1PodSpec(
            containers=[
                client.V1Container(
                    name="main",
                    # image=f"debuggymacr.azurecr.io/{self.base_image}",
                    image=f"{self.base_image}",
                    image_pull_policy="IfNotPresent",
                    command=["/bin/bash"],
                    args=["-c", "sleep infinity"],
                    working_dir=self.working_dir or "/",
                    env=[
                        client.V1EnvVar(name=k, value=v)
                        for k, v in self.env_vars.items()
                    ],
                    resources=client.V1ResourceRequirements(
                        limits={"memory": "16Gi", "cpu": "2"},
                        requests={"memory": "1Gi", "cpu": "0.5"},
                    ),
                    stdin=True,
                    stdin_once=False,
                    tty=True,
                )
            ],
            restart_policy="Never",
            image_pull_secrets=[{"name": "dockerhub-pro"}],
            tolerations=[
                client.V1Toleration(
                    key="kubernetes.azure.com/scalesetpriority",
                    operator="Equal",
                    value="spot",
                    effect="NoSchedule",
                ),
                client.V1Toleration(
                    key="CriticalAddonsOnly",
                    operator="Equal",
                    value="true",
                    effect="NoSchedule",
                ),
            ],
        )

        pod_metadata = client.V1ObjectMeta(
            name=self.pod_name,
            namespace=self.namespace,
            labels={"app": "debug-gym", "component": "terminal"},
        )

        pod_body = client.V1Pod(
            api_version="v1", kind="Pod", metadata=pod_metadata, spec=pod_spec
        )

        try:
            # Create the pod
            self.k8s_client.create_namespaced_pod(
                namespace=self.namespace, body=pod_body
            )

            # Wait for pod to be ready
            self._wait_for_pod_ready()

            # Run setup commands
            self._run_setup_commands()

            self.logger.debug(f"Pod {self.pod_name} started successfully.")
            atexit.register(self.clean_up)

        except ApiException as e:
            raise ValueError(f"Failed to create pod: {e}")

    def _wait_for_pod_ready(self, timeout: int = 3600 * 2):
        """Wait for the pod to be in Running state."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                pod = self.k8s_client.read_namespaced_pod(
                    name=self.pod_name, namespace=self.namespace
                )

                if pod.status.phase == "Running":
                    return
                elif pod.status.phase in ["Failed", "Unknown"]:
                    raise ValueError(
                        f"Pod {self.pod_name} is in {pod.status.phase} state."
                    )
                elif pod.status.phase == "Pending":
                    self.logger.debug(f"Pod {self.pod_name} is still pending...")
                    time.sleep(60)
                elif pod.status.phase == "Succeeded":
                    raise ValueError(
                        f"Pod {self.pod_name} has already succeeded unexpectedly."
                    )
                else:
                    self.logger.debug(
                        f"Pod {self.pod_name} is in {pod.status.phase} state..."
                    )
                    time.sleep(5)

            except ApiException as e:
                self.logger.debug(f"Error checking pod status: {e}")

        raise ValueError(
            f"Pod {self.pod_name} did not become ready within {timeout} seconds"
        )

    def _run_setup_commands(self):
        """Run setup commands if any. If commands fail, delete the pod."""
        if self.setup_commands:
            setup_commands = " && ".join(self.setup_commands)
            self.logger.debug(
                f"Pod {self.pod_name} Running setup commands: {setup_commands}"
            )

            success, output = self.run(setup_commands)
            if not success:
                self.clean_up()
                raise ValueError(
                    f"Failed to run setup command: {setup_commands}\n"
                    f"Output: {output}"
                )
            self.logger.debug("Setup commands ran successfully.")

    def clean_up(self):
        """Clean up the Kubernetes pod."""
        if self.pod_exists():
            try:
                self.k8s_client.delete_namespaced_pod(
                    name=self.pod_name, namespace=self.namespace, grace_period_seconds=5
                )
                self.logger.debug(f"Pod {self.pod_name} deleted successfully.")
            except ApiException as e:
                if e.status != 404:  # Ignore not found errors
                    self.logger.debug(f"Failed to delete pod {self.pod_name}: {e}")
            except Exception as e:
                self.logger.debug(f"Error during pod cleanup: {e}")
            finally:
                self._pod_name = None

    def close(self):
        super().close()
        self.clean_up()

    def pod_exists(self) -> bool:
        """Check if the pod exists in the namespace."""
        try:
            self.k8s_client.read_namespaced_pod(
                name=self.pod_name, namespace=self.namespace
            )
            return True
        except ApiException as e:
            if e.status == 404:
                return False
            self.logger.debug(f"Error checking pod existence: {e}")
            return False

    def is_running(self) -> bool:
        """Check if the pod is currently running."""
        try:
            pod = self.k8s_client.read_namespaced_pod(
                name=self.pod_name, namespace=self.namespace
            )
            return pod.status.phase == "Running"
        except ApiException as e:
            if e.status == 404:
                return False
            self.logger.debug(f"Error checking pod status: {e}")
            return False

    def __str__(self):
        return f"KubernetesTerminal[{self.pod_name}, {self.working_dir}]"

    def copy_content(self, src: str | Path, target: str | Path | None = None) -> None:
        """Copy files or directories from host to pod using kubectl cp.

        kubectl cp natively handles both files and directories, so we can
        simplify this to a single command rather than iterating through files.
        """
        if not self.is_running():
            raise ValueError("Pod is not running. Cannot copy files.")

        src = str(src)
        target = str(target or self.working_dir)

        if not os.path.isdir(src):
            raise ValueError(f"Source {src} must be a directory.")

        self.logger.debug(f"[{self.pod_name}] Copying {src} to {target}.")

        try:
            # kubectl cp can handle both files and directories natively
            # Format: kubectl cp <src> <namespace>/<pod>:<dest>
            # The official Kubernetes Python client does not provide a direct method for file copy.
            # The recommended approach is still to use 'kubectl cp' via subprocess.
            # Alternatives (using tar + exec) are complex and less reliable for directories.
            result = subprocess.run(
                [
                    "kubectl",
                    "--kubeconfig",
                    self.kube_config,
                    "cp",
                    f"{src}/.",
                    f"{self.namespace}/{self.pod_name}:{target}",
                ],
                capture_output=True,
                text=True,
                timeout=120,  # Increased timeout for directory operations
            )

            if result.returncode != 0:
                raise ValueError(f"Failed to copy {src} to {target}: {result.stderr}")

            self.logger.debug(f"Successfully copied {src} to {target}")

        except subprocess.TimeoutExpired:
            raise ValueError(f"Timeout copying {src} to {target}")
        except FileNotFoundError:
            raise ValueError(
                "kubectl command not found. Please ensure kubectl is installed and in PATH."
            )
        except Exception as e:
            self.logger.debug(f"Error copying {src} to {target}: {e}")
            raise
