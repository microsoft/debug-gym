import atexit
import json
import os
import random
import subprocess
import time
import uuid
from pathlib import Path

from jinja2 import Template
from kubernetes import client, config, stream, watch
from kubernetes.client.rest import ApiException
from kubernetes.stream.ws_client import ERROR_CHANNEL
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)
from yaml import dump, safe_load

from debug_gym.gym.terminals.shell_session import ShellSession
from debug_gym.gym.terminals.terminal import DISABLE_ECHO_COMMAND, Terminal
from debug_gym.logger import DebugGymLogger

NB_RETRIES_RUN = 50  # Number of retries for running a command


class SandboxReservationError(Exception):
    """Raised when a pod cannot be created due to sandbox name reservation conflict."""

    pass


def _clean_for_kubernetes(name: str) -> str:
    """Clean pod name to conform to Kubernetes naming conventions."""
    # replace any characters not in the regex with hyphens
    cleaned = "".join(c if c.isalnum() or c in "-." else "-" for c in name).lower()
    # ensure it starts and ends with alphanumeric character
    cleaned = cleaned.strip("-").strip(".")
    # truncate to 253 characters
    return cleaned[:253]


class Pod:
    def __init__(
        self, k8s_client: client.CoreV1Api, pod_body: dict, logger: DebugGymLogger
    ):
        self.k8s_client = k8s_client
        self.pod_body = pod_body
        self.name = self.pod_body["metadata"]["name"]
        self.namespace = self.pod_body["metadata"]["namespace"]
        self.logger = logger
        self._last_pending_reason = None  # Track to avoid duplicate pending logs

        self.create_pod()
        atexit.register(self.clean_up)
        self.wait_for_pod_ready()

    @retry(
        retry=retry_if_exception_type(ApiException),
        wait=wait_random_exponential(multiplier=1, min=1, max=240),
        stop=stop_after_attempt(30),
        reraise=True,
    )
    def create_pod(self):
        """Create a Kubernetes pod with tenacity retry and exponential backoff."""
        try:
            pod = self.k8s_client.create_namespaced_pod(
                namespace=self.namespace, body=self.pod_body, field_validation="Strict"
            )
            self.logger.debug(f"Created {self}.")
            return pod
        except ApiException as e:
            # Handle 409 conflicts specially - check if pod already exists
            if e.status == 409:
                try:
                    existing = self.k8s_client.read_namespaced_pod(
                        name=self.name, namespace=self.namespace
                    )
                    if existing:
                        self.logger.debug(f"{self} Reusing existing pod.")
                        return
                except ApiException:
                    pass  # Fall through to retry logic

            # Only retry on specific HTTP status codes
            if e.status in (409, 429, 500, 503):
                self.logger.warning(f"{self} Retrying pod creation (HTTP {e.status}).")
                raise  # Let tenacity handle the retry
            else:
                # Non-retriable error, fail immediately
                raise ValueError(f"Failed to create pod: {e}")
        except Exception as e:
            # Non-ApiException errors are not retriable
            raise ValueError(f"Failed to create pod: {e}")

    def wait_for_pod_ready(self, timeout: int = 3600 * 2):
        """Wait for the pod to be in Running state using Kubernetes watch."""
        self.logger.debug(f"{self} Waiting to be ready...")

        w = watch.Watch()
        start_time = time.time()
        sandbox_check_interval = 30  # Check for sandbox errors every 30 seconds
        last_sandbox_check = 0

        try:
            for event in w.stream(
                self.k8s_client.list_namespaced_pod,
                namespace=self.namespace,
                field_selector=f"metadata.name={self.name}",
                timeout_seconds=timeout,
            ):
                event_type = event.get("type")
                pod = event.get("object")

                if not pod:
                    continue

                phase = pod.status.phase

                if phase == "Running":
                    self.logger.debug(f"{self} is ready on node {pod.spec.node_name}")
                    return
                elif phase in ["Failed", "Unknown", "Succeeded"]:
                    raise ValueError(f"{self} is in {phase} state instead of running.")
                elif phase == "Pending":
                    if event_type == "ADDED":
                        # Only log pending status on initial ADDED event or when reason changes
                        self._log_pending_status(pod)

                    # Periodically check for sandbox reservation errors while pending
                    elapsed = time.time() - start_time
                    if elapsed - last_sandbox_check >= sandbox_check_interval:
                        last_sandbox_check = elapsed
                        if self._has_sandbox_reservation_error():
                            raise SandboxReservationError(
                                f"{self} has sandbox reservation conflict"
                            )

        except SandboxReservationError:
            raise  # Re-raise sandbox errors without wrapping
        except Exception as e:
            self.logger.debug(f"{self} Error during pod watch: {e}")
            raise ValueError(f"Error watching pod {self.name}: {e}")
        finally:
            w.stop()

        # If we get here, we've timed out
        raise ValueError(
            f"Pod {self.name} did not become ready within {timeout} seconds"
        )

    def _has_sandbox_reservation_error(self) -> bool:
        """Check pod events for sandbox reservation errors."""
        try:
            events = self.k8s_client.list_namespaced_event(
                namespace=self.namespace,
                field_selector=f"involvedObject.name={self.name}",
            )
            for event in events.items:
                if event.reason == "FailedCreatePodSandBox" and event.message:
                    if "is reserved for" in event.message:
                        self.logger.warning(
                            f"{self} Sandbox reservation conflict detected: {event.message}"
                        )
                        return True
        except ApiException as e:
            self.logger.debug(f"{self} Error checking pod events: {e}")
        return False

    def _log_pending_status(self, pod):
        """Log pending status only if it's different from the last one."""
        if pod.status.conditions:
            for condition in pod.status.conditions:
                if condition.status == "False":
                    reason = f"{condition.reason}: {condition.message}"
                    if reason != self._last_pending_reason:
                        self.logger.debug(f"{self} Pending - {reason}")
                        self._last_pending_reason = reason
                    return

        # No specific conditions found
        if self._last_pending_reason != "scheduling":
            self.logger.debug(f"{self} Pending - scheduling...")
            self._last_pending_reason = "scheduling"

    def exists(self) -> bool:
        """Check if the pod exists in the namespace."""
        try:
            self.k8s_client.read_namespaced_pod(
                name=self.name, namespace=self.namespace
            )
            return True
        except ApiException as e:
            if e.status == 404:
                return False
            self.logger.debug(f"{self} Error checking pod existence: {e}")
            raise

    def is_running(self) -> bool:
        """Check if the pod is currently running."""
        try:
            pod = self.k8s_client.read_namespaced_pod(
                name=self.name, namespace=self.namespace
            )
            return pod.status.phase == "Running"
        except ApiException as e:
            if e.status == 404:
                return False
            self.logger.debug(f"{self} Error checking pod status: {e}")
            raise

    def clean_up(self):
        """Clean up the Kubernetes pod."""
        if not self.exists():
            return

        try:
            self.k8s_client.delete_namespaced_pod(
                name=self.name, namespace=self.namespace, grace_period_seconds=5
            )

            w = watch.Watch()
            try:
                for event in w.stream(
                    self.k8s_client.list_namespaced_pod,
                    namespace=self.namespace,
                    field_selector=f"metadata.name={self.name}",
                    timeout_seconds=60,
                ):
                    if event.get("type") == "DELETED":
                        self.logger.debug(f"{self} deleted.")
                        return
                else:
                    self.logger.debug(f"{self} deletion timed out.")
            finally:
                w.stop()

            if self.exists():
                self.logger.debug(
                    f"{self} still exists after delete - manual cleanup may be required"
                )

        except ApiException as e:
            if e.status != 404:  # Ignore not found errors
                self.logger.debug(f"Failed to delete pod {self.name}: {e}")
        except Exception as e:
            self.logger.debug(f"Error during pod cleanup: {e}")

    def __str__(self):
        return f"Pod[{self.name}]"


class KubernetesTerminal(Terminal):

    def __init__(
        self,
        working_dir: str | None = None,
        session_commands: list[str] | None = None,
        env_vars: dict[str, str] | None = None,
        logger: DebugGymLogger | None = None,
        # Kubernetes-specific parameters
        setup_commands: list[str] | None = None,
        pod_name: str | None = None,
        base_image: str | None = None,
        registry: str = "",
        namespace: str = "default",
        kube_config: str | None = None,
        kube_context: str | None = None,
        extra_labels: dict | None = None,
        pod_spec_kwargs: dict = None,
        **kwargs,
    ):
        super().__init__(
            working_dir=working_dir,
            session_commands=session_commands,
            env_vars=env_vars,
            logger=logger,
            **kwargs,
        )
        self.base_image = base_image
        self._task_name = base_image
        self.setup_commands = setup_commands or []
        self.namespace = namespace
        self.kubernetes_kwargs = kwargs  # e.g., nodeSelector, tolerations
        self.registry = registry.rstrip("/") + "/" if registry else ""
        self._pod_name = pod_name
        self.pod_spec_kwargs = pod_spec_kwargs or {}
        user = _clean_for_kubernetes(os.environ.get("USER", "unknown"))
        self.labels = {"app": "dbg-gym", "user": user} | (extra_labels or {})
        self._pod = None

        # Initialize Kubernetes client
        self.kube_config = kube_config
        self.kube_context = kube_context
        if self.kube_config == "incluster":
            self.kube_config = None
            config.load_incluster_config()
            # For in-cluster kubectl access, pass Kubernetes service environment variables
            # This enables kubectl to auto-discover the service account credentials
            for key in ("KUBERNETES_SERVICE_HOST", "KUBERNETES_SERVICE_PORT"):
                if key in os.environ:
                    self.env_vars.setdefault(key, os.environ[key])
        else:
            self.kube_config = self.kube_config or os.environ.get(
                "KUBECONFIG", "~/.kube/config"
            )
            self.kube_config = os.path.expanduser(self.kube_config)
            config.load_kube_config(self.kube_config, self.kube_context)
            self.env_vars.setdefault("KUBECONFIG", self.kube_config)

        # Ensure helper binaries such as kubectl can be discovered even when
        # host environment variables are not inherited.
        if "PATH" in os.environ:
            self.env_vars.setdefault("PATH", os.environ["PATH"])

        self.k8s_client = client.CoreV1Api()
        atexit.register(self.close)

    @property
    def pod_name(self):
        if self._pod is None:
            raise ValueError("Pod not created yet; pod_name is not available.")

        return self._pod.name

    @pod_name.setter
    def pod_name(self, value):
        if self._pod is not None:
            raise ValueError("Cannot change the pod's name after its creation.")

        self._pod_name = value

    @property
    def task_name(self):
        return self._task_name

    @task_name.setter
    def task_name(self, value):
        if self._pod is not None:
            raise ValueError("Cannot change task_name after pod creation.")

        self._task_name = value

    @property
    def working_dir(self):
        """Lazy initialization of the working directory."""
        return super().working_dir

    @working_dir.setter
    def working_dir(self, value):
        if self._pod is not None:
            raise ValueError("Cannot change working directory after pod creation.")

        self._working_dir = value

    @property
    def pod(self):
        """Lazy initialization of the pod."""
        if self._pod is None:
            self.setup_pod()

        return self._pod

    @property
    def default_shell_command(self) -> list[str]:
        """Expects the pod to have bash installed."""
        kubeconfig = f"--kubeconfig {self.kube_config} " if self.kube_config else ""
        bash_cmd = "/bin/bash --noprofile --norc --noediting"
        return f"kubectl {kubeconfig}exec -it {self.pod.name} -c main -n {self.pod.namespace} -- {bash_cmd}"

    def _ensure_pod_running(self) -> None:
        """Ensure the backing pod exists and is in Running phase."""
        if self._pod is None:
            self.setup_pod()
            return

        try:
            if self._pod.is_running():
                return
        except Exception as exc:  # noqa: BLE001 - diagnostics only
            self.logger.debug(f"{self._pod} status check failed: {exc}")

        self.logger.debug(f"{self._pod} not running anymore.")

        # Check logs and describe for diagnostics
        try:
            pod_logs = self.k8s_client.read_namespaced_pod_log(
                name=self._pod.name, namespace=self._pod.namespace
            )
            pod_description = self.k8s_client.read_namespaced_pod(
                name=self._pod.name, namespace=self._pod.namespace
            )
            self.logger.debug(
                f"[{self._pod.name}] Pod logs before failure:\n{pod_logs}\n"
                f"Pod description before failure:\n{pod_description}"
            )
        except Exception as log_exc:
            self.logger.debug(
                f"[{self._pod.name}] Failed to get pod logs/description: {log_exc}"
            )

        self.logger.debug(f"Cleaning up {self._pod} after failure.")
        try:
            self._pod.clean_up()
        except Exception as exc:  # noqa: BLE001 - best-effort cleanup
            self.logger.debug(f"Failed to clean up {self._pod}: {exc}")

        raise RuntimeError("Pod is not running anymore.")

    def new_shell_session(self):
        self._ensure_pod_running()

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
        entrypoint_str = " && ".join(entrypoint)

        # Set environment variables by prefixing the command
        env_prefix = ""
        if self.env_vars:
            env_vars_str = " ".join([f'{k}="{v}"' for k, v in self.env_vars.items()])
            env_prefix = f"export {env_vars_str} && "

        # Build the full command with environment variables and working directory
        command = entrypoint_str
        if self.working_dir and self.working_dir != "/":
            command = f"cd {self.working_dir} && {env_prefix}{command}"
        elif env_prefix:
            command = f"{env_prefix}{command}"

        return command

    def run(
        self,
        entrypoint: str | list[str],
        timeout: int = None,
        raises: bool = False,
        strip_output: bool = True,
    ) -> tuple[bool, str]:
        """Run a command in the pod. Return command status and output."""
        self._ensure_pod_running()

        command = self.prepare_command(entrypoint)

        self.logger.debug(f"[{self.pod.name}] Kubernetes exec run: {command}")
        for _ in range(NB_RETRIES_RUN):
            try:
                # Execute command using Kubernetes stream API
                resp = stream.stream(
                    self.k8s_client.connect_get_namespaced_pod_exec,
                    name=self.pod.name,
                    namespace=self.pod.namespace,
                    command=["/bin/bash", "-c", command],
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
                self.logger.debug(f"[{self.pod.name}] error channel: {error_channel}")
                status = json.loads(error_channel)
                success = status["status"] == "Success"
                break  # Command executed successfully, exit the retry loop

            except ApiException as e:
                success = False
                self.logger.debug(
                    f"[{self.pod.name}] Exception during command `{command}`: {e}"
                )
                # Get kubectl logs and describe for diagnostics
                try:
                    pod_logs = self.k8s_client.read_namespaced_pod_log(
                        name=self.pod.name, namespace=self.pod.namespace
                    )
                    pod_description = self.k8s_client.read_namespaced_pod(
                        name=self.pod.name, namespace=self.pod.namespace
                    )
                    self.logger.debug(
                        f"[{self.pod.name}] Pod logs:\n{pod_logs}\n"
                        f"Pod description:\n{pod_description}"
                    )
                except Exception as log_exc:
                    self.logger.debug(
                        f"[{self.pod.name}] Failed to get pod logs/description: {log_exc}"
                    )

                output = f"Command execution failed: {str(e)}"
                backoff = random.uniform(5, 10)  # seconds
                time.sleep(backoff)

        if strip_output:
            output = output.strip("\r\n").strip("\n")

        if raises and not success:
            self.logger.error(f"Failed to run command `{command}`:\n{output}")
            raise ValueError(f"Failed to run command `{entrypoint}`")

        self.logger.debug(f"[{self.pod.name}] Output success `{success}`:\n{output}")
        return success, output

    def setup_pod(self, max_retries: int = 3) -> None:
        """Create and start a Kubernetes pod.

        If a sandbox reservation conflict is detected, the pod is cleaned up
        and a new pod with a fresh UUID is created.
        """

        # Render pod_spec_kwargs as a Jinja2 template, replace variables, then load as dict.
        pod_spec_yaml = dump(self.pod_spec_kwargs)
        pod_spec_template = Template(pod_spec_yaml)
        rendered_yaml = pod_spec_template.render(os.environ)
        pod_spec_kwargs = safe_load(rendered_yaml)

        for attempt in range(max_retries):
            # Generate a new pod name for each attempt to avoid sandbox conflicts
            pod_name = _clean_for_kubernetes(
                self._pod_name or f"dbg-gym.{self.task_name}.{str(uuid.uuid4())[:8]}"
            )
            self.logger.debug(
                f"Setting up pod {pod_name} (attempt {attempt + 1}/{max_retries}) "
                f"with image: {self.registry}{self.base_image}"
            )

            # Create pod specification for Kubernetes.
            pod_body = {
                "apiVersion": "v1",
                "kind": "Pod",
                "metadata": {
                    "name": pod_name,
                    "namespace": self.namespace,
                    "labels": self.labels,
                },
                "spec": {
                    "activeDeadlineSeconds": 3600 * 24,  # a day
                    "restartPolicy": "Never",
                    "containers": [
                        {
                            "name": "main",
                            "image": f"{self.registry}{self.base_image}",
                            "imagePullPolicy": "IfNotPresent",
                            "command": ["/bin/bash"],
                            "args": ["-c", "sleep infinity"],
                            "workingDir": self.working_dir,
                            "stdin": True,
                            "stdinOnce": False,
                            "tty": True,
                            "env": [
                                {"name": k, "value": v}
                                for k, v in self.env_vars.items()
                            ],
                            "resources": {
                                "requests": {"cpu": "0.5", "memory": "1Gi"},
                                "limits": {"cpu": "2", "memory": "8Gi"},
                            },
                        }
                    ],
                    **pod_spec_kwargs,  # e.g., nodeSelector, tolerations
                },
            }

            try:
                self._pod = Pod(self.k8s_client, pod_body, logger=self.logger)

                # Run setup commands
                self._run_setup_commands()
                self.logger.debug(f"{self.pod} started successfully.")
                return  # Success, exit the retry loop

            except SandboxReservationError as e:
                self.logger.warning(
                    f"Sandbox reservation conflict for {pod_name}: {e}. "
                    f"Cleaning up and retrying with new pod name..."
                )
                # Clean up the failed pod
                if self._pod is not None:
                    self._pod.clean_up()
                    self._pod = None

                # Wait a bit before retrying to allow the container runtime to clean up
                time.sleep(5)

                if attempt == max_retries - 1:
                    raise ValueError(
                        f"Failed to create pod after {max_retries} attempts "
                        f"due to sandbox reservation conflicts"
                    )
            except ApiException as e:
                raise ValueError(f"Failed to create pod: {e}")

    def _run_setup_commands(self):
        """Run setup commands if any. If commands fail, delete the pod."""
        if not self.setup_commands:
            return

        setup_commands = " && ".join(self.setup_commands)
        success, output = self.run(setup_commands, raises=True)
        if not success:
            self.close()
            raise ValueError(
                f"Failed to run setup command: {setup_commands}\n" f"Output: {output}"
            )
        self.logger.debug("Setup commands ran successfully.")

    def close(self):
        super().close()
        if self._pod is not None:
            self._pod.clean_up()
            self._pod = None

    def __del__(self):
        self.close()

    def __str__(self):
        return f"KubernetesTerminal[{self.pod_name}, {self.working_dir}]"

    def copy_content(self, src: str | Path, target: str | Path | None = None) -> None:
        """Copy files or directories from host to pod using kubectl cp.

        kubectl cp natively handles both files and directories, so we can
        simplify this to a single command rather than iterating through files.
        """
        if not self.pod.is_running():
            raise ValueError("Pod is not running. Cannot copy files.")

        src = str(src)
        target = str(target or self.working_dir)

        if not os.path.isdir(src):
            raise ValueError(f"Source {src} must be a directory.")

        self.logger.debug(f"[{self.pod.name}] Copying {src} to {target}.")

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
                    f"{self.pod.namespace}/{self.pod.name}:{target}",
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
