import atexit
import json
import os
import random
import shlex
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
from debug_gym.gym.terminals.terminal import (
    DISABLE_ECHO_COMMAND,
    Terminal,
    UnrecoverableTerminalError,
)
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
    cleaned = cleaned.replace("/", "-")
    cleaned = cleaned.replace(":", "-")
    cleaned = cleaned.replace(".", "-")
    cleaned = cleaned.strip("-").strip(".")
    # truncate to 253 characters
    return cleaned[:253]


class Pod:
    def __init__(
        self,
        k8s_client: client.CoreV1Api,
        pod_body: dict,
        logger: DebugGymLogger,
        image_pull_backoff_timeout: int = 600,
    ):
        self.k8s_client = k8s_client
        self.pod_body = pod_body
        self.name = self.pod_body["metadata"]["name"]
        self.namespace = self.pod_body["metadata"]["namespace"]
        self.logger = logger
        self.image_pull_backoff_timeout = image_pull_backoff_timeout
        self._last_pending_reason = None  # Track to avoid duplicate pending logs
        self._image_pull_backoff_since = None  # Track when ImagePullBackOff started

        self.create_pod()
        atexit.register(self.clean_up)
        self.wait_for_pod_ready(image_pull_backoff_timeout=image_pull_backoff_timeout)

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

    def wait_for_pod_ready(
        self, timeout: int = 3600 * 2, image_pull_backoff_timeout: int = 600
    ):
        """Wait for the pod to be in Running state using Kubernetes watch.

        Args:
            timeout: Overall timeout in seconds (default: 2 hours)
            image_pull_backoff_timeout: How long to wait before failing on persistent
                ImagePullBackOff (default: 600 = 10 minutes). Set to 0 to disable.
                This gives time for transient issues (registry rate limiting,
                network blips) to resolve before giving up.
        """
        self.logger.debug(f"{self} Waiting to be ready...")

        w = watch.Watch()
        start_time = time.time()
        check_interval = 30  # Check for errors every 30 seconds
        last_check_time = 0

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

                    # Periodically check for errors while pending
                    elapsed = time.time() - start_time
                    if elapsed - last_check_time >= check_interval:
                        last_check_time = elapsed

                        # Check for sandbox reservation errors
                        if self._has_sandbox_reservation_error():
                            raise SandboxReservationError(
                                f"{self} has sandbox reservation conflict"
                            )

                        # Check for persistent image pull failures
                        if image_pull_backoff_timeout > 0:
                            has_backoff, message = self._check_image_pull_status()
                            if has_backoff:
                                if self._image_pull_backoff_since is None:
                                    self._image_pull_backoff_since = time.time()
                                    self.logger.debug(
                                        f"{self} ImagePullBackOff detected: {message}"
                                    )
                                else:
                                    backoff_duration = (
                                        time.time() - self._image_pull_backoff_since
                                    )
                                    if backoff_duration >= image_pull_backoff_timeout:
                                        raise ValueError(
                                            f"{self} image pull failed after {backoff_duration:.0f}s: {message}"
                                        )
                            else:
                                # Reset if backoff resolved (e.g., pull succeeded after retry)
                                if self._image_pull_backoff_since is not None:
                                    self.logger.debug(
                                        f"{self} ImagePullBackOff resolved"
                                    )
                                    self._image_pull_backoff_since = None

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

    def _check_image_pull_status(self) -> tuple[bool, str | None]:
        """Check container status for image pull failures.

        Returns:
            Tuple of (has_backoff, message). has_backoff is True if ImagePullBackOff detected.
        """
        try:
            pod = self.k8s_client.read_namespaced_pod(
                name=self.name, namespace=self.namespace
            )
            for cs in pod.status.container_statuses or []:
                if cs.state and cs.state.waiting:
                    reason = cs.state.waiting.reason
                    if reason == "ImagePullBackOff":
                        return True, cs.state.waiting.message
            return False, None
        except ApiException as e:
            self.logger.debug(f"{self} Error checking container status: {e}")
            return False, None

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
    """
    Kubernetes-based terminal for running commands in pods.

    Note: reads values of env variables K8S_NAMESPACE, K8S_DOCKER_SECRET, K8S_DOCKER_CONSTRAINT.
    """

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
        image_pull_secret: str | None = None,
        registry: str = "docker.io",
        namespace: str | None = None,
        kube_config: str | None = None,
        kube_context: str | None = None,
        extra_labels: dict | None = None,
        pod_spec_kwargs: dict = None,
        command_timeout: int = 300,
        image_pull_backoff_timeout: int = 600,
        **kwargs,
    ):
        """
        Args:
            working_dir: Working directory inside the pod.
            session_commands: Commands to run at the start of each session.
            env_vars: Environment variables to set in the pod.
            logger: Logger instance.
            setup_commands: Commands to run once when setting up the pod.
            pod_name: Custom name for the pod.
            base_image: Docker image to use for the pod.
            image_pull_secret: Kubernetes secret for pulling images.
            registry: Docker registry URL.
            namespace: Kubernetes namespace.
            kube_config: Path to kubeconfig or "incluster".
            kube_context: Kubernetes context to use.
            extra_labels: Additional labels for the pod.
            pod_spec_kwargs: Additional pod spec configuration.
            command_timeout: Default timeout in seconds for individual command execution
                (default: 300 = 5 minutes). This is NOT the pod lifetime. Commands that
                exceed this timeout will be killed.
            image_pull_backoff_timeout: How long to wait before failing on persistent
                ImagePullBackOff (default: 600 = 10 minutes). Set to 0 to disable.
                This gives time for transient issues (registry rate limiting,
                network blips) to resolve before giving up.
            **kwargs: Additional arguments passed to pod spec.
        """
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
        self.command_timeout = command_timeout
        self.image_pull_backoff_timeout = image_pull_backoff_timeout
        self.namespace = namespace or os.environ.get("K8S_NAMESPACE", "default")
        self.image_pull_secret = image_pull_secret or os.environ.get(
            "K8S_DOCKER_SECRET"
        )
        self.in_node_constraint = os.environ.get("K8S_IN_NODE_CONSTRAINT", False)
        self.kubernetes_kwargs = kwargs  # e.g., nodeSelector, tolerations
        self.critical_addons_only = self.kubernetes_kwargs.get(
            "critical_addons_only", False
        )
        self.registry = registry.rstrip("/") + "/" if registry else ""
        self._pod_name = pod_name
        self.pod_spec_kwargs = pod_spec_kwargs or {}
        user = _clean_for_kubernetes(os.environ.get("USER", "unknown").split("@")[0])
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
        if not self.pod.is_running():
            raise UnrecoverableTerminalError(
                "Pod is not running. Cannot create shell session."
            )

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
    ) -> str:
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

        # Wrap with timeout command if specified
        if timeout is not None:
            # Use timeout command to kill the process if it exceeds the limit
            # Exit code 124 indicates timeout was reached
            command = f"timeout {timeout} /bin/bash -c {shlex.quote(command)}"

        return command

    def run(
        self,
        entrypoint: str | list[str],
        timeout: int = None,
        raises: bool = False,
        strip_output: bool = True,
    ) -> tuple[bool, str]:
        """Run a command in the pod. Return command status and output.

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
        if not self.pod.is_running():
            raise UnrecoverableTerminalError("Pod is not running. Cannot run commands.")

        # Use command_timeout if not specified per-call
        effective_timeout = timeout if timeout is not None else self.command_timeout
        command = self.prepare_command(entrypoint, timeout=effective_timeout)

        self.logger.debug(
            f"[{self.pod.name}] Kubernetes exec run (timeout={effective_timeout}s): {command}"
        )
        exit_code = None
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

                # Parse exit code from status
                if status["status"] == "Success":
                    exit_code = 0
                    success = True
                else:
                    # Try to extract exit code from status details
                    exit_code = 1  # Default to 1 for failure
                    if "details" in status and "causes" in status["details"]:
                        for cause in status["details"]["causes"]:
                            if cause.get("reason") == "ExitCode":
                                exit_code = int(cause.get("message", "1"))
                                break
                    success = False
                break  # Command executed successfully, exit the retry loop

            except ApiException as e:
                success = False
                exit_code = None
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

        # Check for timeout (exit code 124 from the timeout command)
        if exit_code == 124:
            self.logger.warning(
                f"[{self.pod.name}] Command timed out after {effective_timeout}s: {entrypoint}"
            )
            timeout_msg = f"Command timed out after {effective_timeout} seconds"
            if output:
                output = f"{timeout_msg}\nPartial output:\n{output}"
            else:
                output = timeout_msg
            return False, output

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

            # set image pull secrets, don't override imagePullSecrets
            if self.image_pull_secret and not "imagePullSecrets" in pod_spec_kwargs:
                pod_spec_kwargs["imagePullSecrets"] = [{"name": self.image_pull_secret}]

            # set in node constraint, don't override affinity
            if self.in_node_constraint and not "affinity" in pod_spec_kwargs:
                pod_spec_kwargs["affinity"] = {
                    "nodeAffinity": {
                        "requiredDuringSchedulingIgnoredDuringExecution": {
                            "nodeSelectorTerms": [
                                {
                                    "matchExpressions": [
                                        {
                                            "key": "kubernetes.io/hostname",
                                            "operator": "In",
                                            "values": [os.environ["HOSTNAME"]],
                                        }
                                    ]
                                }
                            ]
                        }
                    }
                }

            if self.critical_addons_only:
                pod_spec_kwargs["tolerations"] = pod_spec_kwargs.get(
                    "tolerations", []
                ) + [
                    {
                        "key": "CriticalAddonsOnly",
                        "operator": "Equal",
                        "value": "true",
                        "effect": "NoSchedule",
                    },
                ]

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
                self._pod = Pod(
                    self.k8s_client,
                    pod_body,
                    logger=self.logger,
                    image_pull_backoff_timeout=self.image_pull_backoff_timeout,
                )

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
        success, output = self.run(setup_commands, raises=False)
        if not success:
            self.close()
            raise UnrecoverableTerminalError(
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
            raise UnrecoverableTerminalError("Pod is not running. Cannot copy files.")

        src = str(src)
        target = str(target or self.working_dir)

        self.logger.debug(f"[{self.pod.name}] Copying {src} to {target}.")

        try:
            # kubectl cp can handle both files and directories natively
            # Format: kubectl cp <src> <namespace>/<pod>:<dest>
            # The official Kubernetes Python client does not provide a direct method for file copy.
            # The recommended approach is still to use 'kubectl cp' via subprocess.
            # Alternatives (using tar + exec) are complex and less reliable for directories.
            cmd = ["kubectl"]
            if self.kube_config:
                cmd.extend(["--kubeconfig", self.kube_config])
            # restore previous behavior
            if os.path.isdir(src):
                src = f"{src}/."
            cmd.extend(
                [
                    "cp",
                    f"{src}",
                    f"{self.pod.namespace}/{self.pod.name}:{target}",
                ]
            )

            for _ in range(NB_RETRIES_RUN):
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=120,  # Increased timeout for directory operations
                )

                if result.returncode != 0:
                    if "Internal Server Error" in result.stderr:
                        self.logger.debug(
                            f"[{self.pod.name}] Transient error during copy, retrying: {result.stderr}"
                        )
                        backoff = random.uniform(5, 10)  # seconds
                        time.sleep(backoff)
                        continue  # Retry

                    raise ValueError(
                        f"Failed to copy {src} to {target}: {result.stderr}"
                    )

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
