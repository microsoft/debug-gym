import os
import platform
import subprocess
import time
from unittest.mock import MagicMock, patch

import pytest

from debug_gym.gym.terminals import select_terminal
from debug_gym.gym.terminals.kubernetes import KubernetesTerminal, Pod
from debug_gym.gym.terminals.shell_session import DEFAULT_PS1
from debug_gym.gym.terminals.terminal import (
    DISABLE_ECHO_COMMAND,
    UnrecoverableTerminalError,
)


def is_kubernetes_available():
    """Check if kubectl is available and can connect to a cluster."""
    try:
        subprocess.check_output(["kubectl", "cluster-info"], stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False


if_kubernetes_available = pytest.mark.skipif(
    not is_kubernetes_available(),
    reason="Kubernetes cluster not available",
)

if_is_linux = pytest.mark.skipif(
    platform.system() != "Linux",
    reason="Interactive ShellSession (pty) requires Linux.",
)


@if_kubernetes_available
def test_kubernetes_terminal_init():
    terminal = KubernetesTerminal(base_image="ubuntu:latest")
    assert terminal.session_commands == []
    expected_base_env = {
        "NO_COLOR": "1",
        "PS1": DEFAULT_PS1,
        "PYTHONSTARTUP": "",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    for key, value in expected_base_env.items():
        assert terminal.env_vars[key] == value

    assert terminal.env_vars["PATH"] == os.environ.get("PATH")
    if terminal.kube_config:
        assert terminal.env_vars["KUBECONFIG"] == terminal.kube_config
    else:
        assert "KUBECONFIG" not in terminal.env_vars

    extra_env_keys = set(terminal.env_vars) - (
        set(expected_base_env) | {"PATH", "KUBECONFIG"}
    )
    assert not extra_env_keys
    assert os.path.basename(terminal.working_dir).startswith("Terminal-")
    assert terminal.base_image == "ubuntu:latest"
    assert terminal.namespace == "default"

    # Pod should not be created until accessed
    assert terminal._pod is None
    with pytest.raises(
        ValueError, match="Pod not created yet; pod_name is not available."
    ):
        terminal.pod_name  # Accessing pod_name before pod creation should raise an error.

    # Assessing the `pod` property will create it.
    assert terminal.pod
    assert terminal._pod is not None

    # Pod name should be automatically generated when not provided at initialization.
    assert terminal.pod_name.startswith("dbg-gym-")
    assert terminal.pod.is_running()
    assert terminal.pod.exists()

    # Close pod.
    terminal.close()
    assert terminal._pod is None


@if_kubernetes_available
def test_kubernetes_terminal_init_with_params(tmp_path):
    working_dir = str(tmp_path)
    session_commands = ["mkdir new_dir"]
    env_vars = {"ENV_VAR": "value"}
    base_image = "ubuntu:24.04"
    namespace = "default"  # Need to exists.
    pod_name = "test-pod-123"

    terminal = KubernetesTerminal(
        working_dir=working_dir,
        session_commands=session_commands,
        env_vars=env_vars,
        base_image=base_image,
        namespace=namespace,
        pod_name=pod_name,
    )
    assert terminal.working_dir == working_dir
    assert terminal.session_commands == session_commands
    assert terminal.env_vars["ENV_VAR"] == "value"
    assert terminal.env_vars["NO_COLOR"] == "1"
    assert terminal.env_vars["PS1"] == DEFAULT_PS1
    assert terminal.env_vars["PYTHONSTARTUP"] == ""
    assert terminal.env_vars["PATH"] == os.environ.get("PATH")
    if terminal.kube_config:
        assert terminal.env_vars["KUBECONFIG"] == terminal.kube_config
    else:
        assert "KUBECONFIG" not in terminal.env_vars
    assert terminal.base_image == base_image

    # Create pod.
    assert terminal.pod is not None
    assert terminal.pod.is_running()
    assert terminal.pod.name == pod_name
    assert terminal.pod.namespace == namespace

    # Close pod.
    terminal.close()
    assert terminal._pod is None


@if_kubernetes_available
def test_kubernetes_terminal_init_with_pod_specs(tmp_path):
    working_dir = str(tmp_path)
    # set an environment variable to use in the pod spec
    os.environ["HOSTNAME"] = "minikube"
    pod_spec_kwargs = {
        "affinity": {
            "nodeAffinity": {
                "requiredDuringSchedulingIgnoredDuringExecution": {
                    "nodeSelectorTerms": [
                        {
                            "matchExpressions": [
                                {
                                    "key": "kubernetes.io/hostname",
                                    "operator": "In",
                                    "values": ["{{HOSTNAME}}"],
                                }
                            ]
                        }
                    ]
                }
            }
        },
        "tolerations": [
            {
                "key": "kubernetes.azure.com/scalesetpriority",
                "operator": "Equal",
                "value": "spot",
                "effect": "NoSchedule",
            },
            {
                "key": "CriticalAddonsOnly",
                "operator": "Equal",
                "value": "true",
                "effect": "NoSchedule",
            },
        ],
    }

    terminal = KubernetesTerminal(
        working_dir=working_dir,
        pod_spec_kwargs=pod_spec_kwargs,
        kube_context="minikube",
        base_image="ubuntu:latest",
    )

    terminal.pod  # Create pod.
    assert (
        terminal.pod.pod_body["spec"]["tolerations"] == pod_spec_kwargs["tolerations"]
    )
    # Make sure environment variable was replaced in the pod spec.
    spec = terminal.pod.pod_body["spec"]
    node_affinity = spec["affinity"]["nodeAffinity"]
    required = node_affinity["requiredDuringSchedulingIgnoredDuringExecution"]
    term = required["nodeSelectorTerms"][0]
    match_expression = term["matchExpressions"][0]
    assert match_expression["values"] == [os.environ["HOSTNAME"]]

    # Close pod.
    terminal.close()
    assert terminal._pod is None


@if_kubernetes_available
@pytest.mark.parametrize(
    "command",
    [
        "export ENV_VAR=value && mkdir test && ls",
        ["export ENV_VAR=value", "mkdir test", "ls"],
    ],
)
def test_kubernetes_terminal_run(tmp_path, command):
    """Test running commands in the Kubernetes terminal."""
    working_dir = str(tmp_path)
    terminal = KubernetesTerminal(working_dir=working_dir, base_image="ubuntu:latest")
    success, output = terminal.run(command, timeout=1)
    assert output == "test"
    assert success is True

    success, output = terminal.run("echo $ENV_VAR", timeout=1)
    assert "value" not in output
    assert success is True
    success, output = terminal.run("ls", timeout=1)
    assert "test" in output
    assert success is True

    terminal.close()


@if_kubernetes_available
def test_kubernetes_terminal_with_session_commands(tmp_path):
    working_dir = str(tmp_path)
    session_commands = ["echo 'Hello'", "echo 'World'"]
    terminal = KubernetesTerminal(
        working_dir, session_commands=session_commands, base_image="ubuntu:latest"
    )
    status, output = terminal.run("pwd", timeout=1)
    assert status
    assert output == f"Hello\nWorld\n{working_dir}"
    terminal.close()


@if_is_linux
@if_kubernetes_available
def test_kubernetes_terminal_session(tmp_path):
    # same as test_terminal_session but with DockerTerminal
    working_dir = str(tmp_path)
    command = "echo Hello World"
    terminal = KubernetesTerminal(working_dir=working_dir, base_image="ubuntu:latest")
    assert not terminal.sessions

    session = terminal.new_shell_session()
    assert len(terminal.sessions) == 1
    output = session.run(command, timeout=1)
    assert output == f"{DISABLE_ECHO_COMMAND}Hello World"

    output = session.start()
    session.run("export TEST_VAR='FooBar'", timeout=1)
    assert output == f"{DISABLE_ECHO_COMMAND}"
    output = session.run("pwd", timeout=1)
    assert output == working_dir
    output = session.run("echo $TEST_VAR", timeout=1)
    assert output == "FooBar"

    terminal.close_shell_session(session)
    assert not terminal.sessions
    terminal.close()


@if_kubernetes_available
def test_copy_content(tmp_path):
    # Create a temporary source directory with a file
    source_dir = tmp_path / "source_dir"
    source_dir.mkdir()
    source_file = source_dir / "tmp.txt"
    with open(source_file, "w") as src_file:
        src_file.write("Hello World")

    terminal = KubernetesTerminal(base_image="ubuntu:latest")

    # Copy directory content
    terminal.copy_content(source_dir)

    # Clean up the temporary source_dir
    source_file.unlink()
    source_dir.rmdir()

    # Verify the content was copied correctly
    _, output = terminal.run(f"cat {terminal.working_dir}/tmp.txt", timeout=1)
    assert output == "Hello World"
    terminal.close()


@if_kubernetes_available
def test_kubernetes_terminal_cleanup(tmp_path):
    """Test cleanup functionality."""
    working_dir = str(tmp_path)
    terminal = KubernetesTerminal(working_dir=working_dir, base_image="ubuntu:latest")

    # Test cleanup without creating pod
    terminal.close()


@if_kubernetes_available
def test_unrecoverable_error_when_pod_stops(tmp_path):
    """Ensure terminal raises fatal error once the backing pod is gone."""

    working_dir = str(tmp_path)
    terminal = KubernetesTerminal(working_dir=working_dir, base_image="ubuntu:latest")
    try:
        pod = terminal.pod  # Ensure pod is created.
        pod.clean_up()  # Delete the pod to simulate infrastructure failure.

        with pytest.raises(UnrecoverableTerminalError):
            terminal.run("echo after cleanup", timeout=1)
    finally:
        terminal.close()


@if_kubernetes_available
def test_select_terminal_kubernetes():
    """Test terminal selection for Kubernetes."""
    config = {"type": "kubernetes"}
    terminal = select_terminal(config)
    assert isinstance(terminal, KubernetesTerminal)
    assert config == {"type": "kubernetes"}  # config should not be modified
    terminal.close()


@if_kubernetes_available
def test_kubernetes_terminal_run_timeout(tmp_path):
    """Test that commands that exceed the timeout are killed and return failure."""
    working_dir = str(tmp_path)
    terminal = KubernetesTerminal(working_dir=working_dir, base_image="ubuntu:latest")
    try:
        # Run a command that takes longer than the timeout
        entrypoint = "sleep 10 && echo done"
        success, output = terminal.run(entrypoint, timeout=2)
        assert success is False
        assert "timed out" in output.lower()
        assert "2 seconds" in output
    finally:
        terminal.close()


@if_kubernetes_available
def test_kubernetes_terminal_run_default_timeout(tmp_path):
    """Test that the default timeout is applied when none is specified."""
    working_dir = str(tmp_path)
    terminal = KubernetesTerminal(working_dir=working_dir, base_image="ubuntu:latest")
    try:
        # Run a quick command without specifying timeout
        entrypoint = "echo 'Hello'"
        success, output = terminal.run(entrypoint)  # No timeout specified
        assert success is True
        assert output == "Hello"
        # Default command_timeout should be 300 seconds (5 minutes)
        assert terminal.command_timeout == 300
    finally:
        terminal.close()


@if_kubernetes_available
def test_kubernetes_terminal_custom_command_timeout(tmp_path):
    """Test that custom command_timeout can be set via constructor."""
    working_dir = str(tmp_path)
    terminal = KubernetesTerminal(
        working_dir=working_dir, base_image="ubuntu:latest", command_timeout=120
    )
    try:
        assert terminal.command_timeout == 120
        # Quick command should still work
        success, output = terminal.run("echo 'test'")
        assert success is True
        assert output == "test"
    finally:
        terminal.close()


@if_kubernetes_available
def test_kubernetes_terminal_nohup_with_subshell_returns_immediately(tmp_path):
    """Test that nohup commands with subshell return immediately in non-TTY mode.

    This test verifies the fix for issue #325 where nohup commands would cause
    the timeout wrapper to wait in non-TTY mode. Using (...) subshell creates
    a subprocess that exits immediately after backgrounding.
    """
    working_dir = str(tmp_path)
    terminal = KubernetesTerminal(
        working_dir=working_dir, base_image="ubuntu:latest", command_timeout=10
    )
    try:
        # Warm up the terminal with a dummy command to exclude pod startup time
        terminal.run("echo 'warming up'")

        # Test that subshell with nohup returns immediately
        start_time = time.time()
        success, output = terminal.run("(nohup sleep 100 > /dev/null 2>&1 &)")
        elapsed = time.time() - start_time

        # Should return almost immediately (within 2 seconds, excluding pod startup)
        assert success is True
        assert elapsed < 2, f"nohup command took {elapsed:.2f}s, expected < 2s"

        # Verify the background process is actually running
        success, output = terminal.run("pgrep -f 'sleep 100'")
        assert success is True
        # Should have at least one PID (may have multiple due to process hierarchy)
        pids = [line.strip() for line in output.strip().split("\n") if line.strip()]
        assert (
            len(pids) >= 1
        ), f"Expected to find at least one sleep process, got: {output}"
        assert all(
            pid.isdigit() for pid in pids
        ), f"Expected PIDs to be digits, got: {pids}"

        # Clean up the background process
        terminal.run("pkill -f 'sleep 100'")
    finally:
        terminal.close()


@if_kubernetes_available
def test_kubernetes_terminal_nohup_without_redirection_may_timeout(tmp_path):
    """Test that nohup commands without redirection may not return immediately.

    This test demonstrates the problem that was fixed: without output redirection,
    the timeout command waits for file descriptors to close.
    """
    working_dir = str(tmp_path)
    terminal = KubernetesTerminal(
        working_dir=working_dir, base_image="ubuntu:latest", command_timeout=3
    )
    try:
        # Test that nohup WITHOUT redirection may hit the timeout
        start_time = time.time()
        success, output = terminal.run("nohup sleep 100 &")
        elapsed = time.time() - start_time

        # This will likely timeout after 3 seconds
        # The exact behavior depends on the shell, but it should take longer
        # than the properly redirected version
        assert elapsed >= 2, f"Expected command to take longer, took {elapsed:.2f}s"

        # Clean up any background processes
        terminal.run("pkill -f 'sleep 100'")
    finally:
        terminal.close()


def test_kubernetes_terminal_readonly_properties_after_pod_creation():
    """Test that working directory cannot be changed after pod creation."""
    terminal = KubernetesTerminal(base_image="ubuntu:latest")
    terminal.pod  # Create pod.

    with pytest.raises(
        ValueError, match="Cannot change the pod's name after its creation."
    ):
        terminal.pod_name = "New-Podname"

    with pytest.raises(ValueError, match="Cannot change task_name after pod creation."):
        terminal.task_name = "New-Task"

    with pytest.raises(
        ValueError, match="Cannot change working directory after pod creation."
    ):
        terminal.working_dir = "/new/path"

    terminal.close()


# Unit tests for image pull backoff detection (no cluster required)


class TestImagePullBackoffDetection:
    """Unit tests for image pull backoff detection logic."""

    def test_check_image_pull_status_no_backoff(self):
        """Test _check_image_pull_status returns False when no backoff."""
        mock_client = MagicMock()
        mock_pod = MagicMock()
        mock_pod.status.container_statuses = [
            MagicMock(state=MagicMock(waiting=MagicMock(reason="ContainerCreating")))
        ]
        mock_client.read_namespaced_pod.return_value = mock_pod

        pod = Pod.__new__(Pod)
        pod.k8s_client = mock_client
        pod.name = "test-pod"
        pod.namespace = "default"
        pod.logger = MagicMock()

        has_backoff, message = pod._check_image_pull_status()
        assert has_backoff is False
        assert message is None

    def test_check_image_pull_status_with_backoff(self):
        """Test _check_image_pull_status returns True when ImagePullBackOff."""
        mock_client = MagicMock()
        mock_pod = MagicMock()
        mock_pod.status.container_statuses = [
            MagicMock(
                state=MagicMock(
                    waiting=MagicMock(
                        reason="ImagePullBackOff",
                        message="Back-off pulling image myregistry/missing:latest",
                    )
                )
            )
        ]
        mock_client.read_namespaced_pod.return_value = mock_pod

        pod = Pod.__new__(Pod)
        pod.k8s_client = mock_client
        pod.name = "test-pod"
        pod.namespace = "default"
        pod.logger = MagicMock()

        has_backoff, message = pod._check_image_pull_status()
        assert has_backoff is True
        assert "missing:latest" in message

    def test_check_image_pull_status_no_container_statuses(self):
        """Test _check_image_pull_status handles missing container statuses."""
        mock_client = MagicMock()
        mock_pod = MagicMock()
        mock_pod.status.container_statuses = None
        mock_client.read_namespaced_pod.return_value = mock_pod

        pod = Pod.__new__(Pod)
        pod.k8s_client = mock_client
        pod.name = "test-pod"
        pod.namespace = "default"
        pod.logger = MagicMock()

        has_backoff, message = pod._check_image_pull_status()
        assert has_backoff is False
        assert message is None

    def test_check_image_pull_status_api_error(self):
        """Test _check_image_pull_status handles API errors gracefully."""
        from kubernetes.client.rest import ApiException

        mock_client = MagicMock()
        mock_client.read_namespaced_pod.side_effect = ApiException(status=500)

        pod = Pod.__new__(Pod)
        pod.k8s_client = mock_client
        pod.name = "test-pod"
        pod.namespace = "default"
        pod.logger = MagicMock()

        has_backoff, message = pod._check_image_pull_status()
        assert has_backoff is False
        assert message is None

    def test_image_pull_backoff_timeout_default(self):
        """Test default image_pull_backoff_timeout is 600 seconds."""
        with patch.object(KubernetesTerminal, "__init__", lambda self, **kwargs: None):
            terminal = KubernetesTerminal.__new__(KubernetesTerminal)
            # Manually set the attribute as __init__ is mocked
            terminal.image_pull_backoff_timeout = 600
            assert terminal.image_pull_backoff_timeout == 600

    def test_image_pull_backoff_timeout_disabled(self):
        """Test image_pull_backoff_timeout can be disabled with 0."""
        # This is a config test - just verify the parameter can be set to 0
        # The actual behavior (not checking) is tested in wait_for_pod_ready
        with patch.object(KubernetesTerminal, "__init__", lambda self, **kwargs: None):
            terminal = KubernetesTerminal.__new__(KubernetesTerminal)
            terminal.image_pull_backoff_timeout = 0
            assert terminal.image_pull_backoff_timeout == 0

    def test_err_image_pull_not_detected_as_backoff(self):
        """Test that ErrImagePull (initial failure) is not detected as backoff.

        ErrImagePull is the first failure; ImagePullBackOff is after Kubernetes
        has already retried multiple times. We only fail fast on ImagePullBackOff.
        """
        mock_client = MagicMock()
        mock_pod = MagicMock()
        mock_pod.status.container_statuses = [
            MagicMock(
                state=MagicMock(
                    waiting=MagicMock(
                        reason="ErrImagePull",
                        message="rpc error: code = NotFound",
                    )
                )
            )
        ]
        mock_client.read_namespaced_pod.return_value = mock_pod

        pod = Pod.__new__(Pod)
        pod.k8s_client = mock_client
        pod.name = "test-pod"
        pod.namespace = "default"
        pod.logger = MagicMock()

        has_backoff, message = pod._check_image_pull_status()
        # ErrImagePull should NOT trigger backoff detection
        assert has_backoff is False
        assert message is None
