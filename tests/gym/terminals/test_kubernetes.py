import os
import platform
import subprocess
import time

import pytest

from debug_gym.gym.terminals import select_terminal
from debug_gym.gym.terminals.kubernetes import KubernetesTerminal
from debug_gym.gym.terminals.shell_session import DEFAULT_PS1
from debug_gym.gym.terminals.terminal import DISABLE_ECHO_COMMAND


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
    terminal = KubernetesTerminal()
    assert terminal.session_commands == []
    assert terminal.env_vars == {"NO_COLOR": "1", "PS1": DEFAULT_PS1}
    assert os.path.basename(terminal.working_dir).startswith("Terminal-")
    assert terminal.base_image == "ubuntu:latest"
    assert terminal.namespace == "default"
    assert terminal.pod_name.startswith("debug-gym.")
    # Pod should not be created until accessed
    assert not terminal._pod_created

    # Create pod.
    pod = terminal.pod
    assert pod is not None
    assert terminal.is_running()

    # Close pod.
    terminal.close()
    time.sleep(5 + 1)  # wait for pod to terminate, i.e., grace_period_seconds
    assert not terminal.is_running()


@if_kubernetes_available
def test_kubernetes_terminal_init_with_params(tmp_path):
    working_dir = str(tmp_path)
    session_commands = ["mkdir new_dir"]
    env_vars = {"ENV_VAR": "value"}
    base_image = "ubuntu:24.04"
    namespace = "test-namespace"
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
    assert terminal.env_vars == env_vars | {"NO_COLOR": "1", "PS1": DEFAULT_PS1}
    assert terminal.base_image == base_image
    assert terminal.namespace == namespace
    assert terminal.pod_name == pod_name

    # Create pod.
    pod = terminal.pod
    assert pod is not None
    assert terminal.is_running()

    # Close pod.
    terminal.close()
    time.sleep(5 + 1)  # wait for pod to terminate, i.e., grace_period_seconds
    assert not terminal.is_running()


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
    terminal = KubernetesTerminal(working_dir=working_dir)
    success, output = terminal.run(command, timeout=1)
    assert output == "test"
    assert success is True

    success, output = terminal.run("echo $ENV_VAR", timeout=1)
    assert "value" not in output
    assert success is True
    success, output = terminal.run("ls", timeout=1)
    assert "test" in output
    assert success is True


@if_kubernetes_available
def test_kubernetes_terminal_multiple_session_commands(tmp_path):
    working_dir = str(tmp_path)
    session_commands = ["echo 'Hello'", "echo 'World'"]
    terminal = KubernetesTerminal(working_dir, session_commands=session_commands)
    status, output = terminal.run("pwd", timeout=1)
    assert status
    assert output == f"Hello\nWorld\n{working_dir}"


@if_is_linux
@if_kubernetes_available
def test_kubernetes_terminal_session(tmp_path):
    # same as test_terminal_session but with DockerTerminal
    working_dir = str(tmp_path)
    command = "echo Hello World"
    terminal = KubernetesTerminal(working_dir=working_dir)
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


@if_kubernetes_available
def test_copy_content(tmp_path):
    # Create a temporary source file
    source_dir = tmp_path / "source_dir"
    source_dir.mkdir()
    source_file = source_dir / "tmp.txt"
    with open(source_file, "w") as src_file:
        src_file.write("Hello World")

    terminal = KubernetesTerminal()
    # Source must be a folder.
    with pytest.raises(ValueError, match="Source .* must be a directory."):
        terminal.copy_content(source_file)

    terminal.copy_content(source_dir)

    # Clean up the temporary source_dir
    source_file.unlink()
    source_dir.rmdir()

    # Verify the content was copied correctly
    _, output = terminal.run(f"cat {terminal.working_dir}/tmp.txt", timeout=1)
    assert output == "Hello World"


@if_kubernetes_available
def test_kubernetes_terminal_cleanup(tmp_path):
    """Test cleanup functionality."""
    working_dir = str(tmp_path)
    terminal = KubernetesTerminal(working_dir=working_dir)
    pod_name = terminal.pod
    assert terminal.is_running()

    # Test cleanup without creating pod
    terminal.clean_up()
    time.sleep(5 + 1)  # wait for pod to terminate, i.e., grace_period_seconds
    assert not terminal._pod_created
    assert not terminal.is_running()

    # Test that cleanup can be called multiple times safely
    terminal.clean_up()
    assert not terminal._pod_created
    assert not terminal.is_running()


@if_kubernetes_available
def test_select_terminal_kubernetes():
    """Test terminal selection for Kubernetes."""
    config = {"type": "kubernetes"}
    terminal = select_terminal(config)
    assert isinstance(terminal, KubernetesTerminal)
    assert config == {"type": "kubernetes"}  # config should not be modified


# @if_kubernetes_available
# def test_kubernetes_terminal_pod_naming():
#     """Test pod naming functionality."""
#     # Test default pod name generation
#     terminal1 = KubernetesTerminal()
#     assert terminal1.pod_name.startswith("debug-gym-")
#     assert len(terminal1.pod_name.split("-")) == 3  # debug-gym-<uuid>

#     # Test custom pod name
#     custom_name = "my-custom-pod"
#     terminal2 = KubernetesTerminal(pod_name=custom_name)
#     assert terminal2.pod_name == custom_name


# @if_kubernetes_available
# def test_kubernetes_terminal_namespace_handling():
#     """Test namespace configuration."""
#     # Test default namespace
#     terminal1 = KubernetesTerminal()
#     assert terminal1.namespace == "default"

#     # Test custom namespace
#     custom_namespace = "test-namespace"
#     terminal2 = KubernetesTerminal(namespace=custom_namespace)
#     assert terminal2.namespace == custom_namespace


def test_kubernetes_terminal_working_dir_readonly_after_pod_creation():
    """Test that working directory cannot be changed after pod creation."""
    terminal = KubernetesTerminal()
    # Simulate pod creation
    terminal.pod = True

    with pytest.raises(
        ValueError, match="Cannot change working directory while pod is running"
    ):
        terminal.working_dir = "/new/path"
