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
    terminal = KubernetesTerminal(base_image="ubuntu:latest")
    assert terminal.session_commands == []
    assert terminal.env_vars == {
        "NO_COLOR": "1",
        "PS1": DEFAULT_PS1,
        "PYTHONSTARTUP": "",
    }
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
    assert terminal.pod_name.startswith("dbg-gym.")
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
    assert terminal.env_vars == env_vars | {"NO_COLOR": "1", "PS1": DEFAULT_PS1}
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
    # Create a temporary source file
    source_dir = tmp_path / "source_dir"
    source_dir.mkdir()
    source_file = source_dir / "tmp.txt"
    with open(source_file, "w") as src_file:
        src_file.write("Hello World")

    terminal = KubernetesTerminal(base_image="ubuntu:latest")
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
    terminal.close()


@if_kubernetes_available
def test_kubernetes_terminal_cleanup(tmp_path):
    """Test cleanup functionality."""
    working_dir = str(tmp_path)
    terminal = KubernetesTerminal(working_dir=working_dir, base_image="ubuntu:latest")

    # Test cleanup without creating pod
    terminal.close()

    assert terminal.pod.is_running()
    assert terminal._pod is not None
    terminal.close()
    assert terminal._pod is None

    # Test that cleanup can be called multiple times safely
    terminal.close()


@if_kubernetes_available
def test_select_terminal_kubernetes():
    """Test terminal selection for Kubernetes."""
    config = {"type": "kubernetes"}
    terminal = select_terminal(config)
    assert isinstance(terminal, KubernetesTerminal)
    assert config == {"type": "kubernetes"}  # config should not be modified
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
