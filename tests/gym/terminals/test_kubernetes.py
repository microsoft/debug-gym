import hashlib
import os
import platform
import subprocess
import time
from pathlib import Path, PurePosixPath
from unittest.mock import Mock, patch

import pytest

from debug_gym.gym.terminals import select_terminal
from debug_gym.gym.terminals.kubernetes import KubernetesTerminal
from debug_gym.gym.terminals.shell_session import DEFAULT_PS1
from debug_gym.gym.terminals.terminal import (
    DISABLE_ECHO_COMMAND,
    TerminalError,
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

    # Accessing pod_name triggers lazy pod creation.
    assert terminal.pod_name.startswith("dbg-gym-")
    assert terminal._pod is not None
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
def test_write_text_preserves_untrusted_content():
    terminal = KubernetesTerminal(base_image="python:3.12-slim", working_dir="/testbed")
    content = "DEBUGGYM_EOF\n$(touch /tmp/side-effect)\nUnicode: 世界 🚀\n" + (
        "x" * (2 * 1024 * 1024)
    )
    expected_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
    try:
        terminal.run("touch /testbed/payload.txt && chmod 0755 /testbed/payload.txt")
        terminal.write_text(f"{terminal.working_dir}/payload.txt", content)

        success, output = terminal.run(
            'python -c "from pathlib import Path; import hashlib; '
            "print(hashlib.sha256(Path('payload.txt').read_bytes()).hexdigest())\""
        )
        assert success
        assert output == expected_hash
        assert terminal.run('test "$(stat -c %a /testbed/payload.txt)" = 755')[0]
        assert terminal.run("test ! -e /tmp/side-effect")[0]

        terminal.run("mkdir -p /tmp/outside && ln -s /tmp/outside /testbed/escape")
        with pytest.raises(TerminalError, match="symbolic-link"):
            terminal.write_text("/testbed/escape/payload.txt", "blocked")
        assert terminal.run("test ! -e /tmp/outside/payload.txt")[0]

        terminal.run("ln -s /tmp/outside-target /testbed/linked-file")
        with pytest.raises(TerminalError, match="symbolic-link"):
            terminal.write_text("/testbed/linked-file", "blocked")
        assert terminal.run("test ! -e /tmp/outside-target")[0]

        terminal.run("mkdir -p /testbed/real && ln -s /testbed/real /testbed/alias")
        with pytest.raises(TerminalError, match="symbolic-link"):
            terminal.write_text("/testbed/alias/in-root.txt", "blocked")
        assert terminal.run("test ! -e /testbed/real/in-root.txt")[0]

        terminal.run("ln -s /testbed /testbed-root")
        terminal._working_dir = "/testbed-root"
        with pytest.raises(TerminalError, match="symbolic-link"):
            terminal.write_text("/testbed-root/root-link.txt", "blocked")
        terminal._working_dir = "/testbed"
        assert terminal.run("test ! -e /testbed/root-link.txt")[0]

        terminal.run(
            "printf outside > /tmp/outside-hard-link && "
            "ln /tmp/outside-hard-link /testbed/hard-link"
        )
        terminal.write_text("/testbed/hard-link", "inside")
        assert terminal.run(
            'test "$(cat /tmp/outside-hard-link)" = outside && '
            'test "$(cat /testbed/hard-link)" = inside'
        )[0]
    finally:
        terminal.close()


def test_write_bytes_transfers_content_out_of_band():
    terminal = KubernetesTerminal.__new__(KubernetesTerminal)
    terminal._working_dir = "/testbed"
    terminal._pod = Mock()
    terminal._pod.is_running.return_value = True
    terminal._pod.name = "test-pod"
    terminal._pod.namespace = "default"
    terminal.sessions = []
    terminal.logger = Mock()
    terminal._destination_metadata = Mock(
        return_value=(0o644, 1000, 1000, 1000, 1000, False)
    )
    terminal.run = Mock(
        side_effect=lambda command, **kwargs: (
            (True, "")
            if command.startswith(("mkdir ", "chmod ", "rm "))
            else (False, "")
        )
    )
    content = b"DEBUGGYM_EOF\n$(touch /tmp/side-effect)\n"

    def assert_copy_content(src, target):
        assert Path(src).read_bytes() == content
        assert "DEBUGGYM_EOF" not in target

    terminal._kubectl_copy = Mock(side_effect=assert_copy_content)
    try:
        terminal.write_bytes("/testbed/nested/payload.txt", content)

        terminal._kubectl_copy.assert_called_once()
        assert all(
            "DEBUGGYM_EOF" not in call.args[0] for call in terminal.run.call_args_list
        )
        replacement_commands = [
            call.args[0]
            for call in terminal.run.call_args_list
            if call.args[0].startswith("chmod ")
        ]
        assert len(replacement_commands) == 1
        assert "mv -f --" in replacement_commands[0]
        assert ".debug-gym-write-" in replacement_commands[0]
    finally:
        terminal._pod = None


def test_destination_metadata_strips_privileged_bits():
    terminal = KubernetesTerminal.__new__(KubernetesTerminal)
    terminal._pod = None
    terminal.sessions = []
    terminal.run = Mock(
        side_effect=[
            (True, "0"),
            (True, "0"),
            (True, "0"),
            (True, ""),
            (True, ""),
            (True, "6755 0 0"),
        ]
    )

    assert terminal._destination_metadata(PurePosixPath("/testbed/privileged")) == (
        0o755,
        0,
        0,
        0,
        0,
        True,
    )


def test_destination_metadata_honors_pod_umask():
    terminal = KubernetesTerminal.__new__(KubernetesTerminal)
    terminal._pod = None
    terminal.sessions = []
    terminal.run = Mock(
        side_effect=[
            (True, "1000"),
            (True, "1000"),
            (True, "1000 1234"),
            (False, ""),
            (True, "0077"),
            (True, "2775 1234"),
        ]
    )

    assert terminal._destination_metadata(PurePosixPath("/testbed/new")) == (
        0o600,
        1000,
        1234,
        1000,
        1000,
        False,
    )


def test_destination_metadata_accepts_supplementary_group():
    terminal = KubernetesTerminal.__new__(KubernetesTerminal)
    terminal._pod = None
    terminal.sessions = []
    terminal.run = Mock(
        side_effect=[
            (True, "1000"),
            (True, "1000"),
            (True, "1000 1234"),
            (True, ""),
            (True, ""),
            (True, "0660 1000 1234"),
        ]
    )

    assert terminal._destination_metadata(PurePosixPath("/testbed/shared")) == (
        0o660,
        1000,
        1234,
        1000,
        1000,
        True,
    )


def test_kubectl_copy_uses_configured_context():
    terminal = KubernetesTerminal.__new__(KubernetesTerminal)
    terminal.kube_config = "/tmp/kubeconfig"
    terminal.kube_context = "isolated-context"
    terminal._pod = Mock()
    terminal._pod.name = "test-pod"
    terminal._pod.namespace = "default"
    terminal.logger = Mock()
    terminal.sessions = []

    with patch("subprocess.run", return_value=Mock(returncode=0)) as run:
        terminal._kubectl_copy("/tmp/source", "/testbed/target")

    command = run.call_args.args[0]
    assert command[:5] == [
        "kubectl",
        "--kubeconfig",
        "/tmp/kubeconfig",
        "--context",
        "isolated-context",
    ]
    terminal._pod = None


def test_interactive_shell_uses_configured_context():
    terminal = KubernetesTerminal.__new__(KubernetesTerminal)
    terminal.kube_config = "/tmp/kube config"
    terminal.kube_context = "isolated-context"
    terminal._pod = Mock()
    terminal._pod.name = "test-pod"
    terminal._pod.namespace = "default"
    terminal.sessions = []

    command = terminal.default_shell_command

    assert "--kubeconfig '/tmp/kube config'" in command
    assert "--context isolated-context" in command
    terminal._pod = None


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
