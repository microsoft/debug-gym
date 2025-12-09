import os
import platform
import subprocess
import time
from unittest.mock import MagicMock, patch

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


# Tests for pod spec shortcut functionality (don't require Kubernetes cluster)


def test_build_pod_spec_from_shortcuts_affinity_same_host():
    """Test building pod spec with same_host affinity mode."""
    from debug_gym.gym.terminals.kubernetes import _build_pod_spec_from_shortcuts

    spec = _build_pod_spec_from_shortcuts(
        affinity_mode="same_host", affinity_hostname_key="MY_HOST"
    )

    assert "affinity" in spec
    node_affinity = spec["affinity"]["nodeAffinity"]
    required = node_affinity["requiredDuringSchedulingIgnoredDuringExecution"]
    term = required["nodeSelectorTerms"][0]
    match_expr = term["matchExpressions"][0]

    assert match_expr["key"] == "kubernetes.io/hostname"
    assert match_expr["operator"] == "In"
    assert match_expr["values"] == ["{{MY_HOST}}"]


def test_build_pod_spec_from_shortcuts_tolerations_single():
    """Test building pod spec with a single toleration preset."""
    from debug_gym.gym.terminals.kubernetes import _build_pod_spec_from_shortcuts

    spec = _build_pod_spec_from_shortcuts(tolerations_preset="spot")

    assert "tolerations" in spec
    assert len(spec["tolerations"]) == 1
    assert spec["tolerations"][0]["key"] == "kubernetes.azure.com/scalesetpriority"
    assert spec["tolerations"][0]["value"] == "spot"


def test_build_pod_spec_from_shortcuts_tolerations_multiple():
    """Test building pod spec with multiple toleration presets."""
    from debug_gym.gym.terminals.kubernetes import _build_pod_spec_from_shortcuts

    spec = _build_pod_spec_from_shortcuts(tolerations_preset=["spot", "critical"])

    assert "tolerations" in spec
    assert len(spec["tolerations"]) == 2

    keys = [t["key"] for t in spec["tolerations"]]
    assert "kubernetes.azure.com/scalesetpriority" in keys
    assert "CriticalAddonsOnly" in keys


def test_build_pod_spec_from_shortcuts_combined():
    """Test building pod spec with both affinity and tolerations."""
    from debug_gym.gym.terminals.kubernetes import _build_pod_spec_from_shortcuts

    spec = _build_pod_spec_from_shortcuts(
        affinity_mode="same_host",
        tolerations_preset=["spot", "critical"],
    )

    assert "affinity" in spec
    assert "tolerations" in spec
    assert len(spec["tolerations"]) == 2


def test_build_pod_spec_from_shortcuts_invalid_affinity_mode():
    """Test that invalid affinity_mode raises ValueError."""
    from debug_gym.gym.terminals.kubernetes import _build_pod_spec_from_shortcuts

    with pytest.raises(ValueError, match="Unknown affinity_mode 'invalid'"):
        _build_pod_spec_from_shortcuts(affinity_mode="invalid")


def test_build_pod_spec_from_shortcuts_invalid_tolerations_preset():
    """Test that invalid tolerations_preset raises ValueError."""
    from debug_gym.gym.terminals.kubernetes import _build_pod_spec_from_shortcuts

    with pytest.raises(ValueError, match="Unknown tolerations_preset 'invalid'"):
        _build_pod_spec_from_shortcuts(tolerations_preset="invalid")


def test_deep_merge_dicts():
    """Test deep merging of dictionaries."""
    from debug_gym.gym.terminals.kubernetes import _deep_merge_dicts

    base = {
        "a": 1,
        "b": {"c": 2, "d": 3},
        "e": [1, 2],
    }
    override = {
        "b": {"c": 10, "f": 4},
        "e": [3, 4],
        "g": 5,
    }

    result = _deep_merge_dicts(base, override)

    assert result["a"] == 1  # Unchanged from base
    assert result["b"]["c"] == 10  # Override wins
    assert result["b"]["d"] == 3  # Preserved from base
    assert result["b"]["f"] == 4  # Added from override
    assert result["e"] == [3, 4]  # Override replaces list
    assert result["g"] == 5  # Added from override


@patch("debug_gym.gym.terminals.kubernetes.config.load_kube_config")
@patch("debug_gym.gym.terminals.kubernetes.client.CoreV1Api")
def test_kubernetes_terminal_with_affinity_mode(mock_api, mock_config, monkeypatch):
    """Test KubernetesTerminal initialization with affinity_mode shortcut."""
    monkeypatch.setenv("MY_NODE", "test-node-1")
    terminal = KubernetesTerminal(
        base_image="ubuntu:latest",
        affinity_mode="same_host",
        affinity_hostname_key="MY_NODE",
    )

    assert "affinity" in terminal.pod_spec_kwargs
    node_affinity = terminal.pod_spec_kwargs["affinity"]["nodeAffinity"]
    required = node_affinity["requiredDuringSchedulingIgnoredDuringExecution"]
    term = required["nodeSelectorTerms"][0]
    match_expr = term["matchExpressions"][0]
    assert match_expr["values"] == ["{{MY_NODE}}"]


@patch("debug_gym.gym.terminals.kubernetes.config.load_kube_config")
@patch("debug_gym.gym.terminals.kubernetes.client.CoreV1Api")
def test_kubernetes_terminal_with_tolerations_preset(mock_api, mock_config):
    """Test KubernetesTerminal initialization with tolerations_preset shortcut."""
    terminal = KubernetesTerminal(
        base_image="ubuntu:latest",
        tolerations_preset=["spot", "critical"],
    )

    assert "tolerations" in terminal.pod_spec_kwargs
    assert len(terminal.pod_spec_kwargs["tolerations"]) == 2


@patch("debug_gym.gym.terminals.kubernetes.config.load_kube_config")
@patch("debug_gym.gym.terminals.kubernetes.client.CoreV1Api")
def test_kubernetes_terminal_shortcuts_with_explicit_pod_spec_kwargs(
    mock_api, mock_config
):
    """Test that explicit pod_spec_kwargs override shortcut-generated values."""
    custom_tolerations = [{"key": "custom", "value": "value", "effect": "NoSchedule"}]
    terminal = KubernetesTerminal(
        base_image="ubuntu:latest",
        tolerations_preset="spot",  # This would normally add spot toleration
        pod_spec_kwargs={"tolerations": custom_tolerations},  # But explicit overrides
    )

    # Explicit pod_spec_kwargs should override the shortcut-generated tolerations
    assert terminal.pod_spec_kwargs["tolerations"] == custom_tolerations


@patch("debug_gym.gym.terminals.kubernetes.config.load_kube_config")
@patch("debug_gym.gym.terminals.kubernetes.client.CoreV1Api")
def test_kubernetes_terminal_shortcuts_merge_with_pod_spec_kwargs(
    mock_api, mock_config
):
    """Test that shortcuts and pod_spec_kwargs are properly merged."""
    terminal = KubernetesTerminal(
        base_image="ubuntu:latest",
        affinity_mode="same_host",  # Adds affinity
        pod_spec_kwargs={"nodeSelector": {"disktype": "ssd"}},  # Adds nodeSelector
    )

    # Both should be present
    assert "affinity" in terminal.pod_spec_kwargs
    assert "nodeSelector" in terminal.pod_spec_kwargs
    assert terminal.pod_spec_kwargs["nodeSelector"]["disktype"] == "ssd"
