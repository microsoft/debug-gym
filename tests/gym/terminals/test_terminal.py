import pytest

from debug_gym.gym.terminals import DockerTerminal, select_terminal
from debug_gym.gym.terminals.shell_session import DEFAULT_PS1, ShellSession
from debug_gym.gym.terminals.terminal import MAX_LOG_OUTPUT_CHARS, Terminal


class TerminalWithoutWriteBytes(Terminal):
    @property
    def default_shell_command(self):
        return "/bin/sh"

    def prepare_command(self, entrypoint):
        return [entrypoint]

    def run(self, entrypoint, timeout=None, raises=False, strip_output=True):
        return True, ""

    def new_shell_session(self):
        return None

    def copy_content(self, src, target=None):
        return None


@pytest.if_is_linux
def test_shell_session_run(tmp_path):
    working_dir = str(tmp_path)
    shell_command = "/bin/bash --noprofile --norc"
    env_vars_1 = {"TEST_VAR": "TestVar"}
    session_1 = ShellSession(
        shell_command=shell_command,
        working_dir=working_dir,
        env_vars=env_vars_1,
    )
    session_2 = ShellSession(
        shell_command=shell_command,
        working_dir=working_dir,
    )

    assert session_1.shell_command == shell_command
    assert session_2.shell_command == shell_command

    assert session_1.working_dir == working_dir
    assert session_2.working_dir == working_dir

    assert session_1.env_vars == env_vars_1 | {"PS1": DEFAULT_PS1}
    assert session_2.env_vars == {"PS1": DEFAULT_PS1}

    output = session_1.run("echo Hello World", timeout=5)
    assert output == "Hello World"

    session_2.run("export TEST_VAR='FooBar'", timeout=5)
    output = session_2.run("echo $TEST_VAR", timeout=5)
    assert output == "FooBar"

    output = session_1.run("echo $TEST_VAR", timeout=5)
    assert output == "TestVar"


def test_shell_session_timeout(tmp_path):
    working_dir = str(tmp_path)
    # Write a long-running command to a file
    long_running_command = "sleep 60"

    shell = ShellSession(
        shell_command="/bin/bash --noprofile --norc",
        working_dir=working_dir,
    )

    timeout = 1
    with pytest.raises(
        TimeoutError,
        match=f"Read timeout after {timeout}",
    ):
        shell.run(long_running_command, timeout=timeout)
    assert shell.is_running is False


def test_select_terminal_default():
    terminal = select_terminal(None)
    assert terminal is None
    terminal = select_terminal()
    assert terminal is None


def test_terminal_bounds_output_for_logging():
    terminal = DockerTerminal(base_image="ubuntu:latest")
    output = "A" * (MAX_LOG_OUTPUT_CHARS + 1)

    assert terminal._output_for_logging("short output") == "short output"

    logged_output = terminal._output_for_logging(output)

    assert logged_output.startswith("A" * MAX_LOG_OUTPUT_CHARS)
    assert logged_output.endswith(f"[LOG OUTPUT TRUNCATED: {len(output)} chars]")


def test_select_terminal_local():
    config = {"type": "local"}
    with pytest.raises(
        ValueError,
        match="Local terminal is no longer supported. Use a Docker or Kubernetes terminal.",
    ):
        select_terminal(config)
    assert config == {"type": "local"}  # config should not be modified


@pytest.if_docker_running
def test_select_terminal_docker():
    config = {"type": "docker"}
    terminal = select_terminal(config)
    assert isinstance(terminal, DockerTerminal)
    assert config == {"type": "docker"}  # config should not be modified


def test_select_terminal_unknown():
    with pytest.raises(ValueError, match="Unknown terminal unknown"):
        select_terminal({"type": "unknown"})


def test_select_terminal_invalid_config():
    with pytest.raises(TypeError):
        select_terminal("not a dict")


def test_custom_terminal_must_implement_confined_write_bytes():
    with pytest.raises(TypeError, match="write_bytes"):
        TerminalWithoutWriteBytes()


def test_select_terminal_kubernetes_extra_labels(monkeypatch):
    captured = {}

    class DummyK8s:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(
        "debug_gym.gym.terminals.KubernetesTerminal",
        DummyK8s,
    )

    config = {
        "type": "kubernetes",
        "namespace": "example",
        "extra_labels": {"foo": "bar"},
        "pod_spec_kwargs": {"tolerations": []},
    }

    terminal = select_terminal(config, uuid="1234")

    assert isinstance(terminal, DummyK8s)
    assert captured["namespace"] == "example"
    assert captured["pod_spec_kwargs"] == {"tolerations": []}
    assert captured["extra_labels"] == {"foo": "bar", "uuid": "1234"}
    assert "logger" in captured
    assert config == {
        "type": "kubernetes",
        "namespace": "example",
        "extra_labels": {"foo": "bar"},
        "pod_spec_kwargs": {"tolerations": []},
    }


def test_select_terminal_docker_extra_labels(monkeypatch):
    """Test that extra_labels are passed to DockerTerminal."""
    captured = {}

    class DummyDocker:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(
        "debug_gym.gym.terminals.DockerTerminal",
        DummyDocker,
    )

    config = {
        "type": "docker",
        "base_image": "ubuntu:latest",
        "extra_labels": {"run-id": "my-run"},
    }

    terminal = select_terminal(config, uuid="1234")

    assert isinstance(terminal, DummyDocker)
    assert captured["base_image"] == "ubuntu:latest"
    assert captured["extra_labels"] == {"run-id": "my-run", "uuid": "1234"}
    assert "logger" in captured
    # Original config should not be modified
    assert config == {
        "type": "docker",
        "base_image": "ubuntu:latest",
        "extra_labels": {"run-id": "my-run"},
    }
