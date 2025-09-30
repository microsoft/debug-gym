import tempfile
from pathlib import Path

import pytest

from debug_gym.gym.terminals import (
    DockerTerminal,
    LocalTerminal,
    ShellSession,
    select_terminal,
)
from debug_gym.gym.terminals.shell_session import DEFAULT_PS1


@pytest.fixture
def tmp_dir_prefix(tmp_path):
    """Expected tmp_dir_prefix for a terminal session."""
    tmp_dir = tempfile.TemporaryDirectory(prefix="Terminal-")
    tmp_dir_prefix = str(Path(tmp_dir.name).resolve()).split("Terminal-")[0]
    return tmp_dir_prefix


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


def test_terminal_init(tmp_dir_prefix):
    terminal = LocalTerminal()
    assert terminal.session_commands == []
    assert terminal.env_vars["NO_COLOR"] == "1"
    assert terminal.env_vars["PS1"] == DEFAULT_PS1
    assert len(terminal.env_vars) > 2  # NO_COLOR, PS1 + os env vars
    assert terminal.working_dir.startswith(tmp_dir_prefix)


def test_terminal_init_no_os_env_vars():
    terminal = LocalTerminal(include_os_env_vars=False)
    assert terminal.env_vars == {"NO_COLOR": "1", "PS1": DEFAULT_PS1}


def test_terminal_init_with_params(tmp_path):
    working_dir = str(tmp_path)
    session_commands = ["echo 'Hello World'"]
    env_vars = {"ENV_VAR": "value"}
    terminal = LocalTerminal(working_dir, session_commands, env_vars)
    assert terminal.working_dir == working_dir
    assert terminal.session_commands == session_commands
    assert terminal.env_vars["NO_COLOR"] == "1"
    assert terminal.env_vars["ENV_VAR"] == "value"
    status, output = terminal.run("pwd", timeout=1)
    assert status
    assert output == f"Hello World\n{working_dir}"
    status, output = terminal.run("echo $ENV_VAR", timeout=1)
    assert status
    assert output == "Hello World\nvalue"


def test_terminal_run(tmp_path):
    working_dir = str(tmp_path)
    terminal = LocalTerminal(working_dir=working_dir)
    entrypoint = "echo 'Hello World'"
    success, output = terminal.run(entrypoint, timeout=1)
    assert success is True
    assert output == "Hello World"
    assert terminal.working_dir == working_dir


def test_terminal_run_tmp_working_dir(tmp_path, tmp_dir_prefix):
    terminal = LocalTerminal()
    entrypoint = "echo 'Hello World'"
    success, output = terminal.run(entrypoint, timeout=1)
    assert success is True
    assert output == "Hello World"
    assert terminal.working_dir.startswith(tmp_dir_prefix)


def test_select_terminal_default():
    terminal = select_terminal(None)
    assert terminal is None
    terminal = select_terminal()
    assert terminal is None


def test_select_terminal_local():
    config = {"type": "local"}
    terminal = select_terminal(config)
    assert isinstance(terminal, LocalTerminal)
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


def test_shell_session_start_with_session_commands(tmp_path):
    terminal = LocalTerminal(
        working_dir=str(tmp_path),
        session_commands=["echo setup"],
    )
    session = terminal.new_shell_session()

    # Test starting without command
    output = session.start()
    assert output == "setup"  # from `echo setup` in session_commands
    assert session.is_running
    assert session.filedescriptor is not None
    assert session.process is not None
    output = session.run("echo Hello World")
    assert output == "Hello World"
    session.close()
    assert not session.is_running
    assert session.filedescriptor is None
    assert session.process is None

    # Test starting with command
    output = session.start("python", ">>>")
    assert output.startswith("setup\r\nPython 3.12")
    assert session.is_running
    assert session.filedescriptor is not None
    assert session.process is not None
    output = session.run("print('test python')", ">>>")
    assert output == "test python"
    session.close()


def test_shell_session_start_without_session_commands(tmp_path):
    terminal = LocalTerminal(working_dir=str(tmp_path))
    session = terminal.new_shell_session()

    # Test starting without command
    output = session.start()
    assert output == ""
    assert session.is_running
    assert session.filedescriptor is not None
    assert session.process is not None
    output = session.run("echo Hello World")
    assert output == "Hello World"
    session.close()
    assert not session.is_running
    assert session.filedescriptor is None
    assert session.process is None

    # Test starting with command
    output = session.start("python", ">>>")
    assert output.startswith("Python 3.12")
    assert session.is_running
    assert session.filedescriptor is not None
    assert session.process is not None
    output = session.run("print('test python')", ">>>")
    assert output == "test python"
    session.close()
