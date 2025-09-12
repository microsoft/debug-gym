import re
import tempfile
import time
from pathlib import Path

import docker
import pytest

from debug_gym.gym.terminal import (
    DEFAULT_PS1,
    DISABLE_ECHO_COMMAND,
    DockerTerminal,
    ShellSession,
    Terminal,
    select_terminal,
)


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
    terminal = Terminal()
    assert terminal.session_commands == []
    assert terminal.env_vars["NO_COLOR"] == "1"
    assert terminal.env_vars["PS1"] == DEFAULT_PS1
    assert len(terminal.env_vars) > 2  # NO_COLOR, PS1 + os env vars
    assert terminal.working_dir.startswith(tmp_dir_prefix)


def test_terminal_init_no_os_env_vars():
    terminal = Terminal(include_os_env_vars=False)
    assert terminal.env_vars == {"NO_COLOR": "1", "PS1": DEFAULT_PS1}


def test_terminal_init_with_params(tmp_path):
    working_dir = str(tmp_path)
    session_commands = ["echo 'Hello World'"]
    env_vars = {"ENV_VAR": "value"}
    terminal = Terminal(working_dir, session_commands, env_vars)
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
    terminal = Terminal(working_dir=working_dir)
    entrypoint = "echo 'Hello World'"
    success, output = terminal.run(entrypoint, timeout=1)
    assert success is True
    assert output == "Hello World"
    assert terminal.working_dir == working_dir


def test_terminal_run_tmp_working_dir(tmp_path, tmp_dir_prefix):
    terminal = Terminal()
    entrypoint = "echo 'Hello World'"
    success, output = terminal.run(entrypoint, timeout=1)
    assert success is True
    assert output == "Hello World"
    assert terminal.working_dir.startswith(tmp_dir_prefix)


@pytest.mark.parametrize(
    "command",
    [
        ["echo Hello", "echo World"],
        "echo Hello && echo World",
    ],
)
def test_terminal_run_multiple_commands(tmp_path, command):
    working_dir = str(tmp_path)
    terminal = Terminal(working_dir=working_dir)
    success, output = terminal.run(command, timeout=1)
    assert success is True
    assert output == "Hello\nWorld"


def test_terminal_run_failure(tmp_path):
    working_dir = str(tmp_path)
    terminal = Terminal(working_dir=working_dir)
    entrypoint = "ls non_existent_dir"
    success, output = terminal.run(entrypoint, timeout=1)
    assert success is False
    # Linux: "ls: cannot access 'non_existent_dir': No such file or directory"
    # MacOS: "ls: non_existent_dir: No such file or directory"
    pattern = r"ls:.*non_existent_dir.*No such file or directory"
    assert re.search(pattern, output)


def test_terminal_session(tmp_path):
    working_dir = str(tmp_path)
    command = "echo Hello World"
    terminal = Terminal(working_dir=working_dir)
    assert not terminal.sessions

    session = terminal.new_shell_session()
    assert len(terminal.sessions) == 1
    output = session.run(command, timeout=1)
    assert output == "Hello World"

    session.run("export TEST_VAR='FooBar'", timeout=1)
    output = session.run("pwd", timeout=1)
    assert output == working_dir
    output = session.run("echo $TEST_VAR", timeout=1)
    assert output == "FooBar"

    terminal.close_shell_session(session)
    assert not terminal.sessions


@pytest.if_docker_running
def test_docker_terminal_init(tmp_dir_prefix):
    terminal = DockerTerminal()
    assert terminal.session_commands == []
    assert terminal.env_vars == {"NO_COLOR": "1", "PS1": DEFAULT_PS1}
    assert terminal.working_dir.startswith(tmp_dir_prefix)
    assert terminal.base_image == "ubuntu:latest"
    assert terminal.container is not None
    assert terminal.container.status == "running"


@pytest.if_docker_running
def test_docker_terminal_init_with_params(tmp_path):
    working_dir = str(tmp_path)
    session_commands = ["mkdir new_dir"]
    env_vars = {"ENV_VAR": "value"}
    base_image = "ubuntu:24.04"
    terminal = DockerTerminal(
        working_dir=working_dir,
        session_commands=session_commands,
        env_vars=env_vars,
        base_image=base_image,
    )
    assert terminal.working_dir == working_dir
    assert terminal.session_commands == session_commands
    assert terminal.env_vars == env_vars | {"NO_COLOR": "1", "PS1": DEFAULT_PS1}
    assert terminal.base_image == base_image
    assert terminal.container.status == "running"

    _, output = terminal.run("pwd", timeout=1)
    assert output == working_dir

    _, output = terminal.run("ls -l", timeout=1)
    assert "new_dir" in output


@pytest.if_docker_running
@pytest.mark.parametrize(
    "command",
    [
        "export ENV_VAR=value && mkdir test && ls",
        ["export ENV_VAR=value", "mkdir test", "ls"],
    ],
)
def test_docker_terminal_run(tmp_path, command):
    working_dir = str(tmp_path)
    docker_terminal = DockerTerminal(working_dir=working_dir)
    success, output = docker_terminal.run(command, timeout=1)
    assert output == "test"
    assert success is True

    success, output = docker_terminal.run("echo $ENV_VAR", timeout=1)
    assert "value" not in output
    assert success is True
    success, output = docker_terminal.run("ls", timeout=1)
    assert "test" in output
    assert success is True


@pytest.if_is_linux
@pytest.if_docker_running
def test_docker_terminal_session(tmp_path):
    # same as test_terminal_session but with DockerTerminal
    working_dir = str(tmp_path)
    command = "echo Hello World"
    terminal = DockerTerminal(working_dir=working_dir)
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


@pytest.mark.parametrize(
    "terminal_cls",
    [
        Terminal,
        pytest.param(DockerTerminal, marks=pytest.if_docker_running),
    ],
)
def test_terminal_multiple_session_commands(tmp_path, terminal_cls):
    working_dir = str(tmp_path)
    session_commands = ["echo 'Hello'", "echo 'World'"]
    terminal = terminal_cls(working_dir, session_commands)
    status, output = terminal.run("pwd", timeout=1)
    assert status
    assert output == f"Hello\nWorld\n{working_dir}"


@pytest.if_docker_running
def test_terminal_sudo_command(tmp_path):
    working_dir = str(tmp_path)
    terminal = DockerTerminal(working_dir=working_dir)
    success, output = terminal.run("vim --version", timeout=1)
    assert "vim: command not found" in output
    assert success is False
    success, output = terminal.run(
        "apt update && apt install -y sudo && sudo apt install -y vim"
    )
    assert success is True
    success, output = terminal.run("vim --version", timeout=1)
    assert success is True
    assert "VIM - Vi IMproved" in output


@pytest.if_docker_running
def test_terminal_cleanup(tmp_path):
    working_dir = str(tmp_path)
    terminal = DockerTerminal(working_dir=working_dir)
    container_name = terminal.container.name
    terminal.clean_up()
    assert terminal._container is None
    time.sleep(10)  # give docker some time to remove the container
    client = docker.from_env()
    containers = client.containers.list(all=True, ignore_removed=True)
    assert container_name not in [c.name for c in containers]


def test_select_terminal_default():
    terminal = select_terminal(None)
    assert terminal is None
    terminal = select_terminal()
    assert terminal is None


def test_select_terminal_local():
    config = {"type": "local"}
    terminal = select_terminal(config)
    assert isinstance(terminal, Terminal)
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
    terminal = Terminal(
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
    terminal = Terminal(working_dir=str(tmp_path))
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


@pytest.if_docker_running
def test_run_setup_commands_success(tmp_path):
    working_dir = str(tmp_path)
    setup_commands = ["touch test1.txt", "echo test > test2.txt"]
    terminal = DockerTerminal(working_dir, setup_commands=setup_commands)
    assert terminal.container is not None
    assert terminal.container.status == "running"
    _, output = terminal.run("ls", timeout=1)
    assert output == "test1.txt\ntest2.txt"


@pytest.if_docker_running
def test_run_setup_commands_failure(tmp_path):
    working_dir = str(tmp_path)
    setup_commands = ["echo install", "ls ./non_existent_dir"]
    with pytest.raises(ValueError, match="Failed to run setup command:*"):
        terminal = DockerTerminal(working_dir, setup_commands=setup_commands)
        terminal.container  # start the container
