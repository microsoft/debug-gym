import os
import time

import docker
import pytest
from docker import errors as docker_errors

from debug_gym.gym.terminals import select_terminal
from debug_gym.gym.terminals.docker import DockerTerminal
from debug_gym.gym.terminals.shell_session import DEFAULT_PS1
from debug_gym.gym.terminals.terminal import (
    DISABLE_ECHO_COMMAND,
    UnrecoverableTerminalError,
)


@pytest.if_docker_running
def test_docker_terminal_init():
    terminal = DockerTerminal(base_image="ubuntu:latest")
    assert terminal.session_commands == []
    assert terminal.env_vars == {
        "NO_COLOR": "1",
        "PS1": DEFAULT_PS1,
        "PYTHONSTARTUP": "",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    assert os.path.basename(terminal.working_dir).startswith("Terminal-")
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
    docker_terminal = DockerTerminal(
        working_dir=working_dir, base_image="ubuntu:latest"
    )
    success, output = docker_terminal.run(command, timeout=1)
    assert output == "test"
    assert success is True

    success, output = docker_terminal.run("echo $ENV_VAR", timeout=1)
    assert "value" not in output
    assert success is True
    success, output = docker_terminal.run("ls", timeout=1)
    assert "test" in output
    assert success is True


@pytest.if_docker_running
def test_terminal_multiple_session_commands(tmp_path):
    working_dir = str(tmp_path)
    session_commands = ["echo 'Hello'", "echo 'World'"]
    terminal = DockerTerminal(working_dir, session_commands, base_image="ubuntu:latest")
    status, output = terminal.run("pwd", timeout=1)
    assert status
    assert output == f"Hello\nWorld\n{working_dir}"


@pytest.if_is_linux
@pytest.if_docker_running
def test_docker_terminal_session(tmp_path):
    # same as test_terminal_session but with DockerTerminal
    working_dir = str(tmp_path)
    command = "echo Hello World"
    terminal = DockerTerminal(working_dir=working_dir, base_image="ubuntu:latest")
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


@pytest.if_docker_running
def test_terminal_sudo_command(tmp_path):
    working_dir = str(tmp_path)
    terminal = DockerTerminal(working_dir=working_dir, base_image="ubuntu:latest")
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
    terminal = DockerTerminal(working_dir=working_dir, base_image="ubuntu:latest")
    container_name = terminal.container.name
    terminal.clean_up()
    assert terminal._container is None
    time.sleep(10)  # give docker some time to remove the container
    client = docker.from_env()
    containers = client.containers.list(all=True, ignore_removed=True)
    assert container_name not in [c.name for c in containers]


@pytest.if_docker_running
def test_select_terminal_docker():
    config = {"type": "docker"}
    terminal = select_terminal(config)
    assert isinstance(terminal, DockerTerminal)
    assert config == {"type": "docker"}  # config should not be modified


@pytest.if_docker_running
def test_run_setup_commands_success(tmp_path):
    working_dir = str(tmp_path)
    setup_commands = ["touch test1.txt", "echo test > test2.txt"]
    terminal = DockerTerminal(
        working_dir, setup_commands=setup_commands, base_image="ubuntu:latest"
    )
    assert terminal.container is not None
    assert terminal.container.status == "running"
    _, output = terminal.run("ls", timeout=1)
    assert output == "test1.txt\ntest2.txt"


@pytest.if_docker_running
def test_run_setup_commands_failure(tmp_path):
    working_dir = str(tmp_path)
    setup_commands = ["echo install", "ls ./non_existent_dir"]
    with pytest.raises(UnrecoverableTerminalError, match="Failed to run setup command"):
        terminal = DockerTerminal(
            working_dir, setup_commands=setup_commands, base_image="ubuntu:latest"
        )
        terminal.container  # start the container


@pytest.if_docker_running
def test_copy_content(tmp_path):
    # Create a temporary source file
    source_dir = tmp_path / "source_dir"
    source_dir.mkdir()
    source_file = source_dir / "tmp.txt"
    with open(source_file, "w") as src_file:
        src_file.write("Hello World")

    terminal = DockerTerminal(base_image="ubuntu:latest")
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


@pytest.if_docker_running
def test_unrecoverable_error_when_container_stops(tmp_path):
    working_dir = str(tmp_path)
    terminal = DockerTerminal(working_dir=working_dir, base_image="ubuntu:latest")
    try:
        container = terminal.container
        try:
            container.stop(timeout=1)
        except docker_errors.APIError:
            pass
        try:
            container.wait()
        except docker_errors.DockerException:
            pass

        with pytest.raises(UnrecoverableTerminalError):
            terminal.run("echo after stop", timeout=1)
    finally:
        terminal.clean_up()
