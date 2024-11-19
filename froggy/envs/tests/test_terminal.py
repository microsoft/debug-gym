import subprocess

import pytest

from froggy.envs.terminal import DockerTerminal, Terminal

if_docker_running = pytest.mark.skipif(
    not subprocess.check_output("docker ps", shell=True),
    reason="Docker not running",
)


def test_terminal_init():
    terminal = Terminal()
    assert terminal.setup_commands == []
    assert terminal.env_vars == {"NO_COLOR": "1", "PS1": ""}
    assert terminal.working_dir == "/tmp/Froggy"


def test_terminal_init_with_params(tmp_path):
    working_dir = str(tmp_path)
    setup_commands = ["echo 'Hello World'"]
    env_vars = {"ENV_VAR": "value"}
    terminal = Terminal(working_dir, setup_commands, env_vars)
    assert terminal.working_dir == working_dir
    assert terminal.setup_commands == setup_commands
    assert terminal.env_vars == env_vars | {"NO_COLOR": "1"}
    output = terminal.run(["pwd"])
    assert output == working_dir
    output = terminal.run(["echo", "$ENV_VAR"])
    assert output == "value"


def test_terminal_run(tmp_path):
    working_dir = str(tmp_path)
    terminal = Terminal()
    entrypoint = ["echo", "Hello World"]
    success, output = terminal.run(entrypoint, working_dir)
    assert success is True
    assert output == "Hello World"


def test_terminal_run_failure(tmp_path):
    working_dir = str(tmp_path)
    terminal = Terminal()
    entrypoint = ["ls", "non_existent_dir"]
    success, output = terminal.run(entrypoint, working_dir)
    assert success is False
    assert output == ("ls: cannot access 'non_existent_dir': No such file or directory")


def test_terminal_pseudo_terminal(tmp_path):
    working_dir = str(tmp_path)
    command = ["echo", "Hello World"]
    terminal = Terminal(working_dir=working_dir)
    assert terminal.has_pseudo_terminal() is False

    terminal.start_pseudo_terminal(timeout=1)
    assert terminal.has_pseudo_terminal() is True
    output = terminal.run_interactive(command, timeout=1)
    assert output == "Hello World"

    terminal.run_interactive(["export", "TEST_VAR='FooBar'"], timeout=1)
    output = terminal.run_interactive(["pwd"], timeout=1)
    assert output == working_dir
    output = terminal.run_interactive(["echo", "$TEST_VAR"], timeout=1)
    assert output == "FooBar"

    terminal.close_pseudo_terminal()

    # starts the pseudo terminal automatically
    output = terminal.run_interactive(command, timeout=1)
    assert terminal.has_pseudo_terminal() is True
    assert output == "Hello World"

    terminal.close_pseudo_terminal()
    assert terminal.has_pseudo_terminal() is False


@if_docker_running
def test_docker_terminal_init():
    terminal = DockerTerminal()
    assert terminal.setup_commands == []
    assert terminal.env_vars == {"NO_COLOR": "1", "PS1": ""}
    assert terminal.working_dir == "/"
    assert terminal.base_image == "ubuntu:latest"
    assert terminal.volumes == {}
    assert terminal.container is not None
    assert terminal.container.status == "created"


@if_docker_running
def test_terminal_init_with_params(tmp_path):
    working_dir = str(tmp_path)
    setup_commands = ["mkdir new_dir"]
    env_vars = {"ENV_VAR": "value"}
    base_image = "ubuntu:24.04"
    volumes = {working_dir: {"bind": working_dir, "mode": "rw"}}
    terminal = DockerTerminal(
        working_dir=working_dir,
        setup_commands=setup_commands,
        env_vars=env_vars,
        base_image=base_image,
        volumes=volumes,
    )
    assert terminal.working_dir == working_dir
    assert terminal.setup_commands == setup_commands
    assert terminal.env_vars == env_vars | {"NO_COLOR": "1", "PS1": ""}
    assert terminal.base_image == base_image
    assert terminal.volumes == volumes
    assert terminal.container.status == "created"

    _, output = terminal.run(["pwd"])
    assert output == working_dir

    _, output = terminal.run(["ls", "-l"])
    assert "new_dir" in output


@if_docker_running
def test_docker_terminal_run(tmp_path):
    working_dir = str(tmp_path)
    volumes = {working_dir: {"bind": working_dir, "mode": "rw"}}
    docker_terminal = DockerTerminal(working_dir=working_dir, volumes=volumes)
    success, output = docker_terminal.run("export ENV_VAR=value && mkdir test && ls")
    assert success is True
    assert output == "test"

    success, output = docker_terminal.run("echo $ENV_VAR")
    assert success is True
    assert "value" not in output
    success, output = docker_terminal.run("ls")
    assert success is True
    assert "test" in output


@if_docker_running
def test_docker_terminal_read_only_volume(tmp_path):
    with open(tmp_path / "test.txt", "w") as f:
        f.write("test")

    working_dir = str(tmp_path)
    volumes = {working_dir: {"bind": working_dir, "mode": "ro"}}
    docker_terminal = DockerTerminal(working_dir=working_dir, volumes=volumes)
    success, ls_output = docker_terminal.run("ls")
    assert success is True
    assert ls_output.startswith("test.txt")

    success, output = docker_terminal.run("touch test2.txt")
    assert success is False
    assert output == "touch: cannot touch 'test2.txt': Read-only file system"


@if_docker_running
def test_docker_terminal_pseudo_terminal(tmp_path):
    # same as test_terminal_pseudo_terminal but with DockerTerminal
    working_dir = str(tmp_path)
    volumes = {working_dir: {"bind": working_dir, "mode": "rw"}}
    command = ["echo", "Hello World"]
    terminal = DockerTerminal(working_dir=working_dir, volumes=volumes)
    assert terminal.has_pseudo_terminal() is False

    terminal.start_pseudo_terminal(timeout=1)
    assert terminal.has_pseudo_terminal() is True
    output = terminal.run_interactive(command, timeout=1)
    assert output == "Hello World"

    terminal.run_interactive(["export", "TEST_VAR='FooBar'"], timeout=1)
    output = terminal.run_interactive(["pwd"], timeout=1)
    assert output == working_dir
    output = terminal.run_interactive(["echo", "$TEST_VAR"], timeout=1)
    assert output == "FooBar"

    terminal.close_pseudo_terminal()
    assert terminal.has_pseudo_terminal() is False

    # starts the pseudo terminal automatically
    output = terminal.run_interactive(command, timeout=1)
    assert terminal.has_pseudo_terminal() is True
    assert output == "Hello World"

    terminal.close_pseudo_terminal()
    assert terminal.has_pseudo_terminal() is False
