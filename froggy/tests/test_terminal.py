import subprocess

import pytest

from froggy.terminal import DockerTerminal, Terminal

if_docker_running = pytest.mark.skipif(
    not subprocess.check_output(["docker", "ps"]),
    reason="Docker not running",
)


def test_terminal_init():
    terminal = Terminal()
    assert terminal.setup_commands == []
    assert terminal.env_vars["NO_COLOR"] == "1"
    assert terminal.env_vars["PS1"] == ""
    assert len(terminal.env_vars) > 2  # NO_COLOR, PS1 + os env vars
    assert terminal.working_dir.startswith("/tmp/Terminal-")


def test_terminal_init_no_os_env_vars():
    terminal = Terminal(include_os_env_vars=False)
    assert terminal.env_vars == {"NO_COLOR": "1", "PS1": ""}


def test_terminal_init_with_params(tmp_path):
    working_dir = str(tmp_path)
    setup_commands = ["echo 'Hello World'"]
    env_vars = {"ENV_VAR": "value"}
    terminal = Terminal(working_dir, setup_commands, env_vars)
    assert terminal.working_dir == working_dir
    assert terminal.setup_commands == setup_commands
    assert terminal.env_vars["NO_COLOR"] == "1"
    assert terminal.env_vars["ENV_VAR"] == "value"
    status, output = terminal.run("pwd")
    assert status
    assert output == f"Hello World\n{working_dir}"
    status, output = terminal.run("echo $ENV_VAR")
    assert status
    assert output == "Hello World\nvalue"


def test_terminal_run(tmp_path):
    working_dir = str(tmp_path)
    terminal = Terminal(working_dir=working_dir)
    entrypoint = "echo 'Hello World'"
    success, output = terminal.run(entrypoint)
    assert success is True
    assert output == "Hello World"
    assert terminal.working_dir == working_dir


def test_terminal_run_tmp_working_dir(tmp_path):
    terminal = Terminal()
    entrypoint = "echo 'Hello World'"
    success, output = terminal.run(entrypoint)
    assert success is True
    assert output == "Hello World"
    assert terminal.working_dir.startswith("/tmp/Terminal-")


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
    success, output = terminal.run(command)
    assert success is True
    assert output == "Hello\nWorld"


def test_terminal_run_failure(tmp_path):
    working_dir = str(tmp_path)
    terminal = Terminal(working_dir=working_dir)
    entrypoint = "ls non_existent_dir"
    success, output = terminal.run(entrypoint)
    assert success is False
    assert output == ("ls: cannot access 'non_existent_dir': No such file or directory")


def test_terminal_pseudo_terminal(tmp_path):
    working_dir = str(tmp_path)
    command = "echo Hello World"
    terminal = Terminal(working_dir=working_dir)
    assert terminal.has_pseudo_terminal() is False

    terminal.start_pseudo_terminal(timeout=1)
    assert terminal.has_pseudo_terminal() is True
    output = terminal.run_interactive(command, timeout=1)
    assert output == "Hello World"

    terminal.run_interactive("export TEST_VAR='FooBar'", timeout=1)
    output = terminal.run_interactive("pwd", timeout=1)
    assert output == working_dir
    output = terminal.run_interactive("echo $TEST_VAR", timeout=1)
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
    assert terminal.working_dir.startswith("/tmp/Terminal-")
    assert terminal.base_image == "ubuntu:latest"
    assert terminal.volumes[terminal.working_dir] == {
        "bind": terminal.working_dir,
        "mode": "rw",
    }
    assert terminal.container is not None
    assert terminal.container.status == "running"


@if_docker_running
def test_docker_terminal_init_with_params(tmp_path):
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
    assert terminal.container.status == "running"

    _, output = terminal.run("pwd")
    assert output == working_dir

    _, output = terminal.run("ls -l")
    assert "new_dir" in output


@if_docker_running
@pytest.mark.parametrize(
    "command",
    [
        "export ENV_VAR=value && mkdir test && ls",
        ["export ENV_VAR=value", "mkdir test", "ls"],
    ],
)
def test_docker_terminal_run(tmp_path, command):
    working_dir = str(tmp_path)
    volumes = {working_dir: {"bind": working_dir, "mode": "rw"}}
    docker_terminal = DockerTerminal(working_dir=working_dir, volumes=volumes)
    success, output = docker_terminal.run(command)
    assert output == "test"
    assert success is True

    success, output = docker_terminal.run("echo $ENV_VAR")
    assert "value" not in output
    assert success is True
    success, output = docker_terminal.run("ls")
    assert "test" in output
    assert success is True


@if_docker_running
def test_docker_terminal_read_only_volume(tmp_path):
    working_dir = str(tmp_path)
    read_only_dir = tmp_path / "read_only"
    read_only_dir.mkdir()
    with open(read_only_dir / "test.txt", "w") as f:
        f.write("test")
    read_only_dir = str(read_only_dir)
    volumes = {read_only_dir: {"bind": read_only_dir, "mode": "ro"}}
    docker_terminal = DockerTerminal(working_dir=working_dir, volumes=volumes)
    volumes = {
        working_dir: {"bind": working_dir, "mode": "rw"},
        read_only_dir: {"bind": read_only_dir, "mode": "ro"},
    }
    success, ls_output = docker_terminal.run(f"ls {read_only_dir}")
    assert success is True
    assert ls_output.startswith("test.txt")

    success, output = docker_terminal.run(f"touch {read_only_dir}/test2.txt")
    assert success is False
    assert (
        output
        == f"touch: cannot touch '{read_only_dir}/test2.txt': Read-only file system"
    )

    success, output = docker_terminal.run(f"touch {working_dir}/test2.txt")
    assert success is True
    assert output == ""


@if_docker_running
def test_docker_terminal_pseudo_terminal(tmp_path):
    # same as test_terminal_pseudo_terminal but with DockerTerminal
    working_dir = str(tmp_path)
    volumes = {working_dir: {"bind": working_dir, "mode": "rw"}}
    command = "echo Hello World"
    terminal = DockerTerminal(working_dir=working_dir, volumes=volumes)
    assert terminal.has_pseudo_terminal() is False

    terminal.start_pseudo_terminal(timeout=1)
    assert terminal.has_pseudo_terminal() is True
    output = terminal.run_interactive(command, timeout=1)
    assert output == "Hello World"

    terminal.run_interactive("export TEST_VAR='FooBar'", timeout=1)
    output = terminal.run_interactive("pwd", timeout=1)
    assert output == working_dir
    output = terminal.run_interactive("echo $TEST_VAR", timeout=1)
    assert output == "FooBar"

    terminal.close_pseudo_terminal()
    assert terminal.has_pseudo_terminal() is False

    # starts the pseudo terminal automatically
    output = terminal.run_interactive(command, timeout=1)
    assert terminal.has_pseudo_terminal() is True
    assert output == "Hello World"

    terminal.close_pseudo_terminal()
    assert terminal.has_pseudo_terminal() is False


@if_docker_running
def test_docker_terminal_update_volumes_with_working_dir(tmp_path):
    working_dir_a = str(tmp_path / "dir_a")
    terminal = DockerTerminal(working_dir=working_dir_a)
    assert terminal.working_dir == working_dir_a
    assert terminal.volumes[working_dir_a] == {"bind": working_dir_a, "mode": "rw"}

    working_dir_b = str(tmp_path / "dir_b")
    terminal.working_dir = working_dir_b
    assert terminal.volumes[working_dir_b] == {"bind": working_dir_b, "mode": "rw"}


@pytest.mark.parametrize(
    "terminal_cls",
    [
        Terminal,
        pytest.param(DockerTerminal, marks=if_docker_running),
    ],
)
def test_terminal_multiple_setup_commands(tmp_path, terminal_cls):
    working_dir = str(tmp_path)
    setup_commands = ["echo 'Hello'", "echo 'World'"]
    terminal = terminal_cls(working_dir, setup_commands)
    status, output = terminal.run("pwd")
    assert status
    assert output == f"Hello\nWorld\n{working_dir}"
