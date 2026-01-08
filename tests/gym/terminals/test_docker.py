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


@pytest.if_docker_running
def test_docker_terminal_run_timeout(tmp_path):
    """Test that commands that exceed the timeout are killed and return failure."""
    working_dir = str(tmp_path)
    terminal = DockerTerminal(working_dir=working_dir, base_image="ubuntu:latest")
    try:
        # Run a command that takes longer than the timeout
        entrypoint = "sleep 10 && echo done"
        success, output = terminal.run(entrypoint, timeout=2)
        assert success is False
        assert "timed out" in output.lower()
        assert "2 seconds" in output
    finally:
        terminal.clean_up()


@pytest.if_docker_running
def test_docker_terminal_run_default_timeout(tmp_path):
    """Test that the default timeout is applied when none is specified."""
    working_dir = str(tmp_path)
    terminal = DockerTerminal(working_dir=working_dir, base_image="ubuntu:latest")
    try:
        # Run a quick command without specifying timeout
        entrypoint = "echo 'Hello'"
        success, output = terminal.run(entrypoint)  # No timeout specified
        assert success is True
        assert output == "Hello"
        # Default command_timeout should be 300 seconds (5 minutes)
        assert terminal.command_timeout == 300
    finally:
        terminal.clean_up()


@pytest.if_docker_running
def test_docker_terminal_custom_command_timeout(tmp_path):
    """Test that custom command_timeout can be set via constructor."""
    working_dir = str(tmp_path)
    terminal = DockerTerminal(
        working_dir=working_dir, base_image="ubuntu:latest", command_timeout=60
    )
    try:
        assert terminal.command_timeout == 60
        # Quick command should still work
        success, output = terminal.run("echo 'test'")
        assert success is True
        assert output == "test"
    finally:
        terminal.clean_up()


@pytest.if_docker_running
def test_docker_terminal_nohup_with_nested_shell_returns_immediately(tmp_path):
    """Test that nohup commands with nested shell return immediately in non-TTY mode.

    This test verifies the fix for issue #325 where nohup commands would cause
    the timeout wrapper to wait in non-TTY mode. Using sh -c creates a nested
    shell that exits immediately after backgrounding the process.
    """
    working_dir = str(tmp_path)
    terminal = DockerTerminal(
        working_dir=working_dir, base_image="ubuntu:latest", command_timeout=10
    )
    try:
        # Test that sh -c with nohup returns immediately
        start_time = time.time()
        success, output = terminal.run("sh -c 'nohup sleep 100 > /dev/null 2>&1 &'")
        elapsed = time.time() - start_time

        # Should return almost immediately (within 2 seconds)
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
        terminal.clean_up()


@pytest.if_docker_running
def test_docker_terminal_nohup_without_redirection_may_timeout(tmp_path):
    """Test that nohup commands without redirection may not return immediately.

    This test demonstrates the problem that was fixed: without output redirection,
    the timeout command waits for file descriptors to close.
    """
    working_dir = str(tmp_path)
    terminal = DockerTerminal(
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
        terminal.clean_up()
