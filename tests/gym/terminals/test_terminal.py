import os
import re
import tempfile
from pathlib import Path

import pytest

from debug_gym.gym.terminals import select_terminal
from debug_gym.gym.terminals.shell_session import DEFAULT_PS1
from debug_gym.gym.terminals.terminal import Terminal


def test_terminal_init():
    terminal = Terminal()
    assert terminal.session_commands == []
    assert terminal.env_vars["NO_COLOR"] == "1"
    assert terminal.env_vars["PS1"] == DEFAULT_PS1
    assert len(terminal.env_vars) > 2  # NO_COLOR, PS1 + os env vars
    assert os.path.basename(terminal.working_dir).startswith("Terminal-")


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


def test_terminal_run_tmp_working_dir():
    terminal = Terminal()
    entrypoint = "pwd -P"
    success, output = terminal.run(entrypoint, timeout=1)
    assert success is True
    assert output == terminal.working_dir


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


def test_terminal_multiple_session_commands(tmp_path):
    working_dir = str(tmp_path)
    session_commands = ["echo 'Hello'", "echo 'World'"]
    terminal = Terminal(working_dir, session_commands)
    status, output = terminal.run("pwd", timeout=1)
    assert status
    assert output == f"Hello\nWorld\n{working_dir}"


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


def test_copy_content(tmp_path):
    # Create a temporary source file
    source_dir = tmp_path / "source_dir"
    source_dir.mkdir()
    source_file = source_dir / "tmp.txt"
    with open(source_file, "w") as src_file:
        src_file.write("Hello World")

    working_dir = tmp_path / "working_dir"
    working_dir.mkdir()

    terminal = Terminal(working_dir=working_dir)
    # Source must be a folder.
    with pytest.raises(ValueError, match="Source .* must be a directory."):
        terminal.copy_content(source_file)

    terminal.copy_content(source_dir)

    # Clean up the temporary source_dir
    source_file.unlink()
    source_dir.rmdir()

    # Verify the content was copied correctly
    with open(working_dir / "tmp.txt", "r") as f:
        content = f.read()
    assert content == "Hello World"
