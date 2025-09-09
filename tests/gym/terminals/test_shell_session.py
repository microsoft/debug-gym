import platform

import pytest

from debug_gym.gym.terminals.shell_session import DEFAULT_PS1, ShellSession

if_is_linux = pytest.mark.skipif(
    platform.system() != "Linux",
    reason="Interactive ShellSession (pty) requires Linux.",
)


@if_is_linux
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
