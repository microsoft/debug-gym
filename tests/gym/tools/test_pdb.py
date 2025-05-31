import copy
import re
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from debug_gym.gym.entities import Event
from debug_gym.gym.envs.env import RepoEnv
from debug_gym.gym.terminal import DockerTerminal, Terminal
from debug_gym.gym.tools.pdb import PDBTool

if_docker_running = pytest.mark.skipif(
    not subprocess.check_output(["docker", "ps"]),
    reason="Docker not running",
)


def clean_up_pytest_path(obs):
    """clean up the pytest path to not depend on the environment"""
    return re.sub(
        r"Current frame:\n.*pytest/__main__\.py",
        "Current frame:\n.../pytest/__main__.py",
        obs,
    )


@pytest.fixture
def setup_test_repo():
    def _setup_test_repo(base_dir):
        """Setup a repo with 2 dummy files, 1 fail test, and 1 pass test"""
        working_dir = base_dir / "tests_pdb"
        working_dir.mkdir()
        with (working_dir / "test_pass.py").open("w") as f:
            f.write("def test_pass():\n    assert True")
        with (working_dir / "test_fail.py").open("w") as f:
            f.write("def test_fail():\n    assert False")
        dummy_files = ["file1.py", "file2.py"]
        for dummy_file in dummy_files:
            with (working_dir / dummy_file).open("w") as f:
                [f.write(f"print({i})\n") for i in range(40)]
        return working_dir

    return _setup_test_repo


@pytest.fixture
def breakpoints_state():
    return {
        "file1.py|||10": "b file1.py:10",
        "file1.py|||20": "b file1.py:20",
        "file1.py|||30": "b file1.py:30",
        "file2.py|||15": "b file2.py:15",
    }


@pytest.fixture
def setup_pdb_repo_env(setup_test_repo, breakpoints_state):
    def _setup_pdb_repo_env(base_dir):
        test_repo = setup_test_repo(base_dir)
        env = RepoEnv(path=str(test_repo))
        pdb_tool = PDBTool()
        pdb_tool.register(env)
        env.reset()
        env.current_breakpoints_state = breakpoints_state
        env.persistent_breakpoints = True
        env.auto_list = True
        pdb_tool.start_pdb(env)
        return pdb_tool, env

    return _setup_pdb_repo_env


def test_pdb_use(tmp_path, setup_test_repo):
    # Test PDBTool with Terminal, verbose pytest
    tests_path = str(setup_test_repo(tmp_path))
    terminal = Terminal()
    environment = RepoEnv(
        path=tests_path,
        terminal=terminal,
        debug_entrypoint="python -m pdb -m pytest -sv .",
    )
    pdb = PDBTool()
    initial_output = pdb.start_pdb(environment)
    assert """The pytest entry point.""" in initial_output
    assert "(Pdb)" not in initial_output
    output = pdb.use(environment, command="l").observation
    assert """The pytest entry point.""" in output
    assert "(Pdb)" not in output
    output = pdb.use(environment, command="c").observation
    assert "1 failed, 1 passed" in output
    assert "test_fail.py::test_fail FAILED" in output
    assert "test_pass.py::test_pass PASSED" in output
    assert "Reached the end of the program. Restarting the debugging session." in output
    assert "pytest/__main__.py" in output
    assert '-> """The pytest entry point."""' in output
    assert 'Context around the current frame:\n1  ->	"""The pytest entry point.""""'
    assert "(Pdb)" not in output


def test_pdb_use_empty_command(tmp_path, setup_test_repo):
    # Test PDBTool with Terminal, verbose pytest
    tests_path = str(setup_test_repo(tmp_path))
    terminal = Terminal()
    environment = RepoEnv(
        path=tests_path,
        terminal=terminal,
        debug_entrypoint="python -m pdb -m pytest -sv .",
    )
    pdb = PDBTool()
    _ = pdb.start_pdb(environment)

    output = pdb.use(environment, command="").observation
    assert "Failure calling pdb:\nEmpty commands are not allowed." in output


def test_pdb_b_fail_blank_or_comment(tmp_path, setup_test_repo):
    # Test PDBTool with Terminal, verbose pytest
    tests_path = str(setup_test_repo(tmp_path))
    terminal = Terminal()
    environment = RepoEnv(
        path=tests_path,
        terminal=terminal,
        debug_entrypoint="python -m pdb -m pytest -sv .",
    )
    pdb = PDBTool()
    _ = pdb.start_pdb(environment)

    output = pdb.use(environment, command="b 1").observation
    output = clean_up_pytest_path(output)
    assert (
        output == "Invalid pdb command: b 1\nInvalid line number: *** Blank or comment."
    )
    assert environment.current_breakpoints_state == {}


def test_pdb_pass_empty_path_if_in_session(tmp_path, setup_test_repo):
    # Test PDBTool with Terminal, verbose pytest
    tests_path = str(setup_test_repo(tmp_path))
    terminal = Terminal()
    environment = RepoEnv(
        path=tests_path,
        terminal=terminal,
        debug_entrypoint="python -m pdb -m pytest -sv .",
    )
    pdb = PDBTool()
    _ = pdb.start_pdb(environment)

    obs = pdb.use(environment, command="b test_pass.py:1").observation
    assert obs.startswith("Pdb command output:\nBreakpoint 1 at test_pass.py:1")
    obs = pdb.use(environment, command="c").observation
    assert "1 B->\tdef test_pass():" in obs
    # Now try to set a breakpoint without specifying the file, it should pass
    obs = pdb.use(environment, command="b 2").observation
    assert obs.startswith("Pdb command output:\nBreakpoint 2 at test_pass.py:2")


def test_pdb_use_default_environment_entrypoint(tmp_path, setup_test_repo):
    # Test PDBTool with default environment entrypoint, quiet pytest
    tests_path = str(setup_test_repo(tmp_path))
    terminal = Terminal()
    environment = RepoEnv(path=tests_path, terminal=terminal)
    pdb = PDBTool()
    initial_output = pdb.start_pdb(environment)  # "python -m pdb -m pytest -sq ."
    assert """The pytest entry point.""" in initial_output
    assert "(Pdb)" not in initial_output

    output = pdb.use(environment, command="l").observation
    assert """The pytest entry point.""" in output
    assert "(Pdb)" not in output

    output = pdb.use(environment, command="c").observation
    assert "1 failed, 1 passed" in output
    assert "test_fail.py::test_fail" in output
    assert "test_pass.py::test_pass" not in output
    assert "Reached the end of the program. Restarting the debugging session." in output
    assert "pytest/__main__.py" in output
    assert '-> """The pytest entry point."""' in output
    assert 'Context around the current frame:\n1  ->	"""The pytest entry point.""""'
    assert "(Pdb)" not in output


@if_docker_running
def test_pdb_use_docker_terminal(tmp_path, setup_test_repo):
    """Test PDBTool similar to test_pdb_use but using DockerTerminal"""
    tests_path = str(setup_test_repo(tmp_path))
    terminal = DockerTerminal(
        base_image="python:3.12-slim",
        session_commands=["pip install pytest"],
        env_vars={"PYTHONDONTWRITEBYTECODE": "1"},  # avoid __pycache__
        map_host_uid_gid=False,  # run as root
    )
    # no:cacheprovider to avoid .pytest_cache
    debug_entrypoint = "python -m pdb -m pytest -p no:cacheprovider -sv ."
    environment = RepoEnv(
        path=tests_path, terminal=terminal, debug_entrypoint=debug_entrypoint
    )
    pdb = PDBTool()
    pdb.start_pdb(environment)

    output = pdb.use(environment, command="l").observation
    assert """The pytest entry point.""" in output
    assert "(Pdb)" not in output

    output = pdb.use(environment, command="c").observation
    assert "1 failed, 1 passed" in output
    assert "test_fail.py::test_fail FAILED" in output
    assert "test_pass.py::test_pass PASSED" in output
    assert "Reached the end of the program. Restarting the debugging session." in output
    assert "pytest/__main__.py" in output
    assert '-> """The pytest entry point."""' in output
    assert 'Context around the current frame:\n1  ->	"""The pytest entry point.""""'
    assert "(Pdb)" not in output


def test_initialization():
    pdb_tool = PDBTool()
    assert pdb_tool.current_frame_file is None
    assert pdb_tool._session is None


def test_register():
    env = RepoEnv()
    pdb_tool = PDBTool()
    pdb_tool.register(env)
    # every tool listen to ENV_RESET event to track history
    assert pdb_tool in env.event_hooks.event_listeners[Event.ENV_RESET]
    assert pdb_tool in env.event_hooks.event_listeners[Event.REWRITE_SUCCESS]


def test_register_invalid_environment():
    pdb_tool = PDBTool()
    with pytest.raises(ValueError, match="The environment must be a RepoEnv instance."):
        pdb_tool.register(MagicMock())


def test_pdb_add_new_breakpoint(tmp_path, setup_pdb_repo_env, breakpoints_state):
    pdb_tool, env = setup_pdb_repo_env(tmp_path)
    pdb_obs = pdb_tool.use(env, "b file1.py:25")

    # Accept both relative and absolute paths in the output
    assert "Pdb command output:\nBreakpoint 5 at file1.py:25" in pdb_obs.observation
    expected_state = {"file1.py|||25": "b file1.py:25"} | breakpoints_state
    assert env.current_breakpoints_state == expected_state


def test_pdb_add_existing_breakpoint(tmp_path, setup_pdb_repo_env, breakpoints_state):
    pdb_tool, env = setup_pdb_repo_env(tmp_path)
    pdb_obs = pdb_tool.use(env, "b file1.py:10")
    assert "Pdb command output:\nBreakpoint 5 at file1.py:10" in pdb_obs.observation
    assert env.current_breakpoints_state == breakpoints_state


def test_pdb_clear_specific(tmp_path, setup_pdb_repo_env):
    pdb_tool, env = setup_pdb_repo_env(tmp_path)
    pdb_obs = pdb_tool.use(env, "cl file1.py:20")
    expected_state = {
        "file1.py|||10": "b file1.py:10",
        "file1.py|||30": "b file1.py:30",
        "file2.py|||15": "b file2.py:15",
    }
    assert pdb_obs.observation.startswith(
        "Pdb command output:\nDeleted breakpoint 2 at file1.py:20\n\nCurrent frame:"
    )
    assert env.current_breakpoints_state == expected_state


def test_pdb_clear_not_found(tmp_path, setup_pdb_repo_env, breakpoints_state):
    pdb_tool, env = setup_pdb_repo_env(tmp_path)
    pdb_obs = pdb_tool.use(env, "cl file1.py:8")
    assert pdb_obs.observation.startswith(
        "Pdb command output:\n*** There is no breakpoint at file1.py:8"
    )
    assert env.current_breakpoints_state == breakpoints_state


def test_breakpoint_modify_remove(tmp_path, setup_pdb_repo_env):
    # Remove breakpoint at line 20 and move breakpoint at line 30 to line 24
    # TODO: 24 or 25?
    pdb_tool, env = setup_pdb_repo_env(tmp_path)
    pdb_tool.breakpoint_modify(env, "file1.py", 15, 25, 5)
    expected_state = {
        "file1.py|||10": "b file1.py:10",
        "file1.py|||24": "b file1.py:24",
        "file2.py|||15": "b file2.py:15",
    }
    assert env.current_breakpoints_state == expected_state


def test_breakpoint_modify_move(tmp_path, setup_pdb_repo_env):
    pdb_tool, env = setup_pdb_repo_env(tmp_path)
    pdb_tool.breakpoint_modify(env, "file1.py", 5, 15, 10)
    expected_state = {
        "file2.py|||15": "b file2.py:15",
        "file1.py|||19": "b file1.py:19",
        "file1.py|||29": "b file1.py:29",
    }
    assert env.current_breakpoints_state == expected_state


def test_breakpoint_modify_remove_all(tmp_path, setup_pdb_repo_env):
    pdb_tool, env = setup_pdb_repo_env(tmp_path)
    pdb_tool.breakpoint_modify(env, "file1.py", None, None, 0)
    expected_state = {"file2.py|||15": "b file2.py:15"}
    assert env.current_breakpoints_state == expected_state


def test_breakpoint_modify_no_change(tmp_path, setup_pdb_repo_env):
    pdb_tool, env = setup_pdb_repo_env(tmp_path)
    pdb_tool.breakpoint_modify(env, "file1.py", 25, 35, 5)
    # Test no change for breakpoints before the rewritten code (change line 30)
    expected_state = {
        "file1.py|||10": "b file1.py:10",
        "file1.py|||20": "b file1.py:20",
        "file2.py|||15": "b file2.py:15",
    }
    assert env.current_breakpoints_state == expected_state


def test_pdb_crashing(tmp_path, setup_test_repo):
    tests_path = setup_test_repo(tmp_path)
    with open(tests_path / "test_fail.py", "w") as f:
        f.write("def test_fail():\nassert False")  # IndentationError

    environment = RepoEnv(
        path=tests_path,
        entrypoint="python -m pytest -s test.py",
        debug_entrypoint="python -m pdb -m pytest -s test_fail.py",
    )
    pdb = PDBTool()

    initial_output = pdb.start_pdb(environment)
    assert "The pytest entry point." in initial_output
    output = pdb.interact_with_pdb("c", 5)
    assert "IndentationError" in output


def test_pdb_timeout(tmp_path, setup_test_repo):
    tests_path = setup_test_repo(tmp_path)
    with open(tests_path / "test_fail.py", "w") as f:
        f.write(
            "def test_fail():\n  print('Sleeping...'); import time; time.sleep(10)"
        )  # IndentationError

    environment = RepoEnv(
        path=tests_path,
        entrypoint="python -m pytest -s test.py",
        debug_entrypoint="python -m pdb -m pytest -sv test_fail.py",
    )
    pdb = PDBTool()

    initial_output = pdb.start_pdb(environment)
    assert "The pytest entry point." in initial_output
    output = pdb.interact_with_pdb("c", timeout=1)
    assert "timed out" in output
    assert not pdb.pdb_is_running


def test_close_pdb_start_and_close_session(tmp_path, setup_pdb_repo_env):
    pdb_tool, env = setup_pdb_repo_env(tmp_path)
    # setup_pdb_repo_env starts the pdb session
    assert pdb_tool.pdb_is_running
    pdb_tool.close_pdb()
    assert not pdb_tool.pdb_is_running
    pdb_tool.start_pdb(env)
    assert pdb_tool.pdb_is_running


def test_deepcopy_sets_session_none(tmp_path, setup_pdb_repo_env):
    pdb_tool, _ = setup_pdb_repo_env(tmp_path)
    assert pdb_tool.current_frame_file.endswith("pytest/__main__.py")
    tool_copy = copy.deepcopy(pdb_tool)
    assert tool_copy._session is None
    assert tool_copy.current_frame_file is None
    assert pdb_tool.current_frame_file.endswith("pytest/__main__.py")


def test_start_pdb_restores_breakpoints(tmp_path, setup_pdb_repo_env):
    pdb_tool, env = setup_pdb_repo_env(tmp_path)
    env.current_breakpoints_state = {"file1.py|||1": "b file1.py:1"}
    out = pdb_tool.start_pdb(env)
    assert "Breakpoints have been restored." in out


def test_on_env_reset_calls_start_pdb(tmp_path, setup_pdb_repo_env):
    pdb_tool, env = setup_pdb_repo_env(tmp_path)
    called = []

    def fake_start(e):
        called.append(True)
        return "reset"

    pdb_tool.start_pdb = fake_start
    obs = pdb_tool.on_env_reset(env)
    assert obs.observation == "reset"
    assert called


def test_on_rewrite_success_calls_breakpoint_modify_and_restart_pdb(
    tmp_path, setup_pdb_repo_env
):
    pdb_tool, env = setup_pdb_repo_env(tmp_path)
    called = []
    pdb_tool.breakpoint_modify = lambda *a, **k: called.append("modify")
    pdb_tool.restart_pdb = lambda e: "restarted"
    obs = pdb_tool.on_rewrite_success(env, "file1.py", 1, 2, 3)
    assert "restarted" in obs.observation
    assert "modify" in called


def test_restart_pdb_calls_close_and_start(tmp_path, setup_pdb_repo_env):
    pdb_tool, env = setup_pdb_repo_env(tmp_path)
    pdb_tool.close_pdb = lambda: setattr(pdb_tool, "closed", True)
    pdb_tool.start_pdb = lambda e: "started"
    out = pdb_tool.restart_pdb(env)
    assert pdb_tool.closed
    assert out == "started"


def test_use_multiple_commands_warning(tmp_path, setup_pdb_repo_env):
    pdb_tool, env = setup_pdb_repo_env(tmp_path)
    env.all_files = ["file1.py"]
    env.current_breakpoints_state = {}
    pdb_tool.current_frame_file = "file1.py"
    obs = pdb_tool.use(env, "b 1; b 2").observation
    assert "Multiple commands are not supported" in obs


def test_use_empty_command_returns_failure(tmp_path, setup_pdb_repo_env):
    pdb_tool, env = setup_pdb_repo_env(tmp_path)
    obs = pdb_tool.use(env, "").observation
    assert "Empty commands are not allowed" in obs


def test_use_breakpoints_and_clear(tmp_path, setup_pdb_repo_env):
    pdb_tool, env = setup_pdb_repo_env(tmp_path)
    env.current_breakpoints_state = {"file1.py|||1": "b file1.py:1"}
    env.all_files = ["file1.py"]
    pdb_tool.current_frame_file = "file1.py"
    obs = pdb_tool.use(env, "b").observation
    # clean up the pytest path to not depend on the environment
    obs = clean_up_pytest_path(obs)
    assert obs == (
        "Breakpoints:\n"
        "line 1 in file1.py\n"
        "\n"
        "Current frame:\n"
        ".../pytest/__main__.py\n"
        "\n"
        "Context around the current frame:\n"
        '1  ->\t"""The pytest entry point."""\r\n'
        "  2  \t\r\n"
        "  3  \tfrom __future__ import annotations\r\n"
        "  4  \t\r\n"
        "  5  \timport pytest\r\n"
        "  6  \t\r\n"
        "  7  \t\r\n"
        '  8  \tif __name__ == "__main__":\r\n'
        "  9  \t    raise SystemExit(pytest.console_main())\r\n"
        "[EOF]\n"
    )

    obs2 = pdb_tool.use(env, "cl").observation
    # clean up the pytest path to not depend on the environment
    obs2 = clean_up_pytest_path(obs2)
    assert obs2 == (
        "All breakpoints have been cleared.\n"
        "\n"
        "Current frame:\n"
        ".../pytest/__main__.py\n"
        "\n"
        "Context around the current frame:\n"
        '1  ->\t"""The pytest entry point."""\r\n'
        "  2  \t\r\n"
        "  3  \tfrom __future__ import annotations\r\n"
        "  4  \t\r\n"
        "  5  \timport pytest\r\n"
        "  6  \t\r\n"
        "  7  \t\r\n"
        '  8  \tif __name__ == "__main__":\r\n'
        "  9  \t    raise SystemExit(pytest.console_main())\r\n"
        "[EOF]\n"
    )


def test_use_b_invalid_file(tmp_path, setup_pdb_repo_env):
    pdb_tool, env = setup_pdb_repo_env(tmp_path)
    pdb_obs = pdb_tool.use(env, "b notafile.py:1")
    pdb_obs.source = "pdb"
    obs = clean_up_pytest_path(pdb_obs.observation)
    assert obs == (
        "Pdb command output:\n"
        "*** 'notafile.py' not found from sys.path\n"
        "\n"
        "Current frame:\n"
        ".../pytest/__main__.py\n"
        "\n"
        "Context around the current frame:\n"
        '1  ->\t"""The pytest entry point."""\r\n'
        "  2  \t\r\n"
        "  3  \tfrom __future__ import annotations\r\n"
        "  4  \t\r\n"
        "  5  \timport pytest\r\n"
        "  6  \t\r\n"
        "  7  \t\r\n"
        '  8  \tif __name__ == "__main__":\r\n'
        "  9  \t    raise SystemExit(pytest.console_main())\r\n"
        "[EOF]\n"
    )


def test_use_pdb_invalid_line(tmp_path, setup_pdb_repo_env):
    pdb_tool, env = setup_pdb_repo_env(tmp_path)
    pdb_obs = pdb_tool.use(env, "b file1.py:100")
    assert pdb_obs.source == "pdb"
    assert (
        pdb_obs.observation
        == "Invalid pdb command: b file1.py:100\nInvalid line number: End of file."
    )
    pdb_obs = pdb_tool.use(env, "b file1.py:-100")
    assert pdb_obs.source == "pdb"
    assert (
        pdb_obs.observation
        == "Invalid pdb command: b file1.py:-100\nInvalid line number: End of file."
    )


def test_use_pdb_var_not_defined(tmp_path, setup_pdb_repo_env):
    pdb_tool, env = setup_pdb_repo_env(tmp_path)
    obs = pdb_tool.use(env, "invalid").observation
    assert "*** NameError: name 'invalid' is not defined" in obs


def test_use_pdb_syntax_error(tmp_path, setup_pdb_repo_env):
    pdb_tool, env = setup_pdb_repo_env(tmp_path)
    pdb_obs = pdb_tool.use(env, "foo file1.py:1")
    assert pdb_obs.observation.startswith(
        "Pdb command output:\n*** SyntaxError: invalid syntax"
    )


def test_set_current_frame_file_sets_file(tmp_path, setup_pdb_repo_env):
    pdb_tool, env = setup_pdb_repo_env(tmp_path)
    # First stop at pytest main file
    assert "pytest/__main__.py" in pdb_tool.current_frame_file
    # pdb_tool.use calls pdb_tool.set_current_frame_file(env)
    obs = pdb_tool.use(env, "b test_fail.py:2")
    assert "Pdb command output:\nBreakpoint 5 at test_fail.py:2" in obs.observation
    # no `continue` command, so current_frame_file should still be pytest main file
    assert "pytest/__main__.py" in pdb_tool.current_frame_file
    obs = pdb_tool.use(env, "c")
    # At this point, current_frame_file should be set to the file where the breakpoint was set
    assert pdb_tool.current_frame_file == "test_fail.py"
    # observation should contain the test file
    assert "Current frame:\ntest_fail.py" in obs.observation
    assert "> test_fail.py(2)" in obs.observation


def test_set_current_frame_file_sets_and_returns(tmp_path, setup_pdb_repo_env):
    pdb_tool, env = setup_pdb_repo_env(tmp_path)
    # Patch interact_with_pdb to simulate output
    test_file = "file1.py"
    test_path = str(env.working_dir / test_file)

    def fake_interact_with_pdb(command, timeout):
        return f"somecontext\n> {test_path}(10)<module>()\n-> some code context"

    pdb_tool.interact_with_pdb = fake_interact_with_pdb
    result = pdb_tool.set_current_frame_file(env)
    assert result == test_file
    assert pdb_tool.current_frame_file == test_file


def test_use_multiple_commands_only_first_executed(tmp_path, setup_pdb_repo_env):
    pdb_tool, env = setup_pdb_repo_env(tmp_path)
    env.current_breakpoints_state = {}
    pdb_tool.restart_pdb(env)
    pdb_obs = pdb_tool.use(env, "b")
    obs = clean_up_pytest_path(pdb_obs.observation)
    assert pdb_obs.source == "pdb"
    assert obs == (
        "Breakpoints:\n"
        "No breakpoints are set.\n"
        "\n"
        "Current frame:\n"
        ".../pytest/__main__.py\n"
        "\n"
        "Context around the current frame:\n"
        '1  ->\t"""The pytest entry point."""\r\n'
        "  2  \t\r\n"
        "  3  \tfrom __future__ import annotations\r\n"
        "  4  \t\r\n"
        "  5  \timport pytest\r\n"
        "  6  \t\r\n"
        "  7  \t\r\n"
        '  8  \tif __name__ == "__main__":\r\n'
        "  9  \t    raise SystemExit(pytest.console_main())\r\n"
        "[EOF]\n"
    )
    pdb_obs = pdb_tool.use(env, "b file1.py:1; b file1.py:2; b file1.py:3")
    assert pdb_obs.source == "pdb"
    assert pdb_obs.observation.startswith(
        "Multiple commands are not supported. Only the first command will be executed.\n"
        "Pdb command output:\nBreakpoint 1 at file1.py:1\n"
        "\n"
        "Current frame:"
    )


def test_use_print_command_allows_semicolon(tmp_path, setup_pdb_repo_env):
    pdb_tool, env = setup_pdb_repo_env(tmp_path)
    env.all_files = ["file1.py"]
    pdb_tool.current_frame_file = "file1.py"
    # Patch interact_with_pdb to check command is not split
    called = []

    def fake_interact_with_pdb(command, timeout):
        called.append(command)
        return "42"

    pdb_tool.interact_with_pdb = fake_interact_with_pdb
    obs = pdb_tool.use(env, "p x; p y").observation
    assert "Multiple commands are not supported" not in obs
    # print + update breakpoints list, free where, and list commands
    assert called == ["p x; p y", "b", "where", "l ."]


def test_use_empty_command_returns_failure_message(tmp_path, setup_pdb_repo_env):
    pdb_tool, env = setup_pdb_repo_env(tmp_path)
    obs = pdb_tool.use(env, "").observation
    assert "Empty commands are not allowed" in obs


def test_use_starts_pdb_if_not_running(tmp_path, setup_pdb_repo_env):
    pdb_tool, env = setup_pdb_repo_env(tmp_path)
    pdb_tool._session = None  # simulate not running
    # Patch start_pdb to simulate output
    pdb_tool.start_pdb = lambda e: "Started PDB"
    pdb_tool.set_current_frame_file = lambda e: None
    obs = pdb_tool.use(env, "b 1").observation
    assert "Started PDB" in obs


def test_use_lists_breakpoints(tmp_path, setup_pdb_repo_env):
    pdb_tool, env = setup_pdb_repo_env(tmp_path)
    env.current_breakpoints_state = {"file1.py|||1": "b file1.py:1"}
    env.current_breakpoints = lambda: "line 1 in file1.py"
    obs = pdb_tool.use(env, "b").observation
    assert "Breakpoints:" in obs
    assert "line 1 in file1.py" in obs


def test_use_clears_all_breakpoints(tmp_path, setup_pdb_repo_env):
    pdb_tool, env = setup_pdb_repo_env(tmp_path)
    env.current_breakpoints_state = {"file1.py|||1": "b file1.py:1"}
    env.all_files = ["file1.py"]
    # Patch restart_pdb to avoid real restart
    pdb_tool.restart_pdb = lambda e: None
    obs = pdb_tool.use(env, "cl").observation
    assert "All breakpoints have been cleared." in obs
    assert env.current_breakpoints_state == {}


def test_use_invalid_command_returns_invalid_message(tmp_path, setup_pdb_repo_env):
    pdb_tool, env = setup_pdb_repo_env(tmp_path)

    # Patch interact_with_pdb to raise exception
    def raise_exc(c, t):
        raise Exception("fail")

    pdb_tool.interact_with_pdb = raise_exc
    obs = pdb_tool.use(env, "invalid").observation
    assert "Invalid pdb command: invalid" in obs
