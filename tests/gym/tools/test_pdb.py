import copy
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
    assert "1 failed, 1 passed" in pdb.pdb_obs
    assert "test_fail.py::test_fail FAILED" in pdb.pdb_obs
    assert "test_pass.py::test_pass PASSED" in pdb.pdb_obs
    assert "Reached the end of the file. Restarting the debugging session." in output
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


def test_pdb_fail_if_empty_path_at_start(tmp_path, setup_test_repo):
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
    assert output.endswith("pytest/__main__.py` is not found in the repository.")


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
    assert obs.startswith(f"Breakpoint 1 at {environment.working_dir/'test_pass.py'}:1")
    obs = pdb.use(environment, command="c").observation
    assert "1 B->\tdef test_pass():" in obs
    # Now try to set a breakpoint without specifying the file, it should pass
    obs = pdb.use(environment, command="b 2").observation
    assert obs.startswith(f"Breakpoint 2 at {environment.working_dir/'test_pass.py'}:2")


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
    assert "1 failed, 1 passed" in pdb.pdb_obs
    assert "test_fail.py::test_fail" in pdb.pdb_obs
    assert "test_pass.py::test_pass" not in pdb.pdb_obs
    assert "Reached the end of the file. Restarting the debugging session." in output
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
    assert "1 failed, 1 passed" in pdb.pdb_obs
    assert "test_fail.py::test_fail FAILED" in pdb.pdb_obs
    assert "test_pass.py::test_pass PASSED" in pdb.pdb_obs
    assert "Reached the end of the file. Restarting the debugging session." in output
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


@patch.object(PDBTool, "interact_with_pdb")
def test_breakpoint_add_clear_add_new_breakpoint(
    mock_interact_with_pdb, tmp_path, setup_pdb_repo_env, breakpoints_state
):
    pdb_message = "Breakpoint 5 at file1.py:25"
    mock_interact_with_pdb.return_value = pdb_message
    pdb_tool, env = setup_pdb_repo_env(tmp_path)
    success, output = pdb_tool.breakpoint_add_clear(env, "b 25", "file1.py")
    assert success
    assert output == pdb_message
    expected_state = {"file1.py|||25": "b file1.py:25"} | breakpoints_state
    assert env.current_breakpoints_state == expected_state


def test_breakpoint_add_clear_add_existing_breakpoint(
    tmp_path, setup_pdb_repo_env, breakpoints_state
):
    pdb_tool, env = setup_pdb_repo_env(tmp_path)
    success, output = pdb_tool.breakpoint_add_clear(env, "b 10", "file1.py")
    assert success
    assert output == "Breakpoint already exists at line 10 in `file1.py`."
    assert env.current_breakpoints_state == breakpoints_state


@patch.object(PDBTool, "interact_with_pdb")
def test_breakpoint_add_clear_clear_specific(
    mock_interact_with_pdb, tmp_path, setup_pdb_repo_env
):
    pdb_message = "Deleted breakpoint 2 at file1.py:20"
    mock_interact_with_pdb.return_value = pdb_message
    pdb_tool, env = setup_pdb_repo_env(tmp_path)
    success, output = pdb_tool.breakpoint_add_clear(env, "cl 20", "file1.py")
    expected_state = {
        "file1.py|||10": "b file1.py:10",
        "file1.py|||30": "b file1.py:30",
        "file2.py|||15": "b file2.py:15",
    }
    assert success
    assert output == pdb_message
    assert env.current_breakpoints_state == expected_state


def test_breakpoint_add_clear_clear_not_found(
    tmp_path, setup_pdb_repo_env, breakpoints_state
):
    pdb_tool, env = setup_pdb_repo_env(tmp_path)
    success, output = pdb_tool.breakpoint_add_clear(env, "cl 8", "file1.py")
    assert success
    assert output == "No breakpoint exists at line 8 in `file1.py`."
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
    assert pdb.pdb_is_running is False


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
    assert obs == (
        "Breakpoints:\n"
        "line 1 in file1.py\n"
        "\n"
        "Current frame:\n"
        "/home/matpereira/miniconda3/envs/Froggy-terminal/lib/python3.12/site-packages/pytest/__main__.py\n"
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
    assert obs2 == (
        "All breakpoints have been cleared.\n"
        "\n"
        "Current frame:\n"
        "/home/matpereira/miniconda3/envs/Froggy-terminal/lib/python3.12/site-packages/pytest/__main__.py\n"
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


def test_use_breakpoint_add_clear_invalid_file(tmp_path, setup_pdb_repo_env):
    pdb_tool, env = setup_pdb_repo_env(tmp_path)
    env.all_files = ["file1.py"]
    pdb_tool.current_frame_file = "notafile.py"
    obs = pdb_tool.use(env, "b 1").observation
    assert "not found in the repository" in obs


def test_use_breakpoint_add_clear_invalid_line(tmp_path, setup_pdb_repo_env):
    pdb_tool, env = setup_pdb_repo_env(tmp_path)
    obs = pdb_tool.use(env, "b file1.py:99").observation
    assert "Invalid line number" in obs


def test_use_other_command_exception(tmp_path, setup_pdb_repo_env):
    pdb_tool, env = setup_pdb_repo_env(tmp_path)
    obs = pdb_tool.use(env, "invalid").observation
    assert "*** NameError: name 'invalid' is not defined" in obs


def test_breakpoint_add_clear_invalid_action(tmp_path, setup_pdb_repo_env):
    pdb_tool, env = setup_pdb_repo_env(tmp_path)
    success, output = pdb_tool.breakpoint_add_clear(env, "foo 1", "file1.py")
    assert not success
    assert output == "Invalid action: `foo 1`. Expected 'b', 'break', 'cl', or 'clear'."


def test_set_current_frame_file_sets_file(tmp_path, setup_pdb_repo_env):
    pdb_tool, env = setup_pdb_repo_env(tmp_path)
    # First stop at pytest main file
    assert "pytest/__main__.py" in pdb_tool.current_frame_file

    test_dir = env.working_dir / "test_fail.py"
    # pdb_tool.use calls pdb_tool.set_current_frame_file(env)
    obs = pdb_tool.use(env, "b test_fail.py:2")
    assert obs.observation.startswith(f"Breakpoint 1 at {test_dir}:2")
    # no `continue` command, so current_frame_file should still be pytest main file
    assert "pytest/__main__.py" in pdb_tool.current_frame_file
    obs = pdb_tool.use(env, "c")
    # At this point, current_frame_file should be set to the file where the breakpoint was set
    assert pdb_tool.current_frame_file == "test_fail.py"
    # observation should contain the test file
    assert "Current frame:\ntest_fail.py" in obs.observation
    assert f"> {test_dir}(2)" in obs.observation


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


def test_breakpoint_add_clear_missing_file(tmp_path, setup_pdb_repo_env):
    pdb_tool, env = setup_pdb_repo_env(tmp_path)
    success, output = pdb_tool.breakpoint_add_clear(env, "b 10", None)
    assert not success
    assert output.startswith(
        "Failed to set breakpoint. No file is specified in the command."
    )


def test_breakpoint_add_clear_file_not_in_repo(tmp_path, setup_pdb_repo_env):
    pdb_tool, env = setup_pdb_repo_env(tmp_path)
    env.all_files = ["file1.py"]
    success, output = pdb_tool.breakpoint_add_clear(env, "b 10", "notafile.py")
    assert not success
    assert output.startswith(
        "Failed to set breakpoint. `notafile.py` is not found in the repository."
    )


def test_breakpoint_add_clear_invalid_line_number(tmp_path, setup_pdb_repo_env):
    pdb_tool, env = setup_pdb_repo_env(tmp_path)
    env.all_files = ["file1.py"]
    # file1.py only has 40 lines, so 100 is invalid
    success, output = pdb_tool.breakpoint_add_clear(env, "b 100", "file1.py")
    assert not success
    assert output == "Invalid line number: 100, expected between 1 and 41."


def test_breakpoint_add_clear_invalid_action_type(tmp_path, setup_pdb_repo_env):
    pdb_tool, env = setup_pdb_repo_env(tmp_path)
    env.all_files = ["file1.py"]
    success, output = pdb_tool.breakpoint_add_clear(env, "foo 1", "file1.py")
    assert not success
    assert output == "Invalid action: `foo 1`. Expected 'b', 'break', 'cl', or 'clear'."
