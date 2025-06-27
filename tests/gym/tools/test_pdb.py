import copy
import platform
import re
import subprocess
from unittest.mock import MagicMock

import pytest

from debug_gym.gym.entities import Event
from debug_gym.gym.envs.env import RepoEnv
from debug_gym.gym.terminal import DockerTerminal, Terminal
from debug_gym.gym.tools.pdb import PDBTool


def is_docker_running():
    try:
        subprocess.check_output(["docker", "ps"])
        return True
    except Exception:
        return False


if_docker_running = pytest.mark.skipif(
    not is_docker_running(),
    reason="Docker not running",
)


if_is_linux = pytest.mark.skipif(
    platform.system() != "Linux",
    reason="Interactive ShellSession (pty) requires Linux.",
)


def clean_up_pytest_path(obs):
    """clean up the pytest path to not depend on the environment"""
    return re.sub(
        r"Current frame:\n.*pytest/__main__\.py",
        "Current frame:\n.../pytest/__main__.py",
        obs,
    )


@pytest.fixture
def setup_breakpoints_state():
    def _breakpoints_state(working_dir):
        return {
            f"{working_dir}/file1.py|||10": f"b {working_dir}/file1.py:10",
            f"{working_dir}/file1.py|||20": f"b {working_dir}/file1.py:20",
            f"{working_dir}/file1.py|||30": f"b {working_dir}/file1.py:30",
            f"{working_dir}/file2.py|||15": f"b {working_dir}/file2.py:15",
        }

    return _breakpoints_state


@pytest.fixture
def setup_pdb_repo_env(setup_test_repo, setup_breakpoints_state):
    def _setup_pdb_repo_env(base_dir):
        test_repo = setup_test_repo(base_dir)
        env = RepoEnv(path=str(test_repo))
        pdb_tool = PDBTool()
        pdb_tool.register(env)
        env.reset()
        breakpoints = setup_breakpoints_state(env.working_dir)
        env.current_breakpoints_state = breakpoints
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
    assert 'Context around the current frame:\n  1  ->	"""The pytest entry point.""""'
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
    assert 'Context around the current frame:\n  1  ->	"""The pytest entry point.""""'
    assert "(Pdb)" not in output


@if_is_linux
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
    assert 'Context around the current frame:\n  1  ->	"""The pytest entry point.""""'
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


def test_pdb_add_new_breakpoint_relative_path(tmp_path, setup_pdb_repo_env):
    pdb_tool, env = setup_pdb_repo_env(tmp_path)
    breakpoints_state = copy.deepcopy(env.current_breakpoints_state)
    wd = env.working_dir
    pdb_obs = pdb_tool.use(env, "b file1.py:25")
    assert "Pdb command output:\nBreakpoint 5 at file1.py:25" in pdb_obs.observation
    new_breakpoint = {f"{wd}/file1.py|||25": f"b {wd}/file1.py:25"}
    expected_state = breakpoints_state | new_breakpoint
    assert env.current_breakpoints_state == expected_state


def test_pdb_add_new_breakpoint_absolute_path(tmp_path, setup_pdb_repo_env):
    pdb_tool, env = setup_pdb_repo_env(tmp_path)
    breakpoints_state = copy.deepcopy(env.current_breakpoints_state)
    wd = env.working_dir
    pdb_obs = pdb_tool.use(env, f"b {wd}/file1.py:25")
    assert "Pdb command output:\nBreakpoint 5 at file1.py:25" in pdb_obs.observation
    new_breakpoint = {f"{wd}/file1.py|||25": f"b {wd}/file1.py:25"}
    expected_state = breakpoints_state | new_breakpoint
    assert env.current_breakpoints_state == expected_state


def test_pdb_add_existing_breakpoint(tmp_path, setup_pdb_repo_env):
    pdb_tool, env = setup_pdb_repo_env(tmp_path)
    breakpoints_state = copy.deepcopy(env.current_breakpoints_state)
    pdb_obs = pdb_tool.use(env, "b file1.py:10")
    assert "Pdb command output:\nBreakpoint 5 at file1.py:10" in pdb_obs.observation
    assert env.current_breakpoints_state == breakpoints_state


def test_pdb_clear_specific(tmp_path, setup_pdb_repo_env):
    pdb_tool, env = setup_pdb_repo_env(tmp_path)
    pdb_obs = pdb_tool.use(env, "cl file1.py:20")
    wd = str(env.working_dir)
    expected_state = {
        f"{wd}/file1.py|||10": f"b {wd}/file1.py:10",
        f"{wd}/file1.py|||30": f"b {wd}/file1.py:30",
        f"{wd}/file2.py|||15": f"b {wd}/file2.py:15",
    }
    assert pdb_obs.observation.startswith(
        "Pdb command output:\nDeleted breakpoint 2 at file1.py:20\n\nCurrent frame:"
    )
    assert env.current_breakpoints_state == expected_state


def test_pdb_clear_not_found(
    tmp_path,
    setup_pdb_repo_env,
):
    pdb_tool, env = setup_pdb_repo_env(tmp_path)
    breakpoints_state = copy.deepcopy(env.current_breakpoints_state)
    pdb_obs = pdb_tool.use(env, "cl file1.py:8")
    assert pdb_obs.observation.startswith(
        "Pdb command output:\n*** There is no breakpoint at file1.py:8"
    )
    assert env.current_breakpoints_state == breakpoints_state


def test_breakpoint_modify_remove(tmp_path, setup_pdb_repo_env):
    # Remove breakpoint at line 20 and move breakpoint at line 30 to line 24
    pdb_tool, env = setup_pdb_repo_env(tmp_path)
    pdb_tool.breakpoint_modify(env, "file1.py", 15, 25, 5)
    wd = str(env.working_dir)
    expected_state = {
        f"{wd}/file1.py|||10": f"b {wd}/file1.py:10",
        f"{wd}/file1.py|||24": f"b {wd}/file1.py:24",
        f"{wd}/file2.py|||15": f"b {wd}/file2.py:15",
    }
    assert env.current_breakpoints_state == expected_state


def test_breakpoint_modify_move(tmp_path, setup_pdb_repo_env):
    pdb_tool, env = setup_pdb_repo_env(tmp_path)
    pdb_tool.breakpoint_modify(env, "file1.py", 5, 15, 10)
    wd = str(env.working_dir)
    expected_state = {
        f"{wd}/file1.py|||19": f"b {wd}/file1.py:19",
        f"{wd}/file1.py|||29": f"b {wd}/file1.py:29",
        f"{wd}/file2.py|||15": f"b {wd}/file2.py:15",
    }
    assert env.current_breakpoints_state == expected_state


def test_breakpoint_modify_remove_all(tmp_path, setup_pdb_repo_env):
    pdb_tool, env = setup_pdb_repo_env(tmp_path)
    wd = str(env.working_dir)
    pdb_tool.breakpoint_modify(env, "file1.py", None, None, 0)
    expected_state = {f"{wd}/file2.py|||15": f"b {wd}/file2.py:15"}
    assert env.current_breakpoints_state == expected_state


def test_breakpoint_modify_no_change(tmp_path, setup_pdb_repo_env):
    pdb_tool, env = setup_pdb_repo_env(tmp_path)
    pdb_tool.breakpoint_modify(env, "file1.py", 25, 35, 5)
    # Test no change for breakpoints before the rewritten code (change line 30)
    wd = str(env.working_dir)
    expected_state = {
        f"{wd}/file1.py|||10": f"b {wd}/file1.py:10",
        f"{wd}/file1.py|||20": f"b {wd}/file1.py:20",
        f"{wd}/file2.py|||15": f"b {wd}/file2.py:15",
    }
    assert env.current_breakpoints_state == expected_state


def test_breakpoint_modify_no_breakpoints(tmp_path, setup_pdb_repo_env):
    pdb_tool, env = setup_pdb_repo_env(tmp_path)
    env.current_breakpoints_state = {}
    initial_output = pdb_tool.start_pdb(env)
    assert "The pytest entry point." in initial_output
    pdb_tool.breakpoint_modify(env, "rewrite_file", 1, 3, 1)
    assert env.current_breakpoints_state == {}


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
    obs = pdb_tool.use(env, "b 1; b 2").observation
    assert "Multiple commands are not supported" in obs


def test_use_empty_command_returns_failure(tmp_path, setup_pdb_repo_env):
    pdb_tool, env = setup_pdb_repo_env(tmp_path)
    obs = pdb_tool.use(env, "").observation
    assert "Empty commands are not allowed" in obs


def test_use_breakpoints_and_clear(tmp_path, setup_pdb_repo_env):
    pdb_tool, env = setup_pdb_repo_env(tmp_path)
    env.current_breakpoints_state = {"file1.py|||1": "b file1.py:1"}
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
        '  1  ->\t"""The pytest entry point."""\r\n'
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
        '  1  ->\t"""The pytest entry point."""\r\n'
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
        '  1  ->\t"""The pytest entry point."""\r\n'
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
    assert pdb_tool.current_frame_file == f"{env.working_dir}/test_fail.py"
    # observation should contain the test file
    assert f"Current frame:\n{env.working_dir}/test_fail.py" in obs.observation
    assert "> test_fail.py(2)" in obs.observation


def test_set_current_frame_file_sets_and_returns(tmp_path, setup_pdb_repo_env):
    pdb_tool, env = setup_pdb_repo_env(tmp_path)
    # Patch interact_with_pdb to simulate output
    test_path = str(env.working_dir / "file1.py")

    def fake_interact_with_pdb(command, timeout):
        return f"somecontext\n> {test_path}(10)<module>()\n-> some code context"

    pdb_tool.interact_with_pdb = fake_interact_with_pdb
    result = pdb_tool.set_current_frame_file(env)
    assert result == test_path
    assert pdb_tool.current_frame_file == test_path


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
        '  1  ->\t"""The pytest entry point."""\r\n'
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


def test_pdb_list_output_indentation(tmp_path, setup_pdb_repo_env):
    """Test PDB list output indentation for line numbers around 100 (3-digit)"""
    pdb_tool, env = setup_pdb_repo_env(tmp_path)
    with (env.working_dir / "large_file.py").open("w") as f:
        f.write("def dummy_function():\n")
        f.write("\n".join(f"    'Line {i+1}'" for i in range(1, 2000)))
        f.write("\n\nif __name__ == '__main__':\n")
        f.write("    dummy_function()\n")
    env.set_entrypoints("python large_file.py", "python -m pdb large_file.py")
    pdb_tool.start_pdb(env)
    pdb_obs = pdb_tool.use(env, "b large_file.py:100")
    assert (
        "Pdb command output:\nBreakpoint 5 at large_file.py:100"
    ) in pdb_obs.observation
    pdb_obs = pdb_tool.use(env, "c")
    expected_output = (
        "Context around the current frame:\n"
        " 95  \t    'Line 95'\r\n"
        " 96  \t    'Line 96'\r\n"
        " 97  \t    'Line 97'\r\n"
        " 98  \t    'Line 98'\r\n"
        " 99  \t    'Line 99'\r\n"
        "100 B->\t    'Line 100'\r\n"
        "101  \t    'Line 101'\r\n"
        "102  \t    'Line 102'\r\n"
        "103  \t    'Line 103'\r\n"
        "104  \t    'Line 104'\r\n"
        "105  \t    'Line 105'\n"
    )
    assert expected_output in pdb_obs.observation

    pdb_obs = pdb_tool.use(env, "b large_file.py:1000")
    assert (
        "Pdb command output:\nBreakpoint 6 at large_file.py:1000" in pdb_obs.observation
    )
    pdb_obs = pdb_tool.use(env, "c")
    expected_output = (
        "Context around the current frame:\n"
        "995  \t    'Line 995'\r\n"
        "996  \t    'Line 996'\r\n"
        "997  \t    'Line 997'\r\n"
        "998  \t    'Line 998'\r\n"
        "999  \t    'Line 999'\r\n"
        "1000B->\t    'Line 1000'\r\n"
        "1001 \t    'Line 1001'\r\n"
        "1002 \t    'Line 1002'\r\n"
        "1003 \t    'Line 1003'\r\n"
        "1004 \t    'Line 1004'\r\n"
        "1005 \t    'Line 1005'\n"
    )
    assert expected_output in pdb_obs.observation

    pdb_obs = pdb_tool.use(env, "b large_file.py:2000")
    assert (
        "Pdb command output:\nBreakpoint 7 at large_file.py:2000" in pdb_obs.observation
    )
    pdb_obs = pdb_tool.use(env, "c")
    expected_output = (
        "Context around the current frame:\n"
        "1995 \t    'Line 1995'\r\n"
        "1996 \t    'Line 1996'\r\n"
        "1997 \t    'Line 1997'\r\n"
        "1998 \t    'Line 1998'\r\n"
        "1999 \t    'Line 1999'\r\n"
        "2000B->\t    'Line 2000'\r\n"
        "2001 \t\r\n"
        "2002 \tif __name__ == '__main__':\r\n"
        "2003 \t    dummy_function()\r\n"
        "[EOF]\n"
    )
    assert expected_output in pdb_obs.observation


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
    obs = pdb_tool.use(env, "b").observation
    assert obs.startswith("Breakpoints:\nline 1 in file1.py")
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


def test_pdbtool_pickle_roundtrip(tmp_path, setup_pdb_repo_env):
    """
    A PDBTool should survive a pickle --> un-pickle cycle.
    The non-serialisable _session and current_frame_file
    must be stripped to None, while the rest of the state
    (e.g. its class-level name) is preserved.
    """
    import pickle

    pdb_tool, _env = setup_pdb_repo_env(tmp_path)
    dumped = pickle.dumps(pdb_tool)
    rehydrated = pickle.loads(dumped)

    assert rehydrated._session is None
    assert rehydrated.current_frame_file is None

    assert rehydrated.name == pdb_tool.name
    assert rehydrated.examples == pdb_tool.examples
