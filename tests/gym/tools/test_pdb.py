import subprocess
from unittest.mock import MagicMock, patch

import pytest

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
        with working_dir.joinpath("test_pass.py").open("w") as f:
            f.write("def test_pass():\n    assert True")
        with working_dir.joinpath("test_fail.py").open("w") as f:
            f.write("def test_fail():\n    assert False")
        dummy_files = ["file1.py", "file2.py"]
        for dummy_file in dummy_files:
            with working_dir.joinpath(dummy_file).open("w") as f:
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
        env.current_breakpoints_state = breakpoints_state
        env.current_file = "file1.py"
        env.all_files = ["file1.py", "file2.py"]
        pdb_tool = PDBTool()
        pdb_tool.register(env)
        return pdb_tool, test_repo, env

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
    pdb.register(environment)
    initial_output = pdb.start_pdb()
    assert """The pytest entry point.""" in initial_output
    assert "(Pdb)" not in initial_output

    output = pdb.use("l").observation
    assert """The pytest entry point.""" in output
    assert "(Pdb)" not in output

    output = pdb.use("c").observation
    assert "1 failed, 1 passed" in pdb.pdb_obs
    assert "test_fail.py::test_fail FAILED" in pdb.pdb_obs
    assert "test_pass.py::test_pass PASSED" in pdb.pdb_obs
    assert "Reached the end of the file. Restarting the debugging session." in output
    assert "(Pdb)" not in output


def test_pdb_use_multiple_commands(tmp_path, setup_test_repo):
    # Test PDBTool with Terminal, verbose pytest
    tests_path = str(setup_test_repo(tmp_path))
    terminal = Terminal()
    environment = RepoEnv(
        path=tests_path,
        terminal=terminal,
        debug_entrypoint="python -m pdb -m pytest -sv .",
    )
    pdb = PDBTool()
    pdb.register(environment)
    _ = pdb.start_pdb()

    output = pdb.use("l ; print('hello')").observation
    assert (
        """Multiple commands are not supported. Only the first command will be executed."""
        in output
    )
    assert """The pytest entry point.""" in output
    assert "(Pdb)" not in output

    output = pdb.use("print('hello;\nhi')").observation
    assert (
        """Multiple commands are not supported. Only the first command will be executed."""
        not in output
    )
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
    pdb.register(environment)
    _ = pdb.start_pdb()

    output = pdb.use("").observation
    assert """Tool failure:\nEmpty command.""" in output


def test_pdb_use_default_environment_entrypoint(tmp_path, setup_test_repo):
    # Test PDBTool with default environment entrypoint, quiet pytest
    tests_path = str(setup_test_repo(tmp_path))
    terminal = Terminal()
    environment = RepoEnv(path=tests_path, terminal=terminal)
    pdb = PDBTool()
    pdb.register(environment)
    initial_output = pdb.start_pdb()  # "python -m pdb -m pytest -sq ."
    assert """The pytest entry point.""" in initial_output
    assert "(Pdb)" not in initial_output

    output = pdb.use("l").observation
    assert """The pytest entry point.""" in output
    assert "(Pdb)" not in output

    output = pdb.use("c").observation
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
    debug_entrypoint = f"python -m pdb -m pytest -p no:cacheprovider -sv ."
    environment = RepoEnv(
        path=tests_path, terminal=terminal, debug_entrypoint=debug_entrypoint
    )
    pdb = PDBTool()
    pdb.register(environment)
    pdb.start_pdb()

    output = pdb.use("l").observation
    assert """The pytest entry point.""" in output
    assert "(Pdb)" not in output

    output = pdb.use("c").observation
    assert "1 failed, 1 passed" in pdb.pdb_obs
    assert "test_fail.py::test_fail FAILED" in pdb.pdb_obs
    assert "test_pass.py::test_pass PASSED" in pdb.pdb_obs
    assert "Reached the end of the file. Restarting the debugging session." in output
    assert "(Pdb)" not in output


def test_initialization():
    pdb_tool = PDBTool()
    assert pdb_tool.pdb_obs == ""
    assert not pdb_tool.persistent_breakpoints
    assert pdb_tool.auto_list
    assert pdb_tool.current_frame_file is None
    assert pdb_tool._session is None


def test_register():
    env = RepoEnv()
    pdb_tool = PDBTool()
    pdb_tool.register(env)
    assert pdb_tool.environment == env


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
    pdb_tool, test_repo, env = setup_pdb_repo_env(tmp_path)
    success, output = pdb_tool.breakpoint_add_clear("b 25")
    assert success
    assert output == pdb_message
    expected_state = {"file1.py|||25": "b file1.py:25"} | breakpoints_state
    assert env.current_breakpoints_state == expected_state


def test_breakpoint_add_clear_add_existing_breakpoint(
    tmp_path, setup_pdb_repo_env, breakpoints_state
):
    pdb_tool, test_repo, env = setup_pdb_repo_env(tmp_path)
    success, output = pdb_tool.breakpoint_add_clear("b 10")
    assert success
    assert output == "Breakpoint already exists at line 10 in file1.py."
    assert env.current_breakpoints_state == breakpoints_state


@patch.object(PDBTool, "interact_with_pdb")
def test_breakpoint_add_clear_clear_specific(
    mock_interact_with_pdb, tmp_path, setup_pdb_repo_env
):
    pdb_message = "Deleted breakpoint 2 at file1.py:20"
    mock_interact_with_pdb.return_value = pdb_message
    pdb_tool, test_repo, env = setup_pdb_repo_env(tmp_path)
    success, output = pdb_tool.breakpoint_add_clear("cl 20")
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
    pdb_tool, test_repo, env = setup_pdb_repo_env(tmp_path)
    success, output = pdb_tool.breakpoint_add_clear("cl 8")
    assert success
    assert output == "No breakpoint exists at line 8 in file1.py."
    assert env.current_breakpoints_state == breakpoints_state


def test_breakpoint_modify_remove(tmp_path, setup_pdb_repo_env):
    # Remove breakpoint at line 20 and move breakpoint at line 30 to line 24
    # TODO: 24 or 25?
    pdb_tool, test_repo, env = setup_pdb_repo_env(tmp_path)
    pdb_tool.breakpoint_modify("file1.py", 15, 25, 5)
    expected_state = {
        "file1.py|||10": "b file1.py:10",
        "file1.py|||24": "b file1.py:24",
        "file2.py|||15": "b file2.py:15",
    }
    assert env.current_breakpoints_state == expected_state


def test_breakpoint_modify_move(tmp_path, setup_pdb_repo_env):
    pdb_tool, test_repo, env = setup_pdb_repo_env(tmp_path)
    pdb_tool.breakpoint_modify("file1.py", 5, 15, 10)
    expected_state = {
        "file2.py|||15": "b file2.py:15",
        "file1.py|||19": "b file1.py:19",
        "file1.py|||29": "b file1.py:29",
    }
    assert env.current_breakpoints_state == expected_state


def test_breakpoint_modify_remove_all(tmp_path, setup_pdb_repo_env):
    pdb_tool, test_repo, env = setup_pdb_repo_env(tmp_path)
    pdb_tool.breakpoint_modify("file1.py", None, None, 0)
    expected_state = {"file2.py|||15": "b file2.py:15"}
    assert env.current_breakpoints_state == expected_state


def test_breakpoint_modify_no_change(tmp_path, setup_pdb_repo_env):
    pdb_tool, test_repo, env = setup_pdb_repo_env(tmp_path)
    pdb_tool.breakpoint_modify("file1.py", 25, 35, 5)
    # Test no change for breakpoints before the rewritten code (change line 30)
    expected_state = {
        "file1.py|||10": "b file1.py:10",
        "file1.py|||20": "b file1.py:20",
        "file2.py|||15": "b file2.py:15",
    }
    assert env.current_breakpoints_state == expected_state


@patch.object(PDBTool, "interact_with_pdb")
def test_get_current_frame_file(mock_interact_with_pdb, tmp_path, setup_pdb_repo_env):
    pdb_tool, test_repo, env = setup_pdb_repo_env(tmp_path)
    fail_test_path = str(env.working_dir / "test_fail.py")
    mock_interact_with_pdb.return_value = (
        f"somecontext > {fail_test_path}(2)<module>()\n-> some code context"
    )
    pdb_tool.get_current_frame_file()
    assert str(fail_test_path).endswith(pdb_tool.current_frame_file)


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
    pdb.register(environment)

    initial_output = pdb.start_pdb()
    assert "The pytest entry point." in initial_output
    output = pdb.interact_with_pdb("c")
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
    pdb.register(environment)

    initial_output = pdb.start_pdb()
    assert "The pytest entry point." in initial_output
    output = pdb.interact_with_pdb("c", timeout=1)
    assert "timed out" in output
    assert pdb.pdb_is_running is False
