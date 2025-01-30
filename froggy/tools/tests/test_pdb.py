import subprocess
from unittest.mock import MagicMock, patch

import pytest

from froggy.envs.env import RepoEnv
from froggy.terminal import DockerTerminal, Terminal
from froggy.tools.pdb import PDBTool

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
    environment = RepoEnv(path=tests_path, terminal=terminal)
    pdb = PDBTool()
    pdb.register(environment)
    initial_output = pdb.start_pdb(pdb_cmd="python -m pdb -m pytest -sv .")
    assert """The pytest entry point.""" in initial_output
    assert "(Pdb)" in initial_output
    output = pdb.use("pdb l")
    assert """The pytest entry point.""" in output
    assert "(Pdb)" in output
    output = pdb.use("pdb c")
    assert "1 failed, 1 passed" in pdb.pdb_obs
    assert "test_fail.py::test_fail FAILED" in pdb.pdb_obs
    assert "test_pass.py::test_pass PASSED" in pdb.pdb_obs
    assert "Reached the end of the file. Restarting the debugging session." in output
    assert "(Pdb)" in output


def test_pdb_use_default_environment_entrypoint(tmp_path, setup_test_repo):
    # Test PDBTool with default environment entrypoint, quite pytest
    tests_path = str(setup_test_repo(tmp_path))
    terminal = Terminal()
    environment = RepoEnv(path=tests_path, terminal=terminal)
    pdb = PDBTool()
    pdb.register(environment)
    initial_output = pdb.start_pdb()  # "python -m pdb -m pytest -sq ."
    assert """The pytest entry point.""" in initial_output
    assert "(Pdb)" in initial_output
    output = pdb.use("pdb l")
    assert """The pytest entry point.""" in output
    assert "(Pdb)" in output
    output = pdb.use("pdb c")
    assert "1 failed, 1 passed" in pdb.pdb_obs
    assert "test_fail.py::test_fail" in pdb.pdb_obs
    assert "test_pass.py::test_pass" not in pdb.pdb_obs
    assert "Reached the end of the file. Restarting the debugging session." in output
    assert "(Pdb)" in output


@if_docker_running
def test_pdb_use_docker_terminal(tmp_path, setup_test_repo):
    """Test PDBTool similar to test_pdb_use but using DockerTerminal"""
    tests_path = str(setup_test_repo(tmp_path))
    terminal = DockerTerminal(
        working_dir=tests_path,
        base_image="python:3.12-slim",
        setup_commands=["pip install pytest"],
        volumes={tests_path: {"bind": tests_path, "mode": "rw"}},
    )
    environment = MagicMock()
    environment.working_dir = tests_path
    pdb = PDBTool()
    pdb.environment = environment
    pdb_cmd = f"python -m pdb -m pytest -sv ."
    pdb.start_pdb(terminal, pdb_cmd)

    output = pdb.use("pdb l")
    assert """The pytest entry point.""" in output
    assert "(Pdb)" in output
    output = pdb.use("pdb c")
    assert "1 failed, 1 passed" in pdb.pdb_obs
    assert "test_fail.py::test_fail FAILED" in pdb.pdb_obs
    assert "test_pass.py::test_pass PASSED" in pdb.pdb_obs
    assert "Reached the end of the file. Restarting the debugging session." in output
    assert "(Pdb)" in output


def test_initialization():
    pdb_tool = PDBTool()
    assert pdb_tool.master is None
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
def test_start_pdb(mock_interact_with_pdb):
    mock_interact_with_pdb.return_value = "(Pdb)"
    env = RepoEnv()
    pdb_tool = PDBTool()
    pdb_tool.register(env)
    output = pdb_tool.start_pdb(pdb_cmd="python script.py")
    assert output == "(Pdb)"
    assert pdb_tool.pdb_obs == "(Pdb)"


@patch.object(PDBTool, "interact_with_pdb")
@patch.object(PDBTool, "close_pdb")
def test_restart_pdb(mock_close_pdb, mock_interact_with_pdb):
    mock_interact_with_pdb.return_value = "(Pdb)"
    env = RepoEnv()
    pdb_tool = PDBTool()
    pdb_tool.register(env)
    env.entrypoint = "python script.py"
    env.debug_entrypoint = "python -m pdb script.py"
    output = pdb_tool.restart_pdb()
    assert output == "(Pdb)"
    assert pdb_tool.pdb_obs == "(Pdb)"


@patch.object(PDBTool, "interact_with_pdb")
def test_use_command(mock_interact_with_pdb, tmp_path):
    mock_interact_with_pdb.return_value = "output"
    env = RepoEnv()
    env.working_dir = str(tmp_path)
    pdb_tool = PDBTool()
    pdb_tool.register(env)
    env.current_file = "script.py"
    env.all_files = ["script.py"]
    output = pdb_tool.use("```pdb p x```")
    assert "output" in output


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


def test_current_breakpoints_no_breakpoints():
    env = RepoEnv()
    pdb_tool = PDBTool()
    pdb_tool.register(env)
    pdb_tool.environment.current_breakpoints_state = {}
    result = pdb_tool.current_breakpoints()
    assert result == "No breakpoints are set."


def test_current_breakpoints_with_breakpoints(tmp_path, setup_pdb_repo_env):
    pdb_tool, test_repo, env = setup_pdb_repo_env(tmp_path)
    result = pdb_tool.current_breakpoints()
    expected_result = (
        "line 10 in file1.py\n"
        "line 20 in file1.py\n"
        "line 30 in file1.py\n"
        "line 15 in file2.py"
    )
    assert result == expected_result


@patch.object(PDBTool, "interact_with_pdb")
def test_get_current_frame_file(mock_interact_with_pdb, tmp_path, setup_pdb_repo_env):
    pdb_tool, test_repo, env = setup_pdb_repo_env(tmp_path)
    fail_test_path = str(env.working_dir / "test_fail.py")
    mock_interact_with_pdb.return_value = (
        f"somecontext > {fail_test_path}(2)<module>()\n-> some code context"
    )
    pdb_tool.get_current_frame_file()
    assert str(fail_test_path).endswith(pdb_tool.current_frame_file)
