from os.path import join as pjoin
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import numpy as np
import pytest

from froggy.envs import RepoEnv, TooledEnv


@pytest.fixture
def env_mock():
    env = TooledEnv()
    return env


def test_seed(env_mock):
    seed_value = 42
    env_mock.seed(seed_value)
    # Check if the rng attribute is set to a numpy random state
    assert isinstance(env_mock.rng, np.random.RandomState)
    # Check if the random state is initialized with the correct seed
    expected_rng = np.random.RandomState(seed_value)
    state1 = env_mock.rng.get_state()
    state2 = expected_rng.get_state()
    assert state1[0] == state2[0]  # Check the algorithm
    np.testing.assert_array_equal(state1[1], state2[1])  # Check the state
    assert state1[2:] == state2[2:]  # Check the remaining elements


def test_add_tool(env_mock):
    tool = MagicMock()
    tool.name = "tool1"
    env_mock.add_tool(tool)
    assert "tool1" in env_mock.tools
    assert env_mock.tools["tool1"] == tool


def test_add_tool_existing(env_mock):
    tool = MagicMock()
    tool.name = "tool1"
    env_mock.add_tool(tool)
    with pytest.raises(ValueError):
        env_mock.add_tool(tool)


def test_has_tool(env_mock):
    tool = MagicMock()
    tool.name = "tool1"
    env_mock.add_tool(tool)
    assert env_mock.has_tool("tool1")
    assert not env_mock.has_tool("tool2")


def test_get_tool(env_mock):
    tool = MagicMock()
    tool.name = "tool1"
    env_mock.add_tool(tool)
    assert env_mock.get_tool("tool1") == tool


def test_get_triggered_tools(env_mock):
    tool1 = MagicMock()
    tool1.name = "tool1"
    tool1.is_triggered.return_value = True
    tool2 = MagicMock()
    tool2.name = "tool2"
    tool2.is_triggered.return_value = False
    env_mock.add_tool(tool1)
    env_mock.add_tool(tool2)
    triggered_tools = env_mock.get_triggered_tools("action")
    assert tool1 in triggered_tools
    assert tool2 not in triggered_tools


def test_actions(env_mock):
    tool1 = MagicMock()
    tool1.name = "tool1"
    tool1.action = "action1"
    tool2 = MagicMock()
    tool2.name = "tool2"
    tool2.action = "action2"
    env_mock.add_tool(tool1)
    env_mock.add_tool(tool2)
    assert env_mock.actions == ["action1", "action2"]


def test_actions_str(env_mock):
    tool1 = MagicMock()
    tool1.name = "tool1"
    tool1.action = "action1"
    tool2 = MagicMock()
    tool2.name = "tool2"
    tool2.action = "action2"
    env_mock.add_tool(tool1)
    env_mock.add_tool(tool2)
    assert env_mock.actions_str == "action1, action2"


def test_tool_instructions(env_mock):
    tool1 = MagicMock()
    tool1.name = "tool1"
    tool1.instructions = "instructions1"
    tool2 = MagicMock()
    tool2.name = "tool2"
    tool2.instructions = "instructions2"
    env_mock.add_tool(tool1)
    env_mock.add_tool(tool2)
    assert env_mock.tool_instructions == {
        "tool1": "instructions1",
        "tool2": "instructions2",
    }


@patch("tempfile.TemporaryDirectory")
@patch("atexit.register")
def test_setup_workspace(mock_atexit_register, mock_tempdir, tmp_path):
    path_dir = tmp_path / "pathdir"
    path_dir.mkdir()
    file_content = 'print("Hello, World!")'
    with open(path_dir / "file.py", "w") as f:
        f.write(file_content)
    working_dir = tmp_path / "tempdir"
    working_dir.mkdir()
    mock_tempdir.return_value.name = str(working_dir)
    repo_env = RepoEnv(run_timeout=10, dir_tree_depth=2, auto_view_change=True)
    repo_env.setup_workspace(
        path=str(path_dir),
        entrypoint="python",
        readonly_patterns=["readonly_pattern"],
    )

    assert repo_env.path == path_dir
    assert repo_env.working_dir == working_dir
    assert repo_env.tempdir.startswith("RepoEnv-")
    with open(working_dir / "file.py", "r") as f:
        assert f.read() == file_content
    mock_atexit_register.assert_called_once_with(repo_env.tempdir.cleanup)


@patch("tempfile.TemporaryDirectory")
@patch("atexit.register")
@patch("shutil.copytree")
def test_setup_workspace_with_none_path(
    mock_copytree, mock_atexit_register, mock_tempdir
):
    repo_env = RepoEnv(run_timeout=10, dir_tree_depth=2, auto_view_change=True)
    repo_env.setup_workspace(None, "/bin/bash")

    assert repo_env.path is None
    mock_tempdir.assert_not_called()
    mock_copytree.assert_not_called()
    mock_atexit_register.assert_not_called()


@patch("tempfile.TemporaryDirectory")
def test_cleanup_workspace(mock_tempdir):
    mock_tempdir_instance = MagicMock()
    mock_tempdir.return_value = mock_tempdir_instance
    env = RepoEnv()
    env.tempdir = mock_tempdir_instance
    env.cleanup_workspace()

    mock_tempdir_instance.cleanup.assert_called_once()


def test_instructions():
    tool1 = MagicMock()
    tool1.name = "tool1"
    tool1.instructions = "instructions1"
    tool1.action = "action1"
    tool2 = MagicMock()
    tool2.name = "tool2"
    tool2.instructions = "instructions2"
    tool2.action = "action2"

    env = RepoEnv()
    env.add_tool(tool1)
    env.add_tool(tool2)

    expected_instructions = {
        "Available tools to solve the problem": {
            "tool1": "instructions1",
            "tool2": "instructions2",
        },
        "Available commands": "action1, action2",
    }

    instructions = env.instructions
    assert instructions == expected_instructions


@patch("shutil.copy2")
@patch("os.path.isdir", return_value=False)
@patch("glob.glob", return_value=["/path/to/repo/file1.txt", "/path/to/repo/file2.txt"])
@patch("os.scandir")
@patch("os.walk")
@patch("shutil.copytree")
def test_restore(
    mock_copytree, mock_os_walk, mock_scandir, mock_glob, mock_isdir, mock_copy2
):
    mock_scandir.return_value.__enter__.return_value = [
        MagicMock(is_dir=lambda: False, path="/path/to/repo/file1.txt"),
        MagicMock(is_dir=lambda: False, path="/path/to/repo/file2.txt"),
    ]
    mock_os_walk.return_value = [
        ("/path/to/repo", ("subdir",), ("file1.txt", "file2.txt")),
        ("/path/to/repo/subdir", (), ("subfile1.txt",)),
    ]
    env = RepoEnv(path="/path/to/repo")
    env.restore("/path/to/repo/file1.txt", "/path/to/repo/file2.txt")

    mock_glob.assert_not_called()
    mock_isdir.assert_any_call(Path("/path/to/repo/file1.txt"))
    mock_isdir.assert_any_call(Path("/path/to/repo/file2.txt"))
    mock_copy2.assert_any_call(
        Path("/path/to/repo/file1.txt"), Path(env.working_dir) / "file1.txt"
    )
    mock_copy2.assert_any_call(
        Path("/path/to/repo/file2.txt"), Path(env.working_dir) / "file2.txt"
    )


@patch.object(RepoEnv, "directory_tree")
def test_display_files(mock_directory_tree):
    mock_directory_tree.return_value = "\n|-- file1.py\n|-- file2.py\n"
    env = RepoEnv()
    result = env.display_files(editable_only=False)

    expected_result = "\nAll files:\n|-- file1.py\n|-- file2.py\n"
    assert result == expected_result
    mock_directory_tree.assert_called_once_with(editable_only=False)


@patch("froggy.utils.show_line_number")
def test_current_code_with_line_number(mock_show_line_number):
    mock_show_line_number.return_value = "1    def foo():\n2        return 42"
    env = RepoEnv(path=".")
    env.current_file = "file.py"
    env.current_file_content = "def foo():\n    return 42"

    result = env.current_code_with_line_number()
    expected_result = {
        "File name": "file.py",
        "Content": "\n     1 def foo():\n     2     return 42\n",
    }
    assert result == expected_result


@patch.object(RepoEnv, "get_triggered_tools")
@patch.object(RepoEnv, "get_tool")
@patch.object(RepoEnv, "has_tool", return_value=False)
@patch.object(RepoEnv, "run")
@patch.object(RepoEnv, "display_files")
@patch.object(RepoEnv, "current_code_with_line_number")
def test_step(
    mock_current_code_with_line_number,
    mock_display_files,
    mock_run,
    mock_has_tool,
    mock_get_tool,
    mock_get_triggered_tools,
):
    mock_pdb_tool = MagicMock()
    mock_pdb_tool.use.return_value = "PDB tool used"
    mock_pdb_tool.rewrite_success = True
    mock_pdb_tool.current_frame_file = "file.py"
    mock_pdb_tool.pdb_obs = "PDB started"
    mock_get_tool.return_value = None
    mock_display_files.return_value = "file list"
    mock_current_code_with_line_number.return_value = "code with line numbers"

    env = RepoEnv(path=".")
    mock_get_triggered_tools.return_value = [mock_pdb_tool]

    obs, score, done, infos = env.step("some action")

    mock_get_triggered_tools.assert_called_once_with("some action")
    mock_pdb_tool.use.assert_called_once_with("some action")
    assert obs == "PDB tool used"
    assert score == 0
    assert not done
    assert "obs" in infos
    assert "last_run_obs" in infos
    assert "dbg_obs" in infos
    assert "dir_tree" in infos
    assert "editable_files" in infos
    assert "current_breakpoints" in infos
    assert "current_code_with_line_number" in infos
    assert "action" in infos
    assert "done" in infos
    assert "score" in infos
    assert "max_score" in infos
    assert "instructions" in infos
    assert "rewrite_counter" in infos


@patch("froggy.utils._walk")
@patch("pathlib.Path.exists", return_value=True)
@patch("pathlib.Path.is_file", return_value=False)
@patch("os.scandir")
@patch("os.walk")
@patch("shutil.copytree")
@patch("tempfile.TemporaryDirectory")
def test_directory_tree(
    mock_tempdir,
    mock_copytree,
    mock_os_walk,
    mock_scandir,
    mock_is_file,
    mock_exists,
    mock_walk,
):
    mock_tempdir.return_value.name = "/mock/tempdir"
    mock_scandir.return_value.__enter__.return_value = [
        MagicMock(is_dir=lambda: False, path="/path/to/repo/file1.txt"),
        MagicMock(is_dir=lambda: False, path="/path/to/repo/file2.txt"),
    ]
    mock_os_walk.return_value = [
        ("/path/to/repo", ("subdir",), ("file1.py", "file2.py")),
        ("/path/to/repo/subdir", (), ("subfile1.txt",)),
    ]
    env = RepoEnv(path="/path/to/repo")
    result = env.directory_tree()
    expected_result = (
        "\n\n" "/mock/tempdir/\n  " "|-- file1.txt\n  " "|-- file2.txt\n\n"
    )
    assert result == expected_result


@patch.object(RepoEnv, "restore")
@patch.object(RepoEnv, "run")
@patch.object(RepoEnv, "has_tool", return_value=False)
@patch.object(RepoEnv, "get_tool")
@patch("os.scandir")
@patch("os.walk")
@patch("shutil.copytree")
@patch("tempfile.TemporaryDirectory")
def test_reset(
    mock_tempdir,
    mock_copytree,
    mock_os_walk,
    mock_scandir,
    mock_get_tool,
    mock_has_tool,
    mock_run,
    mock_restore,
):
    mock_pdb_tool = MagicMock()
    mock_pdb_tool.start_pseudo_terminal.return_value = None
    mock_pdb_tool.pdb_obs = "PDB started"
    mock_get_tool.return_value = mock_pdb_tool
    mock_tempdir.return_value.name = "/mock/tempdir"
    mock_scandir.return_value.__enter__.return_value = [
        MagicMock(is_dir=lambda: False, path="/path/to/repo/file1.txt"),
        MagicMock(is_dir=lambda: False, path="/path/to/repo/file2.txt"),
    ]
    mock_os_walk.return_value = [
        ("/path/to/repo", ("subdir",), ("file1.py", "file2.py")),
        ("/path/to/repo/subdir", (), ("subfile1.txt",)),
    ]
    env = RepoEnv(path="/path/to/repo")
    obs, infos = env.reset(seed=42)

    mock_restore.assert_called_once()
    mock_run.assert_called_once()
    assert env.current_file is None
    assert env.current_file_content is None
    assert env.current_breakpoints_state == {}
    assert env.rewrite_counter == 0
    assert "obs" in infos
    assert "dbg_obs" in infos
    assert "last_run_obs" in infos
    assert "dir_tree" in infos
    assert "editable_files" in infos
    assert "current_breakpoints" in infos
    assert "current_code_with_line_number" in infos
    assert "action" in infos
    assert "done" in infos
    assert "score" in infos
    assert "max_score" in infos
    assert "instructions" in infos
    assert "rewrite_counter" in infos


@patch("os.scandir")
@patch("os.walk")
@patch("shutil.copytree")
@patch("builtins.open", new_callable=mock_open)
def test_overwrite_file(mock_open_fn, mock_copytree, mock_os_walk, mock_scandir):
    mock_scandir.return_value.__enter__.return_value = [
        MagicMock(is_dir=lambda: False, path="/path/to/repo/file1.txt"),
        MagicMock(is_dir=lambda: False, path="/path/to/repo/file2.txt"),
    ]
    mock_os_walk.return_value = [
        ("/path/to/repo", ("subdir",), ("file1.py", "file2.py")),
        ("/path/to/repo/subdir", (), ("subfile1.txt",)),
    ]
    env = RepoEnv(path="/path/to/repo")
    filepath = "file.py"
    content = 'print("Hello, World!")'
    env.overwrite_file(filepath, content)

    mock_open_fn.assert_called_once_with(pjoin(env.working_dir, filepath), "w")
    mock_open_fn().write.assert_called_once_with(content)


@patch("os.scandir")
@patch("os.walk")
@patch("shutil.copytree")
@patch("subprocess.run")
def test_patch(mock_subprocess_run, mock_copytree, mock_os_walk, mock_scandir):
    mock_result = MagicMock()
    mock_result.stdout = "diff --git a/path/to/repo/file1.py b/path/to/repo/file1.py\n"
    mock_subprocess_run.return_value = mock_result
    mock_scandir.return_value.__enter__.return_value = [
        MagicMock(is_dir=lambda: False, path="/path/to/repo/file1.txt"),
        MagicMock(is_dir=lambda: False, path="/path/to/repo/file2.txt"),
    ]
    mock_os_walk.return_value = [
        ("/path/to/repo", ("subdir",), ("file1.py", "file2.py")),
        ("/path/to/repo/subdir", (), ("subfile1.txt",)),
    ]
    env = RepoEnv(path="/path/to/repo")
    result = env.patch
    expected_result = "diff --git a/path/to/repo/file1.py b/path/to/repo/file1.py\n"

    mock_subprocess_run.assert_called_once_with(
        ["git", "diff", "--no-index", env.path, env.working_dir],
        text=True,
        capture_output=True,
    )
    assert result == expected_result


def test_run_success(tmp_path):
    working_dir = str(tmp_path)
    # create a dummy file
    with open(tmp_path / "file.py", "w") as f:
        f.write("print('Hello, World!')")
    env = RepoEnv(path=working_dir, entrypoint="python file.py")
    output, done = env.run()

    assert output == "Hello, World!"
    assert done
    assert env.score == 1


def test_run_timeout(tmp_path):
    working_dir = str(tmp_path)
    # runs for longer than the timeout
    with open(tmp_path / "file.py", "w") as f:
        f.write("import time; time.sleep(5)")
    env = RepoEnv(path=working_dir, entrypoint="python file.py", run_timeout=1)
    output, done = env.run()

    assert output == "Timeout expired."
    assert not done
    assert env.score == 0
