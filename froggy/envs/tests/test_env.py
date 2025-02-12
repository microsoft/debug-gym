import os
from os.path import join as pjoin
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import numpy as np
import pytest

from froggy.envs.env import EnvInfo, EventHooks, RepoEnv, TooledEnv


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


@pytest.fixture
def env(tmp_path):
    tmp_path = Path(tmp_path)
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    subdir_path = repo_path / "subdir"
    subdir_path.mkdir()
    (repo_path / "file1.txt").touch()
    (repo_path / "file2.txt").touch()
    (subdir_path / "subfile1.txt").touch()

    env = RepoEnv(path=repo_path, dir_tree_depth=2)
    return env


def test_restore(env):
    # Change the content of a file
    file1 = env.working_dir / "file1.txt"
    with open(file1, "w") as f:
        f.write("Hello, World!")

    def hash_file(file):
        with open(file, "rb") as f:
            return hash(f.read())

    assert hash_file(env.path / "file1.txt") != hash_file(file1)
    env.restore()
    assert hash_file(env.path / "file1.txt") == hash_file(file1)


def test_display_files(env):
    result = env.display_files()
    assert result == (
        "Listing files in the current working directory. (ro) indicates read-only files. Max depth: 2.\n"
        f"{env.working_dir}/\n"
        "|-- file1.txt\n"
        "|-- file2.txt\n"
        "|-- subdir/\n"
        "  |-- subfile1.txt"
    )


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

    infos = env.step("some action")

    mock_get_triggered_tools.assert_called_once_with("some action")
    mock_pdb_tool.use.assert_called_once_with("some action")
    assert infos.obs == "PDB tool used"
    assert infos.score == 0
    assert not infos.done
    assert isinstance(infos, EnvInfo)


def test_directory_tree(tmp_path):
    tmp_path = Path(tmp_path)
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    subdir_path = repo_path / "subdir"
    subdir_path.mkdir()
    (repo_path / "file1.txt").touch()
    (repo_path / "file2.txt").touch()
    (subdir_path / "subfile1.txt").touch()

    env = RepoEnv(path=repo_path, dir_tree_depth=3)
    result = env.directory_tree()
    expected_result = (
        f"{env.working_dir}/\n"
        "|-- file1.txt\n"
        "|-- file2.txt\n"
        "|-- subdir/\n"
        "  |-- subfile1.txt"
    )
    assert result == expected_result


@patch.object(RepoEnv, "restore")
@patch.object(RepoEnv, "run")
@patch.object(RepoEnv, "get_tool")
def test_reset(
    mock_get_tool,
    mock_run,
    mock_restore,
    env,
):
    mock_pdb_tool = MagicMock()
    mock_pdb_tool.start_pseudo_terminal.return_value = None
    mock_pdb_tool.pdb_obs = "PDB started"
    mock_get_tool.return_value = mock_pdb_tool
    infos = env.reset(seed=42)

    mock_restore.assert_called_once()
    mock_run.assert_called_once()
    assert env.current_file is None
    assert env.current_file_content is None
    assert env.current_breakpoints_state == {}
    assert env.rewrite_counter == 0
    assert infos == EnvInfo(
        obs="",
        last_run_obs=None,
        dbg_obs="",
        dir_tree=f"""Listing files in the current working directory. (ro) indicates read-only files. Max depth: 2.
{env.tempdir.name}/
|-- file1.txt
|-- file2.txt
|-- subdir/
  |-- subfile1.txt""",
        current_code_with_line_number="You are currently not working in a file. You can use ```view path/to/file.py``` to navigate to a file first.",
        current_breakpoints="No breakpoints are set.",
        action=None,
        instructions={
            "Available tools to solve the problem": {},
            "Available commands": "",
        },
        score=0,
        max_score=1,
        done=False,
        rewrite_counter=0,
        tools={},
    )


def test_overwrite_file(env):
    filepath = "file.py"
    content = 'print("Hello, World!")'
    env.overwrite_file(filepath, content)

    with open(env.working_dir / filepath, "r") as f:
        assert f.read() == content


def test_patch(env):
    # Change the content of a file
    file1 = env.working_dir / "file1.txt"
    with open(file1, "w") as f:
        f.write("Hello, World!")

    result = env.patch
    expected = (
        f"diff --git a{env.path}/file1.txt b{env.path}/file1.txt\n"
        "index e69de29..b45ef6f 100644\n"
        f"--- a{env.path}/file1.txt\n"
        f"+++ b{env.path}/file1.txt\n"
        "@@ -0,0 +1 @@\n"
        "+Hello, World!\n"
        "\\ No newline at end of file\n"
    )
    assert result == expected


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


def test_env_info_initialization():
    current_code = {"File name": "test.py", "Content": "print('test')"}
    tools = {"tool1": {"template": "```tool1 ... ```"}}
    info = EnvInfo(
        obs="observation",
        last_run_obs="last_run",
        dbg_obs="debug",
        dir_tree="tree",
        current_code_with_line_number=current_code,
        current_breakpoints="breakpoints",
        action="test_action",
        instructions={"tool": "instruction"},
        score=5,
        max_score=10,
        done=False,
        rewrite_counter=2,
        tools=tools,
    )

    assert info.obs == "observation"
    assert info.last_run_obs == "last_run"
    assert info.dbg_obs == "debug"
    assert info.dir_tree == "tree"
    assert info.current_code_with_line_number == current_code
    assert info.current_breakpoints == "breakpoints"
    assert info.action == "test_action"
    assert info.instructions == {"tool": "instruction"}
    assert info.score == 5
    assert info.max_score == 10
    assert not info.done
    assert info.rewrite_counter == 2
    assert info.tools == tools


def test_event_hooks_initialization():
    event_hooks = EventHooks()
    assert event_hooks.events == ["on_start", "on_reset", "on_step"]
    assert event_hooks.event_listeners == {
        "on_start": [],
        "on_reset": [],
        "on_step": [],
    }


def test_event_hooks_subscribe():
    class ToolMock:
        def on_start(self):
            pass

    event_hooks = EventHooks()
    subscriber = ToolMock()
    event_hooks.subscribe("on_start", subscriber)
    assert subscriber in event_hooks.event_listeners["on_start"]


def test_event_hooks_subscribe_invalid_subscriber():
    class InvalidToolMock:
        pass

    event_hooks = EventHooks()
    subscriber = InvalidToolMock()
    with pytest.raises(ValueError, match="Tool does not implement method on_start"):
        event_hooks.subscribe("on_start", subscriber)
    assert subscriber not in event_hooks.event_listeners["on_start"]


def test_event_hooks_subscribe_invalid_event():
    class ToolMock:
        def invalid(self):
            pass

    event_hooks = EventHooks()
    subscriber = ToolMock()
    with pytest.raises(KeyError):
        event_hooks.subscribe("invalid", subscriber)
    assert "invalid" not in event_hooks.event_listeners


def test_event_hooks_unsubscribe():
    event_hooks = EventHooks()
    subscriber = MagicMock()
    event_hooks.subscribe("on_start", subscriber)
    event_hooks.unsubscribe("on_start", subscriber)
    assert subscriber not in event_hooks.event_listeners["on_start"]


def test_event_hooks_notify():
    event_hooks = EventHooks()
    subscriber = MagicMock()
    subscriber.on_start.return_value = "observation"
    event_hooks.subscribe("on_start", subscriber)
    observations = event_hooks.notify("on_start")
    assert observations == {subscriber.name: "observation"}
    subscriber.on_start.assert_called_once()
