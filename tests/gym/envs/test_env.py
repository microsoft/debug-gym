from pathlib import Path
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

from debug_gym.gym.entities import EvalOutput, Event, Observation
from debug_gym.gym.envs.env import EnvInfo, EventHooks, RepoEnv, TooledEnv
from debug_gym.gym.terminal import Terminal


@pytest.fixture
def env_mock():
    env = RepoEnv()
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
    tool2 = MagicMock()
    tool2.name = "tool2"
    env_mock.add_tool(tool1)
    env_mock.add_tool(tool2)
    _, triggered_tool = env_mock.get_triggered_tools("```tool1 arg1 arg2```")
    assert triggered_tool == [tool1, "arg1 arg2"]
    _, triggered_tool = env_mock.get_triggered_tools("```tool2```")
    assert triggered_tool == [tool2, ""]
    # Test with invalid action
    error, triggered_tool = env_mock.get_triggered_tools("```tool3```")
    assert error == "Unregistered tool: tool3"
    assert triggered_tool is None


def test_tool_names(env_mock):
    tool1 = MagicMock()
    tool1.name = "tool1"
    tool2 = MagicMock()
    tool2.name = "tool2"
    env_mock.add_tool(tool1)
    env_mock.add_tool(tool2)
    assert env_mock.tool_names == "```tool1```, ```tool2```"


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
    tool2 = MagicMock()
    tool2.name = "tool2"
    tool2.instructions = "instructions2"

    env = RepoEnv()
    env.add_tool(tool1)
    env.add_tool(tool2)

    expected_instructions = {
        "Available tools to solve the problem": {
            "tool1": "instructions1",
            "tool2": "instructions2",
        },
        "Available commands": "```tool1```, ```tool2```",
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


@patch("debug_gym.gym.utils.show_line_number")
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
@patch.object(RepoEnv, "eval")
@patch.object(RepoEnv, "display_files")
@patch.object(RepoEnv, "current_code_with_line_number")
def test_step(
    mock_current_code_with_line_number,
    mock_display_files,
    mock_eval,
    mock_has_tool,
    mock_get_tool,
    mock_get_triggered_tools,
):
    mock_pdb_tool = MagicMock()
    observation = Observation("pdb", "PDB tool used")
    mock_pdb_tool.return_value = observation
    mock_pdb_tool.rewrite_success = True
    mock_pdb_tool.current_frame_file = "file.py"
    mock_pdb_tool.pdb_obs = "PDB started"
    mock_get_tool.return_value = None
    mock_display_files.return_value = "file list"
    mock_current_code_with_line_number.return_value = "code with line numbers"

    env = RepoEnv(path=".")
    env.last_eval = EvalOutput(success=False, output="1 failed, 0 passed")
    mock_get_triggered_tools.return_value = [mock_pdb_tool]
    mock_get_triggered_tools.return_value = None, [mock_pdb_tool, "b 10"]

    infos = env.step("```pdb b 10```")

    mock_get_triggered_tools.assert_called_once_with("```pdb b 10```")
    mock_pdb_tool.assert_called_once_with("b 10")
    assert infos.step_observation == observation
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


@patch.object(Terminal, "run", return_value=(False, "1 failed, 0 passed"))
@patch.object(RepoEnv, "get_tool")
def test_reset(
    mock_get_tool,
    mock_eval,
    env,
):
    mock_pdb_tool = MagicMock()
    mock_pdb_tool.start_pseudo_terminal.return_value = None
    mock_pdb_tool.pdb_obs = "PDB started"
    mock_get_tool.return_value = mock_pdb_tool
    infos = env.reset()

    mock_eval.assert_called_once()
    assert env.current_file is None
    assert env.current_file_content is None
    assert env.current_breakpoints_state == {}
    assert env.rewrite_counter == 0
    assert infos == EnvInfo(
        step_observation=Observation(source="env", observation="1 failed, 0 passed"),
        all_observations=[Observation(source="env", observation="1 failed, 0 passed")],
        eval_observation=Observation(source="env", observation="1 failed, 0 passed"),
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


def test_eval_success(tmp_path):
    working_dir = str(tmp_path)
    # create a dummy file
    with open(tmp_path / "file.py", "w") as f:
        f.write("print('Hello, World!')")
    env = RepoEnv(path=working_dir, entrypoint="python file.py")
    output = env.eval()
    assert output == EvalOutput(success=True, output="Hello, World!")


def test_eval_timeout(tmp_path):
    working_dir = str(tmp_path)
    # runs for longer than the timeout
    with open(tmp_path / "file.py", "w") as f:
        f.write("import time; time.sleep(5)")
    env = RepoEnv(path=working_dir, entrypoint="python file.py", run_timeout=1)
    output = env.eval()
    assert output == EvalOutput(success=False, output="Timeout expired.")


def test_event_hooks_initialization():
    event_hooks = EventHooks()
    assert set(event_hooks.event_listeners.keys()) == set(Event)
    for e in Event:
        assert event_hooks.event_listeners[e] == []


def test_event_hooks_subscribe():
    class ToolMock:
        def on_env_start(self):
            pass

    event_hooks = EventHooks()
    subscriber = ToolMock()
    event_hooks.subscribe(Event.ENV_START, subscriber)
    assert subscriber in event_hooks.event_listeners[Event.ENV_START]


def test_event_hooks_subscribe_invalid_subscriber():
    class InvalidToolMock:
        pass

    event_hooks = EventHooks()
    subscriber = InvalidToolMock()
    with pytest.raises(ValueError, match="Tool does not implement method on_env_start"):
        event_hooks.subscribe(Event.ENV_START, subscriber)
    assert subscriber not in event_hooks.event_listeners[Event.ENV_START]


def test_event_hooks_subscribe_invalid_event():
    class ToolMock:
        def invalid(self):
            pass

    event_hooks = EventHooks()
    subscriber = ToolMock()
    with pytest.raises(ValueError, match="Unknown event type: invalid"):
        event_hooks.subscribe("invalid", subscriber)
    assert "invalid" not in event_hooks.event_listeners


def test_event_hooks_unsubscribe():
    event_hooks = EventHooks()
    subscriber = MagicMock()
    assert subscriber not in event_hooks.event_listeners[Event.ENV_START]
    event_hooks.subscribe(Event.ENV_START, subscriber)
    assert subscriber in event_hooks.event_listeners[Event.ENV_START]
    event_hooks.unsubscribe(Event.ENV_START, subscriber)
    assert subscriber not in event_hooks.event_listeners[Event.ENV_START]


def test_event_hooks_notify():
    event_hooks = EventHooks()
    subscriber = MagicMock()
    an_observation = Observation("mock", "observation")
    subscriber.on_env_start.return_value = an_observation
    event_hooks.subscribe(Event.ENV_START, subscriber)
    observations = event_hooks.notify(Event.ENV_START)
    assert observations == [an_observation]
    subscriber.on_env_start.assert_called_once()


def test_current_breakpoints_no_breakpoints():
    env = RepoEnv()
    env.current_breakpoints_state = {}
    result = env.current_breakpoints()
    assert result == "No breakpoints are set."


def test_current_breakpoints_with_breakpoints(tmp_path):
    env = RepoEnv()
    env.current_breakpoints_state = {
        "file1.py|||10": "b file1.py:10",
        "file1.py|||20": "b file1.py:20",
        "file1.py|||30": "b file1.py:30",
        "file2.py|||15": "b file2.py:15",
    }
    result = env.current_breakpoints()
    expected_result = (
        "line 10 in file1.py\n"
        "line 20 in file1.py\n"
        "line 30 in file1.py\n"
        "line 15 in file2.py"
    )
    assert result == expected_result


def test_queue_and_process_events():
    env = TooledEnv()
    obs1 = Observation("tool1", "obs1")
    obs2 = Observation("tool2", "obs2")

    # Queue some test events
    env.queue_event(Event.ENV_START, "source1", arg1="val1")
    env.queue_event(Event.ENV_RESET, "source2", arg2="val2")
    assert len(env.event_queue) == 2

    # Patch the notify method to return some observations
    with patch.object(EventHooks, "notify", return_value=[obs1, obs2]) as mock:
        observations = env.process_events()

    # Verify events were processed
    assert observations == [obs1, obs2, obs1, obs2]
    assert env.all_observations == [obs1, obs2, obs1, obs2]
    assert env.event_queue == []

    # Verify notify was called with correct args
    expected_calls = [
        call(event=Event.ENV_START, source="source1", arg1="val1"),
        call(event=Event.ENV_RESET, source="source2", arg2="val2"),
    ]
    mock.assert_has_calls(expected_calls)
