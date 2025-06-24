from pathlib import Path
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

from debug_gym.gym.entities import EvalOutput, Event, Observation
from debug_gym.gym.envs.env import EnvInfo, EventHooks, RepoEnv, TooledEnv
from debug_gym.gym.terminal import Terminal
from debug_gym.gym.tools.tool import ToolCall
from debug_gym.gym.tools.toolbox import Toolbox


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
    assert tool in env_mock.tools
    assert env_mock.get_tool("tool1") == tool


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


def test_remove_tool(env_mock):
    tool = MagicMock()
    tool.name = "tool1"
    env_mock.add_tool(tool)
    removed = env_mock.remove_tool("tool1")
    assert removed == tool
    assert tool not in env_mock.tools
    assert not env_mock.has_tool("tool1")
    with pytest.raises(KeyError):
        assert env_mock.get_tool("tool1") is None
    # Test removing a non-existing tool
    with pytest.raises(ValueError):
        env_mock.remove_tool("tool2")


def test_get_triggered_tools(env_mock):
    tool1 = MagicMock()
    tool1.name = "tool1"
    tool2 = MagicMock()
    tool2.name = "tool2"
    env_mock.add_tool(tool1)
    env_mock.add_tool(tool2)
    _, triggered_tool = env_mock.get_triggered_tools(
        ToolCall(id="123", name="tool1", arguments={"arg1": "abc", "arg2": 4})
    )
    assert triggered_tool == [tool1, {"arg1": "abc", "arg2": 4}]
    _, triggered_tool = env_mock.get_triggered_tools(
        ToolCall(id="234", name="tool2", arguments={})
    )
    assert triggered_tool == [tool2, {}]
    # Test with invalid action
    error, triggered_tool = env_mock.get_triggered_tools(
        ToolCall(id="345", name="tool3", arguments={})
    )
    assert error == "Unregistered tool: tool3"
    assert triggered_tool is None


def test_tool_names(env_mock):
    tool1 = MagicMock()
    tool1.name = "tool1"
    tool2 = MagicMock()
    tool2.name = "tool2"
    env_mock.add_tool(tool1)
    env_mock.add_tool(tool2)
    assert env_mock.tool_names == "tool1, tool2"


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
    repo_env = RepoEnv(run_timeout=10, dir_tree_depth=2)
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
    repo_env = RepoEnv(run_timeout=10, dir_tree_depth=2)
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


def test_env_tools():
    tool1 = MagicMock()
    tool1.name = "tool1"
    tool1.description = "instructions1"
    tool1.arguments = {
        "command 1": {
            "type": ["string"],
            "description": "command 1 description",
        },
    }
    tool2 = MagicMock()
    tool2.name = "tool2"
    tool2.description = "instructions2"
    tool2.arguments = {
        "command 2": {
            "type": ["string"],
            "description": "command 2 description",
        },
    }

    env = RepoEnv()
    env.add_tool(tool1)
    env.add_tool(tool2)

    assert env.tools == [tool1, tool2]


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
        "Listing files in the current working directory. (read-only) indicates read-only files. Max depth: 2.\n"
        f"{env.working_dir}/\n"
        "|-- file1.txt\n"
        "|-- file2.txt\n"
        "|-- subdir/\n"
        "  |-- subfile1.txt"
    )


def test_display_files_read_only(env):
    read_only_path = env.working_dir / "read-only-file.txt"
    read_only_path.touch()
    env.is_editable = lambda x: x != read_only_path
    result = env.display_files()
    assert result == (
        "Listing files in the current working directory. (read-only) indicates read-only files. Max depth: 2.\n"
        f"{env.working_dir}/\n"
        "|-- file1.txt\n"
        "|-- file2.txt\n"
        "|-- read-only-file.txt (read-only)\n"
        "|-- subdir/\n"
        "  |-- subfile1.txt"
    )


@patch.object(RepoEnv, "get_triggered_tools")
@patch.object(RepoEnv, "get_tool")
@patch.object(RepoEnv, "has_tool", return_value=False)
@patch.object(RepoEnv, "eval")
@patch.object(RepoEnv, "display_files")
def test_step(
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
    mock_get_tool.return_value = None
    mock_display_files.return_value = "file list"

    env = RepoEnv(path=".")
    env.last_eval = EvalOutput(success=False, output="1 failed, 0 passed")
    mock_get_triggered_tools.return_value = None, [mock_pdb_tool, {"command": "b 10"}]
    infos = env.step({"id": "123", "name": "pdb", "arguments": {"command": "b 10"}})

    mock_get_triggered_tools.assert_called_once_with(
        {"id": "123", "name": "pdb", "arguments": {"command": "b 10"}}
    )
    mock_pdb_tool.assert_called_once_with(env, command="b 10")
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
    mock_get_tool.return_value = mock_pdb_tool
    infos = env.reset()

    mock_eval.assert_called_once()
    assert env.current_breakpoints_state == {}
    assert env.rewrite_counter == 0
    assert infos == EnvInfo(
        step_observation=Observation(source="env", observation="1 failed, 0 passed"),
        all_observations=[Observation(source="env", observation="1 failed, 0 passed")],
        eval_observation=Observation(source="env", observation="1 failed, 0 passed"),
        dir_tree=f"""Listing files in the current working directory. (read-only) indicates read-only files. Max depth: 2.
{env.tempdir.name}/
|-- file1.txt
|-- file2.txt
|-- subdir/
  |-- subfile1.txt""",
        current_breakpoints="No breakpoints are set.",
        action=None,
        instructions="",
        score=0,
        max_score=1,
        done=False,
        rewrite_counter=0,
        tools=[],
    )


def test_rewrite_counter(env):
    # env.calculate_score = lambda x: 0
    env_info = env.reset()
    assert env.rewrite_counter == 0
    rewrite_tool = Toolbox.get_tool("rewrite")
    env.add_tool(rewrite_tool)

    rewrite_call = ToolCall(id="rewrite_id", name="rewrite", arguments={})
    env_info = env.step(rewrite_call)
    assert env.rewrite_counter == 1
    assert env_info.rewrite_counter == 1
    rewrite_obs = Observation(
        source="rewrite",
        observation="Rewrite failed. Error message:\nFile path is None.\n",
    )
    assert env_info.step_observation == rewrite_obs
    assert env_info.all_observations == [rewrite_obs]

    rewrite_call = ToolCall(
        id="rewrite_id",
        name="rewrite",
        arguments={
            "path": "file1.txt",
            "new_code": "print('Hello')",
        },
    )
    env_info = env.step(rewrite_call)
    assert env.rewrite_counter == 2
    assert env_info.rewrite_counter == 2
    rewrite_obs = Observation(
        source="rewrite",
        observation="The file `file1.txt` has been updated successfully.\n\nDiff:\n\n--- original\n+++ current\n@@ -0,0 +1 @@\n+print('Hello')",
    )
    assert env_info.step_observation == rewrite_obs
    assert env_info.all_observations == [rewrite_obs]
    with open(env.working_dir / "file1.txt", "r") as f:
        assert f.read() == "print('Hello')"


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
    with pytest.raises(
        ValueError, match=f"Tool already subscribed to event: {Event.ENV_START}"
    ):
        event_hooks.subscribe(Event.ENV_START, subscriber)


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
    env = None
    observations = event_hooks.notify(env, Event.ENV_START)
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
        call(environment=env, event=Event.ENV_START, source="source1", arg1="val1"),
        call(environment=env, event=Event.ENV_RESET, source="source2", arg2="val2"),
    ]
    mock.assert_has_calls(expected_calls)


@pytest.mark.parametrize("debugignore", ["", ".?*"])
def test_resolve_path(tmp_path, debugignore):
    (tmp_path / ".debugignore").write_text(debugignore)
    env = RepoEnv(path=tmp_path)
    abs_path = (env.working_dir / "file.txt").resolve()
    (abs_path).touch()
    # env.working_dir itself
    path_from_env = env.resolve_path(str(env.working_dir), raises=True)
    assert path_from_env == env.working_dir.resolve()
    # relative path
    path_from_env = env.resolve_path("file.txt")
    assert path_from_env == abs_path
    # relative path with ./
    path_from_env = env.resolve_path("./file.txt")
    assert path_from_env == abs_path
    # absolute path
    path_from_env = env.resolve_path(str(abs_path))
    assert path_from_env == abs_path
    # relative path with Path object
    path_from_env = env.resolve_path(Path("file.txt"))
    assert path_from_env == abs_path
    # absolute path with Path object
    path_from_env = env.resolve_path(abs_path)
    assert path_from_env == abs_path
    # return an absolute path regardless of existence
    non_existent_path = env.resolve_path("non_existent_file.txt")
    assert non_existent_path == (env.working_dir / "non_existent_file.txt").resolve()
    # non-existent absolute path
    non_existent_path = env.resolve_path("/tmp/non_existent_file.txt").resolve()
    assert non_existent_path == Path("/tmp/non_existent_file.txt").resolve()


def test_resolve_path_raises(tmp_path):
    env = RepoEnv(path=tmp_path)
    # Non-existent file with raises=True
    with pytest.raises(FileNotFoundError):
        env.resolve_path("non_existent_file.txt", raises=True)
    # Non-existent absolute path with raises=True
    with pytest.raises(FileNotFoundError):
        env.resolve_path("/tmp/non_existent_file.txt", raises=True)
    with pytest.raises(FileNotFoundError):
        env.resolve_path("..", raises=True)
    # Invalid path type
    with pytest.raises(TypeError):
        env.resolve_path(123, raises=True)
    with pytest.raises(TypeError):
        env.resolve_path(None, raises=True)


def test_resolve_path_do_not_raise_working_dir(tmp_path):
    # Do not raise for working directory even if the ignore patterns match
    (tmp_path / ".debugignore").write_text(".*")
    env = RepoEnv(path=tmp_path)
    assert env.resolve_path(env.working_dir, raises=True) == env.working_dir


def test_setup_file_filters_basic(tmp_path):
    # Setup a fake repo structure
    env = RepoEnv(path=tmp_path)
    subdir = env.working_dir / "subdir"
    subdir.mkdir()
    files = [
        env.working_dir / "file1.txt",
        env.working_dir / "file2.txt",
        env.working_dir / "ignored.txt",
        env.working_dir / "readonly.txt",
        env.working_dir / "subdir/file3.txt",
    ]
    [f.touch() for f in files]
    files.append(subdir)
    env.setup_file_filters()
    # All files should be indexed
    assert all(env.has_file(f) for f in files)
    # All files should be editable if no readonly patterns
    assert all(env.is_editable(f) for f in files)


def test_setup_file_filters_with_ignore_patterns(tmp_path):
    (tmp_path / "file1.txt").touch()
    (tmp_path / "file2.txt").touch()
    (tmp_path / "ignoreme.txt").touch()
    (tmp_path / "subdir").mkdir()
    (tmp_path / "subdir" / "file3.txt").touch()

    env = RepoEnv(path=tmp_path)
    # Ignore files matching "ignoreme.txt"
    env.setup_file_filters(ignore_patterns=["ignoreme.txt"])
    assert not env.has_file("ignoreme.txt")
    assert env.has_file("file1.txt")
    assert env.has_file("file2.txt")
    assert env.has_file("subdir/file3.txt")


def test_setup_file_filters_with_readonly_patterns(tmp_path):
    (tmp_path / "file1.txt").touch()
    (tmp_path / "readonly.txt").touch()

    env = RepoEnv(path=tmp_path)
    # Mark "readonly.txt" as read-only
    env.setup_file_filters(readonly_patterns=["readonly.txt"])
    assert env.is_editable("file1.txt")
    assert not env.is_editable("readonly.txt")


def test_setup_file_filters_with_debugignore_and_debugreadonly(tmp_path):
    (tmp_path / "file1.txt").touch()
    (tmp_path / "file2.txt").touch()
    (tmp_path / "ignoreme.txt").touch()
    (tmp_path / "readonly.txt").touch()
    # Write .debugignore and .debugreadonly
    (tmp_path / ".debugignore").write_text("ignoreme.txt\n")
    (tmp_path / ".debugreadonly").write_text("readonly.txt\n")

    env = RepoEnv(path=tmp_path)
    env.setup_file_filters()
    assert not env.has_file("ignoreme.txt")
    assert env.has_file("file1.txt")
    assert env.has_file("file2.txt")
    assert env.has_file("readonly.txt")
    # Check that readonly.txt is not editable
    assert not env.is_editable("readonly.txt")
    # Check that file1.txt and file2.txt are editable
    assert env.is_editable("file1.txt")
    assert env.is_editable("file2.txt")
    with pytest.raises(FileNotFoundError):
        env.is_editable("ignoreme.txt")


def test_setup_file_filters_combined_patterns(tmp_path):
    (tmp_path / "file1.txt").touch()
    (tmp_path / "file2.txt").touch()
    (tmp_path / "ignoreme.txt").touch()
    (tmp_path / "readonly.txt").touch()
    (tmp_path / ".debugignore").write_text("ignoreme.txt\n")
    (tmp_path / ".debugreadonly").write_text("readonly.txt\n")

    env = RepoEnv(path=tmp_path)
    # Also ignore file2.txt and mark file1.txt as readonly via patterns
    env.setup_file_filters(
        ignore_patterns=["file2.txt"],
        readonly_patterns=["file1.txt"],
    )
    assert not env.has_file("ignoreme.txt")
    assert not env.has_file("file2.txt")
    assert env.has_file("file1.txt")
    assert env.has_file("readonly.txt")
    # Both file1.txt and readonly.txt should be readonly
    assert not env.is_editable("file1.txt")
    assert not env.is_editable("readonly.txt")
    with pytest.raises(FileNotFoundError):
        assert not env.is_editable("ignoreme.txt")
    with pytest.raises(FileNotFoundError):
        assert not env.is_editable("file2.txt")


def test_read_file_reads_existing_file(tmp_path):
    env = RepoEnv(path=tmp_path)
    file_path = env.working_dir / "test.txt"
    file_content = "Hello, DebugGym!"
    file_path.write_text(file_content)
    # Read file using relative path
    result = env.read_file(str(env.working_dir / "test.txt"))
    assert result == file_content
    # Read file using just the filename (should also work)
    result = env.read_file("test.txt")
    assert result == file_content


def test_read_file_raises_for_nonexistent_file(tmp_path):
    env = RepoEnv(path=tmp_path)
    (env.working_dir / "test.txt").touch()
    # relative path that does not exist
    with pytest.raises(FileNotFoundError):
        env.read_file("does_not_exist.txt")
    # absolute path matching a file in the working_dir
    with pytest.raises(FileNotFoundError):
        env.read_file("/test.txt")


def test_has_breakpoint_true_and_false(tmp_path):
    env = RepoEnv(path=tmp_path)
    file_path = env.working_dir / "test.py"
    file_path.write_text("print('hello')")
    line_number = 10
    key = f"{file_path}|||{line_number}"
    env.current_breakpoints_state = {key: "b test.py:10"}
    assert env.has_breakpoint(str(file_path), line_number) is True
    assert env.has_breakpoint(str(file_path), 20) is False
    other_file = env.working_dir / "other.py"
    assert env.has_breakpoint(str(other_file), line_number) is False


def test_has_breakpoint_relative_path(tmp_path):
    env = RepoEnv(path=tmp_path)
    file_path = env.working_dir / "foo.py"
    file_path.write_text("print('foo')")
    line_number = 5
    key = f"{file_path}|||{line_number}"
    env.current_breakpoints_state = {key: "b foo.py:5"}
    # Should work with relative path
    assert env.has_breakpoint("foo.py", line_number) is True
    # Should return False for wrong line
    assert env.has_breakpoint("foo.py", 6) is False
    # Should return False for non-existent file
    assert env.has_breakpoint("bar.py", line_number) is False
