from pathlib import Path
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

from debug_gym.gym.entities import EvalOutput, Event, Observation
from debug_gym.gym.envs.env import EnvInfo, EventHooks, RepoEnv, TooledEnv
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


def test_patch(env):
    env.reset()

    # Change the content of a file
    file1 = env.working_dir / "file1.txt"
    with open(file1, "w") as f:
        f.write("Hello, World!")

    result = env.patch
    expected = (
        f"diff --git a/file1.txt b/file1.txt\n"
        "index e69de29..b45ef6f 100644\n"
        f"--- a/file1.txt\n"
        f"+++ b/file1.txt\n"
        "@@ -0,0 +1 @@\n"
        "+Hello, World!\n"
        "\\ No newline at end of file\n"
    )
    assert result == expected


@patch.object(RepoEnv, "get_triggered_tools")
@patch.object(RepoEnv, "get_tool")
@patch.object(RepoEnv, "has_tool", return_value=False)
@patch.object(RepoEnv, "eval")
def test_step(
    mock_eval, mock_has_tool, mock_get_tool, mock_get_triggered_tools, tmp_path
):
    mock_pdb_tool = MagicMock()
    observation = Observation("pdb", "PDB tool used")
    mock_pdb_tool.return_value = observation
    mock_pdb_tool.rewrite_success = True
    mock_pdb_tool.current_frame_file = "file.py"
    mock_get_tool.return_value = None

    env = RepoEnv(path=tmp_path)
    env.reset()
    env.last_eval = EvalOutput(success=False, output="1 failed, 0 passed")
    tool_call = ToolCall(id="123", name="pdb", arguments={"command": "b 10"})
    mock_get_triggered_tools.return_value = None, [mock_pdb_tool, {"command": "b 10"}]
    infos = env.step(
        tool_call,
        "let me set a breakpoint at line 10",
        "some reasoning",
    )

    mock_get_triggered_tools.assert_called_once_with(tool_call)
    mock_pdb_tool.assert_called_once_with(env, command="b 10")
    assert infos.step_observation == observation
    assert infos.score == 0
    assert not infos.done
    assert isinstance(infos, EnvInfo)


def test_reset(tmp_path):
    (tmp_path / "test.py").write_text("def test_1():\n  assert False\n")
    (tmp_path / ".debugignore").write_text("__pycache__/\n.git/\n.pytest_cache/\n")

    env = RepoEnv(path=tmp_path, entrypoint="pytest test.py")
    infos = env.reset()

    assert env.current_breakpoints_state == {}
    assert env.rewrite_counter == 0
    assert "FAILED test.py::test_1 - assert False" in env.last_eval.output
    assert infos == EnvInfo(
        step_observation=Observation(source="env", observation=env.last_eval.output),
        all_observations=[Observation(source="env", observation=env.last_eval.output)],
        eval_observation=Observation(source="env", observation=env.last_eval.output),
        dir_tree=(
            "Listing files in the current working directory. (read-only) indicates read-only files. Max depth: 1.\n"
            f"{env.working_dir}/\n"
            "|-- test.py"
        ),
        current_breakpoints="No breakpoints are set.",
        action_reasoning=None,
        action_content=None,
        action_tool_call=None,
        instructions="",
        score=0,
        max_score=1,
        done=False,
        rewrite_counter=0,
        tools=[],
    )


def test_rewrite_counter(env):
    env_info = env.reset()
    assert env.rewrite_counter == 0
    rewrite_tool = Toolbox.get_tool("rewrite")
    env.add_tool(rewrite_tool)

    rewrite_call = ToolCall(id="rewrite_id", name="rewrite", arguments={})
    env_info = env.step(rewrite_call, "let me rewrite the code")
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
    env_info = env.step(rewrite_call, "let me rewrite the file1.txt")
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


def test_eval_success(tmp_path):
    working_dir = str(tmp_path)
    # create a dummy file
    with open(tmp_path / "file.py", "w") as f:
        f.write("print('Hello, World!')")
    env = RepoEnv(path=working_dir, entrypoint="python file.py")
    env.reset()
    output = env.eval()
    assert output == EvalOutput(success=True, output="Hello, World!")


def test_eval_timeout(tmp_path):
    working_dir = str(tmp_path)
    # runs for longer than the timeout
    with open(tmp_path / "file.py", "w") as f:
        f.write("import time; time.sleep(5)")
    env = RepoEnv(path=working_dir, entrypoint="python file.py", run_timeout=1)
    env.reset()
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


def test_has_breakpoint_true_and_false(tmp_path):
    env = RepoEnv(path=tmp_path)
    env.reset()
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
    env.reset()
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
