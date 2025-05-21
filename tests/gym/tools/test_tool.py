import pytest

from debug_gym.gym.entities import Observation
from debug_gym.gym.envs.env import Event, RepoEnv
from debug_gym.gym.tools.tool import EnvironmentTool, Record
from debug_gym.gym.tools.toolbox import Toolbox


class FakeTool(EnvironmentTool):
    def use(self, env, action):
        return Observation("FakeTool", action)


def test_register_valid_environment():
    tool = FakeTool()
    env = RepoEnv()
    tool.register(env)
    # every tool listen to ENV_RESET event to track history
    assert tool in env.event_hooks.event_listeners[Event.ENV_RESET]


def test_register_invalid_environment():
    tool = FakeTool()
    with pytest.raises(ValueError):
        tool.register(object())


def test_abstract_class():
    with pytest.raises(TypeError):
        EnvironmentTool()


def test_abstract_methods():
    class CompletelyFakeTool(EnvironmentTool):
        pass

    with pytest.raises(
        TypeError,
        match=(
            "Can't instantiate abstract class CompletelyFakeTool "
            "without an implementation for abstract method*"
        ),
    ):
        tool = CompletelyFakeTool()


def test_auto_subscribe(monkeypatch):

    @Toolbox.register()
    class ToolWithHandler(FakeTool):
        def on_env_reset(self, **kwargs):
            return "Handler for Event.ENV_RESET"

    tool = ToolWithHandler()

    env = RepoEnv()
    env.add_tool(tool)

    assert tool in env.event_hooks.event_listeners[Event.ENV_RESET]
    assert len(env.event_hooks.event_listeners[Event.ENV_RESET]) == 1
    for channel in env.event_hooks.event_listeners:
        if channel != Event.ENV_RESET:
            assert tool not in env.event_hooks.event_listeners[channel]


def test_track_history():
    tool = FakeTool()
    env = RepoEnv()

    assert hasattr(tool, "history")
    assert isinstance(tool.history, list)
    assert len(tool.history) == 0

    tool(env, action="first")
    assert len(tool.history) == 1
    assert tool.history[0] == Record(
        args=(),
        kwargs={"action": "first"},
        observation=Observation("FakeTool", "first"),
    )

    tool(env, action="second")
    assert len(tool.history) == 2
    assert tool.history[1] == Record(
        args=(),
        kwargs={"action": "second"},
        observation=Observation("FakeTool", "second"),
    )


def test_unregister():
    tool = FakeTool()
    env = RepoEnv()
    tool.register(env)

    # Verify tool is registered
    assert tool in env.event_hooks.event_listeners[Event.ENV_RESET]

    # Unregister tool
    tool.unregister(env)

    # Verify tool is no longer listening to events
    assert tool not in env.event_hooks.event_listeners[Event.ENV_RESET]


def test_unregister_invalid_environment():
    tool = FakeTool()
    with pytest.raises(ValueError, match="The environment must be a RepoEnv instance."):
        tool.unregister(object())


def test_unregister_with_multiple_handlers():
    class ToolWithMultipleHandlers(FakeTool):
        def on_env_reset(self, environment, **kwargs):
            return "Handler for Event.ENV_RESET"

        def on_env_step(self, environment, **kwargs):
            return "Handler for Event.ENV_STEP"

    tool = ToolWithMultipleHandlers()
    env = RepoEnv()
    tool.register(env)

    # Verify tool is registered for both events
    assert tool in env.event_hooks.event_listeners[Event.ENV_RESET]
    assert tool in env.event_hooks.event_listeners[Event.ENV_STEP]

    # Unregister tool
    tool.unregister(env)

    # Verify tool is no longer listening to any events
    assert tool not in env.event_hooks.event_listeners[Event.ENV_RESET]
    assert tool not in env.event_hooks.event_listeners[Event.ENV_STEP]


def test_tool_call():
    tool = FakeTool()
    env = RepoEnv()

    # Call the tool
    observation = tool(env, action="test_action")
    # Check the observation
    assert isinstance(observation, Observation)
    assert observation.source == "FakeTool"
    assert observation.observation == "test_action"

    # Call the tool with an invalid argument
    observation = tool(env, invalid_action="test_action")
    # Check the observation
    assert isinstance(observation, Observation)
    assert observation.source == "FakeTool"
    # invalid argument
    assert (
        observation.observation
        == "FakeTool.use() got an unexpected keyword argument 'invalid_action'"
    )
