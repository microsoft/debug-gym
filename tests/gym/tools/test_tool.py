from pathlib import Path

import pytest

from debug_gym.gym.entities import Observation
from debug_gym.gym.envs.env import Event
from debug_gym.gym.envs.local import LocalEnv
from debug_gym.gym.tools.tool import EnvironmentTool, Record
from debug_gym.gym.tools.toolbox import Toolbox


class FakeTool(EnvironmentTool):
    name: str = "FakeTool"

    def use(self, env, action):
        return Observation("FakeTool", action)


@pytest.fixture
def env(tmp_path):
    env = LocalEnv(path=tmp_path)
    return env


def test_register_valid_environment(env):
    tool = FakeTool()
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


def test_auto_subscribe(monkeypatch, env):

    @Toolbox.register()
    class ToolWithHandler(FakeTool):
        def on_env_reset(self, **kwargs):
            return "Handler for Event.ENV_RESET"

    tool = ToolWithHandler()

    env.add_tool(tool)

    assert tool in env.event_hooks.event_listeners[Event.ENV_RESET]
    assert len(env.event_hooks.event_listeners[Event.ENV_RESET]) == 1
    for channel in env.event_hooks.event_listeners:
        if channel != Event.ENV_RESET:
            assert tool not in env.event_hooks.event_listeners[channel]


def test_track_history(env):
    tool = FakeTool()

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


def test_unknown_args(env):
    tool = FakeTool()
    obs = tool(env, unknown_arg="unknown_value")
    assert obs == Observation(
        "FakeTool", "FakeTool.use() got an unexpected keyword argument 'unknown_arg'"
    )


def test_unregister(env):
    tool = FakeTool()
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


def test_unregister_with_multiple_handlers(env):
    class ToolWithMultipleHandlers(FakeTool):
        def on_env_reset(self, environment, **kwargs):
            return "Handler for Event.ENV_RESET"

        def on_env_step(self, environment, **kwargs):
            return "Handler for Event.ENV_STEP"

    tool = ToolWithMultipleHandlers()
    tool.register(env)

    # Verify tool is registered for both events
    assert tool in env.event_hooks.event_listeners[Event.ENV_RESET]
    assert tool in env.event_hooks.event_listeners[Event.ENV_STEP]

    # Unregister tool
    tool.unregister(env)

    # Verify tool is no longer listening to any events
    assert tool not in env.event_hooks.event_listeners[Event.ENV_RESET]
    assert tool not in env.event_hooks.event_listeners[Event.ENV_STEP]
