import pytest

from debug_gym.gym.entities import Observation
from debug_gym.gym.envs.env import Event, RepoEnv
from debug_gym.gym.tools.tool import EnvironmentTool, Record, track_history
from debug_gym.gym.tools.toolbox import Toolbox


class FakeTool(EnvironmentTool):
    def use(self, action):
        return Observation("FakeTool", action)


def test_register_valid_environment():
    tool = FakeTool()
    env = RepoEnv()
    tool.register(env)
    assert tool.environment == env


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

    assert hasattr(tool, "history")
    assert isinstance(tool.history, list)
    assert len(tool.history) == 0

    tool(action="first")
    assert len(tool.history) == 1
    assert tool.history[0] == Record(
        args=(),
        kwargs={"action": "first"},
        observation=Observation("FakeTool", "first"),
    )

    tool(action="second")
    assert len(tool.history) == 2
    assert tool.history[1] == Record(
        args=(),
        kwargs={"action": "second"},
        observation=Observation("FakeTool", "second"),
    )
