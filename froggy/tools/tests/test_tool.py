import pytest

from froggy.envs.env import Event, RepoEnv
from froggy.tools.tool import EnvironmentTool
from froggy.tools.toolbox import Toolbox


class FakeTool(EnvironmentTool):
    def use(self, action, environment):
        pass

    def reset(self):
        pass


def test_register_valid_environment():
    tool = FakeTool()
    env = RepoEnv()
    tool.register(env)
    assert tool.environment == env


def test_register_invalid_environment():
    tool = FakeTool()
    with pytest.raises(ValueError):
        tool.register(object())


def test_is_triggered():
    tool = FakeTool()
    tool.action = "pdb"
    assert tool.is_triggered("pdb breakpoints")
    assert not tool.is_triggered("otherstuff")


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
