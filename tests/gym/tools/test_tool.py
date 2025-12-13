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


class FakeToolWithSetup(EnvironmentTool):
    """A fake tool that tracks setup command execution."""

    name: str = "FakeToolWithSetup"
    setup_commands: list[str] = ["echo setup_command_executed"]

    def use(self, env, action):
        return Observation("FakeToolWithSetup", action)


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
        CompletelyFakeTool()


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


class TestSetupCommands:
    """Test cases for tool setup_commands functionality."""

    def test_setup_commands_run_on_reset(self, tmp_path):
        """Test that setup commands run when reset() is called after tool registration."""
        # Create env and add tool before reset
        env = LocalEnv(path=tmp_path)
        tool = FakeToolWithSetup()
        env.add_tool(tool)

        # Create a marker file via setup command to verify it ran
        marker_file = tmp_path / "setup_marker.txt"
        tool.setup_commands = [f"echo 'setup_ran' > {marker_file}"]

        # Reset should trigger on_env_reset which runs setup commands
        env.reset()

        assert marker_file.exists()
        assert "setup_ran" in marker_file.read_text()

    def test_setup_commands_run_immediately_when_added_after_reset(self, tmp_path):
        """Test that setup commands run immediately when tool is added after reset()."""
        # Create env and reset first
        env = LocalEnv(path=tmp_path)
        env.reset()

        # Create a marker file via setup command
        marker_file = tmp_path / "setup_marker_after_reset.txt"

        # Add tool after reset - setup commands should run immediately in register()
        tool = FakeToolWithSetup()
        tool.setup_commands = [f"echo 'setup_ran_after' > {marker_file}"]
        env.add_tool(tool)

        assert marker_file.exists()
        assert "setup_ran_after" in marker_file.read_text()

    def test_setup_commands_run_on_each_reset(self, tmp_path):
        """Test that setup commands run on each reset() call."""
        env = LocalEnv(path=tmp_path)
        tool = FakeToolWithSetup()
        env.add_tool(tool)

        # Use a counter file to track how many times setup ran
        counter_file = tmp_path / "setup_counter.txt"
        tool.setup_commands = [f"echo 'x' >> {counter_file}"]

        # First reset
        env.reset()
        assert counter_file.read_text().count("x") == 1

        # Second reset
        env.reset()
        assert counter_file.read_text().count("x") == 2

    def test_tool_without_setup_commands(self, tmp_path):
        """Test that tools without setup_commands work normally."""
        env = LocalEnv(path=tmp_path)
        tool = FakeTool()  # No setup_commands

<<<<<<< HEAD
        assert tool.setup_commands == []
=======
        assert tool.setup_commands == ()
>>>>>>> origin/main

        env.add_tool(tool)
        env.reset()

        # Tool should work normally
        obs = tool(env, action="test")
        assert obs.observation == "test"

    def test_setup_commands_failure_does_not_raise(self, tmp_path):
        """Test that failing setup commands don't raise exceptions (raises=False)."""
        env = LocalEnv(path=tmp_path)
        tool = FakeToolWithSetup()
        # Command that will fail
        tool.setup_commands = ["exit 1"]
        env.add_tool(tool)

        # Should not raise, even though command fails
        env.reset()

        # Tool should still work
        obs = tool(env, action="test")
        assert obs.observation == "test"
