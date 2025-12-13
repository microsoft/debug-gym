import pytest

from debug_gym.gym.entities import Observation
from debug_gym.gym.envs.env import RepoEnv
from debug_gym.gym.terminals.terminal import Terminal, UnrecoverableTerminalError
from debug_gym.gym.tools.bash import BashTool
from debug_gym.gym.tools.tool import ToolCall


def _ensure_list(entrypoint):
    if isinstance(entrypoint, (list, tuple)):
        return list(entrypoint)
    return [entrypoint]


class AlwaysFailTerminal(Terminal):
    """Terminal that raises an unrecoverable error for every operation."""

    def __init__(self, message="Pod is not running. Cannot run commands.", **kwargs):
        super().__init__(**kwargs)
        self.message = message

    def prepare_command(self, entrypoint):
        return _ensure_list(entrypoint)

    @property
    def default_shell_command(self) -> str:
        return "fails"

    def new_shell_session(self):
        raise UnrecoverableTerminalError(self.message)

    def run(self, entrypoint, timeout=None, raises=False, strip_output=True):
        raise UnrecoverableTerminalError(self.message)

    def copy_content(self, src, target=None):
        return None


class MinimalEnv(RepoEnv):
    def __init__(self, terminal):
        super().__init__(task_data={"id": "dummy"}, terminal=terminal)

    @property
    def instructions(self) -> str:
        return "Do not panic"

    @property
    def task_name(self) -> str:
        return "minimal-task"

    def setup_task(self) -> None:  # pragma: no cover - nothing to setup
        return None

    def setup_workspace(self) -> None:  # pragma: no cover - not used for this test
        return None

    def setup_terminal(self) -> None:  # pragma: no cover - nothing to setup
        return None


@pytest.fixture
def fatal_env():
    env = MinimalEnv(terminal=AlwaysFailTerminal())
    env.add_tool(BashTool())
    env.reset(options={"reset_runtime": False})
    try:
        yield env
    finally:
        env.close()


def test_env_terminates_after_unrecoverable_terminal_error(fatal_env):
    tool_call = ToolCall(id="bash-1", name="bash", arguments={"command": "ls"})

    info = fatal_env.step(tool_call)

    assert info.terminated is True
    assert fatal_env.terminated is True
    assert info.action_tool_call == tool_call
    assert info.step_observation.source == "env"
    observation = info.step_observation.observation
    assert "Fatal terminal error detected" in observation
    assert "Pod is not running" in observation
    assert isinstance(fatal_env.step_observation, Observation)
    assert fatal_env.resolved is False
