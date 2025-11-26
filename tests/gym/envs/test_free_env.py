from typing import Any
from unittest.mock import MagicMock, patch

from debug_gym.gym.envs.free_env import FreeEnv
from debug_gym.gym.terminals.terminal import Terminal


class DummyTerminal(Terminal):
    """Test helper terminal with minimal behavior for FreeEnv interactions."""

    def __init__(
        self,
        *,
        working_dir: str = "/tmp/test",
        logger: Any | None = None,
        base_image: str | None = None,
        setup_commands: list[str] | None = None,
    ):
        super().__init__(working_dir=working_dir, logger=logger)
        self.base_image = base_image
        self.setup_commands = list(setup_commands or [])
        self.closed = False

    def prepare_command(self, entrypoint):
        return ["/bin/true"]

    def run(self, entrypoint, timeout=None, raises=False, strip_output=True):
        return True, ""

    @property
    def default_shell_command(self):
        return "/bin/true"

    def new_shell_session(self):
        return None

    def copy_content(self, src, target=None):
        return None

    def close(self):
        self.closed = True


def test_free_env_uses_docker_terminal_by_default():
    logger = MagicMock()
    docker_instance = MagicMock(spec=Terminal)

    with patch(
        "debug_gym.gym.envs.free_env.DockerTerminal", return_value=docker_instance
    ) as mock_docker:
        env = FreeEnv(
            image="ubuntu:22.04",
            logger=logger,
            setup_commands=["apt update"],
            terminal_kwargs={"foo": "bar"},
        )

    mock_docker.assert_called_once()
    _, kwargs = mock_docker.call_args
    assert kwargs["base_image"] == "ubuntu:22.04"
    assert kwargs["setup_commands"] == ["apt update"]
    assert kwargs["working_dir"] == "/testbed"
    assert kwargs["logger"] is logger
    assert kwargs["foo"] == "bar"
    assert env.terminal is docker_instance


def test_free_env_configures_existing_terminal():
    logger = MagicMock()
    terminal_logger = MagicMock()
    terminal = DummyTerminal(
        working_dir="/initial",
        logger=terminal_logger,
        base_image="base",
        setup_commands=["existing"],
    )

    env = FreeEnv(
        image="ubuntu:22.04",
        terminal=terminal,
        setup_commands=["apt update"],
        workspace_dir="/workspace",
        logger=logger,
    )

    assert env.terminal is terminal
    assert terminal.base_image == "ubuntu:22.04"
    assert terminal.working_dir == "/workspace"
    assert terminal.logger is logger
    assert terminal.setup_commands == ["existing", "apt update"]


def test_free_env_update_terminal_restarts_remote_session():
    logger = MagicMock()
    terminal = DummyTerminal(logger=logger)

    env = FreeEnv(image="ubuntu:22.04", terminal=terminal, logger=logger)
    env.container_image = "ubuntu:24.04"
    env.container_workdir = "/new"
    env._setup_commands = ["echo ready"]

    env._update_terminal(restart=True)

    assert terminal.closed is True
    assert terminal.base_image == "ubuntu:24.04"
    assert terminal.setup_commands == ["echo ready"]
    assert terminal.working_dir == "/new"
    assert env.workspace.terminal is terminal
