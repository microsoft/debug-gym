from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from debug_gym.gym.envs.free_env import FreeEnv
from debug_gym.gym.terminals.local import LocalTerminal
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
        if isinstance(entrypoint, str) and "tree" in entrypoint:
            return True, "/workspace\n"
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


def test_free_env_defaults_to_local_terminal():
    logger = MagicMock()

    env = FreeEnv(image="ubuntu:22.04", logger=logger)

    assert isinstance(env.terminal, LocalTerminal)
    assert env.container_image == "ubuntu:22.04"


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
        init_git=False,
    )

    env.reset()

    assert env.terminal is terminal
    assert terminal.base_image == "ubuntu:22.04"
    assert terminal.working_dir == "/workspace"
    assert terminal.logger is logger
    assert terminal.setup_commands == ["apt update"]


def test_free_env_respects_custom_workspace(tmp_path):
    logger = MagicMock()
    terminal = DummyTerminal(logger=logger)

    env = FreeEnv(
        image="ubuntu:22.04",
        terminal=terminal,
        workspace_dir="/workspace",
        logger=logger,
        init_git=False,
    )

    env.reset()

    assert env.workspace.working_dir == Path("/workspace")
    assert terminal.working_dir == "/workspace"


def test_free_env_reset_allows_dynamic_overrides():
    logger = MagicMock()
    terminal = DummyTerminal(logger=logger, setup_commands=["initial"])

    env = FreeEnv(
        image="ubuntu:22.04",
        terminal=terminal,
        setup_commands=["initial"],
        workspace_dir="/workspace",
        logger=logger,
        init_git=True,
    )

    env.reset(
        options={
            "image": "ubuntu:24.04",
            "workspace_dir": "/new",
            "setup_commands": ["echo ready"],
            "instructions": "Inspect carefully.",
            "init_git": False,
        }
    )

    assert env.container_image == "ubuntu:24.04"
    assert env.instructions == "Inspect carefully."
    assert env.init_git is False
    assert env._workspace_dir == "/new"
    assert terminal.working_dir == "/new"
    assert terminal.setup_commands == ["echo ready"]
    assert terminal.base_image == "ubuntu:24.04"
    assert terminal.closed is True
