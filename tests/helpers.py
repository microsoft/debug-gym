from pathlib import Path

from debug_gym.gym.envs.local import LocalEnv as ProductionLocalEnv
from debug_gym.gym.envs.mini_nightmare import (
    DOCKER_MINI_NIGHTMARE_IMAGE_NAME,
    build_docker_image,
)
from debug_gym.gym.terminals.docker import DockerTerminal
from debug_gym.gym.terminals.terminal import Terminal


def docker_test_terminal() -> DockerTerminal:
    build_docker_image()
    return DockerTerminal(base_image=DOCKER_MINI_NIGHTMARE_IMAGE_NAME)


class LocalEnv(ProductionLocalEnv):
    """Local-path environment using a Docker terminal."""

    def __init__(self, path: str | Path, terminal: Terminal | None = None, **kwargs):
        super().__init__(
            path=str(path),
            terminal=terminal or docker_test_terminal(),
            **kwargs,
        )
