from pathlib import Path

from debug_gym.gym.envs.local import LocalEnv
from debug_gym.gym.envs.mini_nightmare import (
    DOCKER_MINI_NIGHTMARE_IMAGE_NAME,
    build_docker_image,
)
from debug_gym.gym.terminals.docker import DockerTerminal
from debug_gym.gym.terminals.terminal import Terminal


def docker_test_terminal() -> DockerTerminal:
    build_docker_image()
    return DockerTerminal(base_image=DOCKER_MINI_NIGHTMARE_IMAGE_NAME)


def docker_local_env(
    path: str | Path, terminal: Terminal | None = None, **kwargs
) -> LocalEnv:
    return LocalEnv(
        path=str(path),
        terminal=terminal or docker_test_terminal(),
        **kwargs,
    )
