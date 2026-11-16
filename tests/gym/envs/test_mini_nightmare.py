from unittest.mock import Mock, patch

import docker
import pytest

from debug_gym.constants import DEBUG_GYM_CACHE_DIR
from debug_gym.gym.envs.mini_nightmare import (
    DOCKER_MINI_NIGHTMARE_IMAGE_NAME,
    MiniNightmareEnv,
    build_docker_image,
)
from debug_gym.gym.terminals.docker import DockerTerminal
from debug_gym.gym.tools.toolbox import Toolbox


@pytest.fixture
def mini_nightmare_env():
    dataset = MiniNightmareEnv.load_dataset()
    task_data = dataset["config"]
    env = MiniNightmareEnv(task_data=task_data)
    env.add_tool(Toolbox.get_tool("eval"))
    return env


def test_load_dataset(mini_nightmare_env):
    dataset = MiniNightmareEnv.load_dataset(build_image=False)
    subproblems = list(dataset.keys())[::2]
    subset = MiniNightmareEnv.load_dataset(problems=subproblems, build_image=False)
    assert list(subset.keys()) == subproblems


@patch("debug_gym.gym.envs.mini_nightmare.build_docker_image")
def test_build_docker_image(mock_build_docker_image):
    MiniNightmareEnv.load_dataset()
    mock_build_docker_image.assert_called_once()


def test_build_docker_image_serializes_image_creation():
    lock_held = False

    class Lock:
        def __init__(self, path):
            assert path == str(DEBUG_GYM_CACHE_DIR / "mini_nightmare_image.lock")

        def __enter__(self):
            nonlocal lock_held
            lock_held = True

        def __exit__(self, *args):
            nonlocal lock_held
            lock_held = False

    images = Mock()

    def image_not_found(image_name):
        assert lock_held
        assert image_name == DOCKER_MINI_NIGHTMARE_IMAGE_NAME
        raise docker.errors.ImageNotFound("image not found")

    def build_image(**kwargs):
        assert lock_held
        assert kwargs["tag"] == DOCKER_MINI_NIGHTMARE_IMAGE_NAME
        return Mock(), []

    images.get.side_effect = image_not_found
    images.build.side_effect = build_image
    with (
        patch("docker.from_env", return_value=Mock(images=images)),
        patch("debug_gym.gym.envs.mini_nightmare.FileLock", Lock),
    ):
        build_docker_image()

    images.get.assert_called_once_with(DOCKER_MINI_NIGHTMARE_IMAGE_NAME)
    images.build.assert_called_once()


def test_instructions(mini_nightmare_env):
    expected_instructions = (
        "The program doesn't behave as intended."
        " Investigate the repository, figure out the root cause, then edit the code to fix the issue."
        " Beaware that the bug may not be in the code you initially see."
    )
    assert mini_nightmare_env.instructions == expected_instructions


def test_reset(mini_nightmare_env):
    infos = mini_nightmare_env.reset(options={"task_name": "config"})
    assert mini_nightmare_env.instructions == infos.step_observation.observation
    assert "2 failed" in infos.eval_observation.observation
    assert infos.max_score == 2
    assert infos.score == 0
    assert not infos.terminated
    assert not infos.resolved


@pytest.if_docker_running
def test_reset_with_docker_terminal():
    dataset = MiniNightmareEnv.load_dataset()
    task_data = dataset["config"]
    env = MiniNightmareEnv(task_data=task_data)
    env.add_tool(Toolbox.get_tool("eval"))
    assert isinstance(env.terminal, DockerTerminal)

    infos = env.reset()
    assert env.instructions == infos.step_observation.observation
    assert "2 failed" in infos.eval_observation.observation
    assert infos.max_score == 2
    assert infos.score == 0
    assert not infos.terminated
    assert not infos.resolved
