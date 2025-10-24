from unittest.mock import patch

import pytest

from debug_gym.gym.envs.mini_nightmare import MiniNightmareEnv
from debug_gym.gym.terminals.docker import DockerTerminal
from debug_gym.gym.terminals.local import LocalTerminal
from debug_gym.gym.tools.toolbox import Toolbox


@pytest.fixture
def mini_nightmare_env():
    # Initialize the MiniNightmareEnv with LocalTerminal
    terminal = LocalTerminal()
    env = MiniNightmareEnv(terminal=terminal)
    env.add_tool(Toolbox.get_tool("eval"))
    return env


def test_load_dataset(mini_nightmare_env):
    dataset = mini_nightmare_env.load_dataset()
    assert mini_nightmare_env.dataset == dataset

    subproblems = list(dataset.keys())[::2]
    subset = mini_nightmare_env.load_dataset(problems=subproblems)
    assert list(subset.keys()) == subproblems


@patch("debug_gym.gym.envs.mini_nightmare.build_docker_image")
def test_build_docker_image(mock_build_docker_image):
    MiniNightmareEnv()
    mock_build_docker_image.assert_called_once()


def test_instructions(mini_nightmare_env):
    expected_instructions = (
        "The program doesn't behave as intended."
        " Investigate the repository, figure out the root cause, then rewrite the code to fix the issue."
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
    env = MiniNightmareEnv()
    env.add_tool(Toolbox.get_tool("eval"))
    assert isinstance(env.terminal, DockerTerminal)

    infos = env.reset(options={"task_name": "config"})
    assert env.instructions == infos.step_observation.observation
    assert "2 failed" in infos.eval_observation.observation
    assert infos.max_score == 2
    assert infos.score == 0
    assert not infos.terminated
    assert not infos.resolved
