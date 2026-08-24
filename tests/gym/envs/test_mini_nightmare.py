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
    dataset = MiniNightmareEnv.load_dataset(build_image=False)
    task_data = dataset["config"]
    env = MiniNightmareEnv(task_data=task_data, terminal=terminal)
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
