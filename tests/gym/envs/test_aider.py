from unittest.mock import patch

import pytest

from debug_gym.gym.envs import AiderBenchmarkEnv
from debug_gym.gym.terminals.docker import DockerTerminal
from debug_gym.gym.terminals.local import LocalTerminal
from debug_gym.gym.tools.tool import ToolCall
from debug_gym.gym.tools.toolbox import Toolbox


@pytest.fixture(scope="session")
def setup_aider_repo(tmp_path_factory):
    """Set up a minimal Aider repository for testing,
    avoiding cloning Aider repo for each test."""
    aider_path = tmp_path_factory.mktemp("aider_repo")
    # aider_path.mkdir(parents=True, exist_ok=True)
    repo_path = aider_path / "exercises" / "practice" / "clock"
    repo_path.mkdir(parents=True, exist_ok=True)
    (repo_path / "clock.py").write_text(
        "def current_time():  # returns a message with the current time\n"
        "    return 'It is 10:00 a.m.'"
    )
    (repo_path / "clock_test.py").write_text(
        "import clock\n"
        "\n"
        "def test_clock():\n"
        "    assert clock.current_time() == 'It is 12:00 a.m.'"
    )
    (repo_path / ".docs").mkdir(parents=True, exist_ok=True)
    (repo_path / ".docs" / "instructions.md").write_text("What time is it?")
    # Patch the REPO_PATH in AiderBenchmarkEnv
    AiderBenchmarkEnv.REPO_PATH = aider_path
    return aider_path


@pytest.fixture
def env(setup_aider_repo):
    terminal = LocalTerminal()
    dataset = AiderBenchmarkEnv.load_dataset()
    task_data = dataset["clock"]
    env = AiderBenchmarkEnv(task_data=task_data, terminal=terminal)
    env.reset()
    return env


def test_files(env):
    assert env.workspace.has_file("clock_test.py")
    assert env.workspace.has_file("clock.py")


def test_steps(env):
    eval_tool = Toolbox.get_tool("eval")
    env.add_tool(eval_tool)
    eval_call = ToolCall(id="eval_id", name="eval", arguments={})
    infos = env.step(eval_call)
    assert infos.step_observation.source == "eval"
    assert "clock_test.py F" in infos.eval_observation.observation
    assert "1 failed" in infos.eval_observation.observation
    assert infos.score == 0
    edit_tool = Toolbox.get_tool("edit")
    env.add_tool(edit_tool)
    infos = env.step(
        ToolCall(
            id="edit_id",
            name="edit",
            arguments={
                "path": "clock.py",
                "start": 2,
                "new_code": "    return 'It is 12:00 a.m.'",
            },
        )
    )
    assert infos.step_observation.source == "edit"
    assert infos.step_observation.observation.startswith(
        "The file `clock.py` has been updated successfully."
    )
    assert not any(obs.source == "eval" for obs in infos.all_observations)
    assert "1 failed" in infos.eval_observation.observation
    assert infos.score == 0

    infos = env.step(eval_call)
    assert infos.step_observation.source == "eval"
    assert "clock_test.py ." in infos.eval_observation.observation
    assert "1 passed" in infos.eval_observation.observation
    assert infos.score == 1


def test_instructions(env):
    assert env.instructions == "What time is it?"


@patch("debug_gym.gym.envs.aider.build_docker_image")
def test_build_docker_image(mock_build_docker_image):
    AiderBenchmarkEnv.load_dataset()
    mock_build_docker_image.assert_called_once()


@pytest.if_docker_running
def test_reset_with_docker_terminal(setup_aider_repo):
    dataset = AiderBenchmarkEnv.load_dataset()
    task_data = dataset["clock"]
    env = AiderBenchmarkEnv(task_data=task_data)
    env.add_tool(Toolbox.get_tool("eval"))
    assert isinstance(env.terminal, DockerTerminal)

    infos = env.reset(options={"task_name": "clock"})
    assert env.instructions == infos.step_observation.observation
    assert "1 failed" in infos.eval_observation.observation
    assert infos.max_score == 1
    assert infos.score == 0
    assert not infos.terminated
    assert not infos.resolved
