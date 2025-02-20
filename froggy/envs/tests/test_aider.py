from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest

from froggy.entities import Observation
from froggy.envs import AiderBenchmarkEnv
from froggy.envs.env import EnvInfo


@pytest.fixture
def env_info():
    return EnvInfo(
        step_observation=Observation("tool", "obs"),
        all_triggered_observations=[],
        eval_observation="eval_observation",
        dir_tree="dir_tree",
        current_code_with_line_number="current_code_with_line_number",
        current_breakpoints="current_breakpoints",
        action="action",
        instructions={},
        score=5,
        max_score=10,
        done=False,
        rewrite_counter=0,
        tools={},
    )


@pytest.fixture
@patch("subprocess.run")
@patch("os.path.exists", return_value=False)
@patch("pathlib.Path.iterdir")
@patch("pathlib.Path.read_text", return_value="Test instructions")
@patch("os.listdir", return_value=[".gitignore"])
@patch("builtins.open", new_callable=mock_open)
def aider_env(
    mock_open, mock_listdir, mock_read_text, mock_iterdir, mock_exists, mock_run
):
    # Mock the directories
    mock_dir = MagicMock()
    mock_dir.is_dir.return_value = True
    mock_dir.name = "test_task"
    mock_iterdir.return_value = [mock_dir]

    # Initialize the AiderBenchmarkEnv
    env = AiderBenchmarkEnv()
    return env


def test_instructions(aider_env):
    aider_env.current_sample = {"instructions": "Test instructions"}
    expected_instructions = {
        "Problem description": "Test instructions",
        "Available tools to solve the problem": aider_env.tool_instructions,
        "Available commands": aider_env.actions_str,
    }
    assert aider_env.instructions == expected_instructions


# TODO: remove patches if not needed
@patch("froggy.envs.RepoEnv.reset")
@patch("froggy.envs.RepoEnv.current_code_with_line_number", return_value="Current code")
@patch("froggy.envs.AiderBenchmarkEnv.setup_workspace")
@patch("froggy.envs.AiderBenchmarkEnv.load_current_file")
@patch("froggy.utils.cleanup_pytest_output", return_value="Cleaned output")
@patch("froggy.utils.extract_max_score_from_pytest_output", return_value=10)
@patch("froggy.utils.extract_reward_from_pytest_output", return_value=5)
@patch("datasets.load_dataset")
@patch("subprocess.run")
def test_reset(
    mock_run,
    mock_load_dataset,
    mock_extract_reward,
    mock_extract_max_score,
    mock_cleanup,
    mock_load_current_file,
    mock_setup_workspace,
    mock_line_number,
    repo_env,
    aider_env,
    env_info,
):
    repo_env.return_value = env_info
    aider_env.dataset = {
        "test_task": {
            "base_directory": "test_directory",
            "instructions": "Test instructions",
            "filename": "test_task.py",
        }
    }
    options = {"task_name": "test_task"}
    infos = aider_env.reset(options=options)
    assert infos.instructions["Problem description"] == "Test instructions"
    assert infos.step_observation == Observation("tool", "obs")
    assert infos.max_score == 10
    assert infos.score == 5


# TODO: Add proper test, mocking repoenv.step doesn't test anything
@patch("froggy.envs.RepoEnv.step")
def test_step(mock_step, aider_env, env_info):
    mock_step.return_value = env_info
    infos = aider_env.step("action")
    assert infos.step_observation == Observation("tool", "obs")
    assert infos.score == 5


@patch("subprocess.run")
@patch("os.path.exists", return_value=False)
@patch("os.listdir", return_value=[".gitignore"])
def test_load_dataset(mock_listdir, mock_exists, mock_run, aider_env):
    aider_env.load_dataset()
    assert mock_run.called
