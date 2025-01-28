from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest

from froggy.envs import AiderBenchmarkEnv
from froggy.utils import (
    cleanup_pytest_output,
    extract_max_score_from_pytest_output,
    extract_reward_from_pytest_output,
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


@patch(
    "froggy.envs.RepoEnv.reset",
    return_value=(
        "obs",
        {"obs": "obs", "max_score": 10, "score": 5, "last_run_obs": "Raw output"},
    ),
)
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
):
    aider_env.dataset = {
        "test_task": {
            "base_directory": "test_directory",
            "instructions": "Test instructions",
            "filename": "test_task.py",
        }
    }
    options = {"task_name": "test_task"}
    obs, infos = aider_env.reset(options=options)
    assert infos["instructions"]["Problem description"] == "Test instructions"
    assert infos["last_run_obs"] == "Cleaned output"
    assert infos["max_score"] == 10
    assert infos["score"] == 5


@patch(
    "froggy.envs.RepoEnv.step",
    return_value=("obs", 5, True, {"last_run_obs": "Raw output"}),
)
@patch("froggy.utils.cleanup_pytest_output", return_value="Cleaned output")
@patch("froggy.utils.extract_reward_from_pytest_output", return_value=5)
def test_step(mock_extract_reward, mock_cleanup, mock_step, aider_env):
    obs, score, done, infos = aider_env.step("action")
    assert infos["last_run_obs"] == "Cleaned output"
    assert infos["score"] == 5


@patch("subprocess.run")
@patch("os.path.exists", return_value=False)
@patch("os.listdir", return_value=[".gitignore"])
def test_load_dataset(mock_listdir, mock_exists, mock_run, aider_env):
    aider_env.load_dataset()
    assert mock_run.called


@patch("builtins.open", new_callable=mock_open)
@patch("os.listdir", return_value=[".gitignore"])
def test_make_froggyignore(mock_listdir, mock_open, aider_env):
    aider_env.make_froggyignore(Path("test_directory"))
    mock_open.assert_called_with(Path("test_directory/.froggyignore"), "w")
