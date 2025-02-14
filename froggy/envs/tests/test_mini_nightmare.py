import json
import os
import shutil
from unittest.mock import MagicMock, mock_open, patch

import pytest

from froggy.envs.env import EnvInfo
from froggy.envs.mini_nightmare import MiniNightmareEnv


@pytest.fixture
def env_info():
    return EnvInfo(
        obs="obs",
        max_score=10,
        score=5,
        last_run_obs="Raw output",
        observations=[],
        dir_tree="dir_tree",
        current_code_with_line_number="current_code_with_line_number",
        current_breakpoints="current_breakpoints",
        action="action",
        instructions={},
        done=False,
        rewrite_counter=0,
        tools={},
    )


@pytest.fixture
@patch("os.path.exists", return_value=True)
@patch("tempfile.TemporaryDirectory")
@patch(
    "builtins.open",
    new_callable=mock_open,
    read_data='{"data": [{"id": "test_task", "original_code_paths": ["path/to/file.py"], "buggy_code_list": ["print(\\"buggy code\\")"]}]}',
)
def mini_nightmare_env(mock_open, mock_tempdir, mock_exists):
    # Mock the temporary directory
    mock_tempdir.return_value.name = "/tmp/MiniNightmareEnv-tempdir"

    # Initialize the MiniNightmareEnv
    env = MiniNightmareEnv()
    return env


def test_instructions(mini_nightmare_env):
    mini_nightmare_env.current_sample = {"instructions": "Test instructions"}
    expected_instructions = {
        "Problem description": "Test instructions",
        "Available tools to solve the problem": mini_nightmare_env.tool_instructions,
        "Available commands": mini_nightmare_env.actions_str,
    }
    assert mini_nightmare_env.instructions == expected_instructions


@patch("froggy.envs.RepoEnv.reset")
@patch("froggy.envs.RepoEnv.current_code_with_line_number", return_value="Current code")
@patch("froggy.envs.MiniNightmareEnv.setup_workspace")
@patch("froggy.envs.MiniNightmareEnv.load_current_file")
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
    mini_nightmare_env,
    env_info,
):
    repo_env.return_value = env_info
    mini_nightmare_env.dataset = {
        "test_task": {
            "base_directory": "test_directory",
            "instructions": "Test instructions",
            "filename": "test_task.py",
        }
    }
    options = {"task_name": "test_task"}
    infos = mini_nightmare_env.reset(options=options)
    assert infos.instructions["Problem description"] == "Test instructions"
    assert infos.last_run_obs == "Cleaned output"
    assert infos.max_score == 10
    assert infos.score == 5
