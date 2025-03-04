from unittest.mock import mock_open, patch

import pytest

from debug_gym.gym.entities import Observation
from debug_gym.gym.envs.env import EnvInfo
from debug_gym.gym.envs.mini_nightmare import MiniNightmareEnv
from debug_gym.gym.terminal import Terminal


@pytest.fixture
def env_info():
    return EnvInfo(
        step_observation=Observation(source="env", observation="obs"),
        all_observations=[],
        eval_observation=Observation(source="env", observation="eval_observation"),
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
@patch("os.path.exists", return_value=True)
@patch("tempfile.TemporaryDirectory")
@patch(
    "builtins.open",
    new_callable=mock_open,
    read_data='{"data": [{"id": "test_task", "original_code_paths": ["path/to/file.py"], "buggy_code_list": ["print(\\"buggy code\\")"]}]}',
)
def mini_nightmare_env(mock_open, mock_tempdir, mock_exists, tmp_path):
    # Mock the temporary directory
    nightmare_dir = tmp_path / "tmp" / "MiniNightmareEnv-tempdir"
    mock_tempdir.return_value.name = nightmare_dir

    # Initialize the MiniNightmareEnv
    env = MiniNightmareEnv()
    env.working_dir = nightmare_dir
    return env


def test_instructions(mini_nightmare_env):
    mini_nightmare_env.current_sample = {"instructions": "Test instructions"}
    expected_instructions = {
        "Problem description": "Test instructions",
        "Available tools to solve the problem": mini_nightmare_env.tool_instructions,
        "Available commands": mini_nightmare_env.tool_names,
    }
    assert mini_nightmare_env.instructions == expected_instructions


@patch(
    "debug_gym.gym.envs.RepoEnv.current_code_with_line_number",
    return_value="Current code",
)
@patch("debug_gym.gym.envs.MiniNightmareEnv.setup_workspace")
@patch("debug_gym.gym.envs.MiniNightmareEnv.load_current_file")
@patch.object(
    Terminal,
    "run",
    return_value=(False, "collected 10 items, 5 failed, 5 passed ..."),
)
@patch("datasets.load_dataset")
@patch("subprocess.run")
def test_reset(
    mock_run,
    mock_load_dataset,
    mock_terminal_run,
    mock_load_current_file,
    mock_setup_workspace,
    mock_line_number,
    mini_nightmare_env,
):
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
    assert infos.step_observation == Observation(
        source="env",
        observation="collected 10 items, 5 failed, 5 passed ...",
    )
    assert infos.max_score == 10
    assert infos.score == 5
    assert infos.done == False
