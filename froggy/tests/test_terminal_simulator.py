import pytest
from unittest.mock import patch, MagicMock, mock_open
import json
import os
import shutil
from froggy.envs.terminal_simulator import TerminalSimulatorEnv

@pytest.fixture
@patch('os.path.exists', return_value=True)
@patch('tempfile.TemporaryDirectory')
@patch('builtins.open', new_callable=mock_open, read_data='{"data": [{"id": "test_task", "original_code_paths": ["path/to/file.py"], "buggy_code_list": ["print(\\"buggy code\\")"]}]}')
def terminal_simulator_env(mock_open, mock_tempdir, mock_exists):
    # Mock the temporary directory
    mock_tempdir.return_value.name = "/tmp/TerminalSimulatorEnv-tempdir"

    # Initialize the TerminalSimulatorEnv
    env = TerminalSimulatorEnv(bug_free_code_path="/path/to/bug_free_code", buggy_code_path="/path/to/buggy_code.json")
    return env

def test_instructions(terminal_simulator_env):
    terminal_simulator_env.current_sample = {"instructions": "Test instructions"}
    expected_instructions = {
        "Problem description": "Test instructions",
        "Available tools to solve the problem": terminal_simulator_env.tool_instructions,
        "Available commands": terminal_simulator_env.actions_str,
    }
    assert terminal_simulator_env.instructions == expected_instructions

@patch('shutil.copytree')
@patch('builtins.open', new_callable=mock_open, read_data='{"data": [{"id": "test_task", "original_code_paths": ["path/to/file.py"], "buggy_code_list": ["print(\\"buggy code\\")"]}]}')
@patch('os.path.exists', return_value=True)
@patch('froggy.envs.RepoEnv.load_current_file')
# @patch('froggy.envs.RepoEnv.reset', return_value=("obs", {"last_run_obs": "Raw output"}))
def test_reset(mock_loadfile, mock_exists, mock_open, mock_copytree, terminal_simulator_env):
    terminal_simulator_env.dataset = {
        "test_task": {
            "original path": ["path/to/file.py"],
            "new code": ["print('buggy code')"],
            "entry_point": "python -m pytest -sv test.py",
            "instructions": "Test instructions",
            "default_file_name": "code/run_terminal_simulator.py",
        }
    }
    options = {"task_name": "test_task"}
    obs, infos = terminal_simulator_env.reset(options=options)
    assert infos["instructions"]["Problem description"] == "Test instructions"
    assert infos["current_code_with_line_number"] is not None

@patch('os.path.exists', return_value=True)
@patch('builtins.open', new_callable=mock_open, read_data='{"data": [{"id": "test_task", "original_code_paths": ["path/to/file.py"], "buggy_code_list": ["print(\\"buggy code\\")"]}]}')
def test_load_dataset(mock_open, mock_exists, terminal_simulator_env):
    terminal_simulator_env.load_dataset()
    assert "test_task" in terminal_simulator_env.dataset
    assert terminal_simulator_env.dataset["test_task"]["original path"] == ["path/to/file.py"]
    assert terminal_simulator_env.dataset["test_task"]["new code"] == ['print("buggy code")']

@patch('froggy.envs.RepoEnv.current_code_with_line_number', return_value="Current code")
@patch('froggy.envs.RepoEnv.display_files', return_value="list of files")
@patch('froggy.utils.cleanup_pytest_output', return_value="Cleaned output")
@patch('froggy.utils.extract_reward_from_pytest_output', return_value=5)
def test_step(mock_extract_reward, mock_cleanup, mock_displayfiles, mock_currentfile, terminal_simulator_env):
    terminal_simulator_env.current_sample = {"instructions": "Test instructions"}
    terminal_simulator_env.last_run_obs = "obs"
    obs, score, done, infos = terminal_simulator_env.step("action")
    assert infos["last_run_obs"] == "obs"
    assert infos["score"] == 0