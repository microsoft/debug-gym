import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
from froggy.envs import SWEBenchEnv
from froggy.utils import cleanup_pytest_output, extract_max_score_from_pytest_output, extract_reward_from_pytest_output

@pytest.fixture
@patch('subprocess.run')
@patch('os.path.exists', return_value=False)
@patch('datasets.load_dataset')
def swe_env(mock_load_hf_dataset, mock_exists, mock_run):
    # Mock the dataset
    mock_dataset = MagicMock()
    mock_dataset.sort.return_value = [
        {
            "instance_id": "test_task",
            "problem_statement": "Test problem statement",
            "repo": "test_org/test_repo",
            "base_commit": "test_commit",
            "test_patch": "test_patch",
            "FAIL_TO_PASS": "['test_fail_to_pass']",
            "PASS_TO_PASS": "['test_pass_to_pass']",
        }
    ]
    mock_load_hf_dataset.return_value = {"test": mock_dataset}

    # Initialize the SWEBenchEnv
    env = SWEBenchEnv()
    return env

def test_instructions(swe_env):
    swe_env.ds_row = {"problem_statement": "Test problem statement"}
    expected_instructions = {
        "Problem description": "Test problem statement",
        "Available tools to solve the problem": swe_env.tool_instructions,
        "Available commands": swe_env.actions_str,
    }
    assert swe_env.instructions == expected_instructions

@patch('froggy.envs.RepoEnv.reset', return_value=("obs", {"obs": "obs", "max_score": 10, "score": 5, "last_run_obs": "Raw output"}))
@patch('froggy.envs.SWEBenchEnv.setup_workspace')
@patch('froggy.envs.SWEBenchEnv.make_froggyignore')
@patch('froggy.utils.cleanup_pytest_output', return_value="Cleaned output")
@patch('froggy.utils.extract_max_score_from_pytest_output', return_value=10)
@patch('froggy.utils.extract_reward_from_pytest_output', return_value=5)
@patch('datasets.load_dataset')
@patch('subprocess.run')
def test_reset(mock_run, mock_load_dataset, mock_extract_reward, mock_extract_max_score, mock_cleanup, mock_make_froggyignore, mock_setup_workspace, mock_env, swe_env):
    options = {"task_name": "test_task"}
    obs, infos = swe_env.reset(options=options)
    assert infos["last_run_obs"] == "Cleaned output"
    assert infos["max_score"] == 10
    assert infos["score"] == 5

@patch('froggy.envs.RepoEnv.step', return_value=("obs", 5, True, {"last_run_obs": "Raw output"}))
@patch('froggy.utils.cleanup_pytest_output', return_value="Cleaned output")
@patch('froggy.utils.extract_reward_from_pytest_output', return_value=5)
def test_step(mock_extract_reward, mock_cleanup, repo_env, swe_env):
    obs, score, done, infos = swe_env.step("action")
    assert infos["last_run_obs"] == "Cleaned output"
    assert infos["score"] == 5

@patch('subprocess.run')
@patch('os.path.exists', return_value=False)
def test_clone_repo(mock_exists, mock_run, swe_env):
    repo_address = "test_org/test_repo"
    local_repo_path = swe_env.clone_repo(repo_address)
    assert mock_run.called
    assert local_repo_path == swe_env.SWE_BENCH_REPO_PATHS / "test_repo"

@patch('builtins.open', new_callable=mock_open)
@patch('os.listdir', return_value=[".gitignore"])
def test_make_froggyignore(mock_listdir, mock_open, swe_env):
    swe_env.make_froggyignore(Path("test_directory"))
    mock_open.assert_called_with(Path("test_directory/.froggyignore"), "w")