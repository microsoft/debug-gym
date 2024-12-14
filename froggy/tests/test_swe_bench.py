import os
import subprocess
from unittest import mock
from unittest.mock import MagicMock, patch

import pytest

from froggy.envs import SWEBenchEnv
from froggy.envs.env import RepoEnv
from froggy.envs.swe_bench import SWEBenchEnv
from froggy.terminal import DockerTerminal

if_docker_running = pytest.mark.skipif(
    not subprocess.check_output(["docker", "ps"]),
    reason="Docker not running",
)


def test_load_dataset(tmp_path):
    working_dir = str(tmp_path)
    swe_env = SWEBenchEnv(path=working_dir)
    assert swe_env.dataset_id == "princeton-nlp/SWE-bench_Verified"
    # check if the dataset contains features that SWEBenchEnv expects
    assert list(swe_env.ds.features.keys()) == [
        "repo",
        "instance_id",
        "base_commit",
        "patch",  # not required
        "test_patch",
        "problem_statement",
        "hints_text",  # not required
        "created_at",  # not required
        "version",  # not required
        "FAIL_TO_PASS",
        "PASS_TO_PASS",
        "environment_setup_commit",  # not required
    ]


def test_clone_repo(tmp_path):
    working_dir = str(tmp_path)
    swe_env = SWEBenchEnv(path=working_dir)
    task_name = "astropy__astropy-14096"
    row = swe_env.dataset[task_name]
    repo_address = row["repo"]
    local_repo_path = swe_env.clone_repo(repo_address)
    repo_content = os.listdir(local_repo_path)
    assert "astropy" in repo_content


def test_make_froggyignore(tmp_path):
    working_dir = str(tmp_path)
    swe_env = SWEBenchEnv(path=working_dir)
    task_name = "astropy__astropy-14096"
    row = swe_env.dataset[task_name]
    repo_address = row["repo"]
    local_repo_path = swe_env.clone_repo(repo_address)
    swe_env.make_froggyignore(local_repo_path, include_gitignore=False)
    with open(local_repo_path / ".froggyignore", "r") as f:
        froggyignore = f.read()
    assert froggyignore == "*/tests/\n.froggyignore"


def test_mak_froggyignore_include_gitignore(tmp_path):
    working_dir = str(tmp_path)
    swe_env = SWEBenchEnv(path=working_dir)
    task_name = "astropy__astropy-14096"
    row = swe_env.dataset[task_name]
    repo_address = row["repo"]
    local_repo_path = swe_env.clone_repo(repo_address)
    swe_env.make_froggyignore(local_repo_path)
    with open(local_repo_path / ".froggyignore", "r") as f:
        froggyignore = f.read()
    assert froggyignore.startswith("*/tests/\n.froggyignore")
    assert len(froggyignore.split("\n")) > 2


@pytest.fixture
@patch("subprocess.run")
@patch("os.path.exists", return_value=False)
@patch("datasets.load_dataset")
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


@patch(
    "froggy.envs.RepoEnv.step",
    return_value=("obs", 5, True, {"last_run_obs": "Raw output"}),
)
@patch("froggy.utils.cleanup_pytest_output", return_value="Cleaned output")
@patch("froggy.utils.extract_reward_from_pytest_output", return_value=5)
def test_step(mock_extract_reward, mock_cleanup, repo_env, swe_env):
    obs, score, done, infos = swe_env.step("action")
    assert infos["last_run_obs"] == "Cleaned output"
    assert infos["score"] == 5


def test_reset(tmp_path):
    working_dir = str(tmp_path)
    swe_env = SWEBenchEnv(path=working_dir)
    task_name = "astropy__astropy-14096"
    obs = "Some observation"
    last_run_obs = "collected 10 items. 5 passed, 5 failed"
    info = {"obs": obs, "last_run_obs": last_run_obs}
    with (
        mock.patch.object(SWEBenchEnv, "setup_local_repo"),
        mock.patch.object(SWEBenchEnv, "setup_terminal"),
        mock.patch.object(RepoEnv, "reset", return_value=(obs, info)),
    ):
        reset_obs, reset_infos = swe_env.reset(options={"task_name": task_name})

    assert reset_obs == obs
    assert reset_infos == {
        "obs": obs,
        "last_run_obs": last_run_obs,
        "max_score": 10,
        "score": 5,
    }


def test_repo_name(tmp_path):
    working_dir = str(tmp_path)
    swe_env = SWEBenchEnv(path=working_dir)
    repo = "test_org/test_repo"
    expected_repo_name = "test_org__test_repo"
    assert swe_env.repo_name(repo) == expected_repo_name

    repo_with_spaces = "test org/test repo"
    expected_repo_name_with_spaces = "test--org__test--repo"
    assert swe_env.repo_name(repo_with_spaces) == expected_repo_name_with_spaces

    repo_with_apostrophe = "test'org/test'repo"
    expected_repo_name_with_apostrophe = "testorg__testrepo"
    assert swe_env.repo_name(repo_with_apostrophe) == expected_repo_name_with_apostrophe


@if_docker_running
def test_run_command_with_raise(tmp_path):
    working_dir = str(tmp_path)
    terminal = DockerTerminal(working_dir=working_dir)
    swe_env = SWEBenchEnv(path=working_dir, terminal=terminal)
    status, output = swe_env.run_command_with_raise("echo 'Hello World'")
    assert output == "Hello World"
    with pytest.raises(
        ValueError, match="Failed to run command: cat /non_existent_file"
    ):
        swe_env.run_command_with_raise("cat /non_existent_file")
    # add sudo if apt-get in command
    status, output = swe_env.run_command_with_raise("apt-get update")
    assert status
    # don't break if sudo is already there
    status, output = swe_env.run_command_with_raise("sudo apt-get update")
    assert status


@if_docker_running
def test_install_conda(tmp_path):
    working_dir = str(tmp_path)
    terminal = DockerTerminal(working_dir=working_dir)
    swe_env = SWEBenchEnv(path=working_dir, terminal=terminal)
    info_cmd = "source ~/miniconda3/bin/activate || true && conda info"
    with pytest.raises(ValueError, match=".*conda: command not found"):
        _, output = swe_env.run_command_with_raise(info_cmd)
        assert not status

    status, output = swe_env.install_conda()

    assert status
    assert "installation finished." in output
    status, output = swe_env.run_command_with_raise(info_cmd)
    assert status
    assert "active environment : base" in output in output


@pytest.fixture
def install_configs_mock():
    install_configs = {
        "python": "3.12.8",
        "test_cmd": "pytest --help",
        "pre_install": ["apt-get help", "apt-get install -y vim"],
        "eval_commands": ["export TEST_VAR='Test Var'", "echo $TEST_VAR"],
        "install": "python3 -m pip install pytest==8.3.3",
        "post_install": ["echo 'Test file' > test.txt", "cat test.txt"],
        "packages": "pytest==8.3.3 requests==2.32.3",
        "pip_packages": ["pytest"],
        "no_use_env": False,
    }
    return install_configs


@if_docker_running
def test_run_pre_intall(tmp_path, install_configs_mock):
    working_dir = str(tmp_path)
    terminal = DockerTerminal(working_dir=working_dir)
    swe_env = SWEBenchEnv(path=working_dir, terminal=terminal)
    swe_env.install_configs = install_configs_mock
    swe_env.run_pre_install()
    _, output = swe_env.run_command_with_raise("vim --help")
    assert "VIM - Vi IMproved" in output


@if_docker_running
def test_run_install(tmp_path, install_configs_mock):
    working_dir = str(tmp_path)
    terminal = DockerTerminal(working_dir=working_dir, base_image="python:3.12-slim")
    swe_env = SWEBenchEnv(path=working_dir, terminal=terminal)
    swe_env.install_configs = install_configs_mock
    swe_env.run_install()
    _, output = swe_env.run_command_with_raise("python -m pytest --version")
    assert "pytest 8.3.3" in output


@if_docker_running
def test_run_post_install(tmp_path, install_configs_mock):
    working_dir = str(tmp_path)
    terminal = DockerTerminal(working_dir=working_dir)
    swe_env = SWEBenchEnv(path=working_dir, terminal=terminal)
    swe_env.install_configs = install_configs_mock
    swe_env.run_post_install()
    _, output = swe_env.run_command_with_raise("cat test.txt")
    assert output == "Test file"


@if_docker_running
def test_create_conda_environment(tmp_path, install_configs_mock):
    working_dir = str(tmp_path)
    terminal = DockerTerminal(working_dir=working_dir)
    swe_env = SWEBenchEnv(path=working_dir, terminal=terminal)
    swe_env.install_configs = install_configs_mock
    swe_env.repo = "test/repo"
    swe_env.version = "2.0"
    swe_env.create_conda_env()
    status, output = swe_env.run_command_with_raise("echo $CONDA_DEFAULT_ENV")
    assert status
    assert output == "test__repo__2.0"
    status, output = swe_env.run_command_with_raise("pip freeze")
    assert status
    assert "pytest" in output
    assert "requests" in output


@if_docker_running
def test_create_conda_environment_from_requirements(tmp_path, install_configs_mock):
    install_configs_mock["packages"] = "requirements.txt"
    working_dir = str(tmp_path)
    terminal = DockerTerminal(working_dir=working_dir)
    swe_env = SWEBenchEnv(path=working_dir, terminal=terminal)
    swe_env.install_configs = install_configs_mock
    swe_env.repo = "test/repo"
    swe_env.version = "2.0"
    swe_env.ds_row = {}
    requirements = "pytest==8.3.3\nrequests==2.32.3"
    with patch("froggy.envs.swe_bench.get_requirements", return_value=requirements):
        swe_env.create_conda_env()
    status, output = swe_env.run_command_with_raise("echo $CONDA_DEFAULT_ENV")
    assert status
    assert output == "test__repo__2.0"
    status, output = swe_env.run_command_with_raise("pip freeze")
    assert status
    assert "pytest" in output
    assert "requests" in output
