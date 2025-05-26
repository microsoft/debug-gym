import os
import subprocess
from unittest.mock import patch

import pytest

# Create a simple FileLock implementation
class FileLock:
    def __init__(self, lock_file):
        self.lock_file = lock_file
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

# Import our mock setup first
from tests.gym.envs.mock_imports import SWESmithEnv

from debug_gym.gym.entities import Observation
from debug_gym.gym.terminal import DockerTerminal
from debug_gym.gym.tools.tool import ToolCall
from debug_gym.gym.tools.toolbox import Toolbox

if_docker_running = pytest.mark.skipif(
    not subprocess.check_output(["docker", "ps"]),
    reason="Docker not running",
)


# https://pytest-xdist.readthedocs.io/en/stable/how-to.html#making-session-scoped-fixtures-execute-only-once
@pytest.fixture(scope="session")
def build_swe_smith_env_once(tmp_path_factory, worker_id):
    """Build the SWESmith docker image only once.
    Do not run this fixture directly, use get_swe_smith_env instead.
    """
    # For testing, we'll use SWE-bench data since SWE-smith dataset may not be available yet
    _build_swe_smith_env = lambda: SWESmithEnv(
        dataset_id="princeton-nlp/SWE-bench_Verified",
        instance_ids=["astropy__astropy-14096"]
    )
    if worker_id == "master":
        # Not running with pytest-xdist or we are in the master process
        _build_swe_smith_env()
    else:
        # When running with pytest-xdist, synchronize between workers using a lock
        root_tmp_dir = tmp_path_factory.getbasetemp().parent
        lock_file = root_tmp_dir / "swe_smith_init.lock"
        with FileLock(str(lock_file)):
            # Only the first worker to acquire the lock will initialize the DB
            _build_swe_smith_env()


@pytest.fixture
def get_swe_smith_env(build_swe_smith_env_once):
    """Instantiate a SWESmithEnv instance after building the SWESmith docker image."""

    def _swe_smith_env(working_dir=None, map_host_uid_gid=True, **kwargs):
        # For testing, we'll use SWE-bench data since SWE-smith dataset may not be available yet
        instance_ids = ["astropy__astropy-14096"]
        terminal = DockerTerminal(
            path=working_dir, map_host_uid_gid=map_host_uid_gid, **kwargs
        )
        env = SWESmithEnv(
            dataset_id="princeton-nlp/SWE-bench_Verified",
            instance_ids=instance_ids, 
            terminal=terminal
        )
        return env

    return _swe_smith_env


@if_docker_running
def test_load_dataset(tmp_path, get_swe_smith_env):
    working_dir = str(tmp_path)
    swe_smith_env = get_swe_smith_env(working_dir)
    # We're using SWE-bench dataset for testing
    assert swe_smith_env.dataset_id == "princeton-nlp/SWE-bench_Verified"


@if_docker_running
def test_instructions(get_swe_smith_env):
    swe_smith_env = get_swe_smith_env()
    swe_smith_env.ds_row = {"problem_statement": "Test problem statement"}
    expected_instructions = {"Problem description": "Test problem statement"}
    assert swe_smith_env.instructions == expected_instructions


@pytest.fixture
def install_configs_mock():
    install_configs = {
        "python": "3.10",
        "test_cmd": "pytest --help",
        "pre_install": ["apt-get help", "apt-get install -y vim"],
        "eval_commands": ["export TEST_VAR='Test Var'", "echo $TEST_VAR"],
        "install": "python3 -m pip install pytest==7.3.1",
        "post_install": ["echo 'Test file' > test.txt", "cat test.txt"],
        "packages": "pytest requests",
        "pip_packages": ["pytest"],
        "no_use_env": False,
    }
    return install_configs


@if_docker_running
def test_run_install(tmp_path, install_configs_mock, get_swe_smith_env):
    working_dir = str(tmp_path)
    swe_smith_env = get_swe_smith_env(
        working_dir=working_dir, map_host_uid_gid=False, base_image="python:3.10-slim"
    )
    swe_smith_env.install_configs = install_configs_mock
    swe_smith_env.run_install()
    _, output = swe_smith_env.run_command_with_raise("python -m pytest --version")
    assert "pytest 7.3.1" in output


@if_docker_running
def test_run_post_install(tmp_path, install_configs_mock, get_swe_smith_env):
    working_dir = str(tmp_path)
    swe_smith_env = get_swe_smith_env(working_dir)
    swe_smith_env.install_configs = install_configs_mock
    swe_smith_env.run_post_install()
    _, output = swe_smith_env.run_command_with_raise("cat test.txt")
    assert output == "Test file"


@if_docker_running
def test_run_command_with_raise(tmp_path, get_swe_smith_env):
    working_dir = str(tmp_path)
    swe_smith_env = get_swe_smith_env(working_dir=working_dir, map_host_uid_gid=False)
    # install sudo for testing
    success, output = swe_smith_env.terminal.run(
        ["apt update", "apt install -y sudo", "echo 'Terminal ready'"]
    )
    assert success
    assert output.endswith("Terminal ready")

    status, output = swe_smith_env.run_command_with_raise("echo 'Hello World'")
    assert output == "Hello World"
    with pytest.raises(
        ValueError, match="Failed to run command: cat /non_existent_file"
    ):
        swe_smith_env.run_command_with_raise("cat /non_existent_file")
    # add sudo if apt-get in command
    status, output = swe_smith_env.run_command_with_raise("apt-get update")
    assert status
    # don't break if sudo is already there
    status, output = swe_smith_env.run_command_with_raise("sudo apt-get update")
    assert status