import os
import subprocess
from unittest import mock
from unittest.mock import MagicMock, patch

import pytest
from filelock import FileLock

from froggy.envs import SWEBenchEnv
from froggy.envs.env import RepoEnv
from froggy.envs.swe_bench import SWEBenchEnv
from froggy.terminal import DockerTerminal

if_docker_running = pytest.mark.skipif(
    not subprocess.check_output(["docker", "ps"]),
    reason="Docker not running",
)


# https://pytest-xdist.readthedocs.io/en/stable/how-to.html#making-session-scoped-fixtures-execute-only-once
@pytest.fixture(scope="session")
def build_swe_env_once(tmp_path_factory, worker_id):
    """Build the SWEBench docker image only once.
    Do not run this fixture directly, use get_swe_env instead.
    """
    _build_swe_env = lambda: SWEBenchEnv(instance_ids=["astropy__astropy-14096"])
    if worker_id == "master":
        # Not running with pytest-xdist or we are in the master process
        _build_swe_env()
    else:
        # When running with pytest-xdist, synchronize between workers using a lock
        root_tmp_dir = tmp_path_factory.getbasetemp().parent
        lock_file = root_tmp_dir / "db_init.lock"
        with FileLock(str(lock_file)):
            # Only the first worker to acquire the lock will initialize the DB
            _build_swe_env()


@pytest.fixture
def get_swe_env(build_swe_env_once):
    """Instantiate a SWEBenchEnv instance after building the SWEBench docker image."""

    def _swe_env(working_dir=None, map_host_uid_gid=True, **kwargs):
        instance_ids = ["astropy__astropy-14096"]
        terminal = DockerTerminal(
            path=working_dir, map_host_uid_gid=map_host_uid_gid, **kwargs
        )
        env = SWEBenchEnv(instance_ids=instance_ids, terminal=terminal)
        return env

    return _swe_env


@if_docker_running
def test_load_dataset(tmp_path, get_swe_env):
    working_dir = str(tmp_path)
    swe_env = get_swe_env(working_dir)
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


@if_docker_running
def test_clone_repo(tmp_path, get_swe_env):
    working_dir = str(tmp_path)
    swe_env = get_swe_env(working_dir)
    task_name = "astropy__astropy-14096"
    row = swe_env.dataset[task_name]
    repo_address = row["repo"]
    org_name, repo_name = repo_address.split("/")
    local_repo_path = SWEBenchEnv.CACHE / repo_name

    if not local_repo_path.exists():
        with patch("subprocess.run") as mock_run:
            local_repo_path = swe_env.clone_repo(repo_address)
            mock_run.assert_called_once_with(
                [
                    "git",
                    "clone",
                    f"https://github.com/{repo_address.lstrip('/')}",
                    local_repo_path,
                ],
                check=True,
            )
    else:
        repo_content = os.listdir(local_repo_path)
        assert "astropy" in repo_content


@if_docker_running
def test_instructions(get_swe_env):
    swe_env = get_swe_env()
    swe_env.ds_row = {"problem_statement": "Test problem statement"}
    expected_instructions = {
        "Problem description": "Test problem statement",
        "Available tools to solve the problem": swe_env.tool_instructions,
        "Available commands": swe_env.tool_names,
    }
    assert swe_env.instructions == expected_instructions


@if_docker_running
def test_step(get_swe_env, build_env_info):
    swe_env = get_swe_env()
    swe_env.reset(options={"task_name": "astropy__astropy-14096"})
    env_info = build_env_info(obs="obs", score=5, done=True, last_run_obs="Raw output")
    with mock.patch.object(RepoEnv, "step", return_value=env_info):
        infos = swe_env.step("action")
    assert infos.obs == "obs"
    assert infos.score == 0
    assert infos.done == False
    assert infos.last_run_obs == "Raw output"


@if_docker_running
def test_reset(tmp_path, get_swe_env, build_env_info):
    working_dir = str(tmp_path)
    swe_env = get_swe_env(working_dir)
    task_name = "astropy__astropy-14096"
    obs = "Some observation"
    last_run_obs = "collected 10 items. 5 passed, 5 failed"
    env_info = build_env_info(obs=obs, last_run_obs=last_run_obs)
    with mock.patch.object(RepoEnv, "reset", return_value=env_info):
        reset_infos = swe_env.reset(options={"task_name": task_name})

    assert reset_infos.obs == obs
    assert reset_infos.last_run_obs == last_run_obs
    assert reset_infos.max_score == 1
    assert reset_infos.score == 0
    assert reset_infos.done == False


@if_docker_running
def test_repo_name(tmp_path, get_swe_env):
    working_dir = str(tmp_path)
    swe_env = get_swe_env(working_dir)
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
def test_run_command_with_raise(tmp_path, get_swe_env):
    working_dir = str(tmp_path)
    swe_env = get_swe_env(working_dir=working_dir, map_host_uid_gid=False)
    # install sudo for testing, swe-bench images already have sudo
    success, output = swe_env.terminal.run(
        ["apt update", "apt install -y sudo", "echo 'Terminal ready'"]
    )
    assert success
    assert output.endswith("Terminal ready")

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


@pytest.fixture
def install_configs_mock():
    install_configs = {
        "python": "3.12.8",
        "test_cmd": "pytest --help",
        "pre_install": ["apt-get help", "apt-get install -y vim"],
        "eval_commands": ["export TEST_VAR='Test Var'", "echo $TEST_VAR"],
        "install": "python3 -m pip install pytest==8.3.3",
        "post_install": ["echo 'Test file' > test.txt", "cat test.txt"],
        "packages": "pytest requests",
        "pip_packages": ["pytest"],
        "no_use_env": False,
    }
    return install_configs


@if_docker_running
def test_run_install(tmp_path, install_configs_mock, get_swe_env):
    working_dir = str(tmp_path)
    swe_env = get_swe_env(
        working_dir=working_dir, map_host_uid_gid=False, base_image="python:3.12-slim"
    )
    swe_env.install_configs = install_configs_mock
    swe_env.run_install()
    _, output = swe_env.run_command_with_raise("python -m pytest --version")
    assert "pytest 8.3.3" in output


@if_docker_running
def test_run_post_install(tmp_path, install_configs_mock, get_swe_env):
    working_dir = str(tmp_path)
    swe_env = get_swe_env(working_dir)
    swe_env.install_configs = install_configs_mock
    swe_env.run_post_install()
    _, output = swe_env.run_command_with_raise("cat test.txt")
    assert output == "Test file"


@if_docker_running
def test_load_dataset(tmp_path, get_swe_env):
    working_dir = str(tmp_path)
    swe_env = get_swe_env(working_dir)
    swe_env.load_dataset()
    assert swe_env.dataset_id == "princeton-nlp/SWE-bench_Verified"
    task_name = "astropy__astropy-14096"
    assert task_name in swe_env.dataset.keys()
    assert list(swe_env.dataset[task_name].keys()) == [
        "repo",
        "instance_id",
        "base_commit",
        "patch",
        "test_patch",
        "problem_statement",
        "hints_text",
        "created_at",
        "version",
        "FAIL_TO_PASS",
        "PASS_TO_PASS",
        "environment_setup_commit",
        "difficulty",
    ]


@if_docker_running
def test_setup_task_info(tmp_path, get_swe_env):
    working_dir = str(tmp_path)
    swe_env = get_swe_env(working_dir)
    task_name = "astropy__astropy-14096"
    swe_env.load_dataset()
    swe_env.setup_task_info(task_name)
    assert swe_env.task_name == task_name
    assert swe_env.ds_row["repo"] == "astropy/astropy"
    assert swe_env.ds_row["version"] == "5.1"
    assert isinstance(swe_env.ds_row, dict)
    assert isinstance(swe_env.install_configs, dict)


@if_docker_running
def test_setup_local_repo(tmp_path, get_swe_env):
    working_dir = str(tmp_path)
    swe_env = get_swe_env(working_dir)
    task_name = "astropy__astropy-14096"
    swe_env.load_dataset()
    swe_env.setup_task_info(task_name)
    swe_env.setup_local_repo()
    git_commit = subprocess.run(
        f"git -C {swe_env.working_dir} status".split(),
        stdout=subprocess.PIPE,
        text=True,
    ).stdout
    assert f"HEAD detached at {swe_env.ds_row["base_commit"][:8]}" in git_commit

    git_diff = subprocess.run(
        f"git -C {swe_env.working_dir} diff".split(),
        stdout=subprocess.PIPE,
        text=True,
    ).stdout
    git_diff = [l for l in git_diff.split("\n") if not l.startswith("index ")]
    assert git_diff == swe_env.ds_row["test_patch"].split("\n")

    assert ".froggyignore" in os.listdir(swe_env.working_dir)
