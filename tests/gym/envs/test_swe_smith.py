import os
import subprocess
from unittest.mock import patch

import pytest
from filelock import FileLock

from debug_gym.gym.entities import Observation
from debug_gym.gym.envs import SWESmithEnv
from debug_gym.gym.terminal import DockerTerminal
from debug_gym.gym.tools.tool import ToolCall
from debug_gym.gym.tools.toolbox import Toolbox

if_docker_running = pytest.mark.skipif(
    not subprocess.check_output(["docker", "ps"]),
    reason="Docker not running",
)


@pytest.fixture(scope="session")
def build_swe_env_once(tmp_path_factory, worker_id):
    """Build the SWESmith docker image only once.
    Do not run this fixture directly, use get_swe_env instead.
    """
    _build_swe_env = lambda: SWESmithEnv(
        instance_ids=["john-kurkowski__tldextract.3d1bf184.combine_file__1vnuqpt4"]
    )
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
    """Instantiate a SWESmithEnv instance after building the SWESmith docker image."""

    def _swe_env(working_dir=None, map_host_uid_gid=True, **kwargs):
        instance_ids = ["john-kurkowski__tldextract.3d1bf184.combine_file__1vnuqpt4"]
        terminal = DockerTerminal(
            path=working_dir, map_host_uid_gid=map_host_uid_gid, **kwargs
        )
        env = SWESmithEnv(instance_ids=instance_ids, terminal=terminal)
        return env

    return _swe_env


@if_docker_running
def test_load_dataset(tmp_path, get_swe_env):
    working_dir = str(tmp_path)
    swe_env = get_swe_env(working_dir)
    assert swe_env.dataset_id == "SWE-bench/SWE-smith"
    # check if the dataset contains features that SWESmithEnv expects
    assert sorted(swe_env.ds.features.keys()) == [
        "instance_id",
        "repo",
        "patch",
        "FAIL_TO_PASS",
        "PASS_TO_PASS",
        "created_at",
        "image_name",
        "base_commit",
        "problem_statement",
    ]


@if_docker_running
def test_instructions(get_swe_env):
    swe_env = get_swe_env()
    swe_env.ds_row = {"problem_statement": "Test problem statement"}
    expected_instructions = {"Problem description": "Test problem statement"}
    assert swe_env.instructions == expected_instructions


@if_docker_running
def test_setup_task(tmp_path, get_swe_env):
    working_dir = str(tmp_path)
    swe_env = get_swe_env(working_dir)
    task_name = "john-kurkowski__tldextract.3d1bf184.combine_file__1vnuqpt4"
    swe_env.setup_task(
        task_name
    )  # SWE-smith uses setup_task instead of setup_task_info
    assert swe_env.task_name == task_name
    assert swe_env.repo == "john-kurkowski/tldextract"
    assert swe_env.branch_name == task_name
    assert swe_env.package_name == "tldextract"


@if_docker_running
def test_setup_terminal(tmp_path, get_swe_env):
    working_dir = str(tmp_path)
    swe_env = get_swe_env(working_dir)
    task_name = "john-kurkowski__tldextract.3d1bf184.combine_file__1vnuqpt4"
    swe_env.setup_task(task_name)
    swe_env.setup_terminal()
    git_logs = subprocess.run(
        f"git -C {swe_env.working_dir} log -n 4".split(),
        stdout=subprocess.PIPE,
        text=True,
    ).stdout
    # For SWE-Smith the base commit is found in the branch associated to the
    # instance id and is different from the one in the main branch.
    # assert swe_env.base_commit in git_logs
    assert f"Applying test patch for {task_name}" in git_logs
    assert "Add debug-gym ignore and read-only files" in git_logs

    git_diff = subprocess.run(
        f"git -C {swe_env.working_dir} show HEAD^1".split(),
        stdout=subprocess.PIPE,
        text=True,
    ).stdout
    git_diff = git_diff[git_diff.index("diff --git") :]
    assert git_diff == swe_env.test_patch

    assert ".debugignore" in os.listdir(swe_env.working_dir)
    assert ".debugreadonly" in os.listdir(swe_env.working_dir)


@if_docker_running
def test_reset_and_step(get_swe_env):
    swe_env = get_swe_env()
    env_info = swe_env.reset(
        options={
            "task_name": "john-kurkowski__tldextract.3d1bf184.combine_file__1vnuqpt4"
        }
    )

    assert "short test summary info" in env_info.step_observation.observation
    assert env_info.score == swe_env.score == 0
    assert env_info.max_score == swe_env.max_score == len(swe_env.fail_to_pass) == 39
    assert not env_info.done
    assert not swe_env.done

    tool_call = ToolCall(id="listdir_id", name="listdir", arguments={})
    env_info = swe_env.step(tool_call)
    assert env_info.step_observation == Observation(
        source="env",
        observation="Unregistered tool: listdir",
    )

    view_tool = Toolbox.get_tool("listdir")
    swe_env.add_tool(view_tool)

    env_info = swe_env.step(tool_call)
    assert env_info.step_observation.source == "listdir"
    # Verify we can see the tldextract directory structure
    observation = env_info.step_observation.observation
    listdir_start = f"""{swe_env.working_dir}/
|-- CHANGELOG.md
|-- LICENSE
|-- README.md
|-- pyproject.toml
|-- scripts/
|-- tests/
|-- tldextract/
|-- tox.ini"""
    assert env_info.step_observation.observation.startswith(listdir_start)


@if_docker_running
def test_apply_gold_patch(tmp_path, get_swe_env):
    swe_env = get_swe_env()
    env_info = swe_env.reset(
        options={
            "task_name": "john-kurkowski__tldextract.3d1bf184.combine_file__1vnuqpt4"
        }
    )

    assert not env_info.done
    assert env_info.score == swe_env.score == 0

    swe_env.apply_gold_patch()
    eval_output = swe_env.eval()
    score = swe_env.calculate_score(eval_output)

    assert score == swe_env.max_score
