import os
import subprocess

import pytest
from filelock import FileLock

from debug_gym.gym.entities import Observation
from debug_gym.gym.envs import SWEBenchEnv
from debug_gym.gym.terminal import DockerTerminal
from debug_gym.gym.tools.tool import ToolCall
from debug_gym.gym.tools.toolbox import Toolbox


def is_docker_running():
    try:
        subprocess.check_output(["docker", "ps"])
        return True
    except Exception:
        return False


if_docker_running = pytest.mark.skipif(
    not is_docker_running(),
    reason="Docker not running",
)


# https://pytest-xdist.readthedocs.io/en/stable/how-to.html#making-session-scoped-fixtures-execute-only-once
@pytest.fixture(scope="session")
def build_swe_env_once(tmp_path_factory, worker_id):
    """Build the SWEBench docker image only once.
    Do not run this fixture directly, use get_swe_env instead.
    """
    _build_swe_env = lambda: SWEBenchEnv(problems=["astropy__astropy-14096"])
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
        problems = ["astropy__astropy-14096"]
        terminal = DockerTerminal(
            path=working_dir, map_host_uid_gid=map_host_uid_gid, **kwargs
        )
        env = SWEBenchEnv(problems=problems, terminal=terminal)
        return env

    return _swe_env


@if_docker_running
def test_instructions(get_swe_env):
    swe_env = get_swe_env()
    swe_env.ds_row = {"problem_statement": "Test problem statement"}
    expected_instructions = "Test problem statement"
    assert swe_env.instructions == expected_instructions


@if_docker_running
def test_reset_and_step(get_swe_env):
    swe_env = get_swe_env()
    env_info = swe_env.reset(options={"task_name": "astropy__astropy-14096"})

    assert "short test summary info" in env_info.step_observation.observation
    assert env_info.score == swe_env.score == 0
    assert env_info.max_score == swe_env.max_score == len(swe_env.fail_to_pass) == 1
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
    listdir_start = f"""{swe_env.working_dir}/
|-- CHANGES.rst
|-- CITATION
|-- CODE_OF_CONDUCT.md
|-- CONTRIBUTING.md
|-- GOVERNANCE.md
|-- LICENSE.rst
|-- MANIFEST.in
|-- README.rst
|-- astropy/"""
    assert env_info.step_observation.observation.startswith(listdir_start)


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
        ValueError,
        match=(
            "Failed to run command `cat /non_existent_file`:\n"
            "cat: /non_existent_file: No such file or directory"
        ),
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
    assert swe_env.dataset_id == "princeton-nlp/SWE-bench_Verified"
    task_name = "astropy__astropy-14096"
    assert task_name in swe_env.dataset.keys()
    assert list(swe_env.ds.features.keys()) == [
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
def test_setup_task(tmp_path, get_swe_env):
    working_dir = str(tmp_path)
    swe_env = get_swe_env(working_dir)
    task_name = "astropy__astropy-14096"
    swe_env.setup_task(task_name)
    assert swe_env.task_name == task_name
    assert swe_env.ds_row["repo"] == "astropy/astropy"
    assert swe_env.ds_row["version"] == "5.1"
    assert isinstance(swe_env.ds_row, dict)
    assert isinstance(swe_env.install_configs, dict)


@if_docker_running
def test_setup_terminal(tmp_path, get_swe_env):
    working_dir = str(tmp_path)
    swe_env = get_swe_env(working_dir)
    task_name = "astropy__astropy-14096"
    swe_env.setup_task(task_name)
    swe_env.setup_terminal()
    git_logs = subprocess.run(
        f"git -C {swe_env.working_dir} log -n 4".split(),
        stdout=subprocess.PIPE,
        text=True,
    ).stdout
    assert swe_env.base_commit in git_logs
    assert f"Applying test patch for {task_name}" in git_logs
    assert "Add debug-gym ignore and read-only files" in git_logs

    git_diff = subprocess.run(
        f"git -C {swe_env.working_dir} show HEAD^1".split(),
        stdout=subprocess.PIPE,
        text=True,
    ).stdout
    git_diff = git_diff[git_diff.index("diff --git") :]
    git_diff = [l for l in git_diff.split("\n") if not l.startswith("index ")]
    assert git_diff == swe_env.test_patch.split("\n")

    assert ".debugignore" in os.listdir(swe_env.working_dir)
    assert ".debugreadonly" in os.listdir(swe_env.working_dir)


@if_docker_running
def test_patch_property(tmp_path, get_swe_env):
    """Test the patch property that generates git diff output."""
    swe_env = get_swe_env(working_dir=tmp_path)

    # Reset with a task to set up the environment
    swe_env.reset(options={"task_name": "astropy__astropy-14096"})

    # Initially, there should be no changes (empty patch)
    initial_patch = swe_env.patch
    assert initial_patch == "", f"Expected empty patch initially, got: {initial_patch}"

    # Create a test file with some content
    test_file = swe_env.working_dir / "test_patch_file.py"
    test_content = """def hello_world():
    print("Hello, World!")
    return "success"
"""
    test_file.write_text(test_content)

    # Add the file to git
    swe_env.terminal.run(f"git add {test_file}")
    swe_env.terminal.run("git commit -m 'Add test file'")

    # Now modify the file
    modified_content = """def hello_world():
    print("Hello, Modified World!")
    return "modified"

def new_function():
    return "new"
"""
    test_file.write_text(modified_content)

    # Get the patch
    patch = swe_env.patch

    # Verify patch contains expected changes
    assert patch != "", "Patch should not be empty after file modification"
    assert "test_patch_file.py" in patch, "Patch should reference the modified file"
    assert "Hello, World!" in patch, "Patch should contain old content"
    assert "Hello, Modified World!" in patch, "Patch should contain new content"
    assert "-" in patch and "+" in patch, "Patch should contain diff markers"

    # Test edge case: deleted file
    test_file.unlink()
    patch_with_deletion = swe_env.patch
    assert "test_patch_file.py" in patch_with_deletion
    assert "deleted file" in patch_with_deletion.lower() or "---" in patch_with_deletion


@if_docker_running
def test_apply_gold_patch(tmp_path, get_swe_env):
    swe_env = get_swe_env()
    env_info = swe_env.reset(options={"task_name": "astropy__astropy-14096"})

    assert not env_info.done
    assert env_info.score == swe_env.score == 0

    swe_env.apply_gold_patch()
    eval_output = swe_env.eval()
    score = swe_env.calculate_score(eval_output)

    assert score == swe_env.max_score
