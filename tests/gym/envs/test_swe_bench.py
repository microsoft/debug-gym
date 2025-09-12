import pytest

from debug_gym.gym.entities import Observation
from debug_gym.gym.tools.tool import ToolCall
from debug_gym.gym.tools.toolbox import Toolbox


@pytest.if_docker_running
def test_instructions(get_swe_bench_env):
    env = get_swe_bench_env()
    env.ds_row = {"problem_statement": "Test problem statement"}
    expected_instructions = "Test problem statement"
    assert env.instructions == expected_instructions


@pytest.if_docker_running
def test_reset_and_step(get_swe_bench_env):
    env = get_swe_bench_env()
    env_info = env.reset(options={"task_name": "astropy__astropy-14096"})

    assert "short test summary info" in env_info.step_observation.observation
    assert env_info.score == env.score == 0
    assert env_info.max_score == env.max_score == len(env.fail_to_pass) == 1
    assert not env_info.done
    assert not env.done

    tool_call = ToolCall(id="listdir_id", name="listdir", arguments={})
    env_info = env.step(tool_call)
    assert env_info.step_observation == Observation(
        source="env",
        observation="Unregistered tool: listdir",
    )

    view_tool = Toolbox.get_tool("listdir")
    env.add_tool(view_tool)

    env_info = env.step(tool_call)
    assert env_info.step_observation.source == "listdir"
    listdir_start = f"""{env.working_dir}/
|-- CHANGES.rst
|-- CITATION
|-- CODE_OF_CONDUCT.md
|-- CONTRIBUTING.md
|-- GOVERNANCE.md
|-- LICENSE.rst
|-- MANIFEST.in
|-- README.rst
|-- astropy/
|-- cextern/
|-- codecov.yml
|-- conftest.py
|-- docs/
|-- examples/
|-- licenses/
|-- pip-requirements
|-- pyproject.toml
|-- setup.cfg
|-- setup.py*
|-- tox.ini"""
    assert env_info.step_observation.observation.startswith(listdir_start)


@pytest.if_docker_running
def test_load_dataset(get_swe_bench_env):
    env = get_swe_bench_env()
    assert env.dataset_id == "SWE-bench/SWE-bench_Verified"
    task_name = "astropy__astropy-14096"
    assert task_name in env.dataset.keys()
    assert list(env.ds.features.keys()) == [
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


@pytest.if_docker_running
def test_setup_task(get_swe_bench_env):
    env = get_swe_bench_env()
    task_name = "astropy__astropy-14096"
    env.setup_task(task_name)
    assert env.task_name == task_name
    assert env.ds_row["repo"] == "astropy/astropy"
    assert env.ds_row["version"] == "5.1"
    assert isinstance(env.ds_row, dict)
    assert isinstance(env.install_configs, dict)


@pytest.if_docker_running
def test_setup_terminal(get_swe_bench_env):
    env = get_swe_bench_env()
    task_name = "astropy__astropy-14096"
    env.reset(options={"task_name": task_name})
    _, git_logs = env.terminal.run("git log -n 4")
    assert env.base_commit in git_logs
    assert f"Applying test patch for {task_name}" in git_logs

    _, git_diff = env.terminal.run("git show HEAD", strip_output=False)
    git_diff = git_diff[git_diff.index("diff --git") :]
    git_diff = [l for l in git_diff.split("\n") if not l.startswith("index ")]
    assert git_diff == env.test_patch.split("\n")


@pytest.if_docker_running
def test_patch_property(tmp_path, get_swe_bench_env):
    """Test the patch property that generates git diff output."""
    env = get_swe_bench_env()

    # Reset with a task to set up the environment
    env.reset(options={"task_name": "astropy__astropy-14096"})

    # Initially, there should be no changes (empty patch)
    initial_patch = env.patch
    assert initial_patch == "", f"Expected empty patch initially, got: {initial_patch}"

    # Create a test file with some content
    test_file = tmp_path / "test_patch_file.py"
    test_content = """def hello_world():
    print("Hello, World!")
    return "success"
"""
    test_file.write_text(test_content)
    env.workspace.copy_content(test_file)

    # Add the file to git
    env.terminal.run(f"git add {test_file.name}")
    env.terminal.run("git commit -m 'Add test file'")

    # Now modify the file
    modified_content = """def hello_world():
    print("Hello, Modified World!")
    return "modified"

def new_function():
    return "new"
"""
    env.workspace.write_file(test_file.name, modified_content)

    # Get the patch
    patch = env.patch

    # Verify patch contains expected changes
    assert patch != "", "Patch should not be empty after file modification"
    assert "test_patch_file.py" in patch, "Patch should reference the modified file"
    assert "Hello, World!" in patch, "Patch should contain old content"
    assert "Hello, Modified World!" in patch, "Patch should contain new content"
    assert "-" in patch and "+" in patch, "Patch should contain diff markers"

    # Test edge case: deleted file
    test_file.unlink()
    patch_with_deletion = env.patch
    assert "test_patch_file.py" in patch_with_deletion
    assert "deleted file" in patch_with_deletion.lower() or "---" in patch_with_deletion


@pytest.if_docker_running
def test_apply_gold_patch(get_swe_bench_env):
    env = get_swe_bench_env()
    env_info = env.reset(options={"task_name": "astropy__astropy-14096"})

    assert not env_info.done
    assert env_info.score == env.score == 0

    env.apply_gold_patch()
    eval_output = env.eval()
    score = env.calculate_score(eval_output)

    assert score == env.max_score
