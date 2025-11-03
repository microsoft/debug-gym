import pytest
from anyio import Path

from debug_gym.agents.solution_agent import AgentSolution
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
    env.add_tool(Toolbox.get_tool("eval"))
    env_info = env.reset(options={"task_name": "astropy__astropy-14096"})

    assert env.instructions == env_info.step_observation.observation
    assert "short test summary info" in env_info.eval_observation.observation
    assert env_info.score == env.score == 0
    assert env_info.max_score == env.max_score == len(env.fail_to_pass) == 1
    assert not env_info.terminated
    assert not env_info.resolved
    assert not env.terminated
    assert not env.resolved

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
def test_readonly_file(get_swe_bench_env):
    env = get_swe_bench_env()
    env_info = env.reset(options={"task_name": "astropy__astropy-14096"})
    test_filename = Path("/testbed/astropy/coordinates/tests/test_sky_coord.py")
    assert str(test_filename).replace("/testbed/", "") in env.test_directives
    assert env.workspace._is_readonly_func(test_filename)
    assert not env.workspace._is_readonly_func(test_filename.parent)

    env.add_tool(Toolbox.get_tool("view"))
    tool_call = ToolCall(
        id="view_id", name="view", arguments={"path": str(test_filename)}
    )
    env_info = env.step(tool_call)
    assert env_info.step_observation.source == "view"
    assert (
        f"Viewing `{test_filename}`"
        in env_info.step_observation.observation.splitlines()[0]
    )
    assert (
        "The file is read-only."
        in env_info.step_observation.observation.splitlines()[0]
    )

    env.add_tool(Toolbox.get_tool("listdir"))
    tool_call = ToolCall(
        id="listdir_id", name="listdir", arguments={"path": str(test_filename.parent)}
    )
    env_info = env.step(tool_call)
    assert env_info.step_observation.source == "listdir"
    assert "|-- test_sky_coord.py (read-only)" in env_info.step_observation.observation


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
    assert f"Applying test patch for {task_name}" not in git_logs

    # Check that the gold test patch has not been applied.
    _, code_diff = env.terminal.run("git diff")
    for test_directive in env.test_directives:
        assert test_directive not in code_diff

    # The test patch will be applied during eval.
    eval_output = env.eval()
    env.max_score = env.calculate_max_score(eval_output)
    score = env.calculate_score(eval_output)
    assert score < env.max_score
    assert score == 0

    # But after calling eval, the gold test patch is removed.
    _, code_diff = env.terminal.run("git diff")
    for test_directive in env.test_directives:
        assert test_directive not in code_diff


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
    test_dir = str(tmp_path)
    test_file = tmp_path / "test_patch_file.py"
    test_content = """def hello_world():
    print("Hello, World!")
    return "success"
"""
    test_file.write_text(test_content)
    env.workspace.copy_content(test_dir)

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
    env.add_tool(Toolbox.get_tool("eval"))
    env_info = env.reset(options={"task_name": "astropy__astropy-14096"})

    assert not env_info.terminated
    assert not env_info.resolved
    assert env_info.score == env.score == 0

    env.apply_gold_patch()
    env_info = env.step(ToolCall(id="eval_id", name="eval", arguments={}))
    assert env_info.step_observation.source == "eval"
    assert env_info.score == env_info.max_score


@pytest.if_docker_running
def test_running_solution_agent(get_swe_bench_env, tmp_path):
    env = get_swe_bench_env()
    # BaseAgent requires a config dict with at least: output_path, random_seed, memory_size.
    # Provide a minimal config for the SolutionAgent run.
    config = {
        "output_path": str(tmp_path),
        "random_seed": 0,
        "memory_size": 8,
        # Optional values that BaseAgent.run would use; harmless to include here.
        "max_steps": 1,
        "env_kwargs": {},
    }
    for tool_name in ["pdb", "submit"]:
        env.add_tool(Toolbox.get_tool(tool_name))
    agent = AgentSolution(config=config, env=env, llm=None, logger=env.logger)
    success = agent.run(task_name="astropy__astropy-14096")
    assert success


@pytest.if_docker_running
def test_debug_entrypoint_contains_pdb(get_swe_bench_env):
    """Ensure the environment's debug_entrypoint includes '-m pdb' for interactive debugging."""
    env = get_swe_bench_env()
    env.reset(options={"task_name": "astropy__astropy-14096"})
    assert (
        "python -m pdb" in env.debug_entrypoint
    ), f"Expected '-m pdb' in debug_entrypoint, got: {env.debug_entrypoint}"


@pytest.if_docker_running
def test_setup_terminal_debug_mode(get_swe_bench_debug_env):
    env = get_swe_bench_debug_env()
    task_name = "astropy__astropy-14096"
    env.reset(options={"task_name": task_name})
    _, git_logs = env.terminal.run("git log -n 4")
    assert env.base_commit in git_logs
    assert f"Applying test patch for {task_name}" in git_logs

    _, git_diff = env.terminal.run("git show HEAD", strip_output=False)
    git_diff = git_diff[git_diff.index("diff --git") :]


@pytest.if_docker_running
def test_running_solution_agent_in_debug_mode(get_swe_bench_debug_env, tmp_path):
    env = get_swe_bench_debug_env()
    # BaseAgent requires a config dict with at least: output_path, random_seed, memory_size.
    # Provide a minimal config for the SolutionAgent run.
    config = {
        "output_path": str(tmp_path),
        "random_seed": 0,
        "memory_size": 8,
        # Optional values that BaseAgent.run would use; harmless to include here.
        "max_steps": 1,
        "env_kwargs": {},
    }
    for tool_name in ["pdb", "eval", "submit"]:
        env.add_tool(Toolbox.get_tool(tool_name))
    agent = AgentSolution(config=config, env=env, llm=None, logger=env.logger)
    success = agent.run(task_name="astropy__astropy-14096")
    assert success
