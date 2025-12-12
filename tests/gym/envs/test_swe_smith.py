from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from debug_gym.agents.solution_agent import AgentSolution
from debug_gym.gym.entities import Observation
from debug_gym.gym.envs.swe_smith import SWESmithEnv
from debug_gym.gym.tools.tool import ToolCall
from debug_gym.gym.tools.toolbox import Toolbox


@pytest.if_docker_running
def test_load_dataset(get_swe_smith_env):
    env = get_swe_smith_env()

    task_name = "john-kurkowski__tldextract.3d1bf184.combine_file__1vnuqpt4"
    dataset = env.load_dataset(problems=[task_name])
    assert task_name in dataset

    # check if the dataset contains features that SWESmithEnv expects
    task_data = next(iter(dataset.values()))
    assert sorted(task_data.keys()) == sorted(
        [
            "instance_id",
            "env_type",
            "repo",
            "patch",
            "FAIL_TO_PASS",
            "PASS_TO_PASS",
            "created_at",
            "image_name",
            "base_commit",
            "problem_statement",
        ]
    )


def test_load_dataset_from_parquet(tmp_path):
    """Test that loading from a local Parquet file works correctly."""

    # Create a sample parquet file with the required features
    data = {
        "instance_id": ["test-instance-1", "test-instance-2"],
        "repo": ["test/repo1", "test/repo2"],
        "patch": ["diff --git a/file.py", "diff --git b/file2.py"],
        "FAIL_TO_PASS": [["test1"], ["test2"]],
        "PASS_TO_PASS": [["test3"], ["test4"]],
        "created_at": ["2024-01-01", "2024-01-02"],
        "image_name": ["image1", "image2"],
        "base_commit": ["abc123", "def456"],
        "problem_statement": ["Problem 1", "Problem 2"],
    }
    parquet_file = tmp_path / "test_dataset.parquet"

    table = pa.table(data)
    pq.write_table(table, str(parquet_file))

    # Load the dataset from the Parquet file
    dataset = SWESmithEnv.load_dataset(dataset_id=str(parquet_file), split="train")
    dataset_entry = next(iter(dataset.values()))

    # Verify that the dataset was loaded correctly with expected features
    assert sorted(dataset_entry.keys()) == sorted(
        [
            "instance_id",
            "env_type",
            "repo",
            "patch",
            "FAIL_TO_PASS",
            "PASS_TO_PASS",
            "created_at",
            "image_name",
            "base_commit",
            "problem_statement",
        ]
    )
    # Verify that the data is accessible
    assert len(dataset) == 2
    assert sorted(dataset.keys()) == ["test-instance-1", "test-instance-2"]


def test_instructions(get_swe_smith_env):
    env = get_swe_smith_env()
    assert env.instructions == env.task_data["problem_statement"]


def test_setup_task(get_swe_smith_env):
    env = get_swe_smith_env()
    task_name = "john-kurkowski__tldextract.3d1bf184.combine_file__1vnuqpt4"
    assert env.task_name == task_name
    env.setup_task()
    assert env.repo == "john-kurkowski/tldextract"
    assert env.branch_name == task_name
    assert env.package_name == "tldextract"


@pytest.if_docker_running
def test_setup_terminal(get_swe_smith_env):
    env = get_swe_smith_env()
    task_name = "john-kurkowski__tldextract.3d1bf184.combine_file__1vnuqpt4"
    env.reset()
    _, git_logs = env.terminal.run("git log -n 4")
    # For SWE-Smith the base commit is found in the branch associated to the
    # instance id and is different from the one in the main branch.
    assert f"Applying bug patch for {task_name}" in git_logs

    _, git_diff = env.terminal.run("git show HEAD", strip_output=False)
    git_diff = git_diff[git_diff.index("diff --git") :]
    assert git_diff == env.bug_patch


@pytest.if_docker_running
def test_reset_and_step(get_swe_smith_env):
    env = get_swe_smith_env()
    env.add_tool(Toolbox.get_tool("eval"))
    env_info = env.reset()

    assert env.instructions == env_info.step_observation.observation
    assert "short test summary info" in env_info.eval_observation.observation
    assert env_info.score == env.score == 0
    assert env_info.max_score == env.max_score == len(env.fail_to_pass) == 39
    assert not env_info.terminated
    assert not env_info.resolved
    assert not env.terminated
    assert not env.resolved

    tool_call = ToolCall(id="bash_id", name="bash", arguments={"command": "ls"})
    env_info = env.step(tool_call)
    assert env_info.step_observation == Observation(
        source="env",
        observation="Unregistered tool: bash",
    )

    bash_tool = Toolbox.get_tool("bash")
    env.add_tool(bash_tool)

    env_info = env.step(tool_call)
    assert env_info.step_observation.source == "bash"
    # Verify some expected files are listed
    assert "tldextract" in env_info.step_observation.observation
    assert "pyproject.toml" in env_info.step_observation.observation


@pytest.if_docker_running
def test_readonly_file(get_swe_smith_env):
    env = get_swe_smith_env()
    env_info = env.reset()

    env.add_tool(Toolbox.get_tool("view"))

    for test_filename in env.test_directives:
        test_filename = Path("/testbed") / test_filename
        assert env.workspace._is_readonly_func(test_filename)

        tool_call = ToolCall(
            id="view_id", name="view", arguments={"path": str(test_filename)}
        )
        env_info = env.step(tool_call)
        assert (
            f"Viewing `{test_filename}`"
            in env_info.step_observation.observation.splitlines()[0]
        )
        assert (
            "The file is read-only."
            in env_info.step_observation.observation.splitlines()[0]
        )


@pytest.if_docker_running
def test_apply_gold_patch(get_swe_smith_env):
    env = get_swe_smith_env()
    env.add_tool(Toolbox.get_tool("eval"))
    env_info = env.reset()

    assert not env_info.terminated
    assert not env_info.resolved
    assert env_info.score == env.score == 0

    env.apply_gold_patch()
    env_info = env.step(ToolCall(id="eval_id", name="eval", arguments={}))
    assert env_info.step_observation.source == "eval"
    assert env_info.score == env_info.max_score


@pytest.if_docker_running
def test_calculate_score_with_pytest_error(get_swe_smith_env):
    """Test that the indentation error in pytest is handled correctly."""
    env = get_swe_smith_env()
    env.add_tool(Toolbox.get_tool("eval"))
    env.reset()

    # Modify 'tldextract/tldextract.py' in the working_dir to introduce an indentation error.
    content = env.workspace.read_file("tldextract/tldextract.py").split("\n")

    # Introduce an indentation error by adding an extra space at the beginning of a line.
    content[10] = " 1/0   " + content[10]
    env.workspace.write_file("tldextract/tldextract.py", "\n".join(content))

    # Now, when we run the tests, we should see an indentation error.
    eval_output = env.eval()
    # ============================= test session starts ==============================
    # platform linux -- Python 3.10.15, pytest-8.3.4, pluggy-1.5.0 -- /opt/miniconda3/envs/testbed/bin/python
    # cachedir: .pytest_cache
    # rootdir: /tmp/RepoEnv-z_m4s7ts
    # configfile: pyproject.toml
    # plugins: syrupy-4.8.0, gitignore-1.3, mock-3.14.0
    # collecting ... collected 45 items / 1 error

    # =========================== short test summary info ============================
    # ERROR tldextract/tldextract.py - ValueError: line 11 of the docstring for tld...
    # !!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
    # =============================== 1 error in 0.40s ===============================

    score = env.calculate_score(eval_output)
    assert score == 0


@pytest.if_docker_running
def test_running_solution_agent(get_swe_smith_env, tmp_path):
    """Analogous to SWE Bench solution agent test: run SolutionAgent end-to-end and assert success."""
    env = get_swe_smith_env()
    config = {
        "output_path": str(tmp_path),
        "random_seed": 0,
        "max_steps": 1,
        "env": env,
    }
    for tool_name in ["pdb", "eval", "submit"]:
        env.add_tool(Toolbox.get_tool(tool_name))
    agent = AgentSolution(agent_args=config, llm=None, logger=env.logger)
    env.reset()
    success = agent.run(env)
    assert success


@pytest.if_docker_running
def test_debug_entrypoint_contains_pdb(get_swe_smith_env):
    """Ensure the environment's debug_entrypoint includes '-m pdb' for interactive debugging."""
    env = get_swe_smith_env()
    env.reset()
    assert (
        "python -m pdb" in env.debug_entrypoint
    ), f"Expected '-m pdb' in debug_entrypoint, got: {env.debug_entrypoint}"
