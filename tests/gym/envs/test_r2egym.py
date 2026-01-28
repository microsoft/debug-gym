import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from debug_gym.agents.solution_agent import AgentSolution
from debug_gym.gym.entities import Observation
from debug_gym.gym.envs.r2egym import R2EGymEnv
from debug_gym.gym.tools.tool import ToolCall
from debug_gym.gym.tools.toolbox import Toolbox


@pytest.if_docker_running
def test_load_dataset(get_r2egym_env):
    env = get_r2egym_env()

    task_name = "aiohttp_final:d7cd0613472fd4d9940e37f1c55921f6a1515324"
    dataset = env.load_dataset(problems=[task_name])
    assert task_name in dataset

    task_data = next(iter(dataset.values()))
    assert sorted(task_data.keys()) == sorted(
        [
            "commit_hash",
            "env_type",
            "docker_image",
            "execution_result_content",
            "expected_output_json",
            "instance_id",
            "modified_entity_summaries",
            "modified_files",
            "num_non_test_files",
            "num_non_test_func_methods",
            "num_non_test_lines",
            "parsed_commit_content",
            "problem_statement",
            "prompt",
            "relevant_files",
            "repo_name",
        ]
    )


def test_load_dataset_from_parquet(tmp_path):
    """Test loading R2EGym dataset from a local Parquet file."""

    # Create a minimal test Parquet file with expected schema
    parquet_file = tmp_path / "test_dataset.parquet"
    docker_image = "test_repo:test_hash_123"
    data = {
        "commit_hash": ["test_hash_123"],
        "docker_image": [docker_image],
        "execution_result_content": ["test execution result"],
        "expected_output_json": ['{"test": "output"}'],
        "modified_entity_summaries": ["test summaries"],
        "modified_files": [["file1.py", "file2.py"]],
        "num_non_test_files": [5],
        "num_non_test_func_methods": [10],
        "num_non_test_lines": [100],
        "parsed_commit_content": ["test commit content"],
        "problem_statement": ["[ISSUE]Test problem statement[/ISSUE]"],
        "prompt": ["test prompt"],
        "relevant_files": [["file1.py"]],
        "repo_name": ["test_repo"],
    }

    table = pa.table(data)
    pq.write_table(table, str(parquet_file))

    # Load the dataset from the Parquet file
    dataset = R2EGymEnv.load_dataset(dataset_id=str(parquet_file), split="train")
    dataset_entry = next(iter(dataset.values()))

    # Verify the dataset contains the expected features
    assert sorted(dataset_entry) == sorted(
        [
            "commit_hash",
            "env_type",
            "docker_image",
            "execution_result_content",
            "expected_output_json",
            "instance_id",
            "modified_entity_summaries",
            "modified_files",
            "num_non_test_files",
            "num_non_test_func_methods",
            "num_non_test_lines",
            "parsed_commit_content",
            "problem_statement",
            "prompt",
            "relevant_files",
            "repo_name",
        ]
    )

    # Verify the dataset has the expected data
    assert len(dataset) == 1
    task_name = docker_image  # For R2EGym, we use docker_image as instance_id
    assert docker_image in dataset
    assert dataset[task_name]["docker_image"] == "test_repo:test_hash_123"
    assert dataset[task_name]["commit_hash"] == "test_hash_123"
    assert "Test problem statement" in dataset[task_name]["problem_statement"]


@pytest.if_docker_running
def test_instructions(get_r2egym_env):
    env = get_r2egym_env()
    # Instructions might be wrapped by [ISSUE] [/ISSUE]
    assert env.instructions in env.task_data["problem_statement"]


@pytest.if_docker_running
def test_setup_task(get_r2egym_env):
    env = get_r2egym_env()
    assert env.task_name == "aiohttp_final:d7cd0613472fd4d9940e37f1c55921f6a1515324"
    env.setup_task()
    assert (
        env.base_image
        == "namanjain12/aiohttp_final:d7cd0613472fd4d9940e37f1c55921f6a1515324"
    )
    assert env.commit_hash == "d7cd0613472fd4d9940e37f1c55921f6a1515324"
    assert env.package_name == "aiohttp"
    assert len(env.expected_output) == 203


@pytest.if_docker_running
def test_setup_terminal(get_r2egym_env):
    env = get_r2egym_env()
    env.reset()
    _, output = env.terminal.run(f"ls -a")
    assert ".git" in output
    assert "r2e_tests" in output
    assert env.gold_patch != ""


@pytest.if_docker_running
def test_reset_and_step(get_r2egym_env):
    env = get_r2egym_env()
    env.add_tool(Toolbox.get_tool("eval"))
    env_info = env.reset()

    assert env.instructions == env_info.step_observation.observation
    assert "short test summary info" in env_info.eval_observation.observation
    assert env_info.score == env.score == 0
    assert env_info.max_score == 1
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

    listdir_tool = Toolbox.get_tool("listdir")
    env.add_tool(listdir_tool)

    env_info = env.step(tool_call)
    assert env_info.step_observation.source == "listdir"
    # Verify we can see the aiohttp directory structure
    listdir_output = env_info.step_observation.observation
    assert listdir_output.startswith(f"{env.working_dir}/")
    expected_files = [
        "aiohttp/",
        "README.rst",
        "setup.py",
        "tests/",
        "docs/",
    ]
    for expected in expected_files:
        assert expected in listdir_output, f"Expected {expected} in listdir output"


@pytest.if_docker_running
def test_apply_gold_patch(get_r2egym_env):
    env = get_r2egym_env()
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
def test_running_solution_agent(get_r2egym_env, tmp_path):
    """End-to-end SolutionAgent run for R2E-Gym environment, asserting successful resolution after gold patch."""
    env = get_r2egym_env()
    config = {
        "output_path": str(tmp_path),
        "random_seed": 0,
        "max_steps": 1,
    }
    for tool_name in ["pdb", "eval", "submit"]:
        env.add_tool(Toolbox.get_tool(tool_name))
    agent = AgentSolution(agent_args=config, llm=None, logger=env.logger)
    result = agent.run(env)
    assert result["success"]


@pytest.if_docker_running
def test_debug_entrypoint_contains_pdb(get_r2egym_env):
    """Ensure the environment's debug_entrypoint includes '-m pdb' for interactive debugging."""
    env = get_r2egym_env()
    env.reset()
    assert (
        "python -m pdb" in env.debug_entrypoint
    ), f"Expected '-m pdb' in debug_entrypoint, got: {env.debug_entrypoint}"
