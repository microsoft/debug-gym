from pathlib import Path
from unittest.mock import MagicMock, patch

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from debug_gym.agents.solution_agent import AgentSolution
from debug_gym.gym.entities import Observation
from debug_gym.gym.envs.r2egym import R2EGymEnv
from debug_gym.gym.terminals.docker import DockerTerminal
from debug_gym.gym.tools.tool import ToolCall
from debug_gym.gym.tools.toolbox import Toolbox


@pytest.if_docker_running
def test_load_dataset(get_r2egym_env):
    env = get_r2egym_env()
    assert env.dataset_id == "R2E-Gym/R2E-Gym-Lite"
    # check if the dataset contains features that R2EGymEnv expects
    assert sorted(env.ds.features.keys()) == sorted(
        [
            "commit_hash",
            "docker_image",
            "execution_result_content",
            "expected_output_json",
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


@patch("docker.from_env")
def test_load_dataset_from_parquet(mock_docker_from_env, tmp_path):
    """Test loading R2EGym dataset from a local Parquet file."""
    # Mock Docker client to avoid trying to pull images
    mock_docker_client = MagicMock()
    mock_docker_client.images.list.return_value = []
    mock_docker_from_env.return_value = mock_docker_client
    
    # Create a minimal test Parquet file with expected schema
    parquet_file = tmp_path / "test_dataset.parquet"
    
    data = {
        "commit_hash": ["test_hash_123"],
        "docker_image": ["test_repo:test_hash_123"],
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
    
    # Mock the terminal to avoid actual Docker operations
    mock_terminal = MagicMock(spec=DockerTerminal)
    
    # Load the dataset from the Parquet file
    env = R2EGymEnv(
        dataset_id=str(parquet_file), 
        split="train", 
        terminal=mock_terminal
    )
    
    # Verify the dataset contains the expected features
    assert sorted(env.ds.features.keys()) == sorted(
        [
            "commit_hash",
            "docker_image",
            "execution_result_content",
            "expected_output_json",
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
    assert len(env.ds) == 1
    assert env.ds[0]["docker_image"] == "test_repo:test_hash_123"
    assert env.ds[0]["commit_hash"] == "test_hash_123"
    assert "Test problem statement" in env.ds[0]["problem_statement"]


@pytest.if_docker_running
def test_instructions(get_r2egym_env):
    env = get_r2egym_env()
    env.setup_task("aiohttp_final:d7cd0613472fd4d9940e37f1c55921f6a1515324")
    # Instructions might be wrapped by [ISSUE] [/ISSUE]
    assert env.instructions in env.ds_row["problem_statement"]


@pytest.if_docker_running
def test_setup_task(get_r2egym_env):
    env = get_r2egym_env()
    task_name = "aiohttp_final:d7cd0613472fd4d9940e37f1c55921f6a1515324"
    env.setup_task(task_name)
    assert env.task_name == task_name
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
    task_name = "aiohttp_final:d7cd0613472fd4d9940e37f1c55921f6a1515324"
    env.reset(options={"task_name": task_name})
    _, output = env.terminal.run(f"ls -a")
    assert ".git" in output
    assert "r2e_tests" in output
    assert env.gold_patch != ""


@pytest.if_docker_running
def test_reset_and_step(get_r2egym_env):
    env = get_r2egym_env()
    env.add_tool(Toolbox.get_tool("eval"))
    env_info = env.reset(
        options={"task_name": "aiohttp_final:d7cd0613472fd4d9940e37f1c55921f6a1515324"}
    )

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

    view_tool = Toolbox.get_tool("listdir")
    env.add_tool(view_tool)

    env_info = env.step(tool_call)
    assert env_info.step_observation.source == "listdir"
    # Verify we can see the tldextract directory structure
    observation = env_info.step_observation.observation
    listdir_start = f"""{env.working_dir}/
|-- CHANGES/
|-- CHANGES.rst
|-- CODE_OF_CONDUCT.md
|-- CONTRIBUTING.rst
|-- CONTRIBUTORS.txt
|-- HISTORY.rst
|-- LICENSE.txt
|-- MANIFEST.in
|-- Makefile
|-- README.rst
|-- aiohttp/
|-- docs/
|-- examples/
|-- install.sh
|-- process_aiohttp_updateasyncio.py
|-- pyproject.toml
|-- r2e_tests/
|-- requirements/
|-- setup.cfg
|-- setup.py
|-- tests/
|-- tools/
|-- vendor/"""
    assert env_info.step_observation.observation.startswith(listdir_start)


@pytest.if_docker_running
def test_readonly_file(get_r2egym_env):
    env = get_r2egym_env()
    env_info = env.reset(
        options={"task_name": "aiohttp_final:d7cd0613472fd4d9940e37f1c55921f6a1515324"}
    )
    assert env.workspace._is_readonly_func("/testbed/r2e_tests/test_1.py")

    env.add_tool(Toolbox.get_tool("view"))
    env.add_tool(Toolbox.get_tool("listdir"))

    tool_call = ToolCall(
        id="listdir_id", name="listdir", arguments={"path": "r2e_tests"}
    )
    env_info = env.step(tool_call)
    assert f"|-- test_1.py (read-only)" in env_info.step_observation.observation

    tool_call = ToolCall(
        id="view_id", name="view", arguments={"path": "r2e_tests/test_1.py"}
    )
    env_info = env.step(tool_call)
    assert (
        f"Viewing `r2e_tests/test_1.py`"
        in env_info.step_observation.observation.splitlines()[0]
    )
    assert (
        "The file is read-only."
        in env_info.step_observation.observation.splitlines()[0]
    )


@pytest.if_docker_running
def test_apply_gold_patch(get_r2egym_env):
    env = get_r2egym_env()
    env.add_tool(Toolbox.get_tool("eval"))
    env_info = env.reset(
        options={"task_name": "aiohttp_final:d7cd0613472fd4d9940e37f1c55921f6a1515324"}
    )

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
    task_name = "aiohttp_final:d7cd0613472fd4d9940e37f1c55921f6a1515324"
    config = {
        "output_path": str(tmp_path),
        "random_seed": 0,
        "memory_size": 8,
        "max_steps": 1,
        "env_kwargs": {},
    }
    for tool_name in ["pdb", "eval", "submit"]:
        env.add_tool(Toolbox.get_tool(tool_name))
    agent = AgentSolution(config=config, env=env, llm=None, logger=env.logger)
    success = agent.run(task_name=task_name)
    assert success


@pytest.if_docker_running
def test_debug_entrypoint_contains_pdb(get_r2egym_env):
    """Ensure the environment's debug_entrypoint includes '-m pdb' for interactive debugging."""
    env = get_r2egym_env()
    env.reset(
        options={"task_name": "aiohttp_final:d7cd0613472fd4d9940e37f1c55921f6a1515324"}
    )
    assert (
        "python -m pdb" in env.debug_entrypoint
    ), f"Expected '-m pdb' in debug_entrypoint, got: {env.debug_entrypoint}"
