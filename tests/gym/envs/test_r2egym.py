from unittest.mock import MagicMock

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from debug_gym.agents.solution_agent import AgentSolution
from debug_gym.gym.entities import Observation
from debug_gym.gym.envs.r2egym import R2EGymEnv, parse_log_pytest
from debug_gym.gym.terminals.docker import DockerTerminal
from debug_gym.gym.tools.tool import ToolCall
from debug_gym.gym.tools.toolbox import Toolbox

R2E_TEST_OUTPUT = """\
=========================== short test summary info ============================
PASSED r2e_tests/test_example.py::TestExample::test_pass
FAILED r2e_tests/test_example.py::TestExample::test_fail - AssertionError
ERROR r2e_tests/test_example.py::TestExample::test_error
XFAIL r2e_tests/test_example.py::TestExample::test_xfail
XPASS r2e_tests/test_example.py::TestExample::test_xpass
"""


def make_r2egym_env(**task_overrides):
    task_data = {
        "instance_id": "example:abc123",
        "docker_image": "example:abc123",
        "commit_hash": "abc123",
        "problem_statement": "[ISSUE]Fix the bug[/ISSUE]",
        "expected_output_json": '{"TestExample.test_pass": "PASSED"}',
        **task_overrides,
    }
    terminal = MagicMock(spec=DockerTerminal)
    terminal.env_vars = {}
    terminal.session_commands = []
    env = R2EGymEnv(task_data=task_data, terminal=terminal)
    return env, terminal


def test_parse_log_pytest_matches_pond_semantics():
    parsed = parse_log_pytest(R2E_TEST_OUTPUT)

    assert parsed == {
        "TestExample.test_pass": "PASSED",
        "TestExample.test_fail": "FAILED",
        "TestExample.test_error": "ERROR",
        "TestExample.test_xfail": "PASSED",
        "TestExample.test_xpass": "PASSED",
    }


def test_setup_task_allows_missing_repo_name():
    env, _ = make_r2egym_env()

    env.setup_task()

    assert env.package_name == ""
    assert env.expected_output == {"TestExample.test_pass": "PASSED"}


def test_load_expected_output_falls_back_to_container_file():
    env, terminal = make_r2egym_env()
    del env.task_data["expected_output_json"]
    terminal.run.return_value = (
        True,
        '{"\\u001b[31mTestExample.test_pass\\u001b[0m - details": "PASSED"}',
    )
    env.setup_task()
    env.alt_path = "/root"

    env.load_expected_output()

    terminal.run.assert_called_once_with(
        "cat /root/expected_test_output.json", timeout=300
    )
    assert env.expected_output == {"TestExample.test_pass": "PASSED"}


def test_load_expected_output_reports_missing_sources():
    env, terminal = make_r2egym_env()
    del env.task_data["expected_output_json"]
    terminal.run.return_value = (False, "No such file")
    env.setup_task()
    env.alt_path = "/root"

    with pytest.raises(
        ValueError,
        match="expected_output_json.*expected_test_output.json",
    ):
        env.load_expected_output()


def test_setup_task_reports_invalid_dataset_expected_output():
    env, _ = make_r2egym_env(expected_output_json="not JSON")

    with pytest.raises(
        ValueError,
        match=r"task_data\['expected_output_json'\]",
    ):
        env.setup_task()


def test_eval_uses_image_test_script_and_records_details():
    env, terminal = make_r2egym_env(expected_output_json="""{
            "TestExample.test_pass": "PASSED",
            "TestExample.test_fail": "FAILED",
            "TestExample.test_error": "ERROR",
            "TestExample.test_xfail": "PASSED",
            "TestExample.test_xpass": "PASSED"
        }""")
    terminal.run.return_value = (True, R2E_TEST_OUTPUT)
    env.setup_task()

    eval_output = env.eval()

    terminal.run.assert_called_once_with(
        "bash /root/run_tests.sh", timeout=env.run_timeout
    )
    assert eval_output.details is not None
    assert eval_output.details.parsed_tests == {
        "TestExample.test_error": "ERROR",
        "TestExample.test_fail": "FAILED",
        "TestExample.test_pass": "PASSED",
        "TestExample.test_xfail": "PASSED",
        "TestExample.test_xpass": "PASSED",
    }
    assert eval_output.details.n_parsed == 5
    assert eval_output.details.n_passed == 3
    assert eval_output.details.n_failed == 2
    assert eval_output.details.reward == 1
    assert env.calculate_score(eval_output) == 1


def test_eval_requires_exact_expected_test_map():
    env, _ = make_r2egym_env(expected_output_json="""{
            "TestExample.test_pass": "PASSED",
            "TestExample.test_missing": "PASSED"
        }""")
    env.setup_task()

    details = env._calculate_eval_details("""\
=========================== short test summary info ============================
PASSED r2e_tests/test_example.py::TestExample::test_pass
""")

    assert details.reward == 0


def test_eval_ignores_unexpected_xfail_and_xpass_for_legacy_expected_maps():
    env, _ = make_r2egym_env()
    env.setup_task()

    details = env._calculate_eval_details("""\
=========================== short test summary info ============================
PASSED r2e_tests/test_example.py::TestExample::test_pass
XFAIL r2e_tests/test_example.py::TestExample::test_xfail
XPASS r2e_tests/test_example.py::TestExample::test_xpass
""")

    assert details.parsed_tests == {
        "TestExample.test_pass": "PASSED",
        "TestExample.test_xfail": "PASSED",
        "TestExample.test_xpass": "PASSED",
    }
    assert details.n_parsed == 3
    assert details.n_passed == 3
    assert details.reward == 1


def test_eval_grades_xfail_when_expected_map_includes_it():
    env, _ = make_r2egym_env(expected_output_json="""{
            "TestExample.test_pass": "PASSED",
            "TestExample.test_xfail": "FAILED"
        }""")
    env.setup_task()

    details = env._calculate_eval_details("""\
=========================== short test summary info ============================
PASSED r2e_tests/test_example.py::TestExample::test_pass
XFAIL r2e_tests/test_example.py::TestExample::test_xfail
""")

    assert details.reward == 0


def test_setup_terminal_batches_optional_image_setup():
    env, terminal = make_r2egym_env(repo_name="scrapy")
    env.setup_task()
    terminal.run.side_effect = [
        (True, ""),
        (True, "gold patch"),
        (True, ""),
    ]

    env.setup_terminal()

    setup_call = terminal.run.call_args_list[0]
    setup_commands = setup_call.args[0]
    assert isinstance(setup_commands, list)
    assert setup_call.kwargs == {"timeout": 300, "raises": True}
    assert any(
        "if [ -f /testbed/run_tests.sh ]" in command for command in setup_commands
    )
    assert any("if [ -d /r2e_tests ]" in command for command in setup_commands)
    assert terminal.env_vars["RES_OPTIONS"] == "timeout:1 attempts:1"
    assert env.gold_patch == "gold patch"


def test_load_dataset():
    task_name = "aiohttp_final:d7cd0613472fd4d9940e37f1c55921f6a1515324"
    dataset = R2EGymEnv.load_dataset(problems=[task_name])
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
        observation="Tool 'listdir' not found among available tools: eval.",
    )

    listdir_tool = Toolbox.get_tool("listdir")
    env.add_tool(listdir_tool)

    env_info = env.step(tool_call)
    assert env_info.step_observation.source == "listdir"
    # Verify we can see the aiohttp directory structure
    # Hidden files (like .git/) now appear first, followed by the original expected files
    listdir_output = env_info.step_observation.observation
    assert listdir_output.startswith(f"{env.working_dir}/")
    assert (
        ".git/" in listdir_output
    ), "Expected hidden .git/ directory in listdir output"
    # Verify the expected file listing format (after hidden files)
    listdir_expected = """|-- CHANGES/
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
    assert listdir_expected in listdir_output


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
