import pytest

from debug_gym.gym.entities import Observation
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


@pytest.if_docker_running
def test_instructions(get_r2egym_env):
    env = get_r2egym_env()
    env.setup_task("aiohttp_final:d7cd0613472fd4d9940e37f1c55921f6a1515324")
    assert env.ds_row["problem_statement"] in env.instructions


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
    env_info = env.reset(
        options={"task_name": "aiohttp_final:d7cd0613472fd4d9940e37f1c55921f6a1515324"}
    )

    assert "short test summary info" in env_info.step_observation.observation
    assert env_info.score == env.score == 0
    assert env_info.max_score == 1
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
def test_apply_gold_patch(get_r2egym_env):
    env = get_r2egym_env()
    env_info = env.reset(
        options={"task_name": "aiohttp_final:d7cd0613472fd4d9940e37f1c55921f6a1515324"}
    )

    assert not env_info.done
    assert env_info.score == env.score == 0

    env.apply_gold_patch()
    eval_output = env.eval()
    score = env.calculate_score(eval_output)

    assert score == env.max_score
