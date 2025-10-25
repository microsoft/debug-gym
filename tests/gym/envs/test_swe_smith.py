from pathlib import Path

import pytest

from debug_gym.gym.entities import Observation
from debug_gym.gym.tools.tool import ToolCall
from debug_gym.gym.tools.toolbox import Toolbox


@pytest.if_docker_running
def test_load_dataset(get_swe_smith_env):
    env = get_swe_smith_env()
    assert env.dataset_id == "SWE-bench/SWE-smith"
    # check if the dataset contains features that SWESmithEnv expects
    assert sorted(env.ds.features.keys()) == sorted(
        [
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
    )


@pytest.if_docker_running
def test_instructions(get_swe_smith_env):
    env = get_swe_smith_env()
    env.ds_row = {"problem_statement": "Test problem statement"}
    expected_instructions = "Test problem statement"
    assert env.instructions == expected_instructions


@pytest.if_docker_running
def test_setup_task(get_swe_smith_env):
    env = get_swe_smith_env()
    task_name = "john-kurkowski__tldextract.3d1bf184.combine_file__1vnuqpt4"
    env.setup_task(task_name)
    assert env.task_name == task_name
    assert env.repo == "john-kurkowski/tldextract"
    assert env.branch_name == task_name
    assert env.package_name == "tldextract"


@pytest.if_docker_running
def test_setup_terminal(get_swe_smith_env):
    env = get_swe_smith_env()
    task_name = "john-kurkowski__tldextract.3d1bf184.combine_file__1vnuqpt4"
    env.reset(options={"task_name": task_name})
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
    env_info = env.reset(
        options={
            "task_name": "john-kurkowski__tldextract.3d1bf184.combine_file__1vnuqpt4"
        }
    )

    assert env.instructions == env_info.step_observation.observation
    assert "short test summary info" in env_info.eval_observation.observation
    assert env_info.score == env.score == 0
    assert env_info.max_score == env.max_score == len(env.fail_to_pass) == 39
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
|-- CHANGELOG.md
|-- LICENSE
|-- README.md
|-- pyproject.toml
|-- scripts/
|-- tests/
|-- tldextract/
|-- tox.ini"""
    assert env_info.step_observation.observation.startswith(listdir_start)


@pytest.if_docker_running
def test_readonly_file(get_swe_smith_env):
    env = get_swe_smith_env()
    env_info = env.reset(
        options={
            "task_name": "john-kurkowski__tldextract.3d1bf184.combine_file__1vnuqpt4"
        }
    )

    env.add_tool(Toolbox.get_tool("view"))
    env.add_tool(Toolbox.get_tool("listdir"))

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

        tool_call = ToolCall(
            id="listdir_id",
            name="listdir",
            arguments={"path": str(test_filename.parent)},
        )
        env_info = env.step(tool_call)
        assert env_info.step_observation.source == "listdir"
        assert (
            f"|-- {test_filename.name} (read-only)"
            in env_info.step_observation.observation
        )


@pytest.if_docker_running
def test_apply_gold_patch(get_swe_smith_env):
    env = get_swe_smith_env()
    env.add_tool(Toolbox.get_tool("eval"))
    env_info = env.reset(
        options={
            "task_name": "john-kurkowski__tldextract.3d1bf184.combine_file__1vnuqpt4"
        }
    )

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
    task_name = "john-kurkowski__tldextract.3d1bf184.combine_file__1vnuqpt4"
    env.reset(options={"task_name": task_name})

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
