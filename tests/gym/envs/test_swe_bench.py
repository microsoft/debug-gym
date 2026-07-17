import base64
from unittest.mock import MagicMock, patch

import pytest
from anyio import Path
from swebench.harness.constants import END_TEST_OUTPUT, START_TEST_OUTPUT, TESTS_TIMEOUT
from swebench.harness.constants import TestStatus as SWETestStatus
from swebench.harness.log_parsers import MAP_REPO_TO_PARSER

from debug_gym.agents.solution_agent import AgentSolution
from debug_gym.gym.entities import Observation
from debug_gym.gym.envs.swe_bench import SWEBenchEnv
from debug_gym.gym.envs.swe_bench_debug import SWEBenchDebugEnv
from debug_gym.gym.terminals.docker import DockerTerminal
from debug_gym.gym.tools.tool import ToolCall
from debug_gym.gym.tools.toolbox import Toolbox


def make_swe_bench_env(env_class=SWEBenchEnv, **task_overrides):
    task_data = {
        "repo": "astropy/astropy",
        "instance_id": "astropy__astropy-14096",
        "base_commit": "0123456789abcdef",
        "patch": "",
        "test_patch": (
            "diff --git a/tests/test_example.py b/tests/test_example.py\n"
            "--- a/tests/test_example.py\n"
            "+++ b/tests/test_example.py\n"
        ),
        "problem_statement": "Fix the bug",
        "hints_text": "",
        "created_at": "",
        "version": "5.1",
        "FAIL_TO_PASS": '["test_f2p"]',
        "PASS_TO_PASS": '["test_p2p"]',
        "environment_setup_commit": "",
        **task_overrides,
    }
    terminal = MagicMock(spec=DockerTerminal)
    terminal.env_vars = {}
    terminal.session_commands = []
    terminal.setup_commands = []
    env = env_class(task_data=task_data, terminal=terminal)
    return env, terminal


def marked_test_output(content: str = "results") -> str:
    return f"before\n{START_TEST_OUTPUT}\n{content}\n{END_TEST_OUTPUT}\nafter"


def test_setup_task_uses_official_image_and_accepts_test_lists():
    env, _ = make_swe_bench_env(
        FAIL_TO_PASS=["test_f2p"],
        PASS_TO_PASS=["test_p2p"],
    )

    env.setup_task()

    assert (
        env.base_image == "swebench/sweb.eval.x86_64.astropy_1776_astropy-14096:latest"
    )
    assert env.fail_to_pass == ["test_f2p"]
    assert env.pass_to_pass == ["test_p2p"]
    assert START_TEST_OUTPUT in env.eval_script
    assert END_TEST_OUTPUT in env.eval_script
    assert env.test_patch in env.eval_script
    assert "/run_tests.sh" not in env.eval_script


def test_setup_task_allows_explicit_image_override():
    env, _ = make_swe_bench_env(docker_image="registry.example/swebench:test-image")

    env.setup_task()

    assert env.base_image == "registry.example/swebench:test-image"


def test_setup_terminal_uses_official_image_setup_and_strips_future_history():
    env, terminal = make_swe_bench_env()
    terminal.run.return_value = (True, "")
    env.setup_task()

    env.setup_terminal()

    list_calls = [
        call.args[0]
        for call in terminal.run.call_args_list
        if isinstance(call.args[0], list)
    ]
    image_setup = next(
        commands
        for commands in list_calls
        if "ln -s /opt/miniconda3/envs/testbed /root/.venv" in commands
    )
    assert "python -m pip install chardet" in image_setup
    assert all(
        "/run_tests.sh" not in command
        for commands in list_calls
        for command in commands
    )

    history_strip = next(
        commands
        for commands in list_calls
        if any("git remote remove origin" in command for command in commands)
    )
    joined = "\n".join(history_strip)
    assert "TARGET_COMMIT=0123456789abcdef" in joined
    assert 'git checkout --detach "$TARGET_COMMIT"' in joined
    assert joined.index("git checkout --detach") < joined.index(
        "git for-each-ref --format='delete %(refname)'"
    )
    assert "refs/heads refs/remotes" in joined
    assert "git tag -l | while read -r tag" in joined
    assert "git reflog expire --expire=now --all" in joined
    assert "git gc --prune=now" in joined
    assert '[ "$COMMIT_COUNT" -eq 0 ] || exit 1' in joined


def test_setup_terminal_discovers_requests_certificate_directory():
    env, terminal = make_swe_bench_env(
        repo="psf/requests",
        instance_id="psf__requests-1",
        version="2.31",
    )
    terminal.run.return_value = (True, "")
    env.setup_task()

    env.setup_terminal()

    list_calls = [
        call.args[0]
        for call in terminal.run.call_args_list
        if isinstance(call.args[0], list)
    ]
    httpbin_setup = next(
        commands
        for commands in list_calls
        if any("pytest_httpbin.certs" in command for command in commands)
    )
    joined = "\n".join(httpbin_setup)
    assert "CERT_DIR=$(" in joined
    assert "pytest_httpbin.certs" in joined
    assert "lib/python3.9/site-packages" not in joined


def test_eval_runs_official_eval_script_and_records_binary_grading_details():
    env, terminal = make_swe_bench_env()
    env.setup_task()
    terminal.run.return_value = (True, marked_test_output())

    with patch.dict(
        MAP_REPO_TO_PARSER,
        {
            "astropy/astropy": lambda content, test_spec: {
                "test_f2p": SWETestStatus.PASSED.value,
                "test_p2p": SWETestStatus.PASSED.value,
            }
        },
    ):
        eval_output = env.eval()

    eval_call = next(
        call
        for call in terminal.run.call_args_list
        if isinstance(call.args[0], str) and "/bin/bash /eval.sh" in call.args[0]
    )
    assert eval_call.kwargs == {"timeout": env.run_timeout}
    encoded_script = (
        eval_call.args[0].split("printf %s ", 1)[1].split(" | base64 -d", 1)[0]
    )
    assert base64.b64decode(encoded_script).decode() == env.eval_script
    assert "/run_tests.sh" not in eval_call.args[0]
    assert eval_output.details is not None
    assert eval_output.details.reward == 1
    assert eval_output.details.n_fail_to_pass == 1
    assert eval_output.details.n_pass_to_pass == 1
    assert eval_output.details.n_parsed == 2
    assert eval_output.details.n_passed == 2
    assert eval_output.details.n_failed == 0
    assert env.calculate_max_score(eval_output) == 1
    assert env.calculate_score(eval_output) == 1


def test_eval_returns_zero_for_swebench_runner_failure_sentinel():
    env, _ = make_swe_bench_env()
    env.setup_task()

    details = env._calculate_eval_details(TESTS_TIMEOUT)

    assert details.reward == 0
    assert details.parsed_tests == {}


def test_eval_requires_official_output_markers():
    env, terminal = make_swe_bench_env()
    env.setup_task()
    terminal.run.return_value = (True, "results")

    with patch.dict(
        MAP_REPO_TO_PARSER,
        {
            "astropy/astropy": lambda content, test_spec: {
                "test_f2p": SWETestStatus.PASSED.value,
                "test_p2p": SWETestStatus.PASSED.value,
            }
        },
    ):
        eval_output = env.eval()

    assert eval_output.details is not None
    assert eval_output.details.reward == 0
    assert eval_output.details.parsed_tests == {}


def test_official_grading_requires_pass_to_pass_results():
    env, _ = make_swe_bench_env()
    env.setup_task()

    with patch.dict(
        MAP_REPO_TO_PARSER,
        {
            "astropy/astropy": lambda content, test_spec: {
                "test_f2p": SWETestStatus.PASSED.value,
            }
        },
    ):
        details = env._calculate_eval_details(marked_test_output())

    assert details.parsed_tests == {"test_f2p": SWETestStatus.PASSED.value}
    assert details.reward == 0


def test_parser_falls_back_to_full_log_when_marker_body_has_no_results():
    env, _ = make_swe_bench_env()
    env.setup_task()
    parsed_content = []

    def parser(content, test_spec):
        parsed_content.append(content)
        if "outside-result" in content:
            return {
                "test_f2p": SWETestStatus.PASSED.value,
                "test_p2p": SWETestStatus.PASSED.value,
            }
        return {}

    output = (
        f"outside-result\n{START_TEST_OUTPUT}\nno results\n"
        f"{END_TEST_OUTPUT}\noutside-result"
    )
    with patch.dict(MAP_REPO_TO_PARSER, {"astropy/astropy": parser}):
        details = env._calculate_eval_details(output)

    assert len(parsed_content) == 2
    assert "outside-result" not in parsed_content[0]
    assert "outside-result" in parsed_content[1]
    assert details.reward == 1


def test_new_test_files_are_removed_without_resetting_the_worktree():
    test_patch = """\
diff --git a/tests/test_new.py b/tests/test_new.py
new file mode 100644
--- /dev/null
+++ b/tests/test_new.py
@@ -0,0 +1 @@
+def test_new(): pass
"""
    env, terminal = make_swe_bench_env(test_patch=test_patch)

    env.setup_task()
    env._restore_test_files()

    assert f"git checkout {env.base_commit}" not in env.eval_script
    assert env.eval_script.count("rm -f tests/test_new.py") == 2
    terminal.run.assert_called_once_with(["rm -f tests/test_new.py"])


def test_debug_eval_records_structured_grading_details():
    env, terminal = make_swe_bench_env(env_class=SWEBenchDebugEnv)
    env.setup_task()
    terminal.run.return_value = (True, f"{env.test_cmd}\nresults")

    with patch.dict(
        MAP_REPO_TO_PARSER,
        {
            "astropy/astropy": lambda content, test_spec: {
                "test_f2p": SWETestStatus.PASSED.value,
            }
        },
    ):
        eval_output = env.eval()

    terminal.run.assert_called_once_with(
        env.entrypoint,
        timeout=env.run_timeout,
    )
    assert eval_output.details is not None
    assert eval_output.details.reward == 1


def test_eval_binary_reward_requires_pass_to_pass_tests():
    env, _ = make_swe_bench_env(FAIL_TO_PASS=["test_f2p", "test_f2p_2"])
    env.setup_task()

    with patch.dict(
        MAP_REPO_TO_PARSER,
        {
            "astropy/astropy": lambda content, test_spec: {
                "test_f2p": SWETestStatus.PASSED.value,
                "test_f2p_2": SWETestStatus.PASSED.value,
                "test_p2p": SWETestStatus.FAILED.value,
            }
        },
    ):
        details = env._calculate_eval_details(marked_test_output())

    assert details.reward == 0
    assert env.calculate_max_score(MagicMock()) == 1


@pytest.if_docker_running
def test_instructions(get_swe_bench_env):
    env = get_swe_bench_env()
    assert env.instructions == env.task_data["problem_statement"]


@pytest.if_docker_running
def test_reset_and_step(get_swe_bench_env):
    env = get_swe_bench_env()
    env.add_tool(Toolbox.get_tool("eval"))
    env_info = env.reset()

    assert env.instructions == env_info.step_observation.observation
    assert "short test summary info" in env_info.eval_observation.observation
    assert env_info.score == env.score == 0
    assert env_info.max_score == env.max_score == 1
    assert len(env.fail_to_pass) == 1
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
    # Check that expected files are present in the listing
    # Hidden files (like .git/) now appear first, followed by the original expected files
    listdir_output = env_info.step_observation.observation
    assert listdir_output.startswith(f"{env.working_dir}/")
    assert (
        ".git/" in listdir_output
    ), "Expected hidden .git/ directory in listdir output"
    # Verify the expected file listing format (after hidden files)
    listdir_expected = """|-- CHANGES.rst
|-- CITATION
|-- CODE_OF_CONDUCT.md
|-- CONTRIBUTING.md
|-- GOVERNANCE.md
|-- LICENSE.rst
|-- MANIFEST.in
|-- README.rst
|-- astropy/
|-- astropy.egg-info/
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
    assert listdir_expected in listdir_output


def test_load_dataset():
    dataset = SWEBenchEnv.load_dataset()
    task_name = "astropy__astropy-14096"
    assert task_name in dataset

    task_data = next(iter(dataset.values()))
    assert sorted(task_data.keys()) == sorted(
        [
            "repo",
            "env_type",
            "instance_id",
            "base_commit",
            "patch",
            "test_patch",
            "problem_statement",
            "hints_text",
            "created_at",
            "difficulty",
            "version",
            "FAIL_TO_PASS",
            "PASS_TO_PASS",
            "environment_setup_commit",
        ]
    )


@pytest.if_docker_running
def test_setup_task(get_swe_bench_env):
    env = get_swe_bench_env()
    task_name = "astropy__astropy-14096"
    assert env.task_name == task_name
    env.setup_task()
    assert env.repo == "astropy/astropy"
    assert env.version == "5.1"
    assert env.package_name == "astropy"
    assert env.base_image == (
        "swebench/sweb.eval.x86_64.astropy_1776_astropy-14096:latest"
    )


@pytest.if_docker_running
def test_setup_terminal(get_swe_bench_env):
    env = get_swe_bench_env()
    task_name = "astropy__astropy-14096"
    env.reset()
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
    env.reset()

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
    env_info = env.reset()

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
    # Provide a minimal agent config for the SolutionAgent run.
    config = {
        "output_path": str(tmp_path),
        "random_seed": 0,
        # Optional values that BaseAgent.run would use; harmless to include here.
        "max_steps": 1,
    }
    for tool_name in ["pdb", "submit"]:
        env.add_tool(Toolbox.get_tool(tool_name))
    agent = AgentSolution(agent_args=config, llm=None, logger=env.logger)
    result = agent.run(env)
    assert result["success"]


@pytest.if_docker_running
def test_debug_entrypoint_contains_pdb(get_swe_bench_env):
    """Ensure the environment's debug_entrypoint includes '-m pdb' for interactive debugging."""
    env = get_swe_bench_env()
    env.reset()
    assert (
        "python -m pdb" in env.debug_entrypoint
    ), f"Expected '-m pdb' in debug_entrypoint, got: {env.debug_entrypoint}"


@pytest.if_docker_running
def test_setup_terminal_debug_mode(get_swe_bench_debug_env):
    env = get_swe_bench_debug_env()
    task_name = "astropy__astropy-14096"
    env.reset()
    _, git_logs = env.terminal.run("git log -n 4")
    assert env.base_commit in git_logs
    assert f"Applying test patch for {task_name}" in git_logs

    _, git_diff = env.terminal.run("git show HEAD", strip_output=False)
    git_diff = git_diff[git_diff.index("diff --git") :]


@pytest.if_docker_running
def test_running_solution_agent_in_debug_mode(get_swe_bench_debug_env, tmp_path):
    env = get_swe_bench_debug_env()
    # Provide a minimal agent config for the SolutionAgent run.
    config = {
        "output_path": str(tmp_path),
        "random_seed": 0,
        # Optional values that BaseAgent.run would use; harmless to include here.
        "max_steps": 1,
    }
    for tool_name in ["pdb", "eval", "submit"]:
        env.add_tool(Toolbox.get_tool(tool_name))
    agent = AgentSolution(agent_args=config, llm=None, logger=env.logger)
    result = agent.run(env)
    assert result["success"]
