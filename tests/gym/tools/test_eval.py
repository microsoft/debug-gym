from pathlib import Path

import pytest

from debug_gym.gym.envs.env import RepoEnv
from debug_gym.gym.tools.tool import ToolCall
from debug_gym.gym.tools.toolbox import Toolbox


@pytest.fixture
def env(tmp_path):
    tmp_path = Path(tmp_path)
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    with open(repo_path / "test_1.py", "w") as f:
        f.write("def test_1():\n  assert False\n")

    env = RepoEnv(path=repo_path, dir_tree_depth=2)
    env.reset()
    return env


def test_eval(env):
    eval_tool = Toolbox.get_tool("eval")
    env.add_tool(eval_tool)

    eval_call = ToolCall(id="eval_id", name="eval", arguments={})
    env_info = env.step(eval_call)

    assert env_info.step_observation.source == "eval"
    assert "FAILED test_1.py::test_1" in env_info.step_observation.observation

    with open(env.working_dir / "test_1.py", "w") as f:
        f.write("def test_1():\n  assert True\n")
    env_info = env.step(eval_call)
    assert env_info.step_observation.source == "eval"
    assert "1 passed in " in env_info.step_observation.observation


@pytest.mark.parametrize(
    "method,auto_eval_on_rewrite,expected",
    [
        ("on_rewrite_success", True, "1 passed in "),
        ("on_rewrite_success", False, "FAILED test_1.py::test_1"),
    ],
)
def test_eval_on_event(env, method, auto_eval_on_rewrite, expected):
    eval_tool = Toolbox.get_tool("eval", auto_eval_on_rewrite=auto_eval_on_rewrite)
    env.add_tool(eval_tool)

    eval_call = ToolCall(id="eval_id", name="eval", arguments={})
    env_info = env.step(eval_call)
    assert env_info.step_observation.source == "eval"
    assert "FAILED test_1.py::test_1" in env_info.step_observation.observation

    # Edit test file to pass. If eval is called, env.terminated is set to True
    with open(env.working_dir / "test_1.py", "w") as f:
        f.write("def test_1():\n  assert True\n")

    getattr(eval_tool, method)(env, random_arg="random_arg")
    assert expected in env.last_eval.output


def test_eval_tool_auto_eval_on_rewrite_respects_default(env):
    """Test that when EvalTool's auto_eval_on_rewrite is None, it uses the correct default (False)."""
    # Tool init without setting auto_eval_on_rewrite
    eval_tool_default = Toolbox.get_tool("eval")
    env.add_tool(eval_tool_default)

    eval_call = ToolCall(id="eval_id", name="eval", arguments={})
    env_info = env.step(eval_call)
    assert "FAILED test_1.py::test_1" in env_info.step_observation.observation

    # Edit test file to pass
    with open(env.working_dir / "test_1.py", "w") as f:
        f.write("def test_1():\n  assert True\n")

    # Tool's None should defer to env's True setting
    eval_tool_default.on_rewrite_success(env)
    assert "FAILED test_1.py::test_1" in env.last_eval.output
