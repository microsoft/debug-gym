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
    "method,env_auto_eval_on_rewrite,expected",
    [
        ("on_rewrite_success", True, "1 passed in "),
        ("on_rewrite_success", False, "FAILED test_1.py::test_1"),
    ],
)
def test_eval_on_event(env, method, env_auto_eval_on_rewrite, expected):
    eval_tool = Toolbox.get_tool("eval")
    env.add_tool(eval_tool)
    env.auto_eval_on_rewrite = env_auto_eval_on_rewrite

    eval_call = ToolCall(id="eval_id", name="eval", arguments={})
    env_info = env.step(eval_call)
    assert env_info.step_observation.source == "eval"
    assert "FAILED test_1.py::test_1" in env_info.step_observation.observation

    # Edit test file to pass. If eval is called, env.done is set to True
    with open(env.working_dir / "test_1.py", "w") as f:
        f.write("def test_1():\n  assert True\n")

    getattr(eval_tool, method)(env, random_arg="random_arg")
    assert expected in env.last_eval.output
