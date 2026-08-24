from pathlib import Path

import pytest

from debug_gym.gym.entities import Event
from debug_gym.gym.envs.local import LocalEnv
from debug_gym.gym.tools.tool import ToolCall
from debug_gym.gym.tools.toolbox import Toolbox


@pytest.fixture
def env(tmp_path):
    tmp_path = Path(tmp_path)
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    with open(repo_path / "test_1.py", "w") as f:
        f.write("def test_1():\n  assert False\n")

    env = LocalEnv(path=repo_path)
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


def test_eval_does_not_auto_run_on_edit(env):
    eval_tool = Toolbox.get_tool("eval")
    env.add_tool(eval_tool)

    eval_call = ToolCall(id="eval_id", name="eval", arguments={})
    env_info = env.step(eval_call)
    assert env_info.step_observation.source == "eval"
    assert "FAILED test_1.py::test_1" in env_info.step_observation.observation
    failing_output = env.last_eval.output

    with open(env.working_dir / "test_1.py", "w") as f:
        f.write("def test_1():\n  assert True\n")

    env.queue_event(Event.EDIT_SUCCESS, source=None)
    env.process_events()

    assert env.last_eval.output == failing_output
