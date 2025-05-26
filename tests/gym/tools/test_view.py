from pathlib import Path

import pytest

from debug_gym.gym.entities import Observation
from debug_gym.gym.envs.env import RepoEnv
from debug_gym.gym.tools.tool import ToolCall
from debug_gym.gym.tools.toolbox import Toolbox


@pytest.fixture
def env(tmp_path):
    tmp_path = Path(tmp_path)
    repo_path = tmp_path / "repo"
    repo_path.mkdir()

    with open(repo_path / "main.py", "w") as f:
        f.write("print('Hello, World!')")

    with open(repo_path / "test_1.py", "w") as f:
        f.write("def test_1():\n  assert False\n")

    with open(repo_path / ".debugreadonly", "w") as f:
        f.write("test_1.py")

    env = RepoEnv(path=repo_path, dir_tree_depth=2)
    view_tool = Toolbox.get_tool("view")
    env.add_tool(view_tool)
    env.reset()
    return env


def test_view_valid_file(env):
    view_call = ToolCall(id="view_id", name="view", arguments={"path": "main.py"})
    env_info = env.step(view_call)

    assert env_info.step_observation.source == "view"
    assert (
        env_info.step_observation.observation
        == "Viewing `main.py`:\n\n```\nprint('Hello, World!')\n```\n\n"
    )

    view_call = ToolCall(
        id="view_id", name="view", arguments={"path": str(env.working_dir / "main.py")}
    )
    env_info_2 = env.step(view_call)
    assert env_info_2.step_observation == env_info.step_observation


def test_view_valid_read_only_file(env):
    view_call = ToolCall(id="view_id", name="view", arguments={"path": "test_1.py"})
    env_info = env.step(view_call)

    assert env_info.step_observation.source == "view"
    assert (
        env_info.step_observation.observation
        == "Viewing `test_1.py` (read-only):\n\n```\ndef test_1():\n  assert False\n\n```\n\n"
    )


def test_view_invalid_file_empty(env):
    view_call = ToolCall(id="view_id", name="view", arguments={"path": ""})
    env_info = env.step(view_call)
    assert env_info.step_observation == Observation(
        source="view",
        observation="Invalid file path. Please specify a valid file path.",
    )


def test_view_invalid_file_not_in_working_dir(env):
    view_call = ToolCall(
        id="view_id", name="view", arguments={"path": "/nonexistent/main.py"}
    )
    env_info = env.step(view_call)
    assert env_info.step_observation == Observation(
        source="view",
        observation=(
            "Invalid file path. The file path must be inside "
            f"the root directory: `{env.working_dir}`."
        ),
    )


def test_view_invalid_file_do_not_exist(env):
    view_call = ToolCall(
        id="view_id",
        name="view",
        arguments={"path": f"{env.working_dir}/nonexistent.py"},
    )
    env_info = env.step(view_call)
    assert env_info.step_observation == Observation(
        source="view",
        observation=(
            "File not found. Could not navigate to `nonexistent.py`. "
            f"Make sure that the file path is given relative to the root: `{env.working_dir}`."
        ),
    )
