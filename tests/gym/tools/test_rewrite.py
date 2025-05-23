from pathlib import Path

import pytest

from debug_gym.gym.envs.env import RepoEnv
from debug_gym.gym.tools.rewrite import RewriteTool


@pytest.fixture
def env(tmp_path):
    tmp_path = Path(tmp_path)
    repo_path = tmp_path / "repo"
    repo_path.mkdir()

    with open(repo_path / "test.py", "w") as f:
        f.write("def greet():\n    print('Hello, world!')\n")

    env = RepoEnv(path=repo_path, dir_tree_depth=2)

    rewrite_tool = RewriteTool()
    env.add_tool(rewrite_tool)

    env.reset()
    env.load_current_file("test.py")
    return env


def test_rewrite(env):
    rewrite_tool = env.get_tool("rewrite")
    patch = {
        "path": None,
        "start": 2,
        "end": None,
        "new_code": "    print(f'Hello, {name}!')",
    }
    obs = rewrite_tool.use(env, **patch)

    assert obs.observation == "Rewrite successful. The file has been modified."
    assert rewrite_tool.rewrite_success
    with open(env.working_dir / "test.py", "r") as f:
        new_content = f.read()
    assert new_content == "def greet():\n    print(f'Hello, {name}!')\n"


def test_rewrite_with_file_path(env):
    rewrite_tool = env.get_tool("rewrite")

    patch = {
        "path": "test.py",
        "start": 2,
        "end": None,
        "new_code": "    print(f'Hello, {name}!')",
    }
    obs = rewrite_tool.use(env, **patch)

    assert obs.observation == "Rewrite successful. The file has been modified."
    assert rewrite_tool.rewrite_success
    with open(env.working_dir / "test.py", "r") as f:
        new_content = f.read()
    assert new_content == "def greet():\n    print(f'Hello, {name}!')\n"


def test_rewrite_invalid_file(env):
    rewrite_tool = env.get_tool("rewrite")
    env.all_files = ["another_file.py"]

    patch = {
        "path": "test.py",
        "start": 2,
        "end": None,
        "new_code": "    print(f'Hello, {name}!')",
    }
    obs = rewrite_tool.use(env, **patch)

    assert (
        obs.observation
        == "Error while rewriting the file: File test.py does not exist or is not in the current repository.\nRewrite failed."
    )
    assert not rewrite_tool.rewrite_success


def test_rewrite_invalid_line_number(env):
    rewrite_tool = env.get_tool("rewrite")

    patch = {
        "path": "test.py",
        "start": 0,
        "end": None,
        "new_code": "    print(f'Hello, {name}!')",
    }
    obs = rewrite_tool.use(env, **patch)

    assert (
        obs.observation
        == "Invalid line number, line numbers are 1-based.\nRewrite failed."
    )
    assert not rewrite_tool.rewrite_success


def test_rewrite_invalid_line_number_2(env):
    rewrite_tool = env.get_tool("rewrite")

    patch = {
        "path": "test.py",
        "start": 12,
        "end": 4,
        "new_code": "    print(f'Hello, {name}!')",
    }
    obs = rewrite_tool.use(env, **patch)

    assert (
        obs.observation
        == "Invalid line number range, start should be less than or equal to end.\nRewrite failed."
    )
    assert not rewrite_tool.rewrite_success
