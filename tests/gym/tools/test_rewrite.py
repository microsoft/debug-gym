from pathlib import Path

import pytest

from debug_gym.gym.envs.env import RepoEnv
from debug_gym.gym.tools.rewrite import RewriteTool


@pytest.fixture
def env(tmp_path):
    tmp_path = Path(tmp_path)
    repo_path = tmp_path / "repo"
    repo_path.mkdir()

    file_content = """import abc

def greet():
    print('Hello, world!')
    print('Goodbye, world!')
"""

    with open(repo_path / "test.py", "w") as f:
        f.write(file_content)

    env = RepoEnv(path=repo_path, dir_tree_depth=2)

    rewrite_tool = RewriteTool()
    env.add_tool(rewrite_tool)

    env.reset()
    return env


def test_rewrite_no_path_error(env):
    rewrite_tool = env.get_tool("rewrite")
    patch = {
        "path": None,
        "start": 4,
        "end": None,
        "new_code": "    print(f'Hello, {name}!')",
    }
    with pytest.raises(ValueError):
        rewrite_tool.use(env, **patch)


def test_rewrite_with_file_path(env):
    rewrite_tool = env.get_tool("rewrite")

    patch = {
        "path": "test.py",
        "start": 4,
        "end": None,
        "new_code": "    print(f'Hello, {name}!')",
    }
    obs = rewrite_tool.use(env, **patch)

    assert rewrite_tool.rewrite_success
    # using \n to prevent ide from removing trailing spaces
    assert (
        obs.observation
        == """Rewrite was successful. The file has been updated.

Diff:

--- original
+++ current
@@ -1,5 +1,5 @@
 import abc
 \n def greet():
-    print('Hello, world!')
+    print(f'Hello, {name}!')
     print('Goodbye, world!')
"""
    )
    with open(env.working_dir / "test.py", "r") as f:
        new_content = f.read()
    assert (
        new_content
        == """import abc

def greet():
    print(f'Hello, {name}!')
    print('Goodbye, world!')
"""
    )


def test_rewrite_start_end(env):
    rewrite_tool = env.get_tool("rewrite")

    patch = {
        "path": "test.py",
        "start": 4,
        "end": 5,
        "new_code": "    print(f'Hello, {name}!')",
    }
    obs = rewrite_tool.use(env, **patch)

    assert rewrite_tool.rewrite_success
    # using \n to prevent ide from removing trailing spaces
    assert (
        obs.observation
        == """Rewrite was successful. The file has been updated.

Diff:

--- original
+++ current
@@ -1,5 +1,4 @@
 import abc
 \n def greet():
-    print('Hello, world!')
-    print('Goodbye, world!')
+    print(f'Hello, {name}!')
"""
    )
    with open(env.working_dir / "test.py", "r") as f:
        new_content = f.read()
    assert (
        new_content
        == """import abc

def greet():
    print(f'Hello, {name}!')
"""
    )


def test_full_rewrite(env):
    rewrite_tool = env.get_tool("rewrite")

    patch = {
        "path": "test.py",
        "new_code": "print(f'Hello, {name}!')",
    }
    obs = rewrite_tool.use(env, **patch)

    assert rewrite_tool.rewrite_success
    # using \n to prevent ide from removing trailing spaces
    assert (
        obs.observation
        == """Rewrite was successful. The file has been updated.

Diff:

--- original
+++ current
@@ -1,5 +1 @@
-import abc
-
-def greet():
-    print('Hello, world!')
-    print('Goodbye, world!')
+print(f'Hello, {name}!')"""
    )
    with open(env.working_dir / "test.py", "r") as f:
        new_content = f.read()
    assert new_content == """print(f'Hello, {name}!')"""


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
