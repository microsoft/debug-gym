from pathlib import Path

import pytest

from debug_gym.gym.envs.env import RepoEnv
from debug_gym.gym.tools.rewrite import RewriteTool


@pytest.fixture
def env(tmp_path):
    tmp_path = Path(tmp_path)
    repo_path = tmp_path / "repo"
    repo_path.mkdir()

    file_content = (
        "import abc\n"
        "\n"
        "def greet():\n"
        "    print('Hello, world!')\n"
        "    print('Goodbye, world!')\n"
    )

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
    obs = rewrite_tool.use(env, **patch)
    assert obs.source == "rewrite"
    assert obs.observation == "Rewrite failed. Error message:\nFile path is None.\n"


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
    assert obs.observation == (
        "The file `test.py` has been updated successfully.\n"
        "\n"
        "Diff:\n"
        "\n"
        "--- original\n"
        "+++ current\n"
        "@@ -1,5 +1,5 @@\n"
        " import abc\n"
        " \n"
        " def greet():\n"
        "-    print('Hello, world!')\n"
        "+    print(f'Hello, {name}!')\n"
        "     print('Goodbye, world!')\n"
    )
    with open(env.working_dir / "test.py", "r") as f:
        new_content = f.read()
    assert new_content == (
        "import abc\n"
        "\n"
        "def greet():\n"
        "    print(f'Hello, {name}!')\n"
        "    print('Goodbye, world!')\n"
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
    assert obs.observation == (
        "The file `test.py` has been updated successfully.\n"
        "\n"
        "Diff:\n"
        "\n"
        "--- original\n"
        "+++ current\n"
        "@@ -1,5 +1,4 @@\n"
        " import abc\n"
        " \n def greet():\n"
        "-    print('Hello, world!')\n"
        "-    print('Goodbye, world!')\n"
        "+    print(f'Hello, {name}!')\n"
    )
    with open(env.working_dir / "test.py", "r") as f:
        new_content = f.read()
    assert new_content == ("import abc\n\ndef greet():\n    print(f'Hello, {name}!')\n")


def test_full_rewrite(env):
    rewrite_tool = env.get_tool("rewrite")
    # env.editable_files = [f"{env.working_dir}/test.py"]
    patch = {
        "path": "test.py",
        "new_code": "print(f'Hello, {name}!')",
    }
    obs = rewrite_tool.use(env, **patch)

    assert rewrite_tool.rewrite_success
    assert obs.observation == (
        "The file `test.py` has been updated successfully.\n"
        "\n"
        "Diff:\n"
        "\n"
        "--- original\n"
        "+++ current\n"
        "@@ -1,5 +1 @@\n"
        "-import abc\n"
        "-\n"
        "-def greet():\n"
        "-    print('Hello, world!')\n"
        "-    print('Goodbye, world!')\n"
        "+print(f'Hello, {name}!')"
    )
    with open(env.working_dir / "test.py", "r") as f:
        new_content = f.read()
    assert new_content == """print(f'Hello, {name}!')"""


def test_rewrite_invalid_file(env):
    # overwrite the is_editable method to simulate a read-only file
    env.is_editable = lambda x: x != "read_only.py"
    rewrite_tool = env.get_tool("rewrite")
    patch = {
        "path": "another_file.py",
        "start": 2,
        "end": None,
        "new_code": "    print(f'Hello, {name}!')",
    }
    obs = rewrite_tool.use(env, **patch)
    assert obs.observation == (
        "Rewrite failed. Error message:\n"
        "File `another_file.py` does not exist or is not "
        f"in the working directory `{env.working_dir}`.\n"
    )

    patch["path"] = "read_only.py"
    obs = rewrite_tool.use(env, **patch)
    assert obs.observation == (
        "Rewrite failed. Error message:\nFile `read_only.py` is not editable.\n"
    )


def test_rewrite_invalid_line_number(env):
    rewrite_tool = env.get_tool("rewrite")

    patch = {
        "path": "test.py",
        "start": 0,
        "end": None,
        "new_code": "    print(f'Hello, {name}!')",
    }
    obs = rewrite_tool.use(env, **patch)

    assert obs.observation == (
        "Rewrite failed. Error message:\n"
        "Invalid line number, line numbers are 1-based.\n"
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

    assert obs.observation == (
        "Rewrite failed. Error message:\n"
        "Invalid line number range, start should be less than or equal to end.\n"
    )
    assert not rewrite_tool.rewrite_success
