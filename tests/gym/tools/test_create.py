from pathlib import Path

import pytest

from debug_gym.gym.envs.env import RepoEnv
from debug_gym.gym.tools import CreateTool


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

    with open(repo_path / "existingfile.py", "w") as f:
        f.write(file_content)

    env = RepoEnv(path=repo_path, dir_tree_depth=2)

    create_tool = CreateTool()
    env.add_tool(create_tool)

    env.reset()
    return env


def test_create_no_overwrite_error(env):
    create_tool = env.get_tool("create")
    patch = {
        "path": "existingfile.py",
        "content": "print('Hello, world!')",
    }
    obs = create_tool.use(env, **patch)
    assert obs.source == "create"
    assert (
        obs.observation
        == "Create failed. Error message:\nFile already exists. To overwrite, Please specify overwrite=True.\n"
    )


def test_overwrite_success(env):
    create_tool = env.get_tool("create")
    patch = {
        "path": "existingfile.py",
        "content": "print('Hello, world!')",
        "overwrite": True,
    }
    obs = create_tool.use(env, **patch)

    assert create_tool.create_success
    assert obs.observation == (
        "The file `existingfile.py` has been created successfully.\n"
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
        "+print('Hello, world!')"
    )
    with open(env.working_dir / "existingfile.py", "r") as f:
        new_content = f.read()
    assert new_content == "print('Hello, world!')"


def test_create_with_file_path(env):
    create_tool = env.get_tool("create")
    patch = {
        "path": "newdir/newfile.py",
        "content": "print('Hello, world!')",
    }
    obs = create_tool.use(env, **patch)

    assert create_tool.create_success
    assert obs.observation == (
        "The file `newdir/newfile.py` has been created successfully.\n"
        "\n"
        "Diff:\n"
        "\n"
        "--- original\n"
        "+++ current\n"
        "@@ -0,0 +1 @@\n"
        "+print('Hello, world!')"
    )
    with open(env.working_dir / "newdir/newfile.py", "r") as f:
        new_content = f.read()
    assert new_content == "print('Hello, world!')"


def test_create_no_path_error(env):
    create_tool = env.get_tool("create")
    patch = {
        "path": None,
        "content": "print('Hello, world!')",
    }
    obs = create_tool.use(env, **patch)
    assert obs.source == "create"
    assert obs.observation == "Create failed. Error message:\nFile path is None.\n"


def test_overwrite_readonly_file_error(env):
    # overwrite the is_editable method to simulate a read-only file
    env.workspace.is_editable = lambda x: x != "existingfile.py"
    create_tool = env.get_tool("create")
    patch = {
        "path": "existingfile.py",
        "content": "    print(f'Hello, {name}!')",
        "overwrite": True,
    }

    obs = create_tool.use(env, **patch)
    assert obs.observation == (
        "Create failed. Error message:\n`existingfile.py` is not editable.\n"
    )


def test_ignorable_file_error(env):
    # overwrite the _is_ignored_func method to simulate an ignored file
    env.workspace._is_ignored_func = lambda x: x.name == "ignoredfile.py"
    create_tool = env.get_tool("create")
    patch = {
        "path": "ignoredfile.py",
        "content": "    print(f'Hello, {name}!')",
    }

    obs = create_tool.use(env, **patch)
    assert obs.observation == (
        "Create failed. Error message:\n`ignoredfile.py` is ignored by the ignore patterns and cannot be created.\n"
    )


def test_create_with_newlines(env):
    create_tool = env.get_tool("create")
    patch = {
        "path": "existingfile.py",
        "content": "    print(f'Hello, {name}!')\n    print(f'Hello #2!')",
        "overwrite": True,
    }

    obs = create_tool.use(env, **patch)

    assert create_tool.create_success
    with open(env.working_dir / "existingfile.py", "r") as f:
        new_content = f.read()

    assert new_content == ("    print(f'Hello, {name}!')\n" "    print(f'Hello #2!')")
