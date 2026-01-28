from pathlib import Path

import pytest

from debug_gym.gym.envs.local import LocalEnv
from debug_gym.gym.tools.edit import EditTool


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

    env = LocalEnv(path=repo_path)

    edit_tool = EditTool()
    env.add_tool(edit_tool)

    env.reset()
    return env


def test_edit_no_path_error(env):
    edit_tool = env.get_tool("edit")
    patch = {
        "path": None,
        "start": 4,
        "end": None,
        "new_code": "    print(f'Hello, {name}!')",
    }
    obs = edit_tool.use(env, **patch)
    assert obs.source == "edit"
    assert obs.observation == "Edit failed. Error message:\nFile path is None.\n"


def test_edit_with_file_path(env):
    edit_tool = env.get_tool("edit")
    patch = {
        "path": "test.py",
        "start": 4,
        "end": None,
        "new_code": "    print(f'Hello, {name}!')",
    }
    obs = edit_tool.use(env, **patch)

    assert edit_tool.edit_success
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


def test_edit_start_end(env):
    edit_tool = env.get_tool("edit")

    patch = {
        "path": "test.py",
        "start": 4,
        "end": 5,
        "new_code": "    print(f'Hello, {name}!')",
    }
    obs = edit_tool.use(env, **patch)

    assert edit_tool.edit_success
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


def test_full_edit(env):
    edit_tool = env.get_tool("edit")
    patch = {
        "path": "test.py",
        "new_code": "print(f'Hello, {name}!')",
    }
    obs = edit_tool.use(env, **patch)

    assert edit_tool.edit_success
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
    assert new_content == "print(f'Hello, {name}!')"


def test_edit_invalid_file(env):
    edit_tool = env.get_tool("edit")
    patch = {
        "path": "another_file.py",
        "start": 2,
        "end": None,
        "new_code": "    print(f'Hello, {name}!')",
    }
    obs = edit_tool.use(env, **patch)
    assert edit_tool.edit_success
    assert (env.working_dir / "another_file.py").exists()
    assert env.workspace.read_file("another_file.py") == "    print(f'Hello, {name}!')"


def test_edit_invalid_line_number(env):
    edit_tool = env.get_tool("edit")

    patch = {
        "path": "test.py",
        "start": 0,
        "end": None,
        "new_code": "    print(f'Hello, {name}!')",
    }
    obs = edit_tool.use(env, **patch)

    assert obs.observation == (
        "Edit failed. Error message:\n"
        "Invalid line number, line numbers are 1-based.\n"
    )
    assert not edit_tool.edit_success


def test_edit_invalid_line_number_2(env):
    edit_tool = env.get_tool("edit")

    patch = {
        "path": "test.py",
        "start": 12,
        "end": 4,
        "new_code": "    print(f'Hello, {name}!')",
    }
    obs = edit_tool.use(env, **patch)

    assert obs.observation == (
        "Edit failed. Error message:\n"
        "Invalid line number range, start should be less than or equal to end.\n"
    )
    assert not edit_tool.edit_success


def test_edit_with_newlines(env):
    edit_tool = env.get_tool("edit")
    patch = {
        "path": "test.py",
        "start": 4,
        "end": None,
        "new_code": "    print(f'Hello, {name}!')\n    print(f'Hello #2!')",
    }

    obs = edit_tool.use(env, **patch)

    assert edit_tool.edit_success
    # using \n to prevent ide from removing trailing spaces
    assert obs.observation == (
        "The file `test.py` has been updated successfully.\n"
        "\n"
        "Diff:\n"
        "\n"
        "--- original\n"
        "+++ current\n"
        "@@ -1,5 +1,6 @@\n"
        " import abc\n"
        " \n"
        " def greet():\n"
        "-    print('Hello, world!')\n"
        "+    print(f'Hello, {name}!')\n"
        "+    print(f'Hello #2!')\n"
        "     print('Goodbye, world!')\n"
    )
    with open(env.working_dir / "test.py", "r") as f:
        new_content = f.read()

    assert new_content == (
        "import abc\n"
        "\n"
        "def greet():\n"
        "    print(f'Hello, {name}!')\n"
        "    print(f'Hello #2!')\n"
        "    print('Goodbye, world!')\n"
    )


def test_edit_new_file(env):
    """Ensure the edit tool can create a brand new file when it does not already exist."""
    edit_tool = env.get_tool("edit")
    filename = "new_dir/nested/new_module.py"
    assert not (env.working_dir / filename).exists()

    patch = {
        "path": filename,
        "new_code": "def added():\n    return 'created'\n",
    }
    obs = edit_tool.use(env, **patch)

    assert edit_tool.edit_success, f"Edit failed: {obs.observation}"
    # We don't assert the entire diff (more brittle); just key substrings.
    assert f"The file `{filename}` has been updated successfully." in obs.observation
    assert "def added():" in obs.observation

    with open(env.working_dir / filename, "r") as f:
        content = f.read()
    assert content == "def added():\n    return 'created'\n"
