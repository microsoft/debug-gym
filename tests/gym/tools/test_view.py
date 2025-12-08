from pathlib import Path

import pytest

from debug_gym.gym.entities import Observation
from debug_gym.gym.envs.local import LocalEnv
from debug_gym.gym.tools.tool import ToolCall
from debug_gym.gym.tools.toolbox import Toolbox


@pytest.fixture
def env(tmp_path):
    tmp_path = Path(tmp_path)
    repo_path = tmp_path / "repo"
    repo_path.mkdir()

    with open(repo_path / "main.py", "w") as f:
        f.write("print('Hello, World!')")

    with open(repo_path / "ten_lines.py", "w") as f:
        for i in range(10):
            f.write(f"print('Line {i + 1}')\n")

    with open(repo_path / "test_1.py", "w") as f:
        f.write("def test_1():\n  assert False")

    with open(repo_path / ".debugreadonly", "w") as f:
        f.write("test_1.py")

    (repo_path / "empty.py").touch()  # Create an empty file

    env = LocalEnv(path=repo_path)
    view_tool = Toolbox.get_tool("view")
    env.add_tool(view_tool)
    env.reset()
    return env


def test_view_valid_file(env):
    view_call = ToolCall(
        id="view_id",
        name="view",
        arguments={"path": "main.py", "include_line_numbers_and_breakpoints": False},
    )
    env_info = env.step(view_call)

    assert env_info.step_observation.source == "view"
    assert env_info.step_observation.observation == (
        "Viewing `main.py`, lines 1-1 of 1 total lines.\n"
        "\n"
        "```\n"
        "print('Hello, World!')\n"
        "```\n"
        "\n"
    )

    abs_path = str(env.working_dir / "main.py")
    view_call = ToolCall(
        id="view_id",
        name="view",
        arguments={
            "path": abs_path,
            "include_line_numbers_and_breakpoints": False,
        },
    )
    env_info_2 = env.step(view_call)
    assert env_info_2.step_observation.observation == (
        f"Viewing `{abs_path}`, lines 1-1 of 1 total lines.\n"
        "\n"
        "```\n"
        "print('Hello, World!')\n"
        "```\n"
        "\n"
    )


def test_view_valid_file_with_line_numbers_no_breakpoints(env):
    view_call = ToolCall(id="view_id", name="view", arguments={"path": "main.py"})
    env_info = env.step(view_call)

    assert env_info.step_observation.source == "view"
    assert env_info.step_observation.observation == (
        "Viewing `main.py`, lines 1-1 of 1 total lines.\n"
        "\n"
        "```\n"
        "     1 print('Hello, World!')\n"
        "```\n"
        "\n"
    )

    abs_path = str(env.working_dir / "main.py")
    view_call = ToolCall(id="view_id", name="view", arguments={"path": abs_path})
    env_info_2 = env.step(view_call)
    assert env_info_2.step_observation.observation == (
        f"Viewing `{env.working_dir}/main.py`, lines 1-1 of 1 total lines.\n"
        "\n"
        "```\n"
        "     1 print('Hello, World!')\n"
        "```\n"
        "\n"
    )


def test_view_valid_file_with_line_numbers_and_breakpoints(env):
    # set a breakpoint at line 1 of main.py
    abs_path = str(env.working_dir / "main.py")
    env.current_breakpoints_state[f"{abs_path}|||1"] = "b main.py|||1"
    view_call = ToolCall(id="view_id", name="view", arguments={"path": "main.py"})
    env_info = env.step(view_call)

    assert env_info.step_observation.source == "view"
    assert env_info.step_observation.observation == (
        "Viewing `main.py`, lines 1-1 of 1 total lines. B indicates breakpoint before a certain line of code.\n"
        "\n"
        "```\n"
        "B    1 print('Hello, World!')\n"
        "```\n"
        "\n"
    )

    view_call = ToolCall(id="view_id", name="view", arguments={"path": abs_path})
    env_info_2 = env.step(view_call)
    assert env_info_2.step_observation.observation == (
        f"Viewing `{abs_path}`, lines 1-1 of 1 total lines. B indicates breakpoint before a certain line of code.\n"
        "\n"
        "```\n"
        "B    1 print('Hello, World!')\n"
        "```\n"
        "\n"
    )


def test_view_valid_read_only_file(env):
    view_call = ToolCall(
        id="view_id",
        name="view",
        arguments={"path": "test_1.py", "include_line_numbers_and_breakpoints": False},
    )
    env_info = env.step(view_call)

    assert env_info.step_observation.source == "view"
    assert env_info.step_observation.observation == (
        "Viewing `test_1.py`, lines 1-2 of 2 total lines. The file is read-only.\n"
        "\n"
        "```\n"
        "def test_1():\n"
        "  assert False\n"
        "```\n"
        "\n"
    )


def test_view_valid_read_only_file_with_line_numbers_no_breakpoints(env):
    view_call = ToolCall(id="view_id", name="view", arguments={"path": "test_1.py"})
    env_info = env.step(view_call)
    assert env_info.step_observation.source == "view"
    assert env_info.step_observation.observation == (
        "Viewing `test_1.py`, lines 1-2 of 2 total lines. The file is read-only.\n"
        "\n"
        "```\n"
        "     1 def test_1():\n"
        "     2   assert False\n"
        "```\n"
        "\n"
    )


def test_view_valid_read_only_file_with_line_numbers_and_breakpoints(env):
    # set a breakpoint at line 2 of test_1.py
    env.current_breakpoints_state[f"{env.working_dir}/test_1.py|||2"] = (
        "b test_1.py|||2"
    )
    view_call = ToolCall(id="view_id", name="view", arguments={"path": "test_1.py"})
    env_info = env.step(view_call)

    assert env_info.step_observation.source == "view"
    assert env_info.step_observation.observation == (
        "Viewing `test_1.py`, lines 1-2 of 2 total lines. The file is read-only. B indicates breakpoint before a certain line of code.\n"
        "\n"
        "```\n"
        "     1 def test_1():\n"
        "B    2   assert False\n"
        "```\n"
        "\n"
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
            "View failed. Error message:\n"
            "`/nonexistent/main.py` does not exist or "
            f"is not in the working directory `{env.working_dir}`."
        ),
    )


def test_view_invalid_file_do_not_exist(env):
    abs_file = str(env.working_dir / "nonexistent.py")
    view_call = ToolCall(
        id="view_id",
        name="view",
        arguments={"path": abs_file},
    )
    env_info = env.step(view_call)
    assert env_info.step_observation == Observation(
        source="view",
        observation=(
            "View failed. Error message:\n"
            f"`{abs_file}` does not exist or "
            f"is not in the working directory `{env.working_dir}`."
        ),
    )


def test_view_file_with_range_full_content(env):
    view_call = ToolCall(id="view_id", name="view", arguments={"path": "ten_lines.py"})
    env_info = env.step(view_call)
    assert env_info.step_observation.source == "view"
    assert (
        env_info.step_observation.observation
        == """Viewing `ten_lines.py`, lines 1-10 of 10 total lines.

```
     1 print('Line 1')
     2 print('Line 2')
     3 print('Line 3')
     4 print('Line 4')
     5 print('Line 5')
     6 print('Line 6')
     7 print('Line 7')
     8 print('Line 8')
     9 print('Line 9')
    10 print('Line 10')
```

"""
    )
    # Test min and max range
    view_call.arguments = {"path": "ten_lines.py", "start": 1, "end": 10}
    info2 = env.step(view_call)
    assert info2.step_observation.observation == env_info.step_observation.observation

    view_call.arguments = {"path": "ten_lines.py", "start": 1}
    info2 = env.step(view_call)
    assert info2.step_observation.observation == env_info.step_observation.observation

    view_call.arguments = {"path": "ten_lines.py", "end": 10}
    info2 = env.step(view_call)
    assert info2.step_observation.observation == env_info.step_observation.observation


def test_view_file_with_range_start_5(env):
    view_call = ToolCall(
        id="view_id",
        name="view",
        arguments={"path": "ten_lines.py", "start": 5},
    )
    env_info = env.step(view_call)
    assert env_info.step_observation.observation == (
        "Viewing `ten_lines.py`, lines 5-10 of 10 total lines.\n"
        "\n"
        "```\n"
        "     5 print('Line 5')\n"
        "     6 print('Line 6')\n"
        "     7 print('Line 7')\n"
        "     8 print('Line 8')\n"
        "     9 print('Line 9')\n"
        "    10 print('Line 10')\n"
        "```\n"
        "\n"
    )


def test_view_file_with_range_start_5_end_8(env):
    view_call = ToolCall(
        id="view_id",
        name="view",
        arguments={"path": "ten_lines.py", "start": 5, "end": 8},
    )
    env_info = env.step(view_call)
    assert env_info.step_observation.observation == (
        "Viewing `ten_lines.py`, lines 5-8 of 10 total lines.\n"
        "\n"
        "```\n"
        "     5 print('Line 5')\n"
        "     6 print('Line 6')\n"
        "     7 print('Line 7')\n"
        "     8 print('Line 8')\n"
        "```\n"
        "\n"
    )


def test_view_end_line_max(env):
    # adjust end to max number of lines in the file (10 lines)
    view_call = ToolCall(
        id="view_id",
        name="view",
        arguments={"path": "ten_lines.py", "start": 1, "end": 11},
    )
    env_info = env.step(view_call)
    assert env_info.step_observation.observation == (
        "Viewing `ten_lines.py`, lines 1-10 of 10 total lines.\n"
        "\n"
        "```\n"
        "     1 print('Line 1')\n"
        "     2 print('Line 2')\n"
        "     3 print('Line 3')\n"
        "     4 print('Line 4')\n"
        "     5 print('Line 5')\n"
        "     6 print('Line 6')\n"
        "     7 print('Line 7')\n"
        "     8 print('Line 8')\n"
        "     9 print('Line 9')\n"
        "    10 print('Line 10')\n"
        "```\n"
        "\n"
    )


def test_view_file_with_invalid_range(env):
    view_call = ToolCall(
        id="view_id",
        name="view",
        arguments={"path": "ten_lines.py", "start": 0, "end": 3},
    )
    env_info = env.step(view_call)
    assert (
        env_info.step_observation.observation
        == "Invalid start index: `0`. It should be between 1 and 10."
    )

    view_call.arguments = {"path": "ten_lines.py", "start": 5, "end": 4}
    env_info = env.step(view_call)
    assert (
        env_info.step_observation.observation
        == "Invalid range: start index `5` is greater than end index `4`."
    )


def test_view_empty_file(env):
    view_call = ToolCall(
        id="view_id",
        name="view",
        arguments={"path": "empty.py", "start": 1, "end": 3},
    )
    env_info = env.step(view_call)
    assert env_info.step_observation.observation == "The file `empty.py` is empty."
