from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from debug_gym.gym.entities import Observation
from debug_gym.gym.envs.env import RepoEnv
from debug_gym.gym.tools.bash import BashTool
from debug_gym.gym.tools.tool import ToolCall
from debug_gym.gym.tools.toolbox import Toolbox


@pytest.fixture
def env(tmp_path):
    """Create a test environment with a temporary repository."""
    tmp_path = Path(tmp_path)
    repo_path = tmp_path / "repo"
    repo_path.mkdir()

    # Create some test files
    with open(repo_path / "test_file.txt", "w") as f:
        f.write("Hello World\nLine 2\nLine 3")

    with open(repo_path / "script.py", "w") as f:
        f.write("print('Python script')")

    # Create a subdirectory with files
    subdir = repo_path / "subdir"
    subdir.mkdir()
    with open(subdir / "nested.txt", "w") as f:
        f.write("nested file content")

    env = RepoEnv(path=repo_path, dir_tree_depth=2)
    bash_tool = Toolbox.get_tool("bash")
    env.add_tool(bash_tool)
    env.reset()
    return env


@pytest.fixture
def bash_tool():
    """Create a BashTool instance."""
    return BashTool()


def test_bash_tool_registration():
    """Test that the bash tool is properly registered in the toolbox."""
    tool = Toolbox.get_tool("bash")
    assert isinstance(tool, BashTool)
    assert tool.name == "bash"


def test_bash_tool_metadata(bash_tool):
    """Test that the bash tool has the correct metadata."""
    assert bash_tool.name == "bash"
    assert "Run commands in a bash shell" in bash_tool.description
    assert "command" in bash_tool.arguments
    assert bash_tool.arguments["command"]["type"] == ["string"]
    assert len(bash_tool.examples) > 0


def test_bash_successful_command(env):
    """Test executing a successful bash command."""
    bash_call = ToolCall(
        id="bash_test", name="bash", arguments={"command": "echo 'Hello World'"}
    )
    env_info = env.step(bash_call)

    assert env_info.step_observation.source == "bash"
    assert "Hello World" in env_info.step_observation.observation


def test_bash_list_files(env):
    """Test listing files with bash command."""
    bash_call = ToolCall(id="bash_test", name="bash", arguments={"command": "ls -la"})
    env_info = env.step(bash_call)

    assert env_info.step_observation.source == "bash"
    observation = env_info.step_observation.observation
    assert "test_file.txt" in observation
    assert "script.py" in observation
    assert "subdir" in observation


def test_bash_file_content(env):
    """Test reading file content with bash command."""
    bash_call = ToolCall(
        id="bash_test", name="bash", arguments={"command": "cat test_file.txt"}
    )
    env_info = env.step(bash_call)

    assert env_info.step_observation.source == "bash"
    observation = env_info.step_observation.observation
    assert "Hello World" in observation
    assert "Line 2" in observation
    assert "Line 3" in observation


def test_bash_command_with_pipes(env):
    """Test bash command with pipes."""
    bash_call = ToolCall(
        id="bash_test",
        name="bash",
        arguments={"command": "cat test_file.txt | head -1"},
    )
    env_info = env.step(bash_call)

    assert env_info.step_observation.source == "bash"
    observation = env_info.step_observation.observation
    assert "Hello World" in observation
    assert "Line 2" not in observation  # Should only show first line


def test_bash_failed_command(env):
    """Test executing a failed bash command."""
    bash_call = ToolCall(
        id="bash_test", name="bash", arguments={"command": "ls non_existent_directory"}
    )
    env_info = env.step(bash_call)

    assert env_info.step_observation.source == "bash"
    observation = env_info.step_observation.observation
    assert "Command failed with output:" in observation
    assert "No such file or directory" in observation


def test_bash_empty_output_command(env):
    """Test command that produces no output."""
    bash_call = ToolCall(
        id="bash_test", name="bash", arguments={"command": "touch new_file.txt"}
    )
    env_info = env.step(bash_call)

    assert env_info.step_observation.source == "bash"
    observation = env_info.step_observation.observation
    assert "Command executed successfully (no output)" in observation


def test_bash_python_execution(env):
    """Test executing Python script with bash."""
    bash_call = ToolCall(
        id="bash_test", name="bash", arguments={"command": "python script.py"}
    )
    env_info = env.step(bash_call)

    assert env_info.step_observation.source == "bash"
    observation = env_info.step_observation.observation
    assert "Python script" in observation


def test_bash_complex_command(env):
    """Test complex bash command with multiple operations."""
    bash_call = ToolCall(
        id="bash_test",
        name="bash",
        arguments={"command": "find . -name '*.txt' | wc -l"},
    )
    env_info = env.step(bash_call)

    assert env_info.step_observation.source == "bash"
    observation = env_info.step_observation.observation
    # Should find test_file.txt and subdir/nested.txt = 2 files
    assert "2" in observation


def test_bash_working_directory(env):
    """Test that bash commands run in the correct working directory."""
    bash_call = ToolCall(id="bash_test", name="bash", arguments={"command": "pwd"})
    env_info = env.step(bash_call)

    assert env_info.step_observation.source == "bash"
    observation = env_info.step_observation.observation
    assert str(env.working_dir) in observation


def test_bash_environment_variables(env):
    """Test that environment variables are accessible."""
    bash_call = ToolCall(
        id="bash_test", name="bash", arguments={"command": "echo $PATH"}
    )
    env_info = env.step(bash_call)

    assert env_info.step_observation.source == "bash"
    observation = env_info.step_observation.observation
    # PATH should be set in the environment
    assert len(observation.strip()) > 0


@patch("debug_gym.gym.terminals.LocalTerminal.run")
def test_bash_terminal_failure(mock_run, bash_tool):
    """Test handling of terminal execution failure."""
    # Mock terminal.run to return failure
    mock_run.return_value = (False, "Command failed: permission denied")

    # Create mock environment with terminal
    mock_env = MagicMock()
    mock_env.terminal = MagicMock()
    mock_env.terminal.run = mock_run

    result = bash_tool.use(mock_env, "some_command")

    assert isinstance(result, Observation)
    assert result.source == "bash"
    assert "Command failed with output:" in result.observation
    assert "permission denied" in result.observation


@patch("debug_gym.gym.terminals.LocalTerminal.run")
def test_bash_terminal_success(mock_run, bash_tool):
    """Test successful terminal execution."""
    # Mock terminal.run to return success
    mock_run.return_value = (True, "Success output")

    # Create mock environment with terminal
    mock_env = MagicMock()
    mock_env.terminal = MagicMock()
    mock_env.terminal.run = mock_run

    result = bash_tool.use(mock_env, "echo hello")

    assert isinstance(result, Observation)
    assert result.source == "bash"
    assert result.observation == "Success output"

    # Verify terminal.run was called with correct parameters
    mock_run.assert_called_once_with("echo hello", timeout=30)


@patch("debug_gym.gym.terminals.LocalTerminal.run")
def test_bash_empty_output_handling(mock_run, bash_tool):
    """Test handling of commands with empty output."""
    # Mock terminal.run to return success with empty output
    mock_run.return_value = (True, "")

    # Create mock environment with terminal
    mock_env = MagicMock()
    mock_env.terminal = MagicMock()
    mock_env.terminal.run = mock_run

    result = bash_tool.use(mock_env, "touch file")

    assert isinstance(result, Observation)
    assert result.source == "bash"
    assert result.observation == "Command executed successfully (no output)"


def test_bash_direct_use_method(bash_tool):
    """Test using the bash tool directly without environment step."""
    # Create mock environment
    mock_env = MagicMock()
    mock_env.terminal.run.return_value = (True, "direct test output")

    result = bash_tool.use(mock_env, "echo direct")

    assert isinstance(result, Observation)
    assert result.source == "bash"
    assert result.observation == "direct test output"


def test_bash_sed_command(env):
    """Test sed command for line extraction."""
    bash_call = ToolCall(
        id="bash_test",
        name="bash",
        arguments={"command": "sed -n '2,3p' test_file.txt"},
    )
    env_info = env.step(bash_call)

    assert env_info.step_observation.source == "bash"
    observation = env_info.step_observation.observation
    assert "Line 2" in observation
    assert "Line 3" in observation
    assert "Hello World" not in observation  # First line should not be included


def test_bash_grep_command(env):
    """Test grep command for searching."""
    bash_call = ToolCall(
        id="bash_test", name="bash", arguments={"command": "grep -r 'Line' ."}
    )
    env_info = env.step(bash_call)

    assert env_info.step_observation.source == "bash"
    observation = env_info.step_observation.observation
    assert "Line 2" in observation or "Line 3" in observation


def test_bash_multiple_commands(env):
    """Test multiple commands chained together."""
    bash_call = ToolCall(
        id="bash_test",
        name="bash",
        arguments={"command": "echo 'test' > temp.txt && cat temp.txt"},
    )
    env_info = env.step(bash_call)

    assert env_info.step_observation.source == "bash"
    observation = env_info.step_observation.observation
    assert "test" in observation
