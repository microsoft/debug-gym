"""Unit tests for FreeEnv without mocking."""

import shutil
import tempfile
from pathlib import Path

import pytest

from debug_gym.gym.envs.free_env import FreeEnv
from debug_gym.gym.terminals.docker import DockerTerminal
from debug_gym.gym.terminals.local import LocalTerminal
from debug_gym.gym.tools.toolbox import Toolbox
from debug_gym.logger import DebugGymLogger


@pytest.fixture
def test_repo(tmp_path):
    """Create a simple test repository structure."""
    repo_path = tmp_path / "test_repo"
    repo_path.mkdir()

    # Create some test files
    (repo_path / "main.py").write_text(
        "def greet(name):\n    return f'Hello, {name}!'\n\nif __name__ == '__main__':\n    print(greet('World'))\n"
    )
    (repo_path / "test_main.py").write_text(
        "from main import greet\n\ndef test_greet():\n    assert greet('Alice') == 'Hello, Alice!'\n"
    )
    (repo_path / "requirements.txt").write_text("pytest\n")
    (repo_path / "README.md").write_text("# Test Project\n\nA simple test project.\n")

    # Create a subdirectory
    subdir = repo_path / "utils"
    subdir.mkdir()
    (subdir / "__init__.py").write_text("")
    (subdir / "helper.py").write_text("def add(a, b):\n    return a + b\n")

    return repo_path


@pytest.fixture
def logger():
    """Create a test logger."""
    return DebugGymLogger("test_free_env")


class TestFreeEnvInitialization:
    """Test FreeEnv initialization with different configurations."""

    def test_init_default_terminal(self, test_repo, logger):
        """Test that default terminal is DockerTerminal."""
        task_data = {
            "image": "python:3.11",
            "local_path": str(test_repo),
        }

        env = FreeEnv(task_data=task_data, logger=logger)

        assert isinstance(env.terminal, DockerTerminal)
        assert env.task_data == task_data
        assert env.logger == logger

    def test_init_with_local_terminal_raises_error(self, test_repo, logger):
        """Test that LocalTerminal raises ValueError."""
        task_data = {
            "image": "python:3.11",
            "local_path": str(test_repo),
        }
        terminal = LocalTerminal()

        with pytest.raises(
            ValueError, match="only supports DockerTerminal and KubernetesTerminal"
        ):
            FreeEnv(task_data=task_data, terminal=terminal, logger=logger)

    def test_init_default_setup_commands(self, test_repo, logger):
        """Test that default setup_commands are set when not provided."""
        # When task_data is None, kwargs are used to create task_data
        env = FreeEnv(
            image="python:3.11",
            local_path=str(test_repo),
            logger=logger,
        )

        assert "setup_commands" in env.task_data
        assert "apt-get update" in env.task_data["setup_commands"][0]
        assert "git" in env.task_data["setup_commands"][0]


class TestFreeEnvProperties:
    """Test FreeEnv properties."""

    def test_task_name_property(self, test_repo, logger):
        """Test task_name property returns correct format."""
        task_data = {
            "image": "python:3.11-slim",
            "local_path": str(test_repo),
        }
        terminal = LocalTerminal()

        # Bypass terminal type check for testing
        env = FreeEnv.__new__(FreeEnv)
        env.task_data = task_data
        env.terminal = terminal
        env.logger = logger

        assert env.task_name == "FreeEnv(python:3.11-slim)"

    def test_instructions_property(self, test_repo, logger):
        """Test instructions property returns guidance text."""
        task_data = {
            "image": "python:3.11",
            "local_path": str(test_repo),
        }
        terminal = LocalTerminal()

        env = FreeEnv.__new__(FreeEnv)
        env.task_data = task_data
        env.terminal = terminal
        env.logger = logger

        instructions = env.instructions
        assert "isolated Linux environment" in instructions
        assert "available tools" in instructions


class TestFreeEnvSetupTask:
    """Test FreeEnv task setup."""

    def test_setup_task(self, test_repo, logger):
        """Test setup_task configures terminal correctly."""
        task_data = {
            "image": "python:3.11-alpine",
            "local_path": str(test_repo),
        }
        terminal = LocalTerminal()

        env = FreeEnv.__new__(FreeEnv)
        env.task_data = task_data
        env.terminal = terminal
        env.logger = logger

        env.setup_task()

        assert env.terminal.task_name == "FreeEnv(python:3.11-alpine)"
        assert env.terminal.base_image == "python:3.11-alpine"


class TestFreeEnvSetupWorkspace:
    """Test FreeEnv workspace setup."""

    def test_setup_workspace_method_exists(self, test_repo, logger):
        """Test that setup_workspace method exists."""
        task_data = {
            "image": "python:3.11",
            "local_path": str(test_repo),
            "workspace_dir": "/workspace",
        }

        env = FreeEnv(task_data=task_data, logger=logger)

        # Verify method exists and is callable
        assert hasattr(env, "setup_workspace")
        assert callable(env.setup_workspace)

    def test_setup_workspace_sets_working_dir(self, test_repo, logger):
        """Test that setup_workspace uses the workspace_dir from task_data."""
        workspace_dir = "/custom/testbed"
        task_data = {
            "image": "python:3.11",
            "local_path": str(test_repo),
            "workspace_dir": workspace_dir,
        }

        env = FreeEnv(task_data=task_data, logger=logger)

        # The workspace_dir should be in task_data
        assert env.task_data.get("workspace_dir") == workspace_dir


class TestFreeEnvSetupTerminal:
    """Test FreeEnv terminal setup."""

    def test_setup_terminal_with_git_available(self, test_repo, logger):
        """Test terminal setup when git is available."""
        task_data = {
            "image": "python:3.11",
            "local_path": str(test_repo),
            "setup_commands": ["echo 'setup complete'"],
        }
        terminal = LocalTerminal()

        env = FreeEnv.__new__(FreeEnv)
        env.task_data = task_data
        env.terminal = terminal
        env.logger = logger

        # Test _git_available method
        result = env._git_available()
        # On most systems, git should be available
        assert isinstance(result, bool)

    def test_git_available_with_none_terminal(self, logger):
        """Test _git_available returns False when terminal is None."""
        task_data = {"image": "python:3.11"}

        env = FreeEnv.__new__(FreeEnv)
        env.task_data = task_data
        env.terminal = None
        env.logger = logger

        assert env._git_available() is False


class TestFreeEnvIntegration:
    """Integration tests for FreeEnv."""

    def test_add_tools(self, test_repo, logger):
        """Test adding tools to FreeEnv."""
        task_data = {
            "image": "python:3.11",
            "local_path": str(test_repo),
        }

        env = FreeEnv(task_data=task_data, logger=logger)

        # Add a tool
        bash_tool = Toolbox.get_tool("bash")
        env.add_tool(bash_tool)

        assert len(env.tools) == 1
        assert env.tools[0].name == "bash"

        # Add another tool
        view_tool = Toolbox.get_tool("view")
        env.add_tool(view_tool)

        assert len(env.tools) == 2
        tool_names = [tool.name for tool in env.tools]
        assert "bash" in tool_names
        assert "view" in tool_names

    def test_multiple_workspace_dirs(self, test_repo, logger):
        """Test different workspace_dir values."""
        workspace_dirs = ["/testbed", "/workspace", "/app", "/code"]

        for workspace_dir in workspace_dirs:
            task_data = {
                "image": "python:3.11",
                "local_path": str(test_repo),
                "workspace_dir": workspace_dir,
            }

            env = FreeEnv(task_data=task_data, logger=logger)

            assert env.task_data["workspace_dir"] == workspace_dir

    def test_various_images(self, test_repo, logger):
        """Test FreeEnv with various Docker images."""
        images = [
            "python:3.11",
            "python:3.11-slim",
            "python:3.11-alpine",
            "python:3.10",
            "ubuntu:22.04",
        ]

        for image in images:
            task_data = {
                "image": image,
                "local_path": str(test_repo),
            }

            env = FreeEnv(task_data=task_data, logger=logger)

            assert env.task_name == f"FreeEnv({image})"


class TestFreeEnvEdgeCases:
    """Test edge cases and error conditions."""

    def test_minimal_task_data(self, logger):
        """Test with minimal task_data."""
        task_data = {"image": "python:3.11"}

        env = FreeEnv(task_data=task_data, logger=logger)

        # Should have the image
        assert env.task_data["image"] == "python:3.11"

    def test_kwargs_used_when_no_task_data(self, test_repo, logger):
        """Test that kwargs are used to construct task_data when task_data is None."""
        env = FreeEnv(
            image="python:3.11",
            local_path=str(test_repo),
            workspace_dir="/custom",
            setup_commands=["echo test"],
            logger=logger,
        )

        # Kwargs should be used to create task_data
        assert env.task_data["image"] == "python:3.11"
        assert env.task_data["local_path"] == str(test_repo)
        assert env.task_data["workspace_dir"] == "/custom"
        assert env.task_data["setup_commands"] == ["echo test"]

    def test_task_data_priority_over_kwargs(self, test_repo, logger):
        """Test that task_data takes priority over kwargs."""
        task_data = {
            "image": "python:3.11",
            "local_path": str(test_repo),
            "workspace_dir": "/from_task_data",
        }

        env = FreeEnv(
            task_data=task_data,
            image="python:3.10",  # Should be ignored
            workspace_dir="/from_kwargs",  # Should be ignored
            logger=logger,
        )

        # task_data values should be used
        assert env.task_data["image"] == "python:3.11"
        assert env.task_data["workspace_dir"] == "/from_task_data"

    def test_nonexistent_local_path(self, logger):
        """Test with a non-existent local_path."""
        task_data = {
            "image": "python:3.11",
            "local_path": "/nonexistent/path/that/does/not/exist",
            "workspace_dir": "/testbed",
        }

        env = FreeEnv(task_data=task_data, logger=logger)

        # Should not crash during initialization
        assert env.task_data["local_path"] == "/nonexistent/path/that/does/not/exist"
        # The error would occur during setup_workspace when trying to copy

    def test_custom_setup_commands(self, test_repo, logger):
        """Test with custom setup commands."""
        custom_commands = [
            "apt-get update",
            "apt-get install -y vim",
            "pip install numpy pandas",
        ]

        task_data = {
            "image": "python:3.11",
            "local_path": str(test_repo),
            "setup_commands": custom_commands,
        }

        env = FreeEnv(task_data=task_data, logger=logger)

        assert env.task_data["setup_commands"] == custom_commands

    def test_empty_setup_commands(self, test_repo, logger):
        """Test with empty setup commands."""
        task_data = {
            "image": "python:3.11",
            "local_path": str(test_repo),
            "setup_commands": [],
        }

        env = FreeEnv(task_data=task_data, logger=logger)

        assert env.task_data["setup_commands"] == []


@pytest.if_docker_running
class TestFreeEnvWithDocker:
    """Tests that require Docker to be running."""

    def test_init_with_docker_terminal(self, test_repo, logger):
        """Test initialization with DockerTerminal."""
        task_data = {
            "image": "python:3.11-slim",
            "local_path": str(test_repo),
        }

        env = FreeEnv(task_data=task_data, logger=logger)

        assert isinstance(env.terminal, DockerTerminal)
        assert env.terminal.base_image is None  # Not set until setup_task

    def test_full_initialization_with_docker(self, test_repo, logger):
        """Test full FreeEnv initialization with Docker."""
        task_data = {
            "image": "python:3.11-slim",
            "local_path": str(test_repo),
            "workspace_dir": "/testbed",
            "setup_commands": ["apt-get update && apt-get install -y git"],
        }

        env = FreeEnv(task_data=task_data, logger=logger)

        assert env.task_data == task_data
        assert isinstance(env.terminal, DockerTerminal)
        assert env.task_name == "FreeEnv(python:3.11-slim)"
