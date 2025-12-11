import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from debug_gym.experiment import add_tools, create_env, dump_experiment_info
from debug_gym.gym.envs.free_env import FreeEnv
from debug_gym.gym.tools.bash import BashTool
from debug_gym.gym.tools.view import ViewTool
from debug_gym.logger import DebugGymLogger


def create_args_object(**kwargs):
    """Helper function to create an Args object with specified attributes."""

    class Args:
        pass

    args = Args()
    for key, value in kwargs.items():
        setattr(args, key, value)
    return args


class TestCreateEnv:
    """Test cases for create_env function.

    Note: create_env is designed for envs that take task_data (SWEBenchEnv, SWESmithEnv, etc.)
    which require Docker. We use mocks to test the orchestration logic without Docker.
    For LocalEnv testing, see TestLocalEnv below which tests the env directly.
    """

    @patch("debug_gym.experiment.select_env")
    @patch("debug_gym.experiment.select_terminal")
    def test_create_env_basic(self, mock_select_terminal, mock_select_env):
        """Test that create_env correctly wires terminal, env class, and config"""
        # Setup mocks
        mock_terminal = mock_select_terminal.return_value
        mock_env_class = mock_select_env.return_value
        mock_env_instance = mock_env_class.return_value

    def test_create_env_basic(self, tmp_path):
        """Test basic environment creation with FreeEnv"""
        # Setup logger
        logger = DebugGymLogger("test")

        config = {
            "terminal": {"type": "docker"},
            "uuid": "test-uuid",
            "env": {"run_timeout": 300},
        }
        task_data = {"env_type": "swebench", "instance_id": "test-123"}

        # Call function
        result = create_env(config, task_data, logger)

        # Assertions
        mock_select_terminal.assert_called_once_with(
            {"type": "docker"}, logger, uuid="test-uuid"
        )
        mock_select_env.assert_called_once_with("swebench")
        mock_env_class.assert_called_once_with(
            task_data=task_data,
            terminal=mock_terminal,
            logger=logger,
            run_timeout=300,
        )
        assert result == mock_env_instance

    @patch("debug_gym.experiment.select_env")
    @patch("debug_gym.experiment.select_terminal")
    def test_create_env_empty_env_config(self, mock_select_terminal, mock_select_env):
        """Test that missing env config defaults to empty dict"""
        mock_terminal = mock_select_terminal.return_value
        mock_env_class = mock_select_env.return_value
        mock_env_instance = mock_env_class.return_value

        logger = DebugGymLogger("test")

        config = {
            "terminal": {"type": "local"},
            "uuid": "test-uuid",
            # No "env" key
        }
        task_data = {"env_type": "swesmith"}

        # Call function
        result = create_env(config, task_data, logger)

        # Assertions - should use empty dict for env kwargs
        mock_env_class.assert_called_once_with(
            task_data=task_data,
            terminal=mock_terminal,
            logger=logger,
        )
        assert result == mock_env_instance

    @patch("debug_gym.experiment.select_env")
    @patch("debug_gym.experiment.select_terminal")
    def test_create_env_with_terminal_none(self, mock_select_terminal, mock_select_env):
        """Test that terminal=None is passed through correctly"""
        logger = DebugGymLogger("test")

        config = {
            "terminal": None,
            "uuid": "test-uuid",
            "env": {},
        }
        task_data = {"env_type": "swebench"}

        # Call function
        create_env(config, task_data, logger)

        # Assertions - select_terminal should be called with None
        mock_select_terminal.assert_called_once_with(None, logger, uuid="test-uuid")
        mock_select_env.assert_called_once_with("swebench")


class TestLocalEnv:
    """Integration tests for LocalEnv - no mocking needed since it doesn't require Docker."""

    def test_local_env_creation(self, tmp_path):
        """Test LocalEnv can be created with a path"""
        logger = DebugGymLogger("test")

        repo_path = tmp_path / "test_repo"
        repo_path.mkdir()
        (repo_path / "test.py").write_text("# test file")

        # Setup config and task_data for FreeEnv
        config = {
            "terminal": {"type": "docker"},
            "uuid": "test-uuid-123",
        }
        task_data = {
            "env_type": "FreeEnv",
            "image": "python:3.11",
            "local_path": str(repo_path),
        }

        # Call function
        result = create_env(config, task_data, logger)

        # Assertions - verify we got a real FreeEnv instance
        assert isinstance(result, FreeEnv)
        assert result.logger == logger

    def test_create_env_with_env_config(self, tmp_path):
        """Test environment creation with env config"""
        # Setup logger
        logger = DebugGymLogger("test")

        repo_path = tmp_path / "test_repo"
        repo_path.mkdir()

        # Setup config with env config
        config = {
            "terminal": {"type": "docker"},
            "uuid": "test-uuid-456",
            "env": {"some_option": "value"},
        }
        task_data = {
            "env_type": "FreeEnv",
            "image": "python:3.11",
            "local_path": str(repo_path),
        }

        # Call function
        result = create_env(config, task_data, logger)

        # Assertions - FreeEnv should be created
        assert isinstance(result, FreeEnv)

    def test_create_env_with_terminal_none(self, tmp_path):
        """Test environment creation with no terminal (None)"""
        # Setup logger
        logger = DebugGymLogger("test")

        # Create a test repository
        repo_path = tmp_path / "test_repo"
        repo_path.mkdir()

        # Setup config with terminal=None
        config = {
            "terminal": None,
            "uuid": "test-uuid-789",
        }
        task_data = {
            "env_type": "FreeEnv",
            "image": "python:3.11",
            "local_path": str(repo_path),
        }

        # Call function
        result = create_env(config, task_data, logger)

        # Assertions - FreeEnv should be created even with terminal=None
        assert isinstance(result, FreeEnv)


class TestAddTools:
    """Test cases for add_tools function"""

    def test_add_tools_single_tool(self, tmp_path):
        """Test adding a single tool to environment"""
        # Create a real environment - use FreeEnv with task_data
        repo_path = tmp_path / "test_repo"
        repo_path.mkdir()

        task_data = {
            "env_type": "FreeEnv",
            "image": "python:3.11",
            "local_path": str(repo_path),
        }
        env = FreeEnv(task_data=task_data)
        logger = DebugGymLogger("test")

        # Setup config
        config = {"tools": ["bash"]}

        # Call function
        add_tools(env, config, logger)

        # Assertions - verify tool was added
        assert len(env.tools) == 1
        assert isinstance(env.tools[0], BashTool)

    def test_add_tools_multiple_tools(self, tmp_path):
        """Test adding multiple tools to environment"""
        # Create a real environment
        repo_path = tmp_path / "test_repo"
        repo_path.mkdir()

        task_data = {
            "env_type": "FreeEnv",
            "image": "python:3.11",
            "local_path": str(repo_path),
        }
        env = FreeEnv(task_data=task_data)
        logger = DebugGymLogger("test")

        # Setup config with multiple tools
        config = {"tools": ["bash", "view"]}

        # Call function
        add_tools(env, config, logger)

        # Assertions - verify all tools were added
        assert len(env.tools) == 2
        tool_types = [type(tool) for tool in env.tools]
        assert BashTool in tool_types
        assert ViewTool in tool_types

    def test_add_tools_empty_list(self, tmp_path):
        """Test add_tools with empty tool list"""
        # Create a real environment
        repo_path = tmp_path / "test_repo"
        repo_path.mkdir()

        task_data = {
            "env_type": "FreeEnv",
            "image": "python:3.11",
            "local_path": str(repo_path),
        }
        env = FreeEnv(task_data=task_data)
        logger = DebugGymLogger("test")

        # Setup config with no tools
        config = {"tools": []}

        # Call function
        add_tools(env, config, logger)

        # Assertions - no tools should be added
        assert len(env.tools) == 0

    def test_add_tools_with_tool_as_dict(self, tmp_path):
        """Test adding a tool specified as a dict with config"""
        from debug_gym.gym.tools.pdb import PDBTool

        repo_path = tmp_path / "test_repo"
        repo_path.mkdir()
        env = LocalEnv(path=str(repo_path))
        logger = DebugGymLogger("test")

        # Tool specified as dict: {"tool_name": {config}}
        # PDBTool accepts auto_list parameter in its constructor
        config = {"tools": [{"pdb": {"auto_list": False}}]}

        add_tools(env, config, logger)

        assert len(env.tools) == 1
        assert isinstance(env.tools[0], PDBTool)
        # PDBTool should have auto_list set to False
        assert env.tools[0].auto_list is False

    def test_add_tools_no_tools_key(self, tmp_path):
        """Test add_tools when config has no 'tools' key"""
        repo_path = tmp_path / "test_repo"
        repo_path.mkdir()
        env = LocalEnv(path=str(repo_path))
        logger = DebugGymLogger("test")

        # Config without tools key
        config = {}

        add_tools(env, config, logger)

        # No tools should be added
        assert len(env.tools) == 0


class TestDumpExperimentInfo:
    """Test cases for dump_experiment_info function"""

    @patch("debug_gym.experiment.subprocess.check_output")
    def test_dump_experiment_info_basic(self, mock_subprocess):
        """Test basic experiment info dumping"""
        # Setup mocks for subprocess calls
        git_hash = "abc123def456"
        git_diff = "diff --git a/file.py b/file.py\n"
        mock_subprocess.side_effect = [
            git_hash.encode(),
            git_diff.encode(),
        ]

        # Setup temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir)
            uuid = "test-uuid-123"

            # Create the output directory
            exp_dir = output_path / uuid
            exp_dir.mkdir(parents=True)

            # Setup config and args
            config = {
                "output_path": str(output_path),
                "uuid": uuid,
                "benchmark": "mini_nightmare",
                "tools": ["bash", "view"],
            }

            # Create a simple object for args instead of Mock
            args_mock = create_args_object(timeout=3600, force_all=False)

            # Call function
            dump_experiment_info(config, args_mock)

            # Assertions - check file was created
            jsonl_file = exp_dir / "experiment_info.jsonl"
            assert jsonl_file.exists()

            # Read and verify content
            with open(jsonl_file) as f:
                content = f.read()
                data = json.loads(content.strip())

            assert data["git_hash"] == git_hash
            assert data["git_diff"] == git_diff
            assert data["config"] == config
            assert "debug_gym_version" in data
            assert "datetime" in data
            assert "python_version" in data
            assert data["args"]["timeout"] == 3600
            assert data["args"]["force_all"] is False

    @patch("debug_gym.experiment.subprocess.check_output")
    def test_dump_experiment_info_git_errors(self, mock_subprocess):
        """Test experiment info dumping when git commands fail"""
        # Setup mocks to raise exceptions
        mock_subprocess.side_effect = [
            Exception("git not found"),
            Exception("git not found"),
        ]

        # Setup temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir)
            uuid = "test-uuid-456"

            # Create the output directory
            exp_dir = output_path / uuid
            exp_dir.mkdir(parents=True)

            # Setup config and args
            config = {
                "output_path": str(output_path),
                "uuid": uuid,
            }

            # Create a simple object for args instead of Mock
            args_mock = create_args_object(debug=True)

            # Call function - should not fail even if git commands fail
            dump_experiment_info(config, args_mock)

            # Assertions - check file was created
            jsonl_file = exp_dir / "experiment_info.jsonl"
            assert jsonl_file.exists()

            # Read and verify content
            with open(jsonl_file) as f:
                content = f.read()
                data = json.loads(content.strip())

            # Git hash and diff should be empty strings
            assert data["git_hash"] == ""
            assert data["git_diff"] == ""
            assert data["config"] == config

    @patch("debug_gym.experiment.subprocess.check_output")
    def test_dump_experiment_info_append_mode(self, mock_subprocess):
        """Test that experiment info is appended to existing file"""
        # Setup mocks for subprocess calls
        mock_subprocess.side_effect = [
            b"hash1",
            b"diff1",
            b"hash2",
            b"diff2",
        ]

        # Setup temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir)
            uuid = "test-uuid-789"

            # Create the output directory
            exp_dir = output_path / uuid
            exp_dir.mkdir(parents=True)

            # Setup config and args
            config = {
                "output_path": str(output_path),
                "uuid": uuid,
            }

            # Create simple objects for args instead of Mock
            args_mock1 = create_args_object(run=1)
            args_mock2 = create_args_object(run=2)

            # Call function twice
            dump_experiment_info(config, args_mock1)
            dump_experiment_info(config, args_mock2)

            # Assertions - check file was created
            jsonl_file = exp_dir / "experiment_info.jsonl"
            assert jsonl_file.exists()

            # Read and verify content - should have two lines
            with open(jsonl_file) as f:
                lines = f.readlines()

            assert len(lines) == 2
            data1 = json.loads(lines[0])
            data2 = json.loads(lines[1])

            assert data1["git_hash"] == "hash1"
            assert data2["git_hash"] == "hash2"
            assert data1["args"]["run"] == 1
            assert data2["args"]["run"] == 2

    @patch("debug_gym.experiment.subprocess.check_output")
    def test_dump_experiment_info_with_complex_args(self, mock_subprocess):
        """Test experiment info dumping with complex argument types"""
        # Setup mocks
        mock_subprocess.side_effect = [b"abc", b""]

        # Setup temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir)
            uuid = "test-uuid-complex"

            # Create the output directory
            exp_dir = output_path / uuid
            exp_dir.mkdir(parents=True)

            # Setup config
            config = {
                "output_path": str(output_path),
                "uuid": uuid,
                "nested": {"key": "value"},
                "list_value": [1, 2, 3],
            }

            # Create a simple object for args with various types
            args_mock = create_args_object(
                string_arg="test",
                int_arg=42,
                bool_arg=True,
                none_arg=None,
                list_arg=["a", "b", "c"],
            )

            # Call function
            dump_experiment_info(config, args_mock)

            # Assertions
            jsonl_file = exp_dir / "experiment_info.jsonl"
            assert jsonl_file.exists()

            with open(jsonl_file) as f:
                data = json.loads(f.read().strip())

            assert data["args"]["string_arg"] == "test"
            assert data["args"]["int_arg"] == 42
            assert data["args"]["bool_arg"] is True
            assert data["args"]["none_arg"] is None
            assert data["args"]["list_arg"] == ["a", "b", "c"]
            assert data["config"]["nested"]["key"] == "value"
            assert data["config"]["list_value"] == [1, 2, 3]
