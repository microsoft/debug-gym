import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from debug_gym.experiment import add_tools, create_env, dump_experiment_info
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
    """Test cases for create_env function"""

    @patch("debug_gym.experiment.select_terminal")
    @patch("debug_gym.experiment.select_env")
    def test_create_env_basic(self, mock_select_env, mock_select_terminal):
        """Test basic environment creation with minimal config"""
        # Setup mocks
        mock_terminal = Mock()
        mock_select_terminal.return_value = mock_terminal

        mock_env_class = Mock()
        mock_env_instance = Mock()
        mock_env_class.return_value = mock_env_instance
        mock_select_env.return_value = mock_env_class

        # Setup logger
        logger = DebugGymLogger("test")

        # Setup config
        config = {
            "terminal": {"type": "local"},
            "benchmark": "mini_nightmare",
            "env_kwargs": {"max_steps": 10},
            "problems": ["problem1", "problem2"],
        }

        # Call function
        result = create_env(config, logger)

        # Assertions
        mock_select_terminal.assert_called_once_with({"type": "local"}, logger)
        mock_select_env.assert_called_once_with("mini_nightmare")
        mock_env_class.assert_called_once_with(
            max_steps=10,
            problems=["problem1", "problem2"],
            terminal=mock_terminal,
            logger=logger,
        )
        assert result == mock_env_instance

    @patch("debug_gym.experiment.select_terminal")
    @patch("debug_gym.experiment.select_env")
    def test_create_env_default_problems(self, mock_select_env, mock_select_terminal):
        """Test environment creation uses default problems when not specified"""
        # Setup mocks
        mock_terminal = Mock()
        mock_select_terminal.return_value = mock_terminal

        mock_env_class = Mock()
        mock_env_instance = Mock()
        mock_env_class.return_value = mock_env_instance
        mock_select_env.return_value = mock_env_class

        # Setup logger
        logger = DebugGymLogger("test")

        # Setup config without problems
        config = {
            "terminal": {"type": "docker"},
            "benchmark": "swebench",
            "env_kwargs": {},
        }

        # Call function
        result = create_env(config, logger)

        # Assertions - should use default ["custom"]
        mock_env_class.assert_called_once_with(
            problems=["custom"], terminal=mock_terminal, logger=logger
        )
        assert result == mock_env_instance

    @patch("debug_gym.experiment.select_terminal")
    @patch("debug_gym.experiment.select_env")
    def test_create_env_with_multiple_env_kwargs(
        self, mock_select_env, mock_select_terminal
    ):
        """Test environment creation with multiple env_kwargs"""
        # Setup mocks
        mock_terminal = Mock()
        mock_select_terminal.return_value = mock_terminal

        mock_env_class = Mock()
        mock_env_instance = Mock()
        mock_env_class.return_value = mock_env_instance
        mock_select_env.return_value = mock_env_class

        # Setup logger
        logger = DebugGymLogger("test")

        # Setup config with multiple kwargs
        config = {
            "terminal": None,
            "benchmark": "local",
            "env_kwargs": {
                "max_steps": 20,
                "timeout": 3600,
                "working_dir": "/tmp/test",
            },
            "problems": [],
        }

        # Call function
        result = create_env(config, logger)

        # Assertions
        mock_env_class.assert_called_once_with(
            max_steps=20,
            timeout=3600,
            working_dir="/tmp/test",
            problems=[],
            terminal=mock_terminal,
            logger=logger,
        )
        assert result == mock_env_instance


class TestAddTools:
    """Test cases for add_tools function"""

    @patch("debug_gym.experiment.Toolbox.get_tool")
    def test_add_tools_single_tool(self, mock_get_tool):
        """Test adding a single tool to environment"""
        # Setup mocks
        mock_tool = Mock()
        mock_tool.__class__.__name__ = "BashTool"
        mock_get_tool.return_value = mock_tool

        mock_env = Mock()
        logger = DebugGymLogger("test")

        # Setup config
        config = {"tools": ["bash"]}

        # Call function
        add_tools(mock_env, config, logger)

        # Assertions
        mock_get_tool.assert_called_once_with("bash")
        mock_env.add_tool.assert_called_once_with(mock_tool)

    @patch("debug_gym.experiment.Toolbox.get_tool")
    def test_add_tools_multiple_tools(self, mock_get_tool):
        """Test adding multiple tools to environment"""
        # Setup mocks for different tools
        mock_bash_tool = Mock()
        mock_bash_tool.__class__.__name__ = "BashTool"

        mock_view_tool = Mock()
        mock_view_tool.__class__.__name__ = "ViewTool"

        mock_edit_tool = Mock()
        mock_edit_tool.__class__.__name__ = "EditTool"

        mock_get_tool.side_effect = [mock_bash_tool, mock_view_tool, mock_edit_tool]

        mock_env = Mock()
        logger = DebugGymLogger("test")

        # Setup config with multiple tools
        config = {"tools": ["bash", "view", "edit"]}

        # Call function
        add_tools(mock_env, config, logger)

        # Assertions
        assert mock_get_tool.call_count == 3
        assert mock_env.add_tool.call_count == 3
        mock_env.add_tool.assert_any_call(mock_bash_tool)
        mock_env.add_tool.assert_any_call(mock_view_tool)
        mock_env.add_tool.assert_any_call(mock_edit_tool)

    @patch("debug_gym.experiment.Toolbox.get_tool")
    def test_add_tools_empty_list(self, mock_get_tool):
        """Test add_tools with empty tool list"""
        # Setup mocks
        mock_env = Mock()
        logger = DebugGymLogger("test")

        # Setup config with no tools
        config = {"tools": []}

        # Call function
        add_tools(mock_env, config, logger)

        # Assertions
        mock_get_tool.assert_not_called()
        mock_env.add_tool.assert_not_called()

    @patch("debug_gym.experiment.Toolbox.get_tool")
    def test_add_tools_with_parameterized_tools(self, mock_get_tool):
        """Test adding tools with parameters (e.g., 'bash:debug')"""
        # Setup mocks
        mock_tool = Mock()
        mock_tool.__class__.__name__ = "BashTool"
        mock_get_tool.return_value = mock_tool

        mock_env = Mock()
        logger = DebugGymLogger("test")

        # Setup config with parameterized tool
        config = {"tools": ["bash:debug"]}

        # Call function
        add_tools(mock_env, config, logger)

        # Assertions
        mock_get_tool.assert_called_once_with("bash:debug")
        mock_env.add_tool.assert_called_once_with(mock_tool)


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
