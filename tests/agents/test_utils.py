import json
import logging
from unittest.mock import MagicMock

import pytest

from debug_gym.agents.utils import load_config, load_trajectory
from debug_gym.llms.base import LLMResponse


def test_load_config():
    import atexit
    import tempfile
    from pathlib import Path

    import yaml

    # do the test in a tmp folder
    tempdir = tempfile.TemporaryDirectory(prefix="TestLoadConfig-")
    working_dir = Path(tempdir.name)
    config_file = str(working_dir / "config.yaml")
    atexit.register(tempdir.cleanup)  # Make sure to cleanup that folder once done.

    config_contents = {
        "agent": {
            "max_steps": 100,
            "type": "pdb_agent",
        },
        "llm": {"name": "gpt2"},
    }

    # write the config file into yaml
    with open(config_file, "w") as f:
        yaml.dump(config_contents, f)

    # now test
    args = [
        "--config",
        config_file,
        "-v",
        "--debug",
    ]
    _config, _args = load_config(args)
    expected_config = {
        "agent": {
            "type": "pdb_agent",
            "max_steps": 100,
        },
        "llm": {"name": "gpt2"},
    }
    assert _config == expected_config
    assert _args.debug is True
    assert _args.logging_level == logging.INFO

    # another test
    args = [
        "--config",
        config_file,
        "-p",
        "agent.type=edit_only",
        "random_seed=456",
        "cot_style=standard",
        "llm.name=gpt20",
        "-v",
        "--debug",
    ]
    _config, _args = load_config(args)
    expected_config = {
        "agent": {
            "type": "edit_only",
            "max_steps": 100,
        },
        "random_seed": 456,
        "cot_style": "standard",
        "llm": {"name": "gpt20"},
    }
    assert _config == expected_config
    assert _args.debug is True
    assert _args.logging_level == logging.INFO


@pytest.fixture
def mock_logger():
    """Create a mock logger for testing."""
    logger = MagicMock()
    return logger


class TestLoadTrajectory:
    """Tests for the load_trajectory function."""

    def test_load_trajectory_no_file(self, tmp_path, mock_logger):
        """Test that load_trajectory returns None when trajectory file doesn't exist."""
        result = load_trajectory(tmp_path, mock_logger)
        assert result is None

    def test_load_trajectory_empty_log(self, tmp_path, mock_logger):
        """Test that load_trajectory returns None when log is empty."""
        trajectory = {"log": []}
        trajectory_file = tmp_path / "trajectory.json"
        with open(trajectory_file, "w") as f:
            json.dump(trajectory, f)

        result = load_trajectory(tmp_path, mock_logger)
        assert result is None

    def test_load_trajectory_no_log_key(self, tmp_path, mock_logger):
        """Test that load_trajectory returns None when log key is missing."""
        trajectory = {"other_key": "value"}
        trajectory_file = tmp_path / "trajectory.json"
        with open(trajectory_file, "w") as f:
            json.dump(trajectory, f)

        result = load_trajectory(tmp_path, mock_logger)
        assert result is None

    def test_load_trajectory_skips_step_zero(self, tmp_path, mock_logger):
        """Test that load_trajectory skips step 0 (initial state)."""
        trajectory = {
            "log": [
                {"step_id": 0, "action": None},
                {
                    "step_id": 1,
                    "action": {
                        "id": "call_1",
                        "name": "test_tool",
                        "arguments": {"arg1": "value1"},
                    },
                    "prompt_response_pairs": [
                        {
                            "prompt": [{"role": "user", "content": "test"}],
                            "response": "test response",
                            "reasoning_response": None,
                            "token_usage": {"prompt": 10, "response": 20},
                        }
                    ],
                },
            ]
        }
        trajectory_file = tmp_path / "trajectory.json"
        with open(trajectory_file, "w") as f:
            json.dump(trajectory, f)

        result = load_trajectory(tmp_path, mock_logger)
        assert result is not None
        assert len(result) == 1
        assert result[0].tool.name == "test_tool"

    def test_load_trajectory_with_prompt_response_pairs(self, tmp_path, mock_logger):
        """Test load_trajectory with full prompt_response_pairs data."""
        trajectory = {
            "log": [
                {
                    "step_id": 1,
                    "action": {
                        "id": "call_123",
                        "name": "edit_file",
                        "arguments": {"file": "test.py", "content": "print('hello')"},
                    },
                    "prompt_response_pairs": [
                        {
                            "prompt": [
                                {
                                    "role": "system",
                                    "content": "You are a helpful assistant",
                                },
                                {"role": "user", "content": "Edit the file"},
                            ],
                            "response": "I'll edit the file for you",
                            "reasoning_response": "Thinking about the edit...",
                            "token_usage": {"prompt": 100, "response": 50},
                        }
                    ],
                }
            ]
        }
        trajectory_file = tmp_path / "trajectory.json"
        with open(trajectory_file, "w") as f:
            json.dump(trajectory, f)

        result = load_trajectory(tmp_path, mock_logger)
        assert result is not None
        assert len(result) == 1

        llm_response = result[0]
        assert isinstance(llm_response, LLMResponse)
        assert llm_response.response == "I'll edit the file for you"
        assert llm_response.reasoning_response == "Thinking about the edit..."
        assert llm_response.token_usage.prompt == 100
        assert llm_response.token_usage.response == 50
        assert llm_response.tool.id == "call_123"
        assert llm_response.tool.name == "edit_file"
        assert llm_response.tool.arguments == {
            "file": "test.py",
            "content": "print('hello')",
        }
        assert len(llm_response.prompt) == 2

    def test_load_trajectory_fallback_without_prompt_response_pairs(
        self, tmp_path, mock_logger
    ):
        """Test load_trajectory fallback when prompt_response_pairs is missing."""
        trajectory = {
            "log": [
                {
                    "step_id": 1,
                    "action": {
                        "id": "call_456",
                        "name": "run_command",
                        "arguments": {"cmd": "ls"},
                    },
                    "content": "Running the command",
                    "reasoning": "Need to list files",
                }
            ]
        }
        trajectory_file = tmp_path / "trajectory.json"
        with open(trajectory_file, "w") as f:
            json.dump(trajectory, f)

        result = load_trajectory(tmp_path, mock_logger)
        assert result is not None
        assert len(result) == 1

        llm_response = result[0]
        assert llm_response.response == "Running the command"
        assert llm_response.reasoning_response == "Need to list files"
        assert llm_response.prompt == []
        assert llm_response.tool.name == "run_command"

    def test_load_trajectory_multiple_steps(self, tmp_path, mock_logger):
        """Test load_trajectory with multiple steps."""
        trajectory = {
            "log": [
                {"step_id": 0, "action": None},
                {
                    "step_id": 1,
                    "action": {
                        "id": "call_1",
                        "name": "tool_1",
                        "arguments": {},
                    },
                    "prompt_response_pairs": [
                        {
                            "prompt": [],
                            "response": "response 1",
                            "token_usage": {"prompt": 10, "response": 5},
                        }
                    ],
                },
                {
                    "step_id": 2,
                    "action": {
                        "id": "call_2",
                        "name": "tool_2",
                        "arguments": {"key": "value"},
                    },
                    "prompt_response_pairs": [
                        {
                            "prompt": [],
                            "response": "response 2",
                            "token_usage": {"prompt": 20, "response": 10},
                        }
                    ],
                },
                {
                    "step_id": 3,
                    "action": {
                        "id": "call_3",
                        "name": "tool_3",
                        "arguments": {},
                    },
                    "prompt_response_pairs": [
                        {
                            "prompt": [],
                            "response": "response 3",
                            "token_usage": {"prompt": 30, "response": 15},
                        }
                    ],
                },
            ]
        }
        trajectory_file = tmp_path / "trajectory.json"
        with open(trajectory_file, "w") as f:
            json.dump(trajectory, f)

        result = load_trajectory(tmp_path, mock_logger)
        assert result is not None
        assert len(result) == 3
        assert result[0].tool.name == "tool_1"
        assert result[1].tool.name == "tool_2"
        assert result[2].tool.name == "tool_3"
        mock_logger.info.assert_called_once()

    def test_load_trajectory_invalid_json(self, tmp_path, mock_logger):
        """Test that load_trajectory handles invalid JSON gracefully."""
        trajectory_file = tmp_path / "trajectory.json"
        with open(trajectory_file, "w") as f:
            f.write("not valid json {{{")

        result = load_trajectory(tmp_path, mock_logger)
        assert result is None
        mock_logger.warning.assert_called_once()

    def test_load_trajectory_skips_steps_without_action(self, tmp_path, mock_logger):
        """Test that load_trajectory skips steps with no action data."""
        trajectory = {
            "log": [
                {
                    "step_id": 1,
                    "action": {},  # Empty action
                },
                {
                    "step_id": 2,
                    # Missing action key entirely
                },
                {
                    "step_id": 3,
                    "action": {
                        "id": "call_valid",
                        "name": "valid_tool",
                        "arguments": {},
                    },
                    "prompt_response_pairs": [
                        {
                            "prompt": [],
                            "response": "valid response",
                            "token_usage": {"prompt": 5, "response": 5},
                        }
                    ],
                },
            ]
        }
        trajectory_file = tmp_path / "trajectory.json"
        with open(trajectory_file, "w") as f:
            json.dump(trajectory, f)

        result = load_trajectory(tmp_path, mock_logger)
        assert result is not None
        assert len(result) == 1
        assert result[0].tool.name == "valid_tool"

    def test_load_trajectory_missing_optional_fields(self, tmp_path, mock_logger):
        """Test load_trajectory with missing optional fields in action."""
        trajectory = {
            "log": [
                {
                    "step_id": 1,
                    "action": {
                        "name": "minimal_tool",
                        # Missing id and arguments
                    },
                    "prompt_response_pairs": [
                        {
                            "prompt": [],
                            "response": "response",
                            "token_usage": {"prompt": 1, "response": 1},
                        }
                    ],
                }
            ]
        }
        trajectory_file = tmp_path / "trajectory.json"
        with open(trajectory_file, "w") as f:
            json.dump(trajectory, f)

        result = load_trajectory(tmp_path, mock_logger)
        assert result is not None
        assert len(result) == 1
        assert result[0].tool.id == ""
        assert result[0].tool.name == "minimal_tool"
        assert result[0].tool.arguments == {}
