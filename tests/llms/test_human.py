import json
from unittest.mock import patch

import numpy as np
import pytest

from debug_gym.llms import Human
from debug_gym.gym.tools.tool import ToolCall
from debug_gym.gym.tools.toolbox import Toolbox


@pytest.fixture
def example_tools():
    """
    Build a deterministic list of tools (Human.define_tools generates random IDs,
    so we craft a fixed one instead).
    """
    return [
        {
            "id": "print-001",
            "name": "print",
            "arguments": {"msg": ""},
        },
        {
            "id": "sum-002",
            "name": "sum",
            "arguments": {"a": "", "b": ""},
        },
    ]


def test_parse_tool_call_response_valid(logger_mock, example_tools):
    """A well-formed JSON command that matches one of the declared tools
    should be converted into a ToolCall instance and no error logged."""
    human = Human(logger=logger_mock)
    cmd = json.dumps({"id": "print-001", "name": "print", "arguments": {"msg": "hi"}})

    tool_call = human.parse_tool_call_response(cmd, example_tools)
    assert tool_call == ToolCall(id="print-001", name="print", arguments={"msg": "hi"})


@pytest.mark.parametrize(
    "bad_command",
    [
        "not-json",  # invalid JSON
        json.dumps(
            {"id": "print-001", "name": "print", "arguments": {"unknown": "x"}}
        ),  # wrong args
        json.dumps(
            {"id": "non-existent", "name": "print", "arguments": {}}
        ),  # id not found
        json.dumps(
            {"id": "print-001", "name": "unknown", "arguments": {}}
        ),  # name mismatch
        json.dumps(
            {"id": "print-001", "name": "unknown"}
        ),  # missing ToolCalls arguments
    ],
)
def test_parse_tool_call_response_invalid(logger_mock, example_tools, bad_command):
    """For malformed or non-matching commands the method should raise ValueError"""
    human = Human(logger=logger_mock)

    with pytest.raises(ValueError, match="Failed to parse valid tool call from input"):
        human.parse_tool_call_response(bad_command, example_tools)


def test_parse_tool_call_response_no_tools(logger_mock):
    """Should raise ValueError when no tools are provided"""
    human = Human(logger=logger_mock)

    with pytest.raises(
        ValueError, match="No tools provided. At least one tool must be available."
    ):
        human.parse_tool_call_response(
            '{"id": "test", "name": "test", "arguments": {}}', []
        )

    with pytest.raises(
        ValueError, match="No tools provided. At least one tool must be available."
    ):
        human.parse_tool_call_response(
            '{"id": "test", "name": "test", "arguments": {}}', None
        )


def test_parse_tool_call_response_none(logger_mock, example_tools):
    """Should raise ValueError when response is None"""
    human = Human(logger=logger_mock)

    with pytest.raises(ValueError, match="Tool call cannot be None"):
        human.parse_tool_call_response(None, example_tools)


@patch(
    "builtins.input",
    lambda *args, **kwargs: json.dumps(
        {"id": "pdb-637469", "name": "pdb", "arguments": {"command": "b 10"}}
    ),
)
def test_human(build_env_info):
    # always generate the same random toolcall id: "pdb-637469"
    np.random.seed(42)
    human = Human()
    messages = [{"role": "user", "content": "Hello"}]
    env_info = build_env_info(
        action=ToolCall(id="pdb-637469", name="pdb", arguments="b 10"),
        tools=[Toolbox.get_tool("pdb"), Toolbox.get_tool("view")],
    )
    llm_response = human(messages, env_info.tools)
    # human only uses the messages content
    assert llm_response.prompt == [{"role": "user", "content": "Hello"}]
    assert (
        llm_response.response
        == '{"id": "pdb-637469", "name": "pdb", "arguments": {"command": "b 10"}}'
    )
    assert llm_response.token_usage.prompt == 4
    assert llm_response.token_usage.response == 8


@patch(
    "builtins.input",
    side_effect=["invalid input"] * 10,  # Return invalid input 10 times
)
def test_human_max_retries(_, build_env_info):
    human = Human(max_retries=5)  # Set max_retries to 5
    messages = [{"role": "user", "content": "Test message"}]
    env_info = build_env_info(
        tools=[Toolbox.get_tool("pdb"), Toolbox.get_tool("view")],
    )

    # Should raise ValueError when max retries is reached
    with pytest.raises(
        ValueError, match="Maximum retries \\(5\\) reached without valid input."
    ):
        human(messages, env_info.tools)
