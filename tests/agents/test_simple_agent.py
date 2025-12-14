from unittest.mock import Mock

import pytest

from debug_gym.agents.base_agent import AgentArgs
from debug_gym.agents.simple_agent import SimpleAgent


@pytest.fixture
def agent():
    agent = SimpleAgent(agent_args=AgentArgs(max_steps=10))
    agent.logger = Mock()
    return agent


def test_parse_with_parameters(agent):
    """Covers main parsing logic and multiline parameters"""
    response = """
<function=test>
<parameter=x>1</parameter>
<parameter=code>
def hello():
    pass
</parameter>
</function>
"""
    tool_calls = agent.parse_tool_call(response)
    assert len(tool_calls) == 1
    assert tool_calls[0].name == "test"
    assert tool_calls[0].arguments["x"] == "1"
    assert "def hello():" in tool_calls[0].arguments["code"]


def test_parse_multiple_and_empty(agent):
    """Covers multiple functions and parameter scoping"""
    response = (
        "<function=a><parameter=x>1</parameter></function><function=b></function>"
    )
    tool_calls = agent.parse_tool_call(response)
    assert len(tool_calls) == 2
    assert tool_calls[0].arguments == {"x": "1"}
    assert tool_calls[1].arguments == {}


def test_parse_fallback_and_exception(agent):
    """Covers no-match fallback and exception handling"""
    # No match fallback
    tool_calls = agent.parse_tool_call("text")
    assert not tool_calls
