from unittest.mock import Mock

import pytest

from debug_gym.agents.base_agent import AGENT_REGISTRY, AgentArgs
from debug_gym.agents.simple_agent import SimpleAgent


@pytest.fixture
def agent():
    agent = SimpleAgent(agent_args=AgentArgs(max_steps=10))
    agent.logger = Mock()
    return agent


def test_single_tool_call(agent):
    response = """
<function=read_file>
<parameter=file_path>/path/to/file.py</parameter>
<parameter=start_line>1</parameter>
</function>
"""
    tool_calls = agent._parse_tool_call(response)

    assert len(tool_calls) == 1
    assert tool_calls[0].name == "read_file"
    assert tool_calls[0].arguments["file_path"] == "/path/to/file.py"
    assert tool_calls[0].arguments["start_line"] == "1"


def test_multiline_parameter(agent):
    response = """
<function=write_file>
<parameter=content>
def hello():
    print("Hello")
</parameter>
</function>
"""
    tool_calls = agent._parse_tool_call(response)

    assert len(tool_calls) == 1
    assert "def hello():" in tool_calls[0].arguments["content"]
    assert 'print("Hello")' in tool_calls[0].arguments["content"]


def test_multiple_tool_calls(agent):
    response = """
<function=read_file>
<parameter=file_path>/file1.py</parameter>
</function>
<function=write_file>
<parameter=file_path>/file2.py</parameter>
</function>
"""
    tool_calls = agent._parse_tool_call(response)

    assert len(tool_calls) == 2
    assert tool_calls[0].name == "read_file"
    assert tool_calls[1].name == "write_file"


def test_no_tool_calls(agent):
    tool_calls = agent._parse_tool_call("Just some text")

    assert len(tool_calls) == 1
    assert tool_calls[0].name == "unknown_function"
    assert tool_calls[0].arguments == {}


def test_empty_function(agent):
    response = "<function=list_files>\n</function>"
    tool_calls = agent._parse_tool_call(response)

    assert len(tool_calls) == 1
    assert tool_calls[0].name == "list_files"
    assert tool_calls[0].arguments == {}


def test_parameter_scoping(agent):
    """Parameters should only belong to their function block"""
    response = """
<function=func1>
<parameter=param1>value1</parameter>
</function>
<function=func2>
<parameter=param2>value2</parameter>
</function>
"""
    tool_calls = agent._parse_tool_call(response)

    assert tool_calls[0].arguments == {"param1": "value1"}
    assert tool_calls[1].arguments == {"param2": "value2"}


def test_agent_registered():
    assert "simple_agent" in AGENT_REGISTRY
    assert AGENT_REGISTRY["simple_agent"] == SimpleAgent
    assert SimpleAgent.name == "simple_agent"
