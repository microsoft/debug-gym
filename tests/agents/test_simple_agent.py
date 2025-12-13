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
    tool_calls = agent._parse_tool_call(response)
    assert len(tool_calls) == 1
    assert tool_calls[0].name == "test"
    assert tool_calls[0].arguments["x"] == "1"
    assert "def hello():" in tool_calls[0].arguments["code"]


def test_parse_multiple_and_empty(agent):
    """Covers multiple functions and parameter scoping"""
    response = (
        "<function=a><parameter=x>1</parameter></function><function=b></function>"
    )
    tool_calls = agent._parse_tool_call(response)
    assert len(tool_calls) == 2
    assert tool_calls[0].arguments == {"x": "1"}
    assert tool_calls[1].arguments == {}


def test_parse_fallback_and_exception(agent):
    """Covers no-match fallback and exception handling"""
    # No match fallback
    tool_calls = agent._parse_tool_call("text")
    assert tool_calls[0].name == "unknown_function"

    # Exception path
    result = agent._parse_tool_call(None)
    assert result is None
    agent.logger.warning.assert_called_once()


def test_run_resolved_and_loop(agent):
    """Covers already-resolved and main loop with multiple tool calls"""
    mock_env = Mock(task_name="test")
    mock_llm = Mock()

    # Test 1: Already resolved
    mock_env.reset.return_value = Mock(resolved=True, score=100, max_score=100)
    agent.build_system_prompt = Mock(return_value="sys")
    agent.build_instance_prompt = Mock(return_value="inst")
    agent._build_trajectory = Mock(return_value="traj")

    result = agent.run(mock_env, mock_llm)
    assert result == "traj"

    # Test 2: Main loop with multiple tool calls
    mock_env.reset.return_value = Mock(resolved=False, score=0, max_score=100)
    mock_env.step.return_value = Mock(resolved=False, score=50, max_score=100)
    mock_llm.return_value = Mock(
        response="<function=a></function><function=b></function>",
        tool=None,
        reasoning_response=None,
    )
    agent.build_prompt = Mock(return_value=[])
    agent.should_stop = Mock(return_value=(True, "done"))

    agent.run(mock_env, mock_llm)
    info_calls = [str(c) for c in agent.logger.info.call_args_list]
    assert any("Multiple tool calls detected" in c for c in info_calls)


def test_run_exception(agent):
    """Covers exception handling in run"""
    mock_env = Mock(task_name="test")
    mock_env.reset.side_effect = Exception("error")

    with pytest.raises(Exception):
        agent.run(mock_env, Mock())

    assert agent.logger.report_progress.call_args[1]["status"] == "error"
