from unittest.mock import Mock

import pytest

from debug_gym.agents.base_agent import AgentArgs
from debug_gym.agents.simple_agent import SimpleAgent, describe_tools
from debug_gym.gym.tools.tool import EnvironmentTool


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


class MockTool(EnvironmentTool):
    """Mock tool for testing."""

    def __init__(self, name, description, arguments):
        super().__init__()
        self.name = name
        self.description = description
        self.arguments = arguments

    def use(self, *args, **kwargs):
        pass


def test_describe_tools_with_parameters():
    """Test describe_tools with tools that have parameters."""
    bash_tool = MockTool(
        name="bash",
        description="Execute a bash command in the terminal.",
        arguments={
            "command": {
                "type": ["string"],
                "description": "The bash command to execute.",
            }
        },
    )

    result = describe_tools([bash_tool])

    # Check structure
    assert "---- BEGIN FUNCTION #1: bash ----" in result
    assert "---- END FUNCTION #1 ----" in result
    assert "Description: Execute a bash command in the terminal." in result
    assert "(1) command (string, required): The bash command to execute." in result


def test_describe_tools_without_parameters():
    """Test describe_tools with tools that have no parameters."""
    submit_tool = MockTool(
        name="submit",
        description="Finish the interaction when the task is complete.",
        arguments={},
    )

    result = describe_tools([submit_tool])

    # Check structure
    assert "---- BEGIN FUNCTION #1: submit ----" in result
    assert "Description: Finish the interaction when the task is complete." in result
    assert "No parameters are required for this function." in result


def test_describe_tools_with_optional_parameters():
    """Test describe_tools correctly identifies optional parameters."""
    tool = MockTool(
        name="view",
        description="View a file.",
        arguments={
            "path": {"type": ["string"], "description": "File path."},
            "start": {
                "type": ["number", "null"],
                "description": "Optional start line.",
            },
        },
    )

    result = describe_tools([tool])

    assert "(1) path (string, required): File path." in result
    assert "(2) start (number, null, optional): Optional start line." in result


def test_describe_tools_multiple_tools():
    """Test describe_tools with multiple tools."""
    tool1 = MockTool(
        name="bash",
        description="Run bash command.",
        arguments={"command": {"type": ["string"], "description": "Command to run."}},
    )
    tool2 = MockTool(name="submit", description="Submit the task.", arguments={})

    result = describe_tools([tool1, tool2])

    # Check both tools are present with correct indices
    assert "---- BEGIN FUNCTION #1: bash ----" in result
    assert "---- BEGIN FUNCTION #2: submit ----" in result
    assert "---- END FUNCTION #1 ----" in result
    assert "---- END FUNCTION #2 ----" in result


def test_simple_agent_system_prompt_generation():
    """Test that SimpleAgent generates system prompt dynamically from env.tools."""
    from debug_gym.agents.simple_agent import SimpleAgentArgs
    from debug_gym.gym.envs.env import EnvInfo

    agent = SimpleAgent(agent_args=SimpleAgentArgs(max_steps=10))
    agent.logger = Mock()
    agent.llm = Mock()
    agent.llm.convert_observation_to_message = Mock(
        return_value={"role": "user", "content": "test"}
    )

    # Create mock environment with tools
    mock_env = Mock()
    bash_tool = MockTool(
        name="bash",
        description="Execute bash commands.",
        arguments={"command": {"type": ["string"], "description": "The command."}},
    )
    submit_tool = MockTool(name="submit", description="Submit the task.", arguments={})
    mock_env.tools = [bash_tool, submit_tool]
    agent.env = mock_env

    # Create mock info
    mock_info = Mock(spec=EnvInfo)
    mock_info.instructions = "Test instructions"
    mock_info.current_breakpoints = ""
    mock_info.eval_observation = None

    # Before building prompt, system prompt should have placeholder
    assert "{tools_description}" in agent.args.system_prompt
    assert not agent._system_prompt_generated

    # Build prompt - this should trigger generation
    agent.build_prompt(mock_info)

    # After building prompt, system prompt should be generated
    assert agent._system_prompt_generated
    assert "{tools_description}" not in agent.system_prompt
    assert "---- BEGIN FUNCTION #1: bash ----" in agent.system_prompt
    assert "---- BEGIN FUNCTION #2: submit ----" in agent.system_prompt
    assert "Execute bash commands." in agent.system_prompt
    assert "Submit the task." in agent.system_prompt


def test_simple_agent_system_prompt_cached():
    """Test that system prompt is only generated once and then cached."""
    from debug_gym.agents.simple_agent import SimpleAgentArgs
    from debug_gym.gym.envs.env import EnvInfo

    agent = SimpleAgent(agent_args=SimpleAgentArgs(max_steps=10))
    agent.logger = Mock()
    agent.llm = Mock()
    agent.llm.convert_observation_to_message = Mock(
        return_value={"role": "user", "content": "test"}
    )

    # Create mock environment with tools
    mock_env = Mock()
    bash_tool = MockTool(
        name="bash",
        description="Execute bash commands.",
        arguments={"command": {"type": ["string"], "description": "The command."}},
    )
    mock_env.tools = [bash_tool]
    agent.env = mock_env

    # Create mock info
    mock_info = Mock(spec=EnvInfo)
    mock_info.instructions = "Test instructions"
    mock_info.current_breakpoints = ""
    mock_info.eval_observation = None

    # Build prompt first time
    agent.build_prompt(mock_info)
    first_prompt = agent.system_prompt

    # Modify the env tools (simulating runtime change)
    submit_tool = MockTool(name="submit", description="Submit.", arguments={})
    mock_env.tools = [bash_tool, submit_tool]

    # Build prompt second time
    agent.build_prompt(mock_info)
    second_prompt = agent.system_prompt

    # System prompt should be the same (cached)
    assert first_prompt == second_prompt
    assert "---- BEGIN FUNCTION #2: submit ----" not in second_prompt


def test_simple_agent_with_no_env():
    """Test that SimpleAgent handles case when env is not set."""
    from debug_gym.agents.simple_agent import SimpleAgentArgs
    from debug_gym.gym.envs.env import EnvInfo

    agent = SimpleAgent(agent_args=SimpleAgentArgs(max_steps=10))
    agent.logger = Mock()
    agent.llm = Mock()
    agent.llm.convert_observation_to_message = Mock(
        return_value={"role": "user", "content": "test"}
    )

    # Don't set agent.env
    mock_info = Mock(spec=EnvInfo)
    mock_info.instructions = "Test instructions"
    mock_info.current_breakpoints = ""
    mock_info.eval_observation = None

    # Should not crash, but system prompt won't be generated
    agent.build_prompt(mock_info)

    # System prompt should still have placeholder
    assert "{tools_description}" in agent.args.system_prompt
    assert not agent._system_prompt_generated
