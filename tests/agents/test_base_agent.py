import json
from unittest.mock import MagicMock, Mock, patch

import pytest

from debug_gym.agents.base_agent import (
    AGENT_REGISTRY,
    AgentArgs,
    BaseAgent,
    create_agent,
    register_agent,
)


def test_register_agent():
    """Test agent registration functionality"""

    # Test successful registration
    class TestAgent(BaseAgent):
        name = "test_agent"

    # Clear registry to avoid conflicts
    original_registry = AGENT_REGISTRY.copy()
    AGENT_REGISTRY.clear()

    try:
        registered_agent = register_agent(TestAgent)
        assert registered_agent == TestAgent
        assert AGENT_REGISTRY["test_agent"] == TestAgent

        # Test error cases
        class NotAnAgent:
            name = "not_an_agent"

        with pytest.raises(
            ValueError, match="agent_class must be a subclass of BaseAgent"
        ):
            register_agent(NotAnAgent)

        class AgentWithoutName(BaseAgent):
            name = None

        with pytest.raises(ValueError, match="agent_class must have a name attribute"):
            register_agent(AgentWithoutName)
    finally:
        # Restore original registry
        AGENT_REGISTRY.clear()
        AGENT_REGISTRY.update(original_registry)


def test_create_agent():
    """Test agent creation functionality"""

    # Test creation from registry
    class TestRegisteredAgent(BaseAgent):
        name = "test_registered"

        def __init__(self, agent_args, env, **kwargs):
            super().__init__(agent_args, env, **kwargs)

    # Clear and setup registry
    original_registry = AGENT_REGISTRY.copy()
    AGENT_REGISTRY.clear()
    AGENT_REGISTRY["test_registered"] = TestRegisteredAgent

    try:
        # Mock the required parameters
        mock_config = {
            "output_path": "/tmp",
            "random_seed": 42,
            "memory_size": 10,
            "max_steps": 5,
            "max_rewrite_steps": 3,
        }
        agent_args = AgentArgs.from_dict(mock_config)
        mock_env = MagicMock()

        agent = create_agent("test_registered", agent_args=agent_args, env=mock_env)
        assert isinstance(agent, TestRegisteredAgent)

        # Test unknown agent type
        with pytest.raises(ValueError, match="Unknown agent type: unknown_agent"):
            create_agent("unknown_agent", agent_args=agent_args, env=mock_env)

        # Test module import (mock importlib)
        with patch("importlib.import_module") as mock_import:
            mock_module = MagicMock()
            mock_module.TestClass = TestRegisteredAgent
            mock_import.return_value = mock_module

            agent = create_agent(
                "some.module.TestClass", agent_args=agent_args, env=mock_env
            )
            assert isinstance(agent, TestRegisteredAgent)
            mock_import.assert_called_once_with("some.module")
    finally:
        # Restore original registry
        AGENT_REGISTRY.clear()
        AGENT_REGISTRY.update(original_registry)


def test_system_prompt_building_with_no_template():
    """Test system prompt building when no template is provided"""
    mock_env = MagicMock()
    agent_args = AgentArgs.from_dict(
        {
            "random_seed": 42,
            "memory_size": 10,
            "max_steps": 1,
            "max_rewrite_steps": 1,
        }
    )
    llm = MagicMock()
    llm.context_length = 2000
    llm.count_tokens = Mock(return_value=500)
    agent = BaseAgent(agent_args, llm=llm)
    agent.env = mock_env

    # Create a mock info object
    mock_info = MagicMock()
    mock_info.instructions = "test instructions"
    mock_info.current_breakpoints = []
    mock_info.eval_observation = MagicMock()
    mock_info.eval_observation.observation = "test eval"

    # Mock template loading to return None
    with patch.object(agent, "_load_system_prompt_template", return_value=None):
        system_message = agent.build_system_prompt(mock_info)
        assert system_message is not None
        assert isinstance(system_message, dict)
        assert len(system_message) == 2
        assert system_message["role"] == "system"
        assert "content" in system_message


def test_system_prompt_building_with_template():
    """Test system prompt building with template file"""
    agent = BaseAgent(
        {
            "output_path": "/tmp",
            "random_seed": 42,
            "memory_size": 10,
            "max_steps": 1,
            "max_rewrite_steps": 1,
        },
        MagicMock(),
    )

    # Create a mock info object
    mock_info = MagicMock()
    mock_info.instructions = "test instructions"

    # Mock template loading
    mock_template = MagicMock()
    mock_template.render.return_value = "Task: test_task, Data: data"

    with patch.object(
        agent, "_load_system_prompt_template", return_value=mock_template
    ):
        system_message = agent.build_system_prompt(mock_info)
        assert len(system_message) == 2
        assert system_message["role"] == "system"
        assert "Task: test_task" in system_message["content"]
        assert "Data: data" in system_message["content"]
        mock_template.render.assert_called_once_with(agent=agent, info=mock_info)


def test_to_pretty_json():
    """Test JSON formatting"""
    data = {"key": "value", "number": 42, "list": [1, 2, 3]}
    result = BaseAgent.to_pretty_json(data)
    expected = json.dumps(data, indent=2, sort_keys=False)
    assert result == expected
