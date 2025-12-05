import json
from unittest.mock import MagicMock, Mock, patch

import pytest
from jinja2 import Template

from debug_gym.agents.base_agent import (
    AGENT_REGISTRY,
    AgentArgs,
    BaseAgent,
    create_agent,
    register_agent,
)
from debug_gym.llms.human import Human


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
            "max_steps": 5,
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
    llm = MagicMock()
    llm.context_length = 2000
    llm.count_tokens = Mock(return_value=500)
    agent = BaseAgent(llm=llm)

    # Create a mock info object
    mock_info = MagicMock()
    mock_info.instructions = "test instructions"
    mock_info.current_breakpoints = []
    mock_info.eval_observation = MagicMock()
    mock_info.eval_observation.observation = "test eval"

    # Mock template loading to return None
    with patch.object(agent, "_load_prompt_template", return_value=None):
        system_message = agent.build_system_prompt(mock_info)
        assert system_message is not None
        assert isinstance(system_message, dict)
        assert len(system_message) == 2
        assert system_message["role"] == "system"


def test_system_prompt_override_via_agent_args():
    llm = MagicMock()
    agent = BaseAgent(agent_args={"system_prompt": "Custom system prompt"}, llm=llm)

    assert agent.system_prompt == "Custom system prompt"

    mock_info = MagicMock()
    mock_info.instructions = {}
    mock_info.current_breakpoints = []
    mock_info.eval_observation = MagicMock()
    mock_info.eval_observation.observation = ""

    with patch.object(agent, "_load_prompt_template", return_value=None):
        system_message = agent.build_system_prompt(mock_info)
        content = json.loads(system_message["content"])
        assert content["Overall task"] == "Custom system prompt"


def test_instance_prompt_override_via_agent_args():
    llm = MagicMock()
    llm.convert_observation_to_message.return_value = {
        "role": "user",
        "content": "converted",
    }
    agent = BaseAgent(agent_args={"instance_prompt": "Custom instance prompt"}, llm=llm)

    mock_info = MagicMock()
    mock_info.instructions = {}
    mock_info.current_breakpoints = []
    mock_info.eval_observation = MagicMock()
    mock_info.eval_observation.observation = ""

    with patch.object(agent, "_load_prompt_template", return_value=None):
        agent.build_instance_prompt(mock_info)

    llm.convert_observation_to_message.assert_called_once_with("Custom instance prompt")


def test_system_prompt_building_with_template():
    """Test system prompt building with template file"""
    agent = BaseAgent()

    # Create a mock info object
    mock_info = MagicMock()
    mock_info.instructions = "test instructions"

    # Mock template loading
    mock_template = MagicMock()
    mock_template.render.return_value = "Task: test_task, Data: data"

    with patch.object(agent, "_load_prompt_template", return_value=mock_template):
        system_message = agent.build_system_prompt(mock_info)
        assert len(system_message) == 2
        assert system_message["role"] == "system"
        assert "Task: test_task" in system_message["content"]
        assert "Data: data" in system_message["content"]
        mock_template.render.assert_called_once_with(agent=agent, info=mock_info)


def test_load_prompt_template_from_file(tmp_path):
    agent = BaseAgent()
    agent.system_prompt = "test task"
    template_content = "Task: {{ agent.system_prompt }}"
    template_path = tmp_path / "template.jinja"
    template_path.write_text(template_content)
    template = agent._load_prompt_template(template_file=str(template_path))
    assert isinstance(template, Template)
    assert template.render(agent=agent) == "Task: test task"


def test_load_prompt_template_file_not_found():
    agent = BaseAgent()
    with pytest.raises(FileNotFoundError):
        agent._load_prompt_template(template_file="non_existent_template.jinja")


def test_to_pretty_json():
    """Test JSON formatting"""
    data = {"key": "value", "number": 42, "list": [1, 2, 3]}
    result = BaseAgent.to_pretty_json(data)
    expected = json.dumps(data, indent=2, sort_keys=False)
    assert result == expected


def test_build_instance_prompt():
    """Test instance prompt building"""
    agent = BaseAgent(llm=Human())
    info = MagicMock()
    info.instructions = "test instructions"
    message = agent.build_instance_prompt(info)
    assert info.instructions in message["content"]


def test_load_prompt_template_with_filters(tmp_path):
    """Test system prompt template loading with custom filters"""
    llm = MagicMock()
    llm.context_length = 2000
    llm.count_tokens = Mock(return_value=500)
    agent = BaseAgent(llm=llm)
    agent.system_prompt = "Test task"

    # Create template that uses custom filters
    template_content = """
{{ agent.system_prompt }}
{{ {"key": "value"} | to_pretty_json }}
{{ "long message that needs trimming" | trim_message(max_length=10) }}
"""
    template_file = tmp_path / "template.jinja"
    template_file.write_text(template_content)

    template = agent._load_prompt_template(template_file=str(template_file))
    assert template is not None

    # Test that custom filters are available
    rendered = template.render(agent=agent)
    assert "Test task" in rendered
    assert '"key": "value"' in rendered
