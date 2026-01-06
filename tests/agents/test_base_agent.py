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

        def __init__(self, agent_args, **kwargs):
            super().__init__(agent_args, **kwargs)

    # Clear and setup registry
    original_registry = AGENT_REGISTRY.copy()
    AGENT_REGISTRY.clear()
    AGENT_REGISTRY["test_registered"] = TestRegisteredAgent

    try:
        # Mock the required parameters
        mock_config = {
            "type": "test_registered",
            "max_steps": 5,
        }
        agent = create_agent(mock_config)
        assert isinstance(agent, TestRegisteredAgent)

        # Test unknown agent type
        with pytest.raises(ValueError, match="Unknown agent type: unknown_agent"):
            create_agent({"type": "unknown_agent"})

        # Test module import (mock importlib)
        with patch("importlib.import_module") as mock_import:
            mock_module = MagicMock()
            mock_module.TestClass = TestRegisteredAgent
            mock_import.return_value = mock_module

            agent = create_agent({"type": "some.module.TestClass"})
            assert isinstance(agent, TestRegisteredAgent)
            mock_import.assert_called_once_with("some.module")
    finally:
        # Restore original registry
        AGENT_REGISTRY.clear()
        AGENT_REGISTRY.update(original_registry)


def test_load_prompt_template_from_file(tmp_path):
    agent = BaseAgent()
    agent.system_prompt = "test task"
    template_content = "Task: {{ agent.system_prompt }}"
    template_path = tmp_path / "template.jinja"
    template_path.write_text(template_content)
    template = agent._load_prompt_template(template=str(template_path))
    assert isinstance(template, Template)
    assert template.render(agent=agent) == "Task: test task"


def test_load_prompt_template_file_not_found():
    agent = BaseAgent()
    with pytest.raises(FileNotFoundError):
        agent._load_prompt_template(template="non_existent_template.jinja")


def test_to_pretty_json():
    """Test JSON formatting"""
    data = {"key": "value", "number": 42, "list": [1, 2, 3]}
    result = BaseAgent.to_pretty_json(data)
    expected = json.dumps(data, indent=2, sort_keys=False)
    assert result == expected


def test_load_prompt_template_with_filters(tmp_path):
    """Test system prompt template loading with custom filters"""
    agent = BaseAgent()
    agent.llm = Human()
    agent.system_prompt = "Test task"

    # Create template that uses custom filters
    template_content = """
{{ agent.system_prompt }}
{{ {"key": "value"} | to_pretty_json }}
{{ "long message that needs trimming" | trim_message(max_length=10) }}
"""
    template_file = tmp_path / "template.jinja"
    template_file.write_text(template_content)

    template = agent._load_prompt_template(template=str(template_file))
    assert template is not None

    # Test that custom filters are available
    rendered = template.render(agent=agent)
    assert "Test task" in rendered
    assert '"key": "value"' in rendered


def test_build_system_prompt_with_no_template():
    agent = BaseAgent()
    system_message = agent.build_system_prompt()
    assert sorted(system_message.keys()) == ["content", "role"]
    assert system_message["role"] == "system"
    assert system_message["content"] == ""


def test_build_system_prompt_provided_in_args():
    system_prompt = "Custom system prompt"
    agent = BaseAgent(agent_args={"system_prompt": system_prompt})
    assert agent.system_prompt == system_prompt
    system_message = agent.build_system_prompt()
    assert sorted(system_message.keys()) == ["content", "role"]
    assert system_message["role"] == "system"
    assert system_message["content"] == system_prompt


def test_build_system_prompt_with_template():
    system_prompt_template = "Your Mission: {{ info.instructions }}"
    agent = BaseAgent(agent_args={"system_prompt": system_prompt_template})

    mock_info = MagicMock()
    mock_info.instructions = "If you choose to accept it."

    system_message = agent.build_system_prompt(mock_info)
    assert sorted(system_message.keys()) == ["content", "role"]
    assert system_message["role"] == "system"
    assert system_message["content"] == "Your Mission: If you choose to accept it."


def test_build_system_prompt_with_template_file(tmp_path):
    system_prompt_template = "Your Mission: {{ info.instructions }}"
    system_prompt_template_file = tmp_path / "system_prompt.jinja"
    system_prompt_template_file.write_text(system_prompt_template)
    agent = BaseAgent(agent_args={"system_prompt": system_prompt_template_file})

    mock_info = MagicMock()
    mock_info.instructions = "If you choose to accept it."

    system_message = agent.build_system_prompt(mock_info)
    assert sorted(system_message.keys()) == ["content", "role"]
    assert system_message["role"] == "system"
    assert system_message["content"] == "Your Mission: If you choose to accept it."


def test_build_instance_prompt_with_no_template():
    agent = BaseAgent()
    agent.llm = Human()

    mock_info = MagicMock()
    mock_info.instructions = "Test instructions."

    instance_message = agent.build_instance_prompt(mock_info)
    assert sorted(instance_message.keys()) == ["content", "role"]
    assert instance_message["role"] == "user"
    assert mock_info.instructions in instance_message["content"]


def test_build_instance_prompt_provided_in_args():
    instance_prompt = "Custom instance prompt"
    agent = BaseAgent(agent_args={"instance_prompt": instance_prompt})
    agent.llm = Human()
    assert agent.instance_prompt == instance_prompt
    instance_message = agent.build_instance_prompt()
    assert sorted(instance_message.keys()) == ["content", "role"]
    assert instance_message["role"] == "user"
    assert instance_message["content"] == instance_prompt


def test_build_instance_prompt_with_template():
    instance_prompt_template = "Your Mission: {{ info.instructions }}"
    agent = BaseAgent(agent_args={"instance_prompt": instance_prompt_template})
    agent.llm = Human()

    mock_info = MagicMock()
    mock_info.instructions = "If you choose to accept it."

    instance_message = agent.build_instance_prompt(mock_info)
    assert sorted(instance_message.keys()) == ["content", "role"]
    assert instance_message["role"] == "user"
    assert instance_message["content"] == "Your Mission: If you choose to accept it."


def test_build_instance_prompt_with_template_file(tmp_path):
    instance_prompt_template = "Your Mission: {{ info.instructions }}"
    instance_prompt_template_file = tmp_path / "instance_prompt.jinja"
    instance_prompt_template_file.write_text(instance_prompt_template)
    agent = BaseAgent(agent_args={"instance_prompt": instance_prompt_template_file})
    agent.llm = Human()
    mock_info = MagicMock()
    mock_info.instructions = "If you choose to accept it."

    instance_message = agent.build_instance_prompt(mock_info)
    assert sorted(instance_message.keys()) == ["content", "role"]
    assert instance_message["role"] == "user"
    assert instance_message["content"] == "Your Mission: If you choose to accept it."


def test_load_prompt_template_with_include(tmp_path):
    """Test that Jinja2 {% include %} directive works with FileSystemLoader"""
    # Create a partial template in the same directory
    partial_template = tmp_path / "header.jinja"
    partial_template.write_text("=== Header: {{ title }} ===")

    # Create a main template that includes the partial
    main_template = tmp_path / "main.jinja"
    main_template.write_text('{% include "header.jinja" %}\nBody content here.')

    agent = BaseAgent()
    template = agent._load_prompt_template(str(main_template))
    rendered = template.render(title="Test Title")

    assert "=== Header: Test Title ===" in rendered
    assert "Body content here." in rendered


def test_load_prompt_template_with_from_import(tmp_path):
    """Test that Jinja2 {% from %} directive works with FileSystemLoader"""
    # Create a macro template in the same directory
    macro_template = tmp_path / "macros.jinja"
    macro_template.write_text("{% macro greet(name) %}Hello, {{ name }}!{% endmacro %}")

    # Create a main template that imports and uses the macro
    main_template = tmp_path / "main.jinja"
    main_template.write_text(
        '{% from "macros.jinja" import greet %}\n{{ greet("World") }}'
    )

    agent = BaseAgent()
    template = agent._load_prompt_template(str(main_template))
    rendered = template.render()

    assert "Hello, World!" in rendered


def test_load_prompt_template_nested_include(tmp_path):
    """Test nested includes work correctly"""
    # Create base partial
    base_partial = tmp_path / "base.jinja"
    base_partial.write_text("Base: {{ base_content }}")

    # Create intermediate partial that includes base
    intermediate_partial = tmp_path / "intermediate.jinja"
    intermediate_partial.write_text(
        '{% include "base.jinja" %} | Intermediate: {{ inter_content }}'
    )

    # Create main template that includes intermediate
    main_template = tmp_path / "main.jinja"
    main_template.write_text(
        '{% include "intermediate.jinja" %} | Main: {{ main_content }}'
    )

    agent = BaseAgent()
    template = agent._load_prompt_template(str(main_template))
    rendered = template.render(base_content="B", inter_content="I", main_content="M")

    assert "Base: B" in rendered
    assert "Intermediate: I" in rendered
    assert "Main: M" in rendered
