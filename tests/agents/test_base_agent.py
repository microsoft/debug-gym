import json
from unittest.mock import MagicMock, patch

import pytest
from jinja2 import Template

from debug_gym.agents.base_agent import (
    AGENT_REGISTRY,
    BaseAgent,
    create_agent,
    register_agent,
)
from debug_gym.gym.terminals.terminal import UnrecoverableTerminalError
from debug_gym.gym.tools.tool import ToolCall
from debug_gym.llms.base import LLMResponse
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


def test_load_prompt_template_with_custom_loader_root(tmp_path):
    """Test prompt_loader_root allows includes across sibling directories"""
    # Create modular prompt structure:
    # prompts/
    # ├── common/
    # │   └── header.jinja
    # └── exploration/
    #     └── main.jinja (includes common/header.jinja)
    prompts_dir = tmp_path / "prompts"
    common_dir = prompts_dir / "common"
    exploration_dir = prompts_dir / "exploration"
    common_dir.mkdir(parents=True)
    exploration_dir.mkdir(parents=True)

    # Create shared component
    header_template = common_dir / "header.jinja"
    header_template.write_text("=== {{ title }} ===")

    # Create main template that includes from sibling directory
    main_template = exploration_dir / "main.jinja"
    main_template.write_text('{% include "common/header.jinja" %}\nBody content.')

    # Without custom root, this would fail (can't use .. paths in Jinja2)
    # With custom root set to prompts/, it works
    agent = BaseAgent(agent_args={"prompt_loader_root": str(prompts_dir)})
    template = agent._load_prompt_template(str(main_template))
    rendered = template.render(title="Explorer")

    assert "=== Explorer ===" in rendered
    assert "Body content." in rendered


class TestExecuteAction:
    """Tests for BaseAgent.execute_action method."""

    @pytest.fixture
    def agent_with_mocks(self):
        """Create a BaseAgent with mocked env and llm."""
        agent = BaseAgent()
        agent.env = MagicMock()
        agent.llm = MagicMock()
        # Initialize history with mock data
        mock_info = MagicMock()
        mock_info.instructions = "Test instructions"
        agent.history.init(
            {"role": "system", "content": "system"},
            {"role": "user", "content": "instance"},
            mock_info,
        )
        return agent

    @pytest.fixture
    def mock_llm_response(self):
        """Create a mock LLMResponse."""
        tool_call = ToolCall(
            id="call_123",
            name="test_tool",
            arguments={"arg1": "value1"},
        )
        return LLMResponse(
            prompt=[{"role": "user", "content": "test"}],
            response="test response",
            reasoning_response="test reasoning",
            tool=tool_call,
        )

    def test_execute_action_success(self, agent_with_mocks, mock_llm_response):
        """Test that execute_action updates history on successful step."""
        agent = agent_with_mocks
        mock_env_info = MagicMock()
        agent.env.step.return_value = mock_env_info

        # Initial history should have 1 entry (from init)
        initial_history_len = len(agent.history)

        result = agent.execute_action(mock_llm_response)

        assert result == mock_env_info
        assert len(agent.history) == initial_history_len + 1
        agent.env.step.assert_called_once_with(
            mock_llm_response.tool,
            mock_llm_response.response,
            mock_llm_response.reasoning_response,
        )

    def test_execute_action_unrecoverable_error_with_env_info(
        self, agent_with_mocks, mock_llm_response
    ):
        """Test that history is updated when UnrecoverableTerminalError has env_info."""
        agent = agent_with_mocks
        mock_env_info = MagicMock()
        mock_env_info.step_observation = MagicMock()
        mock_env_info.step_observation.observation = "error observation"

        error = UnrecoverableTerminalError("Terminal died", env_info=mock_env_info)
        agent.env.step.side_effect = error

        initial_history_len = len(agent.history)

        with pytest.raises(UnrecoverableTerminalError) as exc_info:
            agent.execute_action(mock_llm_response)

        assert exc_info.value is error
        # History should be updated with the failed step
        assert len(agent.history) == initial_history_len + 1
        # Verify the last llm_response in history is our mock
        assert agent.history.llm_responses[-1] == mock_llm_response
        # Verify the last env_observation in history is from the error
        assert agent.history.env_observations[-1] == mock_env_info

    def test_execute_action_unrecoverable_error_without_env_info(
        self, agent_with_mocks, mock_llm_response
    ):
        """Test that history is NOT updated when UnrecoverableTerminalError has no env_info."""
        agent = agent_with_mocks

        error = UnrecoverableTerminalError("Terminal died", env_info=None)
        agent.env.step.side_effect = error

        initial_history_len = len(agent.history)

        with pytest.raises(UnrecoverableTerminalError) as exc_info:
            agent.execute_action(mock_llm_response)

        assert exc_info.value is error
        # History should NOT be updated since env_info is None
        assert len(agent.history) == initial_history_len

    def test_execute_action_history_contains_correct_data(
        self, agent_with_mocks, mock_llm_response
    ):
        """Test that history contains the correct tool call data after execute_action."""
        agent = agent_with_mocks
        mock_env_info = MagicMock()
        mock_env_info.step_observation = MagicMock()
        mock_env_info.step_observation.observation = "tool output"
        agent.env.step.return_value = mock_env_info

        agent.execute_action(mock_llm_response)

        # Verify the history contains correct llm_response data
        last_llm_response = agent.history.llm_responses[-1]
        assert last_llm_response.tool.id == "call_123"
        assert last_llm_response.tool.name == "test_tool"
        assert last_llm_response.tool.arguments == {"arg1": "value1"}
        assert last_llm_response.response == "test response"
        assert last_llm_response.reasoning_response == "test reasoning"

        # Verify the history contains correct env_observation
        last_env_obs = agent.history.env_observations[-1]
        assert last_env_obs == mock_env_info


class TestBaseAgentRunReplayActions:
    """Tests for replay_actions parameter in BaseAgent.run()."""

    @pytest.fixture
    def mock_env(self):
        """Create a mock environment."""
        env = MagicMock()
        env.task_name = "test_task"
        env.tools = []
        env.resolved = False
        return env

    @pytest.fixture
    def mock_env_info(self):
        """Create a mock EnvInfo."""
        info = MagicMock()
        info.instructions = "Test instructions"
        info.tools = []
        info.resolved = False
        info.terminated = False
        info.score = 0
        info.max_score = 10
        info.step_observation = MagicMock()
        info.step_observation.observation = "observation"
        info.action_tool_call = None
        return info

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM."""
        llm = MagicMock()
        llm.context_length = 4096
        llm.count_tokens = lambda x: len(str(x))
        llm.define_tools = lambda x: []
        llm.convert_observation_to_message = lambda obs, **kwargs: {
            "role": "user",
            "content": obs if isinstance(obs, str) else str(obs),
        }
        llm.convert_response_to_message = lambda resp: {
            "role": "assistant",
            "content": resp.response,
        }
        return llm

    def test_run_uses_replay_actions_instead_of_step(
        self, mock_env, mock_env_info, mock_llm
    ):
        """Test that replay_actions are used instead of calling step()."""
        agent = BaseAgent(agent_args={"max_steps": 5})
        agent.llm = mock_llm

        # Create replay actions
        replay_actions = [
            LLMResponse(
                prompt=[],
                response="replay response 1",
                reasoning_response="replay reasoning 1",
                tool=ToolCall(id="replay_1", name="tool_1", arguments={}),
            ),
            LLMResponse(
                prompt=[],
                response="replay response 2",
                reasoning_response="replay reasoning 2",
                tool=ToolCall(id="replay_2", name="tool_2", arguments={}),
            ),
        ]

        # Set up env to terminate after 2 steps
        step_count = {"count": 0}

        def mock_reset():
            return mock_env_info

        def mock_step(tool, response, reasoning):
            step_count["count"] += 1
            info = MagicMock()
            info.instructions = "Test"
            info.tools = []
            info.resolved = step_count["count"] >= 2
            info.terminated = step_count["count"] >= 2
            info.score = step_count["count"]
            info.max_score = 10
            info.step_observation = MagicMock()
            info.step_observation.observation = f"observation {step_count['count']}"
            info.action_tool_call = tool
            return info

        mock_env.reset.return_value = mock_env_info
        mock_env.step.side_effect = mock_step
        mock_env.info = mock_env_info

        # Mock the step method to track if it's called
        agent.step = MagicMock()

        agent.run(mock_env, replay_actions=replay_actions)

        # step() should NOT have been called since we had replay actions
        agent.step.assert_not_called()

        # env.step should have been called with replay action tools
        assert mock_env.step.call_count == 2
        calls = mock_env.step.call_args_list
        assert calls[0][0][0].id == "replay_1"
        assert calls[1][0][0].id == "replay_2"

    def test_run_switches_to_step_after_replay_exhausted(
        self, mock_env, mock_env_info, mock_llm
    ):
        """Test that run() switches to calling step() after replay_actions are exhausted."""
        agent = BaseAgent(agent_args={"max_steps": 5})
        agent.llm = mock_llm

        # Create only 1 replay action
        replay_actions = [
            LLMResponse(
                prompt=[],
                response="replay response",
                reasoning_response="replay reasoning",
                tool=ToolCall(id="replay_1", name="replay_tool", arguments={}),
            ),
        ]

        step_count = {"count": 0}

        def mock_reset():
            return mock_env_info

        def mock_env_step(tool, response, reasoning):
            step_count["count"] += 1
            info = MagicMock()
            info.instructions = "Test"
            info.tools = []
            info.resolved = step_count["count"] >= 3
            info.terminated = step_count["count"] >= 3
            info.score = step_count["count"]
            info.max_score = 10
            info.step_observation = MagicMock()
            info.step_observation.observation = f"observation {step_count['count']}"
            info.action_tool_call = tool
            return info

        mock_env.reset.return_value = mock_env_info
        mock_env.step.side_effect = mock_env_step
        mock_env.info = mock_env_info

        # Create new LLM responses for step() calls
        new_llm_responses = [
            LLMResponse(
                prompt=[],
                response="new response 1",
                reasoning_response="new reasoning 1",
                tool=ToolCall(id="new_1", name="new_tool_1", arguments={}),
            ),
            LLMResponse(
                prompt=[],
                response="new response 2",
                reasoning_response="new reasoning 2",
                tool=ToolCall(id="new_2", name="new_tool_2", arguments={}),
            ),
        ]
        agent.step = MagicMock(side_effect=new_llm_responses)

        agent.run(mock_env, replay_actions=replay_actions)

        # step() should have been called 2 times (after replay action exhausted)
        assert agent.step.call_count == 2

        # env.step should have been called 3 times total
        assert mock_env.step.call_count == 3
        calls = mock_env.step.call_args_list
        # First call: replay action
        assert calls[0][0][0].id == "replay_1"
        # Second and third calls: from step()
        assert calls[1][0][0].id == "new_1"
        assert calls[2][0][0].id == "new_2"

    def test_run_with_empty_replay_actions(self, mock_env, mock_env_info, mock_llm):
        """Test that empty replay_actions list behaves normally."""
        agent = BaseAgent(agent_args={"max_steps": 3})
        agent.llm = mock_llm

        step_count = {"count": 0}

        def mock_env_step(tool, response, reasoning):
            step_count["count"] += 1
            info = MagicMock()
            info.instructions = "Test"
            info.tools = []
            info.resolved = step_count["count"] >= 2
            info.terminated = step_count["count"] >= 2
            info.score = step_count["count"]
            info.max_score = 10
            info.step_observation = MagicMock()
            info.step_observation.observation = f"observation {step_count['count']}"
            info.action_tool_call = tool
            return info

        mock_env.reset.return_value = mock_env_info
        mock_env.step.side_effect = mock_env_step
        mock_env.info = mock_env_info

        llm_responses = [
            LLMResponse(
                prompt=[],
                response="response 1",
                reasoning_response="reasoning 1",
                tool=ToolCall(id="call_1", name="tool_1", arguments={}),
            ),
            LLMResponse(
                prompt=[],
                response="response 2",
                reasoning_response="reasoning 2",
                tool=ToolCall(id="call_2", name="tool_2", arguments={}),
            ),
        ]
        agent.step = MagicMock(side_effect=llm_responses)

        # Pass empty list
        agent.run(mock_env, replay_actions=[])

        # step() should be called for all actions
        assert agent.step.call_count == 2

    def test_run_with_none_replay_actions(self, mock_env, mock_env_info, mock_llm):
        """Test that None replay_actions behaves normally."""
        agent = BaseAgent(agent_args={"max_steps": 3})
        agent.llm = mock_llm

        step_count = {"count": 0}

        def mock_env_step(tool, response, reasoning):
            step_count["count"] += 1
            info = MagicMock()
            info.instructions = "Test"
            info.tools = []
            info.resolved = step_count["count"] >= 1
            info.terminated = step_count["count"] >= 1
            info.score = step_count["count"]
            info.max_score = 10
            info.step_observation = MagicMock()
            info.step_observation.observation = f"observation {step_count['count']}"
            info.action_tool_call = tool
            return info

        mock_env.reset.return_value = mock_env_info
        mock_env.step.side_effect = mock_env_step
        mock_env.info = mock_env_info

        llm_response = LLMResponse(
            prompt=[],
            response="response",
            reasoning_response="reasoning",
            tool=ToolCall(id="call_1", name="tool_1", arguments={}),
        )
        agent.step = MagicMock(return_value=llm_response)

        # Pass None (default)
        agent.run(mock_env, replay_actions=None)

        # step() should be called
        assert agent.step.call_count == 1

    def test_replay_actions_order_preserved(self, mock_env, mock_env_info, mock_llm):
        """Test that replay actions are executed in the correct order."""
        agent = BaseAgent(agent_args={"max_steps": 10})
        agent.llm = mock_llm

        # Create 5 replay actions with distinct IDs
        replay_actions = [
            LLMResponse(
                prompt=[],
                response=f"response {i}",
                reasoning_response=f"reasoning {i}",
                tool=ToolCall(
                    id=f"action_{i}", name=f"tool_{i}", arguments={"order": i}
                ),
            )
            for i in range(5)
        ]

        step_count = {"count": 0}

        def mock_env_step(tool, response, reasoning):
            step_count["count"] += 1
            info = MagicMock()
            info.instructions = "Test"
            info.tools = []
            info.resolved = step_count["count"] >= 5
            info.terminated = step_count["count"] >= 5
            info.score = step_count["count"]
            info.max_score = 10
            info.step_observation = MagicMock()
            info.step_observation.observation = f"observation {step_count['count']}"
            info.action_tool_call = tool
            return info

        mock_env.reset.return_value = mock_env_info
        mock_env.step.side_effect = mock_env_step
        mock_env.info = mock_env_info

        agent.run(mock_env, replay_actions=replay_actions)

        # Verify all 5 actions were executed in order
        assert mock_env.step.call_count == 5
        calls = mock_env.step.call_args_list
        for i, call in enumerate(calls):
            assert call[0][0].id == f"action_{i}"
            assert call[0][0].arguments == {"order": i}
