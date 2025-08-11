from dataclasses import make_dataclass
from unittest.mock import MagicMock, patch

import pytest

from debug_gym.gym.entities import Observation
from debug_gym.gym.envs.env import EnvInfo
from debug_gym.gym.tools.tool import EnvironmentTool, ToolCall
from debug_gym.llms import AnthropicLLM
from debug_gym.llms.base import LLMConfig, LLMConfigRegistry, LLMResponse


class Tool1(EnvironmentTool):
    name = "tool 1"
    description = "The description of tool 1"
    arguments = {
        "arg1": {
            "type": ["string"],
            "description": "arg1 description",
        },
    }

    def use(self, env, action):
        return Observation("Tool1", action)


tools = [Tool1()]


anthropic_config = {
    "test-anthropic": {
        "model": "claude-3-opus-20240229",
        "tokenizer": "claude-3-opus-20240229",
        "endpoint": "https://test-endpoint",
        "api_key": "test-api-key",
        "context_limit": 128,
        "tags": ["anthropic"],
        "generate_kwargs": {
            "max_tokens": 20000,
            "temperature": 1,
        },
    }
}

anthropic_thinking_config = {
    "test-anthropic-thinking": {
        "model": "claude-3-opus-20240229",
        "tokenizer": "claude-3-opus-20240229",
        "endpoint": "https://test-endpoint",
        "api_key": "test-api-key",
        "context_limit": 128,
        "tags": ["anthropic", "thinking"],
        "generate_kwargs": {
            "max_tokens": 20000,
            "temperature": 1,
            "thinking": {"type": "enabled", "budget_tokens": 16000},
        },
    }
}


@patch.object(
    LLMConfigRegistry,
    "from_file",
    return_value=LLMConfigRegistry.register_all(
        anthropic_config | anthropic_thinking_config
    ),
)
def test_query_anthropic_model_basic(mock_llm_config, logger_mock):
    llm = AnthropicLLM("test-anthropic", logger=logger_mock)

    mock_response = MagicMock()
    mock_response.content = [MagicMock()]
    tmp_dict = dict(id="1", input={"arg 1": 0}, name="tool 1", type="tool_use")
    mock_response.content[0] = make_dataclass(
        "tmp", ((k, type(v)) for k, v in tmp_dict.items())
    )(**tmp_dict)
    mock_response.usage.input_tokens = 10
    mock_response.usage.output_tokens = 10
    llm.client.messages.create = MagicMock(return_value=mock_response)

    messages = [{"role": "user", "content": "Write a Hello World program"}]
    llm_response = llm(messages, tools)

    assert llm_response.prompt == [
        {"role": "user", "content": "Write a Hello World program"}
    ]
    assert llm_response.tool == ToolCall(id="1", name="tool 1", arguments={"arg 1": 0})
    assert llm_response.token_usage.prompt == 10
    assert llm_response.token_usage.response == 10

    llm.client.messages.create.assert_called_once()
    assert llm.client.messages.create.call_args[1]["model"] == "claude-3-opus-20240229"
    assert llm.client.messages.create.call_args[1]["max_tokens"] == 20000
    assert len(llm.client.messages.create.call_args[1]["messages"]) == 1


def test_query_anthropic_model_with_thinking(logger_mock):
    llm = AnthropicLLM(
        "test-anthropic-thinking",
        logger=logger_mock,
        llm_config=LLMConfig(**anthropic_thinking_config["test-anthropic-thinking"]),
    )

    mock_response = MagicMock()
    mock_response.content = [MagicMock(), MagicMock()]
    tmp_dict = dict(id="1", input={"arg 1": 0}, name="tool 1", type="tool_use")
    mock_response.content[1] = make_dataclass(
        "tmp", ((k, type(v)) for k, v in tmp_dict.items())
    )(**tmp_dict)
    mock_response.usage.input_tokens = 10
    mock_response.usage.output_tokens = 10
    llm.client.messages.create = MagicMock(return_value=mock_response)

    messages = [{"role": "user", "content": "Write a Hello World program"}]

    llm_response = llm(messages, tools)
    assert llm_response.prompt == [
        {"role": "user", "content": "Write a Hello World program"}
    ]
    assert llm_response.tool == ToolCall(id="1", name="tool 1", arguments={"arg 1": 0})
    assert llm_response.token_usage.prompt == 10
    assert llm_response.token_usage.response == 10

    llm.client.messages.create.assert_called_once()
    assert llm.client.messages.create.call_args[1]["model"] == "claude-3-opus-20240229"
    assert llm.client.messages.create.call_args[1]["max_tokens"] == 20000
    assert llm.client.messages.create.call_args[1]["temperature"] == 1.0
    assert llm.client.messages.create.call_args[1]["thinking"]["type"] == "enabled"
    assert llm.client.messages.create.call_args[1]["thinking"]["budget_tokens"] == 16000


@patch.object(
    LLMConfigRegistry,
    "from_file",
    return_value=LLMConfigRegistry.register_all(anthropic_config),
)
def test_query_anthropic_model_no_user_messages(mock_llm_config, logger_mock):
    llm = AnthropicLLM("test-anthropic", logger=logger_mock)

    mock_response = MagicMock()
    mock_response.content = [MagicMock()]
    tmp_dict = dict(id="1", input={"arg 1": 0}, name="tool 1", type="tool_use")
    mock_response.content[0] = make_dataclass(
        "tmp", ((k, type(v)) for k, v in tmp_dict.items())
    )(**tmp_dict)
    llm.client.messages.create = MagicMock(return_value=mock_response)
    llm.count_tokens = MagicMock(return_value=10)

    messages = [{"role": "system", "content": "You are a helpful assistant"}]
    llm_response = llm(messages, tools)

    # Verify default user prompt was added
    assert llm_response.prompt == [
        {"role": "system", "content": "You are a helpful assistant"}
    ]
    assert llm_response.tool == ToolCall(id="1", name="tool 1", arguments={"arg 1": 0})
    llm.client.messages.create.assert_called_once()
    assert len(llm.client.messages.create.call_args[1]["messages"]) == 1
    assert llm.client.messages.create.call_args[1]["messages"][0]["role"] == "user"
    assert (
        llm.client.messages.create.call_args[1]["messages"][0]["content"]
        == "Your answer is: "
    )


@patch.object(
    LLMConfigRegistry,
    "from_file",
    return_value=LLMConfigRegistry.register_all(anthropic_config),
)
def test_query_anthropic_model_with_system_prompt(mock_llm_config, logger_mock):
    llm = AnthropicLLM("test-anthropic", logger=logger_mock)

    mock_response = MagicMock()
    mock_response.content = [MagicMock()]
    tmp_dict = dict(id="1", input={"arg 1": 0}, name="tool 1", type="tool_use")
    mock_response.content[0] = make_dataclass(
        "tmp", ((k, type(v)) for k, v in tmp_dict.items())
    )(**tmp_dict)
    llm.client.messages.create = MagicMock(return_value=mock_response)
    llm.count_tokens = MagicMock(return_value=10)

    messages = [
        {"role": "system", "content": "You are a helpful coding assistant"},
        {"role": "user", "content": "Help me with Python"},
    ]
    llm_response = llm(messages, tools)

    assert llm_response.prompt == messages
    assert llm_response.tool == ToolCall(id="1", name="tool 1", arguments={"arg 1": 0})
    llm.client.messages.create.assert_called_once()
    assert (
        llm.client.messages.create.call_args[1]["system"]
        == "You are a helpful coding assistant"
    )
    assert len(llm.client.messages.create.call_args[1]["messages"]) == 1
    assert llm.client.messages.create.call_args[1]["messages"][0]["role"] == "user"


@patch.object(
    LLMConfigRegistry,
    "from_file",
    return_value=LLMConfigRegistry.register_all(anthropic_config),
)
def test_query_anthropic_model_with_conversation(mock_llm_config, logger_mock):
    llm = AnthropicLLM("test-anthropic", logger=logger_mock)
    mock_response = MagicMock()
    mock_response.content = [MagicMock()]
    tmp_dict = dict(id="1", input={"arg 1": 0}, name="tool 1", type="tool_use")
    mock_response.content[0] = make_dataclass(
        "tmp", ((k, type(v)) for k, v in tmp_dict.items())
    )(**tmp_dict)
    llm.client.messages.create = MagicMock(return_value=mock_response)
    llm.count_tokens = MagicMock(return_value=10)

    # Test with a conversation (user and assistant messages)
    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there! How can I help you?"},
        {"role": "user", "content": "I need help with Python"},
    ]
    mock_response = llm(messages, tools)

    # Verify conversation handling
    assert mock_response.prompt == messages
    assert mock_response.tool == ToolCall(id="1", name="tool 1", arguments={"arg 1": 0})
    ToolCall(id="1", name="tool 1", arguments={"arg 1": 0})
    llm.client.messages.create.assert_called_once()
    assert (
        llm.client.messages.create.call_args[1]["system"]
        == "You are a helpful assistant"
    )
    assert len(llm.client.messages.create.call_args[1]["messages"]) == 3
    assert llm.client.messages.create.call_args[1]["messages"][0]["role"] == "user"
    assert llm.client.messages.create.call_args[1]["messages"][1]["role"] == "assistant"
    assert llm.client.messages.create.call_args[1]["messages"][2]["role"] == "user"


@patch.object(
    LLMConfigRegistry,
    "from_file",
    return_value=LLMConfigRegistry.register_all(anthropic_config),
)
def test_query_anthropic_model_empty_content(mock_llm_config, logger_mock):
    llm = AnthropicLLM("test-anthropic", logger=logger_mock)
    mock_response = MagicMock()
    mock_response.content = [MagicMock()]
    tmp_dict = dict(id="1", input={"arg 1": 0}, name="tool 1", type="tool_use")
    mock_response.content[0] = make_dataclass(
        "tmp", ((k, type(v)) for k, v in tmp_dict.items())
    )(**tmp_dict)
    llm.client.messages.create = MagicMock(return_value=mock_response)
    llm.count_tokens = MagicMock(return_value=10)

    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": ""},  # Empty content should be skipped
        {"role": "user", "content": "Real question"},
    ]
    result = llm(messages, tools)
    assert result.tool == ToolCall(id="1", name="tool 1", arguments={"arg 1": 0})
    llm.client.messages.create.assert_called_once()
    assert len(llm.client.messages.create.call_args[1]["messages"]) == 1
    assert (
        llm.client.messages.create.call_args[1]["messages"][0]["content"]
        == "Real question"
    )


@patch.object(
    LLMConfigRegistry,
    "from_file",
    return_value=LLMConfigRegistry.register_all(anthropic_config),
)
def test_query_anthropic_model_unknown_role(mock_llm_config, logger_mock):
    llm = AnthropicLLM("test-anthropic", logger=logger_mock)
    llm.client.messages.create = MagicMock()
    llm.count_tokens = MagicMock(return_value=10)
    messages = [{"role": "unknown", "content": "This has an unknown role"}]
    with pytest.raises(ValueError, match="Unknown role: unknown"):
        llm(messages, tools)


@patch.object(
    LLMConfigRegistry,
    "from_file",
    return_value=LLMConfigRegistry.register_all(
        {
            "test-anthropic": anthropic_config["test-anthropic"]
            | {"generate_kwargs": {"max_tokens": 4000}}
        }
    ),
)
def test_query_anthropic_model_max_tokens_from_config(mock_llm_config, logger_mock):
    llm = AnthropicLLM("test-anthropic", logger=logger_mock)
    mock_response = MagicMock()
    mock_response.content = [MagicMock()]
    tmp_dict = dict(id="1", input={"arg 1": 0}, name="tool 1", type="tool_use")
    mock_response.content[0] = make_dataclass(
        "tmp", ((k, type(v)) for k, v in tmp_dict.items())
    )(**tmp_dict)
    llm.client.messages.create = MagicMock(return_value=mock_response)
    llm.count_tokens = MagicMock(return_value=10)
    messages = [{"role": "user", "content": "Test message"}]
    llm(messages, tools)
    assert llm.client.messages.create.call_args[1]["max_tokens"] == 4000


@patch.object(
    LLMConfigRegistry,
    "from_file",
    return_value=LLMConfigRegistry.register_all(anthropic_config),
)
def test_format_tool_call_history_initial_state(mock_llm_config, logger_mock):
    """Test format_tool_call_history with initial state (no action taken yet)"""
    llm = AnthropicLLM("test-anthropic", logger=logger_mock)

    # Create EnvInfo for initial state
    history_info = EnvInfo(
        step_observation=Observation(source="tool1", observation="Initial observation"),
        all_observations=[],
        eval_observation=Observation(source="tool1", observation=""),
        dir_tree="",
        current_breakpoints="",
        action_reasoning=None,  # No reasoning yet
        action_content=None,  # No content yet
        action_tool_call=None,  # No action taken yet
        instructions={},
        score=0,
        max_score=100,
        done=False,
        rewrite_counter=0,
        tools=[],
    )

    # llm_response is None
    messages = llm.format_tool_call_history(history_info, [])
    assert len(messages) == 1
    # Only message should be the user's initial observation
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "Initial observation"


@patch.object(
    LLMConfigRegistry,
    "from_file",
    return_value=LLMConfigRegistry.register_all(anthropic_config),
)
def test_format_tool_call_history_with_action(mock_llm_config, logger_mock):
    """Test format_tool_call_history with an action taken"""
    llm = AnthropicLLM("test-anthropic", logger=logger_mock)

    # Create action that was taken
    action = ToolCall(
        id="tool_456",
        name="edit",
        arguments={"path": "test.py", "content": "new content"},
    )

    # Create EnvInfo with action taken
    history_info = EnvInfo(
        step_observation=Observation(
            source="tool_456", observation="File edited successfully"
        ),
        all_observations=[],
        eval_observation=Observation(source="tool_456", observation=""),
        dir_tree="",
        current_breakpoints="",
        action_reasoning="Edited the file to fix the bug",
        action_content="Edited the file to fix the bug",
        action_tool_call=action,  # Action was taken
        instructions={},
        score=0,
        max_score=100,
        done=False,
        rewrite_counter=0,
        tools=[],
    )
    # Create LLMResponse with the action
    llm_response = LLMResponse(
        prompt=[{"role": "user", "content": "test"}],
        response="Edited the file to fix the bug",
        tool=action,
    )

    messages = llm.format_tool_call_history(history_info, [llm_response])

    assert len(messages) == 2
    # First message should be the assistant's tool use
    assert messages[0]["role"] == "assistant"
    assert messages[0]["content"][0]["type"] == "text"
    assert messages[0]["content"][0]["text"] == "Edited the file to fix the bug"
    assert messages[0]["content"][1]["type"] == "tool_use"
    assert messages[0]["content"][1]["id"] == "tool_456"
    assert messages[0]["content"][1]["name"] == "edit"
    assert messages[0]["content"][1]["input"] == {
        "path": "test.py",
        "content": "new content",
    }

    # Second message should be the tool result
    assert messages[1]["role"] == "user"
    assert messages[1]["content"][0]["type"] == "tool_result"
    assert messages[1]["content"][0]["tool_use_id"] == "tool_456"
    assert messages[1]["content"][0]["content"] == "File edited successfully"
