from dataclasses import make_dataclass
from unittest.mock import MagicMock, patch

import pytest

from debug_gym.gym.entities import Observation
from debug_gym.gym.tools.tool import EnvironmentTool, ToolCall
from debug_gym.llms import AnthropicLLM
from debug_gym.llms.base import LLM, LLMConfig, LLMConfigRegistry


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


def create_fake_exception(
    module: str, classname: str, message: str, code: str = None, status_code: int = None
):
    """Create a fake exception for testing need_to_be_retried."""
    exc_type = type(classname, (Exception,), {})
    exc = exc_type(message)
    exc.message = message
    if code is not None:
        exc.code = code
    if status_code is not None:
        exc.status_code = status_code
    exc.__class__.__module__ = module
    return exc


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
    llm = LLM.instantiate("test-anthropic", logger=logger_mock)

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
    # Mock the from_file to return our test config
    with patch.object(
        LLMConfigRegistry,
        "from_file",
        return_value=LLMConfigRegistry.register_all(anthropic_thinking_config),
    ):
        llm = LLM.instantiate("test-anthropic-thinking", logger=logger_mock)

    mock_response = MagicMock()
    # Create a proper thinking block with .thinking attribute (not .text)
    thinking_block = make_dataclass(
        "ThinkingBlock",
        [("type", str), ("thinking", str), ("signature", str)],
    )(type="thinking", thinking="Let me analyze this...", signature="abc123signature")
    tool_use_block = make_dataclass(
        "ToolUseBlock",
        [("type", str), ("id", str), ("input", dict), ("name", str)],
    )(type="tool_use", id="1", input={"arg 1": 0}, name="tool 1")
    mock_response.content = [thinking_block, tool_use_block]
    mock_response.usage.input_tokens = 10
    mock_response.usage.output_tokens = 10
    llm.client.messages.create = MagicMock(return_value=mock_response)

    messages = [{"role": "user", "content": "Write a Hello World program"}]

    llm_response = llm(messages, tools)
    assert llm_response.prompt == [
        {"role": "user", "content": "Write a Hello World program"}
    ]
    assert llm_response.tool == ToolCall(id="1", name="tool 1", arguments={"arg 1": 0})
    assert llm_response.reasoning_response == "Let me analyze this..."
    assert llm_response.token_usage.prompt == 10
    assert llm_response.token_usage.response == 10
    # Verify thinking blocks are preserved
    assert hasattr(llm_response, "thinking_blocks")
    assert len(llm_response.thinking_blocks) == 1
    assert llm_response.thinking_blocks[0].type == "thinking"
    assert llm_response.thinking_blocks[0].signature == "abc123signature"

    llm.client.messages.create.assert_called_once()
    assert llm.client.messages.create.call_args[1]["model"] == "claude-3-opus-20240229"
    assert llm.client.messages.create.call_args[1]["max_tokens"] == 20000
    assert llm.client.messages.create.call_args[1]["temperature"] == 1.0
    assert llm.client.messages.create.call_args[1]["thinking"]["type"] == "enabled"
    assert llm.client.messages.create.call_args[1]["thinking"]["budget_tokens"] == 16000


def test_convert_response_to_message_with_thinking(logger_mock):
    """Test that thinking blocks are properly converted with signature preserved."""
    from debug_gym.llms.base import LLMResponse

    llm = AnthropicLLM(
        "test-anthropic-thinking",
        logger=logger_mock,
        llm_config=LLMConfig(**anthropic_thinking_config["test-anthropic-thinking"]),
    )

    # Create a mock thinking block
    thinking_block = make_dataclass(
        "ThinkingBlock",
        [("type", str), ("thinking", str), ("signature", str)],
    )(type="thinking", thinking="My reasoning process...", signature="sig123")

    # Create an LLMResponse with thinking blocks
    response = LLMResponse(
        prompt=[{"role": "user", "content": "test"}],
        response="Here's my answer",
        reasoning_response="My reasoning process...",
        tool=ToolCall(id="tool1", name="test_tool", arguments={"arg": "val"}),
    )
    response.thinking_blocks = [thinking_block]

    # Convert to message
    message = llm.convert_response_to_message(response)

    assert message["role"] == "assistant"
    assert len(message["content"]) == 3  # thinking, text, tool_use

    # Check thinking block format
    thinking_content = message["content"][0]
    assert thinking_content["type"] == "thinking"
    assert thinking_content["thinking"] == "My reasoning process..."
    assert thinking_content["signature"] == "sig123"

    # Check text block
    text_content = message["content"][1]
    assert text_content["type"] == "text"
    assert text_content["text"] == "Here's my answer"

    # Check tool_use block
    tool_content = message["content"][2]
    assert tool_content["type"] == "tool_use"
    assert tool_content["id"] == "tool1"


def test_convert_response_to_message_without_tool(logger_mock):
    """Test that convert_response_to_message omits tool_use when tool is None."""
    from debug_gym.llms.base import LLMResponse

    llm = AnthropicLLM(
        "test-anthropic-thinking",
        logger=logger_mock,
        llm_config=LLMConfig(**anthropic_thinking_config["test-anthropic-thinking"]),
    )

    response = LLMResponse(
        prompt=[{"role": "user", "content": "test"}],
        response="Here's my answer",
        tool=None,
    )

    message = llm.convert_response_to_message(response)

    assert message["role"] == "assistant"
    assert len(message["content"]) == 1  # text only
    assert message["content"][0]["type"] == "text"
    assert message["content"][0]["text"] == "Here's my answer"


def test_convert_response_to_message_with_redacted_thinking(logger_mock):
    """Test that redacted_thinking blocks are properly handled."""
    from debug_gym.llms.base import LLMResponse

    llm = AnthropicLLM(
        "test-anthropic-thinking",
        logger=logger_mock,
        llm_config=LLMConfig(**anthropic_thinking_config["test-anthropic-thinking"]),
    )

    # Create a mock redacted_thinking block
    redacted_block = make_dataclass(
        "RedactedThinkingBlock",
        [("type", str), ("data", str)],
    )(type="redacted_thinking", data="encrypted_data_here")

    # Create an LLMResponse with redacted thinking
    response = LLMResponse(
        prompt=[{"role": "user", "content": "test"}],
        response="Here's my answer",
        tool=ToolCall(id="tool1", name="test_tool", arguments={}),
    )
    response.thinking_blocks = [redacted_block]

    # Convert to message
    message = llm.convert_response_to_message(response)

    # Check redacted_thinking block format
    redacted_content = message["content"][0]
    assert redacted_content["type"] == "redacted_thinking"
    assert redacted_content["data"] == "encrypted_data_here"


@patch.object(
    LLMConfigRegistry,
    "from_file",
    return_value=LLMConfigRegistry.register_all(anthropic_config),
)
def test_query_anthropic_model_no_user_messages(mock_llm_config, logger_mock):
    llm = LLM.instantiate("test-anthropic", logger=logger_mock)

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
    llm = LLM.instantiate("test-anthropic", logger=logger_mock)

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
    llm = LLM.instantiate("test-anthropic", logger=logger_mock)
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
    llm = LLM.instantiate("test-anthropic", logger=logger_mock)
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
    llm = LLM.instantiate("test-anthropic", logger=logger_mock)
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
    llm = LLM.instantiate("test-anthropic", logger=logger_mock)
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


def test_need_to_be_retried(logger_mock):
    """Test that need_to_be_retried correctly identifies retryable errors."""
    llm = AnthropicLLM(
        "test-anthropic",
        logger=logger_mock,
        llm_config=LLMConfig(**anthropic_config["test-anthropic"]),
    )

    # Retryable errors - rate limits
    exception = create_fake_exception(
        "anthropic", "RateLimitError", "Rate limit exceeded"
    )
    assert llm.need_to_be_retried(exception) is True

    # Retryable errors - overloaded
    exception = create_fake_exception(
        "anthropic", "OverloadedError", "API is overloaded"
    )
    assert llm.need_to_be_retried(exception) is True

    # Retryable errors - internal path for OverloadedError
    exception = create_fake_exception(
        "anthropic._exceptions", "OverloadedError", "API is overloaded"
    )
    assert llm.need_to_be_retried(exception) is True

    # Retryable errors - internal server error
    exception = create_fake_exception(
        "anthropic", "InternalServerError", "Internal server error"
    )
    assert llm.need_to_be_retried(exception) is True

    # Retryable errors - connection error
    exception = create_fake_exception(
        "anthropic", "APIConnectionError", "Connection failed"
    )
    assert llm.need_to_be_retried(exception) is True

    # Retryable errors - timeout error
    exception = create_fake_exception(
        "anthropic", "APITimeoutError", "Request timed out"
    )
    assert llm.need_to_be_retried(exception) is True

    # Retryable errors - APIStatusError with retryable status codes
    for status_code in [429, 500, 502, 503, 504, 529]:
        exception = create_fake_exception(
            "anthropic",
            "APIStatusError",
            f"Error with status {status_code}",
            status_code=status_code,
        )
        assert (
            llm.need_to_be_retried(exception) is True
        ), f"Status {status_code} should be retryable"

    # Non-retryable errors - APIStatusError with non-retryable status codes
    for status_code in [400, 401, 403, 404, 413]:
        exception = create_fake_exception(
            "anthropic",
            "APIStatusError",
            f"Error with status {status_code}",
            status_code=status_code,
        )
        assert (
            llm.need_to_be_retried(exception) is False
        ), f"Status {status_code} should NOT be retryable"

    # Non-retryable errors - context length error (prompt too long)
    exception = create_fake_exception(
        "anthropic", "RateLimitError", "prompt is too long"
    )
    assert llm.need_to_be_retried(exception) is False

    # Non-retryable errors - unknown error type
    exception = create_fake_exception("anthropic", "SomeUnknownError", "Unknown error")
    assert llm.need_to_be_retried(exception) is False

    # Non-retryable errors - KeyboardInterrupt
    exception = KeyboardInterrupt()
    assert llm.need_to_be_retried(exception) is False
