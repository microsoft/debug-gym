import json
import logging
from unittest.mock import AsyncMock, MagicMock, mock_open, patch

import pytest
from openai import RateLimitError

from debug_gym.agents.llm_api import (
    AnthropicLLM,
    AzureOpenAILLM,
    Human,
    LLMResponse,
    OpenAILLM,
    TokenUsage,
    instantiate_llm,
    load_llm_config,
    merge_messages,
    print_messages,
    retry_on_rate_limit,
)
from debug_gym.logger import DebugGymLogger


@pytest.fixture
def logger_mock():
    logger = DebugGymLogger("test_logger")
    logger.setLevel(logging.DEBUG)
    logs = []

    class ListHandler(logging.Handler):
        def emit(self, record):
            logs.append(record.getMessage())

    handler = ListHandler()
    logger.addHandler(handler)
    logger._log_history = logs
    return logger


@pytest.fixture
def openai_llm(logger_mock, llm_config_mock):
    # Create an instance of AsyncOpenAILLM with a mock configuration
    model_name = "test_model"
    _async_llm = OpenAILLM(model_name, logger=logger_mock)
    return _async_llm


def test_is_rate_limit_error(openai_llm):
    mock_response = MagicMock()
    mock_response.request = "example"
    mock_response.body = {"error": "Rate limit exceeded"}
    # Instantiate the RateLimitError with the mock response
    exception = RateLimitError(
        "Rate limit exceeded", response=mock_response, body=mock_response.body
    )
    assert openai_llm.is_rate_limit_error(exception) == True


def test_print_messages(logger_mock):
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
        {"role": "system", "content": "System message"},
    ]
    print_messages(messages, logger_mock)
    assert logger_mock._log_history == ["Hello\n", "Hi\n", "System message\n"]


def test_merge_messages():
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "user", "content": "How are you?"},
        {"role": "assistant", "content": "Hi"},
    ]
    merged = merge_messages(messages)
    assert len(merged) == 2
    assert merged[0]["content"] == "Hello\n\nHow are you?"

    # Ignore empty message
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "user", "content": ""},
        {"role": "user", "content": "How are you?"},
        {"role": "user", "content": ""},
        {"role": "assistant", "content": "Hi"},
    ]
    merged = merge_messages(messages)
    assert len(merged) == 2
    assert merged[0]["content"] == "Hello\n\nHow are you?"


@patch("openai.resources.chat.completions.Completions.create")
@patch(
    "debug_gym.agents.llm_api.load_llm_config",
    return_value={
        "openai": {
            "model": "openai",
            "max_tokens": 100,
            "tokenizer": "gpt-4o",
            "context_limit": 4,
            "api_key": "test-api-key",
            "endpoint": "https://test-endpoint",
            "api_version": "v1",
            "tags": ["azure openai"],
        }
    },
)
def test_llm(mock_llm_config, mock_openai, logger_mock):
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Some response from OpenAI"
    mock_openai.return_value = mock_response

    llm = OpenAILLM(model_name="openai", logger=logger_mock)
    messages = [{"role": "user", "content": "Hello World"}]
    llm_response = llm(messages)
    assert llm_response.prompt == messages
    assert llm_response.response == "Some response from OpenAI"
    assert llm_response.token_usage.prompt == 2
    assert llm_response.token_usage.response == 5


@pytest.fixture
def llm_config_mock(tmp_path, monkeypatch):
    config_file = tmp_path / "llm.cfg"
    config_file.write_text(
        json.dumps(
            {
                "test_model": {
                    "model": "test_model",
                    "tokenizer": "gpt-4o",
                    "endpoint": "https://test_endpoint",
                    "api_key": "test_api",
                    "api_version": "1.0",
                    "context_limit": 128,
                    "tags": ["azure openai"],
                }
            }
        )
    )
    monkeypatch.setenv("LLM_CONFIG_FILE", str(config_file))
    return config_file


def test_load_llm_config(llm_config_mock):
    config = load_llm_config()
    assert "test_model" in config


def test_load_llm_config_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_llm_config(str(tmp_path / "llm.cfg"))


@pytest.fixture
def completion_mock():
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "some completion mock."
    return AsyncMock(return_value=mock_response)


@patch("builtins.input", lambda *args, **kwargs: "User input")
def test_human(build_env_info):
    human = Human()
    messages = [{"role": "user", "content": "Hello"}]
    env_info = build_env_info(
        tools={
            "pdb": {"template": "```pdb <command>```"},
            "view": {"template": "```<path/to/file.py>```"},
        }
    )
    llm_response = human(messages, env_info)
    # human only uses the messages content
    assert llm_response.prompt == "Hello"
    assert llm_response.response == "User input"
    assert llm_response.token_usage.prompt == 5
    assert llm_response.token_usage.response == 10


@patch(
    "debug_gym.agents.llm_api.load_llm_config",
    return_value={
        "gpt-4o-mini-azure": {
            "model": "gpt-4o-mini_2024-07-18",
            "max_tokens": 100,
            "tokenizer": "gpt-4o-mini",
            "context_limit": 4,
            "api_key": "test-api-key",
            "endpoint": "https://test-endpoint",
            "api_version": "v1",
            "tags": ["azure openai"],
        },
        "gpt-4o-mini": {
            "model": "gpt-4o-mini_2024-07-18",
            "max_tokens": 100,
            "tokenizer": "gpt-4o-mini",
            "context_limit": 4,
            "api_key": "test-api-key",
            "endpoint": "https://test-endpoint",
            "api_version": "v1",
            "tags": ["openai"],
        },
        "claude-3.7": {
            "model": "claude-3-7-sonnet-20250219",
            "max_tokens": 100,
            "tokenizer": "claude-3-7-sonnet-20250219",
            "context_limit": 4,
            "api_key": "test-api-key",
            "tags": ["anthropic", "claude", "claude-3.7"],
        },
    },
)
def test_instantiate_llm(mock_open, logger_mock):
    # tags are used to filter models
    config = {"llm_name": "gpt-4o-mini"}
    llm = instantiate_llm(config, logger=logger_mock)
    assert isinstance(llm, OpenAILLM)

    config = {"llm_name": "gpt-4o-mini-azure"}
    llm = instantiate_llm(config, logger=logger_mock)
    assert isinstance(llm, AzureOpenAILLM)

    config = {"llm_name": "claude-3.7"}
    llm = instantiate_llm(config, logger=logger_mock)
    assert isinstance(llm, AnthropicLLM)

    config = {"llm_name": "human"}
    llm = instantiate_llm(config, logger=logger_mock)
    assert isinstance(llm, Human)

    unknown = "unknown"
    config = {"llm_name": unknown}
    with pytest.raises(ValueError, match="Model unknown not found in llm.cfg.*"):
        instantiate_llm(config, logger=logger_mock)


def test_llm_response_init_with_prompt_and_response():
    prompt = [{"role": "user", "content": "Hello"}]
    response = "Hi"
    prompt_token_count = 1
    response_token_count = 1
    llm_response = LLMResponse(
        prompt=prompt,
        response=response,
        prompt_token_count=prompt_token_count,
        response_token_count=response_token_count,
    )

    assert llm_response.prompt == prompt
    assert llm_response.response == response
    assert llm_response.token_usage.prompt == prompt_token_count
    assert llm_response.token_usage.response == response_token_count


def test_llm_response_init_with_token_usage():
    llm_response = LLMResponse("prompt", "response", token_usage=TokenUsage(1, 1))
    assert llm_response.prompt == "prompt"
    assert llm_response.response == "response"
    assert llm_response.token_usage.prompt == 1
    assert llm_response.token_usage.response == 1


def test_llm_response_init_with_prompt_and_response_only():
    llm_response = LLMResponse("prompt", "response")
    assert llm_response.prompt == "prompt"
    assert llm_response.response == "response"
    assert llm_response.token_usage == None


anthropic_config = {
    "test-anthropic": {
        "model": "claude-3-opus-20240229",
        "tokenizer": "claude-3-opus-20240229",
        "endpoint": "https://test-endpoint",
        "api_key": "test-api-key",
        "context_limit": 128,
        "tags": ["anthropic"],
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
    }
}


@patch(
    "debug_gym.agents.llm_api.load_llm_config",
    return_value=anthropic_config | anthropic_thinking_config,
)
def test_query_anthropic_model_basic(mock_llm_config, logger_mock):
    llm = AnthropicLLM("test-anthropic", logger=logger_mock)

    mock_response = MagicMock()
    mock_response.content = [MagicMock()]
    mock_response.content[0].text = "```python\nprint('Hello World')\n```"
    llm.client.messages.create = MagicMock(return_value=mock_response)
    llm.count_tokens = MagicMock(return_value=10)

    messages = [{"role": "user", "content": "Write a Hello World program"}]
    llm_response = llm(messages)

    assert llm_response.prompt == [
        {"role": "user", "content": "Write a Hello World program"}
    ]
    assert llm_response.response == "```python\nprint('Hello World')\n```"
    assert llm_response.token_usage.prompt == 10  # from mock
    assert llm_response.token_usage.response == 10  # from mock

    llm.client.messages.create.assert_called_once()
    assert llm.client.messages.create.call_args[1]["model"] == "claude-3-opus-20240229"
    assert llm.client.messages.create.call_args[1]["max_tokens"] == 8192
    assert len(llm.client.messages.create.call_args[1]["messages"]) == 1


@patch(
    "debug_gym.agents.llm_api.load_llm_config",
    return_value=anthropic_thinking_config,
)
def test_query_anthropic_model_with_thinking(mock_llm_config, logger_mock):
    llm = AnthropicLLM("test-anthropic-thinking", logger=logger_mock)

    mock_response = MagicMock()
    mock_response.content = [MagicMock(), MagicMock()]
    mock_response.content[1].text = "```python\nprint('Hello World')\n```"
    llm.client.messages.create = MagicMock(return_value=mock_response)
    llm.count_tokens = MagicMock(return_value=10)

    messages = [{"role": "user", "content": "Write a Hello World program"}]

    llm_response = llm(messages)
    assert llm_response.prompt == [
        {"role": "user", "content": "Write a Hello World program"}
    ]
    assert llm_response.response == "```python\nprint('Hello World')\n```"
    assert llm_response.token_usage.prompt == 10  # from mock
    assert llm_response.token_usage.response == 10  # from mock

    llm.client.messages.create.assert_called_once()
    assert llm.client.messages.create.call_args[1]["model"] == "claude-3-opus-20240229"
    assert llm.client.messages.create.call_args[1]["max_tokens"] == 20000
    assert llm.client.messages.create.call_args[1]["temperature"] == 1.0
    assert llm.client.messages.create.call_args[1]["thinking"]["type"] == "enabled"
    assert llm.client.messages.create.call_args[1]["thinking"]["budget_tokens"] == 16000


# DOES THIS TEST MAKE SENSE?
@patch("debug_gym.agents.llm_api.load_llm_config", return_value=anthropic_config)
def test_query_anthropic_model_empty_messages(mock_llm_config, logger_mock):
    llm = AnthropicLLM("test-anthropic", logger=logger_mock)

    mock_response = MagicMock()
    mock_response.content = [MagicMock()]
    mock_response.content[0].text = "```sample response```"
    llm.client.messages.create = MagicMock(return_value=mock_response)
    llm.count_tokens = MagicMock(return_value=10)

    messages = []
    llm_response = llm(messages)

    # Verify default user prompt was added
    assert llm_response.prompt == []
    assert llm_response.response == "```sample response```"
    llm.client.messages.create.assert_called_once()
    assert len(llm.client.messages.create.call_args[1]["messages"]) == 1
    assert llm.client.messages.create.call_args[1]["messages"][0]["role"] == "user"
    assert (
        llm.client.messages.create.call_args[1]["messages"][0]["content"]
        == "Your answer is: "
    )


@patch("debug_gym.agents.llm_api.load_llm_config", return_value=anthropic_config)
def test_query_anthropic_model_with_system_prompt(mock_llm_config, logger_mock):
    llm = AnthropicLLM("test-anthropic", logger=logger_mock)

    mock_response = MagicMock()
    mock_response.content = [MagicMock()]
    mock_response.content[0].text = "```system response```"
    llm.client.messages.create = MagicMock(return_value=mock_response)
    llm.count_tokens = MagicMock(return_value=10)

    messages = [
        {"role": "system", "content": "You are a helpful coding assistant"},
        {"role": "user", "content": "Help me with Python"},
    ]
    llm_response = llm(messages)

    assert llm_response.prompt == messages
    assert llm_response.response == "```system response```"
    llm.client.messages.create.assert_called_once()
    assert (
        llm.client.messages.create.call_args[1]["system"]
        == "You are a helpful coding assistant"
    )
    assert len(llm.client.messages.create.call_args[1]["messages"]) == 1
    assert llm.client.messages.create.call_args[1]["messages"][0]["role"] == "user"


@patch("debug_gym.agents.llm_api.load_llm_config", return_value=anthropic_config)
def test_query_anthropic_model_with_conversation(mock_llm_config, logger_mock):
    llm = AnthropicLLM("test-anthropic", logger=logger_mock)
    mock_response = MagicMock()
    mock_response.content = [MagicMock()]
    mock_response.content[0].text = "```response to conversation```"
    llm.client.messages.create = MagicMock(return_value=mock_response)
    llm.count_tokens = MagicMock(return_value=10)

    # Test with a conversation (user and assistant messages)
    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there! How can I help you?"},
        {"role": "user", "content": "I need help with Python"},
    ]
    mock_response = llm(messages)

    # Verify conversation handling
    assert mock_response.prompt == messages
    assert mock_response.response == "```response to conversation```"
    llm.client.messages.create.assert_called_once()
    assert (
        llm.client.messages.create.call_args[1]["system"]
        == "You are a helpful assistant"
    )
    assert len(llm.client.messages.create.call_args[1]["messages"]) == 3
    assert llm.client.messages.create.call_args[1]["messages"][0]["role"] == "user"
    assert llm.client.messages.create.call_args[1]["messages"][1]["role"] == "assistant"
    assert llm.client.messages.create.call_args[1]["messages"][2]["role"] == "user"


@patch("debug_gym.agents.llm_api.load_llm_config", return_value=anthropic_config)
def test_query_anthropic_model_empty_content(mock_llm_config, logger_mock):
    llm = AnthropicLLM("test-anthropic", logger=logger_mock)
    mock_response = MagicMock()
    mock_response.content = [MagicMock()]
    mock_response.content[0].text = "```response```"
    llm.client.messages.create = MagicMock(return_value=mock_response)
    llm.count_tokens = MagicMock(return_value=10)

    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": ""},  # Empty content should be skipped
        {"role": "user", "content": "Real question"},
    ]
    result = llm(messages)

    assert result.response == "```response```"
    llm.client.messages.create.assert_called_once()
    assert len(llm.client.messages.create.call_args[1]["messages"]) == 1
    assert (
        llm.client.messages.create.call_args[1]["messages"][0]["content"]
        == "Real question"
    )


@patch("debug_gym.agents.llm_api.load_llm_config", return_value=anthropic_config)
def test_query_anthropic_model_unknown_role(mock_llm_config, logger_mock):
    llm = AnthropicLLM("test-anthropic", logger=logger_mock)
    llm.client.messages.create = MagicMock()
    llm.count_tokens = MagicMock(return_value=10)
    messages = [{"role": "unknown", "content": "This has an unknown role"}]
    with pytest.raises(ValueError, match="Unknown role: .* unknown .*"):
        llm(messages)


@patch(
    "debug_gym.agents.llm_api.load_llm_config",
    return_value=anthropic_config | {"max_tokens": 4000},
)
def test_query_anthropic_model_max_tokens_from_config(mock_llm_config, logger_mock):
    llm = AnthropicLLM("test-anthropic", logger=logger_mock)
    mock_response = MagicMock()
    mock_response.content = [MagicMock()]
    mock_response.content[0].text = "```response```"
    llm.client.messages.create = MagicMock(return_value=mock_response)
    llm.count_tokens = MagicMock(return_value=10)
    messages = [{"role": "user", "content": "Test message"}]
    llm(messages)
    assert llm.client.messages.create.call_args[1]["max_tokens"] == 8192


@patch("debug_gym.agents.llm_api.load_llm_config", return_value=anthropic_config)
def test_query_anthropic_model_no_code_block(mock_llm_config, logger_mock):
    llm = AnthropicLLM("test-anthropic", logger=logger_mock)
    mock_response = MagicMock()
    mock_response.content = [MagicMock()]
    mock_response.content[0].text = "This is a response without any code blocks"
    llm.client.messages.create = MagicMock(return_value=mock_response)
    llm.count_tokens = MagicMock(return_value=10)
    messages = [{"role": "user", "content": "Test message"}]
    result = llm(messages)
    assert result.response == ""


def test_retry_on_rate_limit_success_after_retry():
    mock_func = MagicMock(side_effect=[ValueError(), OSError(), "success"])
    mock_is_rate_limit_error = MagicMock(return_value=True)

    result = retry_on_rate_limit(mock_func, mock_is_rate_limit_error)("test_arg")

    assert result == "success"
    assert mock_func.call_count == 3
    mock_func.assert_called_with("test_arg")
    assert mock_is_rate_limit_error.call_count == 2


def test_retry_on_rate_limit_raises_error():
    mock_func = MagicMock(side_effect=[ValueError(), OSError(), "success"])
    mock_is_rate_limit_error = lambda e: isinstance(e, ValueError)

    with pytest.raises(OSError):
        retry_on_rate_limit(mock_func, mock_is_rate_limit_error)("test_arg")

    assert mock_func.call_count == 2
    mock_func.assert_called_with("test_arg")


def test_retry_on_rate_limit_skip_keyboard_interrupt():
    mock_func = MagicMock(side_effect=KeyboardInterrupt())
    mock_is_rate_limit_error = MagicMock()

    # Do not retry on KeyboardInterrupt and let it propagate
    with pytest.raises(KeyboardInterrupt):
        retry_on_rate_limit(mock_func, mock_is_rate_limit_error)("test_arg")

    mock_func.assert_called_once_with("test_arg")
    # The error checker should never be called for KeyboardInterrupt
    mock_is_rate_limit_error.assert_not_called()
