import json
import logging
from unittest.mock import AsyncMock, MagicMock, mock_open, patch

import pytest
from openai import RateLimitError

from debug_gym.agents.llm_api import (
    LLM,
    AsyncLLM,
    Human,
    LLMResponse,
    TokenCounter,
    TokenUsage,
    instantiate_llm,
    load_llm_config,
    merge_messages,
    print_messages,
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
def async_llm(logger_mock, llm_config_mock):
    # Create an instance of AsyncLLM with a mock configuration
    model_name = "test_model"
    async_llm = AsyncLLM(model_name, logger=logger_mock)
    return async_llm


def test_is_rate_limit_error(async_llm):
    mock_response = MagicMock()
    mock_response.request = "example"
    mock_response.body = {"error": "Rate limit exceeded"}
    # Instantiate the RateLimitError with the mock response
    exception = RateLimitError(
        "Rate limit exceeded", response=mock_response, body=mock_response.body
    )
    assert async_llm.is_rate_limit_error(exception) == True


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


@patch("tiktoken.encoding_for_model")
def test_token_counter(mock_encoding_for_model):
    mock_encoding = MagicMock()
    mock_encoding.encode = lambda x: x.split()
    mock_encoding_for_model.return_value = mock_encoding

    counter = TokenCounter(model="gpt-4o", config={})
    messages = [{"content": "Hello"}, {"content": "How are you?"}]
    assert counter(messages=messages) > 0
    assert counter(text="Hello") > 0


@patch("tiktoken.encoding_for_model")
@patch("openai.resources.chat.completions.Completions.create")
@patch("os.path.exists", return_value=True)
@patch(
    "builtins.open",
    new_callable=mock_open,
    read_data='{"test-model": {"model": "test-model", "max_tokens": 100, "tokenizer": "gpt-4o", "context_limit": 4, "api_key": "test-api-key", "endpoint": "https://test-endpoint", "api_version": "v1", "tags": ["azure openai"]}}',
)
def test_llm(mock_open, mock_exists, mock_openai, mock_encoding_for_model, logger_mock):
    mock_encoding = MagicMock()
    mock_encoding.encode = lambda x: x.split()
    mock_encoding_for_model.return_value = mock_encoding

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Response"
    mock_openai.return_value = mock_response

    llm = LLM(model_name="test-model", logger=logger_mock)
    messages = [{"role": "user", "content": "Hello"}]
    llm_response = llm(messages)
    assert llm_response.prompt == messages
    assert llm_response.response == "Response"
    assert llm_response.token_usage.prompt == 1
    assert llm_response.token_usage.response == 1


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


@pytest.mark.asyncio
async def test_async_llm(llm_config_mock, completion_mock, logger_mock):
    llm = AsyncLLM(model_name="test_model", logger=logger_mock)
    llm.client.chat.completions.create = completion_mock
    messages = [{"role": "user", "content": "Hello"}]
    llm_response = await llm(messages)
    assert llm_response.prompt == messages
    assert llm_response.response == "some completion mock."
    assert llm_response.token_usage.prompt == 1
    assert llm_response.token_usage.response == 4


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


@patch("tiktoken.encoding_for_model")
@patch("os.path.exists", return_value=True)
@patch(
    "builtins.open",
    new_callable=mock_open,
    read_data='{"test-model": {"model": "test-model", "max_tokens": 100, "tokenizer": "gpt-4o", "context_limit": 4, "api_key": "test-api-key", "endpoint": "https://test-endpoint", "api_version": "v1", "tags": ["azure openai"]}}',
)
def test_instantiate_llm(mock_open, mock_exists, mock_encoding_for_model, logger_mock):
    mock_encoding = MagicMock()
    mock_encoding.encode = lambda x: x.split()
    mock_encoding_for_model.return_value = mock_encoding

    config = {"llm_name": "test-model"}
    llm = instantiate_llm(config, logger=logger_mock, use_async=False)
    assert isinstance(llm, LLM)

    config = {"llm_name": "human"}
    llm = instantiate_llm(config, logger=logger_mock, use_async=False)
    assert isinstance(llm, Human)


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
    "builtins.open",
    new_callable=mock_open,
    read_data=json.dumps(anthropic_config | anthropic_thinking_config),
)
def test_query_anthropic_model_basic(mock_open, logger_mock):
    llm = LLM("test-anthropic", logger=logger_mock)

    mock_response = MagicMock()
    mock_response.content = [MagicMock()]
    mock_response.content[0].text = "```python\nprint('Hello World')\n```"
    llm.client.messages.create = MagicMock(return_value=mock_response)

    messages = [{"role": "user", "content": "Write a Hello World program"}]
    result = llm.query_anthropic_model(messages)

    assert result == "```python\nprint('Hello World')\n```"
    llm.client.messages.create.assert_called_once()
    assert llm.client.messages.create.call_args[1]["model"] == "claude-3-opus-20240229"
    assert llm.client.messages.create.call_args[1]["max_tokens"] == 8192
    assert len(llm.client.messages.create.call_args[1]["messages"]) == 1


@patch(
    "builtins.open",
    new_callable=mock_open,
    read_data=json.dumps(anthropic_thinking_config),
)
def test_query_anthropic_model_with_thinking(mock_open, logger_mock):
    llm = LLM("test-anthropic-thinking", logger=logger_mock)

    mock_response = MagicMock()
    mock_response.content = [MagicMock(), MagicMock()]
    mock_response.content[1].text = "```python\nprint('Hello World')\n```"
    llm.client.messages.create = MagicMock(return_value=mock_response)

    messages = [{"role": "user", "content": "Write a Hello World program"}]
    result = llm.query_anthropic_model(messages)

    assert result == "```python\nprint('Hello World')\n```"
    llm.client.messages.create.assert_called_once()
    assert llm.client.messages.create.call_args[1]["model"] == "claude-3-opus-20240229"
    assert llm.client.messages.create.call_args[1]["max_tokens"] == 20000
    assert llm.client.messages.create.call_args[1]["temperature"] == 1.0
    assert llm.client.messages.create.call_args[1]["thinking"]["type"] == "enabled"
    assert llm.client.messages.create.call_args[1]["thinking"]["budget_tokens"] == 16000


@patch("builtins.open", new_callable=mock_open, read_data=json.dumps(anthropic_config))
def test_query_anthropic_model_empty_messages(mock_open, logger_mock):
    llm = LLM("test-anthropic", logger=logger_mock)

    mock_response = MagicMock()
    mock_response.content = [MagicMock()]
    mock_response.content[0].text = "```sample response```"
    llm.client.messages.create = MagicMock(return_value=mock_response)

    messages = []
    result = llm.query_anthropic_model(messages)

    # Verify default user prompt was added
    assert result == "```sample response```"
    llm.client.messages.create.assert_called_once()
    assert len(llm.client.messages.create.call_args[1]["messages"]) == 1
    assert llm.client.messages.create.call_args[1]["messages"][0]["role"] == "user"
    assert (
        llm.client.messages.create.call_args[1]["messages"][0]["content"]
        == "Your answer is: "
    )


@patch("builtins.open", new_callable=mock_open, read_data=json.dumps(anthropic_config))
def test_query_anthropic_model_with_system_prompt(mock_open, logger_mock):
    llm = LLM("test-anthropic", logger=logger_mock)

    mock_response = MagicMock()
    mock_response.content = [MagicMock()]
    mock_response.content[0].text = "```system response```"
    llm.client.messages.create = MagicMock(return_value=mock_response)

    messages = [
        {"role": "system", "content": "You are a helpful coding assistant"},
        {"role": "user", "content": "Help me with Python"},
    ]
    result = llm.query_anthropic_model(messages)

    assert result == "```system response```"
    llm.client.messages.create.assert_called_once()
    assert (
        llm.client.messages.create.call_args[1]["system"]
        == "You are a helpful coding assistant"
    )
    assert len(llm.client.messages.create.call_args[1]["messages"]) == 1
    assert llm.client.messages.create.call_args[1]["messages"][0]["role"] == "user"


@patch("builtins.open", new_callable=mock_open, read_data=json.dumps(anthropic_config))
def test_query_anthropic_model_with_conversation(mock_open, logger_mock):
    llm = LLM("test-anthropic", logger=logger_mock)

    mock_response = MagicMock()
    mock_response.content = [MagicMock()]
    mock_response.content[0].text = "```response to conversation```"
    llm.client.messages.create = MagicMock(return_value=mock_response)

    # Test with a conversation (user and assistant messages)
    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there! How can I help you?"},
        {"role": "user", "content": "I need help with Python"},
    ]
    result = llm.query_anthropic_model(messages)

    # Verify conversation handling
    assert result == "```response to conversation```"
    llm.client.messages.create.assert_called_once()
    assert (
        llm.client.messages.create.call_args[1]["system"]
        == "You are a helpful assistant"
    )
    assert len(llm.client.messages.create.call_args[1]["messages"]) == 3
    assert llm.client.messages.create.call_args[1]["messages"][0]["role"] == "user"
    assert llm.client.messages.create.call_args[1]["messages"][1]["role"] == "assistant"
    assert llm.client.messages.create.call_args[1]["messages"][2]["role"] == "user"


@patch("builtins.open", new_callable=mock_open, read_data=json.dumps(anthropic_config))
def test_query_anthropic_model_empty_content(mock_open, logger_mock):
    llm = LLM("test-anthropic", logger=logger_mock)

    mock_response = MagicMock()
    mock_response.content = [MagicMock()]
    mock_response.content[0].text = "```response```"
    llm.client.messages.create = MagicMock(return_value=mock_response)

    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": ""},  # Empty content should be skipped
        {"role": "user", "content": "Real question"},
    ]
    result = llm.query_anthropic_model(messages)

    assert result == "```response```"
    llm.client.messages.create.assert_called_once()
    assert len(llm.client.messages.create.call_args[1]["messages"]) == 1
    assert (
        llm.client.messages.create.call_args[1]["messages"][0]["content"]
        == "Real question"
    )


@patch("builtins.open", new_callable=mock_open, read_data=json.dumps(anthropic_config))
def test_query_anthropic_model_unknown_role(mock_open, logger_mock):
    llm = LLM("test-anthropic", logger=logger_mock)

    llm.client.messages.create = MagicMock()

    messages = [{"role": "unknown", "content": "This has an unknown role"}]

    with pytest.raises(ValueError, match="Unknown role: unknown"):
        llm.query_anthropic_model(messages)


@patch(
    "builtins.open",
    new_callable=mock_open,
    read_data=json.dumps(anthropic_config | {"max_tokens": 4000}),
)
def test_query_anthropic_model_max_tokens_from_config(mock_open, logger_mock):
    llm = LLM("test-anthropic", logger=logger_mock)

    mock_response = MagicMock()
    mock_response.content = [MagicMock()]
    mock_response.content[0].text = "```response```"
    llm.client.messages.create = MagicMock(return_value=mock_response)

    messages = [{"role": "user", "content": "Test message"}]
    llm.query_anthropic_model(messages)

    assert llm.client.messages.create.call_args[1]["max_tokens"] == 8192


@patch("builtins.open", new_callable=mock_open, read_data=json.dumps(anthropic_config))
def test_query_anthropic_model_no_code_block(mock_open, logger_mock):
    llm = LLM("test-anthropic", logger=logger_mock)

    mock_response = MagicMock()
    mock_response.content = [MagicMock()]
    mock_response.content[0].text = "This is a response without any code blocks"
    llm.client.messages.create = MagicMock(return_value=mock_response)

    messages = [{"role": "user", "content": "Test message"}]
    result = llm.query_anthropic_model(messages)

    assert result == ""
