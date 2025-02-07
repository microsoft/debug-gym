import json
import logging
from unittest.mock import AsyncMock, MagicMock, mock_open, patch

import pytest
from openai import RateLimitError

from example_agent.llm_api import (
    LLM,
    AsyncLLM,
    Human,
    TokenCounter,
    instantiate_llm,
    load_llm_config,
    merge_messages,
    print_messages,
)
from froggy.logger import FroggyLogger


@pytest.fixture
def logger_mock():
    logger = FroggyLogger("test_logger")
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

    counter = TokenCounter(model="gpt-4o")
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
    response, token_usage = llm(messages)
    assert response == "Response"
    assert "prompt" in token_usage
    assert "response" in token_usage


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
    response, token_usage = await llm(messages)
    assert response == "some completion mock."
    assert token_usage == {"prompt": 1, "response": 4}


@patch("builtins.input", lambda *args, **kwargs: "User input")
def test_human():
    human = Human()
    messages = [{"role": "user", "content": "Hello"}]
    info = {
        "tools": {
            "pdb": {"template": "```pdb <command>```"},
            "view": {"template": "```<path/to/file.py>```"},
        }
    }
    response, token_usage = human(messages, info)
    assert response == "User input"
    assert "prompt" in token_usage
    assert "response" in token_usage


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
