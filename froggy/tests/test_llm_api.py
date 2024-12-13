import json
import sys
from io import StringIO
from unittest.mock import AsyncMock, MagicMock, mock_open, patch

import pytest
from openai import RateLimitError

from froggy.agents.llm_api import (
    LLM,
    AsyncLLM,
    Human,
    Random,
    TokenCounter,
    instantiate_llm,
    is_rate_limit_error,
    load_llm_config,
    merge_messages,
    print_messages,
)


@pytest.fixture
def async_llm():
    # Create an instance of AsyncLLM with a mock configuration
    model_name = "test-model"
    verbose = False
    async_llm = AsyncLLM(model_name, verbose)
    return async_llm


def test_is_rate_limit_error():
    mock_response = MagicMock()
    mock_response.request = "example"
    mock_response.body = {"error": "Rate limit exceeded"}
    # Instantiate the RateLimitError with the mock response
    exception = RateLimitError(
        "Rate limit exceeded", response=mock_response, body=mock_response.body
    )
    assert is_rate_limit_error(exception) == True


def test_print_messages():
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
        {"role": "system", "content": "System message"},
    ]
    captured_output = StringIO()
    sys.stdout = captured_output
    print_messages(messages)
    sys.stdout = sys.__stdout__
    captured = captured_output.getvalue()
    assert "Hello" in captured
    assert "Hi" in captured
    assert "System message" in captured


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
def test_llm(mock_open, mock_exists, mock_openai, mock_encoding_for_model):
    mock_encoding = MagicMock()
    mock_encoding.encode = lambda x: x.split()
    mock_encoding_for_model.return_value = mock_encoding

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Response"
    mock_openai.return_value = mock_response

    llm = LLM(model_name="test-model", verbose=False)
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
async def test_async_llm(llm_config_mock, completion_mock):
    llm = AsyncLLM(model_name="test_model", verbose=False)
    llm.client.chat.completions.create = completion_mock
    messages = [{"role": "user", "content": "Hello"}]
    response, token_usage = await llm(messages)
    assert response == "some completion mock."
    assert token_usage == {"prompt": 1, "response": 4}


@patch("builtins.input", return_value="User input")
def test_human(mock_prompt):
    human = Human()
    messages = [{"role": "user", "content": "Hello"}]
    info = {"available_commands": ["command1", "command2"]}
    response, token_usage = human(messages, info)
    assert response == "User input"
    assert "prompt" in token_usage
    assert "response" in token_usage


def test_random():
    random_llm = Random(seed=42, verbose=False)
    messages = [{"role": "user", "content": "Hello"}]
    info = {"available_commands": ["command1", "command2"]}
    response, token_usage = random_llm(messages, info)
    assert response in ["command1", "command2"]
    assert "prompt" in token_usage
    assert "response" in token_usage


@patch("tiktoken.encoding_for_model")
@patch("os.path.exists", return_value=True)
@patch(
    "builtins.open",
    new_callable=mock_open,
    read_data='{"test-model": {"model": "test-model", "max_tokens": 100, "tokenizer": "gpt-4o", "context_limit": 4, "api_key": "test-api-key", "endpoint": "https://test-endpoint", "api_version": "v1", "tags": ["azure openai"]}}',
)
def test_instantiate_llm(mock_open, mock_exists, mock_encoding_for_model):
    mock_encoding = MagicMock()
    mock_encoding.encode = lambda x: x.split()
    mock_encoding_for_model.return_value = mock_encoding

    config = {"llm_name": "test-model"}
    llm = instantiate_llm(config, verbose=False, use_async=False)
    assert isinstance(llm, LLM)

    config = {"llm_name": "random", "random_seed": 42}
    llm = instantiate_llm(config, verbose=False, use_async=False)
    assert isinstance(llm, Random)

    config = {"llm_name": "human"}
    llm = instantiate_llm(config, verbose=False, use_async=False)
    assert isinstance(llm, Human)
