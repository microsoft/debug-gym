import sys
import unittest
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
    merge_messages,
    print_messages,
)


class TestLLMAPI(unittest.TestCase):

    @pytest.fixture
    def async_llm(self):
        # Create an instance of AsyncLLM with a mock configuration
        model_name = "test-model"
        verbose = False
        async_llm = AsyncLLM(model_name, verbose)
        return async_llm

    def test_is_rate_limit_error(self):
        mock_response = MagicMock()
        mock_response.request = "example"
        mock_response.body = {"error": "Rate limit exceeded"}
        # Instantiate the RateLimitError with the mock response
        exception = RateLimitError(
            "Rate limit exceeded", response=mock_response, body=mock_response.body
        )
        assert is_rate_limit_error(exception) == True

    def test_print_messages(self):
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

    def test_merge_messages(self):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "user", "content": "How are you?"},
            {"role": "assistant", "content": "Hi"},
        ]
        merged = merge_messages(messages)
        assert len(merged) == 2
        assert merged[0]["content"] == "Hello\n\nHow are you?"

    @patch("tiktoken.encoding_for_model")
    def test_token_counter(self, mock_encoding_for_model):
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
    def test_llm(self, mock_open, mock_exists, mock_openai, mock_encoding_for_model):
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

    @pytest.mark.asyncio
    @patch("llm_api.AsyncAzureOpenAI")
    async def test_async_llm(mock_async_openai):
        mock_async_openai.return_value.chat.completions.create.return_value.choices[
            0
        ].message.content = "Response"
        llm = AsyncLLM(model_name="test_model", verbose=False)
        messages = [{"role": "user", "content": "Hello"}]
        response, token_usage = await llm(messages)
        assert response == "Response"
        assert "prompt" in token_usage
        assert "response" in token_usage

    @pytest.mark.asyncio
    @patch("llm_api.AsyncLLM.query_model", new_callable=AsyncMock)
    @patch(
        "llm_api.merge_messages",
        return_value=[{"role": "user", "content": "Test message"}],
    )
    @patch(
        "llm_api.trim_prompt_messages",
        return_value=[{"role": "user", "content": "Test message"}],
    )
    async def test_async_llm_call(mock_trim, mock_merge, mock_query_model, async_llm):
        # Set up the mock return value for query_model
        mock_query_model.return_value = "Test response"

        # Define the input messages
        messages = [{"role": "user", "content": "Test message"}]

        # Call the __call__ method
        response, token_usage = await async_llm(messages)

        # Assert the response and token usage
        assert response == "Test response"
        assert token_usage == {
            "prompt": async_llm.token_counter(messages=messages),
            "response": async_llm.token_counter(text="Test response"),
        }

        # Assert that the mock methods were called
        mock_merge.assert_called_once_with(messages)
        mock_trim.assert_called_once_with(
            [{"role": "user", "content": "Test message"}],
            async_llm.context_length,
            async_llm.token_counter,
        )
        mock_query_model.assert_called_once_with(
            [{"role": "user", "content": "Test message"}]
        )

    @patch("builtins.input", return_value="User input")
    def test_human(self, mock_prompt):
        human = Human()
        messages = [{"role": "user", "content": "Hello"}]
        info = {"available_commands": ["command1", "command2"]}
        response, token_usage = human(messages, info)
        assert response == "User input"
        assert "prompt" in token_usage
        assert "response" in token_usage

    def test_random(self):
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
    def test_instantiate_llm(self, mock_open, mock_exists, mock_encoding_for_model):
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
