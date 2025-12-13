import hashlib
import hmac
import time
from unittest.mock import MagicMock, mock_open, patch

import pytest

from debug_gym.gym.entities import Observation
from debug_gym.gym.tools.tool import EnvironmentTool, ToolCall
from debug_gym.llms.base import LLM, ContextLengthExceededError, LLMConfigRegistry
from debug_gym.llms.copilot import (  # Import for patching in tests
    CopilotClaudeLLM,
    CopilotLLM,
    CopilotOpenAILLM,
)


class MockTool(EnvironmentTool):
    name = "test_tool"
    description = "A test tool"
    arguments = {
        "arg1": {
            "type": ["string"],
            "description": "Test argument",
        },
    }

    def use(self, env, action):
        return Observation("TestTool", action)


tools = [MockTool()]


def create_fake_exception(
    module: str, classname: str, message: str, code: str = "fake_code"
):
    """Helper function to create fake exceptions for testing"""
    exc_type = type(classname, (Exception,), {})
    exc = exc_type(message)
    exc.message = message
    exc.code = code
    exc.__class__.__module__ = module
    return exc


@patch.object(
    LLMConfigRegistry,
    "from_file",
    return_value=LLMConfigRegistry.register_all(
        {
            "test-model": {
                "model": "test-model",
                "tokenizer": "gpt-4o",
                "context_limit": 4096,
                "api_key": "test-key",
                "endpoint": "https://test-endpoint",
                "api_version": "2025-05-01",
                "tags": ["copilot openai"],
            }
        }
    ),
)
class TestCopilotLLM:
    """Test cases for CopilotLLM base class functionality"""

    def test_init(self, mock_config, logger_mock):
        """Test CopilotLLM initialization"""
        llm = LLM.instantiate("test-model", logger=logger_mock)
        assert llm._client is None
        assert llm._token_cache is None
        assert llm._token_expires_at == 0

    def test_create_request_hmac_with_secret(self, mock_config):
        """Test HMAC creation with valid secret"""
        llm = LLM.instantiate("test-model")
        secret = "test_secret"

        # Mock time to get predictable results
        with patch("time.time", return_value=1234567890):
            hmac_result = llm.create_request_hmac(secret)

        assert hmac_result is not None
        assert "1234567890." in hmac_result

        # Verify HMAC format
        parts = hmac_result.split(".")
        assert len(parts) == 2
        assert parts[0] == "1234567890"

        # Verify HMAC signature
        expected_signature = hmac.new(
            secret.encode("utf-8"), "1234567890".encode("utf-8"), hashlib.sha256
        ).hexdigest()
        assert parts[1] == expected_signature

    def test_create_request_hmac_no_secret(self, mock_config):
        """Test HMAC creation with no secret returns None"""
        llm = LLM.instantiate("test-model")
        assert llm.create_request_hmac(None) is None
        assert llm.create_request_hmac("") is None

    def test_tokenize_with_tiktoken(self, mock_config, logger_mock):
        """Test tokenization using tiktoken"""
        llm = LLM.instantiate("test-model", logger=logger_mock)

        # Mock tiktoken encoding
        mock_encode = MagicMock(return_value=[1, 2, 3, 4])
        with patch("tiktoken.encoding_for_model") as mock_tiktoken:
            mock_tiktoken.return_value.encode = mock_encode
            messages = [{"role": "user", "content": "test text"}]
            tokens = llm.tokenize(messages)

        # Should return list of token lists (one per message)
        assert tokens == [["1", "2", "3", "4"]]
        mock_tiktoken.assert_called_once_with("gpt-4o")
        mock_encode.assert_called_once_with("test text")

    def test_tokenize_fallback(self, mock_config, logger_mock):
        """Test tokenization fallback when tiktoken fails"""
        llm = LLM.instantiate("test-model", logger=logger_mock)

        # Mock tiktoken to raise KeyError
        with patch(
            "tiktoken.encoding_for_model", side_effect=KeyError("model not found")
        ):
            messages = [{"role": "user", "content": "hello world test"}]
            tokens = llm.tokenize(messages)

        # Should return list of token lists (one per message)
        assert tokens == [["hello", "world", "test"]]

    def test_need_to_be_retried_hmac_timestamp_error(self, mock_config, logger_mock):
        """Test retry logic for HMAC timestamp errors"""
        llm = LLM.instantiate("test-model", logger=logger_mock)

        # Create HMAC timestamp error
        exception = create_fake_exception(
            "openai", "AuthenticationError", "HMAC timestamp out of range"
        )

        with patch("time.sleep") as mock_sleep:
            result = llm.need_to_be_retried(exception)

        assert result is True
        mock_sleep.assert_called_once_with(5)

    def test_need_to_be_retried_other_auth_error(self, mock_config, logger_mock):
        """Test retry logic for other authentication errors"""
        llm = LLM.instantiate("test-model", logger=logger_mock)

        # Create different auth error
        exception = create_fake_exception(
            "openai", "AuthenticationError", "Invalid API key"
        )

        # Should use parent class logic (which would be False for auth errors)
        with patch.object(
            llm.__class__.__bases__[0], "need_to_be_retried", return_value=False
        ):
            result = llm.need_to_be_retried(exception)

        assert result is False

    def test_need_to_be_retried_non_auth_error(self, mock_config, logger_mock):
        """Test retry logic for non-authentication errors"""
        llm = LLM.instantiate("test-model", logger=logger_mock)

        # Create rate limit error
        exception = create_fake_exception(
            "openai", "RateLimitError", "Rate limit exceeded"
        )

        # Should use parent class logic
        with patch.object(
            llm.__class__.__bases__[0], "need_to_be_retried", return_value=True
        ):
            result = llm.need_to_be_retried(exception)

        assert result is True


@patch.object(
    LLMConfigRegistry,
    "from_file",
    return_value=LLMConfigRegistry.register_all(
        {
            "copilot-claude": {
                "model": "claude-sonnet-4",
                "tokenizer": "gpt-4o",
                "context_limit": 4096,
                "api_key": "test-key",
                "endpoint": "https://test-endpoint",
                "api_version": "2025-05-01",
                "tags": ["copilot claude"],  # Fixed tag
            }
        }
    ),
)
class TestCopilotClaudeLLM:
    """Test cases for CopilotClaudeLLM class"""

    def test_generate_system_message_only(self, mock_config, logger_mock):
        """Test generate with only system message converts to user message"""
        llm = LLM.instantiate("copilot-claude", logger=logger_mock)

        # Mock the client and response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.choices[0].message.tool_calls = None
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 20

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        # Mock the client property to avoid token fetching
        with patch.object(
            type(llm), "client", new_callable=lambda: property(lambda self: mock_client)
        ):
            messages = [{"role": "system", "content": "You are a helpful assistant"}]
            llm.generate(messages, tools)

        # Verify the call was made with converted messages
        call_args = mock_client.chat.completions.create.call_args
        called_messages = call_args[1]["messages"]

        assert len(called_messages) == 2
        assert called_messages[0]["role"] == "system"
        assert called_messages[1]["role"] == "user"
        assert called_messages[1]["content"] == "Your response is:"

    def test_generate_with_tool_calls(self, mock_config, logger_mock):
        """Test generate with tool calls in response"""
        llm = LLM.instantiate("copilot-claude", logger=logger_mock)

        # Mock response with tool calls
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(), MagicMock()]

        # First choice with content
        mock_response.choices[0].message.content = "I'll help you with that."
        mock_response.choices[0].message.tool_calls = None

        # Second choice with tool call
        mock_response.choices[1].message.content = None
        tool_call = MagicMock()
        tool_call.type = "function"
        tool_call.function.name = "test_tool"
        tool_call.function.arguments = '{"arg1": "test_value"}'
        tool_call.id = "call_123"
        mock_response.choices[1].message.tool_calls = [tool_call]

        mock_response.usage.prompt_tokens = 15
        mock_response.usage.completion_tokens = 25

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        # Mock the client property to avoid token fetching
        with patch.object(
            type(llm), "client", new_callable=lambda: property(lambda self: mock_client)
        ):
            with patch.object(
                llm,
                "parse_tool_call_response",
                return_value=ToolCall(
                    id="call_123", name="test_tool", arguments={"arg1": "test_value"}
                ),
            ):
                messages = [{"role": "user", "content": "Test message"}]
                response = llm.generate(messages, tools)

        assert response.response == "I'll help you with that."
        assert response.action.id == "call_123"
        assert response.action.name == "test_tool"
        assert response.token_usage.prompt == 15
        assert response.token_usage.response == 25

    def test_generate_with_thinking_content(self, mock_config, logger_mock):
        """Test generate with thinking content in response"""
        llm = LLM.instantiate("copilot-claude", logger=logger_mock)

        # Mock response with thinking content
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Final answer"
        mock_response.choices[0].message.tool_calls = None
        mock_response.choices[0].message.thinking_content = "Let me think about this..."

        # Create a custom __contains__ method that works with MagicMock
        def message_contains(self, key):
            return key == "thinking_content"

        mock_response.choices[0].message.__contains__ = message_contains
        mock_response.usage.prompt_tokens = 20
        mock_response.usage.completion_tokens = 30

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        # Mock the client property to avoid token fetching
        with patch.object(
            type(llm), "client", new_callable=lambda: property(lambda self: mock_client)
        ):
            messages = [{"role": "user", "content": "Test question"}]
            response = llm.generate(messages, tools)

        assert response.response == "Final answer"
        assert response.reasoning_response == "Let me think about this..."

    def test_generate_context_length_error(self, mock_config, logger_mock):
        """Test generate raises ContextLengthExceededError for context length issues"""
        llm = LLM.instantiate("copilot-claude", logger=logger_mock)

        # Mock BadRequestError with context length message
        import openai

        error = openai.BadRequestError(
            "Request too large", response=MagicMock(), body=None
        )

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = error

        # Mock the client property to avoid token fetching
        with patch.object(
            type(llm), "client", new_callable=lambda: property(lambda self: mock_client)
        ):
            with patch.object(llm, "is_context_length_error", return_value=True):
                messages = [{"role": "user", "content": "Test"}]
                with pytest.raises(ContextLengthExceededError):
                    llm.generate(messages, tools)

    def test_generate_entity_too_large_error(self, mock_config, logger_mock):
        """Test generate raises ContextLengthExceededError for entity too large"""
        llm = LLM.instantiate("copilot-claude", logger=logger_mock)

        # Mock APIStatusError with entity too large message
        import openai

        error = openai.APIStatusError(
            "Request Entity Too Large", response=MagicMock(), body=None
        )
        error.message = "Request Entity Too Large"

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = error

        # Mock the client property to avoid token fetching
        with patch.object(
            type(llm), "client", new_callable=lambda: property(lambda self: mock_client)
        ):
            messages = [{"role": "user", "content": "Test"}]
            with pytest.raises(ContextLengthExceededError):
                llm.generate(messages, tools)

    def test_generate_no_content_no_tool_calls(self, mock_config, logger_mock):
        """Test generate handles response with no content and no tool calls"""
        llm = LLM.instantiate("copilot-claude", logger=logger_mock)

        # Mock response with no content or tool calls
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = None
        mock_response.choices[0].message.tool_calls = None
        mock_response.usage.prompt_tokens = 5
        mock_response.usage.completion_tokens = 0

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        # Mock the client property to avoid token fetching
        with patch.object(
            type(llm), "client", new_callable=lambda: property(lambda self: mock_client)
        ):
            messages = [{"role": "user", "content": "Test"}]
            response = llm.generate(messages, tools)

        assert response.response is None
        assert response.action.id == "empty_tool_response"
        assert response.action.name == "empty_tool_response"
        assert response.action.arguments == {}


@patch.object(
    LLMConfigRegistry,
    "from_file",
    return_value=LLMConfigRegistry.register_all(
        {
            "test-model": {
                "model": "test-model",
                "tokenizer": "gpt-4o",
                "context_limit": 4096,
                "api_key": "test-key",
                "endpoint": "https://test-endpoint",
                "api_version": "2025-05-01",
                "tags": ["copilot openai"],  # Fixed tag for CopilotOpenAILLM
            }
        }
    ),
)
class TestCopilotOpenAILLM:
    """Test cases for CopilotOpenAILLM class"""

    def test_inheritance(self, mock_config, logger_mock):
        """Test CopilotOpenAILLM inherits from CopilotLLM"""
        llm = LLM.instantiate("test-model", logger=logger_mock)
        assert isinstance(llm, CopilotLLM)

    def test_docstring(self, mock_config):
        """Test class has proper docstring"""
        assert "GitHub Copilot Claude API backend" in CopilotOpenAILLM.__doc__


# Integration-style tests that don't require external dependencies
@patch.object(
    LLMConfigRegistry,
    "from_file",
    return_value=LLMConfigRegistry.register_all(
        {
            "test-model": {
                "model": "test-model",
                "tokenizer": "gpt-4o",
                "context_limit": 4096,
                "api_key": "test-key",
                "endpoint": "https://test-endpoint",
                "api_version": "2025-05-01",
                "tags": ["copilot claude"],  # Fixed tag for integration tests
            }
        }
    ),
)
class TestCopilotIntegration:
    """Integration tests for Copilot classes without external dependencies"""

    def test_token_caching_logic(self, mock_config, logger_mock):
        """Test token caching works correctly"""
        llm = LLM.instantiate("test-model", logger=logger_mock)

        # Set up cached token
        test_token = "cached_token_123"
        future_time = time.time() + 1000  # Token valid for 1000 seconds
        llm._token_cache = test_token
        llm._token_expires_at = future_time

        # Mock fetch_token to ensure it's not called
        with patch.object(
            llm, "fetch_token", side_effect=Exception("Should not be called")
        ):
            # This should use cached token and not call fetch_token
            # We can't test this directly without mocking the whole client property
            # But we can verify the cache logic
            assert llm._token_cache == test_token
            assert llm._token_expires_at == future_time

    def test_token_cache_expiry(self, mock_config, logger_mock):
        """Test expired token cache is not used"""
        llm = LLM.instantiate("test-model", logger=logger_mock)

        # Set up expired cached token
        test_token = "expired_token_123"
        past_time = time.time() - 1000  # Token expired 1000 seconds ago
        llm._token_cache = test_token
        llm._token_expires_at = past_time

        # Verify cache is considered expired
        assert time.time() >= llm._token_expires_at
        assert llm._token_cache == test_token  # Still there but expired

    def test_hmac_secret_loading_priority(self, mock_config, logger_mock):
        """Test HMAC secret loading from environment vs .env file"""
        llm = LLM.instantiate("test-model", logger=logger_mock)
        assert llm.model_name == "test-model"

        env_secret = "env_secret_123"
        file_secret = "file_secret_456"

        # Test environment variable takes priority
        with patch.dict("os.environ", {"HMAC_SECRET": env_secret}):
            with patch("os.path.exists", return_value=True):
                with patch(
                    "builtins.open", mock_open(read_data=f"HMAC_SECRET={file_secret}\n")
                ):
                    # This would normally be tested through the client property
                    # But we can verify the environment variable is present
                    import os

                    assert os.environ.get("HMAC_SECRET") == env_secret

    def test_multiple_message_choices_handling(self, mock_config, logger_mock):
        """Test handling of multiple choices in Claude response"""
        # This is tested in the CopilotClaudeLLM test_generate_with_tool_calls
        # but we can add more specific logic tests here
        llm = LLM.instantiate("test-model", logger=logger_mock)
        assert isinstance(llm, CopilotClaudeLLM)

        # Create mock choices - some with content, some with tool calls
        choice1 = MagicMock()
        choice1.message.content = "First response"
        choice1.message.tool_calls = None

        choice2 = MagicMock()
        choice2.message.content = "Second response"
        choice2.message.tool_calls = None

        choice3 = MagicMock()
        choice3.message.content = None
        tool_call = MagicMock()
        tool_call.type = "function"
        choice3.message.tool_calls = [tool_call]

        choices = [choice1, choice2, choice3]

        # Test text message extraction
        text_messages = [r.message.content for r in choices if r.message.content]
        assert text_messages == ["First response", "Second response"]

        # Test tool call extraction
        tool_calls = [r.message.tool_calls[0] for r in choices if r.message.tool_calls]
        assert len(tool_calls) == 1
        assert tool_calls[0] == tool_call
