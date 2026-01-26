from dataclasses import make_dataclass
from unittest.mock import MagicMock, patch

from debug_gym.gym.entities import Observation
from debug_gym.gym.tools.tool import EnvironmentTool, ToolCall
from debug_gym.llms.base import LLM, LLMConfigRegistry
from debug_gym.llms.openai import OpenAILLM  # Import for patching in tests


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


def create_fake_exception(module: str, classname: str, message: str, code: str):
    exc_type = type(classname, (Exception,), {})
    exc = exc_type(message)
    exc.message = message
    exc.code = code
    exc.__class__.__module__ = module
    return exc


@patch("openai.resources.chat.completions.Completions.create")
@patch.object(
    LLMConfigRegistry,
    "from_file",
    return_value=LLMConfigRegistry.register_all(
        {
            "openai": {
                "model": "openai",
                "tokenizer": "gpt-4o",
                "context_limit": 4,
                "api_key": "test-api-key",
                "endpoint": "https://test-endpoint",
                "api_version": "v1",
                "tags": ["azure openai"],
            }
        }
    ),
)
def test_llm(mock_llm_config, mock_openai, logger_mock):
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.tool_calls = [MagicMock()]
    mock_response.choices[0].message.content = "Test response content"
    mock_response.usage.prompt_tokens = 2
    mock_response.usage.completion_tokens = 4

    tmp_dict = {"arguments": '{"arg 1":0}', "name": "tool 1"}
    tmp_dataclass = make_dataclass("tmp", ((k, type(v)) for k, v in tmp_dict.items()))(
        **tmp_dict
    )
    tmp_dict = dict(id="1", function=tmp_dataclass, type="function")
    mock_response.choices[0].message.tool_calls[0] = make_dataclass(
        "tmp", ((k, type(v)) for k, v in tmp_dict.items())
    )(**tmp_dict)
    mock_openai.return_value = mock_response

    llm = LLM.instantiate(name="openai", logger=logger_mock)
    messages = [{"role": "user", "content": "Hello World"}]
    llm_response = llm(messages, tools)
    assert llm_response.prompt == messages
    assert llm_response.tool == ToolCall(id="1", name="tool 1", arguments={"arg 1": 0})
    assert llm_response.token_usage.prompt == 2
    assert llm_response.token_usage.response == 4


@patch.object(
    LLMConfigRegistry,
    "from_file",
    return_value=LLMConfigRegistry.register_all(
        {
            "openai": {
                "model": "openai",
                "context_limit": 4096,
                "api_key": "fake",
                "endpoint": "fake",
                "api_version": "1",
                "tags": ["openai"],
            },
            "qwen": {
                "model": "qwen",
                "context_limit": 4096,
                "api_key": "fake",
                "endpoint": "fake",
                "api_version": "1",
                "tags": ["vllm"],
            },
        }
    ),
)
def test_need_to_be_retried(llm_config_registry_mock, logger_mock):
    openai_llm = LLM.instantiate("openai", logger=logger_mock)
    qwen_llm = LLM.instantiate("qwen", logger=logger_mock)

    exception = create_fake_exception(
        "openai", "RateLimitError", "Rate limit exceeded", "fake code"
    )
    assert openai_llm.need_to_be_retried(exception) is True

    exception = create_fake_exception(
        "openai",
        "APIStatusError",
        "Error occurred: 'status': 429 rate limit",
        "fake code",
    )
    assert openai_llm.need_to_be_retried(exception) is True

    exception = create_fake_exception(
        "openai",
        "APIStatusError",
        "Encountered error: 'status': 504 gateway timeout",
        "fake code",
    )
    assert openai_llm.need_to_be_retried(exception) is True

    exception = create_fake_exception(
        "openai",
        "APIStatusError",
        "Encountered error: 'status': 504 gateway timeout",
        "model_max_prompt_tokens_exceeded",
    )
    assert openai_llm.need_to_be_retried(exception) is False

    exception = create_fake_exception(
        "openai",
        "APIStatusError",
        "Encountered error: maximum context length exceeded",
        "fake code",
    )
    assert openai_llm.need_to_be_retried(exception) is False

    exception = create_fake_exception(
        "openai",
        "APIStatusError",
        "Failure: 'status': 413 A previous prompt was too large. Please shorten input.",
        "fake code",
    )
    assert openai_llm.need_to_be_retried(exception) is True

    exception = create_fake_exception(
        "openai",
        "APIStatusError",
        "Error: 'status': 500 internal server error",
        "fake code",
    )
    assert openai_llm.need_to_be_retried(exception) is False

    exception = create_fake_exception(
        "openai", "PermissionDeniedError", "Permission denied error", "fake code"
    )
    assert openai_llm.need_to_be_retried(exception) is True

    exception = create_fake_exception(
        "openai",
        "BadRequestError",
        "Error code: 400 \n Invalid JSON: EOF while parsing a string",
        "fake code",
    )
    assert openai_llm.need_to_be_retried(exception) is False
    assert qwen_llm.need_to_be_retried(exception) is True

    exception = create_fake_exception(
        "openai", "SomeOtherError", "Some other error", "fake code"
    )
    assert openai_llm.need_to_be_retried(exception) is False

    exception = KeyboardInterrupt()  # KeyboardInterrupt should not be retried
    assert openai_llm.need_to_be_retried(exception) is False


@patch("openai.resources.chat.completions.Completions.create")
@patch.object(
    LLMConfigRegistry,
    "from_file",
    return_value=LLMConfigRegistry.register_all(
        {
            "qwen": {
                "model": "qwen-3",
                "tokenizer": "qwen-3",
                "context_limit": 4,
                "api_key": "test-api-key",
                "endpoint": "https://test-endpoint",
                "api_version": "v1",
                "tags": ["openai"],  # Changed from vllm to openai
            }
        }
    ),
)
@patch.object(OpenAILLM, "tokenize", return_value=["test", "token"])
def test_llm_with_reasoning_content(
    mock_tokenize, mock_llm_config, mock_openai, logger_mock
):
    """Test that reasoning content is properly combined with regular content"""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.tool_calls = [MagicMock()]
    mock_response.choices[0].message.content = "Regular response"
    mock_response.choices[0].message.reasoning_content = (
        "Let me think about this step by step..."
    )
    mock_response.usage.prompt_tokens = 5
    mock_response.usage.completion_tokens = 10

    tmp_dict = {"arguments": '{"arg1": "test"}', "name": "test_tool"}
    tmp_dataclass = make_dataclass("tmp", ((k, type(v)) for k, v in tmp_dict.items()))(
        **tmp_dict
    )
    tmp_dict = dict(id="test_id", function=tmp_dataclass, type="function")
    mock_response.choices[0].message.tool_calls[0] = make_dataclass(
        "tmp", ((k, type(v)) for k, v in tmp_dict.items())
    )(**tmp_dict)
    mock_openai.return_value = mock_response

    llm = LLM.instantiate(name="qwen", logger=logger_mock)
    messages = [{"role": "user", "content": "Test with reasoning"}]
    llm_response = llm(messages, tools)

    # The response should be the regular content, reasoning should be separate
    assert llm_response.response == "Regular response"
    assert llm_response.reasoning_response == "Let me think about this step by step..."
    assert llm_response.prompt == messages
    assert llm_response.tool == ToolCall(
        id="test_id", name="test_tool", arguments={"arg1": "test"}
    )


@patch("openai.resources.chat.completions.Completions.create")
@patch.object(
    LLMConfigRegistry,
    "from_file",
    return_value=LLMConfigRegistry.register_all(
        {
            "qwen": {
                "model": "qwen-3",
                "tokenizer": "qwen-3",
                "context_limit": 4,
                "api_key": "test-api-key",
                "endpoint": "https://test-endpoint",
                "api_version": "v1",
                "tags": ["openai"],  # Changed from vllm to openai
            }
        }
    ),
)
@patch.object(OpenAILLM, "tokenize", return_value=["test", "token"])
def test_llm_with_only_reasoning_content(
    mock_tokenize, mock_llm_config, mock_openai, logger_mock
):
    """Test that reasoning content works when regular content is empty"""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.tool_calls = [MagicMock()]
    mock_response.choices[0].message.content = ""
    mock_response.choices[0].message.reasoning_content = "Reasoning only response"
    mock_response.usage.prompt_tokens = 3
    mock_response.usage.completion_tokens = 7

    tmp_dict = {"arguments": '{"arg1": "test"}', "name": "test_tool"}
    tmp_dataclass = make_dataclass("tmp", ((k, type(v)) for k, v in tmp_dict.items()))(
        **tmp_dict
    )
    tmp_dict = dict(id="test_id", function=tmp_dataclass, type="function")
    mock_response.choices[0].message.tool_calls[0] = make_dataclass(
        "tmp", ((k, type(v)) for k, v in tmp_dict.items())
    )(**tmp_dict)
    mock_openai.return_value = mock_response

    llm = LLM.instantiate(name="qwen", logger=logger_mock)
    messages = [{"role": "user", "content": "Test reasoning only"}]
    llm_response = llm(messages, tools)

    # The response should be empty content, reasoning should be in reasoning_response
    assert llm_response.response == ""
    assert llm_response.reasoning_response == "Reasoning only response"


@patch("openai.resources.chat.completions.Completions.create")
@patch.object(
    LLMConfigRegistry,
    "from_file",
    return_value=LLMConfigRegistry.register_all(
        {
            "openai": {
                "model": "gpt-4",
                "tokenizer": "gpt-4",
                "context_limit": 4,
                "api_key": "test-api-key",
                "endpoint": "https://test-endpoint",
                "api_version": "v1",
                "tags": ["openai"],
            }
        }
    ),
)
def test_llm_without_reasoning_content_attribute(
    mock_llm_config, mock_openai, logger_mock
):
    """Test that models without reasoning_content attribute work normally"""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.tool_calls = [MagicMock()]
    mock_response.choices[0].message.content = "Regular response only"
    # Don't set reasoning_content attribute to simulate models that don't have it
    mock_response.usage.prompt_tokens = 2
    mock_response.usage.completion_tokens = 4

    tmp_dict = {"arguments": '{"arg1": "test"}', "name": "test_tool"}
    tmp_dataclass = make_dataclass("tmp", ((k, type(v)) for k, v in tmp_dict.items()))(
        **tmp_dict
    )
    tmp_dict = dict(id="test_id", function=tmp_dataclass, type="function")
    mock_response.choices[0].message.tool_calls[0] = make_dataclass(
        "tmp", ((k, type(v)) for k, v in tmp_dict.items())
    )(**tmp_dict)
    mock_openai.return_value = mock_response

    llm = LLM.instantiate(name="openai", logger=logger_mock)
    messages = [{"role": "user", "content": "Test without reasoning"}]
    llm_response = llm(messages, tools)

    # The response should be just the regular content
    assert llm_response.response == "Regular response only"


@patch.object(
    LLMConfigRegistry,
    "from_file",
    return_value=LLMConfigRegistry.register_all(
        {
            "qwen": {
                "model": "Qwen/Qwen3-0.6B",
                "tokenizer": "Qwen/Qwen3-0.6B",
                "context_limit": 4,
                "api_key": "test-api-key",
                "endpoint": "https://test-endpoint",
                "api_version": "v1",
                "tags": ["openai"],  # Using openai tag to force OpenAILLM class
            }
        }
    ),
)
def test_openai_llm_raises_error_for_non_gpt_tokenizer(mock_llm_config, logger_mock):
    """Test that OpenAILLM raises ValueError when tokenizer is not a GPT model"""
    import pytest

    llm = LLM.instantiate(name="qwen", logger=logger_mock)
    messages = [{"role": "user", "content": "test"}]

    # Should raise ValueError when trying to tokenize with a non-GPT tokenizer
    with pytest.raises(ValueError) as exc_info:
        llm.tokenize(messages)

    assert "Tokenizer `Qwen/Qwen3-0.6B` not found" in str(exc_info.value)
    assert "set tag `vllm`" in str(exc_info.value)


class TestTokenizationSafety:
    """Tests for tiktoken stack overflow protection."""

    @patch.object(
        LLMConfigRegistry,
        "from_file",
        return_value=LLMConfigRegistry.register_all(
            {
                "gpt4": {
                    "model": "gpt-4",
                    "tokenizer": "gpt-4",
                    "context_limit": 128,
                    "api_key": "test-api-key",
                    "endpoint": "https://test-endpoint",
                    "tags": ["openai"],
                }
            }
        ),
    )
    def test_large_content_uses_character_estimate(self, mock_config, logger_mock):
        """Test that content exceeding MAX_TOKENIZE_CHARS uses character-based estimate."""
        llm = LLM.instantiate(name="gpt4", logger=logger_mock)

        # Create content larger than MAX_TOKENIZE_CHARS
        large_content = "x" * (OpenAILLM.MAX_TOKENIZE_CHARS + 1000)
        messages = [{"role": "user", "content": large_content}]

        result = llm.tokenize(messages)

        # Should return estimated tokens (content_length // 4)
        expected_estimate = len(large_content) // 4
        assert len(result) == 1
        assert len(result[0]) == expected_estimate
        assert all(t == "_" for t in result[0])  # Placeholder tokens

    @patch.object(
        LLMConfigRegistry,
        "from_file",
        return_value=LLMConfigRegistry.register_all(
            {
                "gpt4": {
                    "model": "gpt-4",
                    "tokenizer": "gpt-4",
                    "context_limit": 128,
                    "api_key": "test-api-key",
                    "endpoint": "https://test-endpoint",
                    "tags": ["openai"],
                }
            }
        ),
    )
    def test_tokenization_error_fallback(self, mock_config, logger_mock):
        """Test that tokenization errors fall back to character estimate."""
        llm = LLM.instantiate(name="gpt4", logger=logger_mock)

        # Force initialization of _tk_func
        llm.tokenize([{"role": "user", "content": "init"}])

        # Replace _tk_func with one that raises an error
        llm._tk_func = MagicMock(side_effect=RuntimeError("Simulated Rust panic"))

        content = "Test content for fallback"
        messages = [{"role": "user", "content": content}]

        result = llm.tokenize(messages)

        # Should fall back to character estimate
        expected_estimate = len(content) // 4
        assert len(result) == 1
        assert len(result[0]) == expected_estimate
        assert all(t == "_" for t in result[0])

    @patch.object(
        LLMConfigRegistry,
        "from_file",
        return_value=LLMConfigRegistry.register_all(
            {
                "gpt4": {
                    "model": "gpt-4",
                    "tokenizer": "gpt-4",
                    "context_limit": 128,
                    "api_key": "test-api-key",
                    "endpoint": "https://test-endpoint",
                    "tags": ["openai"],
                }
            }
        ),
    )
    def test_normal_tokenization_still_works(self, mock_config, logger_mock):
        """Test that normal-sized content is still tokenized properly."""
        llm = LLM.instantiate(name="gpt4", logger=logger_mock)

        messages = [{"role": "user", "content": "Hello, world!"}]
        result = llm.tokenize(messages)

        # Should return actual tokens (not placeholder "_")
        assert len(result) == 1
        assert len(result[0]) > 0
        assert not all(t == "_" for t in result[0])  # Real tokens, not placeholders

    @patch.object(
        LLMConfigRegistry,
        "from_file",
        return_value=LLMConfigRegistry.register_all(
            {
                "gpt4": {
                    "model": "gpt-4",
                    "tokenizer": "gpt-4",
                    "context_limit": 128,
                    "api_key": "test-api-key",
                    "endpoint": "https://test-endpoint",
                    "tags": ["openai"],
                }
            }
        ),
    )
    def test_special_tokens_in_content(self, mock_config, logger_mock):
        """Test that special tokens like <|endoftext|> are handled correctly."""
        llm = LLM.instantiate(name="gpt4", logger=logger_mock)

        # Content containing special tokens that would normally cause tiktoken to raise
        content_with_special = "Hello <|endoftext|> world"
        messages = [{"role": "user", "content": content_with_special}]

        # Should not raise an exception
        result = llm.tokenize(messages)

        # Should return tokens without error
        assert len(result) == 1
        assert len(result[0]) > 0
