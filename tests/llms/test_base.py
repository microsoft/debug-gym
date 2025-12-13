from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from debug_gym.gym.entities import Observation
from debug_gym.gym.tools.tool import EnvironmentTool
from debug_gym.llms import (
    AnthropicLLM,
    AzureOpenAILLM,
    HuggingFaceLLM,
    Human,
    OpenAILLM,
)
from debug_gym.llms.base import (
    LLM,
    ContextLengthExceededError,
    LLMConfig,
    LLMConfigRegistry,
    LLMResponse,
    TokenUsage,
    retry_on_exception,
)


@patch.object(
    LLMConfigRegistry,
    "from_file",
    return_value=LLMConfigRegistry.register_all(
        {
            "gpt-4o-mini-azure": {
                "model": "gpt-4o-mini_2024-07-18",
                "tokenizer": "gpt-4o-mini",
                "context_limit": 4,
                "api_key": "test-api-key",
                "endpoint": "https://test-endpoint",
                "api_version": "v1",
                "tags": ["azure openai"],
            },
            "gpt-4o-mini": {
                "model": "gpt-4o-mini_2024-07-18",
                "tokenizer": "gpt-4o-mini",
                "context_limit": 4,
                "api_key": "test-api-key",
                "endpoint": "https://test-endpoint",
                "api_version": "v1",
                "tags": ["openai"],
            },
            "claude-3.7": {
                "model": "claude-3-7-sonnet-20250219",
                "tokenizer": "claude-3-7-sonnet-20250219",
                "context_limit": 4,
                "api_key": "test-api-key",
                "tags": ["anthropic", "claude", "claude-3.7"],
            },
            "qwen-3": {
                "model": "Qwen/Qwen3-0.6B",
                "tokenizer": "Qwen/Qwen3-0.6B",
                "context_limit": 4,
                "api_key": "test-api-key",
                "endpoint": "https://test-endpoint",
                "tags": ["vllm"],
                "tokenizer_kwargs": {"trust_remote_code": True},
            },
        }
    ),
)
def test_instantiate_llm(mock_open, logger_mock):
    # None/empty name returns None
    llm = LLM.instantiate(name=None, logger=logger_mock)
    assert llm is None

    llm = LLM.instantiate(name="", logger=logger_mock)
    assert llm is None

    # tags are used to filter models
    llm = LLM.instantiate(name="gpt-4o-mini", logger=logger_mock)
    assert isinstance(llm, OpenAILLM)

    llm = LLM.instantiate(name="gpt-4o-mini-azure", logger=logger_mock)
    assert isinstance(llm, AzureOpenAILLM)

    llm = LLM.instantiate(name="claude-3.7", logger=logger_mock)
    assert isinstance(llm, AnthropicLLM)

    llm = LLM.instantiate(name="qwen-3", logger=logger_mock)
    assert isinstance(llm, HuggingFaceLLM)

    llm = LLM.instantiate(name="human", logger=logger_mock)
    assert isinstance(llm, Human)

    with pytest.raises(ValueError, match="Model unknown not found in llm config .+"):
        LLM.instantiate(name="unknown", logger=logger_mock)

    # Test with explicit generation kwargs
    llm = LLM.instantiate(
        name="gpt-4o-mini",
        logger=logger_mock,
        temperature=0.5,
        max_tokens=1000,
    )
    assert isinstance(llm, OpenAILLM)
    assert llm.runtime_generate_kwargs == {"temperature": 0.5, "max_tokens": 1000}

    # Test with **kwargs unpacking (like config)
    llm_config = {"name": "gpt-4o-mini", "temperature": 0.7}
    llm = LLM.instantiate(**llm_config, logger=logger_mock)
    assert isinstance(llm, OpenAILLM)
    assert llm.runtime_generate_kwargs == {"temperature": 0.7}


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


@pytest.fixture
def llm_cfg_mock(tmp_path, monkeypatch):
    config_file = tmp_path / "llm.yaml"
    config_file.write_text(
        yaml.dump(
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
    return config_file


def test_load_llm_config(llm_cfg_mock):
    config = LLMConfigRegistry.from_file(config_file_path=str(llm_cfg_mock))
    assert "test_model" in config


def test_load_llm_config_from_env_var(llm_cfg_mock, monkeypatch):
    monkeypatch.setenv("LLM_CONFIG_FILE_PATH", str(llm_cfg_mock))
    config = LLMConfigRegistry.from_file()
    assert "test_model" in config


def test_load_llm_config_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        LLMConfigRegistry.from_file(str(tmp_path / "llm.yaml"))


@pytest.fixture
def completion_mock():
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "some completion mock."
    return AsyncMock(return_value=mock_response)


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
    assert llm_response.token_usage is None


def test_retry_on_exception_success_after_retry():
    mock_func = MagicMock(side_effect=[ValueError(), OSError(), "success"])
    mock_is_rate_limit_error = MagicMock(return_value=True)

    result = retry_on_exception(mock_func, mock_is_rate_limit_error)("test_arg")

    assert result == "success"
    assert mock_func.call_count == 3
    mock_func.assert_called_with("test_arg")
    assert mock_is_rate_limit_error.call_count == 2


def test_retry_on_exception_raises_error():
    mock_func = MagicMock(side_effect=[ValueError(), OSError(), "success"])
    mock_is_rate_limit_error = lambda e: isinstance(e, ValueError)

    with pytest.raises(OSError):
        retry_on_exception(mock_func, mock_is_rate_limit_error)("test_arg")

    assert mock_func.call_count == 2
    mock_func.assert_called_with("test_arg")


def test_retry_on_exception_skip_keyboard_interrupt():
    mock_func = MagicMock(side_effect=KeyboardInterrupt())
    mock_is_rate_limit_error = MagicMock()

    # Do not retry on KeyboardInterrupt and let it propagate
    with pytest.raises(KeyboardInterrupt):
        retry_on_exception(mock_func, mock_is_rate_limit_error)("test_arg")

    mock_func.assert_called_once_with("test_arg")
    # The error checker should never be called for KeyboardInterrupt
    mock_is_rate_limit_error.assert_not_called()


@pytest.fixture
def basic_config():
    return LLMConfig(
        model="llm-mock",
        context_limit=4,
        api_key="test-api-key",
        endpoint="https://test-endpoint",
        tokenizer="test-tokenizer",
        reasoning_end_token="<END>",
        system_prompt_support=True,
        ignore_kwargs=["temperature", "top_p"],
        tags=["test-tag-1", "test-tag-2"],
        api_version="v1",
        scope="test-scope",
    )


def test_llm_config_initialization():
    config = LLMConfig(model="llm-mock", context_limit=4)
    assert config.model == "llm-mock"
    assert config.context_limit == 4
    assert config.tokenizer == "llm-mock"  # Default to model when tokenizer is None
    assert config.ignore_kwargs == []  # Default empty list
    assert config.tags == []  # Default empty list
    assert config.tokenizer_kwargs == {}


def test_llm_config_optional_fields(basic_config):
    assert basic_config.api_key == "test-api-key"
    assert basic_config.endpoint == "https://test-endpoint"
    assert basic_config.tokenizer == "test-tokenizer"
    assert basic_config.reasoning_end_token == "<END>"
    assert basic_config.system_prompt_support is True
    assert basic_config.ignore_kwargs == ["temperature", "top_p"]
    assert basic_config.tags == ["test-tag-1", "test-tag-2"]
    assert basic_config.api_version == "v1"
    assert basic_config.scope == "test-scope"


def test_llm_config_registry_initialization():
    registry = LLMConfigRegistry()
    assert registry.configs == {}

    registry = LLMConfigRegistry(
        configs={"model1": LLMConfig(model="model1", context_limit=4)}
    )
    assert "model1" in registry.configs
    assert registry.configs["model1"].model == "model1"


def test_llm_config_registry_get():
    registry = LLMConfigRegistry(
        configs={"model1": LLMConfig(model="model1", context_limit=4)}
    )
    config = registry.get("model1")
    assert config.model == "model1"

    with pytest.raises(
        ValueError, match="Model unknown not found in llm config registry"
    ):
        registry.get("unknown")


def test_llm_config_registry_register():
    registry = LLMConfigRegistry()
    registry.register("model1", {"model": "model1", "context_limit": 4})
    assert "model1" in registry.configs
    assert registry.configs["model1"].model == "model1"


def test_llm_config_registry_register_all():
    configs = {
        "model1": {
            "model": "model1",
            "context_limit": 4,
        },
        "model2": {
            "model": "model2",
            "context_limit": 8,
            "api_key": "test-key",
        },
    }
    registry = LLMConfigRegistry.register_all(configs)
    assert "model1" in registry.configs
    assert "model2" in registry.configs
    assert registry.configs["model1"].model == "model1"
    assert registry.configs["model2"].api_key == "test-key"


def test_llm_config_registry_contains():
    registry = LLMConfigRegistry(
        configs={
            "model1": LLMConfig(model="model1", context_limit=4),
        }
    )
    assert "model1" in registry
    assert "unknown" not in registry


def test_llm_config_registry_getitem():
    registry = LLMConfigRegistry(
        configs={
            "model1": LLMConfig(model="model1", context_limit=4),
        }
    )
    config = registry["model1"]
    assert config.model == "model1"

    with pytest.raises(ValueError):
        _ = registry["unknown"]


def test_token_usage_initialization():
    token_usage = TokenUsage(prompt=10, response=20)
    assert token_usage.prompt == 10
    assert token_usage.response == 20


@patch.object(
    LLMConfigRegistry,
    "from_file",
    return_value=LLMConfigRegistry.register_all(
        {
            "llm-mock": {
                "model": "llm-mock",
                "context_limit": 4,
                "tokenizer": "test-tokenizer",
                "tags": [],
                "generate_kwargs": {
                    "temperature": 0.7,
                    "max_tokens": 100,
                    "thinking": {
                        "type": "enabled",
                        "budget_tokens": 10,
                    },
                },
            }
        }
    ),
)
def test_llm_call_with_generate_kwargs(logger_mock, llm_class_mock):
    messages = [{"role": "user", "content": "Hello"}]
    llm_mock = llm_class_mock("llm-mock", logger=logger_mock)
    llm_response = llm_mock(messages, tools)

    # Check that generate_kwargs were passed to generate
    assert llm_mock.called_kwargs["temperature"] == 0.7
    assert llm_mock.called_kwargs["max_tokens"] == 100
    assert llm_mock.called_kwargs["thinking"]["type"] == "enabled"
    assert llm_mock.called_kwargs["thinking"]["budget_tokens"] == 10
    assert llm_response.response == "Test response"


@patch.object(
    LLMConfigRegistry,
    "from_file",
    return_value=LLMConfigRegistry.register_all(
        {
            "llm-mock": {
                "model": "llm-mock",
                "context_limit": 4,
                "tokenizer": "test-tokenizer",
                "generate_kwargs": {"temperature": 0.7},
                "tags": [],
            }
        }
    ),
)
def test_llm_call_override_generate_kwargs(logger_mock, llm_class_mock):
    messages = [{"role": "user", "content": "Hello"}]
    llm_mock = llm_class_mock("llm-mock", logger=logger_mock)
    # Override the temperature from config
    llm_mock(messages, tools, temperature=0.2)
    # Check that the override worked: 0.2 from kwargs, not 0.7 from config
    assert llm_mock.called_kwargs["temperature"] == 0.2


@patch.object(
    LLMConfigRegistry,
    "from_file",
    return_value=LLMConfigRegistry.register_all(
        {
            "llm-mock": {
                "model": "llm-mock",
                "context_limit": 4,
                "tokenizer": "test-tokenizer",
                "ignore_kwargs": ["temperature"],
                "tags": [],
            }
        }
    ),
)
def test_llm_call_ignore_kwargs(logger_mock, llm_class_mock):
    messages = [{"role": "user", "content": "Hello"}]
    llm_mock = llm_class_mock("llm-mock", logger=logger_mock)
    llm_mock(messages, tools, temperature=0.7, max_tokens=10)
    assert "temperature" not in llm_mock.called_kwargs
    assert llm_mock.called_kwargs["max_tokens"] == 10


@patch.object(
    LLMConfigRegistry,
    "from_file",
    return_value=LLMConfigRegistry.register_all(
        {
            "llm-mock": {
                "model": "llm-mock",
                "context_limit": 4,
                "tokenizer": "test-tokenizer",
                "system_prompt_support": False,
                "tags": [],
            }
        }
    ),
)
def test_llm_call_system_prompt_not_supported(
    mock_llm_config, logger_mock, llm_class_mock
):
    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"},
    ]
    llm_mock = llm_class_mock("llm-mock", logger=logger_mock)
    llm_mock(messages, tools)
    assert llm_mock.called_messages == [
        {"role": "user", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"},
    ]


def test_llm_init_with_config(logger_mock, llm_class_mock):
    llm_config = LLMConfig(
        model="llm-mock",
        context_limit=4,
        api_key="test-api-key",
        endpoint="https://test-endpoint",
        tokenizer="test-tokenizer",
        tags=["test-tag"],
    )
    llm = llm_class_mock(
        model_name="llm-mock", logger=logger_mock, llm_config=llm_config
    )
    assert llm.model_name == "llm-mock"
    assert llm.config == llm_config
    assert llm.tokenizer_name == "test-tokenizer"
    assert llm.context_length == 4000


def test_llm_init_with_runtime_generate_kwargs(logger_mock, llm_class_mock):
    """Test that runtime_generate_kwargs are properly set during initialization."""
    llm_config = LLMConfig(
        model="llm-mock",
        context_limit=4,
        api_key="test-api-key",
        endpoint="https://test-endpoint",
        tokenizer="test-tokenizer",
        tags=["test-tag"],
    )
    runtime_kwargs = {"temperature": 0.5, "max_tokens": 1000}
    llm = llm_class_mock(
        model_name="llm-mock",
        logger=logger_mock,
        llm_config=llm_config,
        runtime_generate_kwargs=runtime_kwargs,
    )
    assert llm.model_name == "llm-mock"
    assert llm.config == llm_config
    assert llm.tokenizer_name == "test-tokenizer"
    assert llm.context_length == 4000
    assert llm.runtime_generate_kwargs == runtime_kwargs


@patch.object(
    LLMConfigRegistry,
    "from_file",
    return_value=LLMConfigRegistry.register_all(
        {
            "llm-mock": {
                "model": "llm-mock",
                "context_limit": 4,
                "tokenizer": "test-tokenizer",
                "tags": [],
            }
        }
    ),
)
def test_context_length_exceeded_prevents_infinite_recursion(
    llm_mock, logger_mock, llm_class_mock
):
    """Test that ContextLengthExceededError handling prevents infinite recursion."""

    class ContextErrorLLM(llm_class_mock):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.generate_call_count = 0

        def generate(self, messages, tools, **kwargs):
            self.generate_call_count += 1
            # Always raise ContextLengthExceededError to test the fix
            raise ContextLengthExceededError("Context length exceeded")

    llm = ContextErrorLLM("llm-mock", logger=logger_mock)
    messages = [{"role": "user", "content": "Long message"}]

    # Mock trim_prompt_messages to return the same messages (no reduction)
    with patch("debug_gym.llms.utils.trim_prompt_messages") as mock_trim:
        mock_trim.return_value = messages

        # Should raise ContextLengthExceededError, not RecursionError
        with pytest.raises(
            ContextLengthExceededError, match="Unable to reduce prompt size"
        ):
            llm(messages, tools)

    # Should only try once due to no improvement in trimming
    assert llm.generate_call_count == 1

    # Should log the "Prompt is too long" message
    prompt_too_long_calls = [
        msg for msg in logger_mock._log_history if "Prompt is too long" in msg
    ]
    assert len(prompt_too_long_calls) == 1


@patch.object(
    LLMConfigRegistry,
    "from_file",
    return_value=LLMConfigRegistry.register_all(
        {
            "llm-mock": {
                "model": "llm-mock",
                "context_limit": 4,
                "tokenizer": "test-tokenizer",
                "tags": [],
            }
        }
    ),
)
def test_context_length_exceeded_with_successful_truncation(
    mock_llm_config, logger_mock, llm_class_mock
):
    """Test that successful truncation logs both messages correctly."""

    class ContextErrorThenSuccessLLM(llm_class_mock):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.generate_call_count = 0

        def generate(self, messages, tools, **kwargs):
            self.generate_call_count += 1
            if self.generate_call_count == 1:
                raise ContextLengthExceededError("Context length exceeded")

            # Success on second call
            return super().generate(messages, tools, **kwargs)

    llm = ContextErrorThenSuccessLLM("llm-mock", logger=logger_mock)
    messages = [{"role": "user", "content": "Long message"}]

    # Mock trim_prompt_messages to return shorter messages
    with patch("debug_gym.llms.base.trim_prompt_messages") as mock_trim:
        shorter_messages = [{"role": "user", "content": "Short"}]
        mock_trim.return_value = shorter_messages

        # Should succeed
        response = llm(messages, tools)
        assert response.response == "Test response"

    # Should try twice: fail once, then succeed
    assert llm.generate_call_count == 2

    # Should log both "Prompt is too long" and "Prompt truncated" messages
    prompt_too_long_calls = [
        msg for msg in logger_mock._log_history if "Prompt is too long" in msg
    ]
    prompt_truncated_calls = [
        msg for msg in logger_mock._log_history if "Prompt truncated" in msg
    ]

    assert (
        len(prompt_too_long_calls) == 1
    ), f"Expected 1 'Prompt is too long' call, got {len(prompt_too_long_calls)}"
    assert (
        len(prompt_truncated_calls) == 1
    ), f"Expected 1 'Prompt truncated' call, got {len(prompt_truncated_calls)}"
