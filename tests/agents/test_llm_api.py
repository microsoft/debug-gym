import json
from dataclasses import make_dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
import yaml

from debug_gym.agents.llm_api import (
    LLM,
    AnthropicLLM,
    AzureOpenAILLM,
    Human,
    LLMConfig,
    LLMConfigRegistry,
    LLMResponse,
    OpenAILLM,
    TokenUsage,
    retry_on_rate_limit,
)
from debug_gym.gym.entities import Observation
from debug_gym.gym.tools.tool import EnvironmentTool, ToolCall
from debug_gym.gym.tools.toolbox import Toolbox


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

    llm = OpenAILLM(model_name="openai", logger=logger_mock)
    messages = [{"role": "user", "content": "Hello World"}]
    llm_response = llm(messages, tools)
    assert llm_response.prompt == messages
    assert llm_response.tool == ToolCall(id="1", name="tool 1", arguments={"arg 1": 0})
    assert llm_response.token_usage.prompt == 2
    assert llm_response.token_usage.response == 4


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


@patch(
    "builtins.input",
    lambda *args, **kwargs: json.dumps(
        {"id": "pdb-637469", "name": "pdb", "arguments": {"command": "b 10"}}
    ),
)
def test_human(build_env_info):
    # always generate the same random toolcall id: "pdb-637469"
    np.random.seed(42)
    human = Human()
    messages = [{"role": "user", "content": "Hello"}]
    env_info = build_env_info(
        action=ToolCall(id="pdb-637469", name="pdb", arguments="b 10"),
        tools=[Toolbox.get_tool("pdb"), Toolbox.get_tool("view")],
    )
    llm_response = human(messages, env_info.tools)
    # human only uses the messages content
    assert llm_response.prompt == [{"role": "user", "content": "Hello"}]
    assert (
        llm_response.response
        == '{"id": "pdb-637469", "name": "pdb", "arguments": {"command": "b 10"}}'
    )
    assert llm_response.token_usage.prompt == 4
    assert llm_response.token_usage.response == 8


@patch(
    "builtins.input",
    side_effect=["invalid input"] * 10,  # Return invalid input 10 times
)
def test_human_max_retries(_, build_env_info):
    human = Human(max_retries=5)  # Set max_retries to 5
    messages = [{"role": "user", "content": "Test message"}]
    env_info = build_env_info(
        tools=[Toolbox.get_tool("pdb"), Toolbox.get_tool("view")],
    )

    # Should raise ValueError when max retries is reached
    with pytest.raises(
        ValueError, match="Maximum retries \\(5\\) reached without valid input."
    ):
        human(messages, env_info.tools)


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
        }
    ),
)
def test_instantiate_llm(mock_open, logger_mock):
    # tags are used to filter models
    llm = LLM.instantiate("gpt-4o-mini", logger=logger_mock)
    assert isinstance(llm, OpenAILLM)

    llm = LLM.instantiate("gpt-4o-mini-azure", logger=logger_mock)
    assert isinstance(llm, AzureOpenAILLM)

    llm = LLM.instantiate("claude-3.7", logger=logger_mock)
    assert isinstance(llm, AnthropicLLM)

    llm = LLM.instantiate("human", logger=logger_mock)
    assert isinstance(llm, Human)

    with pytest.raises(ValueError, match="Model unknown not found in llm config .+"):
        LLM.instantiate("unknown", logger=logger_mock)


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


def create_fake_exception(module: str, classname: str, message: str):
    exc_type = type(classname, (Exception,), {})
    exc = exc_type(message)
    exc.message = message
    exc.__class__.__module__ = module
    return exc


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
            }
        }
    ),
)
def test_is_rate_limit_error(llm_config_registry_mock, logger_mock):
    openai_llm = OpenAILLM("openai", logger=logger_mock)

    exception = create_fake_exception("openai", "RateLimitError", "Rate limit exceeded")
    assert openai_llm.is_rate_limit_error(exception) is True

    exception = create_fake_exception(
        "openai", "APIStatusError", "Error occurred: 'status': 429 rate limit"
    )
    assert openai_llm.is_rate_limit_error(exception) is True

    exception = create_fake_exception(
        "openai", "APIStatusError", "Encountered error: 'status': 504 gateway timeout"
    )
    assert openai_llm.is_rate_limit_error(exception) is True

    exception = create_fake_exception(
        "openai",
        "APIStatusError",
        "Failure: 'status': 413 A previous prompt was too large. Please shorten input.",
    )
    assert openai_llm.is_rate_limit_error(exception) is True

    exception = create_fake_exception(
        "openai", "APIStatusError", "Error: 'status': 500 internal server error"
    )
    assert openai_llm.is_rate_limit_error(exception) is False

    exception = create_fake_exception(
        "openai", "PermissionDeniedError", "Permission denied error"
    )
    assert openai_llm.is_rate_limit_error(exception) is True

    exception = create_fake_exception("openai", "SomeOtherError", "Some other error")
    assert openai_llm.is_rate_limit_error(exception) is False

    exception = KeyboardInterrupt()  # KeyboardInterrupt should not be retried
    assert openai_llm.is_rate_limit_error(exception) is False


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
def test_llm_call_with_generate_kwargs(mock_llm_config, logger_mock, llm_class_mock):
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
def test_llm_call_override_generate_kwargs(
    mock_llm_config, logger_mock, llm_class_mock
):
    messages = [{"role": "user", "content": "Hello"}]
    llm_mock = llm_class_mock("llm-mock", logger=logger_mock)
    # Override the temperature from config
    llm_response = llm_mock(messages, tools, temperature=0.2)
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
def test_llm_call_ignore_kwargs(mock_llm_config, logger_mock, llm_class_mock):
    messages = [{"role": "user", "content": "Hello"}]
    llm_mock = llm_class_mock("llm-mock", logger=logger_mock)
    llm_response = llm_mock(messages, tools, temperature=0.7, max_tokens=10)
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
    llm_response = llm_mock(messages, tools)
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


def test_llm_init_with_both_config_types(logger_mock, llm_class_mock):
    llm_config = LLMConfig(
        model="llm-mock",
        context_limit=4,
        api_key="test-api-key",
        endpoint="https://test-endpoint",
        tokenizer="test-tokenizer",
        tags=["test-tag"],
    )
    llm = llm_class_mock(
        model_name="llm-mock",
        logger=logger_mock,
        llm_config=llm_config,
        llm_config_file="llm.yaml",
    )
    assert llm.model_name == "llm-mock"
    assert llm.config == llm_config
    assert llm.tokenizer_name == "test-tokenizer"
    assert llm.context_length == 4000
    assert (
        "Both llm_config and llm_config_file are provided, using llm_config."
        in logger_mock._log_history
    )


@pytest.fixture
def example_tools():
    """
    Build a deterministic list of tools (Human.define_tools generates random IDs,
    so we craft a fixed one instead).
    """
    return [
        {
            "id": "print-001",
            "name": "print",
            "arguments": {"msg": ""},
        },
        {
            "id": "sum-002",
            "name": "sum",
            "arguments": {"a": "", "b": ""},
        },
    ]


def test_parse_tool_call_response_valid(logger_mock, example_tools):
    """A well-formed JSON command that matches one of the declared tools
    should be converted into a ToolCall instance and no error logged."""
    human = Human(logger=logger_mock)
    cmd = json.dumps({"id": "print-001", "name": "print", "arguments": {"msg": "hi"}})

    tool_call = human.parse_tool_call_response(cmd, example_tools)
    assert tool_call == ToolCall(id="print-001", name="print", arguments={"msg": "hi"})


@pytest.mark.parametrize(
    "bad_command",
    [
        "not-json",  # invalid JSON
        json.dumps(
            {"id": "print-001", "name": "print", "arguments": {"unknown": "x"}}
        ),  # wrong args
        json.dumps(
            {"id": "non-existent", "name": "print", "arguments": {}}
        ),  # id not found
        json.dumps(
            {"id": "print-001", "name": "unknown", "arguments": {}}
        ),  # name mismatch
        json.dumps(
            {"id": "print-001", "name": "unknown"}
        ),  # missing ToolCalls arguments
    ],
)
def test_parse_tool_call_response_invalid(logger_mock, example_tools, bad_command):
    """For malformed or non-matching commands the method should raise ValueError"""
    human = Human(logger=logger_mock)

    with pytest.raises(ValueError, match="Failed to parse valid tool call from input"):
        human.parse_tool_call_response(bad_command, example_tools)


def test_parse_tool_call_response_no_tools(logger_mock):
    """Should raise ValueError when no tools are provided"""
    human = Human(logger=logger_mock)

    with pytest.raises(
        ValueError, match="No tools provided. At least one tool must be available."
    ):
        human.parse_tool_call_response(
            '{"id": "test", "name": "test", "arguments": {}}', []
        )

    with pytest.raises(
        ValueError, match="No tools provided. At least one tool must be available."
    ):
        human.parse_tool_call_response(
            '{"id": "test", "name": "test", "arguments": {}}', None
        )


def test_parse_tool_call_response_none(logger_mock, example_tools):
    """Should raise ValueError when response is None"""
    human = Human(logger=logger_mock)

    with pytest.raises(ValueError, match="Tool call cannot be None"):
        human.parse_tool_call_response(None, example_tools)
