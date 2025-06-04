from dataclasses import make_dataclass
from unittest.mock import MagicMock, patch

from debug_gym.gym.entities import Observation
from debug_gym.gym.tools.tool import EnvironmentTool, ToolCall
from debug_gym.llms import OpenAILLM
from debug_gym.llms.base import LLMConfigRegistry


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


def create_fake_exception(module: str, classname: str, message: str):
    exc_type = type(classname, (Exception,), {})
    exc = exc_type(message)
    exc.message = message
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
