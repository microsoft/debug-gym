from dataclasses import make_dataclass
from unittest.mock import MagicMock, patch

from debug_gym.gym.entities import Observation
from debug_gym.gym.envs.env import EnvInfo
from debug_gym.gym.tools.tool import EnvironmentTool, ToolCall
from debug_gym.llms import OpenAILLM
from debug_gym.llms.base import LLMConfigRegistry, LLMResponse


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
    openai_llm = OpenAILLM("openai", logger=logger_mock)
    qwen_llm = OpenAILLM("qwen", logger=logger_mock)

    exception = create_fake_exception("openai", "RateLimitError", "Rate limit exceeded")
    assert openai_llm.need_to_be_retried(exception) is True

    exception = create_fake_exception(
        "openai", "APIStatusError", "Error occurred: 'status': 429 rate limit"
    )
    assert openai_llm.need_to_be_retried(exception) is True

    exception = create_fake_exception(
        "openai", "APIStatusError", "Encountered error: 'status': 504 gateway timeout"
    )
    assert openai_llm.need_to_be_retried(exception) is True

    exception = create_fake_exception(
        "openai",
        "APIStatusError",
        "Failure: 'status': 413 A previous prompt was too large. Please shorten input.",
    )
    assert openai_llm.need_to_be_retried(exception) is True

    exception = create_fake_exception(
        "openai", "APIStatusError", "Error: 'status': 500 internal server error"
    )
    assert openai_llm.need_to_be_retried(exception) is False

    exception = create_fake_exception(
        "openai", "PermissionDeniedError", "Permission denied error"
    )
    assert openai_llm.need_to_be_retried(exception) is True

    exception = create_fake_exception(
        "openai",
        "BadRequestError",
        "Error code: 400 \n Invalid JSON: EOF while parsing a string",
    )
    assert openai_llm.need_to_be_retried(exception) is False
    assert qwen_llm.need_to_be_retried(exception) is True

    exception = create_fake_exception("openai", "SomeOtherError", "Some other error")
    assert openai_llm.need_to_be_retried(exception) is False

    exception = KeyboardInterrupt()  # KeyboardInterrupt should not be retried
    assert openai_llm.need_to_be_retried(exception) is False


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
def test_format_tool_call_history_initial_state(mock_llm_config, logger_mock):
    """Test format_tool_call_history with initial state (no action taken yet)"""
    llm = OpenAILLM(model_name="openai", logger=logger_mock)

    # Create EnvInfo for initial state
    history_info = EnvInfo(
        step_observation=Observation(source="tool1", observation="Initial observation"),
        all_observations=[],
        eval_observation=Observation(source="tool1", observation=""),
        dir_tree="",
        current_breakpoints="",
        action=None,  # No action taken yet
        instructions={},
        score=0,
        max_score=100,
        done=False,
        rewrite_counter=0,
        tools=[],
    )

    messages = llm.format_tool_call_history(history_info, [])
    assert len(messages) == 1
    # Only message should be the user's initial observation
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "Initial observation"


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
def test_format_tool_call_history_with_action(mock_llm_config, logger_mock):
    """Test format_tool_call_history with an action taken"""
    llm = OpenAILLM(model_name="openai", logger=logger_mock)

    # Create action that was taken
    action = ToolCall(
        id="call_456",
        name="edit",
        arguments={"path": "test.py", "content": "new content"},
    )

    # Create EnvInfo with action taken
    history_info = EnvInfo(
        step_observation=Observation(
            source="tool_456", observation="File edited successfully"
        ),
        all_observations=[],
        eval_observation=Observation(source="tool_456", observation=""),
        dir_tree="",
        current_breakpoints="",
        action=action,  # Action was taken
        instructions={},
        score=0,
        max_score=100,
        done=False,
        rewrite_counter=0,
        tools=[],
    )

    # Create LLMResponse with tool call
    tool_call = ToolCall(
        id="call_789", name="run", arguments={"command": "python test.py"}
    )
    llm_response = LLMResponse(
        prompt=[{"role": "user", "content": "test"}],
        response="test response",
        tool=tool_call,
    )

    messages = llm.format_tool_call_history(history_info, [llm_response])

    assert len(messages) == 2
    # First message should be the assistant's tool call
    assert messages[0]["role"] == "assistant"
    assert messages[0]["tool_calls"][0]["type"] == "function"
    assert messages[0]["tool_calls"][0]["id"] == "call_789"
    assert messages[0]["tool_calls"][0]["function"]["name"] == "run"
    assert (
        messages[0]["tool_calls"][0]["function"]["arguments"]
        == '{"command": "python test.py"}'
    )

    # Second message should be the tool result
    assert messages[1]["role"] == "tool"
    assert messages[1]["tool_call_id"] == "call_456"
    assert messages[1]["name"] == "edit"
    assert messages[1]["content"] == "File edited successfully"


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
def test_format_tool_call_history_complex_arguments(mock_llm_config, logger_mock):
    """Test format_tool_call_history with complex nested arguments"""
    llm = OpenAILLM(model_name="openai", logger=logger_mock)

    # Create action that was taken
    action = ToolCall(
        id="call_456",
        name="edit",
        arguments={"path": "test.py", "content": "new content"},
    )
    # Create EnvInfo for initial state
    history_info = EnvInfo(
        step_observation=Observation(
            source="tool_456", observation="Complex operation completed"
        ),
        all_observations=[],
        eval_observation=Observation(source="tool_456", observation=""),
        dir_tree="",
        current_breakpoints="",
        action=action,
        instructions={},
        score=0,
        max_score=100,
        done=False,
        rewrite_counter=0,
        tools=[],
    )

    # Create LLMResponse with complex tool call arguments
    complex_args = {
        "config": {
            "mode": "debug",
            "options": ["verbose", "trace"],
            "settings": {"timeout": 30, "retries": 3},
        },
        "files": ["test1.py", "test2.py"],
    }
    tool_call = ToolCall(id="call_complex", name="configure", arguments=complex_args)
    llm_response = LLMResponse(
        prompt=[{"role": "user", "content": "test"}],
        response="test response",
        tool=tool_call,
    )

    messages = llm.format_tool_call_history(history_info, [llm_response])

    assert len(messages) == 2
    # Check that complex arguments are properly JSON-serialized
    import json

    assert messages[0]["role"] == "assistant"
    assert messages[0]["tool_calls"][0]["function"]["name"] == "configure"
    parsed_args = json.loads(messages[0]["tool_calls"][0]["function"]["arguments"])
    assert parsed_args == complex_args

    assert messages[1]["role"] == "tool"
    assert messages[1]["tool_call_id"] == "call_456"
    assert messages[1]["name"] == "edit"
    assert messages[1]["content"] == "Complex operation completed"
