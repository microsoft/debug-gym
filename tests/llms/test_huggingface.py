import json
from unittest.mock import MagicMock, patch

from debug_gym.gym.tools.tool import ToolCall
from debug_gym.llms import HuggingFaceLLM
from debug_gym.llms.base import LLMConfigRegistry, LLMResponse


@patch.object(
    LLMConfigRegistry,
    "from_file",
    return_value=LLMConfigRegistry.register_all(
        {
            "qwen-3": {
                "model": "qwen-3",
                "tokenizer": "Qwen/Qwen3",
                "context_limit": 4,
                "api_key": "test-api-key",
                "endpoint": "https://test-endpoint",
                "tags": ["vllm"],
                "tokenizer_kwargs": {"trust_remote_code": True},
            }
        }
    ),
)
@patch("debug_gym.llms.huggingface.AutoTokenizer.from_pretrained")
def test_huggingface_tokenizer_usage(mock_auto_tokenizer, mock_llm_config, logger_mock):
    tokenizer_mock = MagicMock()
    tokenizer_mock.encode.return_value = [10, 20, 30]
    tokenizer_mock.convert_ids_to_tokens.return_value = ["<a>", "<b>", "<c>"]
    tokenizer_mock.pad_token = None
    tokenizer_mock.eos_token = "</s>"
    mock_auto_tokenizer.return_value = tokenizer_mock

    llm = HuggingFaceLLM(model_name="qwen-3", logger=logger_mock)

    tokens = llm.tokenize("hello world")
    assert tokens == ["<a>", "<b>", "<c>"]
    assert llm.count_tokens("hello world") == 3

    mock_auto_tokenizer.assert_called_once_with("Qwen/Qwen3", trust_remote_code=True)
    tokenizer_mock.encode.assert_called_with("hello world", add_special_tokens=False)

    # Ensure pad token fallback was applied
    assert tokenizer_mock.pad_token == "</s>"


@patch.object(
    LLMConfigRegistry,
    "from_file",
    return_value=LLMConfigRegistry.register_all(
        {
            "qwen-3": {
                "model": "qwen-3",
                "tokenizer": "Qwen/Qwen3",
                "context_limit": 4,
                "api_key": "test-api-key",
                "endpoint": "https://test-endpoint",
                "tags": ["vllm"],
            }
        }
    ),
)
@patch("debug_gym.llms.huggingface.AutoTokenizer.from_pretrained")
def test_huggingface_normalizes_messages_for_template(
    mock_auto_tokenizer, mock_llm_config, logger_mock
):
    tokenizer_mock = MagicMock()
    tokenizer_mock.pad_token = None
    tokenizer_mock.eos_token = "</s>"
    mock_auto_tokenizer.return_value = tokenizer_mock

    llm = HuggingFaceLLM(model_name="qwen-3", logger=logger_mock)

    raw_messages = [
        {"role": "tool", "content": "partial output"},
        {
            "role": "developer",
            "content": [{"text": "line1"}, {"text": "line2"}],
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"type": "function", "name": "noop", "arguments": {}}],
        },
        {"role": "user", "content": None},
    ]

    normalized = llm._normalize_messages_for_template(raw_messages)

    assert normalized == [
        {"role": "user", "content": "partial output"},
        {"role": "user", "content": "line1\nline2"},
        {
            "role": "assistant",
            "content": json.dumps(
                [{"type": "function", "name": "noop", "arguments": {}}]
            ),
        },
        {"role": "user", "content": ""},
    ]


@patch.object(
    LLMConfigRegistry,
    "from_file",
    return_value=LLMConfigRegistry.register_all(
        {
            "qwen-3": {
                "model": "qwen-3",
                "tokenizer": "Qwen/Qwen3",
                "context_limit": 4,
                "api_key": "test-api-key",
                "endpoint": "https://test-endpoint",
                "tags": ["vllm"],
            }
        }
    ),
)
@patch("debug_gym.llms.huggingface.AutoTokenizer.from_pretrained")
def test_huggingface_chat_template_token_counts(
    mock_auto_tokenizer, mock_llm_config, logger_mock
):
    tokenizer_mock = MagicMock()
    tokenizer_mock.pad_token = None
    tokenizer_mock.eos_token = "</s>"

    def fake_apply_chat_template(messages, tokenize=True, add_generation_prompt=False):
        length = len(messages)
        return list(range(length * 2))

    tokenizer_mock.apply_chat_template.side_effect = fake_apply_chat_template
    tokenizer_mock.encode.return_value = []
    mock_auto_tokenizer.return_value = tokenizer_mock

    llm = HuggingFaceLLM(model_name="qwen-3", logger=logger_mock)

    messages = [
        {"role": "system", "content": "Instructions"},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
        {"role": "tool", "content": "Result"},
    ]

    counts = llm._get_message_token_counts(messages)

    assert counts == [2, 2, 2, 2]
    assert tokenizer_mock.apply_chat_template.call_count == len(messages)
    normalized_final = tokenizer_mock.apply_chat_template.call_args_list[-1][0][0]
    assert normalized_final[-1]["role"] == "user"
    assert normalized_final[-1]["content"] == "Result"


@patch.object(
    LLMConfigRegistry,
    "from_file",
    return_value=LLMConfigRegistry.register_all(
        {
            "qwen-3": {
                "model": "qwen-3",
                "tokenizer": "Qwen/Qwen3",
                "context_limit": 4,
                "api_key": "test-api-key",
                "endpoint": "https://test-endpoint",
                "tags": ["vllm"],
            }
        }
    ),
)
@patch("debug_gym.llms.huggingface.AutoTokenizer.from_pretrained")
def test_huggingface_chat_template_zero_token_fallback(
    mock_auto_tokenizer, mock_llm_config, logger_mock
):
    tokenizer_mock = MagicMock()
    tokenizer_mock.pad_token = None
    tokenizer_mock.eos_token = "</s>"
    tokenizer_mock.apply_chat_template.return_value = []
    tokenizer_mock.encode.side_effect = [[1, 2], [3, 4, 5]]
    mock_auto_tokenizer.return_value = tokenizer_mock

    llm = HuggingFaceLLM(model_name="qwen-3", logger=logger_mock)

    messages = [
        {"role": "system", "content": "Instructions"},
        {"role": "user", "content": "Hello"},
    ]

    counts = llm._get_message_token_counts(messages)

    assert counts == [2, 3]
    assert tokenizer_mock.encode.call_count == len(messages)


@patch.object(
    LLMConfigRegistry,
    "from_file",
    return_value=LLMConfigRegistry.register_all(
        {
            "qwen-3": {
                "model": "qwen-3",
                "tokenizer": "Qwen/Qwen3",
                "context_limit": 4,
                "api_key": "test-api-key",
                "endpoint": "https://test-endpoint",
                "tags": ["vllm"],
            }
        }
    ),
)
@patch.object(HuggingFaceLLM, "generate")
@patch.object(
    LLMConfigRegistry,
    "from_file",
    return_value=LLMConfigRegistry.register_all(
        {
            "qwen-3": {
                "model": "qwen-3",
                "tokenizer": "Qwen/Qwen3",
                "context_limit": 4,
                "api_key": "test-api-key",
                "endpoint": "https://test-endpoint",
                "tags": ["vllm"],
                "tokenizer_kwargs": {"trust_remote_code": True},
            }
        }
    ),
)
@patch("debug_gym.llms.huggingface.AutoTokenizer.from_pretrained")
def test_huggingface_chat_template_usage(
    mock_auto_tokenizer, mock_llm_config, mock_generate, logger_mock
):
    tokenizer_mock = MagicMock()
    tokenizer_mock.pad_token = None
    tokenizer_mock.eos_token = "</s>"
    tokenizer_mock.convert_ids_to_tokens.side_effect = lambda ids: [
        f"<{i}>" for i in ids
    ]
    tokenizer_mock.apply_chat_template.side_effect = [
        {"input_ids": [[1, 2]]},
        {"input_ids": [[1, 2, 3, 4]]},
    ]
    tokenizer_mock.encode.return_value = [99]
    mock_auto_tokenizer.return_value = tokenizer_mock

    mock_generate.return_value = LLMResponse(
        prompt=[],
        response="ok",
        tool=ToolCall(id="t1", name="noop", arguments={}),
        prompt_token_count=5,
        response_token_count=2,
    )

    llm = HuggingFaceLLM(model_name="qwen-3", logger=logger_mock)

    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello"},
    ]

    response = llm(messages, tools=[])

    assert response.response == "ok"
    assert tokenizer_mock.apply_chat_template.call_count == 2
    mock_generate.assert_called_once()
