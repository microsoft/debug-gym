import json
from unittest.mock import MagicMock, patch

from debug_gym.llms import HuggingFaceLLM
from debug_gym.llms.base import LLMConfigRegistry
from debug_gym.llms.openai import OpenAILLM

MODEL_REGISTRY = {
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


@patch.object(
    LLMConfigRegistry,
    "from_file",
    return_value=LLMConfigRegistry.register_all(MODEL_REGISTRY),
)
@patch("debug_gym.llms.huggingface.AutoTokenizer.from_pretrained")
def test_tokenize_uses_hf_tokenizer_with_pad_fallback(
    mock_auto_tokenizer, mock_llm_config, logger_mock
):
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
    assert tokenizer_mock.pad_token == "</s>"


@patch.object(
    LLMConfigRegistry,
    "from_file",
    return_value=LLMConfigRegistry.register_all(MODEL_REGISTRY),
)
@patch("debug_gym.llms.huggingface.AutoTokenizer.from_pretrained")
def test_normalize_messages_for_chat_template(
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
    return_value=LLMConfigRegistry.register_all(MODEL_REGISTRY),
)
@patch("debug_gym.llms.huggingface.AutoTokenizer.from_pretrained")
def test_message_token_counts_uses_chat_template(
    mock_auto_tokenizer, mock_llm_config, logger_mock
):
    tokenizer_mock = MagicMock()
    tokenizer_mock.pad_token = None
    tokenizer_mock.eos_token = "</s>"
    tokenizer_mock.apply_chat_template.side_effect = [
        {"input_ids": [[1, 2]]},
        {"input_ids": [[1, 2, 3]]},
        {"input_ids": [[1, 2, 3, 4]]},
    ]
    mock_auto_tokenizer.return_value = tokenizer_mock

    llm = HuggingFaceLLM(model_name="qwen-3", logger=logger_mock)

    messages = [
        {"role": "system", "content": "Instructions"},
        {"role": "user", "content": "Hello"},
        {"role": "tool", "content": "Result"},
    ]

    counts = llm._get_message_token_counts(messages)

    assert counts == [2, 1, 1]
    assert tokenizer_mock.apply_chat_template.call_count == len(messages)
    final_normalized = tokenizer_mock.apply_chat_template.call_args_list[-1][0][0]
    assert final_normalized[-1]["role"] == "user"
    assert final_normalized[-1]["content"] == "Result"


@patch.object(
    LLMConfigRegistry,
    "from_file",
    return_value=LLMConfigRegistry.register_all(MODEL_REGISTRY),
)
@patch("debug_gym.llms.huggingface.AutoTokenizer.from_pretrained")
@patch.object(OpenAILLM, "_get_message_token_counts", return_value=[5, 6])
def test_message_token_counts_fallbacks_to_openai_when_template_fails(
    mock_super_counts, mock_auto_tokenizer, mock_llm_config, logger_mock
):
    tokenizer_mock = MagicMock()
    tokenizer_mock.pad_token = None
    tokenizer_mock.eos_token = "</s>"
    tokenizer_mock.apply_chat_template.side_effect = ValueError("no template")
    mock_auto_tokenizer.return_value = tokenizer_mock

    llm = HuggingFaceLLM(model_name="qwen-3", logger=logger_mock)

    messages = [
        {"role": "system", "content": "Instructions"},
        {"role": "user", "content": "Hello"},
    ]

    counts = llm._get_message_token_counts(messages)

    assert counts == [5, 6]
    mock_super_counts.assert_called_once_with(messages)
