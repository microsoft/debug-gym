import json
from unittest.mock import MagicMock, patch

import pytest
from transformers import AutoTokenizer

from debug_gym.llms import HuggingFaceLLM
from debug_gym.llms.base import LLMConfig, LLMConfigRegistry
from debug_gym.llms.openai import OpenAILLM

# Run these tests with `pytest tests/llms/test_huggingface.py -m hf_tokenizer`
# to include the integration case that downloads the real Qwen tokenizer.


HF_MODEL_ID = "Qwen/Qwen3-0.6B"

MODEL_REGISTRY = {
    "qwen-3": {
        "model": HF_MODEL_ID,
        "tokenizer": HF_MODEL_ID,
        "context_limit": 4,
        "api_key": "test-api-key",
        "endpoint": "https://test-endpoint",
        "tags": ["vllm"],
        "tokenizer_kwargs": {"trust_remote_code": True},
    }
}


@pytest.fixture(scope="session")
def real_qwen3_tokenizer():
    try:
        return AutoTokenizer.from_pretrained(HF_MODEL_ID)
    except (
        OSError,
        ValueError,
        ImportError,
    ) as exc:  # pragma: no cover - network-dependent
        pytest.skip(f"Unable to load tokenizer {HF_MODEL_ID}: {exc}")


@patch.object(
    LLMConfigRegistry,
    "from_file",
    return_value=LLMConfigRegistry.register_all(MODEL_REGISTRY),
)
def test_tokenize_uses_hf_tokenizer_with_pad_fallback(mock_llm_config, logger_mock):
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)
    tokenizer.pad_token = None
    tokenizer.eos_token = "</s>"
    with patch(
        "debug_gym.llms.huggingface.AutoTokenizer.from_pretrained"
    ) as mock_auto_tokenizer:
        mock_auto_tokenizer.return_value = tokenizer
        llm = HuggingFaceLLM(model_name="qwen-3", logger=logger_mock)
        assert llm.tokenize("hello world") == ["hello", "Ġworld"]
        assert llm.count_tokens("hello world") == 2
        assert tokenizer.eos_token == "</s>"
        assert tokenizer.pad_token == "</s>"


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


@pytest.mark.hf_tokenizer
def test_chat_template_counts_with_real_tokenizer(real_qwen3_tokenizer, logger_mock):
    config = LLMConfig(
        model=HF_MODEL_ID,
        tokenizer=HF_MODEL_ID,
        context_limit=4,
        api_key="placeholder",
        endpoint="http://localhost",
        tags=["vllm"],
        tokenizer_kwargs={"trust_remote_code": True},
    )

    llm = HuggingFaceLLM(model_name="qwen-3", logger=logger_mock, llm_config=config)
    llm._hf_tokenizer = real_qwen3_tokenizer

    messages = [
        {"role": "system", "content": "Instructions"},
        {"role": "user", "content": "Hello"},
        {"role": "tool", "content": "Result"},
    ]

    counts = llm._get_message_token_counts(messages)

    normalized = llm._normalize_messages_for_template(messages)
    expected_counts = []
    prev_len = 0
    for idx in range(1, len(normalized) + 1):
        try:
            tokenized = real_qwen3_tokenizer.apply_chat_template(
                normalized[:idx], tokenize=True, add_generation_prompt=False
            )
        except TypeError:  # pragma: no cover - version-specific
            tokenized = real_qwen3_tokenizer.apply_chat_template(
                normalized[:idx], tokenize=True
            )
        token_ids = (
            tokenized.get("input_ids") if isinstance(tokenized, dict) else tokenized
        )
        if token_ids and isinstance(token_ids[0], list):
            token_ids = token_ids[0]
        if token_ids is None:
            pytest.skip("Tokenizer did not return token ids")
        expected_counts.append(len(token_ids) - prev_len)
        prev_len = len(token_ids)

    assert counts == expected_counts
    assert counts[-1] > 0


@pytest.mark.hf_tokenizer
def test_tokenize_and_count_tokens_with_real_tokenizer(
    real_qwen3_tokenizer, logger_mock
):
    config = LLMConfig(
        model=HF_MODEL_ID,
        tokenizer=HF_MODEL_ID,
        context_limit=4,
        api_key="placeholder",
        endpoint="http://localhost",
        tags=["vllm"],
        tokenizer_kwargs={"trust_remote_code": True},
    )

    llm = HuggingFaceLLM(model_name="qwen-3", logger=logger_mock, llm_config=config)
    llm._hf_tokenizer = real_qwen3_tokenizer

    text = "Hello world!"
    hf_ids = real_qwen3_tokenizer.encode(text, add_special_tokens=False)
    hf_tokens = real_qwen3_tokenizer.convert_ids_to_tokens(hf_ids)

    tokens = llm.tokenize(text)
    assert tokens == hf_tokens
    assert llm.count_tokens(text) == len(hf_ids)


@patch.object(
    LLMConfigRegistry,
    "from_file",
    return_value=LLMConfigRegistry.register_all(
        {
            "qwen": {
                "model": HF_MODEL_ID,
                "tokenizer": HF_MODEL_ID,
                "apply_chat_template": False,
                "context_limit": 4096,
                "api_key": "fake",
                "endpoint": "fake",
                "api_version": "1",
                "tags": ["vllm"],
            }
        }
    ),
)
def test_hf_tokenize_no_chat_template(mock_llm_config, logger_mock):
    llm = HuggingFaceLLM(model_name="qwen", logger=logger_mock)
    tokens = llm.tokenize("hello world")
    assert tokens == ["hello", "Ġworld"]


@patch.object(
    LLMConfigRegistry,
    "from_file",
    return_value=LLMConfigRegistry.register_all(
        {
            "qwen": {
                "model": HF_MODEL_ID,
                "tokenizer": HF_MODEL_ID,
                "apply_chat_template": True,
                "context_limit": 4096,
                "api_key": "fake",
                "endpoint": "fake",
                "api_version": "1",
                "tags": ["vllm"],
            }
        }
    ),
)
def test_hf_tokenize_apply_chat_template(mock_llm_config, logger_mock):
    llm = HuggingFaceLLM(model_name="qwen", logger=logger_mock)

    tokens = llm.tokenize("hello world")

    assert tokens == [
        "<|im_start|>",
        "assistant",
        "Ċ",
        "<think>",
        "ĊĊ",
        "</think>",
        "ĊĊ",
    ]


@patch.object(
    LLMConfigRegistry,
    "from_file",
    return_value=LLMConfigRegistry.register_all(
        {
            "qwen": {
                "model": HF_MODEL_ID,
                "tokenizer": HF_MODEL_ID,
                "apply_chat_template": True,
                "enable_thinking": True,
                "context_limit": 4096,
                "api_key": "fake",
                "endpoint": "fake",
                "api_version": "1",
                "tags": ["vllm"],
            }
        }
    ),
)
def test_hf_tokenize_apply_chat_template_thinking(mock_llm_config, logger_mock):
    llm = HuggingFaceLLM(model_name="qwen", logger=logger_mock)

    tokens = llm.tokenize("hello world")

    assert tokens == [
        "<|im_start|>",
        "assistant",
        "Ċ",
    ]
