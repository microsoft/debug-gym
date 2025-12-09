from unittest.mock import patch

import pytest
from transformers import AutoTokenizer

from debug_gym.llms import HuggingFaceLLM
from debug_gym.llms.base import LLMConfig, LLMConfigRegistry

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

MODEL_REGISTRY_WITH_CHAT_TEMPLATE = {
    "qwen-3": {
        **MODEL_REGISTRY["qwen-3"],
        "apply_chat_template": True,
    },
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
        messages = [{"role": "user", "content": "hello world"}]
        assert llm.tokenize(messages) == [["hello", "Ġworld"]]
        assert llm.count_tokens(messages) == 2
        assert tokenizer.eos_token == "</s>"
        assert tokenizer.pad_token == "</s>"


@patch.object(
    LLMConfigRegistry,
    "from_file",
    return_value=LLMConfigRegistry.register_all(MODEL_REGISTRY_WITH_CHAT_TEMPLATE),
)
def test_message_token_counts_uses_chat_template(mock_llm_config, logger_mock):
    llm = HuggingFaceLLM(model_name="qwen-3", logger=logger_mock)

    messages = [
        {"role": "system", "content": "Instructions"},
        {"role": "user", "content": "Hello world!"},
        {"role": "tool", "content": "Result"},
    ]

    counts = llm.count_tokens(messages)

    # When using chat template, each message gets template tokens added
    # The exact counts depend on the template format
    assert counts == 31


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
        {"role": "user", "content": "Hello world!"},
        {"role": "tool", "content": "Result"},
    ]

    counts = llm.count_tokens(messages)
    assert counts == 5


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
    messages = [{"role": "user", "content": text}]
    hf_ids = real_qwen3_tokenizer.encode(text, add_special_tokens=False)
    hf_tokens = real_qwen3_tokenizer.convert_ids_to_tokens(hf_ids)

    tokens = llm.tokenize(messages)
    assert tokens == [hf_tokens]
    assert llm.count_tokens(messages) == len(hf_ids)


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
    messages = [{"role": "user", "content": "hello world"}]
    tokens = llm.tokenize(messages)
    assert tokens == [["hello", "Ġworld"]]


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

    messages = [{"role": "user", "content": "hello world"}]
    tokens = llm.tokenize(messages)

    # When using chat template, all messages are tokenized together, so returns single list
    assert tokens == [
        [
            "<|im_start|>",
            "user",
            "Ċ",
            "hello",
            "Ġworld",
            "<|im_end|>",
            "Ċ",
            "<|im_start|>",
            "assistant",
            "Ċ",
            "<think>",
            "ĊĊ",
            "</think>",
            "ĊĊ",
        ]
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

    messages = [{"role": "user", "content": "hello world"}]
    tokens = llm.tokenize(messages)

    # When using chat template, all messages are tokenized together, so returns single list
    assert tokens == [
        [
            "<|im_start|>",
            "user",
            "Ċ",
            "hello",
            "Ġworld",
            "<|im_end|>",
            "Ċ",
            "<|im_start|>",
            "assistant",
            "Ċ",
        ]
    ]
