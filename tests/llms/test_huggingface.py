from unittest.mock import patch

import pytest
from transformers import AutoTokenizer

from debug_gym.llms.base import LLM, LLMConfig, LLMConfigRegistry
from debug_gym.llms.huggingface import HuggingFaceLLM  # Import for patching in tests

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
        llm = LLM.instantiate(name="qwen-3", logger=logger_mock)
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
    llm = LLM.instantiate(name="qwen-3", logger=logger_mock)

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
    llm = LLM.instantiate(name="qwen", logger=logger_mock)
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
    llm = LLM.instantiate(name="qwen", logger=logger_mock)

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
    llm = LLM.instantiate(name="qwen", logger=logger_mock)

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


def test_hf_auto_injects_chat_template_kwargs_from_enable_thinking(logger_mock):
    """enable_thinking should auto-populate extra_body.chat_template_kwargs."""
    config = LLMConfig(
        model=HF_MODEL_ID,
        tokenizer=HF_MODEL_ID,
        context_limit=4,
        api_key="fake",
        endpoint="http://localhost",
        tags=["vllm"],
        enable_thinking=True,
    )
    llm = HuggingFaceLLM(model_name="qwen-3", logger=logger_mock, llm_config=config)
    extra_body = llm.config.generate_kwargs["extra_body"]
    assert extra_body["chat_template_kwargs"]["enable_thinking"] is True


def test_hf_auto_injects_thinking_false(logger_mock):
    """enable_thinking=False should inject the param as False."""
    config = LLMConfig(
        model=HF_MODEL_ID,
        tokenizer=HF_MODEL_ID,
        context_limit=4,
        api_key="fake",
        endpoint="http://localhost",
        tags=["vllm"],
        enable_thinking=False,
    )
    llm = HuggingFaceLLM(model_name="qwen-3", logger=logger_mock, llm_config=config)
    extra_body = llm.config.generate_kwargs["extra_body"]
    assert extra_body["chat_template_kwargs"]["enable_thinking"] is False


def test_hf_custom_thinking_param_name(logger_mock):
    """Models like Kimi-K2 use 'thinking' instead of 'enable_thinking'."""
    config = LLMConfig(
        model="moonshotai/Kimi-K2-Instruct",
        tokenizer="moonshotai/Kimi-K2-Instruct",
        context_limit=4,
        api_key="fake",
        endpoint="http://localhost",
        tags=["vllm"],
        enable_thinking=True,
        thinking_param_name="thinking",
    )
    llm = HuggingFaceLLM(model_name="kimi-k2", logger=logger_mock, llm_config=config)
    extra_body = llm.config.generate_kwargs["extra_body"]
    assert extra_body["chat_template_kwargs"]["thinking"] is True
    assert "enable_thinking" not in extra_body["chat_template_kwargs"]


def test_hf_explicit_chat_template_kwargs_not_overridden(logger_mock):
    """User-specified chat_template_kwargs should not be overridden."""
    config = LLMConfig(
        model=HF_MODEL_ID,
        tokenizer=HF_MODEL_ID,
        context_limit=4,
        api_key="fake",
        endpoint="http://localhost",
        tags=["vllm"],
        enable_thinking=True,
        generate_kwargs={
            "extra_body": {
                "chat_template_kwargs": {"enable_thinking": False},
            }
        },
    )
    llm = HuggingFaceLLM(model_name="qwen-3", logger=logger_mock, llm_config=config)
    # User explicitly set False — should not be overridden to True
    extra_body = llm.config.generate_kwargs["extra_body"]
    assert extra_body["chat_template_kwargs"]["enable_thinking"] is False
