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
