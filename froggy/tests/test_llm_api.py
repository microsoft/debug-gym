import pytest
from unittest.mock import patch, MagicMock
from froggy.agents.llm_api import (
    is_rate_limit_error,
    print_messages,
    merge_messages,
    TokenCounter,
    LLM,
    AsyncLLM,
    Human,
    Random,
    instantiate_llm,
)

# def test_is_rate_limit_error():
#     class MockException:
#         __module__ = "openai.error"
#         __name__ = "RateLimitError"
#         __class__ = type("MockExceptionClass", (object,), {"__module__": "openai.error", "__name__": "RateLimitError"})

#     exception = MockException()
#     assert is_rate_limit_error(exception) == True

def test_print_messages(capsys):
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
        {"role": "system", "content": "System message"},
    ]
    print_messages(messages)
    captured = capsys.readouterr()
    assert "Hello" in captured.out
    assert "Hi" in captured.out
    assert "System message" in captured.out

def test_merge_messages():
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "user", "content": "How are you?"},
        {"role": "assistant", "content": "Hi"},
    ]
    merged = merge_messages(messages)
    assert len(merged) == 2
    assert merged[0]["content"] == "Hello\n\nHow are you?"

def test_token_counter():
    counter = TokenCounter(model="gpt-4o")
    messages = [{"content": "Hello"}, {"content": "How are you?"}]
    assert counter(messages=messages) > 0
    assert counter(text="Hello") > 0

# @patch("openai.OpenAI")
# def test_llm(mock_openai):
#     mock_openai.return_value.chat.completions.create.return_value.choices[0].message.content = "Response"
#     llm = LLM(model_name="test_model", verbose=False)
#     messages = [{"role": "user", "content": "Hello"}]
#     response, token_usage = llm(messages)
#     assert response == "Response"
#     assert "prompt" in token_usage
#     assert "response" in token_usage

# @patch("llm_api.AsyncAzureOpenAI")
# @pytest.mark.asyncio
# async def test_async_llm(mock_async_openai):
#     mock_async_openai.return_value.chat.completions.create.return_value.choices[0].message.content = "Response"
#     llm = AsyncLLM(model_name="test_model", verbose=False)
#     messages = [{"role": "user", "content": "Hello"}]
#     response, token_usage = await llm(messages)
#     assert response == "Response"
#     assert "prompt" in token_usage
#     assert "response" in token_usage

# @patch("llm_api.prompt_toolkit_available", True)
# @patch("llm_api.prompt", return_value="User input")
# def test_human(mock_prompt):
#     human = Human()
#     messages = [{"role": "user", "content": "Hello"}]
#     info = {"available_commands": ["command1", "command2"]}
#     response, token_usage = human(messages, info)
#     assert response == "User input"
#     assert "prompt" in token_usage
#     assert "response" in token_usage

def test_random():
    random_llm = Random(seed=42, verbose=False)
    messages = [{"role": "user", "content": "Hello"}]
    info = {"available_commands": ["command1", "command2"]}
    response, token_usage = random_llm(messages, info)
    assert response in ["command1", "command2"]
    assert "prompt" in token_usage
    assert "response" in token_usage

# @patch("llm_api.LLM_CONFIGS", {"test_model": {"tokenizer": "gpt-4o", "context_limit": 4, "api_key": "key", "endpoint": "endpoint"}})
# def test_instantiate_llm():
#     config = {"llm_name": "test_model"}
#     llm = instantiate_llm(config, verbose=False, use_async=False)
#     assert isinstance(llm, LLM)

#     config = {"llm_name": "random", "random_seed": 42}
#     llm = instantiate_llm(config, verbose=False, use_async=False)
#     assert isinstance(llm, Random)

#     config = {"llm_name": "human"}
#     llm = instantiate_llm(config, verbose=False, use_async=False)
#     assert isinstance(llm, Human)

if __name__ == "__main__":
    pytest.main()