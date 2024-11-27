import sys
import unittest
from io import StringIO
from unittest.mock import patch, mock_open, MagicMock

class TestLLMAPI(unittest.TestCase):

    def setUp(self):
        # Mock the os.path.exists to return True and mock the open function
        self.patcher_exists = patch('os.path.exists', return_value=True)
        self.patcher_open = patch('builtins.open', new_callable=mock_open, read_data='{"test-model": {"model": "test-model", "max_tokens": 100, "tokenizer": "gpt-4o", "context_limit": 4, "api_key": "test-api-key", "endpoint": "https://test-endpoint", "api_version": "v1", "tags": ["azure openai"]}}')
        self.mock_exists = self.patcher_exists.start()
        self.mock_open = self.patcher_open.start()

        # Import the module after applying the mocks
        global is_rate_limit_error, print_messages, merge_messages, TokenCounter, LLM, AsyncLLM, Human, Random, instantiate_llm
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

    def tearDown(self):
        # Stop the patchers
        self.patcher_exists.stop()
        self.patcher_open.stop()

# def test_is_rate_limit_error():
#     class MockException:
#         __module__ = "openai.error"
#         __name__ = "RateLimitError"
#         __class__ = type("MockExceptionClass", (object,), {"__module__": "openai.error", "__name__": "RateLimitError"})

#     exception = MockException()
#     assert is_rate_limit_error(exception) == True

    def test_print_messages(self):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
            {"role": "system", "content": "System message"},
        ]
        captured_output = StringIO()
        sys.stdout = captured_output
        print_messages(messages)
        sys.stdout = sys.__stdout__
        captured = captured_output.getvalue()
        assert "Hello" in captured
        assert "Hi" in captured
        assert "System message" in captured

    def test_merge_messages(self):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "user", "content": "How are you?"},
            {"role": "assistant", "content": "Hi"},
        ]
        merged = merge_messages(messages)
        assert len(merged) == 2
        assert merged[0]["content"] == "Hello\n\nHow are you?"

    @patch('tiktoken.encoding_for_model')
    def test_token_counter(self, mock_encoding_for_model):
        mock_encoding = MagicMock()
        mock_encoding.encode = lambda x: x.split()
        mock_encoding_for_model.return_value = mock_encoding

        counter = TokenCounter(model="gpt-4o")
        messages = [{"content": "Hello"}, {"content": "How are you?"}]
        assert counter(messages=messages) > 0
        assert counter(text="Hello") > 0

    @patch('tiktoken.encoding_for_model')
    @patch("openai.resources.chat.completions.Completions.create")
    def test_llm(self, mock_openai, mock_encoding_for_model):
        mock_encoding = MagicMock()
        mock_encoding.encode = lambda x: x.split()
        mock_encoding_for_model.return_value = mock_encoding

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Response"
        mock_openai.return_value = mock_response

        llm = LLM(model_name="test-model", verbose=False)
        messages = [{"role": "user", "content": "Hello"}]
        response, token_usage = llm(messages)
        assert response == "Response"
        assert "prompt" in token_usage
        assert "response" in token_usage

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

    def test_random(self):
        random_llm = Random(seed=42, verbose=False)
        messages = [{"role": "user", "content": "Hello"}]
        info = {"available_commands": ["command1", "command2"]}
        response, token_usage = random_llm(messages, info)
        assert response in ["command1", "command2"]
        assert "prompt" in token_usage
        assert "response" in token_usage

    @patch('tiktoken.encoding_for_model')
    def test_instantiate_llm(self, mock_encoding_for_model):
        mock_encoding = MagicMock()
        mock_encoding.encode = lambda x: x.split()
        mock_encoding_for_model.return_value = mock_encoding
    
        config = {"llm_name": "test-model"}
        llm = instantiate_llm(config, verbose=False, use_async=False)
        assert isinstance(llm, LLM)

        config = {"llm_name": "random", "random_seed": 42}
        llm = instantiate_llm(config, verbose=False, use_async=False)
        assert isinstance(llm, Random)

        config = {"llm_name": "human"}
        llm = instantiate_llm(config, verbose=False, use_async=False)
        assert isinstance(llm, Human)
