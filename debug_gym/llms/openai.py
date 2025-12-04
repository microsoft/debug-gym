import json
import logging

import openai
import tiktoken
from openai import NOT_GIVEN, OpenAI
from transformers import AutoTokenizer

from debug_gym.gym.envs.env import EnvInfo
from debug_gym.gym.tools.tool import EnvironmentTool, ToolCall
from debug_gym.gym.utils import filter_non_utf8
from debug_gym.llms.base import (
    LLM,
    ContextLengthExceededError,
    LLMResponse,
    retry_on_exception,
)
from debug_gym.llms.constants import LLM_API_KEY_PLACEHOLDER, LLM_ENDPOINT_PLACEHOLDER

# Set logging level down to WARNING for endpoint queries.
logging.getLogger("openai").setLevel(logging.WARNING)


class OpenAILLM(LLM):

    context_length_error_code = [
        "context_length_exceeded",
        "model_max_prompt_tokens_exceeded",
        "string_above_max_length",
    ]
    context_length_error_message_keywords = [
        "maximum context length",
    ]

    def is_context_length_error(self, exception: Exception) -> bool:
        if (
            hasattr(exception, "code")
            and exception.code in self.context_length_error_code
        ):
            return True
        if hasattr(exception, "message"):
            for keyword in self.context_length_error_message_keywords:
                if keyword in exception.message:
                    return True
        return False

    @property
    def client(self):
        if getattr(self, "_client", None) is None:
            if self.config.api_key in [
                LLM_API_KEY_PLACEHOLDER,
                None,
            ] or self.config.endpoint in [LLM_ENDPOINT_PLACEHOLDER, None]:
                raise ValueError(
                    "OpenAI API key and endpoint are required. Please add them to the config. "
                    "If using Azure OpenAI, please add `azure openai` to the tags."
                )
            self._client = OpenAI(
                api_key=self.config.api_key,
                base_url=self.config.endpoint,
                timeout=None,
            )
        return self._client

    def tokenize(self, messages: list[dict]) -> list[list[str]]:
        if getattr(self, "_tk_func", None) is None:
            try:
                encoder = tiktoken.encoding_for_model(self.tokenizer_name)
                # For tiktoken, encode returns list of ints, we need to convert to list of "tokens"
                self._tk_func = lambda text: [str(t) for t in encoder.encode(text)]
            except KeyError:
                raise ValueError(
                    f"Tokenizer `{self.tokenizer_name}` not found for model "
                    f"{self.model_name}. If using Hugging Face models, please "
                    f"set tag `vllm` to load the HuggingFaceLLM class instead."
                )
        # Tokenize each message individually
        result = []
        for msg in messages:
            content = str(msg.get("content", msg.get("tool_calls", msg)))
            tokens = self._tk_func(content)
            result.append(tokens)
        return result

    def need_to_be_retried(self, exception) -> bool:
        # List of fully qualified names of RateLimitError exceptions from various libraries
        _errors = [
            "openai.APIStatusError",
            "openai.APITimeoutError",
            "openai.error.Timeout",
            "openai.error.RateLimitError",
            "openai.error.ServiceUnavailableError",
            "openai.Timeout",
            "openai.APIError",
            "openai.APIConnectionError",
            "openai.RateLimitError",
            "openai.PermissionDeniedError",
            "openai.BadRequestError",
            # Add more as needed
        ]
        exception_full_name = (
            f"{exception.__class__.__module__}.{exception.__class__.__name__}"
        )

        need_to_retry = exception_full_name in _errors
        logger = self.logger.debug

        # Ignore error that are not rate limit errors
        if exception_full_name == "openai.APIStatusError":
            if not (
                "'status': 429" in exception.message  # Rate Limit Exceeded
                or "'status': 504" in exception.message  # Gateway Timeout
                or (  # A previous prompt was too large
                    "'status': 413" in exception.message
                    and "A previous prompt was too large." in exception.message
                )
            ):
                need_to_retry = False
                logger = self.logger.warning
        if (
            exception_full_name == "openai.BadRequestError"
            and len(self.config.tags) > 0
            and "vllm" not in self.config.tags
        ):
            # only retry when a such error occurs on a model hosting on vllm
            need_to_retry = False

        if self.is_context_length_error(exception):
            need_to_retry = False

        logger(
            f"Error calling {self.model_name}: {exception_full_name!r}\n"
            f"{exception.message if hasattr(exception, 'message') else exception}"
        )

        return need_to_retry

    def _perform_chat_completion(self, **kwargs):
        """Invoke the OpenAI chat completion endpoint.

        Kept as a separate method so subclasses can customize how the client is
        retrieved per attempt (for example, to refresh authentication headers
        such as GitHub Copilot HMAC tokens).
        """

        return self.client.chat.completions.create(**kwargs)

    def define_tools(self, tool_call_list: list[EnvironmentTool]) -> list[dict]:
        """Translates the list of tools into a format that is specifically defined by each LLM.
        OpenAI function calling format: https://platform.openai.com/docs/guides/function-calling
        """
        output = []
        for tool in tool_call_list:
            _tool = {"type": "function", "function": {}}
            _function = _tool["function"]
            _function["name"] = tool.name
            _function["description"] = tool.description
            _function["parameters"] = {
                "type": "object",
                "properties": tool.arguments,
                "additionalProperties": False,
            }
            # _function["strict"] = True  # this is not supported by reasoning models such as o3
            if len(tool.arguments) > 0:
                _function["parameters"]["required"] = list(tool.arguments.keys())
            output.append(_tool)
        return output

    def parse_tool_call_response(self, response) -> ToolCall:
        """Parse the tool response from different LLMs and return it as a ToolCall object.
        An example of the OpenAI tool response is:
        {
            "id": "call_12345xyz",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": "{\"latitude\":48.8566,\"longitude\":2.3522}"
            }
        }
        """
        if response is None:
            return ToolCall(
                id="empty_tool_response",
                name="empty_tool_response",
                arguments={},
            )
        return ToolCall(
            id=response.id,
            name=response.function.name,
            arguments=json.loads(response.function.arguments),
        )

    def convert_response_to_message(self, response: LLMResponse) -> dict:
        message = {
            "role": "assistant",
            "tool_calls": [
                {
                    "type": "function",
                    "id": response.tool.id,
                    "function": {
                        "name": response.tool.name,
                        "arguments": json.dumps(response.tool.arguments),
                    },
                },
            ],
            "content": filter_non_utf8(f"{response.response}"),
        }
        if response.reasoning_response:
            message["reasoning_content"] = filter_non_utf8(
                f"{response.reasoning_response}"
            )
        return message

    def convert_observation_to_message(
        self, observation: str, last_tool_call_id=None, last_tool_call_name=None
    ) -> dict:
        if last_tool_call_id is None:
            # This is the initial state, no action taken yet
            return {
                "role": "user",
                "content": filter_non_utf8(observation),
            }
        else:
            # This is a step with an action taken
            return {
                "role": "tool",
                "tool_call_id": last_tool_call_id,
                "name": last_tool_call_name,
                "content": filter_non_utf8(observation),
            }

    def generate(self, messages, tools, **kwargs) -> LLMResponse:
        # set max tokens if not provided
        kwargs["max_tokens"] = kwargs.get("max_tokens", NOT_GIVEN)
        try:
            response = retry_on_exception(
                self._perform_chat_completion, self.need_to_be_retried
            )(
                model=self.config.model,
                messages=messages,
                tools=self.define_tools(tools),
                tool_choice="auto",
                **kwargs,
            )
        except openai.BadRequestError as e:
            # Handle specific error for context length exceeded, otherwise just propagate the error
            if self.is_context_length_error(e):
                raise ContextLengthExceededError
            raise e
        # LLM may select multiple tool calls, we only care about the first action
        if not response.choices[0].message.tool_calls:
            # LLM failed to call a tool
            tool_call = None
        else:
            tool_call = response.choices[0].message.tool_calls[0]
            assert tool_call.type == "function"

        # In openai call, the content is in response.choices[0].message.content
        # In some models hosted on vllm, e.g., qwen-3, there could be content in both (when reasoning is enabled)
        # response.choices[0].message.content and response.choices[0].message.reasoning_content
        # https://qwen.readthedocs.io/en/latest/deployment/vllm.html#parsing-thinking-content
        _content = response.choices[0].message.content
        _reasoning_content = None
        if hasattr(response.choices[0].message, "reasoning_content"):
            _reasoning_content = response.choices[0].message.reasoning_content

        llm_response = LLMResponse(
            prompt=messages,
            response=_content,
            reasoning_response=_reasoning_content,
            tool=self.parse_tool_call_response(tool_call),
            prompt_token_count=response.usage.prompt_tokens,
            response_token_count=response.usage.completion_tokens,
        )
        return llm_response
