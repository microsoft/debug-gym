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
    retry_on_rate_limit,
)
from debug_gym.llms.constants import LLM_API_KEY_PLACEHOLDER, LLM_ENDPOINT_PLACEHOLDER

# Set logging level down to WARNING for endpoint queries.
logging.getLogger("openai").setLevel(logging.WARNING)


class OpenAILLM(LLM):

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

    def tokenize(self, text: str) -> list[str]:
        if getattr(self, "_tk_func", None) is None:
            try:
                self._tk_func = tiktoken.encoding_for_model(self.tokenizer_name).encode
            except KeyError:
                try:  # Try to load from transformers.
                    self._tk_func = AutoTokenizer.from_pretrained(
                        self.tokenizer_name
                    ).tokenize
                except OSError:
                    raise ValueError(
                        f"Tokenizer `{self.tokenizer_name}` not found for model "
                        f"{self.model_name}, make sure you have access to "
                        "the model (e.g., HuggingFace API key is correctly set)."
                    )
        return self._tk_func(text)

    def is_rate_limit_error(self, exception) -> bool:
        # List of fully qualified names of RateLimitError exceptions from various libraries
        rate_limit_errors = [
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

        is_error = exception_full_name in rate_limit_errors
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
                is_error = False
                logger = self.logger.warning
        if (
            exception_full_name == "openai.BadRequestError"
            and len(self.config.tags) > 0
            and "vllm" not in self.config.tags
        ):
            # only retry when a such error occurs on a model hosting on vllm
            is_error = False

        logger(
            f"Error calling {self.model_name}: {exception_full_name!r} {
                exception.message if hasattr(exception, 'message') else exception
            }"
        )

        return is_error

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

    def format_tool_call_history(
        self, history_info: EnvInfo, response: LLMResponse
    ) -> list[dict]:
        _messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "type": "function",
                        "id": response[0].tool.id,
                        "function": {
                            "name": response[0].tool.name,
                            "arguments": json.dumps(response[0].tool.arguments),
                        },
                    },
                ],
            },
            {
                "role": "tool",
                "tool_call_id": history_info.action.id,
                "name": history_info.action.name,
                "content": filter_non_utf8(
                    f"{history_info.step_observation.observation}"
                ),
            },
        ]
        return _messages

    def generate(self, messages, tools, **kwargs) -> LLMResponse:
        # set max tokens if not provided
        kwargs["max_tokens"] = kwargs.get("max_tokens", NOT_GIVEN)
        try:
            response = retry_on_rate_limit(
                self.client.chat.completions.create, self.is_rate_limit_error
            )(
                model=self.config.model,
                messages=messages,
                tools=self.define_tools(tools),
                tool_choice="required",
                **kwargs,
            )
        except openai.BadRequestError as e:
            # Handle specific error for context length exceeded, otherwise just propagate the error
            if e.code == "context_length_exceeded":
                raise ContextLengthExceededError
            raise

        # LLM may select multiple tool calls, we only care about the first action
        if response.choices[0].message.tool_calls is None:
            # LLM failed to call a tool
            tool_call = None
        else:
            tool_call = response.choices[0].message.tool_calls[0]
            assert tool_call.type == "function"

        llm_response = LLMResponse(
            prompt=messages,
            response=response.choices[0].message.content,
            tool=self.parse_tool_call_response(tool_call),
            prompt_token_count=response.usage.prompt_tokens,
            response_token_count=response.usage.completion_tokens,
        )
        return llm_response
