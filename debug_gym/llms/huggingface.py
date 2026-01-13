from functools import lru_cache

from transformers import AutoTokenizer

from debug_gym.llms.openai import OpenAILLM

import json
import logging
from functools import lru_cache

import openai
import tiktoken
from openai import NOT_GIVEN, OpenAI


from debug_gym.gym.tools.tool import EnvironmentTool, ToolCall
from debug_gym.gym.utils import filter_non_utf8
from debug_gym.llms.base import (
    LLM,
    ContextLengthExceededError,
    LLMResponse,
    retry_on_exception,
)
from debug_gym.llms.constants import LLM_API_KEY_PLACEHOLDER, LLM_ENDPOINT_PLACEHOLDER


@lru_cache(maxsize=5)
def _get_hf_tokenizer(tokenizer_name: str, tokenizer_kwargs_tuple: tuple):
    """Cache HuggingFace tokenizers to limit memory usage (max 5 different tokenizers).

    Note: tokenizer_kwargs is converted to a tuple of items for hashability.
    """
    tokenizer_kwargs = dict(tokenizer_kwargs_tuple) if tokenizer_kwargs_tuple else {}
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, **tokenizer_kwargs)

    # Ensure we have a pad token to avoid downstream warnings
    if (
        getattr(tokenizer, "pad_token", None) is None
        and getattr(tokenizer, "eos_token", None) is not None
    ):
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


class HuggingFaceLLM(OpenAILLM):
    """LLM implementation backed by a Hugging Face tokenizer."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._hf_tokenizer = None

    def _load_tokenizer(self):
        if self._hf_tokenizer is None:
            tokenizer_kwargs = getattr(self.config, "tokenizer_kwargs", None) or {}
            # Convert dict to tuple of items for hashability in lru_cache
            tokenizer_kwargs_tuple = tuple(sorted(tokenizer_kwargs.items()))
            try:
                self._hf_tokenizer = _get_hf_tokenizer(
                    self.tokenizer_name, tokenizer_kwargs_tuple
                )
            except OSError:
                raise ValueError(
                    f"Tokenizer `{self.tokenizer_name}` not found for model "
                    f"{self.model_name}, make sure you have access to "
                    "the model (e.g., HuggingFace API key is correctly set)."
                )
        return self._hf_tokenizer

    def tokenize(self, messages: list[dict]) -> list[list[str]]:
        tokenizer = self._load_tokenizer()

        if self.apply_chat_template:
            # When applying chat template, tokenize all messages together
            # then return as a single list
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
            tokens = tokenizer.tokenize(text)
            # Return as list with single element (all tokens together)
            return [tokens]
        else:
            # Tokenize each message individually
            result = []
            for msg in messages:
                content = str(msg["content"])
                tokens = tokenizer.tokenize(content)
                result.append(tokens)
            return result

    def generate(self, messages, tools, **kwargs) -> LLMResponse:
        # set max tokens if not provided
        kwargs["max_tokens"] = kwargs.get("max_tokens", NOT_GIVEN)
        api_call = retry_on_exception(
            self._perform_chat_completion,
            self.need_to_be_retried,
        )
        try:
            if tools:
                response = api_call(
                    model=self.config.model,
                    messages=messages,
                    tools=self.define_tools(tools),
                    tool_choice="required",
                    **kwargs,
                )
            else:
                response = api_call(
                    model=self.config.model,
                    messages=messages,
                    **kwargs,
                )
        except openai.BadRequestError as e:
            # Handle specific error for context length exceeded, otherwise just propagate the error
            if self.is_context_length_error(e):
                raise ContextLengthExceededError
            raise e
        if getattr(response, "choices", None) is None:
            self.logger.debug(
                "OpenAI response missing 'choices' key; response type=%s",
                type(response),
            )
            raise OpenAIResponseParsingError(
                "OpenAI chat completion returned unexpected payload without 'choices'"
            )
        try:
            choice = response.choices[0]
            message = choice.message
        except (IndexError, TypeError, AttributeError) as exc:
            self.logger.debug(
                "OpenAI response choices could not provide a message: %s", exc
            )
            raise OpenAIResponseParsingError(
                "OpenAI chat completion returned no usable choice message"
            ) from exc

        # LLM may select multiple tool calls, we only care about the first action
        if not getattr(message, "tool_calls", None):
            # LLM failed to call a tool
            tool_call = None
        else:
            tool_call = message.tool_calls[0]
            assert tool_call.type == "function"

        # In openai call, the content is in response.choices[0].message.content
        # In some models hosted on vllm, e.g., qwen-3, there could be content in both (when reasoning is enabled)
        # response.choices[0].message.content and response.choices[0].message.reasoning_content
        # https://qwen.readthedocs.io/en/latest/deployment/vllm.html#parsing-thinking-content
        _content = message.content
        _reasoning_content = None
        if hasattr(message, "reasoning_content"):
            _reasoning_content = message.reasoning_content

        parsed_tool = self.parse_tool_call_response(tool_call)

        llm_response = LLMResponse(
            prompt=messages,
            response=_content,
            reasoning_response=_reasoning_content,
            tool=parsed_tool,
            prompt_token_count=response.usage.prompt_tokens,
            response_token_count=response.usage.completion_tokens,
        )
        return llm_response