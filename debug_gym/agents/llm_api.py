import json
import logging
import os
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import openai
import tiktoken
import yaml
from openai import NOT_GIVEN, AzureOpenAI, OpenAI, PermissionDeniedError
from tenacity import (
    retry,
    retry_if_exception,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)
from termcolor import colored
from transformers import AutoTokenizer

from debug_gym.agents.utils import print_messages
from debug_gym.gym.envs.env import EnvInfo
from debug_gym.gym.tools.tool import EnvironmentTool, ToolCall
from debug_gym.logger import DebugGymLogger

prompt_toolkit_available = False
try:
    # For command line history and autocompletion.
    from prompt_toolkit import prompt
    from prompt_toolkit.completion import WordCompleter
    from prompt_toolkit.history import InMemoryHistory

    prompt_toolkit_available = sys.stdout.isatty()
except ImportError:
    pass


# Set logging level down to WARNING for endpoint queries.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("azure.identity").setLevel(logging.WARNING)
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(
    logging.WARNING
)


DEFAULT_LLM_CONFIG = Path.joinpath(Path.home(), ".config", "debug_gym", "llm.yaml")
LLM_API_KEY_PLACEHOLDER = "[YOUR_API_KEY]"
LLM_ENDPOINT_PLACEHOLDER = "[YOUR_ENDPOINT]"
LLM_SCOPE_PLACEHOLDER = "[YOUR_SCOPE]"
LLM_CONFIG_TEMPLATE = f"""# Please edit this file replacing the placeholders with your own values.
gpt-4o:
  model: gpt-4o
  tokenizer: gpt-4o
  endpoint: "{LLM_ENDPOINT_PLACEHOLDER}"
  api_key: "{LLM_API_KEY_PLACEHOLDER}"
  tags: [gpt-4o, azure openai, GCR]
  api_version: "2024-09-01-preview"
  context_limit: 128
  generate_kwargs:
    temperature: 0.5

o1-mini:
  model: o1-mini
  tokenizer: gpt-4o
  endpoint: "{LLM_ENDPOINT_PLACEHOLDER}"
  api_key: "{LLM_API_KEY_PLACEHOLDER}"
  tags: [gpt-4o, azure openai, GCR]
  api_version: "2024-09-01-preview"
  context_limit: 128
  system_prompt_support: false
  ignore_kwargs: [temperature, top_p, presence_penalty, frequency_penalty, logprobs, top_logprobs, logit_bias, max_tokens]

gpt-4o-az-login:
  model: gpt-4o
  tokenizer: gpt-4o
  endpoint: "{LLM_ENDPOINT_PLACEHOLDER}"
  scope: "{LLM_SCOPE_PLACEHOLDER}"
  tags: [gpt-4o, azure openai, GCR]
  api_version: "2024-09-01-preview"
  context_limit: 128
  generate_kwargs:
    temperature: 0.5

deepseek-r1-distill-qwen-32b:
  model: deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
  tokenizer: Qwen/Qwen2.5-32B
  endpoint: "{LLM_ENDPOINT_PLACEHOLDER}"
  api_key: "{LLM_API_KEY_PLACEHOLDER}"
  tags: [DeepSeek-R1-Distill-Qwen-32B, H100]
  system_prompt_support: false
  context_limit: 128
  reasoning_end_token: "</think>"
  generate_kwargs:
    temperature: 0.5

claude-3.7:
  model: claude-3-7-sonnet-20250219
  tokenizer: claude-3-7-sonnet-20250219
  tags: [anthropic, claude, claude-3.7]
  context_limit: 100
  api_key: "{LLM_API_KEY_PLACEHOLDER}"
  generate_kwargs:
    max_tokens: 8192
    temperature: 0.5

claude-3.7-thinking:
  model: claude-3-7-sonnet-20250219
  tokenizer: claude-3-7-sonnet-20250219
  tags: [anthropic, claude, claude-3.7]
  context_limit: 100
  api_key: "{LLM_API_KEY_PLACEHOLDER}"
  generate_kwargs:
    max_tokens: 20000
    temperature: 1
    thinking:
      type: enabled
      budget_tokens: 16000
"""


def retry_on_rate_limit(
    func, is_rate_limit_error_func, multiplier=1, max_wait=40, max_attempts=100
):
    """Executes a function with retry logic for rate limits. Never retries on KeyboardInterrupt.
    Args:
        func: The function to execute with retries
        is_rate_limit_error_func: Function that checks if an exception is a rate limit error
        *args, **kwargs: Arguments to pass to the function

    Returns:
        The result of the function call
    """
    retry_function = retry(
        retry=(
            retry_if_not_exception_type(KeyboardInterrupt)
            & retry_if_exception(is_rate_limit_error_func)
        ),
        wait=wait_random_exponential(multiplier=multiplier, max=max_wait),
        stop=stop_after_attempt(max_attempts),
    )
    return retry_function(func)


@dataclass
class LLMConfig:
    """Configuration dataclass for LLM models"""

    # Required fields
    model: str
    context_limit: int
    # Optional fields
    api_key: Optional[str] = None
    endpoint: Optional[str] = None
    tokenizer: Optional[str] = None
    reasoning_end_token: Optional[str] = None
    system_prompt_support: bool = True
    ignore_kwargs: List[str] = None
    tags: List[str] = None
    # Azure OpenAI specific fields
    api_version: Optional[str] = None
    scope: Optional[str] = None
    # Custom parameters to pass to generate
    generate_kwargs: dict = None

    def __post_init__(self):
        # Set tokenizer to model if not specified
        if self.tokenizer is None:
            self.tokenizer = self.model
        # Initialize empty lists
        if self.ignore_kwargs is None:
            self.ignore_kwargs = []
        if self.tags is None:
            self.tags = []
        if self.generate_kwargs is None:
            self.generate_kwargs = {}


@dataclass
class LLMConfigRegistry:
    """Registry holding a collection of LLM configurations"""

    configs: dict[str, LLMConfig] = None

    def __post_init__(self):
        if self.configs is None:
            self.configs = {}

    def get(self, model_name: str) -> LLMConfig:
        """Get a model configuration by name"""
        if model_name not in self.configs:
            raise ValueError(
                f"Model {model_name} not found in llm config registry. Please make "
                "sure the model is registered and the config file is correctly set."
            )
        return self.configs[model_name]

    def register(self, model_name: str, config: dict) -> LLMConfig:
        """Register a new model configuration from a dictionary"""
        llm_config = LLMConfig(**config)
        self.configs[model_name] = llm_config
        return llm_config

    @classmethod
    def register_all(cls, configs: dict) -> None:
        """Register multiple model configurations from a dictionary"""
        registry = cls()
        # Convert each model configuration to LLMConfig objects
        for model_name, model_config in configs.items():
            registry.register(model_name, model_config)
        return registry

    @classmethod
    def from_file(cls, config_file_path: str | None = None) -> "LLMConfigRegistry":
        """Load the LLM configuration from a JSON file"""
        if config_file_path is None:
            config_file_path = os.environ.get(
                "LLM_CONFIG_FILE_PATH", DEFAULT_LLM_CONFIG
            )
        try:
            with open(config_file_path) as f:
                raw_llm_config = yaml.safe_load(f)
            return cls.register_all(raw_llm_config)
        except FileNotFoundError:
            raise FileNotFoundError(f"Cannot find llm config file: {config_file_path}")

    def __getitem__(self, model_name: str) -> LLMConfig:
        """Allow dictionary-like access to configurations"""
        return self.get(model_name)

    def __contains__(self, model_name: str) -> bool:
        """Check if a model name exists in the registry"""
        return model_name in self.configs


@dataclass
class TokenUsage:
    prompt: int
    response: int


@dataclass
class LLMResponse:
    prompt: list[dict] | str  # either a string or a list of messages.
    response: str
    tool: ToolCall
    token_usage: TokenUsage | None = None

    def __init__(
        self,
        prompt: list[dict] | str,
        response: str,
        tool: ToolCall = None,
        prompt_token_count: int = None,
        response_token_count: int = None,
        token_usage: TokenUsage = None,
    ):
        self.prompt = prompt
        self.response = response
        self.tool = tool
        if prompt_token_count is not None and response_token_count is not None:
            self.token_usage = TokenUsage(prompt_token_count, response_token_count)
        else:
            self.token_usage = token_usage


class ContextLengthExceededError(Exception):
    """Exception raised when the context length of an LLM request is exceeded."""

    pass


class LLM(ABC):

    def __init__(
        self,
        model_name: str,
        logger: DebugGymLogger | None = None,
        llm_config: LLMConfig | None = None,
        llm_config_file: str | None = None,
    ):
        self.model_name = model_name
        self.logger = logger or DebugGymLogger("debug-gym")
        if llm_config is not None and llm_config_file is not None:
            logger.warning(
                "Both llm_config and llm_config_file are provided, using llm_config."
            )
        self.config = (
            llm_config or LLMConfigRegistry.from_file(llm_config_file)[model_name]
        )
        self.tokenizer_name = self.config.tokenizer
        self.context_length = self.config.context_limit * 1000
        self.reasoning_end_token = self.config.reasoning_end_token

        self.logger.debug(
            f"Using {self.model_name} with max context length of {
                self.context_length:,} tokens."
        )

    @classmethod
    def instantiate(
        cls,
        llm_name: str,
        llm_config_file_path: str | None = None,
        logger: DebugGymLogger | None = None,
    ) -> "LLM":
        """Creates an instance of the appropriate LLM class based on the configuration.

        Args:
            llm_name: Name of the LLM model to instantiate.
            llm_config_file_path: Optional path to the LLM configuration file.
            logger: Optional DebugGymLogger for logging.

        Returns:
            An instance of the appropriate LLM class.
        """
        logger = logger or DebugGymLogger("debug-gym")
        if llm_name == "human":
            return Human(llm_name, logger=logger)

        llm_config = LLMConfigRegistry.from_file(llm_config_file_path)[llm_name]

        tags = llm_config.tags
        if "azure openai" in tags:
            klass = AzureOpenAILLM
        elif "anthropic" in tags:
            klass = AnthropicLLM
        else:
            klass = OpenAILLM
        llm = klass(llm_name, logger=logger, llm_config=llm_config)
        return llm

    @abstractmethod
    def generate(self, messages, tools, **kwargs) -> LLMResponse:
        """Generate a response given some messages and return it as an LLMResponse object.
        Raises ContextLengthExceededError if the context length is exceeded.
        The method should be overridden by subclasses."""
        pass

    @abstractmethod
    def tokenize(self, text: str) -> list[str]:
        """Abstract method to tokenize a text."""
        pass

    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text."""
        return len(self.tokenize(text))

    @abstractmethod
    def define_tools(self, tool_call_list: dict[str, EnvironmentTool]) -> list[dict]:
        """Translates the list of tools into a format that is specifically defined by each LLM.
        The method should be overridden by subclasses.
        """
        raise NotImplementedError(
            "The define_tools method should be overridden by subclasses."
        )

    @abstractmethod
    def parse_tool_call_response(self, response) -> ToolCall:
        """Parse the tool response from different LLMs and return it as a ToolCall object.
        The method should be overridden by subclasses.
        """
        raise NotImplementedError(
            "The parse_tool_call_response method should be overridden by subclasses."
        )

    @abstractmethod
    def format_tool_call_history(
        self, history_info: EnvInfo, response: LLMResponse
    ) -> list[dict]:
        """Format the tool call history for different LLMs.
        The method should be overridden by subclasses.
        """
        raise NotImplementedError(
            "The format_tool_call_history method should be overridden by subclasses."
        )

    def __call__(self, messages, tools, *args, **kwargs) -> LLMResponse:
        """Prepares messages and kwargs, then call `generate` which
        should be implemented by subclasses. Returns an LLMResponse object
        with the prompt, response and token usage.
        """
        from debug_gym.agents.utils import trim_prompt_messages

        # Add custom generation parameters from config
        for key, value in self.config.generate_kwargs.items():
            # Only set if not already specified in the call
            if key not in kwargs:
                kwargs[key] = value

        # replace system prompt by user prompt if not supported
        if not self.config.system_prompt_support:
            self.logger.debug(
                "System prompt is not supported by the model, it will be replaced by user prompt."
            )
            for i, m in enumerate(messages):
                if m["role"] == "system":
                    messages[i]["role"] = "user"

        # ignore specific kwargs that are not supported by the model
        if self.config.ignore_kwargs:
            self.logger.debug(
                f"LLM arguments {", ".join(self.config.ignore_kwargs)} "
                "are not supported by the model, they will be ignored."
            )
            for kw in self.config.ignore_kwargs:
                if kw in kwargs:
                    del kwargs[kw]

        def generate_with_drop_message_and_retry(messages, tools, **kwargs):
            """Generate a response. If context length is exceeded, apply trim_prompt_messages and retry."""
            if not messages:
                raise ValueError("No messages provided for generation.")
            try:
                llm_response = self.generate(messages, tools, **kwargs)
            except ContextLengthExceededError:
                self.logger.info(
                    f"Prompt is too long. {self.model_name} only allows for {self.context_length:,} tokens."
                )
                messages = trim_prompt_messages(
                    messages, self.context_length, self.count_tokens
                )
                llm_response = self.generate_with_drop_message_and_retry(
                    messages, tools, **kwargs
                )
                self.logger.info(
                    f"Prompt truncated to {llm_response.token_usage.prompt:,} tokens."
                )

            print_messages(messages, self.logger)
            return llm_response

        llm_response = generate_with_drop_message_and_retry(messages, tools, **kwargs)

        if llm_response.tool is None:
            # for error analysis purposes
            tool = {
                "id": "empty_tool_response",
                "name": "empty_tool_response",
                "arguments": {},
            }
            llm_response.tool = tool
            self.logger.warning(
                "Tool response is empty. The model may not have called a tool."
            )

        self.logger.info(colored(llm_response.tool, "green"))

        return llm_response


class AnthropicLLM(LLM):

    @property
    def client(self):
        if getattr(self, "_client", None) is None:
            from anthropic import Anthropic

            if self.config.api_key in [LLM_API_KEY_PLACEHOLDER, None]:
                raise ValueError(
                    "API key is required for Anthropic. Please add it to the config."
                )
            self._client = Anthropic(api_key=self.config.api_key)
        return self._client

    def tokenize(self, text: str) -> list[str]:
        raise NotImplementedError("Tokenization is not supported by Anthropic.")

    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text using the Anthropic API.
        Dump content to JSON for cases such as:
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "id123",
                        "content": "results",
                    }
                ],
            }
        """
        messages = [{"role": "user", "content": [{"type": "text", "text": text}]}]
        try:
            response = self.client.messages.count_tokens(
                model=self.tokenizer_name, messages=messages
            )
            return response.input_tokens
        except Exception as e:
            self.logger.warning(
                f"Error calling Claude token count API: {e!r}. "
                "The message was: {messages}."
                "Will return 0 tokens."
            )
        return 0

    def is_rate_limit_error(self, exception) -> bool:
        rate_limit_errors = [
            "anthropic.RateLimitError",
            "anthropic.OverloadedError",
            "anthropic._exceptions.OverloadedError",
            "anthropic.InternalServerError",
        ]
        exception_full_name = (
            f"{exception.__class__.__module__}.{exception.__class__.__name__}"
        )

        self.logger.debug(
            f"Error calling {self.model_name}: {exception_full_name!r} "
            f"{exception.message if hasattr(exception, 'message') else exception}"
        )
        return exception_full_name in rate_limit_errors

    def define_tools(self, tool_call_list: dict[str, EnvironmentTool]) -> list[dict]:
        """Translates the list of tools into a format that is specifically defined by each LLM.
        Anthropic function calling format: https://docs.anthropic.com/en/docs/build-with-claude/tool-use/overview
        """
        output = []
        for tool in tool_call_list:
            _tool = {}
            _tool["name"] = tool.name
            _tool["description"] = tool.description
            _tool["input_schema"] = {
                "type": "object",
                "properties": tool.arguments,
            }
            if len(tool.arguments) > 0:
                _tool["input_schema"]["required"] = list(tool.arguments.keys())
            output.append(_tool)
        return output

    def parse_tool_call_response(self, response) -> ToolCall:
        """Parse the tool response from different LLMs and return it as a ToolCall object.
        An example of the Anthropic tool response is:
        ToolUseBlock(
            id='toolu_staging_01FMRQ9pZniZqFUGQwTcFU4N',
            input={
                'positive_score': 0.9,
                'negative_score': 0.0,
                'neutral_score': 0.1
            },
            name='print_sentiment_scores',
            type='tool_use',
        )
        """
        return ToolCall(
            id=response.id,
            name=response.name,
            arguments=response.input,
        )

    def format_tool_call_history(
        self, history_info: EnvInfo, response: LLMResponse
    ) -> list[dict]:
        _messages = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": response[0].tool.id,  # 'toolu_01SdR84CsnTKRpdH4zwFjvGj'
                        "name": response[0].tool.name,  # 'view'
                        "input": response[
                            0
                        ].tool.arguments,  # {'path': 'hangman_test.py'}
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": history_info.action.id,  # 'toolu_01SdR84CsnTKRpdH4zwFjvGj'
                        "content": f"{history_info.step_observation.observation}",  # 'Viewing `hangman_test.py`. The file is read-only, it is not editable.'
                    }
                ],
            },
        ]
        return _messages

    def generate(self, messages, tools, **kwargs) -> LLMResponse:
        import anthropic

        system_prompt = " "  # weird exceptions sometimes if empty
        user_assistant_prompt = []
        for message in messages:
            if message["content"] == "":
                continue
            if message["role"] == "system":
                system_prompt = message["content"]
            elif message["role"] in ["user", "assistant", "tool"]:
                user_assistant_prompt.append(
                    {
                        "role": message["role"],
                        "content": message["content"],
                    }
                )
            else:
                raise ValueError(f"Unknown role: {message['role']}")
        if len(user_assistant_prompt) == 0:
            user_assistant_prompt = [
                {
                    "role": "user",
                    "content": "Your answer is: ",
                }
            ]

        try:
            # https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/overview
            response = retry_on_rate_limit(
                self.client.messages.create, self.is_rate_limit_error
            )(
                model=self.config.model,
                system=system_prompt,
                messages=user_assistant_prompt,
                tools=self.define_tools(tools),
                tool_choice={
                    "type": "any",  # has to call a tool, but can be any
                },
                **kwargs,
            )
        except anthropic.BadRequestError as e:
            # Handle specific error for context length exceeded, otherwise just propagate the error
            if "prompt is too long" in e.message:
                raise ContextLengthExceededError
            raise

        # messages are either of type `text` or `tool_use`
        # https://docs.anthropic.com/en/docs/build-with-claude/tool-use/implement-tool-use#handling-results-from-client-tools

        tool_use_block = [r for r in response.content if r.type == "tool_use"]
        assert tool_use_block, "No tool use found in response."
        tool_use_block = tool_use_block[0]  # Select first tool called
        # select the first text message if there's any
        text_messages = [r.text for r in response.content if r.type == "text"]
        text_messages = text_messages[0] if text_messages else ""

        llm_response = LLMResponse(
            prompt=messages,
            response=text_messages,
            tool=self.parse_tool_call_response(tool_use_block),
            prompt_token_count=response.usage.input_tokens,
            response_token_count=response.usage.output_tokens,
        )

        return llm_response


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

        logger(
            f"Error calling {self.model_name}: {exception_full_name!r} {
                exception.message if hasattr(exception, 'message') else exception
            }"
        )

        return is_error

    def define_tools(self, tool_call_list: dict[str, EnvironmentTool]) -> list[dict]:
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
                "content": f"{history_info.step_observation.observation}",
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


class AzureOpenAILLM(OpenAILLM):

    @property
    def client(self):
        if getattr(self, "_client", None) is None:
            self._client = self._get_azure_oai_client()
        return self._client

    def _get_azure_oai_client(self):
        """
        Returns the Azure OpenAI client. This will use either an API key or Azure Identity.
        If the first attempt with Default and Managed Identity credentials fails,
        try again using only CliCredential (az login).

        Raises ValueError: If neither an API key nor a scope is provided in the configuration.
        """
        api_key = self.config.api_key
        scope = self.config.scope
        kwargs = {
            "azure_endpoint": self.config.endpoint,
            "api_version": self.config.api_version,
            "timeout": None,
        }
        if api_key not in [LLM_API_KEY_PLACEHOLDER, None]:  # api key
            kwargs["api_key"] = api_key
            aoai_client = AzureOpenAI(**kwargs)
        elif scope not in [LLM_SCOPE_PLACEHOLDER, None]:  # az login
            from azure.identity import (
                AzureCliCredential,
                ChainedTokenCredential,
                DefaultAzureCredential,
                ManagedIdentityCredential,
                get_bearer_token_provider,
            )

            credential = get_bearer_token_provider(
                ChainedTokenCredential(
                    DefaultAzureCredential(),
                    ManagedIdentityCredential(),
                    AzureCliCredential(),
                ),
                scope,
            )
            kwargs["azure_ad_token_provider"] = credential
            aoai_client = AzureOpenAI(**kwargs)
            try:
                aoai_client.models.list()  # test the connection
            except PermissionDeniedError:
                # if auth works but permission denied, try AzureCliCredential
                self.logger.warning(
                    "Permission denied for DefaultAzureCredential. Trying AzureCliCredential."
                )
                kwargs["azure_ad_token_provider"] = get_bearer_token_provider(
                    AzureCliCredential(), scope
                )
                aoai_client = AzureOpenAI(**kwargs)
        else:
            raise ValueError(
                "Invalid LLM configuration for AzureOpenAI. "
                "Please provide an `api_key or `scope` in the configuration."
            )
        return aoai_client


class Human(LLM):
    def __init__(
        self, model_name=None, logger: DebugGymLogger | None = None, max_retries=10
    ):
        self.model_name = model_name or "human"
        self.logger = logger or DebugGymLogger("debug-gym")
        self.context_length = None
        self.reasoning_end_token = None
        self._history = None
        self.max_retries = max_retries
        if prompt_toolkit_available:
            self._history = InMemoryHistory()

    def tokenize(self, text: str) -> list[str]:
        """Tokenizes a text by splitting it by spaces."""
        return text.split()

    def count_tokens(self, text: str) -> int:
        return len(self.tokenize(text))

    def define_tools(self, tool_call_list: dict[str, EnvironmentTool]) -> list[dict]:
        available_commands = []
        for tool in tool_call_list:
            random_id = "".join(map(str, np.random.randint(0, 10, size=6)))
            tool_id = f"{tool.name}-{random_id}"
            template = {
                "id": tool_id,
                "name": tool.name,
                "arguments": {k: "" for k in tool.arguments.keys()},
            }
            available_commands.append(template)
        return available_commands

    def parse_tool_call_response(self, response, all_tools) -> ToolCall:
        """Parse user input and return a ToolCall object.
        Validate the input against the available tools."""
        if response is None:
            raise ValueError("Tool call cannot be None")

        if not all_tools:
            raise ValueError("No tools provided. At least one tool must be available.")

        try:
            tool_call = ToolCall(**json.loads(response))
            for t in all_tools:
                if (
                    tool_call.id == t["id"]
                    and tool_call.name == t["name"]
                    and all([k in t["arguments"] for k in tool_call.arguments])
                ):
                    return tool_call
        except Exception:
            pass

        self.logger.error(
            "Invalid action format or command not available, please try again."
        )

        # Raise exception for parsing failures
        raise ValueError("Failed to parse valid tool call from input")

    def format_tool_call_history(
        self, history_info: EnvInfo, response: LLMResponse
    ) -> list[dict]:
        """Anthropic like format for tool call history"""
        _messages = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": response[0].tool.id,
                        "name": response[0].tool.name,
                        "input": response[0].tool.arguments,
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": history_info.action.id,
                        "content": f"{history_info.step_observation.observation}",
                    }
                ],
            },
        ]
        return _messages

    def generate(self, messages, tools, **kwargs) -> LLMResponse:
        # Human overrides the entire __call__ method, so generate is never called
        pass

    def __call__(self, messages, tools, *args, **kwargs) -> LLMResponse:
        print_messages(messages, self.logger)
        all_tools = self.define_tools(tools)
        available_commands = [json.dumps(t) for t in all_tools]
        tool_call = None
        retry_count = 0
        action = ""

        while tool_call is None and retry_count < self.max_retries:
            if prompt_toolkit_available:
                actions_completer = WordCompleter(
                    available_commands, ignore_case=False, sentence=True
                )
                action = prompt(
                    "\n> ",
                    completer=actions_completer,
                    history=self._history,
                    enable_history_search=True,
                )
            else:
                self.logger.info(
                    "\n".join(["Available commands:"] + available_commands)
                )
                action = input("> ")

            try:
                tool_call = self.parse_tool_call_response(action, all_tools)
            except ValueError as e:
                self.logger.error(f"Error parsing tool call: {e}")

            retry_count += 1

        if tool_call is None:
            error_message = (
                f"Maximum retries ({self.max_retries}) reached without valid input."
            )
            self.logger.error(
                f"Maximum retries ({self.max_retries}) reached without a valid tool call."
            )
            raise ValueError(error_message)

        return LLMResponse(
            prompt=messages,
            response=action,
            tool=tool_call,
            prompt_token_count=self.count_tokens(json.dumps(messages)),
            response_token_count=self.count_tokens(action),
        )
