import logging
import os
import re
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import tiktoken
import yaml
from openai import NOT_GIVEN, AzureOpenAI, OpenAI
from tenacity import (
    retry,
    retry_if_exception,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)
from termcolor import colored
from transformers import AutoTokenizer

from debug_gym.agents.utils import merge_messages, print_messages
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
    token_usage: TokenUsage | None = None

    def __init__(
        self,
        prompt: list[dict] | str,
        response: str,
        prompt_token_count: int = None,
        response_token_count: int = None,
        token_usage: TokenUsage = None,
    ):
        self.prompt = prompt
        self.response = response
        if prompt_token_count is not None and response_token_count is not None:
            self.token_usage = TokenUsage(prompt_token_count, response_token_count)
        else:
            self.token_usage = token_usage


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
    def generate(self, messages, **kwargs) -> str:
        """Generate a response given some messages and return it as a string."""
        pass

    @abstractmethod
    def tokenize(self, text: str) -> list[str]:
        """Abstract method to tokenize a text."""
        pass

    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text."""
        return len(self.tokenize(text))

    def count_messages_tokens(self, messages: list[dict]) -> int:
        """Count the number of tokens in a list of messages."""
        return sum(self.count_tokens(msg["content"]) for msg in messages)

    def __call__(self, messages, *args, **kwargs) -> LLMResponse:
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

        # merge consecutive messages with same role
        messages = merge_messages(messages)

        messages_length = self.count_messages_tokens(messages)
        self.logger.debug(f"Prompt size is {messages_length:,} tokens.")

        if messages_length > self.context_length:
            self.logger.info(
                f"Prompt is too long. {self.model_name} only allows for {self.context_length:,} tokens."
            )
            messages = trim_prompt_messages(
                messages, self.context_length, self.count_tokens
            )
            messages_length = self.count_messages_tokens(messages)
            self.logger.info(f"Prompt truncated to {messages_length:,} tokens.")

        print_messages(messages, self.logger)

        response = self.generate(messages, **kwargs)

        if response is None:
            response = ""
        response = response.strip()

        self.logger.info(colored(response, "green"))

        llm_response = LLMResponse(
            prompt=messages,
            response=response,
            prompt_token_count=self.count_messages_tokens(messages),
            response_token_count=self.count_tokens(response),
        )
        return llm_response


class AnthropicLLM(LLM):

    @property
    def client(self):
        if getattr(self, "_client", None) is None:
            from anthropic import Anthropic

            if self.config.api_key in [LLM_API_KEY_PLACEHOLDER, None]:
                raise ValueError(
                    f"API key is required for Anthropic. Please add it to the config."
                )
            self._client = Anthropic(api_key=self.config.api_key)
        return self._client

    def tokenize(self, text: str) -> list[str]:
        raise NotImplementedError("Tokenization is not supported by Anthropic.")

    def count_tokens(self, text: str) -> list[str]:
        messages = [{"role": "user", "content": [{"type": "text", "text": text}]}]
        try:
            response = self.client.beta.messages.count_tokens(
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

    def generate(self, messages, **kwargs):
        system_prompt = " "  # weird exceptions sometimes if empty
        user_assistant_prompt = []
        for message in messages:
            if message["content"] == "":
                continue
            if message["role"] == "system":
                system_prompt = message["content"]
            elif message["role"] in ["user", "assistant"]:
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
        # if thinking is enabled, the first message is the thought,
        # the last messages `content[-1]` is the response in any mode
        response = (
            retry_on_rate_limit(self.client.messages.create, self.is_rate_limit_error)(
                model=self.config.model,
                system=system_prompt,
                messages=user_assistant_prompt,
                **kwargs,
            )
            .content[-1]
            .text
        )
        response = response.strip()
        # only keep the content between the two ```.
        p = re.compile(r"```(.*?)```", re.DOTALL)
        if p.search(response) is not None:
            # ```...```
            response = p.search(response).group(0)
        else:
            response = ""
        return response


class OpenAILLM(LLM):

    @property
    def client(self):
        if getattr(self, "_client", None) is None:
            if self.config.api_key in [
                LLM_API_KEY_PLACEHOLDER,
                None,
            ] or self.config.endpoint in [LLM_ENDPOINT_PLACEHOLDER, None]:
                raise ValueError(
                    f"OpenAI API key and endpoint are required. Please add them to the config. "
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

    def generate(self, messages, **kwargs):
        # set max tokens if not provided
        kwargs["max_tokens"] = kwargs.get("max_tokens", NOT_GIVEN)
        response = retry_on_rate_limit(
            self.client.chat.completions.create, self.is_rate_limit_error
        )(
            model=self.config.model,
            messages=messages,
            **kwargs,
        )
        return response.choices[0].message.content


class AzureOpenAILLM(OpenAILLM):

    @property
    def client(self):
        if getattr(self, "_client", None) is None:
            kwargs = self._get_azure_oai_kwargs()
            self._client = AzureOpenAI(**kwargs)
        return self._client

    def _get_azure_oai_kwargs(self):
        """
        Returns a dictionary of keyword arguments required for connecting to Azure OpenAI.
        This will either use an API key or AzureCliCredential (az login).

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
        elif scope not in [LLM_SCOPE_PLACEHOLDER, None]:  # az login
            from azure.identity import (
                AzureCliCredential,
                ChainedTokenCredential,
                ManagedIdentityCredential,
                get_bearer_token_provider,
            )

            credential = get_bearer_token_provider(
                ChainedTokenCredential(
                    ManagedIdentityCredential(),
                    AzureCliCredential(),
                ),
                scope,
            )
            kwargs["azure_ad_token_provider"] = credential
        else:
            raise ValueError(
                "Invalid LLM configuration for AzureOpenAI. "
                "Please provide an `api_key or `scope` in the configuration."
            )
        return kwargs


class Human(LLM):
    def __init__(self, model_name=None, logger: DebugGymLogger | None = None):
        self.model_name = model_name or "human"
        self.logger = logger or DebugGymLogger("debug-gym")
        self.context_length = None
        self.reasoning_end_token = None
        self._history = None
        if prompt_toolkit_available:
            self._history = InMemoryHistory()

    def tokenize(self, text: str) -> list[str]:
        """Tokenizes a text by splitting it by spaces."""
        return text.split()

    def count_tokens(self, text: str) -> int:
        return len(self.tokenize(text))

    def generate(self, messages, **kwargs):
        # Human overrides the entire __call__ method, so generate is never called
        pass

    def __call__(self, messages, info, *args, **kwargs) -> LLMResponse:
        print_messages(messages, self.logger)
        available_commands = [t["template"] for t in info.tools.values()]
        if prompt_toolkit_available:
            actions_completer = WordCompleter(
                available_commands, ignore_case=True, sentence=True
            )
            action = prompt(
                "\n> ",
                completer=actions_completer,
                history=self._history,
                enable_history_search=True,
            )
        else:
            self.logger.info("\n".join(["Available commands:"] + available_commands))
            action = input("> ")

        prompt_messages = "\n".join([msg["content"] for msg in messages])

        return LLMResponse(
            prompt=prompt_messages,
            response=action,
            prompt_token_count=len(prompt_messages),
            response_token_count=len(action),
        )
