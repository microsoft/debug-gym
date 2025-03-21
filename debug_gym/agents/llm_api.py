import json
import logging
import os
import re
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass

import tiktoken
from openai import NOT_GIVEN, AsyncAzureOpenAI, AsyncOpenAI, AzureOpenAI, OpenAI
from tenacity import (
    retry,
    retry_if_exception,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)
from termcolor import colored
from transformers import AutoTokenizer

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


def load_llm_config(config_file_path: str | None = None):
    if config_file_path is None:
        config_file_path = os.environ.get("LLM_CONFIG_FILE", "llm.cfg")
    try:
        llm_config = json.load(open(config_file_path))
    except FileNotFoundError:
        raise FileNotFoundError(f"Cannot find llm config file: {config_file_path}")
    return llm_config


def print_messages(messages: list[dict], logger: DebugGymLogger):
    """Print messages coloring each role differently.
    Messages is a list of dictionaries with role and content keys."""
    for m in messages:
        if m["role"] == "user":
            logger.info(colored(f"{m['content']}\n", "cyan"))
        elif m["role"] == "assistant":
            logger.info(colored(f"{m['content']}\n", "green"))
        elif m["role"] == "system":
            logger.info(colored(f"{m['content']}\n", "yellow"))
        else:
            raise ValueError(f"Unknown role: {m['content']}")


def merge_messages(messages: list[dict]) -> list[dict]:
    """Merge consecutive messages with same role into one message."""
    messages_out = []
    to_merge = []

    def merge():
        content = "\n\n".join(m["content"] for m in to_merge if m["content"])
        if content:
            messages_out.append({"role": current_role, "content": content})

    current_role = None
    for message in messages:
        if current_role == message["role"]:
            to_merge.append(message)
        else:
            merge()  # merge the previous messages
            current_role = message["role"]
            to_merge = [message]
    merge()  # merge the last messages
    return messages_out


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
    retry_decorator = retry(
        retry=(
            retry_if_not_exception_type(KeyboardInterrupt)
            & retry_if_exception(is_rate_limit_error_func)
        ),
        wait=wait_random_exponential(multiplier=multiplier, max=max_wait),
        stop=stop_after_attempt(max_attempts),
    )
    return retry_decorator(func)


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

    def __init__(self, model_name: str, logger: DebugGymLogger | None = None):
        self.logger = logger or DebugGymLogger("debug-gym")
        configs = load_llm_config()
        if model_name not in configs:
            raise ValueError(f"Model {model_name} not found in llm.cfg")
        self.model_name = model_name
        self.config = configs[model_name]
        self.tokenizer_name = self.config.get("tokenizer", self.config["model"])
        self.context_length = self.config["context_limit"] * 1000
        self.reasoning_end_token = self.config.get("reasoning_end_token", None)

        self.logger.debug(
            f"Using {self.model_name} with max context length of {
                self.context_length:,} tokens."
        )

    @abstractmethod
    def generate(self, messages, **kwargs) -> str:
        """Generate a response given some messages and return it as a string."""
        pass

    @abstractmethod
    def tokenize(self, text: str) -> list[str]:
        """Abstract method to tokenize a text."""
        pass

    def token_counter(self, text: str) -> int:
        """Count the number of tokens in a text."""
        return self.count_tokens(text)

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

        # set max tokens if not provided
        kwargs["max_tokens"] = kwargs.get(
            "max_tokens", self.config.get("max_tokens", NOT_GIVEN)  # OpenAI
        )

        # replace system prompt by user prompt if not supported
        if not self.config.get("system_prompt_support", True):
            self.logger.debug(
                "System prompt is not supported by the model, it will be replaced by user prompt."
            )
            for i, m in enumerate(messages):
                if m["role"] == "system":
                    messages[i]["role"] = "user"

        # ignore specific kwargs that are not supported by the model
        if "ignore_kwargs" in self.config:
            self.logger.debug(
                f"LLM arguments {", ".join(self.config['ignore_kwargs'])} "
                "are not supported by the model, they will be ignored."
            )
            for kw in self.config["ignore_kwargs"]:
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

            self._client = Anthropic(api_key=self.config["api_key"])
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

        if "thinking" in self.config.get("tags", []):
            kwargs["max_tokens"] = 20000
            kwargs["temperature"] = 1.0

            response = (
                retry_on_rate_limit(
                    self.client.messages.create, self.is_rate_limit_error
                )(
                    model=self.config["model"],
                    thinking={"type": "enabled", "budget_tokens": 16000},
                    system=system_prompt,
                    messages=user_assistant_prompt,
                    **kwargs,
                )
                .content[1]
                .text
            )
        else:
            kwargs["max_tokens"] = 8192
            response = (
                retry_on_rate_limit(
                    self.client.messages.create, self.is_rate_limit_error
                )(
                    model=self.config["model"],
                    system=system_prompt,
                    messages=user_assistant_prompt,
                    **kwargs,
                )
                .content[0]
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
            self._client = OpenAI(
                api_key=self.config["api_key"],
                base_url=self.config["endpoint"],
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
        if isinstance(exception, KeyboardInterrupt):
            return False

        # Look for error code that shouldn't be retry.
        match exception_full_name:
            case "openai.APIStatusError":
                if "'status': 429" in exception.message:  # Rate Limit Exceeded
                    self.logger.debug(
                        f"Error calling {self.model_name}: {exception_full_name!r} {exception.message}"
                    )
                    return True
                elif "'status': 504" in exception.message:  # Gateway Timeout
                    self.logger.debug(
                        f"Error calling {self.model_name}: {exception_full_name!r} {exception.message}"
                    )
                    return True
                elif (
                    "'status': 413" in exception.message
                    and "A previous prompt was too large." in exception.message
                ):  # A previous prompt was too large.
                    self.logger.debug(
                        f"Error calling {self.model_name}: {exception_full_name!r} {exception.message}"
                    )
                    return True
                else:
                    self.logger.warning(
                        f"Error calling {self.model_name}: {exception_full_name!r} {exception.message}"
                    )
                    return False
            case _:
                self.logger.debug(
                    f"Error calling {self.model_name}: {exception_full_name!r} {
                        exception.message if hasattr(exception, 'message') else exception
                    }"
                )

        return exception_full_name in rate_limit_errors

    def generate(self, messages, **kwargs):
        response = retry_on_rate_limit(
            self.client.chat.completions.create, self.is_rate_limit_error
        )(
            model=self.config["model"],
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
        api_key = self.config.get("api_key")
        scope = self.config.get("scope")
        kwargs = {
            "azure_endpoint": self.config["endpoint"],
            "api_version": self.config["api_version"],
            "timeout": None,
        }
        if api_key:  # api key
            kwargs["api_key"] = api_key
        elif scope:  # az login
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
                "Invalid LLM configuration. Please provide an `api_key or `scope` in the configuration."
            )
        return kwargs


class Human:
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

    def token_counter(self, text: str) -> int:
        """Count the number of tokens in a text."""
        return self.count_tokens(text)

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


def instantiate_llm(config: dict, logger: DebugGymLogger | None = None) -> LLM:
    llm_name = config["llm_name"]
    if llm_name == "human":
        return Human(llm_name, logger=logger)

    llm_config = load_llm_config()
    try:
        tags = llm_config[llm_name].get("tags", [])
    except KeyError:
        raise ValueError(
            f"Model {llm_name} not found in llm.cfg, please "
            "make sure the LLM config file is correctly set."
        )
    if "azure openai" in tags:
        klass = AzureOpenAILLM
    elif "anthropic" in tags:
        klass = AnthropicLLM
    else:
        klass = OpenAILLM
    llm = klass(llm_name, logger=logger)
    return llm
