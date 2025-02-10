import json
import logging
import os
import sys
from dataclasses import dataclass

import tiktoken
from openai import AsyncAzureOpenAI, AsyncOpenAI, AzureOpenAI, OpenAI
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)
from termcolor import colored
from transformers import AutoTokenizer

from froggy.logger import FroggyLogger

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


def print_messages(messages: list[dict], logger: FroggyLogger):
    for m in messages:
        if m["role"] == "user":
            logger.info(colored(f"{m['content']}\n", "cyan"))
        elif m["role"] == "assistant":
            logger.info(colored(f"{m['content']}\n", "green"))
        elif m["role"] == "system":
            logger.info(colored(f"{m['content']}\n", "yellow"))
        else:
            raise ValueError(f"Unknown role: {m['content']}")


def merge_messages(messages):
    messages_out = [dict(messages[0])]
    for message in messages[1:]:
        if message["role"] == messages_out[-1]["role"]:
            messages_out[-1]["content"] += "\n\n" + message["content"]
        else:
            messages_out.append(dict(message))

    return messages_out


@dataclass
class TokenUsage:
    prompt: int
    response: int


@dataclass
class LLMResponse:
    prompt: list[dict] | str  # Either a string or a list of messages.
    response: str
    token_usage: TokenUsage | None = None


class TokenCounter:
    def __init__(self, model: str = "gpt-4o"):
        self.model = model
        try:
            self.tokenize = tiktoken.encoding_for_model(model).encode
        except KeyError:
            try:
                # Try to load from transformers.
                self.tokenize = AutoTokenizer.from_pretrained(model).tokenize
            except OSError:
                msg = (
                    f"Tokenizer not found for model {model},"
                    " make sure you have access to the model"
                    " (e.g., HuggingFace API key is correctly set)."
                )
                raise ValueError(msg)

    def __call__(self, *, messages=None, text=None) -> int:
        nb_tokens = 0
        if messages is not None:
            nb_tokens += sum(len(self.tokenize(msg["content"])) for msg in messages)

        if text is not None:
            nb_tokens += len(self.tokenize(text))

        return nb_tokens


class LLM:
    def __init__(self, model_name: str, logger: FroggyLogger | None = None):
        configs = load_llm_config()
        if model_name not in configs:
            raise ValueError(f"Model {model_name} not found in llm.cfg")

        self.model_name = model_name
        self.config = configs[model_name]
        self.logger = logger or FroggyLogger("froggy")
        self.token_counter = TokenCounter(self.config["tokenizer"])
        self.context_length = self.config["context_limit"] * 1000

        self.logger.debug(
            f"Using {self.model_name} with max context length of {
                self.context_length:,} tokens."
        )

        if "azure openai" in self.config.get("tags", []):
            kwargs = self._get_azure_oai_kwargs()
            self.client = AzureOpenAI(**kwargs)
        else:
            self.client = OpenAI(
                api_key=self.config["api_key"],
                base_url=self.config["endpoint"],
                timeout=None,
            )

        self.call_with_retry = retry(
            retry=retry_if_exception(self.is_rate_limit_error),
            wait=wait_random_exponential(multiplier=1, max=40),
            stop=stop_after_attempt(100),
        )

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
                    AzureCliCredential(),
                    ManagedIdentityCredential(),
                ),
                scope,
            )
            kwargs["azure_ad_token_provider"] = credential
        else:
            raise ValueError(
                "Invalid LLM configuration. Please provide an `api_key or `scope` in the configuration."
            )
        return kwargs

    def is_rate_limit_error(self, exception):
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
            # Add more as needed
        ]
        exception_full_name = (
            f"{exception.__class__.__module__}.{exception.__class__.__name__}"
        )
        if isinstance(exception, KeyboardInterrupt):
            return False

        self.logger.warning(f"Error calling {self.model_name}: {exception_full_name!r}")
        self.logger.debug(f"Exception: {exception.message}")
        return exception_full_name in rate_limit_errors

    def query_model(self, messages, **kwargs):
        kwargs["max_tokens"] = kwargs.get("max_tokens", self.config.get("max_tokens"))

        reponse = self.call_with_retry(self.client.chat.completions.create)(
            model=self.config["model"],
            messages=messages,
            **kwargs,
        )
        return reponse.choices[0].message.content

    def __call__(self, messages, *args, **kwargs) -> LLMResponse:
        from example_agent.utils import trim_prompt_messages

        if not self.config.get("system_prompt_support", True):
            # Replace system by user
            for i, m in enumerate(messages):
                if m["role"] == "system":
                    messages[i]["role"] = "user"

        # Merge consecutive messages with same role.
        messages = merge_messages(messages)
        messages = trim_prompt_messages(
            messages, self.context_length, self.token_counter
        )

        # Message is a list of dictionaries with role and content keys.
        # Color each role differently.
        print_messages(messages, self.logger)

        response = self.query_model(messages, **kwargs)
        response = response.strip()

        self.logger.debug(colored(response, "green"))

        token_usage = TokenUsage(
            prompt=self.token_counter(messages=messages),
            response=self.token_counter(text=response),
        )

        return LLMResponse(messages, response, token_usage)


class AsyncLLM(LLM):
    def __init__(self, model_name, logger: FroggyLogger | None = None):
        super().__init__(model_name, logger)

        if "azure openai" in self.config.get("tags", []):
            kwargs = self._get_azure_oai_kwargs()
            self.client = AsyncAzureOpenAI(**kwargs)
        else:
            self.client = AsyncOpenAI(
                api_key=self.config["api_key"],
                base_url=self.config["endpoint"],
                timeout=None,
            )

    async def query_model(self, messages, **kwargs):
        kwargs["max_tokens"] = kwargs.get("max_tokens", self.config.get("max_tokens"))

        response = await self.call_with_retry(self.client.chat.completions.create)(
            model=self.config["model"],
            messages=messages,
            **kwargs,
        )
        return response.choices[0].message.content

    async def __call__(self, messages, *args, **kwargs) -> LLMResponse:
        if not self.config.get("system_prompt_support", True):
            # Replace system by user
            for i, m in enumerate(messages):
                if m["role"] == "system":
                    messages[i]["role"] = "user"

        response = await self.query_model(messages, **kwargs)
        response = response.strip()

        token_usage = TokenUsage(
            prompt=self.token_counter(messages=messages),
            response=self.token_counter(text=response),
        )

        return LLMResponse(messages, response, token_usage)


class Human:
    def __init__(self, logger: FroggyLogger | None = None):
        self._history = None
        self.logger = logger or FroggyLogger("froggy")
        if prompt_toolkit_available:
            self._history = InMemoryHistory()

    def __call__(self, messages, info, *args, **kwargs) -> LLMResponse:
        # Color each role differently.
        print_messages(messages, self.logger)
        available_commands = [t["template"] for t in info.tools.values()]
        if prompt_toolkit_available:
            actions_completer = WordCompleter(
                available_commands, ignore_case=True, sentence=True
            )
            action = prompt(
                "> ",
                completer=actions_completer,
                history=self._history,
                enable_history_search=True,
            )
        else:
            self.logger.info("\n".join(["Available commands:"] + available_commands))
            action = input("> ")

        prompt_messages = "\n".join([msg["content"] for msg in messages])
        token_usage = TokenUsage(
            prompt=len(prompt_messages),
            response=len(action),
        )

        return LLMResponse(prompt_messages, action, token_usage)


def instantiate_llm(
    config: dict, logger: FroggyLogger | None = None, use_async: bool = False
):
    llm_config = load_llm_config()
    available_models = list(llm_config.keys()) + ["human"]
    if config["llm_name"] not in available_models:
        raise ValueError(
            f"Model {config['llm_name']} is not available, "
            "please make sure the LLM config file is correctly set."
        )
    if config["llm_name"] == "human":
        llm = Human(logger=logger)
    else:
        if use_async:
            llm = AsyncLLM(config["llm_name"], logger=logger)
        else:
            llm = LLM(config["llm_name"], logger=logger)
    return llm
