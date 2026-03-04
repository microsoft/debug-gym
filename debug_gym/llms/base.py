import logging
import multiprocessing
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import yaml
from tenacity import (
    retry,
    retry_if_exception,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from debug_gym.gym.tools.tool import EnvironmentTool, ToolCall
from debug_gym.llms.constants import DEFAULT_LLM_CONFIG
from debug_gym.llms.utils import print_messages, trim_prompt_messages
from debug_gym.logger import DebugGymLogger

# Set logging level down to WARNING for endpoint queries.
logging.getLogger("httpx").setLevel(logging.WARNING)

# Module-level counter for round-robin endpoint selection, shared across
# all LLMConfig instances and safe across forked worker processes.
# Initialized with a random offset so independent Python processes
# (e.g., separate `python run.py` invocations) start at different endpoints.
_endpoint_counter = multiprocessing.Value("i", os.getpid())


def retry_on_exception(
    func, exception_filter_func, multiplier=1, max_wait=40, max_attempts=100
):
    """Executes a function with retry logic for certain exceptions. Never retries on KeyboardInterrupt.
    Args:
        func: The function to execute with retries
        exception_filter_func: Function that checks if an exception needs to be retried
        *args, **kwargs: Arguments to pass to the function

    Returns:
        The result of the function call
    """
    retry_function = retry(
        retry=(
            retry_if_not_exception_type(KeyboardInterrupt)
            & retry_if_exception(exception_filter_func)
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
    endpoints: Optional[List[str]] = None
    tokenizer: Optional[str] = None
    apply_chat_template: Optional[bool] = False
    enable_thinking: Optional[bool] = False
    thinking_param_name: Optional[str] = "enable_thinking"
    reasoning_end_token: Optional[str] = None
    system_prompt_support: bool = True
    ignore_kwargs: List[str] = None
    tags: List[str] = None
    # Azure OpenAI specific fields
    api_version: Optional[str] = None
    scope: Optional[str] = None
    # Custom parameters to pass to generate
    generate_kwargs: dict = None
    # Additional kwargs for tokenizer construction (e.g., trust_remote_code)
    tokenizer_kwargs: dict | None = None

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
        if self.tokenizer_kwargs is None:
            self.tokenizer_kwargs = {}
        # Reconcile endpoint/endpoints: build a unified endpoints list
        if self.endpoints and self.endpoint:
            # Both provided — prefer endpoints list, ignore singular
            pass
        elif self.endpoint and not self.endpoints:
            self.endpoints = [self.endpoint]
        elif not self.endpoints:
            self.endpoints = []

    def select_endpoint(self) -> Optional[str]:
        """Select the next endpoint using round-robin.

        Uses a process-safe counter so that workers spawned via
        ProcessPoolExecutor each get a different endpoint.
        Returns None if no endpoints are configured.
        """
        if not self.endpoints:
            return self.endpoint  # original behavior (None)
        with _endpoint_counter.get_lock():
            idx = _endpoint_counter.value % len(self.endpoints)
            _endpoint_counter.value += 1
        return self.endpoints[idx]


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
            msg = (
                f"Cannot find llm config file: {config_file_path}. "
                "Use `python -m debug_gym.llms.configure` to create one and edit it."
            )
            raise FileNotFoundError(msg)

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
    response: str | None
    reasoning_response: str | None
    tool: ToolCall
    token_usage: TokenUsage | None = None
    # Raw thinking blocks from the API response (for Anthropic extended thinking)
    # These preserve signatures needed when passing back to the API during tool use
    thinking_blocks: list | None = None

    def __init__(
        self,
        prompt: list[dict] | str,
        response: str = None,
        reasoning_response: str = None,
        tool: ToolCall = None,
        prompt_token_count: int = None,
        response_token_count: int = None,
        token_usage: TokenUsage = None,
        thinking_blocks: list = None,
    ):
        self.prompt = prompt
        self.response = response
        self.reasoning_response = reasoning_response
        self.tool = tool
        if prompt_token_count is not None and response_token_count is not None:
            self.token_usage = TokenUsage(prompt_token_count, response_token_count)
        else:
            self.token_usage = token_usage
        self.thinking_blocks = thinking_blocks


class ContextLengthExceededError(Exception):
    """Exception raised when the context length of an LLM request is exceeded."""

    pass


class LLM(ABC):

    def __init__(
        self,
        model_name: str,
        llm_config: LLMConfig,
        logger: DebugGymLogger | None = None,
        runtime_generate_kwargs: dict | None = None,
    ):
        self.model_name = model_name
        self.logger = logger or DebugGymLogger("debug-gym")
        self.config = llm_config
        # Select endpoint via round-robin when multiple endpoints are configured
        selected = self.config.select_endpoint()
        if selected:
            self.config.endpoint = selected
            self.logger.debug(f"Selected endpoint: {selected}")
        self.tokenizer_name = self.config.tokenizer
        self.context_length = self.config.context_limit * 1000
        self.apply_chat_template = self.config.apply_chat_template
        self.enable_thinking = self.config.enable_thinking
        self.reasoning_end_token = self.config.reasoning_end_token
        # Runtime generation kwargs from experiment config (temperature, max_tokens, etc.)
        self.runtime_generate_kwargs = runtime_generate_kwargs or {}

        self.logger.debug(f"Using {self.model_name} with max context length of {
                self.context_length:,} tokens.")

    @classmethod
    def instantiate(
        cls,
        name: str | None = None,
        llm_config_file_path: str | None = None,
        logger: DebugGymLogger | None = None,
        **runtime_generate_kwargs,
    ) -> "LLM":
        """Creates an instance of the appropriate LLM class based on the configuration.

        Args:
            name: Name of the LLM model (e.g., "gpt-4o", "claude-3.7").
            llm_config_file_path: Optional path to the LLM configuration file.
            logger: Optional DebugGymLogger for logging.
            **runtime_generate_kwargs: Additional generation kwargs to pass to the LLM.
                e.g. temperature, max_tokens, tool_choice ("auto", "required", "none")

        Returns:
            An instance of the appropriate LLM class.
        """

        logger = logger or DebugGymLogger("debug-gym")

        if not name:
            return None

        if name == "human":
            from debug_gym.llms import Human

            return Human(name, logger=logger)

        llm_config = LLMConfigRegistry.from_file(llm_config_file_path)[name]

        tags = llm_config.tags

        if "copilot openai" in tags:
            from debug_gym.llms.copilot import CopilotOpenAILLM

            klass = CopilotOpenAILLM

        elif "copilot claude" in tags:
            from debug_gym.llms.copilot import CopilotClaudeLLM

            klass = CopilotClaudeLLM

        elif "azure openai" in tags:
            from debug_gym.llms import AzureOpenAILLM

            klass = AzureOpenAILLM

        elif "vllm" in tags:
            from debug_gym.llms import HuggingFaceLLM

            klass = HuggingFaceLLM

        elif "anthropic" in tags:
            from debug_gym.llms import AnthropicLLM

            klass = AnthropicLLM

        else:
            from debug_gym.llms import OpenAILLM

            klass = OpenAILLM

        llm = klass(
            name,
            llm_config=llm_config,
            logger=logger,
            runtime_generate_kwargs=runtime_generate_kwargs,
        )
        return llm

    @abstractmethod
    def generate(self, messages, tools, **kwargs) -> LLMResponse:
        """Generate a response given some messages and return it as an LLMResponse object.
        Raises ContextLengthExceededError if the context length is exceeded.
        The method should be overridden by subclasses."""
        pass

    @abstractmethod
    def tokenize(self, messages: list[dict]) -> list[list[str]]:
        """Abstract method to tokenize messages.

        Args:
            messages: List of message dicts

        Returns:
            List of token lists, one per message
        """
        pass

    def count_tokens(self, messages: list[dict] | dict | str) -> int:
        """Count the total number of tokens across all messages.

        Args:
            messages: List of message dicts, a single message dict, or a string

        Returns:
            Total token count across all messages
        """
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        elif isinstance(messages, dict):
            messages = [messages]
        tokenized = self.tokenize(messages)
        return sum(len(tokens) for tokens in tokenized)

    def _estimate_tokens(self, content: str) -> list[str]:
        """Estimate token count using character-based heuristic.

        Uses ~4 characters per token as a conservative estimate for English/code.
        Returns placeholder tokens to maintain the expected return type.

        This is useful as a fallback when tokenization fails (e.g., stack overflow
        on very large inputs) or when a tokenizer is not available.
        """
        estimated_count = max(1, len(content) // 4)
        return ["_"] * estimated_count

    @abstractmethod
    def define_tools(self, tool_call_list: list[EnvironmentTool]) -> list[dict]:
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
    def convert_response_to_message(
        self,
        response: LLMResponse,
    ) -> dict:
        """Format the tool call history for different LLMs.
        The method should be overridden by subclasses.
        """
        raise NotImplementedError(
            "The convert_response_to_message method should be overridden by subclasses."
        )

    @abstractmethod
    def convert_observation_to_message(
        self,
        observation: str,
        action_tool_call_id: int = None,
        action_tool_call_name: str = None,
    ) -> dict:
        """Format the tool call history for different LLMs.
        The method should be overridden by subclasses.
        """
        raise NotImplementedError(
            "The convert_observation_to_message method should be overridden by subclasses."
        )

    def close(self):
        """Clean up resources (HTTP clients, etc.). Override in subclasses."""
        pass

    def __call__(self, messages, tools, *args, **kwargs) -> LLMResponse:
        """Prepares messages and kwargs, then call `generate` which
        should be implemented by subclasses. Returns an LLMResponse object
        with the prompt, response and token usage.

        Priority for generation kwargs (highest to lowest):
        1. kwargs passed directly to this call
        2. runtime_generate_kwargs from experiment config
        3. config.generate_kwargs from llm.yaml
        """
        # Add runtime generation kwargs from experiment config (higher priority)
        for key, value in self.runtime_generate_kwargs.items():
            if key not in kwargs:
                kwargs[key] = value

        # Add custom generation parameters from llm.yaml config (lowest priority)
        for key, value in self.config.generate_kwargs.items():
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

        def generate_with_trim_retry(messages, tools, **kwargs):
            """Generate a response, trimming messages if context length is exceeded.

            Strategy:
            1. Pre-trim if messages exceed context length
            2. Try to generate
            3. If API still complains (tokenizer mismatch), reduce limit by 5% and retry
            """
            if not messages:
                raise ValueError("No messages provided for generation.")

            max_retries = 3
            current_limit = self.context_length

            # Pre-trim if obviously too long
            if self.count_tokens(messages) > current_limit:
                self.logger.info(
                    f"Messages exceed context length ({current_limit:,} tokens). Trimming..."
                )
                messages = trim_prompt_messages(
                    messages, current_limit, self.count_tokens
                )

            for attempt in range(max_retries + 1):
                try:
                    return self.generate(messages, tools, **kwargs)
                except ContextLengthExceededError:
                    if attempt >= max_retries:
                        raise ContextLengthExceededError(
                            f"Unable to fit messages within context limit after {max_retries} retries. "
                            f"Final limit: {current_limit:,} tokens."
                        )

                    # Reduce limit by 5% and retry
                    current_limit = int(current_limit * 0.95)
                    self.logger.info(
                        f"API reported context exceeded. Retrying with reduced limit: {current_limit:,} tokens..."
                    )
                    messages = trim_prompt_messages(
                        messages, current_limit, self.count_tokens
                    )

        llm_response = generate_with_trim_retry(messages, tools, **kwargs)

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

        print_messages(messages, self.logger)
        response_parts = []
        if llm_response.reasoning_response:
            response_parts.append(
                f"LLM response - reasoning: {llm_response.reasoning_response}"
            )
        if llm_response.response:
            response_parts.append(f"LLM response - content: {llm_response.response}")
        if llm_response.tool:
            tool_name = (
                llm_response.tool.get("name")
                if isinstance(llm_response.tool, dict)
                else getattr(llm_response.tool, "name", None)
            )
            if tool_name != "empty_tool_response":
                response_parts.append(f"LLM response - tool call: {llm_response.tool}")
        if response_parts:
            self.logger.info("\n".join(response_parts))
        else:
            self.logger.info(
                "LLM response - empty (no content, reasoning, or tool call)"
            )
        return llm_response
