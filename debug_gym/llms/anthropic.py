from debug_gym.gym.tools.tool import EnvironmentTool, ToolCall
from debug_gym.gym.utils import filter_non_utf8
from debug_gym.llms.base import (
    LLM,
    ContextLengthExceededError,
    LLMResponse,
    retry_on_exception,
)
from debug_gym.llms.constants import LLM_API_KEY_PLACEHOLDER


class AnthropicLLM(LLM):

    context_length_error_code = []
    context_length_error_message_keywords = [
        "prompt is too long",
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
            from anthropic import Anthropic

            if self.config.api_key in [LLM_API_KEY_PLACEHOLDER, None]:
                raise ValueError(
                    "API key is required for Anthropic. Please add it to the config."
                )
            self._client = Anthropic(api_key=self.config.api_key)
        return self._client

    def tokenize(self, messages: list[dict]) -> list[list[str]]:
        """Tokenization is not directly supported by Anthropic.
        This method returns empty token lists as a placeholder."""
        raise NotImplementedError("Direct tokenization is not supported by Anthropic.")

    def count_tokens(self, messages: list[dict] | str) -> int:
        """Count the number of tokens in a text using the Anthropic API."""
        if isinstance(messages, str):
            messages = [
                {"role": "user", "content": [{"type": "text", "text": messages}]}
            ]

        try:
            response = self.client.messages.count_tokens(
                model=self.tokenizer_name, messages=messages
            )
            return response.input_tokens
        except Exception as e:
            self.logger.warning(
                f"Error calling Claude token count API: {e!r}. "
                f"The messages were: {messages}. Will return 0 tokens."
            )
        return 0

    def need_to_be_retried(self, exception) -> bool:
        # Errors that are worth retrying (transient issues)
        # See: https://docs.anthropic.com/en/api/errors
        # - 429: rate_limit_error - Rate limit exceeded
        # - 500: api_error - Internal server error
        # - 529: overloaded_error - API temporarily overloaded
        # - Connection/timeout errors - Network issues
        _retryable_errors = [
            # Rate limit and overload errors
            "anthropic.RateLimitError",  # 429
            "anthropic.OverloadedError",  # 529
            "anthropic._exceptions.OverloadedError",  # 529 (internal path)
            # Server errors
            "anthropic.InternalServerError",  # 500
            # Connection and timeout errors
            "anthropic.APIConnectionError",
            "anthropic.APITimeoutError",
            # Generic API errors (will filter by status code below)
            "anthropic.APIStatusError",
        ]

        exception_full_name = (
            f"{exception.__class__.__module__}.{exception.__class__.__name__}"
        )
        need_to_retry = exception_full_name in _retryable_errors
        logger = self.logger.debug

        # For generic APIStatusError, only retry on specific status codes
        if exception_full_name == "anthropic.APIStatusError":
            status_code = getattr(exception, "status_code", None)
            # Only retry on server errors (5xx) or rate limits (429)
            if status_code not in [429, 500, 502, 503, 504, 529]:
                need_to_retry = False
                logger = self.logger.warning

        # Never retry context length errors - these require reducing input
        if self.is_context_length_error(exception):
            need_to_retry = False

        logger(
            f"Error calling {self.model_name}: {exception_full_name!r} "
            f"{exception.message if hasattr(exception, 'message') else exception}"
        )
        return need_to_retry

    def define_tools(self, tool_call_list: list[EnvironmentTool]) -> list[dict]:
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
        if response is None:
            return ToolCall(
                id="empty_tool_response",
                name="empty_tool_response",
                arguments={},
            )
        return ToolCall(
            id=response.id,
            name=response.name,
            arguments=response.input,
        )

    def convert_response_to_message(self, response: LLMResponse) -> dict:
        """Convert an LLMResponse to an assistant message for the API.

        For thinking blocks, we must preserve the complete block including the
        signature field for verification when passing back to the API during
        tool use continuation.
        """
        content = []

        # Add thinking blocks - preserve complete blocks with signatures
        # These are stored as raw API response objects in thinking_blocks
        if response.thinking_blocks:
            for block in response.thinking_blocks:
                if block.type == "thinking":
                    content.append(
                        {
                            "type": "thinking",
                            "thinking": block.thinking,
                            "signature": block.signature,
                        }
                    )
                elif block.type == "redacted_thinking":
                    # Redacted thinking blocks have encrypted data
                    content.append(
                        {
                            "type": "redacted_thinking",
                            "data": block.data,
                        }
                    )

        if response.response:
            content.append(
                {
                    "type": "text",
                    "text": filter_non_utf8(response.response),
                }
            )
        if response.action:
            content.append(
                {
                    "type": "tool_use",
                    "id": response.action.id,
                    "name": response.action.name,
                    "input": response.action.arguments,
                }
            )

        message = {
            "role": "assistant",
            "content": content,
        }
        return message

    def convert_observation_to_message(
        self,
        observation: str,
        action_tool_call_id=None,
        action_tool_call_name: str = None,
    ) -> dict:
        if action_tool_call_id is None:
            # This is the initial state, no action taken yet
            return {
                "role": "user",
                "content": filter_non_utf8(observation),
            }
        else:
            # This is a step with an action taken
            return {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": action_tool_call_id,  # 'toolu_01SdR84CsnTKRpdH4zwFjvGj'
                        "content": filter_non_utf8(
                            observation
                        ),  # 'Viewing `hangman_test.py`. The file is read-only, it is not editable.'
                    }
                ],
            }

    def generate(self, messages, tools, **kwargs) -> LLMResponse:
        import anthropic

        system_prompt = None
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
            # Build API call parameters
            # https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/overview
            api_params = {
                "model": self.config.model,
                "messages": user_assistant_prompt,
                **kwargs,
            }

            # Only add system prompt if we have one with non-whitespace content
            if system_prompt and system_prompt.strip():
                api_params["system"] = system_prompt

            # Only add tools and tool_choice if tools are provided
            if tools:
                api_params["tools"] = self.define_tools(tools)
                # Only force tool choice if not using extended thinking
                # Extended thinking conflicts with tool_choice="any"
                if "thinking" not in kwargs:
                    api_params["tool_choice"] = {
                        "type": "any",  # has to call a tool, but can be any
                    }

            response = retry_on_exception(
                self.client.messages.create, self.need_to_be_retried
            )(**api_params)
        except anthropic.BadRequestError as e:
            # Handle specific error for context length exceeded, otherwise just propagate the error
            if self.is_context_length_error(e):
                raise ContextLengthExceededError
            raise

        # messages are either of type `text` or `tool_use`
        # https://docs.anthropic.com/en/docs/build-with-claude/tool-use/implement-tool-use#handling-results-from-client-tools

        # tool use block is a list of tool calls, we select the first one
        tool_use_block = [r for r in response.content if r.type == "tool_use"]
        tool_use_block = (
            tool_use_block[0] if tool_use_block else None
        )  # Select first tool called
        # select the first text message if there's any
        text_messages = [r.text for r in response.content if r.type == "text"]
        text_messages = text_messages[0] if text_messages else None
        # thinking - use .thinking attribute (not .text) for thinking blocks
        # Also preserve the full blocks for signature verification when passing back to API
        thinking_blocks = [r for r in response.content if r.type == "thinking"]
        # Include redacted_thinking blocks as well (safety-flagged content)
        redacted_thinking_blocks = [
            r for r in response.content if r.type == "redacted_thinking"
        ]
        # Extract text content for reasoning_response (use .thinking, not .text)
        thinking_messages = thinking_blocks[0].thinking if thinking_blocks else None

        # Combine thinking and redacted_thinking blocks for signature preservation
        # These need to be passed back to the API for tool use continuation
        all_thinking_blocks = thinking_blocks + redacted_thinking_blocks

        llm_response = LLMResponse(
            prompt=messages,
            response=text_messages,
            reasoning_response=thinking_messages,
            tool=self.parse_tool_call_response(tool_use_block),
            prompt_token_count=response.usage.input_tokens,
            response_token_count=response.usage.output_tokens,
            thinking_blocks=all_thinking_blocks if all_thinking_blocks else None,
        )

        return llm_response
