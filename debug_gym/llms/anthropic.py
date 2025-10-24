from debug_gym.gym.envs.env import EnvInfo
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
        _errors = [
            "anthropic.RateLimitError",
            "anthropic.OverloadedError",
            "anthropic._exceptions.OverloadedError",
            "anthropic.InternalServerError",
        ]
        exception_full_name = (
            f"{exception.__class__.__module__}.{exception.__class__.__name__}"
        )
        need_to_retry = exception_full_name in _errors

        if self.is_context_length_error(exception):
            need_to_retry = False

        self.logger.debug(
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

    def format_tool_call_history(
        self, history_info: EnvInfo, response: list[LLMResponse]
    ) -> list[dict]:
        _messages = []
        if isinstance(response, list) and len(response) > 0:
            content = []
            if response[0].reasoning_response:
                content.append(
                    {
                        "type": "thinking",
                        "text": filter_non_utf8(response[0].reasoning_response),
                    }
                )
            if response[0].response:
                content.append(
                    {
                        "type": "text",
                        "text": filter_non_utf8(response[0].response),
                    }
                )
            if response[0].tool:
                content.append(
                    {
                        "type": "tool_use",
                        "id": response[0].tool.id,
                        "name": response[0].tool.name,
                        "input": response[0].tool.arguments,
                    }
                )
            _messages.append(
                {
                    "role": "assistant",
                    "content": content,
                }
            )
        if (
            history_info.action_tool_call is None
            and history_info.action_content is None
            and history_info.action_reasoning is None
        ):
            # This is the initial state, no action taken yet
            _messages.append(
                {
                    "role": "user",
                    "content": filter_non_utf8(
                        history_info.step_observation.observation
                    ),
                }
            )
        else:
            # This is a step with an action taken
            _messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": history_info.action_tool_call.id,  # 'toolu_01SdR84CsnTKRpdH4zwFjvGj'
                            "content": filter_non_utf8(
                                history_info.step_observation.observation
                            ),  # 'Viewing `hangman_test.py`. The file is read-only, it is not editable.'
                        }
                    ],
                }
            )

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
            response = retry_on_exception(
                self.client.messages.create, self.need_to_be_retried
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
        # thinking
        thinking_messages = [r.text for r in response.content if r.type == "thinking"]
        thinking_messages = thinking_messages[0] if thinking_messages else None

        llm_response = LLMResponse(
            prompt=messages,
            response=text_messages,
            reasoning_response=thinking_messages,
            tool=self.parse_tool_call_response(tool_use_block),
            prompt_token_count=response.usage.input_tokens,
            response_token_count=response.usage.output_tokens,
        )

        return llm_response
