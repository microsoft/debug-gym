import json
import sys

import numpy as np

from debug_gym.gym.envs.env import EnvInfo
from debug_gym.gym.tools.tool import EnvironmentTool, ToolCall
from debug_gym.llms.base import LLM, LLMResponse
from debug_gym.llms.utils import print_messages
from debug_gym.logger import DebugGymLogger


prompt_toolkit_available = False
try:
    # For command line history and autocompletion.
    from prompt_toolkit import prompt
    from prompt_toolkit.shortcuts import CompleteStyle
    from prompt_toolkit.completion import WordCompleter
    from prompt_toolkit.history import InMemoryHistory

    prompt_toolkit_available = sys.stdout.isatty()
except ImportError:
    pass


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

    def define_tools(self, tool_call_list: list[EnvironmentTool]) -> list[dict]:
        available_commands = []
        meta_dict = {}
        for tool in tool_call_list:
            random_id = "".join(map(str, np.random.randint(0, 10, size=6)))
            tool_id = f"{tool.name}-{random_id}"
            template = {
                "id": tool_id,
                "name": tool.name,
                "arguments": tool.arguments,
                "description": tool.description,
            }
            available_commands.append(template)
            meta_dict[json.dumps(template)] = tool.description

        return available_commands, meta_dict

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
        all_tools, meta_dict = self.define_tools(tools)
        available_commands = [json.dumps(t) for t in all_tools]
        tool_call = None
        retry_count = 0
        action = ""

        while tool_call is None and retry_count < self.max_retries:
            if prompt_toolkit_available:
                actions_completer = WordCompleter(
                    available_commands, ignore_case=False, sentence=True,
                    meta_dict=meta_dict
                )
                action = prompt(
                    "\n> ",
                    completer=actions_completer,
                    complete_style=CompleteStyle.MULTI_COLUMN,
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
