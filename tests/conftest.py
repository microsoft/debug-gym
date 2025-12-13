import json
import logging
from unittest.mock import patch

import pytest

from debug_gym.gym.entities import Observation
from debug_gym.gym.envs.env import EnvInfo
from debug_gym.gym.tools.tool import ToolCall
from debug_gym.llms.base import LLM, LLMConfigRegistry, LLMResponse
from debug_gym.logger import DebugGymLogger


@pytest.fixture
def logger_mock():
    logger = DebugGymLogger("test_logger")
    logger.setLevel(logging.DEBUG)
    logs = []

    class ListHandler(logging.Handler):
        def emit(self, record):
            logs.append(record.getMessage())

    handler = ListHandler()
    logger.addHandler(handler)
    logger._log_history = logs
    return logger


@pytest.fixture
def llm_class_mock():
    class LLMMock(LLM):
        def generate(self, messages, tools, **kwargs):
            self.called_messages = messages
            self.called_tools = tools
            self.called_kwargs = kwargs
            return LLMResponse(
                prompt="Prompt",
                response="Test response",
                action=ToolCall(
                    id="tool_id",
                    name="tool_name",
                    arguments={"arg1": "value1", "arg2": "value2"},
                ),
                prompt_token_count=10,
                response_token_count=20,
            )

        def tokenize(self, messages):
            # Return list of token lists, one per message
            result = []
            for msg in messages:
                content = str(msg.get("content", msg.get("tool_calls", msg)))
                result.append([c for c in content])
            return result

        def define_tools(self, tool_call_list):
            return tool_call_list

        def parse_tool_call_response(self, response):
            return response

        def convert_response_to_message(self, response: LLMResponse) -> dict:
            message = {
                "role": "assistant",
                "content": response.response,
            }

            tool = getattr(response, "tool", None)
            if isinstance(tool, ToolCall):
                message["tool_calls"] = [
                    {
                        "type": "function",
                        "id": tool.id,
                        "function": {
                            "name": tool.name,
                            "arguments": json.dumps(tool.arguments),
                        },
                    }
                ]

            return message

        def convert_observation_to_message(
            self,
            observation: str,
            action_tool_call_id: str | None = None,
            action_tool_call_name: str | None = None,
        ) -> dict:
            if action_tool_call_id:
                return {
                    "role": "tool",
                    "tool_call_id": action_tool_call_id,
                    "name": action_tool_call_name,
                    "content": observation,
                }
            return {
                "role": "user",
                "content": observation,
            }

    return LLMMock


@pytest.fixture
@patch.object(
    LLMConfigRegistry,
    "from_file",
    return_value=LLMConfigRegistry.register_all(
        {
            "llm-mock": {
                "model": "llm-mock",
                "context_limit": 4,
                "tokenizer": "test-tokenizer",
                "tags": [],
                "generate_kwargs": {
                    "temperature": 0.7,
                    "max_tokens": 100,
                    "thinking": {
                        "type": "enabled",
                        "budget_tokens": 10,
                    },
                },
            }
        }
    ),
)
def llm_mock(mock_llm_config, logger_mock, llm_class_mock):
    llm = llm_class_mock("llm-mock", logger=logger_mock)
    return llm


@pytest.fixture
def build_env_info():
    def _env_info(
        step_observation="obs",
        all_observations=[],
        eval_observation="eval_observation",
        current_breakpoints="current_breakpoints",
        action_tool_call=ToolCall(id="tool_id", name="tool_name", arguments={}),
        action_reasoning="",
        action_content="",
        instructions=None,
        score=5,
        max_score=10,
        terminated=False,
        resolved=False,
        tools=[],
    ):
        return EnvInfo(
            step_observation=Observation("tool", step_observation),
            all_observations=all_observations,
            eval_observation=Observation("env", eval_observation),
            current_breakpoints=current_breakpoints,
            action_reasoning=action_reasoning,
            action_content=action_content,
            action_tool_call=action_tool_call,
            instructions=instructions if instructions is not None else {},
            score=score,
            max_score=max_score,
            terminated=terminated,
            resolved=resolved,
            tools=tools if tools is not None else [],
        )

    return _env_info
