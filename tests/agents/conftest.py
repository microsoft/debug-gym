import json
import logging
from unittest.mock import MagicMock, mock_open, patch

import pytest

from debug_gym.agents.llm_api import LLM
from debug_gym.gym.entities import Observation
from debug_gym.gym.envs.env import EnvInfo
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
        def generate(self, messages, **kwargs):
            self.called_messages = messages
            self.called_kwargs = kwargs
            return "Test response"

        def tokenize(self, text):
            return [c for c in text]

    return LLMMock


@pytest.fixture
def build_env_info():
    def _env_info(
        step_observation="obs",
        all_observations=[],
        eval_observation="eval_observation",
        dir_tree="dir_tree",
        current_code_with_line_number="current_code_with_line_number",
        current_breakpoints="current_breakpoints",
        action="action",
        instructions=None,
        score=5,
        max_score=10,
        done=False,
        rewrite_counter=0,
        tools=None,
    ):
        return EnvInfo(
            step_observation=Observation("tool", step_observation),
            all_observations=all_observations,
            eval_observation=Observation("env", eval_observation),
            dir_tree=dir_tree,
            current_code_with_line_number=current_code_with_line_number,
            current_breakpoints=current_breakpoints,
            action=action,
            instructions=instructions if instructions is not None else {},
            score=score,
            max_score=max_score,
            done=done,
            rewrite_counter=rewrite_counter,
            tools=tools if tools is not None else {},
        )

    return _env_info


@pytest.fixture
def open_data():
    data = json.dumps(
        {
            "test-model": {
                "model": "test-model",
                "tokenizer": "gpt-4o",
                "context_limit": 4,
                "api_key": "test-api-key",
                "endpoint": "https://test-endpoint",
                "api_version": "v1",
                "tags": ["azure openai"],
            }
        }
    )
    return data


@pytest.fixture
def agent_setup(tmp_path, open_data):
    def _length(text):
        return len(text)

    def _agent_setup(agent_class):
        with (
            patch("tiktoken.encoding_for_model") as mock_encoding_for_model,
            patch("os.path.exists", return_value=True),
            patch("builtins.open", new_callable=mock_open, read_data=open_data),
        ):
            mock_encoding = MagicMock()
            mock_encoding.encode = lambda x: x.split()
            mock_encoding_for_model.return_value = mock_encoding

            config_dict = {
                "llm_name": "test-model",
                "max_steps": 10,
                "max_rewrite_steps": 5,
                "use_conversational_prompt": True,
                "n_rewrites_before_pdb": 2,
                "reset_prompt_history_after_rewrite": False,
                "memory_size": 10,
                "output_path": str(tmp_path),
                "random_seed": 42,
            }
            env = MagicMock()
            llm = MagicMock()
            llm.reasoning_end_token = None
            llm.context_length = 4096
            llm.count_tokens = _length
            history = MagicMock()
            agent = agent_class(config_dict, env)
            agent.llm = llm
            agent.history = history
            yield agent, env, llm, history

    return _agent_setup
