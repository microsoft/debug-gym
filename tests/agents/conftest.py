import json
from unittest.mock import MagicMock, mock_open, patch

import pytest


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
            llm.define_tools = lambda x: x
            agent = agent_class(config_dict, env)
            agent.llm = llm
            yield agent, env, llm

    return _agent_setup
