import json
from unittest.mock import MagicMock, mock_open, patch

import pytest

from froggy.envs.env import EnvInfo


@pytest.fixture
def build_env_info():
    def _env_info(
        obs="obs",
        max_score=10,
        score=5,
        last_run_obs="last_run_obs",
        tools_obs={},
        dir_tree="dir_tree",
        current_code_with_line_number="current_code_with_line_number",
        current_breakpoints="current_breakpoints",
        action="action",
        instructions=None,
        done=False,
        rewrite_counter=0,
        tools=None,
    ):
        return EnvInfo(
            obs=obs,
            max_score=max_score,
            score=score,
            last_run_obs=last_run_obs,
            tools_obs=tools_obs,
            dir_tree=dir_tree,
            current_code_with_line_number=current_code_with_line_number,
            current_breakpoints=current_breakpoints,
            action=action,
            instructions=instructions if instructions is not None else {},
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
                "max_tokens": 100,
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
                "llm_temperature": [0.5, 0.7],
                "use_conversational_prompt": True,
                "n_rewrites_before_pdb": 2,
                "reset_prompt_history_after_rewrite": False,
                "memory_size": 10,
                "output_path": str(tmp_path),
                "random_seed": 42,
            }
            env = MagicMock()
            llm = MagicMock()
            history = MagicMock()
            agent = agent_class(config_dict, env)
            agent.llm = llm
            agent.history = history
            yield agent, env, llm, history

    return _agent_setup
