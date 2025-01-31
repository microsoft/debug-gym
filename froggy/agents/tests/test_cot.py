import json
from unittest.mock import MagicMock, mock_open, patch

import pytest

from froggy.agents.cot import AgentCoT, AgentCoT_NoPDB


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
def agent_cot_setup(tmp_path, open_data):
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
            "reset_prompt_history_after_rewrite": False,
            "memory_size": 10,
            "output_path": str(tmp_path),
            "random_seed": 42,
        }
        env = MagicMock()
        llm = MagicMock()
        history = MagicMock()
        agent = AgentCoT(config_dict, env)
        agent.llm = llm
        agent.history = history
        yield agent, env, llm, history


def test_build_cot_prompt(agent_cot_setup):
    agent, _, _, _ = agent_cot_setup
    messages = agent.build_cot_prompt()
    assert len(messages) == 1
    assert "Let's think step by step" in messages[0]["content"]


def test_build_prompt_step_1(agent_cot_setup):
    agent, _, _, _ = agent_cot_setup
    info = {
        "instructions": "Test instructions",
        "dir_tree": "Test dir tree",
        "editable_files": "Test editable files",
        "current_code_with_line_number": "Test code",
        "current_breakpoints": "Test breakpoints",
        "last_run_obs": "Test last run obs",
    }
    messages = agent.build_prompt_step_1(info)
    assert len(messages) > 0


def test_fill_in_cot_response(agent_cot_setup):
    agent, _, _, _ = agent_cot_setup
    response = "Test response"
    messages = agent.fill_in_cot_response(response)
    assert len(messages) == 1
    assert "assistant" in messages[0]["role"]


def test_build_question_prompt(agent_cot_setup):
    agent, _, _, _ = agent_cot_setup
    messages = agent.build_question_prompt()
    assert len(messages) == 1
    assert "what is the best next command?" in messages[0]["content"]


def test_build_prompt_step_2(agent_cot_setup):
    agent, _, _, _ = agent_cot_setup
    info = {
        "instructions": "Test instructions",
        "dir_tree": "Test dir tree",
        "editable_files": "Test editable files",
        "current_code_with_line_number": "Test code",
        "current_breakpoints": "Test breakpoints",
        "last_run_obs": "Test last run obs",
    }
    response = "Test response"
    messages = agent.build_prompt_step_2(info, response)
    assert len(messages) > 0


def test_run(agent_cot_setup):
    agent, env, llm, _ = agent_cot_setup
    env.reset.return_value = (
        None,
        {
            "done": False,
            "score": 0,
            "max_score": 10,
            "instructions": "Test instructions",
            "dir_tree": "Test dir tree",
            "editable_files": "Test editable files",
            "current_code_with_line_number": "Test code",
            "current_breakpoints": "Test breakpoints",
            "last_run_obs": "Test last run obs",
        },
    )
    env.step.return_value = (
        None,
        None,
        True,
        {
            "done": True,
            "score": 10,
            "max_score": 10,
            "instructions": "Test instructions",
            "dir_tree": "Test dir tree",
            "editable_files": "Test editable files",
            "current_code_with_line_number": "Test code",
            "current_breakpoints": "Test breakpoints",
            "last_run_obs": "Test last run obs",
        },
    )
    llm.return_value = ("Expected answer", "Expected token usage")
    assert agent.run(task_name="test_task", debug=False)


@pytest.fixture
def agent_cot_no_pdb_setup(tmp_path, open_data):
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
            "memory_size": 10,
            "output_path": str(tmp_path),
            "random_seed": 42,
        }
        env = MagicMock()
        llm = MagicMock()
        history = MagicMock()
        agent = AgentCoT_NoPDB(config_dict, env)
        agent.llm = llm
        agent.history = history
        yield agent, env, llm, history


def test_build_system_prompt(agent_cot_no_pdb_setup):
    agent, _, _, _ = agent_cot_no_pdb_setup
    info = {
        "instructions": "Test instructions",
        "dir_tree": "Test dir tree",
        "editable_files": "Test editable files",
        "current_code_with_line_number": "Test code",
        "current_breakpoints": "Test breakpoints",
        "last_run_obs": "Test last run obs",
    }
    messages = agent.build_system_prompt(info)
    assert len(messages) == 1
    assert "Overall task" in messages[0]["content"]


def test_no_pdb_build_cot_prompt(agent_cot_no_pdb_setup):
    agent, _, _, _ = agent_cot_no_pdb_setup
    messages = agent.build_cot_prompt()
    assert len(messages) == 1
    assert "Let's think step by step" in messages[0]["content"]
