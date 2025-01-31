import json
from unittest.mock import MagicMock, mock_open, patch

import pytest

from froggy.agents.zero_shot import (
    AgentZeroShot,
    AgentZeroShot_NoPDB,
    AgentZeroShot_PdbAfterRewrites,
)


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
def agent_zero_shot(tmp_path, open_data):
    with (
        patch("tiktoken.encoding_for_model"),
        patch("os.path.exists", return_value=True),
        patch("builtins.open", new_callable=mock_open, read_data=open_data),
    ):
        config = {
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
        agent = AgentZeroShot(config, env)
        agent.llm = llm
        agent.history = history
        yield agent, env, llm, history


def test_build_question_prompt(agent_zero_shot):
    agent, _, _, _ = agent_zero_shot
    messages = agent.build_question_prompt()
    assert len(messages) == 1
    assert "continue your debugging" in messages[0]["content"]


def test_build_prompt(agent_zero_shot):
    agent, _, _, _ = agent_zero_shot
    info = {
        "instructions": "Test instructions",
        "dir_tree": "Test dir tree",
        "editable_files": "Test editable files",
        "current_code_with_line_number": "Test code",
        "current_breakpoints": "Test breakpoints",
        "last_run_obs": "Test last run obs",
    }
    messages = agent.build_prompt(info)
    assert len(messages) > 0


def test_run(agent_zero_shot):
    agent, env, llm, _ = agent_zero_shot
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
    result = agent.run(task_name="test_task", debug=False)
    assert result


@pytest.fixture
def agent_zero_shot_no_pdb(tmp_path, open_data):
    with (
        patch("tiktoken.encoding_for_model"),
        patch("os.path.exists", return_value=True),
        patch("builtins.open", new_callable=mock_open, read_data=open_data),
    ):
        config = {
            "llm_name": "test-model",
            "max_steps": 10,
            "max_rewrite_steps": 5,
            "llm_temperature": [0.5, 0.7],
            "use_conversational_prompt": True,
            "memory_size": 10,
            "output_path": str(tmp_path),
            "random_seed": 42,
            "reset_prompt_history_after_rewrite": False,
        }
        env = MagicMock()
        llm = MagicMock()
        history = MagicMock()
        agent = AgentZeroShot_NoPDB(config, env)
        agent.llm = llm
        agent.history = history
        yield agent, env, llm, history


def test_build_system_prompt_no_pdb(agent_zero_shot_no_pdb):
    agent, _, _, _ = agent_zero_shot_no_pdb
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


def test_build_question_prompt_no_pdb(agent_zero_shot_no_pdb):
    agent, _, _, _ = agent_zero_shot_no_pdb
    messages = agent.build_question_prompt()
    assert len(messages) == 1
    assert "continue your debugging" in messages[0]["content"]


@pytest.fixture
def agent_zero_shot_pdb_after_rewrites(tmp_path, open_data):
    with (
        patch("tiktoken.encoding_for_model"),
        patch("os.path.exists", return_value=True),
        patch("builtins.open", new_callable=mock_open, read_data=open_data),
    ):
        config = {
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
        agent = AgentZeroShot_PdbAfterRewrites(config, env)
        agent.llm = llm
        agent.history = history
        yield agent, env, llm, history


def test_run_pdb_after_rewrites(agent_zero_shot_pdb_after_rewrites):
    agent, env, llm, _ = agent_zero_shot_pdb_after_rewrites
    env.reset.return_value = (
        None,
        {
            "done": False,
            "score": 0,
            "max_score": 10,
            "rewrite_counter": 0,
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
            "rewrite_counter": 0,
            "instructions": "Test instructions",
            "dir_tree": "Test dir tree",
            "editable_files": "Test editable files",
            "current_code_with_line_number": "Test code",
            "current_breakpoints": "Test breakpoints",
            "last_run_obs": "Test last run obs",
        },
    )
    llm.return_value = ("Expected answer", "Expected token usage")
    env.tools = {"pdb": MagicMock()}
    result = agent.run(task_name="test_task", debug=False)
    assert result
