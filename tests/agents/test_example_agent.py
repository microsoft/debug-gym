from unittest.mock import MagicMock

from froggy.agents.example_agent import PdbAfterRewrites, PdbAgent, RewriteOnly
from froggy.agents.llm_api import LLMResponse, TokenUsage


def test_build_question_prompt(agent_setup):
    agent, _, _, _ = next(agent_setup(PdbAgent))
    messages = agent.build_question_prompt()
    assert len(messages) == 1
    assert "continue your debugging" in messages[0]["content"]


def test_build_prompt(agent_setup, build_env_info):
    agent, _, _, _ = next(agent_setup(PdbAgent))
    info = build_env_info(
        instructions="Test instructions",
        dir_tree="Test dir tree",
        current_code_with_line_number="Test code",
        current_breakpoints="Test breakpoints",
        step_observation="Test last run obs",
    )
    messages = agent.build_prompt(info)
    assert len(messages) > 0


def test_run(agent_setup, build_env_info):
    agent, env, llm, _ = next(agent_setup(PdbAgent))
    env.reset.return_value = build_env_info(
        done=False,
        score=0,
        max_score=10,
        instructions="Test instructions",
        dir_tree="Test dir tree",
        current_code_with_line_number="Test code",
        current_breakpoints="Test breakpoints",
        step_observation="Test last run obs",
    )
    env.step.return_value = build_env_info(
        done=True,
        score=10,
        max_score=10,
        instructions="Test instructions",
        dir_tree="Test dir tree",
        current_code_with_line_number="Test code",
        current_breakpoints="Test breakpoints",
        step_observation="Test last run obs",
    )
    llm.return_value = LLMResponse("Prompt", "Expected answer", TokenUsage(2, 4))
    result = agent.run(task_name="test_task", debug=False)
    assert result


def test_build_system_prompt_no_pdb(agent_setup, build_env_info):
    agent, _, _, _ = next(agent_setup(RewriteOnly))
    info = build_env_info(
        instructions="Test instructions",
        dir_tree="Test dir tree",
        current_code_with_line_number="Test code",
        current_breakpoints="Test breakpoints",
        step_observation="Test last run obs",
    )
    messages = agent.build_system_prompt(info)
    assert len(messages) == 1
    assert "Overall task" in messages[0]["content"]


def test_build_question_prompt_no_pdb(agent_setup):
    agent, _, _, _ = next(agent_setup(RewriteOnly))
    messages = agent.build_question_prompt()
    assert len(messages) == 1
    assert "continue your debugging" in messages[0]["content"]


def test_run_pdb_after_rewrites(agent_setup, build_env_info):
    agent, env, llm, _ = next(agent_setup(PdbAfterRewrites))
    env.reset.return_value = build_env_info(
        done=False,
        score=0,
        max_score=10,
        rewrite_counter=0,
        instructions="Test instructions",
        dir_tree="Test dir tree",
        current_code_with_line_number="Test code",
        current_breakpoints="Test breakpoints",
        step_observation="Test last run obs",
    )
    env.step.return_value = build_env_info(
        done=True,
        score=10,
        max_score=10,
        rewrite_counter=0,
        instructions="Test instructions",
        dir_tree="Test dir tree",
        current_code_with_line_number="Test code",
        current_breakpoints="Test breakpoints",
        step_observation="Test last run obs",
    )
    llm.return_value = LLMResponse("Prompt", "Expected answer", TokenUsage(2, 4))
    env.tools = {"pdb": MagicMock()}
    result = agent.run(task_name="test_task", debug=False)
    assert result
