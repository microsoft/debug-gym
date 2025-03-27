from unittest.mock import MagicMock

from debug_gym.agents.debug_agent import Debug_5_Agent, DebugAgent
from debug_gym.agents.llm_api import LLMResponse, TokenUsage
from debug_gym.agents.rewrite_agent import RewriteAgent


def test_build_question_prompt(agent_setup):
    agent, _, _, _ = next(agent_setup(DebugAgent))
    messages = agent.build_question_prompt()
    assert len(messages) == 1
    assert "continue your debugging" in messages[0]["content"]


def test_build_prompt(agent_setup, build_env_info):
    agent, _, _, _ = next(agent_setup(DebugAgent))
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
    agent, env, llm, _ = next(agent_setup(DebugAgent))
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


def test_build_system_prompt_rewrite_agent(agent_setup, build_env_info):
    agent, _, _, _ = next(agent_setup(RewriteAgent))
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


def test_build_question_prompt_rewrite_agent(agent_setup):
    agent, _, _, _ = next(agent_setup(RewriteAgent))
    messages = agent.build_question_prompt()
    assert len(messages) == 1
    assert "continue your debugging" in messages[0]["content"]


def test_run_debug_5_agent(agent_setup, build_env_info):
    agent, env, llm, _ = next(agent_setup(Debug_5_Agent))
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
