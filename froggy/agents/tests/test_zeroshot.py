from unittest.mock import MagicMock

from froggy.agents.zero_shot import (
    AgentZeroShot,
    AgentZeroShot_NoPDB,
    AgentZeroShot_PdbAfterRewrites,
)


def test_build_question_prompt(agent_setup):
    agent, _, _, _ = next(agent_setup(AgentZeroShot))
    messages = agent.build_question_prompt()
    assert len(messages) == 1
    assert "continue your debugging" in messages[0]["content"]


def test_build_prompt(agent_setup):
    agent, _, _, _ = next(agent_setup(AgentZeroShot))
    info = {
        "instructions": "Test instructions",
        "dir_tree": "Test dir tree",
        "current_code_with_line_number": "Test code",
        "current_breakpoints": "Test breakpoints",
        "last_run_obs": "Test last run obs",
    }
    messages = agent.build_prompt(info)
    assert len(messages) > 0


def test_run(agent_setup):
    agent, env, llm, _ = next(agent_setup(AgentZeroShot))
    env.reset.return_value = (
        None,
        {
            "done": False,
            "score": 0,
            "max_score": 10,
            "instructions": "Test instructions",
            "dir_tree": "Test dir tree",
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
            "current_code_with_line_number": "Test code",
            "current_breakpoints": "Test breakpoints",
            "last_run_obs": "Test last run obs",
        },
    )
    llm.return_value = ("Expected answer", "Expected token usage")
    result = agent.run(task_name="test_task", debug=False)
    assert result


def test_build_system_prompt_no_pdb(agent_setup):
    agent, _, _, _ = next(agent_setup(AgentZeroShot_NoPDB))
    info = {
        "instructions": "Test instructions",
        "dir_tree": "Test dir tree",
        "current_code_with_line_number": "Test code",
        "current_breakpoints": "Test breakpoints",
        "last_run_obs": "Test last run obs",
    }
    messages = agent.build_system_prompt(info)
    assert len(messages) == 1
    assert "Overall task" in messages[0]["content"]


def test_build_question_prompt_no_pdb(agent_setup):
    agent, _, _, _ = next(agent_setup(AgentZeroShot_NoPDB))
    messages = agent.build_question_prompt()
    assert len(messages) == 1
    assert "continue your debugging" in messages[0]["content"]


def test_run_pdb_after_rewrites(agent_setup):
    agent, env, llm, _ = next(agent_setup(AgentZeroShot_PdbAfterRewrites))
    env.reset.return_value = (
        None,
        {
            "done": False,
            "score": 0,
            "max_score": 10,
            "rewrite_counter": 0,
            "instructions": "Test instructions",
            "dir_tree": "Test dir tree",
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
            "current_code_with_line_number": "Test code",
            "current_breakpoints": "Test breakpoints",
            "last_run_obs": "Test last run obs",
        },
    )
    llm.return_value = ("Expected answer", "Expected token usage")
    env.tools = {"pdb": MagicMock()}
    result = agent.run(task_name="test_task", debug=False)
    assert result
