from froggy.agents.tadpole import AgentTadpole


def test_build_task_decomposer_prompt(agent_setup):
    agent, _, _, _ = next(agent_setup(AgentTadpole))
    messages = agent.build_task_decomposer_prompt()
    assert len(messages) == 1
    assert "Let's identify the current subgoal" in messages[0]["content"]


def test_build_prompt_step_1(agent_setup):
    agent, _, _, _ = next(agent_setup(AgentTadpole))
    info = {
        "instructions": "Test instructions",
        "dir_tree": "Test dir tree",
        "current_code_with_line_number": "Test code",
        "current_breakpoints": "Test breakpoints",
        "last_run_obs": "Test last run obs",
    }
    messages = agent.build_prompt_step_1(info)
    assert len(messages) > 0


def test_build_question_prompt(agent_setup):
    agent, _, _, _ = next(agent_setup(AgentTadpole))
    agent.current_subgoal = "Test subgoal"
    messages = agent.build_question_prompt()
    assert len(messages) == 1
    assert "what is the best next command?" in messages[0]["content"]


def test_build_prompt_step_2(agent_setup):
    agent, _, _, _ = next(agent_setup(AgentTadpole))
    info = {
        "instructions": "Test instructions",
        "dir_tree": "Test dir tree",
        "current_code_with_line_number": "Test code",
        "current_breakpoints": "Test breakpoints",
        "last_run_obs": "Test last run obs",
    }
    agent.current_subgoal = "Test subgoal"
    messages = agent.build_prompt_step_2(info)
    assert len(messages) > 0


def test_run(agent_setup):
    agent, env, llm, _ = next(agent_setup(AgentTadpole))
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
