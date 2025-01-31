from froggy.agents.cot import AgentCoT, AgentCoT_NoPDB


def test_build_cot_prompt(agent_setup):
    agent, _, _, _ = next(agent_setup(AgentCoT))
    messages = agent.build_cot_prompt()
    assert len(messages) == 1
    assert "Let's think step by step" in messages[0]["content"]


def test_build_prompt_step_1(agent_setup):
    agent, _, _, _ = next(agent_setup(AgentCoT))
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


def test_fill_in_cot_response(agent_setup):
    agent, _, _, _ = next(agent_setup(AgentCoT))
    response = "Test response"
    messages = agent.fill_in_cot_response(response)
    assert len(messages) == 1
    assert "assistant" in messages[0]["role"]


def test_build_question_prompt(agent_setup):
    agent, _, _, _ = next(agent_setup(AgentCoT))
    messages = agent.build_question_prompt()
    assert len(messages) == 1
    assert "what is the best next command?" in messages[0]["content"]


def test_build_prompt_step_2(agent_setup):
    agent, _, _, _ = next(agent_setup(AgentCoT))
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


def test_run(agent_setup):
    agent, env, llm, _ = next(agent_setup(AgentCoT))
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


def test_build_system_prompt(agent_setup):
    agent, _, _, _ = next(agent_setup(AgentCoT_NoPDB))
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


def test_no_pdb_build_cot_prompt(agent_setup):
    agent, _, _, _ = next(agent_setup(AgentCoT_NoPDB))
    messages = agent.build_cot_prompt()
    assert len(messages) == 1
    assert "Let's think step by step" in messages[0]["content"]
