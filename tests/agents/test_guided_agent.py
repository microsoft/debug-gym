from unittest.mock import MagicMock, patch

from debug_gym.agents import GuidedRewriteAgent
from debug_gym.llms import Human
from debug_gym.llms.base import LLMResponse, TokenUsage


@patch.object(
    Human,
    "__call__",
    return_value=LLMResponse(
        "Prompt",
        '{"id": "pdb-267437", "name": "pdb", "arguments": {"command": "c"}}',
        TokenUsage(2, 4),
    ),
)
def test_human_in_the_loop(human, agent_setup, build_env_info):
    agent, env, llm = next(agent_setup(GuidedRewriteAgent))
    env.reset.return_value = build_env_info(
        done=False,
        score=0,
        max_score=10,
        rewrite_counter=0,
        instructions="Test instructions",
        dir_tree="Test dir tree",
        current_breakpoints="Test breakpoints",
        step_observation="Test last run obs",
    )
    env.step.return_value = build_env_info(
        done=False,
        score=10,
        max_score=10,
        rewrite_counter=0,
        instructions="Test instructions",
        dir_tree="Test dir tree",
        current_breakpoints="Test breakpoints",
        step_observation="Test last run obs",
    )

    env.clone.return_value = MagicMock()
    llm.return_value = LLMResponse("Prompt", "Expected answer", TokenUsage(2, 4))
    env.tools = {"pdb": MagicMock()}

    env.clone().step.return_value = build_env_info(
        done=True,
        score=10,
        max_score=10,
        rewrite_counter=0,
        instructions="Test instructions",
        dir_tree="Test dir tree",
        current_breakpoints="Test breakpoints",
        step_observation="Test last run obs",
    )
    result = agent.run(task_name="test_task", debug=False)

    assert result is False
    # test that llm actions were executed
    assert env.step.called
    env.step.assert_called_with(human().response)
    assert env.step().done is False

    # test that llm actions were logged
    _history, _prompt_response_pairs = agent.history.get()
    assert [[], [human()]] == _prompt_response_pairs

    # test that env was cloned
    assert env.clone.called
    assert env.clone().reset.called

    # assert that cloned env was called with history steps
    env.clone().step.assert_has_calls(
        [
            call(agent.history.get_all()[0].action),
        ]
    )

    # test that human action was executed
    assert env.clone().step.called
    env.clone().step.assert_called_with(llm().response)

    # ensure that human action was not recorded in history
    assert env.clone().step() not in agent.history.get_all()
