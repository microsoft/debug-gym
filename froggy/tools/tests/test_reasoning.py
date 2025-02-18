from unittest.mock import MagicMock

import pytest

from froggy.envs.env import EnvInfo, RepoEnv
from froggy.tools.reasoning import ReasoningTool


@pytest.fixture
def setup_reasoning_tool():
    env = MagicMock(spec=RepoEnv)
    environment = MagicMock()
    reasoning_tool = ReasoningTool()
    reasoning_tool.environment = environment
    return reasoning_tool, env


def test_register(setup_reasoning_tool):
    reasoning_tool, env = setup_reasoning_tool
    reasoning_tool.register(env)
    assert reasoning_tool.environment == env


def test_register_invalid_environment(setup_reasoning_tool):
    reasoning_tool, _ = setup_reasoning_tool
    with pytest.raises(ValueError):
        reasoning_tool.register(MagicMock())


def test_use_without_chaining(setup_reasoning_tool):
    reasoning_tool, _ = setup_reasoning_tool
    assert (
        reasoning_tool.use_without_chaining(reasoning_text="reasoning text")
        == "Reasoning:\nreasoning text"
    )


def test_use_with_chaining(setup_reasoning_tool):
    reasoning_tool, _ = setup_reasoning_tool
    env_info = EnvInfo(
        obs="obs",
        max_score=10,
        score=5,
        last_run_obs="last_run_obs",
        dbg_obs="dbg_obs",
        dir_tree="dir_tree",
        current_code_with_line_number="current_code_with_line_number",
        current_breakpoints="current_breakpoints",
        action="action",
        instructions={},
        done=False,
        rewrite_counter=0,
        tools={},
    )
    reasoning_tool.environment.step = MagicMock(
        return_value=env_info  # ("obs", 0.0, False, {"key": "value"})
    )
    assert (
        reasoning_tool.use_with_chaining(
            reasoning_text="reasoning", next_action="next_action"
        )
        == "Reasoning:\nreasoning\nExecuting action:\nnext_action\nNext observation:\nobs"
    )

    reasoning_tool.environment.step = MagicMock(side_effect=ValueError)
    assert (
        reasoning_tool.use_with_chaining(
            reasoning_text="reasoning", next_action="next_action"
        )
        == "Error while executing the action after reasoning.\nSyntaxError: invalid syntax."
    )

    reasoning_tool.environment.step = MagicMock(
        return_value=("obs", 0.0, False, {"key": "value"})
    )
    assert (
        reasoning_tool.use_with_chaining(
            reasoning_text="reasoning", next_action="reasoning(something else)"
        )
        == "SyntaxError: invalid syntax. You cannot chain reasoning actions."
    )

    env_info.obs = "Invalid action: action."
    reasoning_tool.environment.step = MagicMock(return_value=env_info)
    assert (
        reasoning_tool.use_with_chaining(
            reasoning_text="reasoning", next_action="next_action"
        )
        == "Error while executing the action after reasoning.\nInvalid action: action."
    )

    env_info.obs = "Error while using tool: cot"
    reasoning_tool.environment.step = MagicMock(return_value=env_info)
    assert (
        reasoning_tool.use_with_chaining(
            reasoning_text="reasoning", next_action="next_action"
        )
        == "Error while executing the action after reasoning.\nError while using tool: cot"
    )
