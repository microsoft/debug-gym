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


def test_split_reasoning(setup_reasoning_tool):
    reasoning_tool, _ = setup_reasoning_tool
    action = "```reasoning\nreasoning text\n</reasoning>\nnext_action ```"
    assert reasoning_tool.split_reasoning(action) == ("reasoning text", "next_action")

    action = "```reasoning\nreasoning text```"
    with pytest.raises(ValueError):
        reasoning_tool.split_reasoning(action)

    action = "reasoning\nreasoning text\n</reasoning>\nnext_action"
    with pytest.raises(IndexError):
        reasoning_tool.split_reasoning(action)


def test_use_without_chaining(setup_reasoning_tool):
    reasoning_tool, _ = setup_reasoning_tool
    action = "```reasoning\nreasoning text```"
    assert reasoning_tool.use_without_chaining(action) == "Reasoning:\nreasoning text"

    action = "```reasoning\nreasoning text\n</reasoning>\nnext_action```"
    assert (
        reasoning_tool.use_without_chaining(action)
        == "Reasoning:\nreasoning text\n</reasoning>\nnext_action"
    )

    action = "reasoning\nreasoning text```"
    assert reasoning_tool.use_without_chaining(action) == "SyntaxError: invalid syntax."


def test_use_with_chaining(setup_reasoning_tool):
    reasoning_tool, _ = setup_reasoning_tool
    reasoning_tool.split_reasoning = MagicMock(
        return_value=("reasoning", "next_action")
    )
    env_info = EnvInfo(
        obs="obs",
        max_score=10,
        score=5,
        last_run_obs="last_run_obs",
        observations=[],
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
        reasoning_tool.use_with_chaining("action")
        == "Reasoning:\nreasoning\nExecuting action:\nnext_action\nNext observation:\nobs"
    )

    reasoning_tool.split_reasoning = MagicMock(side_effect=ValueError)
    assert reasoning_tool.use_with_chaining("action") == "SyntaxError: invalid syntax."

    reasoning_tool.split_reasoning = MagicMock(
        return_value=("reasoning", "next_action")
    )
    reasoning_tool.environment.step = MagicMock(side_effect=ValueError)
    assert (
        reasoning_tool.use_with_chaining("action")
        == "Error while executing the action after reasoning.\nSyntaxError: invalid syntax."
    )

    reasoning_tool.split_reasoning = MagicMock(
        return_value=("reasoning", "```reasoning something else```")
    )
    reasoning_tool.environment.step = MagicMock(
        return_value=("obs", 0.0, False, {"key": "value"})
    )
    assert (
        reasoning_tool.use_with_chaining("action")
        == "SyntaxError: invalid syntax. You cannot chain reasoning actions."
    )

    reasoning_tool.split_reasoning = MagicMock(
        return_value=("reasoning", "next_action")
    )
    env_info.obs = "Invalid action: action."
    reasoning_tool.environment.step = MagicMock(return_value=env_info)
    assert (
        reasoning_tool.use_with_chaining("action")
        == "Error while executing the action after reasoning.\nInvalid action: action."
    )

    reasoning_tool.split_reasoning = MagicMock(
        return_value=("reasoning", "next_action")
    )
    env_info.obs = "Error while using tool: cot"
    reasoning_tool.environment.step = MagicMock(return_value=env_info)
    assert (
        reasoning_tool.use_with_chaining("action")
        == "Error while executing the action after reasoning.\nError while using tool: cot"
    )
