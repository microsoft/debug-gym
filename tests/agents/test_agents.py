import json
from unittest.mock import MagicMock, Mock

import pytest
from jinja2 import Template

from debug_gym.agents.debug_agent import Debug_5_Agent, DebugAgent
from debug_gym.agents.rewrite_agent import RewriteAgent
from debug_gym.llms.base import LLMResponse, TokenUsage


def test_default_system_prompt(agent_setup, build_env_info):
    agent, _, _ = next(agent_setup(DebugAgent))
    agent.system_prompt = "some task"
    agent.shortcut_features = Mock(return_value=["f1", "f2"])
    info = build_env_info(
        instructions="some instruction",
        dir_tree="dir tree",
        current_breakpoints=[],
        eval_observation="eval obs",
    )
    system_prompt = agent.build_system_prompt(info)
    expected = [
        {
            "role": "system",
            "content": json.dumps(
                {
                    "Overall task": "some task",
                    "Instructions": "some instruction",
                    "Repo directory tree": "dir tree",
                    "Current breakpoints": [],
                    "Shortcut features": ["f1", "f2"],
                },
                indent=2,
            ),
        }
    ]
    assert system_prompt == expected


def test_default_system_prompt_auto_eval(agent_setup, build_env_info):
    agent, _, _ = next(agent_setup(DebugAgent))
    agent.system_prompt = "some task"
    agent.shortcut_features = Mock(return_value=["f1", "f2"])
    agent.config["env_kwargs"] = {"auto_eval_on_rewrite": True}
    info = build_env_info(
        instructions="some instruction",
        dir_tree="dir tree",
        current_breakpoints=[],
        eval_observation="eval obs",
    )
    system_prompt = agent.build_system_prompt(info)
    expected = [
        {
            "role": "system",
            "content": json.dumps(
                {
                    "Overall task": "some task",
                    "Instructions": "some instruction",
                    "Repo directory tree": "dir tree",
                    "Current breakpoints": [],
                    "Evaluation output of current code": "eval obs",
                    "Shortcut features": ["f1", "f2"],
                },
                indent=2,
            ),
        }
    ]
    assert system_prompt == expected


def test_load_system_prompt_template_default_no_shortcuts_or_eval(
    agent_setup, build_env_info
):
    agent, _, _ = next(agent_setup(DebugAgent))
    agent.system_prompt = "some task"
    agent.shortcut_features = Mock(return_value=[])
    info = build_env_info(
        instructions="some instruction",
        dir_tree="dir tree",
        current_breakpoints=[1, 2],
        eval_observation="",
    )
    system_prompt = agent.build_system_prompt(info)
    expected = [
        {
            "role": "system",
            "content": json.dumps(
                {
                    "Overall task": "some task",
                    "Instructions": "some instruction",
                    "Repo directory tree": "dir tree",
                    "Current breakpoints": [1, 2],
                },
                indent=2,
            ),
        }
    ]
    assert system_prompt == expected


def test_load_system_prompt_template_from_file(tmp_path, agent_setup):
    agent, _, _ = next(agent_setup(DebugAgent))
    agent.system_prompt = "test task"
    template_content = "Task: {{ agent.system_prompt }}"
    template_path = tmp_path / "template.jinja"
    template_path.write_text(template_content)
    agent.config["system_prompt_template_file"] = str(template_path)
    template = agent._load_system_prompt_template()
    assert isinstance(template, Template)
    assert template.render(agent=agent) == "Task: test task"


def test_load_system_prompt_template_file_not_found(agent_setup):
    agent, _, _ = next(agent_setup(DebugAgent))
    agent.config["system_prompt_template_file"] = "non_existent_template.jinja"
    with pytest.raises(FileNotFoundError):
        agent._load_system_prompt_template()


def test_build_system_prompt(agent_setup, build_env_info):
    agent, _, _ = next(agent_setup(DebugAgent))
    agent.config["env_kwargs"] = {"auto_eval_on_rewrite": True}
    agent.system_prompt = "Test overall task"
    info = build_env_info(
        instructions="Do X",
        dir_tree="repo/tree",
        current_breakpoints=[1, 2],
        eval_observation="eval obs",
    )
    messages = agent.build_system_prompt(info)
    expected = {
        "Overall task": "Test overall task",
        "Instructions": "Do X",
        "Repo directory tree": "repo/tree",
        "Current breakpoints": [1, 2],
        "Evaluation output of current code": "eval obs",
        "Shortcut features": [
            "After successful rewrites, the environment will automatically call "
            "the Eval tool to evaluate the rewritten code. Therefore, you do not "
            "need to call the Eval tool yourself. The evaluation output will be "
            "updated automatically in the system prompt."
        ],
    }
    assert messages == [{"role": "system", "content": json.dumps(expected, indent=2)}]


def test_build_prompt(agent_setup, build_env_info):
    agent, _, _ = next(agent_setup(DebugAgent))
    info = build_env_info(
        instructions="Test instructions",
        dir_tree="Test dir tree",
        current_breakpoints="Test breakpoints",
        step_observation="Test last run obs",
    )
    messages = agent.build_prompt(info)
    assert len(messages) > 0


def test_run(agent_setup, build_env_info):
    agent, env, llm = next(agent_setup(DebugAgent))
    env.reset.return_value = build_env_info(
        done=False,
        score=0,
        max_score=10,
        instructions="Test instructions",
        dir_tree="Test dir tree",
        current_breakpoints="Test breakpoints",
        step_observation="Test last run obs",
    )
    env.step.return_value = build_env_info(
        done=True,
        score=10,
        max_score=10,
        instructions="Test instructions",
        dir_tree="Test dir tree",
        current_breakpoints="Test breakpoints",
        step_observation="Test last run obs",
    )
    llm.return_value = LLMResponse("Prompt", "Expected answer", TokenUsage(2, 4))
    result = agent.run(task_name="test_task", debug=False)
    assert result


def test_build_system_prompt_rewrite_agent(agent_setup, build_env_info):
    agent, _, _ = next(agent_setup(RewriteAgent))
    info = build_env_info(
        instructions="Test instructions",
        dir_tree="Test dir tree",
        current_breakpoints="Test breakpoints",
        step_observation="Test last run obs",
    )
    messages = agent.build_system_prompt(info)
    assert len(messages) == 1
    assert "Overall task" in messages[0]["content"]


def test_run_debug_5_agent(agent_setup, build_env_info):
    agent, env, llm = next(agent_setup(Debug_5_Agent))
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
        done=True,
        score=10,
        max_score=10,
        rewrite_counter=0,
        instructions="Test instructions",
        dir_tree="Test dir tree",
        current_breakpoints="Test breakpoints",
        step_observation="Test last run obs",
    )
    llm.return_value = LLMResponse("Prompt", "Expected answer", TokenUsage(2, 4))
    env.tools = {"pdb": MagicMock()}
    result = agent.run(task_name="test_task", debug=False)
    assert result
