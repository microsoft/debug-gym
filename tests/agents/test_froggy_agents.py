import json
from unittest.mock import MagicMock, Mock, patch

import pytest
from jinja2 import Template

from debug_gym.agents.debug_agent import Debug_5_Agent, DebugAgent
from debug_gym.agents.froggy_agent import build_history_prompt
from debug_gym.agents.history_tracker import HistoryTracker
from debug_gym.agents.rewrite_agent import RewriteAgent
from debug_gym.agents.utils import save_patch, save_trajectory
from debug_gym.gym.tools.tool import ToolCall
from debug_gym.gym.tools.toolbox import Toolbox
from debug_gym.llms.base import LLMConfigRegistry, LLMResponse, TokenUsage
from debug_gym.llms.openai import OpenAILLM


def test_default_system_prompt(agent_setup, build_env_info):
    agent, env, _ = next(agent_setup(DebugAgent))
    env.get_tool = MagicMock(
        side_effect=KeyError("no tools for testing")
    )  # KeyError to simulate missing tool
    agent.system_prompt = "some task"
    agent.shortcut_features = Mock(return_value=["f1", "f2"])
    info = build_env_info(
        instructions="some instruction",
        current_breakpoints=[],
        eval_observation="eval obs",
    )
    system_prompt = agent.build_system_prompt(info)
    expected = {
        "role": "system",
        "content": json.dumps(
            {
                "Overall task": "some task",
                "Instructions": "some instruction",
                "Shortcut features": ["f1", "f2"],
            },
            indent=2,
        ),
    }
    assert system_prompt == expected


def test_default_system_prompt_auto_eval(agent_setup, build_env_info):
    agent, env, _ = next(agent_setup(DebugAgent))
    agent.system_prompt = "some task"
    agent.shortcut_features = Mock(return_value=["f1", "f2"])

    eval_tool = Toolbox.get_tool("eval", auto_eval_on_rewrite=True)
    env.add_tool(eval_tool)
    info = build_env_info(
        instructions="some instruction",
        current_breakpoints=[],
        eval_observation="eval obs",
    )
    system_prompt = agent.build_system_prompt(info)
    expected = {
        "role": "system",
        "content": json.dumps(
            {
                "Overall task": "some task",
                "Instructions": "some instruction",
                "Evaluation output of current code": "eval obs",
                "Shortcut features": ["f1", "f2"],
            },
            indent=2,
        ),
    }
    assert system_prompt == expected


def test_load_system_prompt_template_default_no_shortcuts_or_eval(
    agent_setup, build_env_info
):
    agent, env, _ = next(agent_setup(DebugAgent))
    env.get_tool = MagicMock(
        side_effect=KeyError("no tools for testing")
    )  # KeyError to simulate missing tool
    agent.system_prompt = "some task"
    agent.shortcut_features = Mock(return_value=[])
    info = build_env_info(
        instructions="some instruction",
        current_breakpoints=[1, 2],
        eval_observation="",
    )
    system_prompt = agent.build_system_prompt(info)
    expected = {
        "role": "system",
        "content": json.dumps(
            {
                "Overall task": "some task",
                "Instructions": "some instruction",
            },
            indent=2,
        ),
    }
    assert system_prompt == expected


def test_load_system_prompt_template_from_file(tmp_path, agent_setup):
    agent, _, _ = next(agent_setup(DebugAgent))
    agent.system_prompt = "test task"
    template_content = "Task: {{ agent.system_prompt }}"
    template_path = tmp_path / "template.jinja"
    template_path.write_text(template_content)
    agent.args.system_prompt_template_file = str(template_path)
    template = agent._load_system_prompt_template()
    assert isinstance(template, Template)
    assert template.render(agent=agent) == "Task: test task"


def test_load_system_prompt_template_file_not_found(agent_setup):
    agent, _, _ = next(agent_setup(DebugAgent))
    agent.args.system_prompt_template_file = "non_existent_template.jinja"
    with pytest.raises(FileNotFoundError):
        agent._load_system_prompt_template()


def test_build_system_prompt(agent_setup, build_env_info):
    agent, env, _ = next(agent_setup(DebugAgent))
    eval_tool = Toolbox.get_tool("eval", auto_eval_on_rewrite=True)
    pdb_tool = Toolbox.get_tool("pdb", auto_list=True, persistent_breakpoints=True)
    env.add_tool(eval_tool)
    env.add_tool(pdb_tool)
    env.workspace = MagicMock()
    env.workspace.display_files = MagicMock(return_value="repo/tree")
    agent.args.show_directory_tree = 2
    agent.args.show_current_breakpoints = True
    agent.system_prompt = "Test overall task"
    agent.env = env
    info = build_env_info(
        instructions="Do X",
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
            "updated automatically in the system prompt.",
            "The environment will show the directory tree of the repository in the system prompt.",
            "The environment will show the current breakpoints in the system prompt.",
            "The environment will automatically restore existing breakpoints when a new PDB session is started (e.g., after a rewrite).",
            "After every valid PDB tool calling, the environment will automatically call the PDB tool again with a `list .` command, which will show the code around the current frame.",
        ],
    }
    assert messages == {"role": "system", "content": json.dumps(expected, indent=2)}


def test_build_prompt(agent_setup, build_env_info):
    agent, _, _ = next(agent_setup(DebugAgent))
    info = build_env_info(
        instructions="Test instructions",
        current_breakpoints="Test breakpoints",
        step_observation="Test last run obs",
    )
    messages = agent.build_prompt(info)
    assert len(messages) > 0


def test_run(agent_setup, build_env_info):
    agent, env, llm = next(agent_setup(DebugAgent))
    env.reset.return_value = build_env_info(
        terminated=False,
        resolved=False,
        score=0,
        max_score=10,
        instructions="Test instructions",
        current_breakpoints="Test breakpoints",
        step_observation="Test last run obs",
    )
    env.step.return_value = build_env_info(
        terminated=True,
        resolved=True,
        score=10,
        max_score=10,
        instructions="Test instructions",
        current_breakpoints="Test breakpoints",
        step_observation="Test last run obs",
    )
    llm.return_value = LLMResponse("Prompt", "Expected answer", TokenUsage(2, 4))
    result = agent.run(env, debug=False)
    assert result


def test_build_system_prompt_rewrite_agent(agent_setup, build_env_info):
    agent, _, _ = next(agent_setup(RewriteAgent))
    info = build_env_info(
        instructions="Test instructions",
        current_breakpoints="Test breakpoints",
        step_observation="Test last run obs",
    )
    messages = agent.build_system_prompt(info)
    assert len(messages) == 2
    assert "Overall task" in messages["content"]


def test_run_debug_5_agent(agent_setup, build_env_info):
    agent, env, llm = next(agent_setup(Debug_5_Agent))
    env.reset.return_value = build_env_info(
        terminated=False,
        resolved=False,
        score=0,
        max_score=10,
        rewrite_counter=0,
        instructions="Test instructions",
        current_breakpoints="Test breakpoints",
        step_observation="Test last run obs",
    )
    env.step.return_value = build_env_info(
        terminated=True,
        resolved=True,
        score=10,
        max_score=10,
        rewrite_counter=0,
        instructions="Test instructions",
        current_breakpoints="Test breakpoints",
        step_observation="Test last run obs",
    )
    llm.return_value = LLMResponse("Prompt", "Expected answer", TokenUsage(2, 4))
    env.tools = {"pdb": MagicMock()}
    result = agent.run(env, debug=False)
    assert result


def test_shortcut_features_comprehensive(agent_setup):
    """Test all shortcut features combinations"""
    agent, env, _ = next(agent_setup(DebugAgent))
    eval_tool = Toolbox.get_tool("eval", auto_eval_on_rewrite=True)
    env.add_tool(eval_tool)
    # Test with all features enabled
    agent.args.show_directory_tree = 1
    agent.args.show_current_breakpoints = True
    env.has_tool.return_value = True

    features = agent.shortcut_features()
    assert len(features) == 5
    assert any("automatically call the Eval tool" in f for f in features)
    assert any("directory tree" in f for f in features)
    assert any("current breakpoints" in f for f in features)
    assert any("restore existing breakpoints" in f for f in features)
    assert any("list ." in f for f in features)  # Fixed to match actual text

    # Test with no PDB tool
    env.has_tool.return_value = False
    features = agent.shortcut_features()
    assert len(features) == 2  # Only auto_eval and directory_tree

    # Test with no features
    agent.args.show_directory_tree = 0
    agent.args.show_current_breakpoints = False
    env.get_tool("eval").auto_eval_on_rewrite = False
    env.get_tool("pdb").auto_list = False
    env.get_tool("pdb").persistent_breakpoints = False
    features = agent.shortcut_features()
    print(features)
    assert len(features) == 0


def test_trim_message(agent_setup):
    """Test message trimming functionality"""
    agent, _, llm = next(agent_setup(DebugAgent))
    llm.context_length = 1000
    llm.count_tokens = Mock(return_value=500)

    # Test with normal message (no trimming needed)
    message = "This is a test message"
    result = agent.trim_message(message, max_length=1000)
    assert result == message

    # Test with message that needs trimming
    llm.count_tokens.return_value = 1500  # Exceeds max_length
    result = agent.trim_message(message, max_length=1000)
    # The actual trim function returns "…" for short messages
    assert result == "…"

    # Test with percentage-based max_length
    llm.count_tokens.return_value = 600  # Exceeds 50% of 1000
    result = agent.trim_message(message, max_length_percentage=0.5)
    # Should use 50% of context_length (500)
    assert result == "…"

    # Test with no count_tokens function
    agent.llm.count_tokens = None
    result = agent.trim_message(message, count_tokens=None)
    assert result == message

    # Test with max_length <= 0
    result = agent.trim_message(message, max_length=0)
    assert result == message


def test_run_early_completion(agent_setup, build_env_info):
    """Test run method when task is already completed on reset"""
    agent, env, llm = next(agent_setup(DebugAgent))
    env.resolved = True

    # Mock environment to return completed task immediately
    env.reset.return_value = build_env_info(
        terminated=True,
        resolved=env.resolved,
        score=10,
        max_score=10,
        instructions="Test instructions",
        current_breakpoints="Test breakpoints",
        step_observation="Test last run obs",
    )

    result = agent.run(env)
    assert result["success"] is True
    env.step.assert_not_called()  # Should not step if already done


def test_run_max_rewrite_steps(agent_setup, build_env_info):
    """Test run method when max rewrite steps is reached"""
    agent, env, llm = next(agent_setup(DebugAgent))
    env.resolved = False
    agent.args.max_rewrite_steps = 2

    env.reset.return_value = build_env_info(
        terminated=False,
        resolved=False,
        score=0,
        max_score=10,
        rewrite_counter=0,
        instructions="Test instructions",
        current_breakpoints="Test breakpoints",
        step_observation="Test last run obs",
    )

    # First step - increase rewrite counter to max
    env.step.return_value = build_env_info(
        terminated=False,
        resolved=False,
        score=5,
        max_score=10,
        rewrite_counter=2,  # Reaches max_rewrite_steps
        instructions="Test instructions",
        current_breakpoints="Test breakpoints",
        step_observation="Test last run obs",
    )

    llm.return_value = LLMResponse("Prompt", "Expected answer", TokenUsage(2, 4))

    result = agent.run(env)
    assert (
        result["success"] is False
    )  # Task not completed, but stopped due to max rewrites


def test_run_exception_handling(agent_setup, build_env_info):
    """Test run method exception handling"""
    agent, env, llm = next(agent_setup(DebugAgent))

    env.reset.return_value = build_env_info(
        terminated=False,
        resolved=False,
        score=0,
        max_score=10,
        instructions="Test instructions",
        current_breakpoints="Test breakpoints",
        step_observation="Test last run obs",
    )

    # Make LLM raise an exception
    llm.side_effect = RuntimeError("Test error")

    with pytest.raises(RuntimeError, match="Test error"):
        agent.run(env)


def test_save_patch(agent_setup, tmp_path):
    """Test patch saving functionality"""
    agent, env, _ = next(agent_setup(DebugAgent))
    env.patch = "test patch content"
    logger = MagicMock()

    problem_path = tmp_path / "test_task"
    save_patch(env, problem_path, logger)

    patch_file = problem_path / "debug_gym.patch"
    assert patch_file.exists()
    assert patch_file.read_text() == "test patch content"


def test_build_trajectory(agent_setup, tmp_path):
    """Test trajectory building and persistence helpers"""
    agent, env, llm = next(agent_setup(DebugAgent))
    env.terminated = True
    env.resolved = True

    agent.args.uuid = "test-uuid-123"

    mock_tool = MagicMock()
    mock_tool.name = "test_tool"
    mock_tool.arguments = "test_args"
    env.tools = [mock_tool]

    class MockHistory:
        def __len__(self):
            return 2

        def json(self, step_id):
            return {"step": step_id, "action": f"test_action_{step_id}"}

    agent.history = MockHistory()

    agent.logger = MagicMock()
    agent.logger.log_file = "/tmp/test.log"
    llm.define_tools = lambda tools: [
        {"name": tool.name, "args": tool.arguments} for tool in tools
    ]

    trajectory = agent._build_trajectory()
    assert trajectory["problem"] == env.task_name
    assert trajectory["uuid"] == "test-uuid-123"
    assert len(trajectory["log"]) == 2
    assert trajectory["logger"] == "/tmp/test.log"
    assert trajectory["config"]["random_seed"] == agent.args.random_seed

    problem_path = tmp_path / "test_task"
    save_trajectory(agent, problem_path, MagicMock())

    trajectory_file = problem_path / "trajectory.json"
    assert trajectory_file.exists()

    saved = json.loads(trajectory_file.read_text())
    assert saved["problem"] == env.task_name
    assert saved["uuid"] == agent.args.uuid


def test_build_instance_prompt(agent_setup):
    """Test instance prompt building"""
    agent, _, _ = next(agent_setup(DebugAgent))

    # Test with action_prompt
    agent.action_prompt = "What should I do next?"
    messages = agent.build_instance_prompt()
    expected = [{"role": "user", "content": "What should I do next?"}]
    assert messages == expected

    # Test without action_prompt
    agent.action_prompt = None
    messages = agent.build_instance_prompt()
    assert messages == []


def test_load_system_prompt_template_with_filters(agent_setup, tmp_path):
    """Test system prompt template loading with custom filters"""
    agent, _, _ = next(agent_setup(DebugAgent))

    # Create template that uses custom filters
    template_content = """
{{ agent.system_prompt }}
{{ {"key": "value"} | to_pretty_json }}
{{ "long message that needs trimming" | trim_message(max_length=10) }}
"""
    template_file = tmp_path / "template.jinja"
    template_file.write_text(template_content)

    agent.args.system_prompt_template_file = str(template_file)
    agent.system_prompt = "Test task"

    template = agent._load_system_prompt_template()
    assert template is not None

    # Test that custom filters are available
    rendered = template.render(agent=agent)
    assert "Test task" in rendered
    assert '"key": "value"' in rendered


def test_set_seed(agent_setup):
    """Test seed setting functionality"""
    agent, _, _ = next(agent_setup(DebugAgent))

    with patch("numpy.random.seed") as mock_seed:
        agent.set_seed(42)
        mock_seed.assert_called_once_with(42)


@patch.object(
    LLMConfigRegistry,
    "from_file",
    return_value=LLMConfigRegistry.register_all(
        {
            "openai": {
                "model": "openai",
                "tokenizer": "gpt-4o",
                "context_limit": 4,
                "api_key": "test-api-key",
                "endpoint": "https://test-endpoint",
                "api_version": "v1",
                "tags": ["azure openai"],
            }
        }
    ),
)
def test_build_history_prompt(mock_llm_config, build_env_info):
    # test with empty history
    ht = HistoryTracker()
    llm = OpenAILLM("openai")
    messages = build_history_prompt(ht, llm)
    expected = []
    assert messages == expected

    # test with non-empty history
    ht = HistoryTracker()
    # prepare some data
    tool_1 = ToolCall(id="1", name="action1", arguments={"a1_args": "a1_args"})
    tool_2 = ToolCall(id="2", name="action2", arguments={"a2_args": "a2_args"})
    tool_3 = ToolCall(id="3", name="action3", arguments={})
    tool_4 = ToolCall(id="4", name="action4", arguments={"a4_args": "a4_args"})
    tool_5 = ToolCall(id="5", name="action5", arguments={})
    action_content_1 = "content_1_1"
    action_content_2 = "content_2_1"
    action_content_3 = "content_3_2"
    action_content_4 = "content_4_1"
    action_content_5 = "content_5_2"
    action_reasoning_1 = "reasoning_1_1"
    action_reasoning_2 = "reasoning_2_1"
    action_reasoning_3 = "reasoning_3_2"
    action_reasoning_4 = "reasoning_4_1"
    action_reasoning_5 = "reasoning_5_2"
    env_info_0 = build_env_info(
        step_observation="initial_obs",
        action_tool_call=None,
        action_reasoning=None,
        action_content=None,
        score=0,
        rewrite_counter=0,
    )
    env_info_1 = build_env_info(
        step_observation="obs1",
        action_tool_call=tool_1,
        action_reasoning=action_reasoning_1,
        action_content=action_content_1,
        score=1,
        rewrite_counter=0,
    )
    env_info_2 = build_env_info(
        step_observation="obs2",
        action_tool_call=tool_2,
        action_reasoning=action_reasoning_2,
        action_content=action_content_2,
        score=2,
        rewrite_counter=0,
    )
    env_info_3 = build_env_info(
        step_observation="obs3",
        action_tool_call=tool_3,
        action_reasoning=action_reasoning_3,
        action_content=action_content_3,
        score=3,
        rewrite_counter=1,
    )
    env_info_4 = build_env_info(
        step_observation="obs4",
        action_tool_call=tool_4,
        action_reasoning=action_reasoning_4,
        action_content=action_content_4,
        score=4,
        rewrite_counter=1,
    )
    env_info_5 = build_env_info(
        step_observation="obs5",
        action_tool_call=tool_5,
        action_reasoning=action_reasoning_5,
        action_content=action_content_5,
        score=5,
        rewrite_counter=2,
    )

    # single prompt format
    llm_response_1 = LLMResponse("prompt_1_1", "response_1_1", tool=tool_1)
    llm_response_2 = LLMResponse("prompt_2_1", "response_2_1", tool=tool_2)
    # list of messages format
    llm_response_3 = LLMResponse(
        prompt=[
            {"role": "user", "content": "prompt_3_1"},
            {"role": "assistent", "content": "response_3_1"},
            {"role": "user", "content": "prompt_3_2"},
        ],
        response="content_3_2",
        reasoning_response="reasoning_3_2",
        tool=tool_3,
    )
    llm_response_4 = LLMResponse(
        "prompt_4_1",
        "response_4_1",
        tool=tool_4,
        prompt_token_count=4321,
        response_token_count=1234,
    )
    llm_response_5 = LLMResponse(
        prompt=[
            {"role": "user", "content": "prompt_5_1"},
            {"role": "assistent", "content": "response_5_1"},
            {"role": "user", "content": "prompt_5_2"},
        ],
        response="content_5_2",
        reasoning_response="reasoning_5_2",
        tool=tool_5,
    )

    # push some steps and prompt-response pairs
    # at 0-th step, there is no prompt-response pair
    ht.init(None, None, env_info_0)
    ht.step(env_info_1, llm_response_1)
    ht.step(env_info_2, llm_response_2)
    ht.step(env_info_3, llm_response_3)
    ht.step(env_info_4, llm_response_4)
    ht.step(env_info_5, llm_response_5)

    # reset_prompt_history_after_rewrite is False
    messages = build_history_prompt(
        ht, llm, reset_prompt_history_after_rewrite=False, history_cutoff=3
    )
    expected = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "type": "function",
                    "id": "2",
                    "function": {
                        "name": "action2",
                        "arguments": json.dumps({}),
                    },
                },
            ],
            "content": "response_2",
        },
        {
            "role": "tool",
            "tool_call_id": "2",
            "name": "action2",
            "content": "obs3",
        },
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "type": "function",
                    "id": "3",
                    "function": {
                        "name": "action3",
                        "arguments": json.dumps({}),
                    },
                },
            ],
            "content": "response_3",
        },
        {
            "role": "tool",
            "tool_call_id": "3",
            "name": "action3",
            "content": "obs4",
        },
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "type": "function",
                    "id": "4",
                    "function": {
                        "name": "action4",
                        "arguments": json.dumps({"a4_args": "a4_args"}),
                    },
                },
            ],
            "content": "response_4",
        },
        {
            "role": "tool",
            "tool_call_id": "4",
            "name": "action4",
            "content": "obs5",
        },
    ]

    assert messages == expected

    # reset_prompt_history_after_rewrite is True
    messages = build_history_prompt(
        ht, llm, reset_prompt_history_after_rewrite=True, history_cutoff=3
    )
    expected = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "type": "function",
                    "id": "3",
                    "function": {
                        "name": "action3",
                        "arguments": json.dumps({}),
                    },
                },
            ],
            "content": "response_3",
        },
        {
            "role": "tool",
            "tool_call_id": "3",
            "name": "action3",
            "content": "obs4",
        },
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "type": "function",
                    "id": "4",
                    "function": {
                        "name": "action4",
                        "arguments": json.dumps({"a4_args": "a4_args"}),
                    },
                },
            ],
            "content": "response_4",
        },
        {
            "role": "tool",
            "tool_call_id": "4",
            "name": "action4",
            "content": "obs5",
        },
    ]
    assert messages == expected
