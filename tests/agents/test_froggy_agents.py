import json
from unittest.mock import MagicMock, Mock

import pytest

from debug_gym.agents.debug_agent import DebugAgent
from debug_gym.agents.rewrite_agent import RewriteAgent
from debug_gym.agents.utils import save_patch, save_trajectory
from debug_gym.gym.tools.toolbox import Toolbox
from debug_gym.llms.base import LLMResponse, TokenUsage


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
        eval_observation="",
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


def test_default_system_prompt_with_eval_output(agent_setup, build_env_info):
    agent, env, _ = next(agent_setup(DebugAgent))
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


def test_build_system_prompt(agent_setup, build_env_info):
    agent, env, _ = next(agent_setup(DebugAgent))
    eval_tool = Toolbox.get_tool("eval")
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


def test_shortcut_features_comprehensive(agent_setup):
    """Test all shortcut features combinations"""
    agent, env, _ = next(agent_setup(DebugAgent))
    # Test with all features enabled
    agent.args.show_directory_tree = 1
    agent.args.show_current_breakpoints = True
    env.has_tool.return_value = True

    features = agent.shortcut_features()
    assert len(features) == 3
    assert any("current breakpoints" in f for f in features)
    assert any("restore existing breakpoints" in f for f in features)
    assert any("list ." in f for f in features)  # Fixed to match actual text

    # Test with no PDB tool
    env.has_tool.return_value = False
    features = agent.shortcut_features()
    assert len(features) == 0

    # Test with no features
    agent.args.show_directory_tree = 0
    agent.args.show_current_breakpoints = False
    env.has_tool.return_value = True
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

    problem_path = tmp_path / "test_task"
    save_trajectory(agent, problem_path, MagicMock())

    trajectory_file = problem_path / "trajectory.json"
    assert trajectory_file.exists()

    saved = json.loads(trajectory_file.read_text())
    assert saved["problem"] == env.task_name
    assert saved["uuid"] == agent.args.uuid
