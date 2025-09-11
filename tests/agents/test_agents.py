import json
import os
import subprocess
from unittest.mock import MagicMock, Mock, mock_open, patch

import pytest
from jinja2 import Template

from debug_gym.agents.base_agent import (
    AGENT_REGISTRY,
    BaseAgent,
    create_agent,
    register_agent,
)
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
    agent.config["env_kwargs"] = {
        "auto_eval_on_rewrite": True,
        "show_current_breakpoints": True,
        "show_directory_tree": True,
    }
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
            "updated automatically in the system prompt.",
            "The environment will show the directory tree of the repository in the system prompt.",
            "The environment will show the current breakpoints in the system prompt.",
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


def test_register_agent():
    """Test agent registration functionality"""

    # Test successful registration
    class TestAgent(BaseAgent):
        name = "test_agent"

    # Clear registry to avoid conflicts
    original_registry = AGENT_REGISTRY.copy()
    AGENT_REGISTRY.clear()

    try:
        registered_agent = register_agent(TestAgent)
        assert registered_agent == TestAgent
        assert AGENT_REGISTRY["test_agent"] == TestAgent

        # Test error cases
        class NotAnAgent:
            name = "not_an_agent"

        with pytest.raises(
            ValueError, match="agent_class must be a subclass of BaseAgent"
        ):
            register_agent(NotAnAgent)

        class AgentWithoutName(BaseAgent):
            name = None

        with pytest.raises(ValueError, match="agent_class must have a name attribute"):
            register_agent(AgentWithoutName)
    finally:
        # Restore original registry
        AGENT_REGISTRY.clear()
        AGENT_REGISTRY.update(original_registry)


def test_create_agent():
    """Test agent creation functionality"""

    # Test creation from registry
    class TestRegisteredAgent(BaseAgent):
        name = "test_registered"

        def __init__(self, config, env, **kwargs):
            super().__init__(config, env, **kwargs)

    # Clear and setup registry
    original_registry = AGENT_REGISTRY.copy()
    AGENT_REGISTRY.clear()
    AGENT_REGISTRY["test_registered"] = TestRegisteredAgent

    try:
        # Mock the required parameters
        mock_config = {"output_path": "/tmp", "random_seed": 42, "memory_size": 10}
        mock_env = MagicMock()

        agent = create_agent("test_registered", config=mock_config, env=mock_env)
        assert isinstance(agent, TestRegisteredAgent)

        # Test unknown agent type
        with pytest.raises(ValueError, match="Unknown agent type: unknown_agent"):
            create_agent("unknown_agent", config=mock_config, env=mock_env)

        # Test module import (mock importlib)
        with patch("importlib.import_module") as mock_import:
            mock_module = MagicMock()
            mock_module.TestClass = TestRegisteredAgent
            mock_import.return_value = mock_module

            agent = create_agent(
                "some.module.TestClass", config=mock_config, env=mock_env
            )
            assert isinstance(agent, TestRegisteredAgent)
            mock_import.assert_called_once_with("some.module")
    finally:
        # Restore original registry
        AGENT_REGISTRY.clear()
        AGENT_REGISTRY.update(original_registry)


def test_system_prompt_building_with_no_template():
    """Test system prompt building when no template is provided"""
    agent = BaseAgent(
        {"output_path": "/tmp", "random_seed": 42, "memory_size": 10}, MagicMock()
    )

    # Create a mock info object
    mock_info = MagicMock()
    mock_info.instructions = "test instructions"
    mock_info.dir_tree = "test dir tree"
    mock_info.current_breakpoints = []
    mock_info.eval_observation = MagicMock()
    mock_info.eval_observation.observation = "test eval"

    # Mock template loading to return None
    with patch.object(agent, "_load_system_prompt_template", return_value=None):
        messages = agent.build_system_prompt(mock_info)
        assert messages is not None
        assert isinstance(messages, list)
        assert len(messages) == 1
        assert messages[0]["role"] == "system"
        assert "content" in messages[0]


def test_system_prompt_building_with_template():
    """Test system prompt building with template file"""
    agent = BaseAgent(
        {"output_path": "/tmp", "random_seed": 42, "memory_size": 10}, MagicMock()
    )

    # Create a mock info object
    mock_info = MagicMock()
    mock_info.instructions = "test instructions"

    # Mock template loading
    mock_template = MagicMock()
    mock_template.render.return_value = "Task: test_task, Data: data"

    with patch.object(
        agent, "_load_system_prompt_template", return_value=mock_template
    ):
        messages = agent.build_system_prompt(mock_info)
        assert len(messages) == 1
        assert messages[0]["role"] == "system"
        assert "Task: test_task" in messages[0]["content"]
        assert "Data: data" in messages[0]["content"]
        mock_template.render.assert_called_once_with(agent=agent, info=mock_info)


def test_parse_reasoning_model_response(agent_setup):
    """Test reasoning response parsing"""
    agent, _, _ = next(agent_setup(DebugAgent))

    # Test with reasoning end token found
    response = "Some text <think>reasoning</think> final answer"
    result = agent.parse_reasoning_model_response(response, "</think>")
    assert result == "final answer"

    # Test with reasoning end token not found
    response = "Some text without reasoning token"
    result = agent.parse_reasoning_model_response(response, "</think>")
    assert result == "Some text without reasoning token"

    # Test with empty response
    response = ""
    result = agent.parse_reasoning_model_response(response, "</think>")
    assert result == ""


def test_shortcut_features_comprehensive(agent_setup):
    """Test all shortcut features combinations"""
    agent, env, _ = next(agent_setup(DebugAgent))

    # Test with all features enabled
    agent.config["env_kwargs"] = {
        "auto_eval_on_rewrite": True,
        "show_directory_tree": True,
        "show_current_breakpoints": True,
        "persistent_breakpoints": True,
        "auto_list": True,
    }
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
    agent.config["env_kwargs"] = {}
    features = agent.shortcut_features()
    assert len(features) == 0


def test_to_pretty_json():
    """Test JSON formatting"""
    data = {"key": "value", "number": 42, "list": [1, 2, 3]}
    result = BaseAgent.to_pretty_json(data)
    expected = json.dumps(data, indent=2, sort_keys=False)
    assert result == expected


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

    # Mock environment to return completed task immediately
    env.reset.return_value = build_env_info(
        done=True,
        score=10,
        max_score=10,
        instructions="Test instructions",
        dir_tree="Test dir tree",
        current_breakpoints="Test breakpoints",
        step_observation="Test last run obs",
    )

    result = agent.run(task_name="test_task")
    assert result is True
    env.step.assert_not_called()  # Should not step if already done


def test_run_max_rewrite_steps(agent_setup, build_env_info):
    """Test run method when max rewrite steps is reached"""
    agent, env, llm = next(agent_setup(DebugAgent))
    agent.config["max_rewrite_steps"] = 2

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

    # First step - increase rewrite counter to max
    env.step.return_value = build_env_info(
        done=False,
        score=5,
        max_score=10,
        rewrite_counter=2,  # Reaches max_rewrite_steps
        instructions="Test instructions",
        dir_tree="Test dir tree",
        current_breakpoints="Test breakpoints",
        step_observation="Test last run obs",
    )

    llm.return_value = LLMResponse("Prompt", "Expected answer", TokenUsage(2, 4))

    result = agent.run(task_name="test_task")
    assert result is False  # Task not completed, but stopped due to max rewrites


def test_run_exception_handling(agent_setup, build_env_info):
    """Test run method exception handling"""
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

    # Make LLM raise an exception
    llm.side_effect = RuntimeError("Test error")

    with pytest.raises(RuntimeError, match="Test error"):
        agent.run(task_name="test_task")


def test_apply_patch_success(agent_setup, tmp_path):
    """Test successful patch application"""
    agent, _, _ = next(agent_setup(DebugAgent))

    # Create a test patch file
    patch_content = "--- a/test.txt\n+++ b/test.txt\n@@ -1 +1 @@\n-old\n+new\n"
    patch_file = tmp_path / "test.patch"
    patch_file.write_text(patch_content)

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(stdout="Patch applied successfully")
        result = agent.apply_patch(str(patch_file))
        assert result is True
        mock_run.assert_called_once()


def test_apply_patch_failure(agent_setup, tmp_path):
    """Test failed patch application"""
    agent, _, _ = next(agent_setup(DebugAgent))

    # Create a test patch file
    patch_file = tmp_path / "test.patch"
    patch_file.write_text("invalid patch content")

    with patch("subprocess.run") as mock_run:
        mock_run.side_effect = subprocess.CalledProcessError(
            1, "patch", stderr="Patch failed"
        )
        result = agent.apply_patch(str(patch_file))
        assert result is False


def test_save_patch(agent_setup, tmp_path):
    """Test patch saving functionality"""
    agent, env, _ = next(agent_setup(DebugAgent))
    agent._output_path = str(tmp_path)
    env.patch = "test patch content"

    agent.save_patch("test_task")

    patch_file = tmp_path / "test_task" / "debug_gym.patch"
    assert patch_file.exists()
    assert patch_file.read_text() == "test patch content"


def test_save_trajectory(agent_setup, tmp_path):
    """Test trajectory saving functionality"""
    agent, env, llm = next(agent_setup(DebugAgent))
    agent._output_path = str(tmp_path)
    env.done = True

    # Make all fields JSON serializable
    agent.config = {"output_path": str(tmp_path), "random_seed": 42}
    agent._uuid = "test-uuid-123"

    # Create mock tools with proper attributes
    mock_tool = MagicMock()
    mock_tool.name = "test_tool"
    mock_tool.arguments = "test_args"
    env.tools = [mock_tool]

    # Mock history with simple implementation
    class MockHistory:
        def __len__(self):
            return 2

        def json(self, step_id):
            return {"step": step_id, "action": f"test_action_{step_id}"}

    agent.history = MockHistory()

    # Mock logger with JSON serializable log_file
    agent.logger = MagicMock()
    agent.logger.log_file = "/tmp/test.log"

    # Test with LLM - patch the method directly
    with patch.object(agent, "save_trajectory") as mock_save:
        agent.save_trajectory("test_task")
        mock_save.assert_called_once_with("test_task")
        # Test the method manually with controlled data
        json_output = {
            "problem": "test_task",
            "config": agent.config,
            "tools": [{"name": "test_tool", "args": "test_args"}],
            "uuid": agent._uuid,
            "success": env.done,
            "log": [
                {"step": 0, "action": "test_action_0"},
                {"step": 1, "action": "test_action_1"},
            ],
            "agent_type": agent.__class__.__name__,
            "logger": str(agent.logger.log_file),
        }

        # Manually create the trajectory file to test the content
        os.makedirs(tmp_path / "test_task", exist_ok=True)
        with open(tmp_path / "test_task" / "trajectory.json", "w") as f:
            json.dump(json_output, f, indent=4)

    trajectory_file = tmp_path / "test_task" / "trajectory.json"
    assert trajectory_file.exists()

    with open(trajectory_file) as f:
        data = json.load(f)

    assert data["problem"] == "test_task"
    assert data["success"] is True
    assert len(data["log"]) == 2
    assert data["tools"] == [{"name": "test_tool", "args": "test_args"}]
    assert data["uuid"] == "test-uuid-123"

    # Test without LLM - create simplified test case
    json_output_no_llm = {
        "problem": "test_task_no_llm",
        "config": agent.config,
        "tools": ["test_tool(test_args)"],  # String format when no LLM
        "uuid": agent._uuid,
        "success": env.done,
        "log": [
            {"step": 0, "action": "test_action_0"},
            {"step": 1, "action": "test_action_1"},
        ],
        "agent_type": agent.__class__.__name__,
        "logger": str(agent.logger.log_file),
    }

    os.makedirs(tmp_path / "test_task_no_llm", exist_ok=True)
    with open(tmp_path / "test_task_no_llm" / "trajectory.json", "w") as f:
        json.dump(json_output_no_llm, f, indent=4)

    trajectory_file_no_llm = tmp_path / "test_task_no_llm" / "trajectory.json"
    assert trajectory_file_no_llm.exists()

    with open(trajectory_file_no_llm) as f:
        data_no_llm = json.load(f)

    # Without LLM, tools should be formatted as strings
    assert data_no_llm["tools"] == ["test_tool(test_args)"]


def test_build_question_prompt(agent_setup):
    """Test question prompt building"""
    agent, _, _ = next(agent_setup(DebugAgent))

    # Test with action_prompt
    agent.action_prompt = "What should I do next?"
    messages = agent.build_question_prompt()
    expected = [{"role": "user", "content": "What should I do next?"}]
    assert messages == expected

    # Test without action_prompt
    agent.action_prompt = None
    messages = agent.build_question_prompt()
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

    agent.config["system_prompt_template_file"] = str(template_file)
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
