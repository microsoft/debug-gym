from jinja2 import Template

from debug_gym.agents.debug_agent import DebugAgent


def test_load_system_prompt_template_default(agent_setup):
    agent, _, _ = next(agent_setup(DebugAgent))
    template = agent._load_system_prompt_template()
    assert isinstance(template, Template)
    assert template.render(
        overall_task="some task",
        instructions="some instruction",
        repo_dir_tree="dir tree",
        current_breakpoints=[],
        eval_observation="eval obs",
        shortcut_features=["f1", "f2"],
    ) == (
        "{\n"
        '    "Overall task": "some task",\n'
        '    "Instructions": "some instruction",\n'
        '    "Repo directory tree": "dir tree",\n'
        '    "Current breakpoints": [],\n'
        '    "Eval observation": "eval obs",\n'
        "    \"Shortcut features\": ['f1', 'f2']\n"
        "}"
    )


def test_load_system_prompt_template_default_no_conditional(agent_setup):
    agent, _, _ = next(agent_setup(DebugAgent))
    template = agent._load_system_prompt_template()
    assert isinstance(template, Template)
    assert template.render(
        overall_task="some task",
        instructions="some instruction",
        repo_dir_tree="dir tree",
        current_breakpoints=[1, 2],
    ) == (
        "{\n"
        '    "Overall task": "some task",\n'
        '    "Instructions": "some instruction",\n'
        '    "Repo directory tree": "dir tree",\n'
        '    "Current breakpoints": [1, 2]\n'
        "}"
    )


def test_load_system_prompt_template_from_file(agent_setup, tmp_path):
    agent, _, _ = next(agent_setup(DebugAgent))
    template_content = "Task: {{ overall_task }}"
    template_path = tmp_path / "template.jinja"
    template_path.write_text(template_content)
    agent.config["system_prompt_template"] = str(template_path)
    template = agent._load_system_prompt_template()
    assert isinstance(template, Template)
    assert template.render(overall_task="test task") == "Task: test task"


def test_load_system_prompt_template_from_plain_text(agent_setup, tmp_path):
    agent, _, _ = next(agent_setup(DebugAgent))
    template_content = "Task: {{ overall_task }}"
    agent.config["system_prompt_template"] = template_content
    template = agent._load_system_prompt_template()
    assert isinstance(template, Template)
    assert template.render(overall_task="test task") == "Task: test task"


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
    assert messages == [
        {
            "role": "system",
            "content": (
                "{\n"
                '    "Overall task": "Test overall task",\n'
                '    "Instructions": "Do X",\n'
                '    "Repo directory tree": "repo/tree",\n'
                '    "Current breakpoints": [1, 2],\n'
                '    "Eval observation": "eval obs",\n'
                '    "Shortcut features": ["After successful rewrites, the environment will automatically call the Eval tool to evaluate the rewritten code. Therefore, you do not need to call the Eval tool yourself. The evaluation output will be updated automatically in the system prompt."]\n'
                "}"
            ),
        }
    ]
