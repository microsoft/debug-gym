from unittest.mock import MagicMock, patch

import pytest

from debug_gym.agents.base_agent import BaseAgent
from debug_gym.agents.free_agent import FreeAgent


@pytest.fixture
def make_free_agent(agent_setup):
    def _factory(*, config_override=None):
        agent, env, llm = next(agent_setup(FreeAgent, config_override=config_override))
        agent.logger = MagicMock()
        return agent, env, llm

    return _factory


def test_free_agent_run_delegates_to_base(make_free_agent):
    agent, _, _ = make_free_agent()

    with patch.object(BaseAgent, "run", return_value=True) as mock_run:
        result = agent.run(task_name="demo", debug=True)

    mock_run.assert_called_once_with(task_name="demo", debug=True)
    assert result is True


def test_free_agent_reraises_root_cause_for_missing_reset(make_free_agent):
    agent, _, _ = make_free_agent()

    def side_effect(*args, **kwargs):
        try:
            raise RuntimeError("reset failed")
        except RuntimeError as exc:  # pragma: no cover - exercised below
            raise AttributeError(
                "'NoneType' object has no attribute 'max_score'"
            ) from exc

    with patch.object(BaseAgent, "run", side_effect=side_effect):
        with pytest.raises(RuntimeError) as excinfo:
            agent.run(task_name="demo")

    assert str(excinfo.value) == "reset failed"
    agent.logger.error.assert_called_once()


def test_free_agent_bubbles_unrelated_attribute_error(make_free_agent):
    agent, _, _ = make_free_agent()

    with patch.object(BaseAgent, "run", side_effect=AttributeError("other")):
        with pytest.raises(AttributeError, match="other"):
            agent.run(task_name="demo")

    agent.logger.error.assert_not_called()


def test_free_agent_system_prompt_override(make_free_agent):
    custom_prompt = "Inspect quietly."
    agent, _, _ = make_free_agent(config_override={"system_prompt": custom_prompt})

    assert agent.system_prompt == custom_prompt
