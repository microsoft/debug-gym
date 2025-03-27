from pathlib import Path

import pytest

from debug_gym.agents.llm_api import LLM_CONFIG_TEMPLATE
from debug_gym.init_llm_config import init_llm_config


@pytest.fixture
def mock_argv(monkeypatch):
    """Fixture to mock sys.argv with different values in tests."""

    def _set_argv(args):
        monkeypatch.setattr("sys.argv", ["init_llm_config"] + args)

    return _set_argv


def test_init_llm_config_dest_default(tmp_path, mock_argv, monkeypatch, capsys):
    expected_path = Path(tmp_path) / ".config" / "debug_gym"
    # Mock home directory to use tmp_path
    monkeypatch.setattr(Path, "home", lambda: Path(tmp_path))
    mock_argv([])
    init_llm_config()
    template_file = expected_path / "llm.yaml"
    assert template_file.exists()
    assert template_file.read_text() == LLM_CONFIG_TEMPLATE
    assert f"LLM config template created" in capsys.readouterr().out


def test_init_llm_config_with_dest_positional(tmp_path, mock_argv, capsys):
    mock_argv([str(tmp_path)])
    init_llm_config()
    template_path = tmp_path / "llm.yaml"
    assert template_path.exists()
    assert template_path.read_text() == LLM_CONFIG_TEMPLATE
    assert f"LLM config template created" in capsys.readouterr().out


def test_init_llm_config_with_dest_named(tmp_path, mock_argv, capsys):
    mock_argv(["--dest", str(tmp_path)])
    init_llm_config()
    template_path = tmp_path / "llm.yaml"
    assert template_path.exists()
    assert template_path.read_text() == LLM_CONFIG_TEMPLATE
    assert f"LLM config template created" in capsys.readouterr().out


def test_init_llm_config_override(tmp_path, monkeypatch, mock_argv, capsys):
    llm_template_path = "debug_gym.init_llm_config.LLM_CONFIG_TEMPLATE"
    monkeypatch.setattr(llm_template_path, "config")

    destination = tmp_path / "destination"
    # os.makedirs(destination, exist_ok=True)
    destination_file = destination / "llm.yaml"

    mock_argv(["--dest", str(destination)])
    init_llm_config()  # First copy should work
    assert destination_file.read_text() == "config"
    assert f"LLM config template created" in capsys.readouterr().out

    monkeypatch.setattr(llm_template_path, "new config")
    init_llm_config()  # No force, should not override
    assert destination_file.read_text() == "config"
    assert f"LLM config template already exists" in capsys.readouterr().out

    mock_argv(["--dest", str(destination), "--force"])
    init_llm_config()  # Force override
    assert destination_file.read_text() == "new config"
    assert f"LLM config template overridden" in capsys.readouterr().out
