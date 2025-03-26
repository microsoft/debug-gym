import os
from pathlib import Path

import pytest

from debug_gym.entrypoints import copy_llm_config_template


@pytest.fixture
def mock_argv(monkeypatch):
    """Fixture to mock sys.argv with different values in tests."""

    def _set_argv(args):
        monkeypatch.setattr("sys.argv", ["copy_llm_config_template"] + args)

    return _set_argv


def test_copy_llm_config_template_dest_default(
    tmp_path, mock_argv, monkeypatch, capsys
):
    expected_path = Path(tmp_path) / ".config" / "debug_gym"
    # Mock home directory to use tmp_path
    monkeypatch.setattr(Path, "home", lambda: Path(tmp_path))
    mock_argv([])
    copy_llm_config_template()
    template_file = expected_path / "llm.template.yaml"
    assert template_file.exists()
    assert template_file.read_text().startswith("gpt-4o:\n  model: gpt-4o")
    assert f"LLM config template created" in capsys.readouterr().out


def test_copy_llm_config_template_with_dest_positional(tmp_path, mock_argv, capsys):
    mock_argv([str(tmp_path)])
    copy_llm_config_template()
    template_path = tmp_path / "llm.template.yaml"
    assert template_path.exists()
    assert template_path.read_text().startswith("gpt-4o:\n  model: gpt-4o")
    assert f"LLM config template created" in capsys.readouterr().out


def test_copy_llm_config_template_with_dest_named(tmp_path, mock_argv, capsys):
    mock_argv(["--dest", str(tmp_path)])
    copy_llm_config_template()
    template_path = tmp_path / "llm.template.yaml"
    assert template_path.exists()
    assert template_path.read_text().startswith("gpt-4o:\n  model: gpt-4o")
    assert f"LLM config template created" in capsys.readouterr().out


def test_copy_llm_config_template_override(tmp_path, monkeypatch, mock_argv, capsys):

    monkeypatch.setattr("importlib.resources.files", lambda _: tmp_path)
    source = tmp_path / "llm.template.yaml"
    source.write_text("initial content")

    destination = tmp_path / "destination"
    os.makedirs(destination, exist_ok=True)
    destination_file = destination / "llm.template.yaml"

    mock_argv(["--dest", str(destination)])

    copy_llm_config_template()  # First copy should work
    assert destination_file.read_text() == "initial content"
    assert f"LLM config template created" in capsys.readouterr().out

    source.write_text("new content")
    copy_llm_config_template()  # No force, should not override
    assert destination_file.read_text() == "initial content"
    assert f"LLM config template already exists" in capsys.readouterr().out

    mock_argv(["--dest", str(destination), "--force"])
    copy_llm_config_template()  # Force override
    assert destination_file.read_text() == "new content"
    assert f"LLM config template overridden" in capsys.readouterr().out
