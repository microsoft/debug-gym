import logging
from unittest.mock import patch

from debug_gym.agents.utils import BaseConfig, load_config


def test_load_config():
    import atexit
    import tempfile
    from pathlib import Path

    import yaml

    # do the test in a tmp folder
    tempdir = tempfile.TemporaryDirectory(prefix="TestLoadConfig-")
    working_dir = Path(tempdir.name)
    config_file = working_dir / "config.yaml"
    atexit.register(tempdir.cleanup)  # Make sure to cleanup that folder once done.

    config_contents = {}
    config_contents["base"] = {
        "random_seed": 42,
        "max_steps": 100,
    }
    config_contents["pdb_agent"] = {
        "llm_name": "gpt2",
    }
    config_contents["rewrite_only"] = {
        "cot_style": "standard",
        "llm_name": "gpt20",
    }

    # write the config file into yaml
    with open(config_file, "w") as f:
        yaml.dump(config_contents, f)

    # now test
    with patch(
        "sys.argv",
        [
            "config_file",
            str(config_file),
            "--agent",
            "pdb_agent",
            "-p",
            "base.random_seed=123",
            "-v",
            "--debug",
        ],
    ):
        _config, _args = load_config()
    assert _args.agent == "pdb_agent"
    expected_config = {
        "agent_type": "pdb_agent",
        "random_seed": 123,
        "max_steps": 100,
        "llm_name": "gpt2",
    }
    assert _config == expected_config
    assert _args.debug is True
    assert _args.logging_level == logging.INFO

    # another test
    with patch(
        "sys.argv",
        [
            "config_file",
            str(config_file),
            "--agent",
            "rewrite_only",
            "-p",
            "base.random_seed=123",
            "rewrite_only.random_seed=456",
            "-v",
            "--debug",
        ],
    ):
        _config, _args = load_config()
    assert _args.agent == "rewrite_only"
    expected_config = {
        "agent_type": "rewrite_only",
        "random_seed": 456,
        "max_steps": 100,
        "cot_style": "standard",
        "llm_name": "gpt20",
    }
    assert _config == expected_config
    assert _args.debug is True
    assert _args.logging_level == logging.INFO


def test_load_config_without_yaml_file():
    """Test that config can be loaded using only CLI arguments without a YAML file."""
    # Test with no config file - should use defaults
    with patch(
        "sys.argv",
        [
            "run.py",
            "--agent",
            "debug_agent",
        ],
    ):
        _config, _args = load_config()

    assert _args.agent == "debug_agent"
    assert _config["agent_type"] == "debug_agent"
    # Check that BaseConfig defaults are used
    assert _config["random_seed"] == 42
    assert _config["max_steps"] == 50
    assert _config["llm_name"] == "gpt-4o"
    assert _config["output_path"] == "exps/default"


def test_load_config_with_cli_overrides():
    """Test that CLI arguments override config file values."""
    import atexit
    import tempfile
    from pathlib import Path

    import yaml

    tempdir = tempfile.TemporaryDirectory(prefix="TestCLIOverrides-")
    working_dir = Path(tempdir.name)
    config_file = working_dir / "config.yaml"
    atexit.register(tempdir.cleanup)

    config_contents = {
        "base": {
            "random_seed": 42,
            "max_steps": 100,
            "llm_name": "gpt-4o",
            "output_path": "exps/original",
        },
        "rewrite_agent": {
            "tools": ["view", "rewrite"],
        },
    }

    with open(config_file, "w") as f:
        yaml.dump(config_contents, f)

    # Test CLI overrides
    with patch(
        "sys.argv",
        [
            "run.py",
            str(config_file),
            "--agent",
            "rewrite_agent",
            "--llm-name",
            "claude-3-sonnet",
            "--max-steps",
            "200",
            "--output-path",
            "exps/custom",
        ],
    ):
        _config, _args = load_config()

    assert _config["llm_name"] == "claude-3-sonnet"
    assert _config["max_steps"] == 200
    assert _config["output_path"] == "exps/custom"
    # Non-overridden values should come from config file
    assert _config["random_seed"] == 42


def test_load_config_cli_only_all_options():
    """Test using all CLI options without a config file."""
    with patch(
        "sys.argv",
        [
            "run.py",
            "--agent",
            "debug_agent",
            "--output-path",
            "exps/test",
            "--benchmark",
            "mini_nightmare",
            "--problems",
            "problem1,problem2",
            "--llm-name",
            "gpt-4",
            "--random-seed",
            "123",
            "--max-steps",
            "30",
            "--max-rewrite-steps",
            "5",
            "--memory-size",
            "10",
            "--terminal-type",
            "docker",
            "--tools",
            "pdb,view,rewrite",
        ],
    ):
        _config, _args = load_config()

    assert _config["output_path"] == "exps/test"
    assert _config["benchmark"] == "mini_nightmare"
    assert _config["problems"] == ["problem1", "problem2"]
    assert _config["llm_name"] == "gpt-4"
    assert _config["random_seed"] == 123
    assert _config["max_steps"] == 30
    assert _config["max_rewrite_steps"] == 5
    assert _config["memory_size"] == 10
    assert _config["terminal"]["type"] == "docker"
    assert _config["tools"] == ["pdb", "view", "rewrite"]


def test_load_config_problems_all():
    """Test that 'all' problems is handled correctly."""
    with patch(
        "sys.argv",
        [
            "run.py",
            "--problems",
            "all",
        ],
    ):
        _config, _args = load_config()

    assert _config["problems"] == "all"


def test_load_config_env_kwargs():
    """Test that env_kwargs can be passed as JSON."""
    with patch(
        "sys.argv",
        [
            "run.py",
            "--env-kwargs",
            '{"run_timeout": 60, "dataset_id": "test-dataset"}',
        ],
    ):
        _config, _args = load_config()

    assert _config["env_kwargs"]["run_timeout"] == 60
    assert _config["env_kwargs"]["dataset_id"] == "test-dataset"


def test_load_config_no_save_patch():
    """Test --no-save-patch flag."""
    with patch(
        "sys.argv",
        [
            "run.py",
            "--no-save-patch",
        ],
    ):
        _config, _args = load_config()

    assert _config["save_patch"] is False


def test_load_config_default_agent():
    """Test that default agent is set when no config file and no --agent is specified."""
    with patch(
        "sys.argv",
        [
            "run.py",
        ],
    ):
        _config, _args = load_config()

    assert _args.agent == "rewrite_agent"
    assert _config["agent_type"] == "rewrite_agent"


def test_base_config_dataclass():
    """Test BaseConfig dataclass methods."""
    config = BaseConfig()

    # Test defaults
    assert config.random_seed == 42
    assert config.max_steps == 50
    assert config.llm_name == "gpt-4o"

    # Test to_dict
    config_dict = config.to_dict()
    assert isinstance(config_dict, dict)
    assert config_dict["random_seed"] == 42

    # Test from_dict
    new_config = BaseConfig.from_dict({"random_seed": 123, "unknown_field": "ignored"})
    assert new_config.random_seed == 123
    assert new_config.max_steps == 50  # Should use default

    # Test update
    config.update({"max_steps": 100})
    assert config.max_steps == 100
