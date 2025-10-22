import logging
from unittest.mock import patch

from debug_gym.agents.utils import load_config


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
