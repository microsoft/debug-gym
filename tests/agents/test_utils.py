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
    config_file = str(working_dir / "config.yaml")
    atexit.register(tempdir.cleanup)  # Make sure to cleanup that folder once done.

    config_contents = {
        "agent": {
            "max_steps": 100,
            "type": "pdb_agent",
        },
        "llm": {"name": "gpt2"},
    }

    # write the config file into yaml
    with open(config_file, "w") as f:
        yaml.dump(config_contents, f)

    # now test
    args = [
        "--config",
        config_file,
        "-v",
        "--debug",
    ]
    _config, _args = load_config(args)
    expected_config = {
        "agent": {
            "type": "pdb_agent",
            "max_steps": 100,
        },
        "llm": {"name": "gpt2"},
    }
    assert _config == expected_config
    assert _args.debug is True
    assert _args.logging_level == logging.INFO

    # another test
    args = [
        "--config",
        config_file,
        "-p",
        "agent.type=edit_only",
        "random_seed=456",
        "cot_style=standard",
        "llm.name=gpt20",
        "-v",
        "--debug",
    ]
    _config, _args = load_config(args)
    expected_config = {
        "agent": {
            "type": "edit_only",
            "max_steps": 100,
        },
        "random_seed": 456,
        "cot_style": "standard",
        "llm": {"name": "gpt20"},
    }
    assert _config == expected_config
    assert _args.debug is True
    assert _args.logging_level == logging.INFO
