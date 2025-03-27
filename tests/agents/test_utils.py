import logging
from unittest.mock import patch

import pytest

from debug_gym.agents.utils import (
    load_config,
    merge_messages,
    print_messages,
    trim,
    trim_prompt_messages,
)


def test_print_messages(logger_mock):
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
        {"role": "system", "content": "System message"},
    ]
    print_messages(messages, logger_mock)
    assert logger_mock._log_history == ["Hello\n", "Hi\n", "System message\n"]


def test_merge_messages():
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "user", "content": "How are you?"},
        {"role": "assistant", "content": "Hi"},
    ]
    merged = merge_messages(messages)
    assert len(merged) == 2
    assert merged[0]["content"] == "Hello\n\nHow are you?"

    # Ignore empty message
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "user", "content": ""},
        {"role": "user", "content": "How are you?"},
        {"role": "user", "content": ""},
        {"role": "assistant", "content": "Hi"},
    ]
    merged = merge_messages(messages)
    assert len(merged) == 2
    assert merged[0]["content"] == "Hello\n\nHow are you?"


def test_trim_prompt_messages():
    def count_tokens(text):
        return len(text)

    with pytest.raises(Exception, match="messages should not be empty"):
        trim_prompt_messages([], 5, count_tokens)

    with pytest.raises(
        Exception,
        match='all messages should be dictionaries with keys "content" and "role"',
    ):
        messages = [{"role": "system", "key": "System message"}]
        trim_prompt_messages(messages, 20, count_tokens)

    with pytest.raises(Exception, match="the last message should be from the user"):
        messages = [
            {"role": "system", "content": "System message"},
            {"role": "assistant", "content": "Assistant message"},
        ]
        trim_prompt_messages(messages, 20, count_tokens)

    with pytest.raises(
        Exception,
        match="if two consecutive messages are from the same role, they should be merged first",
    ):
        messages = [
            {"role": "system", "content": "System message 1"},
            {"role": "system", "content": "System message 2"},
            {"role": "user", "content": "User message"},
        ]
        trim_prompt_messages(messages, 20, count_tokens)

    with pytest.raises(Exception, match="context_length should be non-negative"):
        messages = [{"role": "user", "content": "User message"}]
        trim_prompt_messages(messages, -1, count_tokens)

    messages = [{"role": "user", "content": "User message"}]
    trimmed_messages = [{"role": "user", "content": "Us…ge"}]
    assert trim_prompt_messages(messages, 5, count_tokens) == trimmed_messages

    messages = [
        {"role": "system", "content": "System message"},
        {"role": "user", "content": "User message"},
    ]
    expected = [
        {"role": "system", "content": "System message"},
        {"role": "user", "content": "Us…ge"},
    ]
    assert trim_prompt_messages(messages, 20, count_tokens) == expected

    messages = [
        {"role": "user", "content": "User message"},
        {"role": "assistant", "content": "Assistant message"},
        {"role": "user", "content": "User message 2"},
    ]
    expected = messages
    assert trim_prompt_messages(messages, 200, count_tokens) == expected

    messages = [
        {"role": "user", "content": "User message 1"},
        {"role": "assistant", "content": "Assistant message"},
        {"role": "user", "content": "User message 2"},
    ]
    expected = [
        {"role": "assistant", "content": "Assistant message"},
        {"role": "user", "content": "User message 2"},
    ]
    assert trim_prompt_messages(messages, 35, count_tokens) == expected

    messages = [
        {"role": "system", "content": "System message"},
        {"role": "user", "content": "User message"},
        {"role": "assistant", "content": "Assistant message"},
        {"role": "user", "content": "User message 2"},
    ]
    expected = [
        {"role": "system", "content": "System message"},
        {"role": "user", "content": "User message 2"},
    ]
    assert trim_prompt_messages(messages, 35, count_tokens) == expected

    messages = [
        {"role": "system", "content": "System message"},
        {"role": "user", "content": "User message 1"},
        {"role": "assistant", "content": "Assistant message 1"},
        {"role": "user", "content": "User message 2"},
        {"role": "assistant", "content": "Assistant message 2"},
        {"role": "user", "content": "User message 3"},
        {"role": "assistant", "content": "Assistant message 3"},
        {"role": "user", "content": "User message 4"},
    ]
    expected = [
        {"role": "system", "content": "System message"},
        {"role": "user", "content": "User message 3"},
        {"role": "assistant", "content": "Assistant message 3"},
        {"role": "user", "content": "User message 4"},
    ]
    assert trim_prompt_messages(messages, 65, count_tokens) == expected


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


def test_trim():
    def count_tokens(text):
        return len(text)

    # Test trimming from the middle
    assert trim("Hello world", 5, count_tokens) == "He…ld"
    assert trim("Hello world", 11, count_tokens) == "Hello world"

    # Test trimming from the end
    assert trim("Hello world", 5, count_tokens, where="end") == "Hell…"
    assert trim("Hello world", 11, count_tokens, where="end") == "Hello world"

    # Test trimming from the start
    assert trim("Hello world", 5, count_tokens, where="start") == "…orld"
    assert trim("Hello world", 11, count_tokens, where="start") == "Hello world"

    # Test trimming with very short max_length
    assert trim("Hello world", 1, count_tokens) == "…"
    assert trim("Hello world", 0, count_tokens) == ""

    # Test trimming with exact length
    assert trim("Hi", 2, count_tokens) == "Hi"
    assert trim("Hi", 1, count_tokens) == "…"

    # Test invalid `where` value
    with pytest.raises(ValueError, match="Invalid value for `where`"):
        trim("Hello world", 5, count_tokens, where="invalid")

    def another_count_tokens(text):
        return len(text) // 2

    # Test trimming with a different token counter
    assert trim("1234567890", 3, another_count_tokens) == "12…90"
    assert trim("1234567890", 4, another_count_tokens) == "123…890"
    assert trim("1234567890", 5, another_count_tokens) == "1234567890"
